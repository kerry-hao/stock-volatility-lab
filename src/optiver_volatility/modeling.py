"""Group validation, fold-specific baselines and optional LightGBM models."""
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import feature_spec, write_json
from .features import load_features
from .metrics import (fit_scale, lgb_rmspe, optimal_constant, positive_targets,
                      rmspe, rmspe_weights)

BASELINES = ["constant", "persistence", "scaled_persistence"]


def group_splits(frame, n_splits):
    if frame.time_id.nunique() < n_splits:
        raise ValueError("Not enough distinct time_id groups for this number of folds")
    for tr, va in GroupKFold(n_splits=n_splits).split(frame, groups=frame.time_id):
        if set(frame.iloc[tr].time_id) & set(frame.iloc[va].time_id):
            raise AssertionError("time_id overlap")
        yield tr, va


def model_matrix(frame, columns):
    x = frame[columns].copy()
    x["stock_id"] = x.stock_id.astype("int32")
    return x


def train_models(feature_dir, run_dir, cfg, baseline_only=False):
    frame, manifest = load_features(feature_dir, "train")
    if manifest["feature_spec"] != feature_spec(cfg):
        raise ValueError("Feature configuration changed; rebuild features or use the original config")
    y = positive_targets(frame.target)
    raw_past = frame.book_w600_wap1_rv.to_numpy(dtype=float)
    if not np.isfinite(raw_past).all() or (raw_past < 0).any():
        raise ValueError("Full-window past realized volatility is missing or invalid")
    floor = cfg["prediction_floor"]
    past = np.maximum(raw_past, floor)
    columns = manifest["feature_columns"]
    x = model_matrix(frame, columns)
    folds = list(group_splits(frame, cfg["n_splits"]))
    if not baseline_only:
        import lightgbm as lgb
    out = Path(run_dir)
    # A run directory is write-once, preventing mixtures of incompatible runs.
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    models = BASELINES + ([] if baseline_only else ["lightgbm"])
    oof = frame[["row_id", "stock_id", "time_id", "target"]].copy()
    oof["fold"] = -1
    for name in models:
        oof[name] = np.nan
    records, parameters, importances = [], [], []
    for fold, (tr, va) in enumerate(folds):
        constant = optimal_constant(y[tr])
        scale = fit_scale(y[tr], past[tr])
        parameters.append({"fold": fold, "constant": constant, "scale": scale})
        predictions = {
            "constant": np.full(len(va), constant), "persistence": past[va],
            "scaled_persistence": np.maximum(scale * past[va], floor),
        }
        if not baseline_only:
            params = dict(cfg["lightgbm"])
            params.update(objective="regression", metric="None", verbosity=-1,
                          random_state=cfg["seed"] + fold,
                          deterministic=True, force_col_wise=True)
            model = lgb.LGBMRegressor(**params)
            model.fit(
                x.iloc[tr], y[tr], sample_weight=rmspe_weights(y[tr]),
                eval_set=[(x.iloc[va], y[va])], eval_metric=lgb_rmspe,
                categorical_feature=["stock_id"],
                callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)],
            )
            predictions["lightgbm"] = np.maximum(model.predict(x.iloc[va]), floor)
            model.booster_.save_model(str(out / "models" / f"fold_{fold}.txt"))
            parameters[-1]["best_iteration"] = model.best_iteration_
            importances.append(pd.DataFrame({"feature": columns, "fold": fold,
                                             "gain": model.booster_.feature_importance("gain")}))
        oof.loc[va, "fold"] = fold
        for name, pred in predictions.items():
            oof.loc[va, name] = pred
            score = rmspe(y[va], pred)
            records.append({"fold": fold, "model": name, "n_train": len(tr),
                            "n_valid": len(va), "rmspe": score})
            print(f"Fold {fold}: {name} RMSPE={score:.6f}")
    fold_scores = pd.DataFrame(records)
    summary = pd.DataFrame([
        {"model": name, "oof_rmspe": rmspe(y, oof[name]),
         "mean_fold_rmspe": float(fold_scores.loc[fold_scores.model == name, "rmspe"].mean())}
        for name in models
    ])
    oof.to_csv(out / "oof_predictions.csv", index=False)
    fold_scores.to_csv(out / "fold_scores.csv", index=False)
    summary.to_csv(out / "model_comparison.csv", index=False)
    if importances:
        pd.concat(importances, ignore_index=True).to_csv(out / "feature_importance.csv", index=False)
    versions = {}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    write_json(out / "run.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
        "feature_spec": manifest["feature_spec"], "train_signature": manifest["signature"],
        "feature_columns": columns, "partial_training": manifest["partial"],
        "models": models, "fold_parameters": parameters, "versions": versions,
        "validation": "GroupKFold by time_id; not chronological validation",
        "scores": summary.to_dict(orient="records"),
    })
    return summary


def align_submission(index, predictions):
    if index.row_id.duplicated().any() or predictions.row_id.duplicated().any():
        raise ValueError("Duplicate submission IDs")
    if set(index.row_id) != set(predictions.row_id):
        raise ValueError("Prediction IDs do not exactly match the test index")
    result = index[["row_id"]].merge(predictions[["row_id", "target"]],
                                     on="row_id", how="left", validate="one_to_one", sort=False)
    positive_targets(result.target)
    return result


def predict_models(feature_dir, run_dir, output, model_name="lightgbm"):
    frame, manifest = load_features(feature_dir, "test")
    run = json.loads((Path(run_dir) / "run.json").read_text())
    if manifest["partial"]:
        raise ValueError("Partial test features cannot produce a complete submission")
    if run["partial_training"]:
        raise ValueError("This run trained on a stock subset. Train on all stocks before prediction.")
    if model_name not in run["models"]:
        raise ValueError(f"Model not trained; available models: {run['models']}")
    if run["feature_spec"] != manifest["feature_spec"] or run["feature_columns"] != manifest["feature_columns"]:
        raise ValueError("Train/test feature schemas do not match; rebuild with the same config")
    floor = run["config"]["prediction_floor"]
    past = np.maximum(frame.book_w600_wap1_rv.to_numpy(dtype=float), floor)
    parts = []
    if model_name == "lightgbm":
        import lightgbm as lgb
        x = model_matrix(frame, run["feature_columns"])
    for params in run["fold_parameters"]:
        if model_name == "constant":
            pred = np.full(len(frame), params["constant"])
        elif model_name == "persistence":
            pred = past
        elif model_name == "scaled_persistence":
            pred = params["scale"] * past
        else:
            model = lgb.Booster(model_file=str(Path(run_dir) / "models" / f"fold_{params['fold']}.txt"))
            pred = model.predict(x)
        parts.append(np.maximum(pred, floor))
    predictions = pd.DataFrame({"row_id": frame.row_id, "target": np.mean(parts, axis=0)})
    result = align_submission(frame, predictions)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Prediction output exists; choose a new output filename")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result
