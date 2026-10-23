"""Explicit JSON configuration; relative CLI paths use the current directory."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

DEFAULT = {
    "window_starts": [0, 300, 480], "prediction_floor": 1e-8,
    "n_splits": 5, "seed": 42, "early_stopping_rounds": 100,
    "lightgbm": {
        "n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 31,
        "min_child_samples": 100, "colsample_bytree": 0.8,
        "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 1.0, "n_jobs": 4,
    },
}


def load_config(path=None):
    cfg = deepcopy(DEFAULT)
    if path:
        supplied = json.loads(Path(path).read_text())
        unknown = set(supplied) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        model = supplied.pop("lightgbm", {})
        cfg.update(supplied)
        cfg["lightgbm"].update(model)
    starts = cfg["window_starts"]
    if (not starts or starts[0] != 0 or len(set(starts)) != len(starts)
            or any(type(s) is not int or not 0 <= s < 600 for s in starts)):
        raise ValueError("window_starts must be distinct integers in [0, 599], starting with 0")
    if not 0 < cfg["prediction_floor"] < 1:
        raise ValueError("prediction_floor must lie in (0, 1)")
    if cfg["n_splits"] < 2 or cfg["early_stopping_rounds"] < 1:
        raise ValueError("Use at least two folds and positive early stopping rounds")
    return cfg


def feature_spec(cfg):
    return {"version": 1, "window_starts": cfg["window_starts"],
            "sampling": "observed_updates", "returns": "within_each_window"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
