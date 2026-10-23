"""CLI commands do nothing on import and never download data automatically."""
import argparse
import json
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Learn realized volatility prediction with Optiver data")
    parser.add_argument("--config", help="Optional JSON config; defaults are built in")
    subs = parser.add_subparsers(dest="command", required=True)
    check = subs.add_parser("check-data", help="Check CSV metadata and partition paths, without loading Parquet")
    check.add_argument("--data-dir", default="data/raw")
    check.add_argument("--split", choices=["train", "test"], default="train")
    build = subs.add_parser("build-features", help="Read Parquet and build features only")
    build.add_argument("--data-dir", default="data/raw")
    build.add_argument("--output-dir", default="data/processed")
    build.add_argument("--split", choices=["train", "test"], required=True)
    build.add_argument("--max-stocks", type=int)
    build.add_argument("--overwrite", action="store_true")
    train = subs.add_parser("train", help="Evaluate baselines and LightGBM with grouped folds")
    train.add_argument("--feature-dir", default="data/processed")
    train.add_argument("--run-dir", default="outputs/run01")
    train.add_argument("--baseline-only", action="store_true")
    predict = subs.add_parser("predict", help="Average fold predictions and write a CSV")
    predict.add_argument("--feature-dir", default="data/processed")
    predict.add_argument("--run-dir", default="outputs/run01")
    predict.add_argument("--output", default="outputs/submission.csv")
    predict.add_argument("--model", choices=["constant", "persistence", "scaled_persistence", "lightgbm"], default="lightgbm")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "check-data":
        from .data import check_layout
        print(json.dumps(check_layout(args.data_dir, args.split), indent=2))
    elif args.command == "build-features":
        from .features import build_features
        frame, _ = build_features(args.data_dir, args.output_dir, args.split, cfg,
                                   args.max_stocks, args.overwrite)
        print(f"Saved {len(frame)} {args.split} rows")
    elif args.command == "train":
        from .modeling import train_models
        print(train_models(args.feature_dir, args.run_dir, cfg, args.baseline_only).to_string(index=False))
    else:
        from .modeling import predict_models
        frame = predict_models(args.feature_dir, args.run_dir, args.output, args.model)
        print(f"Saved {len(frame)} predictions to {args.output}")


if __name__ == "__main__":
    main()
