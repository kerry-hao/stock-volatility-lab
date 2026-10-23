"""Read competition metadata and one stock partition at a time."""
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["stock_id", "time_id"]
BOOK_COLUMNS = ["time_id", "seconds_in_bucket"] + [
    f"{side}_{kind}{level}" for kind in ("price", "size")
    for side in ("bid", "ask") for level in (1, 2)
]
TRADE_COLUMNS = ["time_id", "seconds_in_bucket", "price", "size", "order_count"]


def read_index(data_dir, split):
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    path = Path(data_dir) / f"{split}.csv"
    frame = pd.read_csv(path)
    required = KEYS + (["target"] if split == "train" else ["row_id"])
    missing = set(required) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"{path}: missing {missing} or empty index")
    for col in KEYS:
        v = pd.to_numeric(frame[col], errors="raise")
        if v.isna().any() or not np.isfinite(v).all() or (v < 0).any() or (v % 1 != 0).any():
            raise ValueError(f"{col} must contain nonnegative integer IDs")
        frame[col] = v.astype("int32")
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate stock_id/time_id pairs in index")
    expected = frame.stock_id.astype(str) + "-" + frame.time_id.astype(str)
    if "row_id" in frame and not frame.row_id.astype(str).equals(expected):
        raise ValueError("row_id does not match stock_id-time_id")
    frame["row_id"] = expected
    if split == "train":
        from .metrics import positive_targets
        frame["target"] = positive_targets(frame.target)
    return frame[required + (["row_id"] if split == "train" else [])]


def partition_path(data_dir, kind, split, stock_id):
    return Path(data_dir) / f"{kind}_{split}.parquet" / f"stock_id={int(stock_id)}"


def read_partition(data_dir, kind, split, stock_id, time_ids):
    path = partition_path(data_dir, kind, split, stock_id)
    columns = BOOK_COLUMNS if kind == "book" else TRADE_COLUMNS
    # An empty trade partition is allowed; a missing directory is a download error.
    if not path.exists():
        raise FileNotFoundError(f"Missing partition: {path}. Check the extracted dataset.")
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[frame.time_id.isin(time_ids), columns].copy()
    return validate_events(frame, kind)


def validate_events(frame, kind):
    columns = BOOK_COLUMNS if kind == "book" else TRADE_COLUMNS
    if set(columns) - set(frame.columns):
        raise ValueError(f"Missing {kind} columns")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite raw {kind} values")
    for key in ("time_id", "seconds_in_bucket"):
        if (frame[key] % 1 != 0).any():
            raise ValueError(f"Noninteger {key}")
    if not frame.seconds_in_bucket.between(0, 599).all():
        raise ValueError("seconds_in_bucket must be in [0, 599]")
    if frame.duplicated(["time_id", "seconds_in_bucket"]).any():
        raise ValueError(f"Duplicate {kind} timestamps within a stock")
    prices = [c for c in columns if "price" in c]
    sizes = [c for c in columns if "size" in c or c == "order_count"]
    if (frame[prices] <= 0).any().any() or (frame[sizes] < 0).any().any():
        raise ValueError("Prices must be positive and sizes nonnegative")
    if kind == "book":
        if ((frame.bid_size1 + frame.ask_size1) <= 0).any():
            raise ValueError("Top-of-book total size must be positive")
        if (frame.bid_price1 > frame.ask_price1).any():
            raise ValueError("Crossed top-of-book quotes")
    return frame.sort_values(["time_id", "seconds_in_bucket"], kind="stable").reset_index(drop=True)


def check_layout(data_dir, split):
    index = read_index(data_dir, split)
    stocks = sorted(index.stock_id.unique().tolist())
    missing = [str(partition_path(data_dir, kind, split, s))
               for s in stocks for kind in ("book", "trade")
               if not partition_path(data_dir, kind, split, s).exists()]
    if missing:
        raise FileNotFoundError("Missing partitions: " + ", ".join(missing[:6]))
    return {"split": split, "rows": len(index), "stocks": len(stocks),
            "time_ids": int(index.time_id.nunique()), "status": "layout_ok"}
