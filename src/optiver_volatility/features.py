"""Target-free features computed separately inside each observed stock window.

Sampling is at reported updates, with no second-grid filling. Means and standard
deviations are therefore event-weighted. Returns never cross a time_id or the
left boundary of a selected subwindow.
"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from .config import digest, feature_spec, write_json
from .data import KEYS, read_index, read_partition


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def realized_volatility(returns):
    return float(np.sqrt(np.square(returns.dropna()).sum()))


def book_features(book, starts):
    frame = book.copy()
    for level in (1, 2):
        bid, ask = frame[f"bid_size{level}"], frame[f"ask_size{level}"]
        denom = (bid + ask).replace(0, np.nan)
        frame[f"wap{level}"] = (
            frame[f"bid_price{level}"] * ask + frame[f"ask_price{level}"] * bid
        ) / denom
    frame["midpoint"] = (frame.ask_price1 + frame.bid_price1) / 2
    frame["relative_spread"] = (frame.ask_price1 - frame.bid_price1) / frame.midpoint
    frame["depth"] = frame[["bid_size1", "ask_size1", "bid_size2", "ask_size2"]].sum(axis=1)
    frame["imbalance"] = (frame.bid_size1 - frame.ask_size1) / (frame.bid_size1 + frame.ask_size1)
    frame["wap_gap"] = (frame.wap1 - frame.wap2).abs()
    results = []
    for start in starts:
        part = frame.loc[frame.seconds_in_bucket >= start].copy()
        for level in (1, 2):
            part[f"log_return{level}"] = np.log(part[f"wap{level}"]).groupby(part.time_id).diff()
        stats = {
            "updates": ("seconds_in_bucket", "size"),
            "first_second": ("seconds_in_bucket", "min"),
            "last_second": ("seconds_in_bucket", "max"),
            "wap1_rv": ("log_return1", realized_volatility),
            "wap2_rv": ("log_return2", realized_volatility),
            "wap1_return_abs_sum": ("log_return1", lambda s: s.abs().sum()),
        }
        for column in ("wap1", "wap2", "relative_spread", "depth", "imbalance", "wap_gap"):
            for stat in ("mean", "std", "min", "max"):
                stats[f"{column}_{stat}"] = (column, stat)
        summary = part.groupby("time_id").agg(**stats)
        summary.columns = [f"book_w{600-start}_{c}" for c in summary.columns]
        results.append(summary)
    return pd.concat(results, axis=1)


def trade_features(trade, starts):
    frame = trade.copy()
    frame["size_per_order"] = frame["size"] / frame.order_count.replace(0, np.nan)
    results = []
    for start in starts:
        part = frame.loc[frame.seconds_in_bucket >= start].copy()
        part["log_return"] = np.log(part.price).groupby(part.time_id).diff()
        summary = part.groupby("time_id").agg(
            active_seconds=("seconds_in_bucket", "size"),
            volume=("size", "sum"), orders=("order_count", "sum"),
            price_mean=("price", "mean"), price_std=("price", "std"),
            price_min=("price", "min"), price_max=("price", "max"),
            size_mean=("size", "mean"), size_max=("size", "max"),
            size_per_order_mean=("size_per_order", "mean"),
            rv=("log_return", realized_volatility),
        )
        summary.columns = [f"trade_w{600-start}_{c}" for c in summary.columns]
        results.append(summary)
    return pd.concat(results, axis=1)


def stock_features(book, trade, stock_id, time_ids, starts):
    missing = set(time_ids) - set(book.time_id)
    if missing:
        raise ValueError(f"No book observations for stock {stock_id}, time IDs {sorted(missing)[:5]}")
    base = pd.DataFrame(index=pd.Index(time_ids, name="time_id"))
    frame = base.join(book_features(book, starts)).join(trade_features(trade, starts))
    for start in starts:
        w = 600-start
        for c in (f"book_w{w}_updates", f"trade_w{w}_active_seconds",
                  f"trade_w{w}_volume", f"trade_w{w}_orders", f"trade_w{w}_rv"):
            frame[c] = frame[c].fillna(0)
        frame[f"trade_w{w}_has_trade"] = (frame[f"trade_w{w}_active_seconds"] > 0).astype("int8")
    frame["stock_id"] = int(stock_id)
    return frame.reset_index()


def build_features(data_dir, output_dir, split, cfg, max_stocks=None, overwrite=False):
    if max_stocks is not None and max_stocks < 1:
        raise ValueError("max_stocks must be positive")
    index = read_index(data_dir, split)
    stocks = sorted(index.stock_id.unique().tolist())
    if max_stocks:
        stocks = stocks[:max_stocks]
        index = index.loc[index.stock_id.isin(stocks)].reset_index(drop=True)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"{split}.parquet"
    manifest_path = out / f"{split}_manifest.json"
    if not overwrite and (destination.exists() or manifest_path.exists()):
        raise FileExistsError("Feature output exists. Use a fresh directory or explicit --overwrite.")
    pieces, sources = [], []
    for stock in stocks:
        ids = index.loc[index.stock_id == stock, "time_id"].tolist()
        book = read_partition(data_dir, "book", split, stock, ids)
        trade = read_partition(data_dir, "trade", split, stock, ids)
        pieces.append(stock_features(book, trade, stock, ids, cfg["window_starts"]))
        for kind in ("book", "trade"):
            p = Path(data_dir) / f"{kind}_{split}.parquet" / f"stock_id={stock}"
            for f in sorted(p.rglob("*.parquet")):
                sources.append({"path": str(f.resolve()), "size": f.stat().st_size,
                                "mtime_ns": f.stat().st_mtime_ns})
        print(f"Built {split} stock_id={stock} ({len(ids)} windows)")
    features = pd.concat(pieces, ignore_index=True)
    frame = index.merge(features, on=KEYS, how="left", validate="one_to_one", sort=False)
    feature_columns = [c for c in frame if c not in ("row_id", "target", "time_id")]
    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    numeric = [c for c in feature_columns if c != "stock_id"]
    frame[numeric] = frame[numeric].astype("float32")
    temp = destination.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False)
    temp.replace(destination)
    manifest = {
        "split": split, "rows": len(frame), "stocks": stocks,
        "partial": len(stocks) < read_index(data_dir, split).stock_id.nunique(),
        "feature_spec": feature_spec(cfg), "feature_columns": feature_columns,
        "index_sha256": file_sha256(Path(data_dir) / f"{split}.csv"),
        "feature_sha256": file_sha256(destination), "source_files": sources,
    }
    manifest["signature"] = digest(manifest)
    write_json(manifest_path, manifest)
    return frame, manifest


def load_features(feature_dir, split):
    import json
    root = Path(feature_dir)
    manifest = json.loads((root / f"{split}_manifest.json").read_text())
    if manifest.get("split") != split:
        raise ValueError("Feature split does not match its manifest")
    if manifest["feature_sha256"] != file_sha256(root / f"{split}.parquet"):
        raise ValueError("Feature file changed after its manifest was written; rebuild features")
    frame = pd.read_parquet(root / f"{split}.parquet")
    if len(frame) != manifest["rows"] or frame.duplicated(KEYS).any():
        raise ValueError("Invalid feature cache")
    return frame, manifest
