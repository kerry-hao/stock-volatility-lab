# Data guide

Source: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data

## Extracted layout

| Relative path | Expected contents |
|---|---|
| `data/raw/train.csv` | `stock_id`, `time_id`, `target` |
| `data/raw/test.csv` | `stock_id`, `time_id`, `row_id` |
| `data/raw/sample_submission.csv` | `row_id`, `target` |
| `data/raw/book_train.parquet/stock_id=0/*.parquet` | One or more files for this stock |
| `data/raw/trade_train.parquet/stock_id=0/*.parquet` | One or more files for this stock |
| `data/raw/book_test.parquet/stock_id=0/*.parquet` | Test book data |
| `data/raw/trade_test.parquet/stock_id=0/*.parquet` | Test trade data |

Repeat partitions for the supplied stock IDs; IDs need not be contiguous. A
`.parquet` suffix here can belong to a **directory**, not a single file.
Reading a stock partition may omit the partition key, so the pipeline explicitly
adds `stock_id` from the folder/index rather than expecting it in the file.

## Identifiers and raw fields

| Field | Meaning |
|---|---|
| `stock_id` | Anonymous instrument identifier |
| `time_id` | Shared anonymous window label across stocks |
| `seconds_in_bucket` | Elapsed second within the observed 600-second window |
| `bid_price1`, `ask_price1` | Most competitive buy/sell prices |
| `bid_price2`, `ask_price2` | Second levels |
| `bid_size1/2`, `ask_size1/2` | Shares available at the corresponding levels |
| Trade `price` | Volume-weighted execution price in a reported second |
| Trade `size` | Total shares executed in that second |
| Trade `order_count` | Number of executed orders represented by the record |
| `target` | Realized volatility of the following 10-minute window |
| `row_id` | Submission key, formatted as `stock_id-time_id` |

Prices are normalized. Do not interpret them as raw currency prices or use them
to calculate portfolio profit. Observation counts can differ across windows and
between book and trade files. This pipeline sorts observations within each
window and keeps only the indexed stock/window pairs.

No-trade windows keep their label row: trade counts, volume and RV are zero;
unobserved trade-price statistics stay missing and `has_trade` is zero. Missing
book observations for a labelled pair fail explicitly. A missing stock partition
directory also fails, so an incomplete download is not silently treated as
inactivity. An existing empty trade partition is allowed.

## One example from the supplied reference

For `stock_id=0`, `time_id=5`, its saved output shows observed-window RV about
0.004499, while the supplied future target is about 0.004136. These are two
different windows. Using the first number to predict the second is persistence.
The numbers are reference outputs, not values recalculated for this repository.
