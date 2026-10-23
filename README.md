# Optiver Realized Volatility Lab

A learning repository for the [Optiver Realized Volatility
Prediction competition](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction).
Use order-book and trade observations from one 10-minute window to predict
realized volatility in the **next** 10-minute window.

**Current status:** source code, learning notebooks and documentation prepared.
No competition data has been downloaded, no model has been trained, and no
new competition score is reported. You can read the entire workflow now and
add data later. All notebook outputs are intentionally empty.

## What you can learn

- Turn bid/ask prices and sizes into WAP, spreads, depth and imbalance signals.
- Calculate log returns and realized volatility without crossing stock/window boundaries.
- Aggregate observed updates over the full 600 seconds and the last 300/120 seconds.
- Compare a constant, persistence, scaled persistence and LightGBM using RMSPE.
- Validate by `time_id`, save out-of-fold predictions, and average fold models for inference.

This predicts the **magnitude** of future price movement. It does not predict
price direction or establish a profitable trading strategy.

## Start by reading

1. `notebooks/00_workflow.ipynb`: the task, data relationships and reading map.
2. `notebooks/01_microstructure_and_baseline.ipynb`: WAP, returns, volatility and a tiny illustrative example.
3. `notebooks/02_feature_engineering.ipynb`: inspect one local stock and understand the feature table.
4. `notebooks/03_validation_and_prediction.ipynb`: grouped evaluation, result interpretation and prediction.

The first two notebooks can be understood without the competition files.
Notebooks 02 and 03 need data only when you choose to execute their data cells.

## Data at a glance

| Input | Observation | Purpose |
|---|---|---|
| `train.csv` | One `stock_id`, `time_id` pair | Known future `target` |
| `test.csv` | One pair plus `row_id` | Requested prediction rows |
| `book_train.parquet/stock_id=.../` | Reported book snapshot within the feature window | Top two bid/ask price and size levels |
| `trade_train.parquet/stock_id=.../` | Trades aggregated within a reported second | Price, shares traded and order count |
| Corresponding `*_test.parquet/` | Same schema | Test feature windows |
| `sample_submission.csv` | `row_id`, `target` | Example submission format |

The two IDs join the tables. Many raw observations become one feature row per
stock/window. `seconds_in_bucket` runs from 0 through 599; not every second has
a reported observation. The target comes from a different, subsequent window.
**Never reconstruct the training target from the available feature window.**

See [data details](docs/DATA.md) and the [official data page](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data).

## Method

For level 1 of the order book:

$$\mathrm{WAP}_t = \frac{\mathrm{bid}_t\,\mathrm{asksize}_t + \mathrm{ask}_t\,\mathrm{bidsize}_t}{\mathrm{bidsize}_t+\mathrm{asksize}_t}$$

$$r_t=\log(\mathrm{WAP}_t/\mathrm{WAP}_{t-1}),\qquad \mathrm{RV}=\sqrt{\sum_t r_t^2}$$

The competition measure is not annualized. Persistence predicts the next
window's RV using the observed window's RV. Scaled persistence learns a single
multiplier on each training fold. LightGBM uses the engineered feature table.

$$\mathrm{RMSPE}=\sqrt{\frac{1}{N}\sum_i\left(\frac{y_i-\hat y_i}{y_i}\right)^2}$$

Lower is better. A relative error of 10% for every row gives RMSPE 0.10.
Squared-error training uses weights proportional to `1 / target**2`, which
aligns the training loss with squared percentage error. Targets must be positive;
invalid targets fail validation. A configurable floor applies to predictions.

**Validation:** 5-fold `GroupKFold` with `groups=time_id`. Related observations
from the same window stay together. `time_id` is an anonymized label, not a
verified chronological index; neither raw `time_id` nor `row_id` enters the
model. This estimates performance on held-out window groups, not a future-period
backtest. Early stopping also uses these validation folds, so OOF results are
development estimates rather than an untouched final test.

Feature construction is target-free and needs no fitted preprocessing. Means
are calculated over reported updates, not over a forward-filled one-second
grid. No global target encoding, target-derived market features or chronological
lags are included. See [design choices](docs/METHODOLOGY.md).

## Historical reference versus this repository

The supplied `introduction-to-financial-concepts-and-data.ipynb` is the
[Optiver introductory tutorial](https://www.kaggle.com/code/jiashenliu/introduction-to-financial-concepts-and-data)
by Jiashen Liu. It provides a useful persistence benchmark, rather than a
high-ranking model recipe.

| Method | Evaluation in the supplied notebook | Historical score |
|---|---|---|
| Persistence | All provided training rows, without fitted parameters | RMSPE 0.341; R² 0.628 |
| Persistence | Public leaderboard reported in its text | RMSPE 0.327 |
| This repository's grouped baselines and LightGBM | Not run | Pending |

These historical scores are not our measurements and do not promise a current
leaderboard result. The new pipeline also sorts observations explicitly and
validates inputs, so exact numerical reproduction is not claimed. Once run,
`model_comparison.csv` records comparable pooled OOF RMSPE for all implemented
models. A lower LightGBM score would support predictive improvement over these
baselines under this split; it would not by itself establish trading profits.

## Install when you are ready

Python 3.10+ is required; Python 3.11 is a sensible starting environment.
Commands below assume a terminal at the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[notebooks]'
```

On Windows, activate with `.venv\Scripts\activate` instead. On macOS, if
LightGBM reports a missing OpenMP library, install `libomp` using your package
manager and retry. Dependencies have compatible version ranges; this is not
a fully locked environment. Each completed run records installed package versions.

## Add data later

Download and extract the competition files yourself into `data/raw`, preserving
the partition directories. Browser download is perfectly adequate for a first
run. Kaggle's CLI is convenient for repeatable downloads after you configure
authentication and accept any required competition terms:

```bash
python -m pip install -e '.[download]'
kaggle competitions download -c optiver-realized-volatility-prediction -p data/raw
python -m zipfile -e data/raw/optiver-realized-volatility-prediction.zip data/raw
```

Do not commit raw data, tokens or `kaggle.json`. They are excluded by `.gitignore`.
The expected layout is illustrated in `docs/DATA.md`. No command in this
repository downloads data automatically.

## Run the pipeline later

Check paths and metadata first; this command does not load Parquet:

```bash
optiver-lab check-data --data-dir data/raw --split train
```

Build full training features, then compare all four models:

```bash
optiver-lab --config configs/default.json build-features --split train
optiver-lab --config configs/default.json train --run-dir outputs/run01
```

To start with baselines only, use a separate run directory:

```bash
optiver-lab train --baseline-only --run-dir outputs/baselines01
```

For a smaller first feature run, use **separate directories**. This selects the
first three numeric stock IDs, not a representative benchmark:

```bash
optiver-lab build-features --split train --max-stocks 3 --output-dir data/processed/small
optiver-lab train --feature-dir data/processed/small --run-dir outputs/small01
```

Partial training runs are labelled and cannot produce test predictions. Full datasets may need
substantial RAM and time. Processing one stock at a time bounds raw-data memory,
but all aggregated stock features are combined for model fitting. Thread count
is 4 by default and is configurable.

When you have complete test files:

```bash
optiver-lab build-features --split test
optiver-lab predict --run-dir outputs/run01 --model lightgbm --output outputs/submission.csv
```

The prediction command averages the saved fold models, requires matching feature
schemas, validates positive predictions, and preserves `test.csv` row order.
The provided downloadable test sample is small; generating its CSV does not
recreate Kaggle's hidden evaluation. This repository does not package a Kaggle
submission environment or submit anything to Kaggle.

## What a completed run writes

| File | Read it to answer |
|---|---|
| `data/processed/train.parquet` | What features enter each prediction? |
| `data/processed/train_manifest.json` | Which feature settings and source files were used? |
| `outputs/run01/model_comparison.csv` | Which method has the lowest pooled OOF RMSPE? |
| `outputs/run01/fold_scores.csv` | Is the improvement stable across held-out groups? |
| `outputs/run01/oof_predictions.csv` | Where does the model overpredict or underpredict? |
| `outputs/run01/feature_importance.csv` | Which features receive gain in LightGBM? |
| `outputs/run01/models/fold_*.txt` | Which fitted boosters are used for inference? |
| `outputs/run01/run.json` | What configuration, versions and validation produced the scores? |

Gain importance describes model use, not causal importance. A baseline-only run
does not write boosters or feature importances. Run directories are write-once;
choose a new name to rerun. Existing feature files need explicit `--overwrite`.
Manifests hash the resulting feature file and record source sizes/timestamps;
they do not hash every raw Parquet file or automatically detect later raw-data
edits. Rebuild features after replacing raw data.

## Check the code without competition data

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
```

The included checks use tiny in-memory examples for formulas, return boundaries,
missing trades, group separation and submission alignment. No model is fitted.
They do not replace a real-data integration run. GitHub Actions runs the same
checks after you publish the repository.

## Repository map

| Path | Role |
|---|---|
| `src/optiver_volatility/data.py` | CSV keys, partition readers and input checks |
| `src/optiver_volatility/features.py` | WAP, book/trade summaries and feature manifests |
| `src/optiver_volatility/metrics.py` | RMSPE, loss weights and baseline formulas |
| `src/optiver_volatility/modeling.py` | Group folds, training, OOF scores and inference |
| `src/optiver_volatility/cli.py` | Command-line entry point |
| `configs/default.json` | Windows, fold count, prediction floor and tree settings |
| `notebooks/` | Guided reading and optional execution |
| `docs/` | Data guide, design choices, publishing and results status |
| `tests/` | Small deterministic checks |

## Put this on GitHub

Extract the ZIP, open the inner `optiver-volatility-lab` folder, and upload
**its contents** to an empty repository. Include `.gitignore` and `.github`.
For terminal instructions and an existing repository, see
[publishing instructions](docs/PUBLISHING.md).

Suggested repository name: `optiver-volatility-lab`.
Suggested description: `Order-book feature engineering and grouped evaluation for realized-volatility prediction.`

Next improvements can include a genuine real-data run, feature ablations,
per-stock error analysis and a separate final holdout. More complex ensembles
make sense after these comparisons are established.

See [third-party attribution](THIRD_PARTY_NOTICES.md) and [results status](docs/RESULTS.md).
