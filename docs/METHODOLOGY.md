# Methodology and interpretation

## Observation and prediction unit

One sample is a stock and observed 10-minute bucket. The supplied label is RV
over the subsequent 10 minutes. No target-window order-book data is available
as input. This is a supervised regression exercise with market microstructure
features, not a directional trading backtest.

## Features

All features use only the row's own observed bucket. No other stock's future
target or target summary is used. Feature generation can run before splitting
because it is deterministic, target-free and independent across rows.

| Family | Examples | Interpretation |
|---|---|---|
| Historical volatility | WAP1/WAP2 RV; absolute return sum | Recent price movement |
| Liquidity | Relative spread; quoted depth | Cost and capacity of immediate trading |
| Pressure | Bid/ask size imbalance; WAP gap | Differences in displayed interest |
| Activity | Book updates; trade seconds; order counts | Frequency of observed market activity |
| Trade flow | Volume; average trade size; trade-price RV | Executed activity and price movement |

Each family is summarized over `[0,600)`, `[300,600)` and `[480,600)`, in
seconds. Returns are recomputed **inside each subwindow**: the first included
observation has no return. The last-120-second feature does not include the
price jump from before second 480. A window with one price observation has RV 0;
its sample standard deviation is missing. WAP2 is missing when level-2 total
size is zero. LightGBM handles remaining missing numeric features natively.

No one-second forward filling or backward filling is performed. Means are over
reported observations, so active periods receive more weight. This is a clear
baseline choice, not an assertion that event-weighting is optimal. Compare a
properly forward-filled grid as a later feature ablation, keeping folds fixed.

## Baselines and loss

- Constant: the RMSPE-optimal constant estimated only on a training fold.
- Persistence: observed full-window WAP1 RV, with no fitted parameters.
- Scaled persistence: one positive least-squares multiplier estimated on the
  training fold with inverse-squared-target weights.
- LightGBM: regression trees with the same relative-error weighting, a numeric
  feature table and `stock_id` treated as categorical.

Using `w = 1/y²` makes weighted squared loss proportional to mean squared
percentage error. Normalizing these weights to mean one keeps their scale
manageable. The custom evaluation metric computes unweighted RMSPE and is the
only metric used for early stopping. Inverse-squared weighting gives large
influence to small targets; invalid/nonpositive targets raise an error rather
than receiving an arbitrary clipped denominator.

## Validation limits

All stocks from a given `time_id` stay in the same fold. This avoids testing on
some stocks from a bucket whose other stocks were in the training fold. It does
not prove that all dependency across buckets has been removed. The numeric
order of anonymized IDs does not justify `TimeSeriesSplit` or lagging across
them. Chronological validation would need verified timestamps or another
defensible time ordering.

The folds serve both early stopping and reported development evaluation. If you
repeatedly adjust features using these scores, select a separate untouched group
holdout before claiming final performance. This initial repository implements
grouped development CV, not nested validation or a final holdout experiment.

OOF RMSPE is calculated on all OOF rows together. It need not equal mean fold
RMSPE because RMSPE takes a square root after aggregation. Both are saved.
Inference averages each fold's predictions; it does not fit an additional
full-data model. All baseline constants/scales are also fold-specific.

## Useful questions after the first real run

1. Does LightGBM improve pooled OOF RMSPE over scaled persistence?
2. Is the improvement stable across folds and across stocks?
3. How much do trades and recent subwindows add over full-window book RV?
4. Are errors dominated by very small target values?
5. Do the findings survive a genuinely untouched group holdout?

Feature importance is predictive and model-specific. It is not evidence of
causality, execution quality or realizable net trading returns.

## Implementation references

- Supplied Optiver tutorial: https://www.kaggle.com/code/jiashenliu/introduction-to-financial-concepts-and-data
- GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
- LightGBM fit and sample weights: https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMRegressor.html
- LightGBM early stopping: https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.early_stopping.html
