"""Small in-memory checks only: no competition data or model fitting."""
import unittest
import numpy as np
import pandas as pd

from optiver_volatility.config import load_config
from optiver_volatility.data import BOOK_COLUMNS, TRADE_COLUMNS, validate_events
from optiver_volatility.features import stock_features
from optiver_volatility.metrics import fit_scale, optimal_constant, rmspe, rmspe_weights
from optiver_volatility.modeling import align_submission, group_splits


def book_fixture():
    records = []
    for time, base in ((10, 1.0), (20, 3.0)):
        for second, shift in ((0, 0), (300, .01), (301, .02), (500, .03)):
            row = {"time_id": time, "seconds_in_bucket": second}
            for level in (1, 2):
                row[f"bid_price{level}"] = base + shift - .001 * level
                row[f"ask_price{level}"] = base + shift + .001 * level
                row[f"bid_size{level}"] = 10
                row[f"ask_size{level}"] = 10
            records.append(row)
    return validate_events(pd.DataFrame(records)[BOOK_COLUMNS], "book")


def empty_trades():
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in TRADE_COLUMNS})


class MetricTests(unittest.TestCase):
    def test_rmspe(self):
        self.assertAlmostEqual(rmspe([1, 2], [1.1, 2.2]), .1)

    def test_weighted_squared_error(self):
        y = np.array([.1, .3, .6]); p = np.array([.2, .2, .4])
        w = rmspe_weights(y)
        self.assertAlmostEqual(np.mean(w), 1)
        factor = np.mean(1 / y**2)
        self.assertAlmostEqual(np.mean(w * (y-p)**2) * factor, rmspe(y, p)**2)

    def test_constant_and_scale_optima(self):
        y = np.array([1., 2., 4.])
        c = optimal_constant(y)
        for delta in (-.1, .1):
            self.assertLess(rmspe(y, np.full(3, c)), rmspe(y, np.full(3, c+delta)))
        self.assertAlmostEqual(fit_scale(y, y / 2), 2)

    def test_reject_nonpositive_target(self):
        for y in ([0, 1], [-1, 1], [np.nan, 1]):
            with self.assertRaises(ValueError):
                rmspe(y, [1, 1])


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.book = book_fixture()

    def test_wap_and_bucket_boundary(self):
        f = stock_features(self.book, empty_trades(), 0, [10, 20], [0, 300, 480]).set_index("time_id")
        for time, base in ((10, 1.), (20, 3.)):
            expected = np.sqrt(np.sum(np.diff(np.log(base + np.array([0, .01, .02, .03])))**2))
            self.assertAlmostEqual(f.loc[time, "book_w600_wap1_rv"], expected)

    def test_subwindow_excludes_preboundary_return(self):
        f = stock_features(self.book, empty_trades(), 0, [10], [0, 300, 480])
        expected = np.sqrt(np.sum(np.diff(np.log([1.01, 1.02, 1.03]))**2))
        self.assertAlmostEqual(f.book_w300_wap1_rv.iloc[0], expected)
        self.assertEqual(f.book_w120_wap1_rv.iloc[0], 0)

    def test_no_trade_has_zero_activity_and_missing_price(self):
        f = stock_features(self.book, empty_trades(), 0, [10, 20], [0])
        self.assertTrue((f.trade_w600_has_trade == 0).all())
        self.assertTrue((f.trade_w600_rv == 0).all())
        self.assertTrue(f.trade_w600_price_mean.isna().all())

    def test_missing_book_fails(self):
        with self.assertRaises(ValueError):
            stock_features(self.book, empty_trades(), 0, [99], [0])

    def test_duplicate_timestamp_fails(self):
        with self.assertRaises(ValueError):
            validate_events(pd.concat([self.book, self.book.iloc[:1]]), "book")

    def test_sorting_is_stable(self):
        shuffled = self.book.sample(frac=1, random_state=4)
        pd.testing.assert_frame_equal(validate_events(shuffled, "book"), self.book)

    def test_wap_uses_opposite_side_size(self):
        book = self.book.iloc[:1].copy()
        book["bid_price1"], book["ask_price1"] = 99., 101.
        book["bid_size1"], book["ask_size1"] = 30., 10.
        f = stock_features(book, empty_trades(), 0, [10], [0])
        self.assertAlmostEqual(f.book_w600_wap1_mean.iloc[0], 100.5)

    def test_trade_returns_do_not_cross_buckets(self):
        trade = pd.DataFrame({"time_id": [10, 10, 20, 20],
                              "seconds_in_bucket": [1, 2, 1, 2],
                              "price": [1., 1.1, 10., 11.],
                              "size": [4., 6., 8., 10.],
                              "order_count": [2., 3., 4., 5.]})
        f = stock_features(self.book, validate_events(trade, "trade"), 0, [10, 20], [0])
        np.testing.assert_allclose(f.trade_w600_rv, np.log(1.1))
        self.assertEqual(f.trade_w600_volume.tolist(), [10., 18.])

    def test_empty_recent_subwindow(self):
        book = self.book.loc[self.book.seconds_in_bucket < 480]
        f = stock_features(book, empty_trades(), 0, [10, 20], [0, 480])
        self.assertTrue((f.book_w120_updates == 0).all())
        self.assertTrue(f.book_w120_wap1_mean.isna().all())


class ValidationTests(unittest.TestCase):
    def test_group_folds_are_disjoint_and_cover_rows_once(self):
        frame = pd.DataFrame({"time_id": np.repeat([7, 9, 18, 20], 3)})
        seen = []
        for tr, va in group_splits(frame, 2):
            self.assertFalse(set(frame.iloc[tr].time_id) & set(frame.iloc[va].time_id))
            seen.extend(va)
        self.assertEqual(sorted(seen), list(range(len(frame))))

    def test_submission_alignment(self):
        index = pd.DataFrame({"row_id": ["1-2", "1-1"]})
        predictions = pd.DataFrame({"row_id": ["1-1", "1-2"], "target": [.1, .2]})
        result = align_submission(index, predictions)
        self.assertEqual(result.target.tolist(), [.2, .1])
        with self.assertRaises(ValueError):
            align_submission(index, predictions.iloc[:1])

    def test_config_rejects_bad_windows(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"window_starts": [300]}))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
