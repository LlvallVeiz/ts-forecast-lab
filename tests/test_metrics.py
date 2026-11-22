import math
import unittest

import numpy as np

from tsforecast.backtest import backtest, train_test_split
from tsforecast.metrics import lag1_autocorrelation, mae, mape, residual_diagnostics, rmse
from tsforecast.series import generate_series


class TestMetrics(unittest.TestCase):
    def test_mae_known_case(self):
        # errors: |10-12|=2, |20-18|=2, |30-30|=0 -> MAE = 4/3
        self.assertAlmostEqual(mae([10, 20, 30], [12, 18, 30]), 4.0 / 3.0)

    def test_rmse_known_case(self):
        # squared errors: 4, 4, 0 -> MSE=8/3 -> RMSE=sqrt(8/3)
        self.assertAlmostEqual(rmse([10, 20, 30], [12, 18, 30]), math.sqrt(8.0 / 3.0))

    def test_mape_known_case(self):
        # |2/10| + |2/20| + 0 = 0.3 -> /3 *100 = 10%
        self.assertAlmostEqual(mape([10, 20, 30], [12, 18, 30]), 10.0)

    def test_perfect_forecast_zero_error(self):
        y = [1.0, 2.0, 3.0]
        self.assertEqual(mae(y, y), 0.0)
        self.assertEqual(rmse(y, y), 0.0)
        self.assertEqual(mape(y, y), 0.0)

    def test_mape_skips_zero_actuals(self):
        # zero actual skipped: only |10-12|/10 = 20%
        self.assertAlmostEqual(mape([0, 10], [5, 12]), 20.0)

    def test_mape_all_zero_actuals_is_nan(self):
        self.assertTrue(math.isnan(mape([0, 0], [1, 2])))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mae([1, 2], [1])

    def test_lag1_autocorr_perfectly_correlated(self):
        # strictly increasing residuals: centred products all positive -> near 1
        r = lag1_autocorrelation(np.arange(1.0, 21.0))
        self.assertGreater(r, 0.5)

    def test_lag1_autocorr_constant_is_zero(self):
        self.assertEqual(lag1_autocorrelation([2.0, 2.0, 2.0]), 0.0)

    def test_residual_diagnostics(self):
        d = residual_diagnostics([1.0, -1.0, 1.0, -1.0])
        self.assertAlmostEqual(d["mean"], 0.0)
        self.assertEqual(d["n"], 4)
        self.assertGreater(d["std"], 0.0)


class TestBacktest(unittest.TestCase):
    def test_split_sizes(self):
        train, test = train_test_split(np.arange(20.0), horizon=5)
        self.assertEqual(len(train), 15)
        self.assertEqual(len(test), 5)
        np.testing.assert_array_equal(test, np.arange(15.0, 20.0))

    def test_backtest_holt_on_trended_series(self):
        y = generate_series(n=120, seed=42)
        result = backtest(y, method="holt", horizon=12)
        self.assertEqual(result.forecasts.shape, (12,))
        self.assertEqual(result.train_size, 108)
        self.assertTrue(np.isfinite(result.mae))
        self.assertTrue(np.isfinite(result.rmse))
        self.assertIn("MAE=", result.summary())

    def test_backtest_all_methods_run(self):
        y = generate_series(n=80, seed=3)
        for method in ("ma", "ses", "holt", "ar"):
            result = backtest(y, method=method, horizon=8)
            self.assertEqual(result.forecasts.shape, (8,), msg=method)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            backtest([1, 2, 3, 4, 5], method="nope", horizon=2)


if __name__ == "__main__":
    unittest.main()
