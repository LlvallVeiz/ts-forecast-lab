import unittest

import numpy as np

from tsforecast.forecast import (
    ar_fit,
    ar_forecast,
    holt_forecast,
    moving_average_forecast,
    ses_forecast,
    ses_level,
)


class TestMovingAverage(unittest.TestCase):
    def test_known_case(self):
        # mean of last 3 of [1..6] = (4+5+6)/3 = 5
        fc = moving_average_forecast([1, 2, 3, 4, 5, 6], horizon=4, window=3)
        np.testing.assert_allclose(fc, [5.0, 5.0, 5.0, 5.0])

    def test_window_larger_than_series(self):
        fc = moving_average_forecast([2, 4], horizon=2, window=10)
        np.testing.assert_allclose(fc, [3.0, 3.0])

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            moving_average_forecast([1, 2], window=0)


class TestSES(unittest.TestCase):
    def test_constant_series_stays_constant(self):
        y = np.full(30, 7.5)
        for alpha in (0.1, 0.3, 0.9, 1.0):
            fc = ses_forecast(y, horizon=5, alpha=alpha)
            np.testing.assert_allclose(fc, np.full(5, 7.5), atol=1e-12)

    def test_hand_computed_level(self):
        # y = [10, 12], alpha=0.5 -> l = 0.5*12 + 0.5*10 = 11
        self.assertAlmostEqual(ses_level([10, 12], alpha=0.5), 11.0)

    def test_alpha_validation(self):
        with self.assertRaises(ValueError):
            ses_forecast([1, 2, 3], alpha=0.0)
        with self.assertRaises(ValueError):
            ses_forecast([1, 2, 3], alpha=1.5)


class TestHolt(unittest.TestCase):
    def test_recovers_linear_trend(self):
        # Perfect line y = 10 + 2t, t=0..49. True continuation: 110, 112, ...
        t = np.arange(50)
        y = 10.0 + 2.0 * t
        fc = holt_forecast(y, horizon=5, alpha=0.5, beta=0.3)
        expected = 10.0 + 2.0 * np.arange(50, 55)
        np.testing.assert_allclose(fc, expected, atol=1e-6)

    def test_constant_series_flat(self):
        fc = holt_forecast(np.full(20, 3.0), horizon=3)
        np.testing.assert_allclose(fc, [3.0, 3.0, 3.0], atol=1e-9)

    def test_forecast_is_increasing_on_rising_series(self):
        y = 1.0 + 0.5 * np.arange(40) + np.sin(np.arange(40))
        fc = holt_forecast(y, horizon=4)
        self.assertTrue(np.all(np.diff(fc) > 0))


class TestAR(unittest.TestCase):
    def test_ar1_coefficient_recovery(self):
        # Simulate y_t = 0.8 * y_{t-1} + e_t, e ~ N(0, 1), seeded.
        rng = np.random.default_rng(123)
        n = 2000
        e = rng.normal(0.0, 1.0, size=n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.8 * y[t - 1] + e[t]
        model = ar_fit(y, p=1)
        self.assertAlmostEqual(float(model.coefficients[0]), 0.8, delta=0.05)
        self.assertAlmostEqual(model.intercept, 0.0, delta=0.1)

    def test_ar2_coefficient_recovery(self):
        rng = np.random.default_rng(99)
        n = 3000
        e = rng.normal(0.0, 1.0, size=n)
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.5 * y[t - 1] + 0.25 * y[t - 2] + e[t]
        model = ar_fit(y, p=2)
        np.testing.assert_allclose(model.coefficients, [0.5, 0.25], atol=0.06)

    def test_forecast_shape_and_recursion_decay(self):
        # AR(1) with phi=0.5, no intercept: forecasts decay toward 0.
        rng = np.random.default_rng(5)
        n = 500
        e = rng.normal(0.0, 1.0, size=n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + e[t]
        fc = ar_forecast(y, horizon=10, p=1)
        self.assertEqual(fc.shape, (10,))
        self.assertLess(abs(fc[-1]), abs(fc[0]) + 1e-9)

    def test_too_short_series_raises(self):
        with self.assertRaises(ValueError):
            ar_fit([1.0, 2.0], p=1)


if __name__ == "__main__":
    unittest.main()
