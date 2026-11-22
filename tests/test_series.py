import unittest

import numpy as np

from tsforecast.series import SeriesConfig, generate_series, true_continuation


class TestSeries(unittest.TestCase):
    def test_deterministic_same_seed(self):
        a = generate_series(SeriesConfig(n=50, seed=7))
        b = generate_series(SeriesConfig(n=50, seed=7))
        np.testing.assert_array_equal(a, b)

    def test_different_seed_differs(self):
        a = generate_series(SeriesConfig(n=50, seed=7))
        b = generate_series(SeriesConfig(n=50, seed=8))
        self.assertFalse(np.array_equal(a, b))

    def test_shape_and_dtype(self):
        y = generate_series(n=33, seed=1)
        self.assertEqual(y.shape, (33,))
        self.assertEqual(y.dtype, np.dtype(float))

    def test_no_noise_matches_formula(self):
        cfg = SeriesConfig(n=10, level=5.0, slope=2.0, amplitude=0.0, sigma=0.0, seed=0)
        y = generate_series(cfg)
        expected = 5.0 + 2.0 * np.arange(10)
        np.testing.assert_allclose(y, expected)

    def test_true_continuation_length(self):
        cfg = SeriesConfig(n=20, sigma=0.0)
        cont = true_continuation(cfg, horizon=6)
        self.assertEqual(cont.shape, (6,))

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            generate_series(n=1)
        with self.assertRaises(ValueError):
            generate_series(period=0)


if __name__ == "__main__":
    unittest.main()
