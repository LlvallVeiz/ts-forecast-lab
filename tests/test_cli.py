import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from tsforecast.cli import main


def run_cli(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


class TestCLI(unittest.TestCase):
    def test_generate_smoke(self):
        code, out = run_cli(["generate", "--n", "30", "--seed", "1"])
        self.assertEqual(code, 0)
        self.assertIn("generated n=30", out)

    def test_forecast_smoke(self):
        code, out = run_cli(
            ["forecast", "--method", "holt", "--horizon", "3", "--n", "60"]
        )
        self.assertEqual(code, 0)
        self.assertIn("method=holt horizon=3", out)
        self.assertIn("step 3:", out)

    def test_forecast_ar_smoke(self):
        code, out = run_cli(["forecast", "--method", "ar", "--p", "2", "--horizon", "2"])
        self.assertEqual(code, 0)
        self.assertIn("method=ar", out)

    def test_backtest_smoke(self):
        code, out = run_cli(["backtest", "--method", "ses", "--horizon", "6"])
        self.assertEqual(code, 0)
        self.assertIn("MAE=", out)
        self.assertIn("RMSE=", out)

    def test_plot_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.png")
            code, out = run_cli(["plot", "--out", path, "--horizon", "6"])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
            self.assertIn("saved plot", out)

    def test_generate_csv_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "series.csv")
            code, _ = run_cli(["generate", "--n", "10", "--out", path])
            self.assertEqual(code, 0)
            with open(path) as fh:
                lines = fh.read().strip().splitlines()
            # header + 10 values
            self.assertEqual(len(lines), 11)


if __name__ == "__main__":
    unittest.main()
