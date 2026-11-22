"""Train/test backtesting harness.

Splits a series into train (first part) and test (last ``horizon``
points), fits the chosen forecaster on train, forecasts the test
window in one shot, and scores it with MAE / RMSE / MAPE plus
residual diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forecast import FORECASTERS
from .metrics import mae, rmse, mape, residual_diagnostics


@dataclass
class BacktestResult:
    method: str
    horizon: int
    train_size: int
    test_size: int
    forecasts: np.ndarray
    actuals: np.ndarray
    residuals: np.ndarray
    mae: float
    rmse: float
    mape: float
    diagnostics: dict

    def summary(self) -> str:
        mape_str = "nan" if self.mape != self.mape else f"{self.mape:.2f}%"
        return (
            f"method={self.method} horizon={self.horizon} "
            f"train={self.train_size} test={self.test_size} "
            f"MAE={self.mae:.4f} RMSE={self.rmse:.4f} MAPE={mape_str} "
            f"resid_mean={self.diagnostics['mean']:.4f} "
            f"resid_std={self.diagnostics['std']:.4f} "
            f"lag1={self.diagnostics['lag1_autocorr']:.4f}"
        )


def train_test_split(series, horizon: int):
    """Return (train, test) where test is the last ``horizon`` points."""
    y = np.asarray(series, dtype=float).ravel()
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if y.size <= horizon:
        raise ValueError("series must be longer than horizon")
    return y[:-horizon], y[-horizon:]


def backtest(series, method: str = "holt", horizon: int = 12, **params) -> BacktestResult:
    """Run a single train/test backtest.

    Extra keyword arguments are forwarded to the forecaster, e.g.
    ``backtest(y, method="ses", alpha=0.5)`` or
    ``backtest(y, method="ar", p=3)``.
    """
    if method not in FORECASTERS:
        raise ValueError(
            f"unknown method {method!r}; choose from {sorted(FORECASTERS)}"
        )
    train, test = train_test_split(series, horizon)
    forecasts = FORECASTERS[method](train, horizon=horizon, **params)
    residuals = test - forecasts
    return BacktestResult(
        method=method,
        horizon=horizon,
        train_size=int(train.size),
        test_size=int(test.size),
        forecasts=forecasts,
        actuals=test,
        residuals=residuals,
        mae=mae(test, forecasts),
        rmse=rmse(test, forecasts),
        mape=mape(test, forecasts),
        diagnostics=residual_diagnostics(residuals),
    )
