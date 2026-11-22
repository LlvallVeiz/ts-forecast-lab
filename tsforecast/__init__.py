"""tsforecast — time-series forecasting lab built from scratch.

Public API re-exports for convenience.
"""

from .series import generate_series, SeriesConfig
from .forecast import (
    moving_average_forecast,
    ses_forecast,
    holt_forecast,
    ar_fit,
    ar_forecast,
    ARModel,
)
from .metrics import mae, rmse, mape, residual_diagnostics
from .backtest import backtest, BacktestResult

__all__ = [
    "generate_series",
    "SeriesConfig",
    "moving_average_forecast",
    "ses_forecast",
    "holt_forecast",
    "ar_fit",
    "ar_forecast",
    "ARModel",
    "mae",
    "rmse",
    "mape",
    "residual_diagnostics",
    "backtest",
    "BacktestResult",
]

__version__ = "1.0.0"
