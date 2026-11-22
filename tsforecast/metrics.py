"""Forecast accuracy metrics and residual diagnostics.

Definitions (e_t = y_t - f_t, n = number of points)::

    MAE  = (1/n) * sum |e_t|
    RMSE = sqrt( (1/n) * sum e_t^2 )
    MAPE = (100/n') * sum |e_t / y_t|    over points with y_t != 0

MAPE handling of zero actuals: points where the actual is exactly 0
are excluded from the average (division by zero would otherwise blow
up).  If *every* actual is zero, MAPE returns ``nan`` — there is no
meaningful percentage error.  Callers that need a finite number can
check ``math.isnan``.
"""

from __future__ import annotations

import math

import numpy as np


def _pair(actual, predicted):
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs predicted {p.shape}")
    if a.size == 0:
        raise ValueError("empty input")
    return a, p


def mae(actual, predicted) -> float:
    """Mean absolute error."""
    a, p = _pair(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    """Root mean squared error."""
    a, p = _pair(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual, predicted) -> float:
    """Mean absolute percentage error, in percent (0–100+ scale).

    Zero actuals are skipped; returns ``nan`` if all actuals are zero.
    """
    a, p = _pair(actual, predicted)
    mask = a != 0.0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100.0)


def lag1_autocorrelation(residuals) -> float:
    """Lag-1 autocorrelation of a residual series.

        r1 = sum_{t=2..n} (e_t - e_bar)(e_{t-1} - e_bar)
             / sum_{t=1..n} (e_t - e_bar)^2

    Returns 0.0 for constant residuals (zero variance) or n < 2.
    """
    e = np.asarray(residuals, dtype=float).ravel()
    if e.size < 2:
        return 0.0
    centred = e - np.mean(e)
    denom = float(np.sum(centred ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(centred[1:] * centred[:-1]) / denom)


def residual_diagnostics(residuals) -> dict:
    """Summarise residuals: mean, std (ddof=1), lag-1 autocorrelation, n."""
    e = np.asarray(residuals, dtype=float).ravel()
    if e.size == 0:
        raise ValueError("empty residuals")
    std = float(np.std(e, ddof=1)) if e.size > 1 else 0.0
    return {
        "mean": float(np.mean(e)),
        "std": std,
        "lag1_autocorr": lag1_autocorrelation(e),
        "n": int(e.size),
    }
