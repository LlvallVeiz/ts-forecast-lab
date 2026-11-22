"""Forecasters implemented from scratch.

Methods
-------
1. Simple moving average (SMA)
2. Simple exponential smoothing (SES)
3. Holt's linear trend method
4. Autoregressive AR(p) fitted by ordinary least squares

Every function returns a 1-D ``np.ndarray`` of length ``horizon``
containing point forecasts for steps n+1 … n+horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_1d_float(series) -> np.ndarray:
    y = np.asarray(series, dtype=float).ravel()
    if y.size == 0:
        raise ValueError("series must not be empty")
    if not np.all(np.isfinite(y)):
        raise ValueError("series must contain only finite values")
    return y


def _check_horizon(horizon: int) -> None:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")


# ---------------------------------------------------------------------------
# 1. Simple moving average
# ---------------------------------------------------------------------------

def moving_average_forecast(series, horizon: int = 1, window: int = 5) -> np.ndarray:
    """Forecast with a simple moving average.

    The forecast for every future step is the mean of the last
    ``window`` observations::

        f_{n+h} = (1/w) * sum_{i=0}^{w-1} y_{n-i}     for all h >= 1

    Parameters
    ----------
    series : array-like
    horizon : int
        Number of steps ahead to forecast.
    window : int
        Averaging window ``w``.  If ``w > len(series)`` the whole
        series is averaged.
    """
    y = _as_1d_float(series)
    _check_horizon(horizon)
    if window < 1:
        raise ValueError("window must be >= 1")
    w = min(window, y.size)
    level = float(np.mean(y[-w:]))
    return np.full(horizon, level, dtype=float)


# ---------------------------------------------------------------------------
# 2. Simple exponential smoothing (SES)
# ---------------------------------------------------------------------------

def ses_forecast(series, horizon: int = 1, alpha: float = 0.3) -> np.ndarray:
    """Simple exponential smoothing forecast.

    Recursion (level only, no trend)::

        l_0 = y_0
        l_t = alpha * y_t + (1 - alpha) * l_{t-1}
        f_{n+h} = l_n        for all h >= 1

    ``alpha`` in (0, 1] controls how fast old observations decay.
    On a constant series the level stays exactly at that constant,
    which the test-suite checks.
    """
    y = _as_1d_float(series)
    _check_horizon(horizon)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    level = float(y[0])
    for t in range(1, y.size):
        level = alpha * float(y[t]) + (1.0 - alpha) * level
    return np.full(horizon, level, dtype=float)


def ses_level(series, alpha: float = 0.3) -> float:
    """Return the final SES level (handy for tests / diagnostics)."""
    y = _as_1d_float(series)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    level = float(y[0])
    for t in range(1, y.size):
        level = alpha * float(y[t]) + (1.0 - alpha) * level
    return level


# ---------------------------------------------------------------------------
# 3. Holt's linear trend method
# ---------------------------------------------------------------------------

def holt_forecast(
    series,
    horizon: int = 1,
    alpha: float = 0.3,
    beta: float = 0.1,
) -> np.ndarray:
    """Holt's linear trend forecast.

    State equations::

        l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
        b_t = beta  * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
        f_{n+h} = l_n + h * b_n

    Initialisation: ``l_0 = y_0``, ``b_0 = y_1 - y_0`` (0 if n == 1).
    On a perfectly linear series Holt recovers the slope exactly,
    so forecasts lie on the true continuation line.
    """
    y = _as_1d_float(series)
    _check_horizon(horizon)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1]")

    level = float(y[0])
    trend = float(y[1] - y[0]) if y.size > 1 else 0.0

    for t in range(1, y.size):
        prev_level = level
        level = alpha * float(y[t]) + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend

    steps = np.arange(1, horizon + 1, dtype=float)
    return level + steps * trend


# ---------------------------------------------------------------------------
# 4. Autoregressive AR(p) by least squares
# ---------------------------------------------------------------------------

@dataclass
class ARModel:
    """Fitted AR(p) model:  y_t = c + phi_1 y_{t-1} + … + phi_p y_{t-p} + e_t."""

    intercept: float
    coefficients: np.ndarray  # shape (p,), phi_1 … phi_p
    p: int
    residuals: np.ndarray
    sigma2: float  # residual variance (unbiased-ish: RSS / (n - p - 1))

    def summary(self) -> str:
        coefs = ", ".join(f"{c:.4f}" for c in self.coefficients)
        return (
            f"AR({self.p})  intercept={self.intercept:.4f}  "
            f"phi=[{coefs}]  sigma2={self.sigma2:.4f}"
        )


def _lag_matrix(y: np.ndarray, p: int):
    """Build design matrix X and target vector for AR(p) OLS.

    Rows correspond to t = p … n-1::

        X_t = [1, y_{t-1}, y_{t-2}, …, y_{t-p}]
        target_t = y_t
    """
    n = y.size
    rows = n - p
    X = np.empty((rows, p + 1), dtype=float)
    X[:, 0] = 1.0
    for lag in range(1, p + 1):
        X[:, lag] = y[p - lag : n - lag]
    target = y[p:]
    return X, target


def ar_fit(series, p: int = 1) -> ARModel:
    """Fit AR(p) by ordinary least squares via ``numpy.linalg.lstsq``.

    Solves ``min ||X beta - y||^2`` where ``beta = [c, phi_1, …, phi_p]``.
    """
    y = _as_1d_float(series)
    if p < 1:
        raise ValueError("p must be >= 1")
    if y.size <= p + 1:
        raise ValueError(f"need at least p+2 = {p + 2} observations to fit AR({p})")

    X, target = _lag_matrix(y, p)
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    fitted = X @ beta
    residuals = target - fitted
    dof = max(len(target) - (p + 1), 1)
    sigma2 = float(np.sum(residuals ** 2) / dof)
    return ARModel(
        intercept=float(beta[0]),
        coefficients=np.asarray(beta[1:], dtype=float),
        p=p,
        residuals=residuals,
        sigma2=sigma2,
    )


def ar_forecast(series, horizon: int = 1, p: int = 1, model: ARModel | None = None) -> np.ndarray:
    """Multi-step AR(p) forecast by recursion.

    Each forecast step feeds back into the lag window::

        f_{n+1} = c + phi_1 y_n + phi_2 y_{n-1} + …
        f_{n+2} = c + phi_1 f_{n+1} + phi_2 y_n + …
    """
    y = _as_1d_float(series)
    _check_horizon(horizon)
    if model is None:
        model = ar_fit(y, p=p)
    if model.p != p and model is not None:
        # trust the supplied model
        p = model.p
    if y.size < p:
        raise ValueError("series shorter than AR order p")

    history = list(y[-p:])  # oldest … newest
    out = []
    for _ in range(horizon):
        # history[-1] is lag-1, history[-2] lag-2, …
        lags = np.array([history[-lag] for lag in range(1, p + 1)], dtype=float)
        nxt = model.intercept + float(model.coefficients @ lags)
        out.append(nxt)
        history.append(nxt)
    return np.asarray(out, dtype=float)


FORECASTERS = {
    "ma": moving_average_forecast,
    "ses": ses_forecast,
    "holt": holt_forecast,
    "ar": ar_forecast,
}
