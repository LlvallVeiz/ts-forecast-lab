"""Synthetic time-series generation: trend + seasonality + noise.

Model
-----
    y_t = level + slope * t
          + amplitude * sin(2*pi*t / period + phase)
          + epsilon_t,   epsilon_t ~ N(0, sigma^2)

All randomness flows through an explicit ``seed`` so every series is
fully reproducible.  We use ``numpy.random.default_rng(seed)`` (PCG64)
rather than the legacy global RandomState.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeriesConfig:
    """Parameters for :func:`generate_series`."""

    n: int = 120
    level: float = 100.0
    slope: float = 0.5
    amplitude: float = 10.0
    period: float = 12.0
    phase: float = 0.0
    sigma: float = 2.0
    seed: int = 42

    def validate(self) -> None:
        if self.n < 2:
            raise ValueError("n must be >= 2")
        if self.period <= 0:
            raise ValueError("period must be > 0")
        if self.sigma < 0:
            raise ValueError("sigma must be >= 0")


def generate_series(config: SeriesConfig | None = None, **kwargs) -> np.ndarray:
    """Generate a synthetic series.

    Parameters
    ----------
    config : SeriesConfig, optional
        Full configuration.  Any keyword arguments override fields.
    **kwargs
        Field overrides, e.g. ``generate_series(n=50, seed=7)``.

    Returns
    -------
    np.ndarray of shape (n,), dtype float64.
    """
    if config is None:
        config = SeriesConfig()
    if kwargs:
        # Build a new config with overrides (dataclasses.replace equivalent)
        data = {**config.__dict__, **kwargs}
        config = SeriesConfig(**data)
    config.validate()

    rng = np.random.default_rng(config.seed)
    t = np.arange(config.n, dtype=float)

    trend = config.level + config.slope * t
    seasonality = config.amplitude * np.sin(
        2.0 * np.pi * t / config.period + config.phase
    )
    noise = rng.normal(0.0, config.sigma, size=config.n) if config.sigma > 0 else np.zeros(config.n)

    return (trend + seasonality + noise).astype(float)


def true_continuation(config: SeriesConfig, horizon: int) -> np.ndarray:
    """Noise-free continuation of the series for ``horizon`` steps past n.

    Useful for testing forecasters against the known ground truth.
    """
    config.validate()
    t = np.arange(config.n, config.n + horizon, dtype=float)
    trend = config.level + config.slope * t
    seasonality = config.amplitude * np.sin(
        2.0 * np.pi * t / config.period + config.phase
    )
    return (trend + seasonality).astype(float)
