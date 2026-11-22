# ts-forecast-lab

A time-series forecasting lab built **from scratch** in pure Python + NumPy.
No forecasting libraries — every method (moving average, exponential
smoothing, Holt's linear trend, AR(p) by least squares) is implemented by
hand with its formulas documented, so the code doubles as a study reference.

## What it does

- **Synthetic series generator** — trend + sinusoidal seasonality + Gaussian
  noise, fully seeded/deterministic (`numpy.random.default_rng(seed)`).
- **Forecasters from scratch**
  - Simple moving average
  - Simple exponential smoothing (SES)
  - Holt's linear trend method
  - AR(p) fitted by ordinary least squares (`numpy.linalg.lstsq` on a lag
    matrix), multi-step forecasts by recursion
- **Backtesting** — train/test split, MAE / RMSE / MAPE, residual diagnostics
  (mean, std, lag-1 autocorrelation).
- **CLI** with `generate`, `forecast`, `backtest`, and `plot` subcommands
  (matplotlib actual-vs-forecast PNG).

## Formulas

### Synthetic series

```
y_t = level + slope * t + amplitude * sin(2*pi*t / period + phase) + e_t,
e_t ~ N(0, sigma^2)
```

### Simple moving average (window w)

```
f_{n+h} = (1/w) * sum_{i=0}^{w-1} y_{n-i}      for all h >= 1
```

### Simple exponential smoothing (SES)

```
l_0 = y_0
l_t = alpha * y_t + (1 - alpha) * l_{t-1}
f_{n+h} = l_n                                  for all h >= 1
```

### Holt's linear trend

```
l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
f_{n+h} = l_n + h * b_n
```

Initialisation: `l_0 = y_0`, `b_0 = y_1 - y_0`.

### AR(p)

```
y_t = c + phi_1*y_{t-1} + ... + phi_p*y_{t-p} + e_t
```

Fitted by OLS: build the lag design matrix `X_t = [1, y_{t-1}, ..., y_{t-p}]`
and solve `min ||X beta - y||^2` with `numpy.linalg.lstsq`. Multi-step
forecasts recurse: each forecast feeds back into the lag window.

### Metrics (e_t = y_t - f_t)

```
MAE  = (1/n) * sum |e_t|
RMSE = sqrt( (1/n) * sum e_t^2 )
MAPE = (100/n') * sum |e_t / y_t|     over points with y_t != 0
```

Zero actuals are excluded from MAPE (no division by zero); if every actual
is zero, MAPE returns `nan`.

Lag-1 residual autocorrelation:

```
r1 = sum_{t=2..n} (e_t - e_bar)(e_{t-1} - e_bar) / sum_{t=1..n} (e_t - e_bar)^2
```

## Requirements

- Python 3.12
- numpy 1.26.4, matplotlib 3.6.3 (see `requirements.txt`)

```bash
pip install -r requirements.txt
```

## Usage

All commands run from the project root via the module entry point:

```bash
python3 -m tsforecast <subcommand> [options]
# equivalently: python3 -m tsforecast.cli <subcommand> ...
```

Shared series options: `--n`, `--level`, `--slope`, `--amplitude`,
`--period`, `--sigma`, `--seed`.
Method options (forecast/backtest/plot): `--method {ma,ses,holt,ar}`,
`--horizon`, `--alpha`, `--beta`, `--window`, `--p`.

### generate — create a synthetic series

```bash
python3 -m tsforecast generate --n 120 --seed 42
python3 -m tsforecast generate --n 120 --seed 42 --out series.csv
```

### forecast — fit a method and print point forecasts

```bash
python3 -m tsforecast forecast --method holt --horizon 12
python3 -m tsforecast forecast --method ar --p 3 --horizon 6
python3 -m tsforecast forecast --method ses --alpha 0.5 --horizon 4
python3 -m tsforecast forecast --method ma --window 7 --horizon 4
```

### backtest — train/test split with metrics

```bash
python3 -m tsforecast backtest --method holt --horizon 12
python3 -m tsforecast backtest --method ar --p 2 --horizon 12
```

Example output:

```
method=holt horizon=12 train=108 test=12 MAE=... RMSE=... MAPE=...% resid_mean=... resid_std=... lag1=...
```

### plot — actual vs forecast PNG

```bash
python3 -m tsforecast plot --method holt --horizon 12 --out forecast.png
```

## Project structure

```
ts-forecast-lab/
├── README.md
├── requirements.txt
├── .gitignore
├── tsforecast/
│   ├── __init__.py      # public API re-exports
│   ├── __main__.py      # python3 -m tsforecast
│   ├── series.py        # synthetic generator (trend+seasonality+noise)
│   ├── forecast.py      # MA, SES, Holt, AR(p)
│   ├── metrics.py       # MAE/RMSE/MAPE + residual diagnostics
│   ├── backtest.py      # train/test split harness
│   └── cli.py           # argparse CLI (generate/forecast/backtest/plot)
└── tests/
    ├── test_series.py
    ├── test_forecast.py
    ├── test_metrics.py
    └── test_cli.py
```

## Tests

Stdlib `unittest` only (no pytest needed):

```bash
python3 -m unittest discover -v
```

Coverage highlights: SES on a constant series stays constant; Holt recovers
a perfect linear trend; AR(1)/AR(2) coefficient recovery on seeded synthetic
data; hand-computed MAE/RMSE/MAPE cases; moving-average known case; CLI
smoke tests for every subcommand (including that `plot` writes a real PNG).

## Notes

- Generated PNG/CSV outputs are git-ignored — the repo holds source only.
- Everything is deterministic for a given `--seed`.
