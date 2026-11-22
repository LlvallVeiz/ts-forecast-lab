"""Command-line interface.

Subcommands
-----------
generate   Create a synthetic series (prints summary, optional CSV out)
forecast   Fit a forecaster on a synthetic series and print forecasts
backtest   Train/test split backtest with MAE/RMSE/MAPE + diagnostics
plot       Save an actual-vs-forecast PNG (matplotlib, Agg backend)

Run as ``python3 -m tsforecast.cli <subcommand> ...`` or via the
``main()`` entry point.  ``main(argv)`` returns an exit code (0 = ok)
so tests can call it directly without spawning a subprocess.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .backtest import backtest as run_backtest
from .forecast import FORECASTERS
from .series import SeriesConfig, generate_series


def _add_series_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=120, help="series length")
    parser.add_argument("--level", type=float, default=100.0)
    parser.add_argument("--slope", type=float, default=0.5)
    parser.add_argument("--amplitude", type=float, default=10.0)
    parser.add_argument("--period", type=float, default=12.0)
    parser.add_argument("--sigma", type=float, default=2.0, help="noise std")
    parser.add_argument("--seed", type=int, default=42)


def _add_method_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--method",
        choices=sorted(FORECASTERS),
        default="holt",
        help="forecasting method",
    )
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.3, help="SES/Holt alpha")
    parser.add_argument("--beta", type=float, default=0.1, help="Holt beta")
    parser.add_argument("--window", type=int, default=5, help="MA window")
    parser.add_argument("--p", type=int, default=2, help="AR order")


def _config_from(args) -> SeriesConfig:
    return SeriesConfig(
        n=args.n,
        level=args.level,
        slope=args.slope,
        amplitude=args.amplitude,
        period=args.period,
        sigma=args.sigma,
        seed=args.seed,
    )


def _method_kwargs(args) -> dict:
    if args.method == "ma":
        return {"window": args.window}
    if args.method == "ses":
        return {"alpha": args.alpha}
    if args.method == "holt":
        return {"alpha": args.alpha, "beta": args.beta}
    if args.method == "ar":
        return {"p": args.p}
    return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsforecast",
        description="Time-series forecasting lab (from scratch).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="generate a synthetic series")
    _add_series_args(p_gen)
    p_gen.add_argument("--out", default=None, help="optional CSV output path")
    p_gen.add_argument("--show", type=int, default=5, help="how many values to print")

    p_fc = sub.add_parser("forecast", help="forecast from a synthetic series")
    _add_series_args(p_fc)
    _add_method_args(p_fc)

    p_bt = sub.add_parser("backtest", help="train/test backtest")
    _add_series_args(p_bt)
    _add_method_args(p_bt)

    p_plot = sub.add_parser("plot", help="save actual-vs-forecast PNG")
    _add_series_args(p_plot)
    _add_method_args(p_plot)
    p_plot.add_argument("--out", default="forecast.png", help="PNG output path")

    return parser


def cmd_generate(args) -> int:
    y = generate_series(_config_from(args))
    print(
        f"generated n={y.size} seed={args.seed} "
        f"mean={y.mean():.4f} std={y.std():.4f} min={y.min():.4f} max={y.max():.4f}"
    )
    k = max(0, min(args.show, y.size))
    if k:
        print("first values: " + ", ".join(f"{v:.4f}" for v in y[:k]))
    if args.out:
        np.savetxt(args.out, y, delimiter=",", header="value", comments="")
        print(f"wrote {args.out}")
    return 0


def cmd_forecast(args) -> int:
    y = generate_series(_config_from(args))
    fc = FORECASTERS[args.method](y, horizon=args.horizon, **_method_kwargs(args))
    print(f"method={args.method} horizon={args.horizon} n={y.size}")
    for i, v in enumerate(fc, start=1):
        print(f"  step {i}: {v:.4f}")
    return 0


def cmd_backtest(args) -> int:
    y = generate_series(_config_from(args))
    result = run_backtest(y, method=args.method, horizon=args.horizon, **_method_kwargs(args))
    print(result.summary())
    return 0


def cmd_plot(args) -> int:
    # Import here so headless environments only pay for matplotlib on plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = generate_series(_config_from(args))
    result = run_backtest(y, method=args.method, horizon=args.horizon, **_method_kwargs(args))

    x_all = np.arange(y.size)
    x_fc = np.arange(result.train_size, result.train_size + result.horizon)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x_all, y, label="actual", linewidth=1.2)
    ax.plot(x_fc, result.forecasts, label=f"forecast ({args.method})", linewidth=1.6)
    ax.axvline(result.train_size - 0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title(f"Actual vs forecast — {args.method} (horizon={args.horizon})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"saved plot to {args.out} ({result.summary()})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "generate": cmd_generate,
        "forecast": cmd_forecast,
        "backtest": cmd_backtest,
        "plot": cmd_plot,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
