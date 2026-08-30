"""Generate a synthetic daily time series with trend, seasonality, and shock events."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def generate_series(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """Build a daily series: linear trend + weekly/yearly seasonality + noise + shocks."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    trend = np.linspace(50, 200, n)
    weekly = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
    yearly = 20 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = rng.normal(0, 5, size=n)
    series = trend + weekly + yearly + noise
    for _ in range(max(3, n // 300)):
        idx = rng.integers(0, n)
        series[idx : idx + 3] += rng.uniform(20, 60)
    for _ in range(max(2, n // 400)):
        idx = rng.integers(0, n)
        series[idx : idx + 2] -= rng.uniform(15, 40)
    return pd.DataFrame({"date": dates, "value": series})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default="2025-12-31")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/daily_series.csv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = generate_series(args.start, args.end, args.seed)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} with {len(df):,} rows")


if __name__ == "__main__":
    main()
