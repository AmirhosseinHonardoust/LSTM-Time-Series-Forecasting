"""Evaluate a trained LSTM checkpoint on a series and refresh metrics/plot."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from model import LSTMForecaster
from scaler import JsonStandardScaler
from utils import inverse_scale, mae, make_windows, mape, rmse, scale_with


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--outdir", type=str, default="outputs")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input series not found: {args.input}")
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    os.makedirs(args.outdir, exist_ok=True)

    scaler_path = os.path.join(args.outdir, "scaler.json")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Run train_lstm.py first, "
            "or pass --outdir pointing at the training run's output directory."
        )

    df = pd.read_csv(args.input, parse_dates=["date"])
    series = df["value"].values.astype("float32")

    if len(series) <= args.lookback + args.horizon:
        raise ValueError(
            f"Series has {len(series)} rows, too short for "
            f"lookback={args.lookback} + horizon={args.horizon}"
        )

    scaler = JsonStandardScaler.load(scaler_path)
    scaled = scale_with(series, scaler)
    x, _ = make_windows(scaled, args.lookback, args.horizon)

    model = LSTMForecaster(horizon=args.horizon)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state", state))
    model.eval()

    last_input = torch.tensor(x[-1][:, None]).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(last_input).numpy().flatten()
    pred = inverse_scale(pred_scaled, scaler)

    y_true = df["value"].values[-args.horizon :]
    r = {"rmse": rmse(y_true, pred), "mae": mae(y_true, pred), "mape": mape(y_true, pred)}
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(r, f, indent=2)

    dates = df["date"]
    pred_start = len(df) - args.horizon
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, df["value"], label="actual")
    ax.plot(dates[pred_start : pred_start + args.horizon], pred, label="forecast")
    ax.set_title("Forecast vs Actual (Evaluate)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "forecast_plot.png"), dpi=160)
    plt.close(fig)
    print("[OK] Evaluation complete. Metrics saved.")


if __name__ == "__main__":
    main()
