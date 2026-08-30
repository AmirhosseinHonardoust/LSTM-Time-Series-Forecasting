"""Evaluate a trained LSTM checkpoint on a series and refresh metrics/plot."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd
import torch

from model import LSTMForecaster
from scaler import JsonStandardScaler
from utils import (
    inverse_scale,
    mae,
    make_windows,
    mape,
    plot_forecast,
    require_columns,
    resolve_device,
    rmse,
    scale_with,
)

# Defaults mirror LSTMForecaster's own constructor defaults, used only as a
# fallback for checkpoints saved before this config was persisted.
_LEGACY_DEFAULTS: dict[str, int | float] = {
    "lookback": 60,
    "horizon": 30,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="must match the checkpoint's training lookback; defaults to the saved value",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="must match the checkpoint's training horizon; defaults to the saved value",
    )
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto' (default, prefers CUDA/MPS then CPU), 'cpu', 'cuda', 'cuda:0', 'mps', ...",
    )
    ap.add_argument("--date-col", type=str, default="date", help="date column name in --input")
    ap.add_argument("--value-col", type=str, default="value", help="value column name in --input")
    return ap.parse_args()


def load_checkpoint_config(state: dict[str, Any]) -> dict[str, int | float]:
    """Pull window/architecture config saved by train_lstm.py's save_checkpoint(),
    falling back to LSTMForecaster's defaults for older checkpoints that predate it."""
    return {key: state.get(key, default) for key, default in _LEGACY_DEFAULTS.items()}


def resolve_window_arg(name: str, cli_value: int | None, checkpoint_value: int) -> int:
    """Use the checkpoint's saved value unless the CLI explicitly overrides it, in
    which case the two must agree -- a mismatch means the model's weights were
    trained for a different window size than the one about to be used."""
    if cli_value is None:
        return checkpoint_value
    if cli_value != checkpoint_value:
        raise ValueError(
            f"--{name}={cli_value} does not match the checkpoint's training "
            f"{name}={checkpoint_value}. Omit --{name} to use the checkpoint's "
            f"value, or retrain with the desired {name}."
        )
    return cli_value


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

    device = resolve_device(args.device)
    print(f"[device] using {device}")

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    ckpt_cfg = load_checkpoint_config(state)
    lookback = resolve_window_arg("lookback", args.lookback, int(ckpt_cfg["lookback"]))
    horizon = resolve_window_arg("horizon", args.horizon, int(ckpt_cfg["horizon"]))

    df = pd.read_csv(args.input)
    require_columns(df, [args.date_col, args.value_col], args.input)
    df = df.rename(columns={args.date_col: "date", args.value_col: "value"})
    df["date"] = pd.to_datetime(df["date"])
    series = df["value"].values.astype("float32")

    if len(series) <= lookback + horizon:
        raise ValueError(
            f"Series has {len(series)} rows, too short for lookback={lookback} + horizon={horizon}"
        )

    scaler = JsonStandardScaler.load(scaler_path)
    scaled = scale_with(series, scaler)
    x, _ = make_windows(scaled, lookback, horizon)

    model = LSTMForecaster(
        hidden_size=int(ckpt_cfg["hidden_size"]),
        num_layers=int(ckpt_cfg["num_layers"]),
        dropout=float(ckpt_cfg["dropout"]),
        horizon=horizon,
    ).to(device)
    model.load_state_dict(state.get("model_state", state))
    model.eval()

    last_input = torch.tensor(x[-1][:, None]).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_scaled = model(last_input).cpu().numpy().flatten()
    pred = inverse_scale(pred_scaled, scaler)

    y_true = df["value"].values[-horizon:]
    r = {"rmse": rmse(y_true, pred), "mae": mae(y_true, pred), "mape": mape(y_true, pred)}
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(r, f, indent=2)

    pred_start = len(df) - horizon
    plot_forecast(
        df["date"],
        df["value"],
        pred_start,
        pred,
        os.path.join(args.outdir, "forecast_plot.png"),
        title="Forecast vs Actual (Evaluate)",
    )
    print("[OK] Evaluation complete. Metrics saved.")


if __name__ == "__main__":
    main()
