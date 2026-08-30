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
    return ap.parse_args()


def load_checkpoint_config(state: dict) -> dict[str, int | float]:
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

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    ckpt_cfg = load_checkpoint_config(state)
    lookback = resolve_window_arg("lookback", args.lookback, int(ckpt_cfg["lookback"]))
    horizon = resolve_window_arg("horizon", args.horizon, int(ckpt_cfg["horizon"]))

    df = pd.read_csv(args.input, parse_dates=["date"])
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
    )
    model.load_state_dict(state.get("model_state", state))
    model.eval()

    last_input = torch.tensor(x[-1][:, None]).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(last_input).numpy().flatten()
    pred = inverse_scale(pred_scaled, scaler)

    y_true = df["value"].values[-horizon:]
    r = {"rmse": rmse(y_true, pred), "mae": mae(y_true, pred), "mape": mape(y_true, pred)}
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(r, f, indent=2)

    dates = df["date"]
    pred_start = len(df) - horizon
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, df["value"], label="actual")
    ax.plot(dates[pred_start : pred_start + horizon], pred, label="forecast")
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
