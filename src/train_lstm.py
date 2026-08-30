"""Train an LSTM forecaster on a daily time series and save the best checkpoint."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import LSTMForecaster
from scaler import JsonStandardScaler
from utils import (
    FloatArray,
    inverse_scale,
    mae,
    make_windows,
    mape,
    plot_forecast,
    raw_fit_cutoff,
    require_columns,
    resolve_device,
    rmse,
    scale_series,
)


def plot_curves(history: dict[str, list[float]], outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["train_loss"], label="train_loss")
    ax.plot(history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience in epochs")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto' (default, prefers CUDA/MPS then CPU), 'cpu', 'cuda', 'cuda:0', 'mps', ...",
    )
    ap.add_argument("--date-col", type=str, default="date", help="date column name in --input")
    ap.add_argument("--value-col", type=str, default="value", help="value column name in --input")
    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input series not found: {args.input}")
    if args.lookback <= 0 or args.horizon <= 0:
        raise ValueError("--lookback and --horizon must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.hidden_size <= 0 or args.num_layers <= 0:
        raise ValueError("--hidden-size and --num-layers must be positive")
    if args.patience <= 0:
        raise ValueError("--patience must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")


def load_series(
    input_path: str,
    lookback: int,
    horizon: int,
    date_col: str = "date",
    value_col: str = "value",
) -> pd.DataFrame:
    """Read the CSV, validate its columns, and fail fast if it's too short for the
    requested window sizes. Returns a frame with columns renamed to "date"/"value"
    so the rest of the pipeline doesn't need to know about custom column names."""
    df = pd.read_csv(input_path)
    require_columns(df, [date_col, value_col], input_path)
    df = df.rename(columns={date_col: "date", value_col: "value"})
    df["date"] = pd.to_datetime(df["date"])
    if len(df) <= lookback + horizon:
        raise ValueError(
            f"Series has {len(df)} rows, too short for lookback={lookback} + horizon={horizon}"
        )
    return df


def build_dataloaders(scaled: FloatArray, lookback: int, horizon: int, batch_size: int) -> tuple[
    DataLoader[tuple[torch.Tensor, ...]],
    DataLoader[tuple[torch.Tensor, ...]],
    FloatArray,
]:
    """Slice into windows, time-split 80/20 (no shuffle, so val is the most recent 20%)."""
    x, y = make_windows(scaled, lookback, horizon)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False)
    tr_ds = TensorDataset(torch.tensor(x_train[:, :, None]), torch.tensor(y_train))
    va_ds = TensorDataset(torch.tensor(x_val[:, :, None]), torch.tensor(y_val))
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    return tr_dl, va_dl, x


def train_one_epoch(
    model: LSTMForecaster,
    dl: DataLoader[tuple[torch.Tensor, ...]],
    opt: torch.optim.Optimizer,
    crit: nn.Module,
    desc: str,
    device: torch.device,
) -> float:
    model.train()
    tloss, n = 0.0, 0
    for xb, yb in tqdm(dl, desc=desc):
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        preds = model(xb)
        loss = crit(preds, yb)
        loss.backward()
        opt.step()
        tloss += loss.item() * xb.size(0)
        n += xb.size(0)
    return tloss / n


def validate_epoch(
    model: LSTMForecaster,
    dl: DataLoader[tuple[torch.Tensor, ...]],
    crit: nn.Module,
    desc: str,
    device: torch.device,
) -> float:
    model.eval()
    vloss, vn = 0.0, 0
    with torch.no_grad():
        for xb, yb in tqdm(dl, desc=desc):
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = crit(preds, yb)
            vloss += loss.item() * xb.size(0)
            vn += xb.size(0)
    return vloss / vn


def save_checkpoint(path: str, model: LSTMForecaster, args: argparse.Namespace) -> None:
    """Save weights plus the full window/architecture config so evaluate.py can
    reconstruct an identical model instead of guessing from its own CLI defaults."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "horizon": args.horizon,
            "lookback": args.lookback,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        path,
    )


def run_final_forecast(
    model: LSTMForecaster,
    x: FloatArray,
    df: pd.DataFrame,
    scaler: JsonStandardScaler,
    horizon: int,
    outdir: str,
    device: torch.device,
) -> dict[str, float]:
    """Forecast the last `horizon` steps of the input series (a backtest against the
    most recent known values, not a prediction of dates beyond the input CSV) and
    save the comparison plot."""
    model.eval()
    last_input = torch.tensor(x[-1][:, None]).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_scaled = model(last_input).cpu().numpy().flatten()
    pred = inverse_scale(pred_scaled, scaler)

    pred_start = len(df) - horizon
    forecast_path = os.path.join(outdir, "forecast_plot.png")
    plot_forecast(df["date"], df["value"], pred_start, pred, forecast_path)

    y_true = df["value"].values[-horizon:]
    return {"rmse": rmse(y_true, pred), "mae": mae(y_true, pred), "mape": mape(y_true, pred)}


def main() -> None:
    args = parse_args()
    validate_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[device] using {device}")

    df = load_series(args.input, args.lookback, args.horizon, args.date_col, args.value_col)
    series = df["value"].values.astype("float32")

    # Fit the scaler only on raw values that fall entirely within training
    # windows, so no validation-window statistic leaks into mean/std.
    cutoff = raw_fit_cutoff(len(series), args.lookback, args.horizon)
    scaled, scaler = scale_series(series, os.path.join(args.outdir, "scaler.json"), fit_upto=cutoff)
    tr_dl, va_dl, x = build_dataloaders(scaled, args.lookback, args.horizon, args.batch_size)

    model = LSTMForecaster(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        horizon=args.horizon,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()
    best_val = float("inf")
    stale = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_path = os.path.join(args.outdir, "best_lstm.pt")

    for ep in range(1, args.epochs + 1):
        tl = train_one_epoch(
            model, tr_dl, opt, crit, desc=f"Epoch {ep}/{args.epochs} [train]", device=device
        )
        vl = validate_epoch(
            model, va_dl, crit, desc=f"Epoch {ep}/{args.epochs} [val]", device=device
        )

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        print(f"[epoch {ep}] train_loss={tl:.4f} val_loss={vl:.4f}")

        if vl < best_val:
            best_val = vl
            stale = 0
            save_checkpoint(best_path, model, args)
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.")
                break

    plot_curves(history, os.path.join(args.outdir, "training_curves.png"))

    state = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"])

    r = run_final_forecast(model, x, df, scaler, args.horizon, args.outdir, device)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("[OK] Training complete. Metrics saved.")


if __name__ == "__main__":
    main()
