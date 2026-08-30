"""Shared helpers: sliding-window dataset prep, scaling, and error metrics."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from scaler import JsonStandardScaler

FloatArray = npt.NDArray[np.floating[Any]]


def make_windows(series: FloatArray, lookback: int, horizon: int) -> tuple[FloatArray, FloatArray]:
    """Slice a 1-D series into overlapping (X, y) windows for supervised learning.

    X[i] is ``lookback`` steps ending right before y[i], which is the next
    ``horizon`` steps.
    """
    x, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        x.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])
    return np.array(x), np.array(y)


def train_val_split_sizes(n_windows: int, test_size: float = 0.2) -> tuple[int, int]:
    """Mirror sklearn.model_selection.train_test_split's sizing for shuffle=False
    (it rounds the test count up via ceil) without importing sklearn here."""
    n_test = math.ceil(test_size * n_windows)
    n_train = n_windows - n_test
    return n_train, n_test


def raw_fit_cutoff(n_total: int, lookback: int, horizon: int, test_size: float = 0.2) -> int:
    """Number of leading raw series values fully contained within the training
    windows that make_windows() + a shuffle=False/test_size train_test_split
    would produce. Fitting the scaler on ``series[:cutoff]`` (via
    ``scale_series(..., fit_upto=cutoff)``) means it never sees a value that
    only appears in a validation window, avoiding leakage.
    """
    n_windows = n_total - lookback - horizon + 1
    n_train, _ = train_val_split_sizes(n_windows, test_size)
    return n_train - 1 + lookback + horizon


def scale_series(
    arr: FloatArray, out_path: str, fit_upto: int | None = None
) -> tuple[FloatArray, JsonStandardScaler]:
    """Fit a scaler on ``arr[:fit_upto]``, save it to ``out_path`` as JSON, and
    return the *whole* array scaled with those statistics.

    ``fit_upto`` (default: the full array) restricts the fit to a leading
    prefix so validation-window statistics can't leak into it -- pass
    ``raw_fit_cutoff(...)`` to align exactly with a downstream windowed
    train/val split.
    """
    cutoff = len(arr) if fit_upto is None else max(1, min(fit_upto, len(arr)))
    scaler = JsonStandardScaler()
    scaler.fit(arr[:cutoff])
    scaled = scaler.transform(arr)
    scaler.save(out_path)
    return scaled, scaler


def scale_with(arr: FloatArray, scaler: JsonStandardScaler) -> FloatArray:
    return scaler.transform(arr)


def inverse_scale(arr: FloatArray, scaler: JsonStandardScaler) -> FloatArray:
    return scaler.inverse_transform(arr)


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: FloatArray, y_pred: FloatArray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def plot_forecast(
    dates: pd.Series,
    values: pd.Series,
    pred_start_idx: int,
    preds: FloatArray,
    outpath: str,
    title: str = "Forecast vs Actual",
) -> None:
    """Plot the actual series with the forecasted horizon overlaid and save it.

    Shared by ``train_lstm.py`` (post-training backtest plot) and
    ``evaluate.py`` (standalone evaluation plot), previously duplicated in
    each with only the title differing.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, values, label="actual")
    fut_dates = dates[pred_start_idx : pred_start_idx + len(preds)]
    ax.plot(fut_dates, preds, label="forecast")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
