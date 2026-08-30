"""Shared helpers: sliding-window dataset prep, scaling, and error metrics."""

from __future__ import annotations

import numpy as np

from scaler import JsonStandardScaler


def make_windows(series: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice a 1-D series into overlapping (X, y) windows for supervised learning.

    X[i] is ``lookback`` steps ending right before y[i], which is the next
    ``horizon`` steps.
    """
    x, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        x.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])
    return np.array(x), np.array(y)


def scale_series(
    arr: np.ndarray, out_path: str, fit_frac: float = 1.0
) -> tuple[np.ndarray, JsonStandardScaler]:
    """Fit a scaler on the first ``fit_frac`` of ``arr``, save it to ``out_path`` as
    JSON, and return the *whole* array scaled with those statistics.

    ``fit_frac`` < 1.0 avoids leaking validation-window statistics into the fit:
    pass the same fraction used for the train/val split (e.g. 0.8 to match a
    time-ordered 80/20 split) so only the training portion informs mean/std.
    """
    split = max(1, int(len(arr) * fit_frac))
    scaler = JsonStandardScaler()
    scaler.fit(arr[:split])
    scaled = scaler.transform(arr)
    scaler.save(out_path)
    return scaled, scaler


def scale_with(arr: np.ndarray, scaler: JsonStandardScaler) -> np.ndarray:
    return scaler.transform(arr)


def inverse_scale(arr: np.ndarray, scaler: JsonStandardScaler) -> np.ndarray:
    return scaler.inverse_transform(arr)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
