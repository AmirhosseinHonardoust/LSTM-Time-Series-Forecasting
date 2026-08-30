"""A minimal, JSON-serializable standard scaler.

Replaces the previous ``joblib``/pickle-based scaler. Pickle files can execute
arbitrary code on load; a plain mean/std pair does not need that risk and is
human-readable. Behavior is equivalent to
``sklearn.preprocessing.StandardScaler`` for the 1-D case used in this project
(population standard deviation, ddof=0).
"""

from __future__ import annotations

import json
import os

import numpy as np


class JsonStandardScaler:
    """Standardizes a 1-D array to zero mean / unit variance and back."""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def fit(self, arr: np.ndarray) -> JsonStandardScaler:
        self.mean = float(np.mean(arr))
        std = float(np.std(arr))
        self.std = std if std > 0 else 1.0
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.std

    def fit_transform(self, arr: np.ndarray) -> np.ndarray:
        return self.fit(arr).transform(arr)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.std + self.mean

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"mean": self.mean, "std": self.std}, f)

    @classmethod
    def load(cls, path: str) -> JsonStandardScaler:
        with open(path) as f:
            data = json.load(f)
        return cls(mean=data["mean"], std=data["std"])
