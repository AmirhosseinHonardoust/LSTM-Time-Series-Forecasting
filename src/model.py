"""LSTM forecasting model definition.

Single source of truth for the model architecture, shared by both
``train_lstm.py`` and ``evaluate.py`` (previously duplicated in each).
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Multi-step direct forecaster: LSTM encoder + linear head over the last hidden state."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 30,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.fc(h_last)
