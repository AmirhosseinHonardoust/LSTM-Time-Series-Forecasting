import torch

from model import LSTMForecaster


def test_forward_output_shape_matches_horizon():
    model = LSTMForecaster(input_size=1, hidden_size=8, num_layers=1, dropout=0.0, horizon=5)
    x = torch.zeros(4, 20, 1)  # (batch, lookback, input_size)
    out = model(x)
    assert out.shape == (4, 5)
    assert out.dtype == torch.float32


def test_forward_respects_custom_hyperparams():
    model = LSTMForecaster(input_size=1, hidden_size=16, num_layers=2, dropout=0.1, horizon=12)
    x = torch.zeros(2, 30, 1)
    out = model(x)
    assert out.shape == (2, 12)


def test_forward_single_sample_batch():
    model = LSTMForecaster(hidden_size=8, num_layers=1, dropout=0.0, horizon=3)
    x = torch.zeros(1, 10, 1)
    out = model(x)
    assert out.shape == (1, 3)
