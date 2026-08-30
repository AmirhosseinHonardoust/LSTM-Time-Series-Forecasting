"""Unit tests for evaluate.py's checkpoint-config resolution (no training/torch
compute involved, but the module import itself requires torch to be installed,
same as evaluate.py in production use).
"""

import pytest

from evaluate import load_checkpoint_config, resolve_window_arg


def test_load_checkpoint_config_uses_saved_values():
    state = {
        "lookback": 45,
        "horizon": 14,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.1,
        "model_state": {},
    }
    cfg = load_checkpoint_config(state)
    assert cfg == {
        "lookback": 45,
        "horizon": 14,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.1,
    }


def test_load_checkpoint_config_defaults_for_legacy_checkpoints():
    cfg = load_checkpoint_config({"model_state": {}})
    assert cfg == {
        "lookback": 60,
        "horizon": 30,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
    }


def test_resolve_window_arg_defaults_to_checkpoint_value():
    assert resolve_window_arg("lookback", None, 45) == 45


def test_resolve_window_arg_accepts_matching_override():
    assert resolve_window_arg("horizon", 14, 14) == 14


def test_resolve_window_arg_rejects_mismatched_override():
    with pytest.raises(ValueError, match="does not match the checkpoint"):
        resolve_window_arg("lookback", 60, 45)
