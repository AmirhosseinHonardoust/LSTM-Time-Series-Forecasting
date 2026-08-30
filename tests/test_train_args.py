import argparse

import pytest

from train_lstm import validate_args


def _base_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        input="",  # filled in per-test with a real path
        lookback=60,
        horizon=30,
        epochs=30,
        batch_size=64,
        lr=1e-3,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        patience=5,
        outdir="outputs",
        seed=42,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_valid_args_do_not_raise(tmp_path):
    csv = tmp_path / "series.csv"
    csv.write_text("date,value\n2020-01-01,1.0\n")
    validate_args(_base_args(input=str(csv)))  # should not raise


def test_missing_input_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_args(_base_args(input=str(tmp_path / "missing.csv")))


@pytest.mark.parametrize("field,value", [("lookback", 0), ("lookback", -1), ("horizon", 0)])
def test_nonpositive_lookback_horizon_raises(tmp_path, field, value):
    csv = tmp_path / "series.csv"
    csv.write_text("date,value\n2020-01-01,1.0\n")
    with pytest.raises(ValueError, match="lookback.*horizon"):
        validate_args(_base_args(input=str(csv), **{field: value}))


def test_nonpositive_batch_size_raises(tmp_path):
    csv = tmp_path / "series.csv"
    csv.write_text("date,value\n2020-01-01,1.0\n")
    with pytest.raises(ValueError, match="batch-size"):
        validate_args(_base_args(input=str(csv), batch_size=0))


@pytest.mark.parametrize("field,value", [("hidden_size", 0), ("num_layers", 0)])
def test_nonpositive_hidden_size_or_num_layers_raises(tmp_path, field, value):
    csv = tmp_path / "series.csv"
    csv.write_text("date,value\n2020-01-01,1.0\n")
    with pytest.raises(ValueError, match="hidden-size.*num-layers"):
        validate_args(_base_args(input=str(csv), **{field: value}))


def test_nonpositive_patience_raises(tmp_path):
    csv = tmp_path / "series.csv"
    csv.write_text("date,value\n2020-01-01,1.0\n")
    with pytest.raises(ValueError, match="patience"):
        validate_args(_base_args(input=str(csv), patience=0))


def test_nonpositive_epochs_raises(tmp_path):
    csv = tmp_path / "series.csv"
    csv.write_text("date,value\n2020-01-01,1.0\n")
    with pytest.raises(ValueError, match="epochs"):
        validate_args(_base_args(input=str(csv), epochs=0))


@pytest.mark.parametrize("lr", [0, -0.1])
def test_nonpositive_lr_raises(tmp_path, lr):
    csv = tmp_path / "series.csv"
    csv.write_text("date,value\n2020-01-01,1.0\n")
    with pytest.raises(ValueError, match="lr"):
        validate_args(_base_args(input=str(csv), lr=lr))
