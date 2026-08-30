"""End-to-end smoke test: train for a couple epochs on tiny synthetic data, then
evaluate the resulting checkpoint. Exercises the full pipeline (windows -> scaler
-> model -> metrics) the way the CLI scripts do, without shelling out, so it's
fast and Windows/CI-friendly.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_train_then_evaluate_produce_metrics(tmp_path):
    from generate_series import generate_series

    data_path = tmp_path / "series.csv"
    df = generate_series("2020-01-01", "2020-12-31", seed=42)
    df.to_csv(data_path, index=False)

    outdir = tmp_path / "outputs"
    train_script = REPO_ROOT / "src" / "train_lstm.py"
    eval_script = REPO_ROOT / "src" / "evaluate.py"

    train_result = subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--input",
            str(data_path),
            "--lookback",
            "20",
            "--horizon",
            "10",
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--outdir",
            str(outdir),
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert train_result.returncode == 0, train_result.stderr

    assert (outdir / "best_lstm.pt").is_file()
    assert (outdir / "scaler.json").is_file()
    assert (outdir / "metrics.json").is_file()
    assert (outdir / "training_curves.png").is_file()
    assert (outdir / "forecast_plot.png").is_file()

    eval_result = subprocess.run(
        [
            sys.executable,
            str(eval_script),
            "--input",
            str(data_path),
            "--model",
            str(outdir / "best_lstm.pt"),
            "--lookback",
            "20",
            "--horizon",
            "10",
            "--outdir",
            str(outdir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert eval_result.returncode == 0, eval_result.stderr

    import json

    with open(outdir / "metrics.json") as f:
        metrics = json.load(f)
    assert set(metrics.keys()) == {"rmse", "mae", "mape"}
    for v in metrics.values():
        assert v == v  # not NaN
        assert v >= 0


def test_train_rejects_series_too_short_for_lookback_horizon(tmp_path):
    from generate_series import generate_series

    data_path = tmp_path / "tiny.csv"
    generate_series("2020-01-01", "2020-01-10", seed=1).to_csv(data_path, index=False)

    train_script = REPO_ROOT / "src" / "train_lstm.py"
    result = subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--input",
            str(data_path),
            "--lookback",
            "60",
            "--horizon",
            "30",
            "--epochs",
            "1",
            "--outdir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "too short" in result.stderr
