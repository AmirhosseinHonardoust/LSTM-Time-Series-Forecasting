"""Tests for configurable --date-col/--value-col and the friendly error raised
when a CSV is missing a required column, exercised the same way
test_train_eval_smoke.py does: via subprocess against the real CLI scripts.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "src" / "train_lstm.py"
EVAL_SCRIPT = REPO_ROOT / "src" / "evaluate.py"


def test_train_and_evaluate_accept_custom_column_names(tmp_path):
    from generate_series import generate_series

    df = generate_series("2020-01-01", "2020-12-31", seed=42)
    df = df.rename(columns={"date": "day", "value": "reading"})
    data_path = tmp_path / "series.csv"
    df.to_csv(data_path, index=False)

    outdir = tmp_path / "outputs"
    train_result = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--input",
            str(data_path),
            "--lookback",
            "20",
            "--horizon",
            "10",
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--outdir",
            str(outdir),
            "--device",
            "cpu",
            "--date-col",
            "day",
            "--value-col",
            "reading",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert train_result.returncode == 0, train_result.stderr

    eval_result = subprocess.run(
        [
            sys.executable,
            str(EVAL_SCRIPT),
            "--input",
            str(data_path),
            "--model",
            str(outdir / "best_lstm.pt"),
            "--outdir",
            str(outdir),
            "--device",
            "cpu",
            "--date-col",
            "day",
            "--value-col",
            "reading",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert eval_result.returncode == 0, eval_result.stderr


def test_train_rejects_csv_missing_value_column(tmp_path):
    data_path = tmp_path / "series.csv"
    data_path.write_text("date\n2020-01-01\n2020-01-02\n")

    result = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--input",
            str(data_path),
            "--outdir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "missing required column" in result.stderr
    assert "value" in result.stderr


def test_evaluate_rejects_csv_missing_date_column(tmp_path):
    from generate_series import generate_series

    # Build a valid checkpoint/scaler first via a normal training run.
    good_df = generate_series("2020-01-01", "2020-12-31", seed=42)
    good_path = tmp_path / "good.csv"
    good_df.to_csv(good_path, index=False)
    outdir = tmp_path / "outputs"
    subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--input",
            str(good_path),
            "--lookback",
            "20",
            "--horizon",
            "10",
            "--epochs",
            "1",
            "--outdir",
            str(outdir),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )

    bad_path = tmp_path / "bad.csv"
    bad_path.write_text("value\n1.0\n2.0\n")

    result = subprocess.run(
        [
            sys.executable,
            str(EVAL_SCRIPT),
            "--input",
            str(bad_path),
            "--model",
            str(outdir / "best_lstm.pt"),
            "--outdir",
            str(outdir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "missing required column" in result.stderr
    assert "date" in result.stderr
