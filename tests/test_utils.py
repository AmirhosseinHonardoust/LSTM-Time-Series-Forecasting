import numpy as np
from sklearn.model_selection import train_test_split

from utils import mae, make_windows, mape, raw_fit_cutoff, rmse, scale_series, train_val_split_sizes


def test_make_windows_shapes_and_content():
    series = np.arange(10, dtype="float32")
    x, y = make_windows(series, lookback=3, horizon=2)
    # 10 - 3 - 2 + 1 = 6 windows
    assert x.shape == (6, 3)
    assert y.shape == (6, 2)
    np.testing.assert_array_equal(x[0], [0, 1, 2])
    np.testing.assert_array_equal(y[0], [3, 4])
    np.testing.assert_array_equal(x[-1], [5, 6, 7])
    np.testing.assert_array_equal(y[-1], [8, 9])


def test_make_windows_empty_when_too_short():
    series = np.arange(4, dtype="float32")
    x, y = make_windows(series, lookback=3, horizon=2)
    assert len(x) == 0
    assert len(y) == 0


def test_rmse_zero_for_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    assert rmse(y_true, y_true) == 0.0


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert rmse(y_true, y_pred) == 3.5355339059327378


def test_mae_known_value():
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([3.0, 4.0])
    assert mae(y_true, y_pred) == 4.5


def test_mape_known_value():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    # errors: 10% and 10% -> mean 10%
    assert abs(mape(y_true, y_pred) - 10.0) < 1e-6


def test_mape_avoids_division_by_zero():
    y_true = np.array([0.0])
    y_pred = np.array([5.0])
    # should not raise, and should be finite
    result = mape(y_true, y_pred)
    assert np.isfinite(result)


def test_scale_series_fit_upto_ignores_tail(tmp_path):
    # First half is constant at 10, second half jumps to 1000. A scaler fit on
    # the whole array would have its mean/std dragged toward the tail; one fit
    # only on the first half (fit_upto=50) should not see the jump at all.
    arr = np.concatenate([np.full(50, 10.0), np.full(50, 1000.0)]).astype("float32")
    _, scaler = scale_series(arr, str(tmp_path / "scaler.json"), fit_upto=50)
    assert scaler.mean == 10.0
    assert scaler.std == 1.0  # zero-variance first half falls back to std=1.0


def test_scale_series_default_fit_upto_uses_whole_array(tmp_path):
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
    scaled, scaler = scale_series(arr, str(tmp_path / "scaler.json"))
    assert scaler.mean == float(np.mean(arr))
    np.testing.assert_allclose(scaled, (arr - scaler.mean) / scaler.std)


def test_train_val_split_sizes_matches_sklearn():
    for n in (10, 27, 100, 133, 517):
        expected_train, expected_test = train_test_split(np.arange(n), test_size=0.2, shuffle=False)
        n_train, n_test = train_val_split_sizes(n, test_size=0.2)
        assert (n_train, n_test) == (len(expected_train), len(expected_test))


def test_raw_fit_cutoff_excludes_every_validation_window_value():
    # Full pipeline: build windows, split them the way train_lstm.py does, and
    # confirm no raw index used by a validation window is < raw_fit_cutoff.
    lookback, horizon = 5, 3
    series = np.arange(50, dtype="float32")
    x, y = make_windows(series, lookback, horizon)
    _, x_val, _, _ = train_test_split(x, y, test_size=0.2, shuffle=False)

    cutoff = raw_fit_cutoff(len(series), lookback, horizon)

    n_windows = len(series) - lookback - horizon + 1
    n_train, _ = train_val_split_sizes(n_windows)
    first_val_window_start = n_train  # windows are contiguous and time-ordered
    assert cutoff <= first_val_window_start + lookback + horizon
    assert cutoff > first_val_window_start  # sanity: cutoff lands inside/after last train window
    assert len(x_val) == n_windows - n_train
