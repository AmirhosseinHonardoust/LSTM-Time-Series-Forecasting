import numpy as np

from utils import mae, make_windows, mape, rmse, scale_series


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


def test_scale_series_fit_frac_ignores_tail(tmp_path):
    # First half is constant at 10, second half jumps to 1000. A scaler fit on
    # the whole array would have its mean/std dragged toward the tail; one fit
    # only on the first half (fit_frac=0.5) should not see the jump at all.
    arr = np.concatenate([np.full(50, 10.0), np.full(50, 1000.0)]).astype("float32")
    _, scaler = scale_series(arr, str(tmp_path / "scaler.json"), fit_frac=0.5)
    assert scaler.mean == 10.0
    assert scaler.std == 1.0  # zero-variance first half falls back to std=1.0


def test_scale_series_default_fit_frac_uses_whole_array(tmp_path):
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
    scaled, scaler = scale_series(arr, str(tmp_path / "scaler.json"))
    assert scaler.mean == float(np.mean(arr))
    np.testing.assert_allclose(scaled, (arr - scaler.mean) / scaler.std)
