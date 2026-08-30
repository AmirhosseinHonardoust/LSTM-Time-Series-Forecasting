import json

import numpy as np
from sklearn.preprocessing import StandardScaler

from scaler import JsonStandardScaler


def test_matches_sklearn_standard_scaler():
    rng = np.random.default_rng(0)
    arr = rng.normal(100, 20, size=500).astype("float32")

    sk = StandardScaler()
    sk_scaled = sk.fit_transform(arr.reshape(-1, 1)).flatten()

    js = JsonStandardScaler()
    js_scaled = js.fit_transform(arr)

    np.testing.assert_allclose(sk_scaled, js_scaled, rtol=1e-5)
    np.testing.assert_allclose(sk.mean_[0], js.mean, rtol=1e-5)
    np.testing.assert_allclose(sk.scale_[0], js.std, rtol=1e-5)


def test_inverse_transform_round_trips():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
    scaler = JsonStandardScaler()
    scaled = scaler.fit_transform(arr)
    recovered = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(recovered, arr, rtol=1e-5)


def test_save_and_load_round_trip(tmp_path):
    arr = np.array([10.0, 20.0, 30.0], dtype="float32")
    scaler = JsonStandardScaler()
    scaler.fit(arr)
    path = tmp_path / "scaler.json"
    scaler.save(str(path))

    loaded = JsonStandardScaler.load(str(path))
    assert loaded.mean == scaler.mean
    assert loaded.std == scaler.std

    with open(path) as f:
        data = json.load(f)
    assert set(data.keys()) == {"mean", "std"}


def test_handles_zero_variance_without_division_by_zero():
    arr = np.array([5.0, 5.0, 5.0], dtype="float32")
    scaler = JsonStandardScaler()
    scaled = scaler.fit_transform(arr)
    assert np.all(np.isfinite(scaled))
    assert scaler.std == 1.0
