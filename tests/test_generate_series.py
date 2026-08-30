from generate_series import generate_series


def test_row_count_matches_date_range():
    df = generate_series("2020-01-01", "2020-01-10", seed=42)
    assert len(df) == 10
    assert list(df.columns) == ["date", "value"]


def test_deterministic_with_same_seed():
    df1 = generate_series("2020-01-01", "2020-06-30", seed=42)
    df2 = generate_series("2020-01-01", "2020-06-30", seed=42)
    assert (df1["value"] == df2["value"]).all()


def test_different_seeds_differ():
    df1 = generate_series("2020-01-01", "2020-06-30", seed=1)
    df2 = generate_series("2020-01-01", "2020-06-30", seed=2)
    assert not (df1["value"] == df2["value"]).all()


def test_values_are_finite():
    df = generate_series("2020-01-01", "2021-12-31", seed=7)
    assert df["value"].notna().all()
    assert df["value"].abs().max() < 1e6
