import pandas as pd
import pytest

from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_xy


def test_numeric_xy_extracts_clean_pairs():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    x, y = numeric_xy(df, "x", "y")
    assert list(x) == [1, 2, 3]
    assert list(y) == [4, 5, 6]


def test_numeric_xy_drops_non_numeric_rows():
    df = pd.DataFrame(
        {
            "x": ["1", "bad", "3", "4"],
            "y": ["10", "20", "n/a", "40"],
        }
    )
    x, y = numeric_xy(df, "x", "y", min_points=2)
    assert list(x) == [1, 4]
    assert list(y) == [10, 40]


def test_numeric_xy_does_not_mutate_original_dataframe():
    df = pd.DataFrame({"x": ["1", "bad"], "y": ["10", "20"]})
    numeric_xy(df, "x", "y", min_points=1)
    assert df["x"].tolist() == ["1", "bad"]
    assert df["y"].tolist() == ["10", "20"]


def test_numeric_xy_raises_when_too_few_valid_points_remain():
    df = pd.DataFrame({"x": ["1", "bad", "also bad"], "y": ["10", "20", "30"]})
    with pytest.raises(InsufficientNumericDataError):
        numeric_xy(df, "x", "y", min_points=2)


def test_numeric_xy_missing_column_raises_key_error():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(KeyError):
        numeric_xy(df, "x", "does_not_exist")
