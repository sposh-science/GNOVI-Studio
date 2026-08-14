import pandas as pd
import pytest

from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_column, numeric_xy


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


def test_numeric_column_extracts_clean_values():
    df = pd.DataFrame({"current": [1.0, 2.0, 3.0]})
    values = numeric_column(df, "current")
    assert list(values) == [1.0, 2.0, 3.0]


def test_numeric_column_drops_non_numeric_rows():
    df = pd.DataFrame({"current": ["1", "bad", "3", "4"]})
    values = numeric_column(df, "current")
    assert list(values) == [1, 3, 4]


def test_numeric_column_does_not_mutate_original_dataframe():
    df = pd.DataFrame({"current": ["1", "bad", "3"]})
    numeric_column(df, "current")
    assert df["current"].tolist() == ["1", "bad", "3"]


def test_numeric_column_raises_when_too_few_valid_points_remain():
    df = pd.DataFrame({"current": ["bad", "also bad"]})
    with pytest.raises(InsufficientNumericDataError):
        numeric_column(df, "current", min_points=1)


def test_numeric_column_missing_column_raises_key_error():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(KeyError):
        numeric_column(df, "does_not_exist")
