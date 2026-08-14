import pandas as pd
import pytest

from gnovi_plot.data.dataset import Dataset


def test_dataset_wraps_dataframe_and_metadata():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    dataset = Dataset(name="sample", dataframe=df, source_path="sample.csv")

    assert dataset.name == "sample"
    assert dataset.source_path == "sample.csv"
    assert dataset.columns == ["x", "y"]
    assert dataset.row_count == 2
    assert dataset.dataframe is df


def test_dataset_requires_a_dataframe():
    with pytest.raises(TypeError):
        Dataset(name="bad", dataframe=[1, 2, 3])


def test_dataset_requires_a_name():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError):
        Dataset(name="", dataframe=df)


def test_dataset_ids_are_unique():
    df = pd.DataFrame({"x": [1]})
    a = Dataset(name="a", dataframe=df)
    b = Dataset(name="b", dataframe=df)
    assert a.id != b.id
