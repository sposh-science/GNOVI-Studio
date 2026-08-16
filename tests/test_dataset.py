import pandas as pd
import pytest

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.transforms import CalculatedColumnInfo, Transformation


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


# --- Project-save serialization -----------------------------------------------


def _dataset_with_history():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    dataset = Dataset(name="sample", dataframe=df, source_path="/data/sample.csv", metadata={"delimiter": ","})
    dataset.add_calculated_column("z", "x + y")
    dataset.exclude_rows([0])
    return dataset


def test_dataset_to_dict_never_includes_dataframes():
    dataset = _dataset_with_history()
    data = dataset.to_dict()
    assert "dataframe" not in data
    assert "raw_dataframe" not in data


def test_dataset_to_dict_from_project_data_round_trips_raw_and_working_frames():
    dataset = _dataset_with_history()
    data = dataset.to_dict()

    restored = Dataset.from_project_data(
        data,
        raw_dataframe=dataset.raw_dataframe.copy(),
        working_dataframe=dataset.dataframe.copy(),
    )

    assert restored.id == dataset.id
    pd.testing.assert_frame_equal(restored.raw_dataframe, dataset.raw_dataframe)
    pd.testing.assert_frame_equal(restored.dataframe, dataset.dataframe)
    # Raw and working stay genuinely distinct after reload -- the exclude_rows
    # transformation is baked into working.csv only, never raw.csv.
    assert restored.raw_row_count == 4
    assert restored.row_count == 3


def test_dataset_from_project_data_mutating_working_frame_does_not_affect_raw():
    dataset = _dataset_with_history()
    data = dataset.to_dict()
    restored = Dataset.from_project_data(
        data, raw_dataframe=dataset.raw_dataframe.copy(), working_dataframe=dataset.dataframe.copy()
    )

    restored.reset_working_data()

    assert restored.raw_row_count == 4
    assert restored.row_count == 4  # reset restores the raw row count
    assert "z" not in restored.dataframe.columns  # calculated column dropped by reset


def test_dataset_to_dict_preserves_calculated_column_metadata_and_formula():
    dataset = _dataset_with_history()
    data = dataset.to_dict()

    assert set(data["calculated_columns"].keys()) == {"z"}
    info = data["calculated_columns"]["z"]
    assert info["formula"] == "x + y"
    assert info["source_columns"] == ["x", "y"]

    restored = Dataset.from_project_data(
        data, raw_dataframe=dataset.raw_dataframe.copy(), working_dataframe=dataset.dataframe.copy()
    )
    assert isinstance(restored.calculated_columns["z"], CalculatedColumnInfo)
    assert restored.calculated_columns["z"].formula == "x + y"


def test_dataset_to_dict_preserves_full_transformation_history_order_and_kind():
    dataset = _dataset_with_history()
    data = dataset.to_dict()

    kinds = [t["kind"] for t in data["transformations"]]
    assert kinds == ["calculated_column", "exclude_rows"]

    restored = Dataset.from_project_data(
        data, raw_dataframe=dataset.raw_dataframe.copy(), working_dataframe=dataset.dataframe.copy()
    )
    assert len(restored.transformations) == 2
    assert all(isinstance(t, Transformation) for t in restored.transformations)
    assert restored.transformations[0].kind == "calculated_column"
    assert restored.transformations[1].detail["row_positions"] == [0]


def test_dataset_source_path_is_retained_as_provenance_only():
    dataset = _dataset_with_history()
    data = dataset.to_dict()
    assert data["source_path"] == "/data/sample.csv"

    # Reconstruction never touches the filesystem -- an unreachable path is
    # fine, it's carried through as plain metadata, not re-read.
    data["source_path"] = "/this/path/does/not/exist.csv"
    restored = Dataset.from_project_data(
        data, raw_dataframe=dataset.raw_dataframe.copy(), working_dataframe=dataset.dataframe.copy()
    )
    assert restored.source_path == "/this/path/does/not/exist.csv"
    assert restored.row_count == dataset.row_count


def test_dataset_from_project_data_works_with_no_source_path_at_all():
    dataset = Dataset(name="no-source", dataframe=pd.DataFrame({"x": [1, 2]}))
    data = dataset.to_dict()
    assert data["source_path"] is None

    restored = Dataset.from_project_data(
        data, raw_dataframe=dataset.raw_dataframe.copy(), working_dataframe=dataset.dataframe.copy()
    )
    assert restored.source_path is None
