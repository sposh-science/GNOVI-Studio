import pandas as pd
import pytest

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d", rows=10):
    df = pd.DataFrame({"x": list(range(rows)), "y": [float(i) * 2 for i in range(rows)]})
    return Dataset(name=name, dataframe=df)


# --- Raw vs. working data ---------------------------------------------------


def test_dataframe_and_raw_dataframe_start_identical():
    df = pd.DataFrame({"x": [1, 2, 3]})
    dataset = Dataset(name="d", dataframe=df)
    assert dataset.dataframe is df
    assert dataset.raw_dataframe is df


def test_raw_dataframe_survives_calculated_column():
    dataset = _make_dataset(rows=5)
    original = dataset.raw_dataframe.copy(deep=True)
    dataset.add_calculated_column("y2", "x * 2")
    pd.testing.assert_frame_equal(dataset.raw_dataframe, original)
    assert "y2" not in dataset.raw_dataframe.columns
    assert "y2" in dataset.dataframe.columns


def test_raw_dataframe_survives_exclude_and_keep():
    dataset = _make_dataset(rows=10)
    original = dataset.raw_dataframe.copy(deep=True)
    dataset.exclude_rows([0, 1])
    dataset.keep_rows([0, 1])
    pd.testing.assert_frame_equal(dataset.raw_dataframe, original)
    assert dataset.raw_row_count == 10


# --- Calculated columns -----------------------------------------------------


def test_calculated_column_appears_in_working_data_and_is_plottable():
    dataset = _make_dataset(rows=5)
    dataset.add_calculated_column("y2", "x * 2")
    assert "y2" in dataset.columns

    series = PlotSeries.line(dataset, "x", "y2")
    assert list(series.dataframe["y2"]) == [0, 2, 4, 6, 8]


def test_calculated_column_metadata_is_retained():
    dataset = _make_dataset(rows=4)
    info = dataset.add_calculated_column("y2", "x * 2")
    assert info.name == "y2"
    assert info.formula == "x * 2"
    assert info.source_columns == ["x"]
    assert dataset.calculated_columns["y2"] is info


def test_calculated_column_rejects_duplicate_name():
    dataset = _make_dataset(rows=4)
    with pytest.raises(ValueError):
        dataset.add_calculated_column("x", "x * 2")
    dataset.add_calculated_column("y2", "x * 2")
    with pytest.raises(ValueError):
        dataset.add_calculated_column("y2", "x * 3")


def test_calculated_column_can_reference_another_calculated_column():
    dataset = _make_dataset(rows=4)
    dataset.add_calculated_column("y2", "x * 2")
    dataset.add_calculated_column("y3", "y2 * 2")
    assert list(dataset.dataframe["y3"]) == [0, 4, 8, 12]


# --- Row selection operations ------------------------------------------------


def test_exclude_selected_range_removes_rows_from_working_data_only():
    dataset = _make_dataset(rows=10)
    dataset.exclude_rows([3, 4, 5])
    assert dataset.row_count == 7
    assert dataset.raw_row_count == 10
    assert list(dataset.dataframe["x"]) == [0, 1, 2, 6, 7, 8, 9]


def test_keep_only_selected_range_restricts_working_data():
    dataset = _make_dataset(rows=10)
    dataset.keep_rows([2, 3, 4])
    assert dataset.row_count == 3
    assert list(dataset.dataframe["x"]) == [2, 3, 4]


def test_multiple_exclusions_compound():
    dataset = _make_dataset(rows=10)
    dataset.exclude_rows([0, 1])
    dataset.exclude_rows([0])  # now positional row 0 of the *remaining* 8 rows (original x=2)
    assert list(dataset.dataframe["x"]) == [3, 4, 5, 6, 7, 8, 9]


def test_row_operations_reject_empty_or_out_of_bounds_selection():
    dataset = _make_dataset(rows=5)
    with pytest.raises(ValueError):
        dataset.exclude_rows([])
    with pytest.raises(ValueError):
        dataset.exclude_rows([99])
    with pytest.raises(ValueError):
        dataset.keep_rows([-1])


def test_row_operation_failure_leaves_working_data_in_previous_state():
    dataset = _make_dataset(rows=5)
    dataset.add_calculated_column("y2", "x * 2")
    before = dataset.dataframe.copy(deep=True)
    with pytest.raises(ValueError):
        dataset.exclude_rows([999])
    pd.testing.assert_frame_equal(dataset.dataframe, before)


def test_working_row_count_updates_after_each_operation():
    dataset = _make_dataset(rows=10)
    assert dataset.row_count == 10
    dataset.exclude_rows([0])
    assert dataset.row_count == 9
    dataset.keep_rows([0, 1, 2])
    assert dataset.row_count == 3


# --- Reset ------------------------------------------------------------------


def test_reset_working_data_restores_raw_rows_and_drops_calculated_columns():
    dataset = _make_dataset(rows=10)
    dataset.add_calculated_column("y2", "x * 2")
    dataset.exclude_rows([0, 1, 2])
    assert dataset.row_count == 7
    assert "y2" in dataset.columns

    dataset.reset_working_data()

    assert dataset.row_count == 10
    assert "y2" not in dataset.columns
    assert dataset.calculated_columns == {}
    pd.testing.assert_frame_equal(dataset.dataframe, dataset.raw_dataframe)


# --- Transformation history --------------------------------------------------


def test_transformation_history_is_ordered_and_never_cleared():
    dataset = _make_dataset(rows=10)
    dataset.add_calculated_column("y2", "x * 2")
    dataset.exclude_rows([0])
    dataset.keep_rows([0, 1])
    dataset.reset_working_data()

    kinds = [t.kind for t in dataset.transformations]
    assert kinds == ["calculated_column", "exclude_rows", "keep_rows", "reset"]


def test_transformation_history_records_readable_descriptions():
    dataset = _make_dataset(rows=10)
    dataset.add_calculated_column("y2", "x * 2")
    dataset.exclude_rows([3, 4, 5])

    descriptions = [t.description for t in dataset.transformations]
    assert descriptions[0] == "Created calculated column: y2 = x * 2"
    assert descriptions[1] == "Excluded rows: 3-5"


# --- PlotSeries staleness after transformations ------------------------------


def test_calculated_column_creation_never_invalidates_existing_series():
    dataset = _make_dataset(rows=10)
    figure = GnoviFigure()
    series = PlotSeries.line(dataset, "x", "y")
    figure.add_series(series)

    dataset.add_calculated_column("y2", "x * 2")
    newly_stale = figure.invalidate_series_for_dataset(dataset, row_set_changed=False)

    assert newly_stale == []
    assert series.stale is False


def test_row_range_series_becomes_stale_after_exclude():
    dataset = _make_dataset(rows=10)
    figure = GnoviFigure()
    whole = PlotSeries.line(dataset, "x", "y")
    ranged = PlotSeries.line(dataset, "x", "y", row_range=(2, 5))
    figure.add_series(whole)
    figure.add_series(ranged)

    dataset.exclude_rows([0, 1])
    newly_stale = figure.invalidate_series_for_dataset(dataset, row_set_changed=True)

    assert ranged in newly_stale
    assert ranged.stale is True
    assert whole.stale is False


def test_series_becomes_stale_when_its_column_is_removed_by_reset():
    dataset = _make_dataset(rows=10)
    dataset.add_calculated_column("y2", "x * 2")
    figure = GnoviFigure()
    series = PlotSeries.line(dataset, "x", "y2")
    figure.add_series(series)

    dataset.reset_working_data()
    newly_stale = figure.invalidate_series_for_dataset(dataset, row_set_changed=True)

    assert series in newly_stale
    assert series.stale is True


def test_already_stale_series_is_not_returned_again():
    dataset = _make_dataset(rows=10)
    figure = GnoviFigure()
    ranged = PlotSeries.line(dataset, "x", "y", row_range=(2, 5))
    figure.add_series(ranged)

    dataset.exclude_rows([0])
    first_pass = figure.invalidate_series_for_dataset(dataset, row_set_changed=True)
    dataset.exclude_rows([0])
    second_pass = figure.invalidate_series_for_dataset(dataset, row_set_changed=True)

    assert ranged in first_pass
    assert second_pass == []


def test_series_for_other_datasets_are_not_touched():
    dataset_a = _make_dataset(name="a", rows=10)
    dataset_b = _make_dataset(name="b", rows=10)
    figure = GnoviFigure()
    series_b = PlotSeries.line(dataset_b, "x", "y", row_range=(0, 3))
    figure.add_series(series_b)

    dataset_a.exclude_rows([0])
    newly_stale = figure.invalidate_series_for_dataset(dataset_a, row_set_changed=True)

    assert newly_stale == []
    assert series_b.stale is False
