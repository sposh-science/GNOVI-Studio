import pandas as pd
import pytest

from gnovi_plot.analysis.segments import (
    InvalidRowRangeError,
    OverlappingRowRangeError,
    RowRangeCollection,
    contiguous_row_range,
)
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(rows: int = 4764, name: str = "cv") -> Dataset:
    df = pd.DataFrame(
        {
            "Potential/V": [float(i) for i in range(rows)],
            "Current/A": [float(i) * 0.5 for i in range(rows)],
        }
    )
    return Dataset(name=name, dataframe=df)


def _manual_series(dataset: Dataset, x_col: str, y_col: str, ranges: list[tuple[int, int]]) -> list[PlotSeries]:
    return [
        PlotSeries.line(
            dataset,
            x_col,
            y_col,
            label=f"{dataset.name} — Cycle {i + 1}",
            row_range=row_range,
        )
        for i, row_range in enumerate(ranges)
    ]


def test_creates_one_manual_cycle_from_table_selection():
    ranges = RowRangeCollection(row_count=10)
    start, end = ranges.add_from_positions([2, 3, 4, 5])
    assert (start, end) == (2, 6)
    assert ranges.ranges == [(2, 6)]


def test_creates_five_or_more_manual_cycles():
    ranges = RowRangeCollection(row_count=100)
    for i in range(6):
        ranges.add_from_positions(range(i * 10, i * 10 + 5))
    assert len(ranges) == 6
    assert ranges.ranges == [(i * 10, i * 10 + 5) for i in range(6)]


def test_overlapping_selection_is_rejected():
    ranges = RowRangeCollection(row_count=20)
    ranges.add_from_positions([0, 1, 2, 3])
    with pytest.raises(OverlappingRowRangeError):
        ranges.add_from_positions([2, 3, 4, 5])


def test_non_contiguous_selection_is_rejected():
    with pytest.raises(InvalidRowRangeError):
        contiguous_row_range([0, 1, 3, 4])

    ranges = RowRangeCollection(row_count=20)
    with pytest.raises(InvalidRowRangeError):
        ranges.add_from_positions([0, 2])
    assert len(ranges) == 0


def test_fewer_than_two_rows_is_rejected():
    with pytest.raises(InvalidRowRangeError):
        contiguous_row_range([5])
    with pytest.raises(InvalidRowRangeError):
        contiguous_row_range([])

    ranges = RowRangeCollection(row_count=20)
    with pytest.raises(InvalidRowRangeError):
        ranges.add_from_positions([5])
    assert len(ranges) == 0


def test_ranges_are_sorted_by_starting_row_position_regardless_of_add_order():
    ranges = RowRangeCollection(row_count=30)
    ranges.add_from_positions([20, 21, 22])
    ranges.add_from_positions([0, 1, 2])
    ranges.add_from_positions([10, 11, 12])
    assert ranges.ranges == [(0, 3), (10, 13), (20, 23)]


def test_range_out_of_bounds_is_rejected():
    ranges = RowRangeCollection(row_count=10)
    with pytest.raises(InvalidRowRangeError):
        ranges.add(8, 15)


def test_remove_and_clear():
    ranges = RowRangeCollection(row_count=30)
    ranges.add_from_positions([0, 1, 2])
    ranges.add_from_positions([10, 11, 12])
    ranges.remove_at(0)
    assert ranges.ranges == [(10, 13)]
    ranges.clear()
    assert ranges.ranges == []


def test_source_dataframe_is_not_modified_by_manual_selection():
    dataset = _make_dataset(rows=4764)
    original = dataset.dataframe.copy(deep=True)

    ranges = RowRangeCollection(row_count=dataset.row_count)
    ranges.add_from_positions(range(0, 1588))
    ranges.add_from_positions(range(1588, 3176))
    ranges.add_from_positions(range(3176, 4764))
    _manual_series(dataset, "Potential/V", "Current/A", ranges.ranges)

    pd.testing.assert_frame_equal(dataset.dataframe, original)


def test_manual_cycles_become_independently_stylable_plot_series():
    dataset = _make_dataset(rows=4764)
    ranges = RowRangeCollection(row_count=dataset.row_count)
    ranges.add_from_positions(range(0, 1588))
    ranges.add_from_positions(range(1588, 3176))
    ranges.add_from_positions(range(3176, 4764))

    series_list = _manual_series(dataset, "Potential/V", "Current/A", ranges.ranges)

    assert [s.label for s in series_list] == [
        "cv — Cycle 1",
        "cv — Cycle 2",
        "cv — Cycle 3",
    ]
    assert [s.row_range for s in series_list] == [(0, 1588), (1588, 3176), (3176, 4764)]
    assert len({s.id for s in series_list}) == 3

    series_list[0].color = "#111111"
    series_list[0].line_width = 4.0
    series_list[0].visible = False
    series_list[1].color = "#222222"

    assert series_list[0].color == "#111111"
    assert series_list[0].line_width == 4.0
    assert series_list[0].visible is False
    assert series_list[1].color == "#222222"
    assert series_list[1].line_width == 1.5
    assert series_list[1].visible is True
    assert series_list[2].color is None
