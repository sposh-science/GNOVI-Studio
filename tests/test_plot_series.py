import pandas as pd
import pytest

from gnovi_plot.analysis.cycles import detect_cycles
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.series import PlotSeries, PlotType


def _make_dataset(name="cv"):
    df = pd.DataFrame({"Potential/V": [-0.2, -0.1, 0.0, 0.1], "Current/A": [1e-5, 2e-5, 3e-5, 4e-5]})
    return Dataset(name=name, dataframe=df)


def _make_three_cycle_dataset(name="cv"):
    leg = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0]
    x = leg + leg[1:] + leg[1:]
    y = [float(i) for i in range(len(x))]
    df = pd.DataFrame({"Potential/V": x, "Current/A": y})
    return Dataset(name=name, dataframe=df)


def test_line_series_configuration():
    dataset = _make_dataset("SR-0.01")
    series = PlotSeries.line(dataset, "Potential/V", "Current/A")

    assert series.plot_type == PlotType.LINE
    assert series.dataset is dataset
    assert series.x_column == "Potential/V"
    assert series.y_column == "Current/A"
    assert series.label == "SR-0.01 — Current/A"
    assert series.marker == ""
    assert series.visible is True
    assert series.id


def test_scatter_series_configuration():
    dataset = _make_dataset("SR-0.05")
    series = PlotSeries.scatter(dataset, "Potential/V", "Current/A")

    assert series.plot_type == PlotType.SCATTER
    assert series.x_column == "Potential/V"
    assert series.y_column == "Current/A"
    assert series.marker == "o"


def test_histogram_series_configuration():
    dataset = _make_dataset("SR-0.10")
    series = PlotSeries.histogram(dataset, "Current/A")

    assert series.plot_type == PlotType.HISTOGRAM
    assert series.x_column == "Current/A"
    assert series.y_column is None
    assert series.bins == "auto"
    assert series.label == "SR-0.10 — Current/A"


def test_histogram_series_accepts_manual_bin_count():
    dataset = _make_dataset()
    series = PlotSeries.histogram(dataset, "Current/A", bins=25)
    assert series.bins == 25


def test_histogram_series_rejects_invalid_bins():
    dataset = _make_dataset()
    with pytest.raises(ValueError):
        PlotSeries.histogram(dataset, "Current/A", bins=0)
    with pytest.raises(ValueError):
        PlotSeries.histogram(dataset, "Current/A", bins="not-a-count")


def test_histogram_series_rejects_a_y_column():
    dataset = _make_dataset()
    with pytest.raises(ValueError):
        PlotSeries(
            dataset=dataset,
            plot_type=PlotType.HISTOGRAM,
            label="bad",
            x_column="Current/A",
            y_column="Potential/V",
        )


def test_xy_series_require_both_columns():
    dataset = _make_dataset()
    with pytest.raises(ValueError):
        PlotSeries(dataset=dataset, plot_type=PlotType.LINE, label="bad", x_column="Potential/V")


def test_series_ids_are_unique():
    dataset = _make_dataset()
    a = PlotSeries.line(dataset, "Potential/V", "Current/A")
    b = PlotSeries.line(dataset, "Potential/V", "Current/A")
    assert a.id != b.id


def test_style_property_changes_are_independent_per_series():
    dataset = _make_dataset()
    a = PlotSeries.line(dataset, "Potential/V", "Current/A", color="#111111")
    b = PlotSeries.line(dataset, "Potential/V", "Current/A", color="#222222")

    a.color = "#ff0000"
    a.line_width = 3.0
    a.visible = False

    assert a.color == "#ff0000"
    assert a.line_width == 3.0
    assert a.visible is False

    assert b.color == "#222222"
    assert b.line_width == 1.5
    assert b.visible is True


def test_alpha_must_be_within_unit_range():
    dataset = _make_dataset()
    with pytest.raises(ValueError):
        PlotSeries.line(dataset, "Potential/V", "Current/A", alpha=1.5)


def test_series_construction_does_not_mutate_source_dataframe():
    dataset = _make_dataset()
    original = dataset.dataframe.copy(deep=True)

    series = PlotSeries.line(dataset, "Potential/V", "Current/A")
    series.color = "#abcdef"
    series.label = "renamed"
    series.visible = False

    pd.testing.assert_frame_equal(dataset.dataframe, original)


def test_row_range_dataframe_property_slices_without_mutating_source():
    dataset = _make_dataset()
    original = dataset.dataframe.copy(deep=True)

    series = PlotSeries.line(dataset, "Potential/V", "Current/A", row_range=(1, 3))

    assert list(series.dataframe["Potential/V"]) == [-0.1, 0.0]
    pd.testing.assert_frame_equal(dataset.dataframe, original)


def test_row_range_none_uses_the_whole_dataframe():
    dataset = _make_dataset()
    series = PlotSeries.line(dataset, "Potential/V", "Current/A")
    pd.testing.assert_frame_equal(series.dataframe, dataset.dataframe)


def test_row_range_out_of_bounds_is_rejected():
    dataset = _make_dataset()
    with pytest.raises(ValueError):
        PlotSeries.line(dataset, "Potential/V", "Current/A", row_range=(0, 999))
    with pytest.raises(ValueError):
        PlotSeries.line(dataset, "Potential/V", "Current/A", row_range=(3, 1))


def test_detected_cycles_become_independent_plot_series():
    dataset = _make_three_cycle_dataset()
    cycles = detect_cycles(dataset.dataframe, "Potential/V")
    assert len(cycles) == 3

    series_list = [
        PlotSeries.line(
            dataset,
            "Potential/V",
            "Current/A",
            label=f"{dataset.name} — Cycle {i + 1}",
            row_range=row_range,
        )
        for i, row_range in enumerate(cycles)
    ]

    assert [s.label for s in series_list] == [
        "cv — Cycle 1",
        "cv — Cycle 2",
        "cv — Cycle 3",
    ]
    assert len({s.id for s in series_list}) == 3
    assert all(s.dataset is dataset for s in series_list)

    # Every series shares the same source Dataset but references its own
    # row range, so styling one leaves the others untouched.
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
