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


# --- Project-save serialization -----------------------------------------------


def test_to_dict_stores_dataset_id_not_a_nested_dataset():
    dataset = _make_dataset()
    series = PlotSeries.line(dataset, "Potential/V", "Current/A")
    data = series.to_dict()

    assert data["dataset_id"] == dataset.id
    assert "dataset" not in data
    assert "dataframe" not in data


def test_from_dict_resolves_dataset_via_lookup():
    dataset = _make_dataset()
    series = PlotSeries.line(dataset, "Potential/V", "Current/A")
    data = series.to_dict()

    restored = PlotSeries.from_dict(data, {dataset.id: dataset})

    assert restored is not None
    assert restored.dataset is dataset  # same live instance, not a copy
    assert restored.id == series.id
    assert restored.x_column == "Potential/V"
    assert restored.y_column == "Current/A"


def test_from_dict_returns_none_when_dataset_id_is_missing_from_lookup():
    dataset = _make_dataset()
    series = PlotSeries.line(dataset, "Potential/V", "Current/A")
    data = series.to_dict()

    assert PlotSeries.from_dict(data, {}) is None


def test_round_trip_preserves_row_range_and_visibility():
    dataset = _make_dataset()
    series = PlotSeries.line(dataset, "Potential/V", "Current/A", row_range=(1, 3), visible=False)
    restored = PlotSeries.from_dict(series.to_dict(), {dataset.id: dataset})

    assert restored.row_range == (1, 3)
    assert isinstance(restored.row_range, tuple)
    assert restored.visible is False


def test_round_trip_preserves_row_range_none():
    dataset = _make_dataset()
    series = PlotSeries.line(dataset, "Potential/V", "Current/A")
    restored = PlotSeries.from_dict(series.to_dict(), {dataset.id: dataset})
    assert restored.row_range is None


def test_round_trip_preserves_color_and_style_fields():
    dataset = _make_dataset()
    series = PlotSeries.line(
        dataset,
        "Potential/V",
        "Current/A",
        color="#abcdef",
        color_is_manual=True,
        line_width=3.5,
        line_style="--",
        marker="s",
        marker_size=9.0,
        marker_filled=False,
        marker_edge_width=2.0,
        alpha=0.75,
        zorder=5.0,
    )
    restored = PlotSeries.from_dict(series.to_dict(), {dataset.id: dataset})

    assert restored.color == "#abcdef"
    assert restored.color_is_manual is True
    assert restored.line_width == 3.5
    assert restored.line_style == "--"
    assert restored.marker == "s"
    assert restored.marker_size == 9.0
    assert restored.marker_filled is False
    assert restored.marker_edge_width == 2.0
    assert restored.alpha == 0.75
    assert restored.zorder == 5.0


def test_round_trip_preserves_stacking_offset_and_normalization():
    dataset = _make_dataset()
    series = PlotSeries.line(dataset, "Potential/V", "Current/A", y_offset=2.5, normalize_to_max=True)
    restored = PlotSeries.from_dict(series.to_dict(), {dataset.id: dataset})

    assert restored.y_offset == 2.5
    assert restored.normalize_to_max is True


def test_round_trip_preserves_histogram_mode_and_bins():
    dataset = _make_dataset()
    series = PlotSeries.histogram(dataset, "Current/A", bins=17, hist_mode="cumulative")
    restored = PlotSeries.from_dict(series.to_dict(), {dataset.id: dataset})

    assert restored.plot_type == PlotType.HISTOGRAM
    assert restored.bins == 17
    assert restored.hist_mode == "cumulative"
    assert restored.y_column is None


def test_round_trip_preserves_stale_flag():
    dataset = _make_dataset()
    series = PlotSeries.line(dataset, "Potential/V", "Current/A")
    series.stale = True
    restored = PlotSeries.from_dict(series.to_dict(), {dataset.id: dataset})
    assert restored.stale is True
