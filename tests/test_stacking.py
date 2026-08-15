import pandas as pd
import pytest

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.figure import Panel
from gnovi_plot.plotting.series import PlotSeries
from gnovi_plot.plotting.stacking import auto_stack_offsets, reset_offsets, suggest_offset_step


def _make_dataset(name, y_values):
    df = pd.DataFrame({"two_theta": list(range(len(y_values))), "intensity": y_values})
    return Dataset(name=name, dataframe=df)


def test_auto_stack_offsets_assigns_sequential_multiples_of_step():
    panel = Panel()
    a = PlotSeries.line(_make_dataset("A", [0, 10, 0]), "two_theta", "intensity")
    b = PlotSeries.line(_make_dataset("B", [0, 20, 0]), "two_theta", "intensity")
    c = PlotSeries.line(_make_dataset("C", [0, 5, 0]), "two_theta", "intensity")
    panel.add_series(a)
    panel.add_series(b)
    panel.add_series(c)

    used_step = auto_stack_offsets(panel, step=1000.0)

    assert used_step == 1000.0
    assert a.y_offset == 0.0
    assert b.y_offset == 1000.0
    assert c.y_offset == 2000.0


def test_auto_stack_offsets_computes_a_step_when_none_given():
    panel = Panel()
    a = PlotSeries.line(_make_dataset("A", [0, 10, 0]), "two_theta", "intensity")
    b = PlotSeries.line(_make_dataset("B", [0, 30, 0]), "two_theta", "intensity")
    panel.add_series(a)
    panel.add_series(b)

    used_step = auto_stack_offsets(panel)

    assert used_step == pytest.approx(30.0)
    assert a.y_offset == 0.0
    assert b.y_offset == pytest.approx(30.0)


def test_suggest_offset_step_falls_back_to_one_with_no_data():
    panel = Panel()
    assert suggest_offset_step(panel) == 1.0


def test_reset_offsets_zeroes_every_stackable_series():
    panel = Panel()
    a = PlotSeries.line(_make_dataset("A", [0, 10, 0]), "two_theta", "intensity")
    b = PlotSeries.line(_make_dataset("B", [0, 20, 0]), "two_theta", "intensity")
    panel.add_series(a)
    panel.add_series(b)
    auto_stack_offsets(panel, step=500.0)

    reset_offsets(panel)

    assert a.y_offset == 0.0
    assert b.y_offset == 0.0


def test_histograms_are_excluded_from_stacking():
    panel = Panel()
    line = PlotSeries.line(_make_dataset("A", [0, 10, 0]), "two_theta", "intensity")
    hist = PlotSeries.histogram(_make_dataset("B", [1, 2, 3]), "intensity")
    panel.add_series(line)
    panel.add_series(hist)

    auto_stack_offsets(panel, step=100.0)

    assert line.y_offset == 0.0
    assert hist.y_offset == 0.0


def test_stacking_does_not_mutate_source_dataframe():
    dataset = _make_dataset("A", [0, 10, 0])
    original = dataset.dataframe.copy(deep=True)
    panel = Panel()
    panel.add_series(PlotSeries.line(dataset, "two_theta", "intensity"))

    auto_stack_offsets(panel, step=250.0)

    pd.testing.assert_frame_equal(dataset.dataframe, original)
