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


# --- Issue #44: Auto-Stack must use the normalized display range, not the raw range ---


def test_suggest_offset_step_uses_normalized_range_when_normalize_to_max_is_set():
    panel = Panel()
    a = PlotSeries.line(
        _make_dataset("A", [0, 3000, 0]), "two_theta", "intensity", normalize_to_max=True
    )
    panel.add_series(a)

    step = suggest_offset_step(panel)

    # Normalized display range is [0, 1] -> range 1.0, not the raw 3000.0.
    assert step == pytest.approx(1.0)


def test_auto_stack_offsets_normalized_series_at_normalized_scale_not_raw_scale():
    panel = Panel()
    series = [
        PlotSeries.line(
            _make_dataset(name, y), "two_theta", "intensity", normalize_to_max=True
        )
        for name, y in [
            ("A", [0, 1000, 0]),
            ("B", [0, 1800, 0]),
            ("C", [0, 900, 0]),
            ("D", [0, 4700, 0]),
        ]
    ]
    for s in series:
        panel.add_series(s)

    used_step = auto_stack_offsets(panel)

    # Every series normalizes to a peak of 1.0, so the displayed range is
    # 1.0 for all of them regardless of raw amplitude -- the step (and
    # every resulting offset) must stay on that normalized scale, not
    # balloon into the thousands the raw data would suggest.
    assert used_step == pytest.approx(1.0)
    for i, s in enumerate(series):
        assert s.y_offset == pytest.approx(i * 1.0)


def test_suggest_offset_step_mixed_normalized_and_raw_uses_largest_displayed_range():
    panel = Panel()
    normalized = PlotSeries.line(
        _make_dataset("A", [0, 3000, 0]), "two_theta", "intensity", normalize_to_max=True
    )
    raw = PlotSeries.line(_make_dataset("B", [0, 50, 0]), "two_theta", "intensity")
    panel.add_series(normalized)
    panel.add_series(raw)

    step = suggest_offset_step(panel)

    # normalized's displayed range is 1.0; raw's displayed (== actual)
    # range is 50.0 -- the larger of the two effective displayed ranges,
    # not the larger of the raw values (3000 vs 50).
    assert step == pytest.approx(50.0)


def test_toggling_normalize_after_auto_stack_does_not_recompute_existing_offset():
    panel = Panel()
    a = PlotSeries.line(_make_dataset("A", [0, 1000, 0]), "two_theta", "intensity")
    b = PlotSeries.line(_make_dataset("B", [0, 1000, 0]), "two_theta", "intensity")
    panel.add_series(a)
    panel.add_series(b)
    auto_stack_offsets(panel)
    assert b.y_offset == pytest.approx(1000.0)

    # Auto-Stack is an explicit, one-shot action: enabling normalize on an
    # already-stacked series afterward must not silently recompute or
    # touch the offset it was already given.
    b.normalize_to_max = True

    assert b.y_offset == pytest.approx(1000.0)
