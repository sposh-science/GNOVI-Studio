import pandas as pd
import pytest

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.figure import GnoviFigure, Panel
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return Dataset(name=name, dataframe=df)


# --- layout / panel grid --------------------------------------------------


@pytest.mark.parametrize("rows,cols", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_set_layout_creates_the_requested_panel_count(rows, cols):
    figure = GnoviFigure()
    figure.set_layout(rows, cols)

    assert figure.layout == (rows, cols)
    assert len(figure.panels) == rows * cols


def test_set_layout_preserves_existing_panels_when_growing():
    figure = GnoviFigure()
    figure.active_panel.title = "Panel 1 title"
    figure.set_layout(1, 2)

    assert figure.panels[0].title == "Panel 1 title"
    assert figure.panels[1].title == ""


def test_set_layout_drops_trailing_panels_when_shrinking():
    figure = GnoviFigure()
    figure.set_layout(2, 2)
    figure.panels[3].title = "last panel"
    figure.set_active_panel(3)

    figure.set_layout(1, 1)

    assert len(figure.panels) == 1
    # Active panel index is clamped back into range rather than left dangling.
    assert figure.active_panel_index == 0


def test_set_layout_rejects_zero_panels():
    figure = GnoviFigure()
    with pytest.raises(ValueError):
        figure.set_layout(0, 1)


# --- active panel selection ------------------------------------------------


def test_set_active_panel_switches_which_panel_is_delegated_to():
    figure = GnoviFigure()
    figure.set_layout(1, 2)

    figure.set_active_panel(0)
    figure.title = "left panel"
    figure.set_active_panel(1)
    figure.title = "right panel"

    assert figure.panels[0].title == "left panel"
    assert figure.panels[1].title == "right panel"


def test_set_active_panel_rejects_out_of_range_index():
    figure = GnoviFigure()
    with pytest.raises(ValueError):
        figure.set_active_panel(5)


# --- series assignment to independent panels --------------------------------


def test_add_series_adds_to_the_active_panel_only():
    figure = GnoviFigure()
    figure.set_layout(1, 2)

    figure.set_active_panel(0)
    figure.add_series(PlotSeries.line(_make_dataset("a"), "x", "y"))
    figure.set_active_panel(1)
    figure.add_series(PlotSeries.line(_make_dataset("b"), "x", "y"))

    assert len(figure.panels[0].series) == 1
    assert len(figure.panels[1].series) == 1
    assert figure.panels[0].series[0].dataset.name == "a"
    assert figure.panels[1].series[0].dataset.name == "b"


def test_each_panel_has_its_own_color_cycle():
    figure = GnoviFigure()
    figure.set_layout(1, 2)

    figure.set_active_panel(0)
    a1 = PlotSeries.line(_make_dataset("a1"), "x", "y")
    figure.add_series(a1)
    figure.set_active_panel(1)
    b1 = PlotSeries.line(_make_dataset("b1"), "x", "y")
    figure.add_series(b1)

    # Both panels' first series get the same first color from their own
    # independent cycle, not a continuation of a shared one.
    assert a1.color == b1.color


def test_panels_have_independent_limits_and_titles():
    figure = GnoviFigure()
    figure.set_layout(2, 1)

    figure.set_active_panel(0)
    figure.title = "top"
    figure.xlim = (0.0, 1.0)
    figure.set_active_panel(1)
    figure.title = "bottom"
    figure.ylim = (-1.0, 1.0)

    assert figure.panels[0].title == "top"
    assert figure.panels[0].xlim == (0.0, 1.0)
    assert figure.panels[0].ylim is None
    assert figure.panels[1].title == "bottom"
    assert figure.panels[1].ylim == (-1.0, 1.0)
    assert figure.panels[1].xlim is None


def test_panels_have_independent_legends():
    figure = GnoviFigure()
    figure.set_layout(1, 2)

    figure.set_active_panel(0)
    figure.legend_visible = False
    figure.set_active_panel(1)
    figure.legend_visible = True
    figure.legend_loc = "upper left"

    assert figure.panels[0].legend_visible is False
    assert figure.panels[1].legend_visible is True
    assert figure.panels[1].legend_loc == "upper left"


# --- panel labels ------------------------------------------------------------


def test_panel_labels_are_auto_assigned_in_order():
    figure = GnoviFigure()
    figure.set_layout(2, 2)

    assert [p.panel_label for p in figure.panels] == ["(a)", "(b)", "(c)", "(d)"]


def test_panel_labels_default_hidden_and_configurable():
    figure = GnoviFigure()
    assert figure.panel_labels_visible is False
    figure.panel_labels_visible = True
    assert figure.panel_labels_visible is True


# --- backward compatibility: default figure behaves like the old single-axes one


def test_default_figure_has_exactly_one_panel():
    figure = GnoviFigure()
    assert figure.layout == (1, 1)
    assert len(figure.panels) == 1
    assert figure.active_panel_index == 0


def test_invalidate_series_for_dataset_checks_every_panel():
    dataset = _make_dataset()
    figure = GnoviFigure()
    figure.set_layout(1, 2)

    figure.set_active_panel(0)
    series_a = PlotSeries.line(dataset, "x", "y", row_range=(0, 2))
    figure.add_series(series_a)
    figure.set_active_panel(1)
    series_b = PlotSeries.line(dataset, "x", "y", row_range=(0, 2))
    figure.add_series(series_b)

    newly_stale = figure.invalidate_series_for_dataset(dataset, row_set_changed=True)

    assert {s.id for s in newly_stale} == {series_a.id, series_b.id}
    assert series_a.stale is True
    assert series_b.stale is True


def test_get_series_and_remove_series_search_across_panels():
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    figure.set_active_panel(0)
    series_a = PlotSeries.line(_make_dataset("a"), "x", "y")
    figure.add_series(series_a)
    figure.set_active_panel(1)

    assert figure.get_series(series_a.id) is series_a

    figure.remove_series(series_a.id)
    assert figure.panels[0].series == []


# --- Panel.id: stable identity ----------------------------------------------


def test_a_fresh_panel_has_a_non_empty_id():
    figure = GnoviFigure()
    assert figure.active_panel.id
    assert isinstance(figure.active_panel.id, str)


def test_two_fresh_panels_get_different_ids():
    assert Panel().id != Panel().id


def test_every_panel_in_a_multi_panel_layout_has_a_unique_id():
    figure = GnoviFigure()
    figure.set_layout(2, 2)

    ids = [p.id for p in figure.panels]
    assert len(set(ids)) == len(ids) == 4


def test_set_layout_growth_preserves_the_existing_panels_own_id():
    figure = GnoviFigure()
    original_id = figure.active_panel.id

    figure.set_layout(1, 2)

    assert figure.panels[0].id == original_id


def test_set_layout_shrink_then_regrow_never_reuses_a_dropped_panel_id():
    figure = GnoviFigure()
    figure.set_layout(2, 2)
    dropped_ids = {p.id for p in figure.panels[1:]}

    figure.set_layout(1, 1)
    figure.set_layout(2, 2)

    regrown_ids = {p.id for p in figure.panels[1:]}
    assert regrown_ids.isdisjoint(dropped_ids)


def test_switching_active_panel_never_changes_any_panel_id():
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    ids_before = [p.id for p in figure.panels]

    figure.set_active_panel(1)
    figure.set_active_panel(0)

    assert [p.id for p in figure.panels] == ids_before


# --- Panel.id: to_dict/from_dict round trip ---------------------------------


def test_panel_id_round_trips_through_to_dict_from_dict():
    panel = Panel()
    restored = Panel.from_dict(panel.to_dict(), {})

    assert restored.id == panel.id


def test_panel_from_dict_generates_a_fresh_id_when_the_key_is_missing():
    """A project saved before `Panel.id` existed has no `"id"` key in its
    panel dicts at all -- loading it must not crash, and must not leave
    the reconstructed panel with a falsy/missing id."""
    data = Panel().to_dict()
    del data["id"]

    restored = Panel.from_dict(data, {})

    assert restored.id
    assert isinstance(restored.id, str)


def test_panels_missing_ids_each_get_a_different_generated_id():
    data_a = Panel().to_dict()
    data_b = Panel().to_dict()
    del data_a["id"]
    del data_b["id"]

    restored_a = Panel.from_dict(data_a, {})
    restored_b = Panel.from_dict(data_b, {})

    assert restored_a.id != restored_b.id
