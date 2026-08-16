"""Unit tests for `ActivePanelLabel`'s "Graph: ..." / "Data: ..." context
lines (see `gui.widgets.active_panel_label`) -- distinguishing a WORKING
panel from the persistent Graph Library snapshot it may have originated
from, and surfacing which project dataset(s) it's plotting. Exercises the
widget directly against a plain `GnoviFigure`/`GraphLibrary`, without a
`MainWindow` -- `tests/test_main_window_project.py` covers the same
behavior wired through the real GUI.
"""

import pandas as pd

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.active_panel_label import ActivePanelLabel
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.graph_library import GraphLibrary
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return Dataset(name=name, dataframe=df)


def _manager_with(*datasets):
    manager = DatasetManager()
    for dataset in datasets:
        manager.add(dataset)
    return manager


# --- Graph line: unsaved vs. working copy ---------------------------------------


def test_never_saved_panel_shows_unsaved_graph(qapp):
    figure = GnoviFigure()
    library = GraphLibrary()

    label = ActivePanelLabel(figure, lambda: library)

    assert label.text() == "Active panel: Panel 1\nGraph: Unsaved graph\nData: No data"


def test_panel_saved_as_a_graph_shows_it_as_a_working_copy(qapp):
    dataset = _make_dataset("Ferricyanide SR-0.05")
    manager = _manager_with(dataset)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))
    library = GraphLibrary()
    library.save_panel_as_graph(figure, "Ferricyanide 50 mV/s", manager)

    label = ActivePanelLabel(figure, lambda: library)

    assert label.text() == (
        "Active panel: Panel 1\n"
        "Graph: Ferricyanide 50 mV/s (working copy)\n"
        "Data: Ferricyanide SR-0.05"
    )


def test_panel_loaded_from_a_graph_shows_it_as_a_working_copy(qapp):
    dataset = _make_dataset("Ferricyanide SR-0.05")
    manager = _manager_with(dataset)
    source_figure = GnoviFigure()
    source_figure.add_series(PlotSeries.line(dataset, "x", "y"))
    library = GraphLibrary()
    graph = library.save_panel_as_graph(source_figure, "Graph 6", manager)

    target_figure = GnoviFigure()
    library.load_graph_into_panel(graph.id, target_figure, manager)
    label = ActivePanelLabel(target_figure, lambda: library)

    assert label.text() == "Active panel: Panel 1\nGraph: Graph 6 (working copy)\nData: Ferricyanide SR-0.05"


def test_editing_the_working_copy_keeps_identifying_its_origin_graph(qapp):
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))
    library = GraphLibrary()
    library.save_panel_as_graph(figure, "G1", manager)
    label = ActivePanelLabel(figure, lambda: library)

    figure.active_panel.title = "Edited"  # the stored Graph 6 is untouched
    label.refresh(figure)

    assert "Graph: G1 (working copy)" in label.text()


def test_deleted_origin_graph_falls_back_to_unsaved_graph(qapp):
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))
    library = GraphLibrary()
    graph = library.save_panel_as_graph(figure, "G1", manager)
    label = ActivePanelLabel(figure, lambda: library)
    assert "G1 (working copy)" in label.text()

    library.remove(graph.id)
    label.refresh(figure)

    assert "Graph: Unsaved graph" in label.text()


def test_get_graph_library_none_omits_graph_and_data_lines(qapp):
    figure = GnoviFigure()

    label = ActivePanelLabel(figure, None)

    assert label.text() == "Active panel: Panel 1"


# --- Data line: dataset provenance ----------------------------------------------


def test_no_series_shows_no_data(qapp):
    figure = GnoviFigure()
    label = ActivePanelLabel(figure, lambda: GraphLibrary())

    assert "Data: No data" in label.text()
    assert label.toolTip() == ""


def test_one_dataset_shows_its_name(qapp):
    dataset = _make_dataset("Ferricyanide SR-0.05")
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))

    label = ActivePanelLabel(figure, lambda: GraphLibrary())

    assert "Data: Ferricyanide SR-0.05" in label.text()
    assert label.toolTip() == ""


def test_repeated_series_from_the_same_dataset_still_counts_as_one_dataset(qapp):
    dataset = _make_dataset("d")
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y", label="s1"))
    figure.add_series(PlotSeries.line(dataset, "x", "y", label="s2"))

    label = ActivePanelLabel(figure, lambda: GraphLibrary())

    assert "Data: d" in label.text()
    assert "Data: 2 datasets" not in label.text()


def test_multiple_datasets_shows_a_count_with_a_names_tooltip(qapp):
    d1 = _make_dataset("Ferricyanide SR-0.05")
    d2 = _make_dataset("Ferricyanide SR-0.10")
    d3 = _make_dataset("Ascorbic Acid")
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(d1, "x", "y"))
    figure.add_series(PlotSeries.line(d2, "x", "y"))
    figure.add_series(PlotSeries.line(d3, "x", "y"))

    label = ActivePanelLabel(figure, lambda: GraphLibrary())

    assert "Data: 3 datasets" in label.text()
    assert label.toolTip() == "Ferricyanide SR-0.05\nFerricyanide SR-0.10\nAscorbic Acid"


# --- Follows the active panel, not a fixed one ----------------------------------


def test_context_follows_the_active_panel_after_switching(qapp):
    d1 = _make_dataset("A")
    d2 = _make_dataset("B")
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    figure.set_active_panel(0)
    figure.add_series(PlotSeries.line(d1, "x", "y"))
    figure.set_active_panel(1)
    figure.add_series(PlotSeries.line(d2, "x", "y"))
    library = GraphLibrary()
    manager = _manager_with(d1, d2)
    library.save_panel_as_graph(figure, "Panel 2 Graph", manager)
    label = ActivePanelLabel(figure, lambda: library)

    figure.set_active_panel(0)
    label.refresh(figure)
    assert label.text() == "Active panel: Panel 1\nGraph: Unsaved graph\nData: A"

    figure.set_active_panel(1)
    label.refresh(figure)
    assert label.text() == "Active panel: Panel 2\nGraph: Panel 2 Graph (working copy)\nData: B"
