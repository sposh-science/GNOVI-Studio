import pandas as pd

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.graph import Graph
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


# --- Graph: creation and serialization ----------------------------------------


def test_graph_created_from_a_panel_holds_the_given_panel():
    dataset = _make_dataset()
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))

    graph = Graph(name="My Graph", panel=figure.active_panel)
    assert graph.name == "My Graph"
    assert graph.panel is figure.active_panel
    assert graph.id


def test_graph_to_dict_from_dict_round_trip_preserves_identity_and_content():
    dataset = _make_dataset()
    figure = GnoviFigure()
    figure.active_panel.title = "Overlay"
    figure.add_series(PlotSeries.line(dataset, "x", "y", color="#ff0000"))

    graph = Graph(name="Saved Graph", panel=figure.active_panel)
    data = graph.to_dict()
    restored = Graph.from_dict(data, {dataset.id: dataset})

    assert restored.id == graph.id
    assert restored.name == "Saved Graph"
    assert restored.panel.title == "Overlay"
    assert restored.panel.series[0].dataset is dataset
    assert restored.panel.series[0].color == "#ff0000"


def test_graph_from_dict_resolves_dataset_references_via_the_lookup():
    dataset = _make_dataset()
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))
    graph = Graph(name="G", panel=figure.active_panel)

    data = graph.to_dict()
    restored = Graph.from_dict(data, {dataset.id: dataset})

    assert restored.panel.series[0].dataset is dataset  # same live object, not a copy


def test_graph_remains_editable_after_round_trip():
    dataset = _make_dataset()
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))
    graph = Graph(name="G", panel=figure.active_panel)

    restored = Graph.from_dict(graph.to_dict(), {dataset.id: dataset})
    restored.panel.title = "Edited After Load"
    restored.panel.series[0].color = "#00ff00"
    restored.panel.add_series(PlotSeries.line(dataset, "x", "y"))

    assert restored.panel.title == "Edited After Load"
    assert restored.panel.series[0].color == "#00ff00"
    assert len(restored.panel.series) == 2


# --- GraphLibrary: Save/Load/Rename/Duplicate/Delete workflow ----------------


def test_save_current_panel_as_graph_adds_an_independent_snapshot():
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    figure = GnoviFigure()
    figure.active_panel.title = "Original"
    figure.add_series(PlotSeries.line(dataset, "x", "y"))

    library = GraphLibrary()
    graph = library.save_panel_as_graph(figure, "Graph 1", manager)

    assert len(library.graphs) == 1
    assert library.get(graph.id) is graph
    assert graph.panel is not figure.active_panel  # independent snapshot
    assert graph.panel.series[0].dataset is dataset  # but shares the live Dataset


def test_editing_the_live_panel_after_save_does_not_affect_the_stored_graph():
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y", color="#111111"))

    library = GraphLibrary()
    graph = library.save_panel_as_graph(figure, "Graph 1", manager)

    figure.active_panel.title = "Changed Later"
    figure.active_panel.series[0].color = "#999999"
    figure.add_series(PlotSeries.line(dataset, "x", "y"))

    assert graph.panel.title == ""
    assert graph.panel.series[0].color == "#111111"
    assert len(graph.panel.series) == 1


def test_rename_graph():
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    figure = GnoviFigure()
    library = GraphLibrary()
    graph = library.save_panel_as_graph(figure, "Original Name", manager)

    library.rename(graph.id, "New Name")

    assert library.get(graph.id).name == "New Name"


def test_rename_unknown_graph_is_a_no_op():
    library = GraphLibrary()
    library.rename("does-not-exist", "New Name")  # must not raise


def test_duplicate_graph_produces_an_independent_copy_with_a_new_id():
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    figure = GnoviFigure()
    figure.active_panel.title = "Base"
    figure.add_series(PlotSeries.line(dataset, "x", "y"))
    library = GraphLibrary()
    graph = library.save_panel_as_graph(figure, "Base Graph", manager)

    copy_graph = library.duplicate(graph.id, manager)

    assert copy_graph is not None
    assert copy_graph.id != graph.id
    assert copy_graph.name == "Base Graph (Copy)"
    assert copy_graph.panel is not graph.panel
    assert copy_graph.panel.title == "Base"
    assert len(library.graphs) == 2

    copy_graph.panel.title = "Edited Copy"
    assert graph.panel.title == "Base"  # original untouched


def test_duplicate_unknown_graph_returns_none():
    library = GraphLibrary()
    manager = DatasetManager()
    assert library.duplicate("does-not-exist", manager) is None


def test_delete_graph_removes_it_from_the_library():
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    figure = GnoviFigure()
    library = GraphLibrary()
    graph = library.save_panel_as_graph(figure, "G", manager)

    library.remove(graph.id)

    assert library.get(graph.id) is None
    assert len(library.graphs) == 0


def test_delete_unknown_graph_is_a_no_op():
    library = GraphLibrary()
    library.remove("does-not-exist")  # must not raise


def test_load_graph_into_active_panel_replaces_it_with_an_independent_copy():
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    source_figure = GnoviFigure()
    source_figure.active_panel.title = "Saved"
    source_figure.add_series(PlotSeries.line(dataset, "x", "y", color="#123456"))
    library = GraphLibrary()
    graph = library.save_panel_as_graph(source_figure, "G", manager)

    target_figure = GnoviFigure()
    target_figure.active_panel.title = "Blank"
    loaded = library.load_graph_into_panel(graph.id, target_figure, manager)

    assert loaded is True
    assert target_figure.active_panel is not graph.panel  # independent copy
    assert target_figure.active_panel.title == "Saved"
    assert target_figure.active_panel.series[0].dataset is dataset
    assert target_figure.active_panel.series[0].color == "#123456"


def test_load_graph_into_active_panel_returns_false_for_unknown_graph_id():
    manager = DatasetManager()
    figure = GnoviFigure()
    original_panel = figure.active_panel

    loaded = GraphLibrary().load_graph_into_panel("nope", figure, manager)

    assert loaded is False
    assert figure.active_panel is original_panel  # untouched


def test_editing_the_loaded_panel_does_not_mutate_the_stored_graph():
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    source_figure = GnoviFigure()
    source_figure.add_series(PlotSeries.line(dataset, "x", "y", color="#111111"))
    library = GraphLibrary()
    graph = library.save_panel_as_graph(source_figure, "G", manager)

    target_figure = GnoviFigure()
    library.load_graph_into_panel(graph.id, target_figure, manager)

    # Edit the panel that was just loaded into the active figure.
    target_figure.active_panel.title = "Edited After Load"
    target_figure.active_panel.series[0].color = "#ff00ff"
    target_figure.add_series(PlotSeries.line(dataset, "x", "y"))

    assert graph.panel.title == ""
    assert graph.panel.series[0].color == "#111111"
    assert len(graph.panel.series) == 1


def test_stored_graph_survives_the_active_panel_being_replaced_again():
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    source_figure = GnoviFigure()
    source_figure.active_panel.title = "Original"
    source_figure.add_series(PlotSeries.line(dataset, "x", "y"))
    library = GraphLibrary()
    graph = library.save_panel_as_graph(source_figure, "G", manager)

    target_figure = GnoviFigure()
    library.load_graph_into_panel(graph.id, target_figure, manager)
    # Replace the active panel a second time (e.g. loading a different graph,
    # or Load Selected Graph clicked again) -- the first stored Graph must
    # remain exactly as it was.
    target_figure.panels[target_figure.active_panel_index] = type(target_figure.active_panel)()

    assert graph.panel.title == "Original"
    assert len(graph.panel.series) == 1


# --- GraphLibrary: independence from panel layout -----------------------------
#
# A Dataset -> Saved Graph -> Panel Display is meant to be a one-way chain:
# a Graph is a reusable, project-bound configuration, and a Panel is only a
# viewport that currently happens to display one. Changing the panel grid
# (`GnoviFigure.set_layout`) legitimately drops/creates *panels*, but must
# never delete anything from the GraphLibrary itself.


def test_graph_library_survives_layout_reduction_and_reexpansion():
    """1x2 -> save Graph A (panel 0) and Graph B (panel 1) -> 1x1 (panel 1's
    content is gone from the figure, as expected) -> 1x2 again: both Graphs
    must still be in the library, retrievable to load into any panel --
    the figure is NOT expected to auto-restore panel 1's old content just
    because the layout returned to 1x2 (that's what the Graph Library is
    for)."""
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    figure = GnoviFigure()
    figure.set_layout(1, 2)

    figure.set_active_panel(0)
    figure.active_panel.title = "Graph A"
    figure.add_series(PlotSeries.line(dataset, "x", "y", label="a"))
    figure.set_active_panel(1)
    figure.active_panel.title = "Graph B"
    figure.add_series(PlotSeries.line(dataset, "x", "y", label="b"))

    library = GraphLibrary()
    figure.set_active_panel(0)
    graph_a = library.save_panel_as_graph(figure, "Graph A", manager)
    figure.set_active_panel(1)
    graph_b = library.save_panel_as_graph(figure, "Graph B", manager)

    figure.set_layout(1, 1)  # panel 1 (Graph B's display) is dropped
    assert len(figure.panels) == 1
    assert len(library.graphs) == 2  # library untouched by the layout shrink

    figure.set_layout(1, 2)  # back to 1x2 -- fresh, blank panels
    assert len(figure.panels) == 2
    assert figure.panels[1].series == []  # not auto-restored -- that's expected
    assert {g.id for g in library.graphs} == {graph_a.id, graph_b.id}
    assert library.get(graph_a.id).panel.title == "Graph A"
    assert library.get(graph_b.id).panel.title == "Graph B"


def test_saved_graphs_can_be_reloaded_into_panels_after_a_layout_round_trip():
    """Continuing the scenario above: after 1x2 -> 1x1 -> 1x2, both
    previously-saved Graphs must be loadable into any of the new (blank)
    panels, in any arrangement -- not just their original panel index."""
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    figure.set_active_panel(0)
    figure.active_panel.title = "Graph A"
    figure.set_active_panel(1)
    figure.active_panel.title = "Graph B"

    library = GraphLibrary()
    figure.set_active_panel(0)
    graph_a = library.save_panel_as_graph(figure, "Graph A", manager)
    figure.set_active_panel(1)
    graph_b = library.save_panel_as_graph(figure, "Graph B", manager)

    figure.set_layout(1, 1)
    figure.set_layout(1, 2)

    # Load them back in swapped order to prove any Graph can go in any panel.
    figure.set_active_panel(0)
    library.load_graph_into_panel(graph_b.id, figure, manager)
    figure.set_active_panel(1)
    library.load_graph_into_panel(graph_a.id, figure, manager)

    assert figure.panels[0].title == "Graph B"
    assert figure.panels[1].title == "Graph A"
    # Still independent copies -- the library entries are untouched.
    assert library.get(graph_a.id).panel.title == "Graph A"
    assert library.get(graph_b.id).panel.title == "Graph B"


def test_multiple_saved_graphs_load_independently_into_different_panels_of_a_2x2():
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    scratch = GnoviFigure()
    library = GraphLibrary()
    graphs = []
    for i in range(4):
        scratch.active_panel.clear_series()
        scratch.active_panel.title = f"G{i}"
        scratch.add_series(PlotSeries.line(dataset, "x", "y", label=f"s{i}", color=f"#{i}{i}{i}{i}{i}{i}"))
        graphs.append(library.save_panel_as_graph(scratch, f"G{i}", manager))

    figure = GnoviFigure()
    figure.set_layout(2, 2)
    for panel_index, graph in enumerate(graphs):
        figure.set_active_panel(panel_index)
        assert library.load_graph_into_panel(graph.id, figure, manager)

    assert [panel.title for panel in figure.panels] == ["G0", "G1", "G2", "G3"]
    # Each panel is its own independent copy -- editing one never touches
    # another panel or any stored Graph.
    figure.panels[0].title = "Edited"
    assert figure.panels[1].title == "G1"
    assert library.get(graphs[0].id).panel.title == "G0"


# --- GraphLibrary: serialization ---------------------------------------------


def test_graph_library_to_dict_from_dict_round_trip():
    dataset = _make_dataset()
    manager = _manager_with(dataset)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))
    library = GraphLibrary()
    g1 = library.save_panel_as_graph(figure, "Graph 1", manager)
    g2 = library.save_panel_as_graph(figure, "Graph 2", manager)

    data = library.to_dict()
    restored = GraphLibrary.from_dict(data, {dataset.id: dataset})

    assert {g.id for g in restored.graphs} == {g1.id, g2.id}
    assert {g.name for g in restored.graphs} == {"Graph 1", "Graph 2"}
    for g in restored.graphs:
        assert g.panel.series[0].dataset is dataset


def test_empty_graph_library_round_trips_to_an_empty_list():
    library = GraphLibrary()
    assert library.to_dict() == []
    restored = GraphLibrary.from_dict([], {})
    assert len(restored.graphs) == 0
