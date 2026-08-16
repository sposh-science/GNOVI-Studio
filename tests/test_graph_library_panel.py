import pandas as pd
import pytest
from PySide6.QtWidgets import QInputDialog, QMessageBox

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.graph_library_panel import GraphLibraryPanel
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.graph_library import GraphLibrary
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return Dataset(name=name, dataframe=df)


def _make_panel(dataset=None, library=None):
    dataset = dataset or _make_dataset()
    manager = DatasetManager()
    manager.add(dataset)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))
    library = library if library is not None else GraphLibrary()
    panel = GraphLibraryPanel(library, lambda: figure, lambda: manager)
    return panel, figure, manager, dataset, library


def _stub_get_text(monkeypatch, text, ok=True):
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: (text, ok)))


@pytest.fixture(autouse=True)
def _no_blocking_information(monkeypatch):
    """QMessageBox.information (no-selection guards) would otherwise block
    offscreen waiting for a click. Records calls so tests can assert one
    was shown without needing a real dialog."""
    calls = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: calls.append(a)))
    return calls


# --- Save Current Panel as Graph ----------------------------------------------


def test_save_button_click_prompts_for_a_name_and_adds_a_graph(qapp, monkeypatch):
    panel, figure, manager, dataset, library = _make_panel()
    _stub_get_text(monkeypatch, "My Graph")

    panel.save_button.click()

    assert len(library.graphs) == 1
    assert library.graphs[0].name == "My Graph"
    assert panel.graph_list.count() == 1
    assert panel.graph_list.item(0).text() == "My Graph"


def test_save_button_click_cancelled_does_not_add_a_graph(qapp, monkeypatch):
    panel, *_rest, library = _make_panel()
    _stub_get_text(monkeypatch, "Ignored", ok=False)

    panel.save_button.click()

    assert len(library.graphs) == 0


def test_save_button_click_with_empty_name_does_not_add_a_graph(qapp, monkeypatch):
    panel, *_rest, library = _make_panel()
    _stub_get_text(monkeypatch, "   ")

    panel.save_button.click()

    assert len(library.graphs) == 0


def test_save_button_click_emits_graph_library_changed(qapp, monkeypatch):
    panel, *_rest = _make_panel()
    _stub_get_text(monkeypatch, "G1")
    received = []
    panel.graph_library_changed.connect(lambda: received.append(True))

    panel.save_button.click()

    assert received == [True]


# --- Load Selected Graph into Active Panel ------------------------------------


def test_load_button_replaces_active_panel_with_an_independent_copy(qapp, monkeypatch):
    dataset = _make_dataset()
    library = GraphLibrary()
    source_figure = GnoviFigure()
    source_figure.active_panel.title = "Saved Title"
    source_figure.add_series(PlotSeries.line(dataset, "x", "y", color="#123456"))
    manager = DatasetManager()
    manager.add(dataset)
    graph = library.save_panel_as_graph(source_figure, "G1", manager)

    target_figure = GnoviFigure()
    panel = GraphLibraryPanel(library, lambda: target_figure, lambda: manager)
    panel.graph_list.setCurrentRow(0)

    panel.load_button.click()

    assert target_figure.active_panel is not graph.panel
    assert target_figure.active_panel.title == "Saved Title"
    assert target_figure.active_panel.series[0].color == "#123456"


def test_load_button_emits_graph_loaded_into_panel_not_graph_library_changed(qapp):
    dataset = _make_dataset()
    library = GraphLibrary()
    figure = GnoviFigure()
    manager = DatasetManager()
    manager.add(dataset)
    library.save_panel_as_graph(figure, "G1", manager)

    panel = GraphLibraryPanel(library, lambda: figure, lambda: manager)
    panel.graph_list.setCurrentRow(0)
    loaded_events = []
    library_changed_events = []
    panel.graph_loaded_into_panel.connect(lambda: loaded_events.append(True))
    panel.graph_library_changed.connect(lambda: library_changed_events.append(True))

    panel.load_button.click()

    assert loaded_events == [True]
    assert library_changed_events == []


def test_load_button_with_no_selection_shows_a_message_and_does_not_emit(qapp, _no_blocking_information):
    panel, *_rest = _make_panel()
    events = []
    panel.graph_loaded_into_panel.connect(lambda: events.append(True))

    panel.load_button.click()

    assert events == []
    assert len(_no_blocking_information) == 1


def test_editing_the_loaded_panel_does_not_mutate_the_stored_graph(qapp):
    dataset = _make_dataset()
    library = GraphLibrary()
    source_figure = GnoviFigure()
    source_figure.add_series(PlotSeries.line(dataset, "x", "y", color="#111111"))
    manager = DatasetManager()
    manager.add(dataset)
    graph = library.save_panel_as_graph(source_figure, "G1", manager)

    target_figure = GnoviFigure()
    panel = GraphLibraryPanel(library, lambda: target_figure, lambda: manager)
    panel.graph_list.setCurrentRow(0)
    panel.load_button.click()

    target_figure.active_panel.title = "Edited After Load"
    target_figure.active_panel.series[0].color = "#ff00ff"

    assert graph.panel.title == ""
    assert graph.panel.series[0].color == "#111111"


# --- Rename / Duplicate / Delete ----------------------------------------------


def test_rename_button_updates_the_graph_name_and_list(qapp, monkeypatch):
    panel, figure, manager, dataset, library = _make_panel()
    _stub_get_text(monkeypatch, "Original")
    panel.save_button.click()
    panel.graph_list.setCurrentRow(0)

    _stub_get_text(monkeypatch, "Renamed")
    panel.rename_button.click()

    assert library.graphs[0].name == "Renamed"
    assert panel.graph_list.item(0).text() == "Renamed"


def test_rename_button_emits_graph_library_changed(qapp, monkeypatch):
    panel, *_rest = _make_panel()
    _stub_get_text(monkeypatch, "G1")
    panel.save_button.click()
    panel.graph_list.setCurrentRow(0)
    _stub_get_text(monkeypatch, "G1 renamed")
    received = []
    panel.graph_library_changed.connect(lambda: received.append(True))

    panel.rename_button.click()

    assert received == [True]


def test_duplicate_button_adds_an_independent_copy(qapp, monkeypatch):
    panel, figure, manager, dataset, library = _make_panel()
    _stub_get_text(monkeypatch, "Base")
    panel.save_button.click()
    panel.graph_list.setCurrentRow(0)

    panel.duplicate_button.click()

    assert len(library.graphs) == 2
    assert panel.graph_list.count() == 2
    names = {library.graphs[0].name, library.graphs[1].name}
    assert names == {"Base", "Base (Copy)"}


def test_delete_button_removes_the_graph(qapp, monkeypatch):
    panel, figure, manager, dataset, library = _make_panel()
    _stub_get_text(monkeypatch, "G1")
    panel.save_button.click()
    panel.graph_list.setCurrentRow(0)

    panel.delete_button.click()

    assert len(library.graphs) == 0
    assert panel.graph_list.count() == 0


def test_delete_button_with_no_selection_shows_a_message_and_does_not_raise(qapp, _no_blocking_information):
    panel, *_rest, library = _make_panel()
    panel.delete_button.click()
    assert len(library.graphs) == 0
    assert len(_no_blocking_information) == 1


# --- Repointing at a different project ----------------------------------------


def test_set_library_repoints_and_reloads_the_list(qapp, monkeypatch):
    panel, figure, manager, dataset, library = _make_panel()
    _stub_get_text(monkeypatch, "Old Project Graph")
    panel.save_button.click()
    assert panel.graph_list.count() == 1

    new_library = GraphLibrary()
    panel.set_library(new_library)

    assert panel.graph_list.count() == 0
    _stub_get_text(monkeypatch, "New Project Graph")
    panel.save_button.click()
    assert len(new_library.graphs) == 1
    assert len(library.graphs) == 1  # old library untouched
