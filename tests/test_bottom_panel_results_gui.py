from __future__ import annotations

from PySide6.QtWidgets import QLabel

from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.analysis_result_view import AnalysisResultView
from gnovi_plot.gui.widgets.bottom_panel import BottomPanel
from gnovi_plot.plotting.figure import GnoviFigure


def _results_tab_index(panel: BottomPanel) -> int:
    for i in range(panel.count()):
        if panel.tabText(i) == "Results":
            return i
    raise AssertionError("Results tab not found")


def test_results_tab_exists_and_starts_empty(qapp):
    panel = BottomPanel()
    index = _results_tab_index(panel)
    results_tab = panel.widget(index)

    assert results_tab.layout() is not None
    assert results_tab.layout().count() == 0


def test_set_results_widget_places_widget_in_results_tab(qapp):
    panel = BottomPanel()
    view = AnalysisResultView(GnoviFigure(), DatasetManager())

    panel.set_results_widget(view)

    index = _results_tab_index(panel)
    results_tab = panel.widget(index)
    assert results_tab.layout().count() == 1
    assert results_tab.layout().itemAt(0).widget() is view
    assert view.parent() is results_tab


def test_set_results_widget_follows_the_same_pattern_as_the_other_setters(qapp):
    """set_results_widget should reparent an arbitrary widget exactly like
    set_data_widget/set_graphs_widget/set_transformations_widget do --
    BottomPanel must not require its Results content to be an
    AnalysisResultView specifically."""
    panel = BottomPanel()
    placeholder = QLabel("stand-in widget")

    panel.set_results_widget(placeholder)

    index = _results_tab_index(panel)
    results_tab = panel.widget(index)
    assert results_tab.layout().itemAt(0).widget() is placeholder


def test_results_tab_no_longer_has_static_placeholder_text(qapp):
    panel = BottomPanel()
    index = _results_tab_index(panel)
    results_tab = panel.widget(index)

    assert not isinstance(results_tab, QLabel)
