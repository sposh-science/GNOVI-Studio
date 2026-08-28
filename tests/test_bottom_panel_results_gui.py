from __future__ import annotations

from PySide6.QtWidgets import QLabel, QScrollArea

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
    """`set_results_widget` wraps `view` in a `QScrollArea` (see that
    method's own docstring -- a structural fix so a large result's
    content can never dictate this tab's, and therefore the central
    splitter's, size). The layout's one child is now that QScrollArea,
    with `view` reparented onto it as its `.widget()` -- this test's
    actual intent (does `set_results_widget` place/reparent `view` into
    the Results tab) is unchanged, just checked through the wrapper."""
    panel = BottomPanel()
    view = AnalysisResultView(GnoviFigure(), DatasetManager())

    panel.set_results_widget(view)

    index = _results_tab_index(panel)
    results_tab = panel.widget(index)
    assert results_tab.layout().count() == 1
    scroll_area = results_tab.layout().itemAt(0).widget()
    assert isinstance(scroll_area, QScrollArea)
    assert scroll_area.widget() is view
    assert view.parent() is scroll_area.viewport()


def test_set_results_widget_follows_the_same_pattern_as_the_other_setters(qapp):
    """set_results_widget should reparent an arbitrary widget exactly like
    set_data_widget/set_graphs_widget/set_transformations_widget do --
    BottomPanel must not require its Results content to be an
    AnalysisResultView specifically. See `test_set_results_widget_places_
    widget_in_results_tab`'s own docstring for the QScrollArea wrapper
    this now checks through."""
    panel = BottomPanel()
    placeholder = QLabel("stand-in widget")

    panel.set_results_widget(placeholder)

    index = _results_tab_index(panel)
    results_tab = panel.widget(index)
    scroll_area = results_tab.layout().itemAt(0).widget()
    assert isinstance(scroll_area, QScrollArea)
    assert scroll_area.widget() is placeholder


def test_results_tab_no_longer_has_static_placeholder_text(qapp):
    panel = BottomPanel()
    index = _results_tab_index(panel)
    results_tab = panel.widget(index)

    assert not isinstance(results_tab, QLabel)
