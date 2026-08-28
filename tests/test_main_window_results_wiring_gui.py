from __future__ import annotations

from PySide6.QtWidgets import QScrollArea

from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.widgets.analysis_result_view import AnalysisResultView


def test_main_window_wires_an_analysis_result_view_into_the_results_tab(qapp):
    window = MainWindow()

    assert isinstance(window.analysis_result_view, AnalysisResultView)
    assert window.analysis_result_view.result is None  # nothing produces one yet

    index = next(
        i for i in range(window.bottom_panel.count()) if window.bottom_panel.tabText(i) == "Results"
    )
    results_tab = window.bottom_panel.widget(index)
    # `BottomPanel.set_results_widget` wraps the results widget in a
    # `QScrollArea` (a structural fix so a large result can never dictate
    # this tab's -- and therefore the central splitter's -- size); the
    # layout's one child is that QScrollArea, with `analysis_result_view`
    # reparented onto it as its `.widget()`.
    scroll_area = results_tab.layout().itemAt(0).widget()
    assert isinstance(scroll_area, QScrollArea)
    assert scroll_area.widget() is window.analysis_result_view

    window.close()
