from __future__ import annotations

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
    assert results_tab.layout().itemAt(0).widget() is window.analysis_result_view

    window.close()
