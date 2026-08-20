"""End-to-end "select plotted curve -> Analysis -> Curve Fitting -> Run Fit
-> Results" workflow, driven through the real `MainWindow` -- mirrors
`test_workbench_switching_gui.py`'s style of exercising the real handlers
rather than calling `AnalysisPanel` in isolation (see
`test_analysis_panel_gui.py` for the isolated widget-level coverage).
"""

import pandas as pd

from gnovi_plot.analysis.fitting import LINEAR
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    x = list(range(20))
    y = [3.0 * v + 2.0 for v in x]
    return Dataset(name=name, dataframe=pd.DataFrame({"x": x, "y": y}))


# --- Page registration -------------------------------------------------------


def test_analysis_page_is_registered_as_a_single_left_drawer_entry(qapp):
    window = MainWindow()

    assert "analysis" in window.tool_drawer._buttons
    window.tool_drawer._buttons["analysis"].click()
    assert window.tool_drawer.active_key == "analysis"

    window.close()


def _all_action_texts(menu) -> list[str]:
    texts = []
    for action in menu.actions():
        if action.text():
            texts.append(action.text())
        if action.menu() is not None:
            texts.extend(_all_action_texts(action.menu()))
    return texts


def test_analysis_is_the_only_new_navigation_entry(qapp):
    """No menu/toolbar duplicate for Analysis -- the drawer page is the one
    and only place to reach it."""
    from PySide6.QtWidgets import QToolBar

    window = MainWindow()

    menu_texts = _all_action_texts(window.menuBar())
    assert not any("analysis" in text.lower() for text in menu_texts)

    toolbar_texts = [
        action.text()
        for toolbar in window.findChildren(QToolBar)
        for action in toolbar.actions()
        if action.text()
    ]
    assert not any("analysis" in text.lower() for text in toolbar_texts)

    window.close()


# --- Active-panel changes (real UI-driven panel switch) ----------------------


def test_switching_the_active_panel_retargets_analysis_panel(qapp):
    window = MainWindow()
    figure_a = window.figure_model
    assert window.analysis_panel._figure is figure_a

    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2"
    window._on_add_to_plot([PlotSeries.line(_make_dataset("panel1"), "x", "y", label="Panel 1 curve")])

    window.toolbar_panel_combo.setCurrentIndex(1)  # switch to Panel 2
    window._on_add_to_plot([PlotSeries.line(_make_dataset("panel2"), "x", "y", label="Panel 2 curve")])

    labels = [
        window.analysis_panel.source_combo.itemText(i)
        for i in range(window.analysis_panel.source_combo.count())
    ]
    assert labels == ["Panel 2 curve"]

    window.toolbar_panel_combo.setCurrentIndex(0)  # back to Panel 1
    labels = [
        window.analysis_panel.source_combo.itemText(i)
        for i in range(window.analysis_panel.source_combo.count())
    ]
    assert labels == ["Panel 1 curve"]

    window.close()


# --- Workbench switch retargets the panel like the other four ----------------


def test_workbench_switch_retargets_analysis_panel(qapp):
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    window.workbench_tab_bar.new_button.click()
    workbench_b_id = window._project.active_workbench_id
    figure_b = window._project.get_workbench(workbench_b_id).figure
    assert window.analysis_panel._figure is figure_b

    window._on_workbench_tab_selected(workbench_a_id)
    figure_a = window._project.get_workbench(workbench_a_id).figure
    assert window.analysis_panel._figure is figure_a

    window.close()


# --- Full Run Fit workflow: result routing + automatic Results-tab activation


def test_run_fit_routes_result_to_results_view_and_shows_results_tab(qapp):
    window = MainWindow()
    window._on_add_to_plot([PlotSeries.line(_make_dataset(), "x", "y", label="Fittable curve")])

    # Start from the Analysis page, bottom panel hidden, on a different tab --
    # the workflow must not require the user to manually open Results.
    window.tool_drawer._buttons["analysis"].click()
    window.toggle_bottom_panel_action.setChecked(False)
    window.bottom_panel.setCurrentIndex(0)  # Data tab
    assert window.analysis_result_view.result is None

    window.analysis_panel.model_combo.setCurrentIndex(window.analysis_panel.model_combo.findData(LINEAR))
    window.analysis_panel.run_fit_button.click()

    assert window.analysis_result_view.result is not None
    assert window.analysis_result_view.result.model == LINEAR
    assert window.bottom_panel.isVisibleTo(window)
    assert window.bottom_panel.tabText(window.bottom_panel.currentIndex()) == "Results"
    assert window.toggle_bottom_panel_action.isChecked()

    window.close()


def test_run_fit_creates_no_new_dataset_in_the_project(qapp):
    window = MainWindow()
    window._on_add_to_plot([PlotSeries.line(_make_dataset(), "x", "y", label="Fittable curve")])
    before_count = len(window.dataset_manager.datasets)

    window.tool_drawer._buttons["analysis"].click()
    window.analysis_panel.run_fit_button.click()

    assert len(window.dataset_manager.datasets) == before_count
    assert len(window.figure_model.series) == 1  # unchanged -- no fit curve plotted

    window.close()
