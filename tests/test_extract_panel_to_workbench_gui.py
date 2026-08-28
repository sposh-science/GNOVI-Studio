""""Panels -> Extract Active Panel to New Workbench", driven through the
real `MainWindow` -- mirrors `test_workbench_switching_gui.py`'s style
(exercising real handlers) and reuses `test_main_window_analysis_workflow_
gui.py`'s fit-curve helper pattern for the analysis-isolation scenarios.
"""

import numpy as np
import pandas as pd

from gnovi_plot.analysis.fitting import GAUSSIAN, LINEAR
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    x = list(range(20))
    y = [3.0 * v + 2.0 for v in x]
    return Dataset(name=name, dataframe=pd.DataFrame({"x": x, "y": y}))


def _gaussian_dataset(name="peak"):
    x = np.linspace(-6.0, 6.0, 60)
    y = 4.0 * np.exp(-((x - 1.0) ** 2) / (2 * 1.5**2)) + 0.5
    return Dataset(name=name, dataframe=pd.DataFrame({"x": x, "y": y}))


def _run_fit_on_active_panel(window, dataset, model=LINEAR, label="curve"):
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y", label=label)])
    window.analysis_panel.model_combo.setCurrentIndex(window.analysis_panel.model_combo.findData(model))
    window.analysis_panel.run_fit_button.click()
    return window.analysis_result_view.result


def _fit_curve_series_id(window, result):
    return next(
        s.id
        for s in window.figure_model.active_panel.series
        if s.dataset.metadata.get("result_id") == result.result_id
    )


# --- Menu action: basic extraction ---------------------------------------------


def test_extract_action_creates_and_activates_a_new_1x1_workbench(qapp):
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    window.figure_size_panel.layout_combo.setCurrentIndex(4)  # "1 x 3"
    window.toolbar_panel_combo.setCurrentIndex(1)
    original_panel_id = window.figure_model.active_panel.id

    window._on_extract_panel_requested()

    assert window._project.active_workbench_id != workbench_a_id
    assert window.figure_model.layout == (1, 1)
    assert window.figure_model.active_panel.id != original_panel_id
    # Source Workbench keeps its original 1x3 layout, untouched.
    source = window._project.get_workbench(workbench_a_id)
    assert source.figure.layout == (1, 3)
    assert source.figure.panels[1].id == original_panel_id

    window.close()


def test_extract_action_is_enabled_for_a_1x1_source_workbench(qapp):
    window = MainWindow()
    assert window.figure_model.layout == (1, 1)
    assert window.extract_panel_action.isEnabled()
    window.close()


def test_extract_action_marks_the_project_dirty(qapp):
    window = MainWindow()
    window._set_dirty(False)

    window._on_extract_panel_requested()

    assert window._dirty is True
    window.close()


# --- Results/history sync on activation ----------------------------------------


def test_activating_the_extracted_workbench_immediately_shows_its_current_result(qapp):
    window = MainWindow()
    result = _run_fit_on_active_panel(window, _make_dataset(), label="curve")

    window._on_extract_panel_requested()

    assert window.analysis_result_view.result is not None
    assert window.analysis_result_view.result.result_id == result.result_id
    assert window.analysis_result_view.result is not result  # independent copy


# --- Add/Remove Fit Curve isolation ---------------------------------------------


def test_removing_a_fit_curve_in_the_extracted_workbench_does_not_affect_the_source(qapp):
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    _run_fit_on_active_panel(window, _make_dataset(), label="curve")
    window.analysis_panel.add_fit_curve_button.click()
    source_series_count = len(window.figure_model.active_panel.series)

    window._on_extract_panel_requested()
    assert window.analysis_panel.remove_fit_curve_button.isEnabled()
    window.analysis_panel.remove_fit_curve_button.click()

    assert len(window.figure_model.active_panel.series) == source_series_count - 1

    window._on_workbench_tab_selected(workbench_a_id)
    assert len(window.figure_model.active_panel.series) == source_series_count  # untouched

    window.close()


def test_adding_a_historical_fit_curve_in_the_extracted_workbench_works_independently(qapp):
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    result = _run_fit_on_active_panel(window, _make_dataset(), label="curve")
    # Not added to the plot in the source -- Add is available in the extraction too.

    window._on_extract_panel_requested()

    assert window.analysis_panel.add_fit_curve_button.isEnabled()
    window.analysis_panel.add_fit_curve_button.click()

    fit_series_id = _fit_curve_series_id(window, result)
    assert fit_series_id in [s.id for s in window.figure_model.active_panel.series]

    # The source Workbench's own panel never got this fit curve.
    window._on_workbench_tab_selected(workbench_a_id)
    assert window.figure_model.active_panel.series[0].dataset.metadata.get("result_id") is None

    window.close()


def test_new_fit_in_extracted_workbench_does_not_alter_source_history(qapp):
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    _run_fit_on_active_panel(window, _make_dataset(), label="curve")
    source_panel_id = window.figure_model.active_panel.id
    source_history_len = len(window._project.active_workbench.analysis_results.all(source_panel_id))

    window._on_extract_panel_requested()
    extracted_panel_id = window.figure_model.active_panel.id
    _run_fit_on_active_panel(window, _gaussian_dataset(), model=GAUSSIAN, label="peak")

    assert len(window._project.active_workbench.analysis_results.all(extracted_panel_id)) == 2

    window._on_workbench_tab_selected(workbench_a_id)
    assert len(window._project.active_workbench.analysis_results.all(source_panel_id)) == source_history_len

    window.close()


# --- Residuals -------------------------------------------------------------------


def test_residuals_resolve_live_data_in_the_extracted_workbench(qapp):
    window = MainWindow()
    _run_fit_on_active_panel(window, _make_dataset(), label="curve")

    window._on_extract_panel_requested()
    window.analysis_result_view._view_residuals_button.click()

    residual_window = window.analysis_result_view._residual_window
    assert residual_window is not None
    assert residual_window.isVisible()

    window.close()


def test_open_residual_window_updates_when_switching_to_the_extracted_workbench(qapp):
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    _run_fit_on_active_panel(window, _make_dataset(), label="curve")
    window.analysis_result_view._view_residuals_button.click()
    residual_window = window.analysis_result_view._residual_window
    assert residual_window is not None
    assert residual_window.isVisible()
    first_title = residual_window.windowTitle()

    window._on_extract_panel_requested()

    # Same reusable window instance, retitled for the extracted panel's own
    # (copied) current result -- never a second window.
    assert window.analysis_result_view._residual_window is residual_window
    assert residual_window.isVisible()
    assert residual_window.windowTitle() == first_title  # same model/result, same subtitle

    window._on_workbench_tab_selected(workbench_a_id)
    assert window.analysis_result_view._residual_window is residual_window
    assert residual_window.isVisible()

    window.close()


def test_open_residual_window_hides_when_extracting_a_panel_with_no_history(qapp):
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    window.toolbar_panel_combo.setCurrentIndex(0)
    _run_fit_on_active_panel(window, _make_dataset(), label="curve")
    window.analysis_result_view._view_residuals_button.click()
    residual_window = window.analysis_result_view._residual_window
    assert residual_window.isVisible()

    window.toolbar_panel_combo.setCurrentIndex(1)  # panel with no history
    window._on_extract_panel_requested()

    assert window.analysis_result_view._residual_window is residual_window  # kept alive, not destroyed
    assert not residual_window.isVisible()

    window.close()


# --- Double extraction -----------------------------------------------------------


def test_extracting_the_same_panel_twice_creates_independent_workbenches(qapp):
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    _run_fit_on_active_panel(window, _make_dataset(), label="curve")

    window._on_workbench_tab_selected(workbench_a_id)
    window._on_extract_panel_requested()
    first_id = window._project.active_workbench_id
    first_panel_id = window.figure_model.active_panel.id

    window._on_workbench_tab_selected(workbench_a_id)
    window._on_extract_panel_requested()
    second_id = window._project.active_workbench_id
    second_panel_id = window.figure_model.active_panel.id

    assert first_id != second_id
    assert first_panel_id != second_panel_id
    assert len({workbench_a_id, first_id, second_id}) == 3

    window.close()


# --- Save / reopen -----------------------------------------------------------------


def test_save_reopen_preserves_both_workbenches_and_analysis_linkage(qapp, tmp_path):
    from gnovi_plot.core.project_io import load_project, save_project

    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    result = _run_fit_on_active_panel(window, _make_dataset(), label="curve")
    window.analysis_panel.add_fit_curve_button.click()

    window._on_extract_panel_requested()
    extracted_id = window._project.active_workbench_id
    extracted_panel_id = window.figure_model.active_panel.id

    out_path = tmp_path / "proj.gnovi"
    save_project(window._project, out_path)
    reloaded = load_project(out_path)
    window._load_project_into_window(reloaded)

    assert {w.id for w in window._project.workbenches} == {workbench_a_id, extracted_id}

    extracted_workbench = window._project.get_workbench(extracted_id)
    restored_result = extracted_workbench.analysis_results.current(extracted_panel_id)
    assert restored_result is not None
    assert restored_result.result_id == result.result_id

    window._on_workbench_tab_selected(extracted_id)
    assert window.analysis_result_view.result.result_id == result.result_id
    assert not window.analysis_panel.add_fit_curve_button.isEnabled()
    assert window.analysis_panel.remove_fit_curve_button.isEnabled()

    window._on_workbench_tab_selected(workbench_a_id)
    assert window.analysis_result_view.result.result_id == result.result_id
    assert window.analysis_panel.remove_fit_curve_button.isEnabled()

    window.close()
