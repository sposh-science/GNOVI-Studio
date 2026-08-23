"""End-to-end "select plotted curve -> Analysis -> Curve Fitting -> Run Fit
-> Results" workflow, driven through the real `MainWindow` -- mirrors
`test_workbench_switching_gui.py`'s style of exercising the real handlers
rather than calling `AnalysisPanel` in isolation (see
`test_analysis_panel_gui.py` for the isolated widget-level coverage).
"""

import numpy as np
import pandas as pd

from gnovi_plot.analysis.fitting import GAUSSIAN, LINEAR, POLYNOMIAL
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    x = list(range(20))
    y = [3.0 * v + 2.0 for v in x]
    return Dataset(name=name, dataframe=pd.DataFrame({"x": x, "y": y}))


def _quadratic_dataset(name="q"):
    x = list(range(20))
    y = [0.5 * v * v + 1.0 for v in x]
    return Dataset(name=name, dataframe=pd.DataFrame({"x": x, "y": y}))


def _gaussian_dataset(name="peak"):
    x = np.linspace(-6.0, 6.0, 60)
    y = 4.0 * np.exp(-((x - 1.0) ** 2) / (2 * 1.5**2)) + 0.5
    return Dataset(name=name, dataframe=pd.DataFrame({"x": x, "y": y}))


def _run_two_fits_on_active_panel(window, label_prefix="peak"):
    """Plots one series and runs Gaussian then Linear against it, adding
    both curves to the plot -- the exact repro scenario from the manual
    testing report: a panel accumulating two completed fits, both plotted,
    with the History list's default selection landing on the latest
    (Linear). Returns `(gaussian_result, linear_result)`."""
    window._on_add_to_plot([PlotSeries.line(_gaussian_dataset(), "x", "y", label=label_prefix)])
    analysis_panel = window.analysis_panel

    analysis_panel.model_combo.setCurrentIndex(analysis_panel.model_combo.findData(GAUSSIAN))
    analysis_panel.run_fit_button.click()
    gaussian_result = window.analysis_result_view.result
    analysis_panel.add_fit_curve_button.click()

    analysis_panel.model_combo.setCurrentIndex(analysis_panel.model_combo.findData(LINEAR))
    analysis_panel.run_fit_button.click()
    linear_result = window.analysis_result_view.result
    analysis_panel.add_fit_curve_button.click()

    return gaussian_result, linear_result


def _fit_curve_series_id(window, result):
    return next(
        s.id
        for s in window.figure_model.active_panel.series
        if s.dataset.metadata.get("result_id") == result.result_id
    )


def _run_fit_on_active_panel(window, dataset, model=LINEAR, label="curve"):
    """Add `dataset` as a line series to the currently active panel and
    run `model` against it -- mirrors `test_run_fit_routes_result_to_
    results_view_and_shows_results_tab`'s own sequence. Returns the
    `AnalysisResult` now shown in Results (same object `analysis_panel.
    run_fit_button` produced -- callers use `is` to prove later panel/
    Workbench switches restore this exact object, never recompute it)."""
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y", label=label)])
    window.analysis_panel.model_combo.setCurrentIndex(window.analysis_panel.model_combo.findData(model))
    window.analysis_panel.run_fit_button.click()
    return window.analysis_result_view.result


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


# --- Panel-scoped analysis-result history: restore on switch, not "most ------
# recent fit run anywhere" (the exact bug this architecture fixes) ------------


def test_panel_switch_restores_each_panels_own_result_and_residual_window_follows(qapp):
    """The exact workflow that exposed the bug: 3 panels, two independent
    fits, Results/ResidualWindow must always reflect the *active* panel's
    own result -- never "the most recent fit run anywhere", and switching
    must never recompute anything (proven via object identity)."""
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(4)  # "1 x 3"

    window.toolbar_panel_combo.setCurrentIndex(0)
    result_a = _run_fit_on_active_panel(window, _make_dataset("panel1"), model=LINEAR, label="Panel 1 curve")

    window.toolbar_panel_combo.setCurrentIndex(1)
    result_b = _run_fit_on_active_panel(
        window, _quadratic_dataset("panel2"), model=POLYNOMIAL, label="Panel 2 curve"
    )
    assert result_a is not result_b

    # Panel 3 never had a fit run against it.
    window.toolbar_panel_combo.setCurrentIndex(2)
    assert window.analysis_result_view.result is None

    # Panel 1 -- its own result restores, the exact same object (no recomputation).
    window.toolbar_panel_combo.setCurrentIndex(0)
    assert window.analysis_result_view.result is result_a

    window.analysis_result_view._view_residuals_button.click()
    residual_window = window.analysis_result_view._residual_window
    assert residual_window is not None
    assert residual_window.isVisible()
    first_title = residual_window.windowTitle()

    # Panel 2 -- Results shows B, the SAME ResidualWindow instance updates to B.
    window.toolbar_panel_combo.setCurrentIndex(1)
    assert window.analysis_result_view.result is result_b
    assert window.analysis_result_view._residual_window is residual_window
    assert residual_window.isVisible()
    assert residual_window.windowTitle() != first_title

    # Panel 3 -- Results clears, residual window hides (kept alive, not destroyed).
    window.toolbar_panel_combo.setCurrentIndex(2)
    assert window.analysis_result_view.result is None
    assert window.analysis_result_view._residual_window is residual_window
    assert not residual_window.isVisible()

    # Back to Panel 1 -- result A restores (same object), View Residuals reopens it.
    window.toolbar_panel_combo.setCurrentIndex(0)
    assert window.analysis_result_view.result is result_a

    window.analysis_result_view._view_residuals_button.click()
    assert window.analysis_result_view._residual_window is residual_window
    assert residual_window.isVisible()

    window.close()


def test_panel_with_no_result_shows_empty_results_state(qapp):
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    window.toolbar_panel_combo.setCurrentIndex(0)
    _run_fit_on_active_panel(window, _make_dataset(), label="curve")

    window.toolbar_panel_combo.setCurrentIndex(1)

    assert window.analysis_result_view.result is None
    assert window.analysis_result_view._empty_label.isVisibleTo(window.analysis_result_view)

    window.close()


def test_workbench_switch_restores_each_workbenchs_own_result(qapp):
    """Ownership is by stable Panel.id, not panel index/name: both
    Workbenches' active panel is index 0 here, yet each restores its own,
    distinct result."""
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    result_a = _run_fit_on_active_panel(window, _make_dataset("wa"), label="Workbench A curve")

    window.workbench_tab_bar.new_button.click()
    workbench_b_id = window._project.active_workbench_id
    assert workbench_b_id != workbench_a_id
    result_b = _run_fit_on_active_panel(
        window, _quadratic_dataset("wb"), model=POLYNOMIAL, label="Workbench B curve"
    )
    assert window.analysis_result_view.result is result_b

    figure_a = window._project.get_workbench(workbench_a_id).figure
    figure_b = window._project.get_workbench(workbench_b_id).figure
    assert figure_a.active_panel_index == figure_b.active_panel_index == 0
    assert figure_a.active_panel.id != figure_b.active_panel.id  # ownership is by Panel.id

    window._on_workbench_tab_selected(workbench_a_id)
    assert window.analysis_result_view.result is result_a

    window._on_workbench_tab_selected(workbench_b_id)
    assert window.analysis_result_view.result is result_b

    # No cross-contamination even after repeated switching.
    window._on_workbench_tab_selected(workbench_a_id)
    window._on_workbench_tab_selected(workbench_b_id)
    window._on_workbench_tab_selected(workbench_a_id)
    assert window.analysis_result_view.result is result_a

    window.close()


def test_running_three_fits_on_the_same_panel_preserves_all_of_them(qapp):
    window = MainWindow()
    ds = _make_dataset()
    window._on_add_to_plot([PlotSeries.line(ds, "x", "y", label="curve")])
    panel_id = window.figure_model.active_panel.id

    results = []
    for _ in range(3):
        window.analysis_panel.run_fit_button.click()
        results.append(window.analysis_result_view.result)

    history = window._project.active_workbench.analysis_results
    assert len(history.all(panel_id)) == 3
    assert history.all(panel_id) == results
    assert history.current(panel_id) is results[2]

    window.close()


def test_layout_shrink_prunes_the_removed_panels_history_only(qapp):
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2"

    window.toolbar_panel_combo.setCurrentIndex(0)
    result_survivor = _run_fit_on_active_panel(window, _make_dataset("p1"), label="Panel 1 curve")
    survivor_panel_id = window.figure_model.panels[0].id

    window.toolbar_panel_combo.setCurrentIndex(3)
    result_removed = _run_fit_on_active_panel(
        window, _quadratic_dataset("p4"), model=POLYNOMIAL, label="Panel 4 curve"
    )
    removed_panel_id = window.figure_model.panels[3].id

    history = window._project.active_workbench.analysis_results
    assert history.current(removed_panel_id) is result_removed

    # Shrink back to a single panel -- panels 2-4 (including the one with a
    # result) are dropped.
    window.toolbar_panel_combo.setCurrentIndex(0)
    window.figure_size_panel.layout_combo.setCurrentIndex(0)  # "1 x 1"

    assert history.current(removed_panel_id) is None
    assert history.all(removed_panel_id) == []
    # The surviving panel's own history is untouched -- no reassignment to
    # a different panel.
    assert history.current(survivor_panel_id) is result_survivor
    assert window.analysis_result_view.result is result_survivor

    window.close()


def test_duplicate_workbench_via_gui_action_starts_with_empty_history(qapp):
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    result_a = _run_fit_on_active_panel(window, _make_dataset("orig"), label="Original curve")
    original_panel_id = window.figure_model.active_panel.id

    window._on_duplicate_workbench_requested(workbench_a_id)

    copy_id = window._project.active_workbench_id
    assert copy_id != workbench_a_id
    assert window.analysis_result_view.result is None  # fresh, empty history
    assert window.figure_model.active_panel.id != original_panel_id  # fresh Panel.id (PR #5)

    # The original Workbench's own history is completely untouched.
    window._on_workbench_tab_selected(workbench_a_id)
    assert window.analysis_result_view.result is result_a

    window.close()


def test_undo_redo_of_a_layout_change_does_not_duplicate_history_or_misattribute_results(qapp):
    window = MainWindow()
    result_a = _run_fit_on_active_panel(window, _make_dataset("p1"), label="Panel 1 curve")
    panel0_id = window.figure_model.panels[0].id
    history = window._project.active_workbench.analysis_results

    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2" -- undoable
    assert window.undo_action.isEnabled() is True

    window._on_undo()
    assert len(window.figure_model.panels) == 1
    assert window.figure_model.panels[0].id == panel0_id
    assert window.analysis_result_view.result is result_a
    assert len(history.all(panel0_id)) == 1  # not duplicated by the undo

    window._on_redo()
    assert len(window.figure_model.panels) == 2
    assert window.figure_model.panels[0].id == panel0_id
    assert len(history.all(panel0_id)) == 1  # still not duplicated by the redo
    assert history.current(panel0_id) is result_a

    window.close()


def test_a_result_with_no_source_panel_id_is_shown_but_never_added_to_history(qapp):
    """A result not associated with any panel (`source_panel_id=None` --
    reachable today by calling `fit_curve()` directly without one, e.g. a
    future analysis tool with no panel context) must still display
    normally, just never be recorded in any Workbench's panel-scoped
    history -- there is no panel id to key it under."""
    import numpy as np

    from gnovi_plot.analysis.fitting import fit_curve

    window = MainWindow()
    x = np.linspace(0, 10, 10)
    y = 2.0 * x
    result = fit_curve(x, y, LINEAR, source_dataset_id="dataset-x", x_column="x", y_column="y")
    assert result.source_panel_id is None

    window._on_analysis_result_ready(result)

    assert window.analysis_result_view.result is result
    for workbench in window._project.workbenches:
        assert workbench.analysis_results.all(window.figure_model.active_panel.id) == []

    window.close()


# --- Add / Remove Fit Curve: strict one-curve-per-result invariant -----------


def test_run_fit_enables_add_and_disables_remove(qapp):
    window = MainWindow()
    _run_fit_on_active_panel(window, _make_dataset(), label="curve")

    assert window.analysis_panel.add_fit_curve_button.isEnabled()
    assert not window.analysis_panel.remove_fit_curve_button.isEnabled()

    window.close()


def test_add_fit_curve_disables_add_and_enables_remove(qapp):
    window = MainWindow()
    _run_fit_on_active_panel(window, _make_dataset(), label="curve")

    window.analysis_panel.add_fit_curve_button.click()

    assert not window.analysis_panel.add_fit_curve_button.isEnabled()
    assert window.analysis_panel.remove_fit_curve_button.isEnabled()

    window.close()


def test_attempting_to_add_a_second_time_creates_no_second_fit_series(qapp):
    """Required regression: Run Fit -> Add Fit Curve -> attempt to add
    again -> no second fit series is created. The button is disabled
    after the first add (a real click on a disabled QPushButton is a
    Qt no-op), and the handler itself defends against it too, even if
    called directly."""
    window = MainWindow()
    _run_fit_on_active_panel(window, _make_dataset(), label="curve")

    window.analysis_panel.add_fit_curve_button.click()
    count_after_first_add = len(window.figure_model.active_panel.series)
    assert count_after_first_add == 2  # source curve + fit curve

    assert not window.analysis_panel.add_fit_curve_button.isEnabled()
    window.analysis_panel.add_fit_curve_button.click()  # disabled -- Qt no-op
    assert len(window.figure_model.active_panel.series) == count_after_first_add

    window.analysis_panel._on_add_fit_curve_clicked()  # defensive, even direct
    assert len(window.figure_model.active_panel.series) == count_after_first_add

    window.close()


def test_remove_fit_curve_removes_exactly_the_generated_series(qapp):
    """Required regression: Remove Fit Curve -> exactly the generated
    series disappears -> FitResult remains current -> Results remain
    visible -> residuals still work -> Add Fit Curve becomes available
    again."""
    window = MainWindow()
    result = _run_fit_on_active_panel(window, _make_dataset(), label="curve")
    window.analysis_panel.add_fit_curve_button.click()
    series_before = list(window.figure_model.active_panel.series)
    assert len(series_before) == 2
    fit_series = next(s for s in series_before if s.dataset.metadata.get("result_id") == result.result_id)
    source_series = next(s for s in series_before if s is not fit_series)

    window.analysis_panel.remove_fit_curve_button.click()

    remaining_ids = [s.id for s in window.figure_model.active_panel.series]
    assert fit_series.id not in remaining_ids
    assert source_series.id in remaining_ids  # only the fit curve is removed
    assert len(remaining_ids) == 1

    # FitResult remains current; Results remain visible.
    assert window.analysis_result_view.result is result
    active_panel_id = window.figure_model.active_panel.id
    assert window._project.active_workbench.analysis_results.current(active_panel_id) is result

    # Residuals still work.
    window.analysis_result_view._view_residuals_button.click()
    assert window.analysis_result_view._residual_window is not None
    assert window.analysis_result_view._residual_window.isVisible()

    # Add Fit Curve becomes available again.
    assert window.analysis_panel.add_fit_curve_button.isEnabled()
    assert not window.analysis_panel.remove_fit_curve_button.isEnabled()

    window.close()


def test_removing_the_fit_curve_via_the_series_page_resets_analysis_panel_state(qapp):
    """If the user deletes the fit curve through the normal Series page
    instead of Analysis's own button, the Analysis panel must detect
    this through the existing central figure-content-change funnel and
    return to Add enabled / Remove disabled -- never leave it believing a
    deleted curve still exists."""
    window = MainWindow()
    result = _run_fit_on_active_panel(window, _make_dataset(), label="curve")
    window.analysis_panel.add_fit_curve_button.click()
    assert not window.analysis_panel.add_fit_curve_button.isEnabled()
    assert window.analysis_panel.remove_fit_curve_button.isEnabled()

    fit_series = next(
        s for s in window.figure_model.active_panel.series if s.dataset.metadata.get("result_id") == result.result_id
    )
    window.series_panel.refresh(select_id=fit_series.id)
    window.series_panel.remove_button.click()

    assert fit_series.id not in [s.id for s in window.figure_model.active_panel.series]
    assert window.analysis_panel.add_fit_curve_button.isEnabled()
    assert not window.analysis_panel.remove_fit_curve_button.isEnabled()

    window.close()


def test_reopening_a_project_with_a_fit_curve_restores_disabled_add_enabled_remove(qapp, tmp_path):
    """If both the FitResult history and its fit PlotSeries were saved,
    reopening must restore: Results -> correct FitResult, fit curve ->
    present, Add Fit Curve to Plot -> disabled, Remove Fit Curve from
    Plot -> enabled. Determined from the stable result_id stored in the
    derived Dataset's metadata, never from labels."""
    from gnovi_plot.core.project_io import load_project, save_project

    window = MainWindow()
    result = _run_fit_on_active_panel(window, _make_dataset(), label="curve")
    window.analysis_panel.add_fit_curve_button.click()
    out_path = tmp_path / "proj.gnovi"
    save_project(window._project, out_path)

    reloaded = load_project(out_path)
    window._load_project_into_window(reloaded)

    restored_result = window.analysis_result_view.result
    assert restored_result is not None
    assert restored_result.result_id == result.result_id
    assert not window.analysis_panel.add_fit_curve_button.isEnabled()
    assert window.analysis_panel.remove_fit_curve_button.isEnabled()

    window.close()


# --- Dirty state: adding to history dirties; display sync alone never does ---


def test_running_a_fit_marks_the_project_dirty(qapp):
    window = MainWindow()
    window._on_add_to_plot([PlotSeries.line(_make_dataset(), "x", "y", label="curve")])
    window._set_dirty(False)

    window.analysis_panel.run_fit_button.click()

    assert window._dirty is True

    window.close()


def test_switching_panels_never_marks_the_project_dirty(qapp):
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    _run_fit_on_active_panel(window, _make_dataset(), label="curve")
    window._set_dirty(False)

    window.toolbar_panel_combo.setCurrentIndex(1)
    window.toolbar_panel_combo.setCurrentIndex(0)

    assert window._dirty is False

    window.close()


def test_viewing_residuals_never_marks_the_project_dirty(qapp):
    window = MainWindow()
    _run_fit_on_active_panel(window, _make_dataset(), label="curve")
    window._set_dirty(False)

    window.analysis_result_view._view_residuals_button.click()

    assert window._dirty is False

    window.close()


def test_removing_a_fit_curve_marks_the_project_dirty(qapp):
    """The plotted figure changed -- same as any other series removal."""
    window = MainWindow()
    _run_fit_on_active_panel(window, _make_dataset(), label="curve")
    window.analysis_panel.add_fit_curve_button.click()
    window._set_dirty(False)

    window.analysis_panel.remove_fit_curve_button.click()

    assert window._dirty is True

    window.close()


# --- Analysis History: selecting an older result (required manual repro) ----


def test_selecting_an_older_history_result_targets_add_remove_at_it_independently(qapp):
    """Required regression, from manual testing: a panel with two
    completed fits (Gaussian, then Linear), both curves on the plot.
    Selecting the older Gaussian entry in Analysis History must make
    Results/Remove act on Gaussian specifically -- removing it must leave
    the Linear curve untouched and keep both FitResults in history.
    Selecting Linear afterward must work independently, the same way."""
    window = MainWindow()
    gaussian_result, linear_result = _run_two_fits_on_active_panel(window)
    analysis_panel = window.analysis_panel
    panel_id = window.figure_model.active_panel.id
    history = window._project.active_workbench.analysis_results

    assert [r.result_id for r in history.all(panel_id)] == [gaussian_result.result_id, linear_result.result_id]
    assert history.current(panel_id).result_id == linear_result.result_id  # latest auto-current
    assert analysis_panel.history_list.count() == 2

    gaussian_series_id = _fit_curve_series_id(window, gaussian_result)
    linear_series_id = _fit_curve_series_id(window, linear_result)

    # Select the older (Gaussian, row 0) entry.
    analysis_panel.history_list.setCurrentRow(0)

    assert window.analysis_result_view.result.result_id == gaussian_result.result_id
    assert analysis_panel.remove_fit_curve_button.isEnabled()
    assert history.current(panel_id).result_id == gaussian_result.result_id

    analysis_panel.remove_fit_curve_button.click()

    remaining_ids = {s.id for s in window.figure_model.active_panel.series}
    assert gaussian_series_id not in remaining_ids
    assert linear_series_id in remaining_ids  # only the Gaussian curve disappeared
    assert [r.result_id for r in history.all(panel_id)] == [
        gaussian_result.result_id,
        linear_result.result_id,
    ]  # both FitResults remain in history
    assert analysis_panel.add_fit_curve_button.isEnabled()

    # Select Linear -- its curve remains, and Remove now targets it instead.
    analysis_panel.history_list.setCurrentRow(1)
    assert window.analysis_result_view.result.result_id == linear_result.result_id
    assert analysis_panel.remove_fit_curve_button.isEnabled()

    analysis_panel.remove_fit_curve_button.click()

    assert linear_series_id not in {s.id for s in window.figure_model.active_panel.series}
    assert len(history.all(panel_id)) == 2  # still both FitResults, nothing deleted from history

    window.close()


def test_history_selection_and_removal_are_isolated_per_panel(qapp):
    """The same scenario, repeated on a second panel: each panel's own
    two fits/selection/removal must never affect the other panel's."""
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"

    window.toolbar_panel_combo.setCurrentIndex(0)
    panel1_gaussian, panel1_linear = _run_two_fits_on_active_panel(window, label_prefix="p1")
    panel1_id = window.figure_model.active_panel.id

    window.toolbar_panel_combo.setCurrentIndex(1)
    panel2_gaussian, panel2_linear = _run_two_fits_on_active_panel(window, label_prefix="p2")
    panel2_id = window.figure_model.active_panel.id

    history = window._project.active_workbench.analysis_results
    window.analysis_panel.history_list.setCurrentRow(0)  # select Panel 2's Gaussian entry
    window.analysis_panel.remove_fit_curve_button.click()

    assert history.current(panel2_id).result_id == panel2_gaussian.result_id
    assert history.current(panel1_id).result_id == panel1_linear.result_id  # untouched

    window.toolbar_panel_combo.setCurrentIndex(0)
    assert window.analysis_result_view.result.result_id == panel1_linear.result_id
    assert len(history.all(panel1_id)) == 2
    assert _fit_curve_series_id(window, panel1_gaussian)  # Panel 1's Gaussian curve still exists
    assert _fit_curve_series_id(window, panel1_linear)  # Panel 1's Linear curve still exists

    window.close()


# --- Save/reopen: required test -- history order + explicit selection survive ---


def test_reopening_a_project_preserves_history_order_and_the_explicitly_selected_current_result(qapp, tmp_path):
    """Required regression: two fits in history, the older (Gaussian)
    explicitly selected as current, then saved and reopened -- history
    order, the selection itself, Results, and Add/Remove Fit Curve state
    must all survive exactly."""
    from gnovi_plot.core.project_io import load_project, save_project

    window = MainWindow()
    gaussian_result, linear_result = _run_two_fits_on_active_panel(window)
    window.analysis_panel.history_list.setCurrentRow(0)  # explicitly select Gaussian (not the latest)
    assert window.analysis_result_view.result.result_id == gaussian_result.result_id

    out_path = tmp_path / "history.gnovi"
    save_project(window._project, out_path)

    reloaded = load_project(out_path)
    window._load_project_into_window(reloaded)

    panel_id = window.figure_model.active_panel.id
    history = window._project.active_workbench.analysis_results
    assert [r.result_id for r in history.all(panel_id)] == [gaussian_result.result_id, linear_result.result_id]
    assert history.current(panel_id).result_id == gaussian_result.result_id  # selection survived

    restored_result = window.analysis_result_view.result
    assert restored_result.result_id == gaussian_result.result_id
    assert window.analysis_panel.remove_fit_curve_button.isEnabled()  # Gaussian's curve is present
    assert not window.analysis_panel.add_fit_curve_button.isEnabled()

    # Reselecting the restored Linear entry works too, from the reloaded state.
    window.analysis_panel.history_list.setCurrentRow(1)
    assert window.analysis_result_view.result.result_id == linear_result.result_id
    assert window.analysis_panel.remove_fit_curve_button.isEnabled()

    window.close()


def test_reopening_a_project_with_legacy_history_format_defaults_current_to_latest(qapp, tmp_path):
    """A project's `analysis_results` blob saved before `current_result_id`
    existed (plain list per panel) must still load and default each
    panel's current result to the latest one -- preserving pre-selection
    behavior exactly (see `PanelResultHistory.from_dict`)."""
    import json
    import zipfile

    from gnovi_plot.core.project_io import load_project, save_project

    window = MainWindow()
    gaussian_result, linear_result = _run_two_fits_on_active_panel(window)
    out_path = tmp_path / "legacy.gnovi"
    save_project(window._project, out_path)

    # Downgrade the saved analysis_results blob to the pre-selection shape
    # (a bare list per panel, no current_result_id) to simulate a project
    # saved before this feature existed.
    with zipfile.ZipFile(out_path, "r") as zf:
        manifest = json.loads(zf.read("project.json"))
        other_names = [n for n in zf.namelist() if n != "project.json"]
        other_contents = {n: zf.read(n) for n in other_names}

    workbench = manifest["workbenches"][0]
    legacy_history = {
        panel_id: entry["history"] for panel_id, entry in workbench["analysis_results"].items()
    }
    workbench["analysis_results"] = legacy_history

    with zipfile.ZipFile(out_path, "w") as zf:
        zf.writestr("project.json", json.dumps(manifest))
        for name, content in other_contents.items():
            zf.writestr(name, content)

    reloaded = load_project(out_path)
    window._load_project_into_window(reloaded)

    panel_id = window.figure_model.active_panel.id
    history = window._project.active_workbench.analysis_results
    assert [r.result_id for r in history.all(panel_id)] == [gaussian_result.result_id, linear_result.result_id]
    assert history.current(panel_id).result_id == linear_result.result_id  # defaulted to latest

    window.close()


# --- Residual window: updates in place on reselect (never a second window) ---


def test_selecting_a_different_history_result_updates_an_open_residual_window_in_place(qapp):
    window = MainWindow()
    gaussian_result, linear_result = _run_two_fits_on_active_panel(window)

    window.analysis_panel.history_list.setCurrentRow(0)  # Gaussian is current
    window.analysis_result_view._view_residuals_button.click()
    residual_window = window.analysis_result_view._residual_window
    assert residual_window is not None
    assert residual_window.isVisible()
    assert gaussian_result.residual_window_subtitle() in residual_window.windowTitle()

    window.analysis_panel.history_list.setCurrentRow(1)  # reselect Linear

    assert window.analysis_result_view._residual_window is residual_window  # same instance, not a new one
    assert residual_window.isVisible()
    assert linear_result.residual_window_subtitle() in residual_window.windowTitle()

    window.close()
