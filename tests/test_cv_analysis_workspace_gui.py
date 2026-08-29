"""CV-2A: the Cyclic Voltammetry analysis workspace inside AnalysisPanel /
CVAnalysisSection.

Covers the CV-2 Workflow Audit §37 test plan: tool selector, Panel3D
rejection, source series, sign convention (+ in-place reinterpretation),
cycle source precedence / last-complete default / incomplete + monotonic
handling, sweep selection, Find Peaks -> one history entry, manual add with
the wrong-panel guard, enable/disable + Set Process on the Results-table
selection, the auto couple summary + which-peaks identification, history
load_result, panel/Workbench switch + Focus/Extract, save/reopen, no source
mutation, and Curve-Fitting / XRD regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnovi_plot.core.project import Project
from gnovi_plot.core.project_io import load_project, save_project
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.widgets.analysis_panel import AnalysisPanel
from gnovi_plot.modules.electrochemistry.cv import PROCESS_ANODIC, PROCESS_CATHODIC
from gnovi_plot.modules.electrochemistry.results import CVCycleAnalysisResult
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series import PlotSeries
from tests.data import generate_synthetic_cv as model


def _cv_dataset(name="cv run") -> Dataset:
    return Dataset(name=name, dataframe=model.build_reversible())


def _panel_with_cv_series(figure: GnoviFigure, dataset: Dataset) -> PlotSeries:
    series = PlotSeries.line(dataset, "Potential/V", "Current/A")
    figure.add_series(series)
    return series


def _make_panel(figure=None):
    figure = figure or GnoviFigure()
    dataset = _cv_dataset()
    manager = DatasetManager()
    manager.add(dataset)
    _panel_with_cv_series(figure, dataset)
    panel = AnalysisPanel(figure, manager)
    panel.refresh()
    panel.tool_combo.setCurrentText("Cyclic Voltammetry")
    return panel, panel.cv_section_widget, dataset


# --- tool selector -------------------------------------------------


def test_tool_selector_shows_cv_and_hides_the_others(qapp):
    panel, _cv, _ds = _make_panel()
    assert [panel.tool_combo.itemText(i) for i in range(panel.tool_combo.count())] == [
        "Curve Fitting", "XRD Peak Analysis", "Cyclic Voltammetry",
    ]
    assert panel.cv_section.isVisibleTo(panel)
    assert not panel.fit_section.isVisibleTo(panel)
    assert not panel.xrd_section.isVisibleTo(panel)


def test_panel3d_disables_the_cv_section(qapp):
    figure = GnoviFigure()
    figure.panels[0] = Panel3D(panel_label="3D")
    panel = AnalysisPanel(figure, DatasetManager())
    panel.tool_combo.setCurrentText("Cyclic Voltammetry")
    cv = panel.cv_section_widget
    assert not cv.source_combo.isEnabled()
    assert not cv.find_peaks_button.isEnabled()
    assert cv.status_label.isVisibleTo(cv)
    assert "2D panel" in cv.status_label.text()


def test_no_eligible_series_disables_with_a_message(qapp):
    panel = AnalysisPanel(GnoviFigure(), DatasetManager())
    panel.tool_combo.setCurrentText("Cyclic Voltammetry")
    cv = panel.cv_section_widget
    assert not cv.find_peaks_button.isEnabled()
    assert cv.status_label.isVisibleTo(cv)


# --- source / sign convention -------------------------------------


def test_source_and_columns_shown_read_only(qapp):
    _panel, cv, _ds = _make_panel()
    assert cv.source_combo.count() == 1
    assert "Potential/V" in cv.columns_label.text()
    assert "Current/A" in cv.columns_label.text()


def test_default_sign_convention_is_anodic_positive(qapp):
    _panel, cv, _ds = _make_panel()
    assert cv.sign_combo.currentData() == "anodic_positive"


def test_sign_convention_change_reinterprets_current_result_in_place(qapp):
    _panel, cv, _ds = _make_panel()
    ready = []
    updated = []
    cv.analysis_result_ready.connect(ready.append)
    cv.result_updated.connect(updated.append)
    cv.find_peaks_button.click()
    result = ready[-1]
    processes_before = [p.process for p in result.peaks]

    cv.sign_combo.setCurrentIndex(cv.sign_combo.findData("cathodic_positive"))

    assert len(ready) == 1  # NO new history entry
    assert len(updated) == 1  # an in-place update instead
    assert result.sign_convention == "cathodic_positive"
    for before, after in zip(processes_before, (p.process for p in result.peaks)):
        assert {before, after} == {PROCESS_ANODIC, PROCESS_CATHODIC} or before == after


# --- cycle selection --------------------------------------------


def test_auto_detect_status_and_last_complete_default(qapp):
    _panel, cv, _ds = _make_panel()
    assert "2 cycle(s), 2 complete" in cv.cycle_status_label.text()
    assert cv.cycle_combo.count() == 2
    assert cv.cycle_combo.currentData() == 2  # last complete cycle
    assert "detected" in cv.cycle_confidence_label.text()


def test_metadata_column_source(qapp):
    df = model.build_reversible()
    df["segment"] = (np.arange(len(df)) // (len(df) // 4)).clip(max=3) + 1
    ds = Dataset(name="cv", dataframe=df)
    manager = DatasetManager()
    manager.add(ds)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(ds, "Potential/V", "Current/A"))
    panel = AnalysisPanel(figure, manager)
    panel.refresh()
    panel.tool_combo.setCurrentText("Cyclic Voltammetry")
    cv = panel.cv_section_widget
    cv.cycle_source_combo.setCurrentIndex(cv.cycle_source_combo.findData("metadata"))
    idx = cv.metadata_column_combo.findText("segment")
    assert idx >= 0
    cv.metadata_column_combo.setCurrentIndex(idx)
    assert cv.cycle_combo.count() == 4
    assert "explicit" in cv.cycle_confidence_label.text()


def test_manual_row_ranges_source(qapp):
    _panel, cv, _ds = _make_panel()
    cv.cycle_source_combo.setCurrentIndex(cv.cycle_source_combo.findData("manual"))
    cv.manual_ranges_edit.setText("0-1600, 1600-3200")
    cv.manual_ranges_edit.editingFinished.emit()
    assert cv.cycle_combo.count() == 2
    assert "manual" in cv.cycle_confidence_label.text()


def test_monotonic_data_is_handled_gracefully(qapp):
    df = pd.DataFrame({"Potential/V": np.linspace(-0.2, 0.6, 500),
                       "Current/A": np.linspace(1e-6, 5e-6, 500)})
    ds = Dataset(name="lsv", dataframe=df)
    manager = DatasetManager()
    manager.add(ds)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(ds, "Potential/V", "Current/A"))
    panel = AnalysisPanel(figure, manager)
    panel.refresh()
    panel.tool_combo.setCurrentText("Cyclic Voltammetry")
    cv = panel.cv_section_widget
    assert "Single sweep" in cv.cycle_status_label.text()
    # Find Peaks still works; no couple metrics
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.find_peaks_button.click()
    assert ready[-1].delta_ep_v is None


# --- sweep selection ------------------------------------------


def test_sweep_selection_readout_and_restriction(qapp):
    _panel, cv, _ds = _make_panel()
    assert "Rising:" in cv.sweep_readout_label.text()
    assert "Falling:" in cv.sweep_readout_label.text()
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.sweep_combo.setCurrentIndex(cv.sweep_combo.findData("rising"))
    cv.find_peaks_button.click()
    assert all(p.sweep == "rising" for p in ready[-1].peaks)


# --- Find Peaks + couple ------------------------------------


def test_find_peaks_creates_one_history_entry_and_a_couple(qapp):
    panel, cv, ds = _make_panel()
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.find_peaks_button.click()
    result = ready[-1]
    assert isinstance(result, CVCycleAnalysisResult)
    assert "candidate(s)" in cv.detection_status_label.text()
    assert result.delta_ep_v == pytest.approx(model.DELTA_EP_TRUE, abs=2 * model.STEP)
    assert result.e_half_v == pytest.approx(model.E_HALF_TRUE, abs=model.STEP)
    assert result.couple_anodic_peak_id is not None
    assert result.couple_cathodic_peak_id is not None
    labels = {row[0]: row[1] for row in result.details()}
    assert labels["Couple"].startswith("peak #")
    assert "raw extremum" in labels["|Ipa| / |Ipc|"]


def test_find_peaks_twice_makes_two_history_entries(qapp):
    panel, cv, ds = _make_panel()
    project = Project.new()
    workbench = project.workbenches[0]
    ready = []

    def _record(r):
        workbench.analysis_results.add(r.source_panel_id, r)
        ready.append(r)

    cv.analysis_result_ready.connect(_record)
    cv.find_peaks_button.click()
    cv.find_peaks_button.click()
    assert len(workbench.analysis_results.all(ready[0].source_panel_id)) == 2


# --- manual peak add ------------------------------------------


def test_manual_add_snaps_and_assigns_process(qapp):
    _panel, cv, _ds = _make_panel()
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.find_peaks_button.click()
    updated = []
    cv.result_updated.connect(updated.append)
    n_before = len(ready[-1].peaks)
    # a click near the cathodic wave on the falling sweep
    cv.add_peak_button.setChecked(True)
    cv.add_manual_peak(model.EPC_TRUE, -1.0e-5)
    assert len(ready[-1].peaks) == n_before + 1
    assert ready[-1].peaks[-1].origin == "manual"
    assert not cv.is_manual_peak_mode()  # disarmed after a successful add


def test_manual_peak_mode_disarms_on_context_change(qapp):
    _panel, cv, _ds = _make_panel()
    cv.add_peak_button.setChecked(True)
    assert cv.is_manual_peak_mode()
    cv.sweep_combo.setCurrentIndex(cv.sweep_combo.findData("rising"))
    assert not cv.is_manual_peak_mode()


# --- Results-table-driven edits ---------------------------


def test_enable_disable_and_set_process_act_on_selection(qapp):
    panel, cv, _ds = _make_panel()
    ready = []
    updated = []
    cv.analysis_result_ready.connect(ready.append)
    cv.result_updated.connect(updated.append)
    cv.find_peaks_button.click()
    result = ready[-1]

    panel.cv_set_selected_peak_rows([0])
    cv.toggle_enabled_button.click()
    assert result.peaks[0].enabled is False
    assert len(updated) == 1

    panel.cv_set_selected_peak_rows([0])
    cv._on_set_process(PROCESS_CATHODIC)
    assert result.peaks[0].process == PROCESS_CATHODIC
    assert len(ready) == 1  # still no new history entry


def test_remove_selected(qapp):
    _panel_, cv, _ds = _make_panel()
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.find_peaks_button.click()
    n = len(ready[-1].peaks)
    _panel_.cv_set_selected_peak_rows([0])
    cv.remove_peak_button.click()
    assert len(ready[-1].peaks) == n - 1


# --- history load_result ---------------------------------


def test_load_result_restores_settings_without_rerunning(qapp):
    _panel, cv, _ds = _make_panel()
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.sweep_combo.setCurrentIndex(cv.sweep_combo.findData("rising"))
    cv.find_peaks_button.click()
    result = ready[-1]

    # simulate a different UI state, then reload the result
    cv.sweep_combo.setCurrentIndex(cv.sweep_combo.findData("both"))
    cv.load_result(result)
    assert cv.sweep_combo.currentData() == "rising"
    assert cv.current_result() is result
    assert len(ready) == 1  # load_result never re-runs detection


# --- overlay payload ------------------------------------


def test_overlay_payload_gated_on_panel_and_has_expected_keys(qapp):
    _panel, cv, _ds = _make_panel()
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.find_peaks_button.click()
    payload = cv.overlay_payload()
    assert payload is not None
    assert "cycle_rising_xy" in payload and "cycle_falling_xy" in payload
    assert "candidate_xy" in payload
    # a result on another panel -> no peak markers
    ready[-1].source_panel_id = "some-other-panel"
    payload2 = cv.overlay_payload()
    assert "candidate_xy" not in (payload2 or {})


# --- MainWindow integration: dirty, no source mutation, save/reopen ----


def test_mainwindow_find_peaks_dirties_and_records_history(qapp):
    window = MainWindow()
    ds = _cv_dataset()
    window.dataset_manager.add(ds)
    window.figure_model.active_panel.add_series(PlotSeries.line(ds, "Potential/V", "Current/A"))
    window.analysis_panel.refresh()
    window.analysis_panel.tool_combo.setCurrentText("Cyclic Voltammetry")
    window._dirty = False
    original = ds.dataframe.copy(deep=True)

    window.analysis_panel.cv_section_widget.find_peaks_button.click()

    panel_id = window.figure_model.active_panel.id
    history = window._project.active_workbench.analysis_results
    assert len(history.all(panel_id)) == 1
    assert isinstance(history.current(panel_id), CVCycleAnalysisResult)
    assert window._dirty is True
    pd.testing.assert_frame_equal(ds.dataframe, original)  # source never mutated


def test_mainwindow_cv_result_survives_save_reopen(qapp, tmp_path):
    window = MainWindow()
    ds = _cv_dataset()
    window.dataset_manager.add(ds)
    window.figure_model.active_panel.add_series(PlotSeries.line(ds, "Potential/V", "Current/A"))
    window.analysis_panel.refresh()
    window.analysis_panel.tool_combo.setCurrentText("Cyclic Voltammetry")
    cv = window.analysis_panel.cv_section_widget
    cv.scan_rate_spin.setValue(100.0)
    cv.find_peaks_button.click()
    panel_id = window.figure_model.active_panel.id
    result = window._project.active_workbench.analysis_results.current(panel_id)

    reloaded = load_project(save_project(window._project, tmp_path / "cv.gnovi"))
    rr = reloaded.workbenches[0].analysis_results.current(panel_id)
    assert isinstance(rr, CVCycleAnalysisResult)
    assert rr.result_id == result.result_id
    assert rr.sign_convention == "anodic_positive"
    assert rr.delta_ep_v == pytest.approx(result.delta_ep_v)
    assert rr.couple_anodic_peak_id == result.couple_anodic_peak_id
    assert rr.parameters.get("scan_rate_v_per_s") == pytest.approx(0.1)


def test_mainwindow_manual_peak_click_wrong_panel_is_ignored(qapp):
    window = MainWindow()
    ds = _cv_dataset()
    window.dataset_manager.add(ds)
    window.figure_model.active_panel.add_series(PlotSeries.line(ds, "Potential/V", "Current/A"))
    window.analysis_panel.refresh()
    window.analysis_panel.tool_combo.setCurrentText("Cyclic Voltammetry")
    cv = window.analysis_panel.cv_section_widget
    cv.find_peaks_button.click()
    n_before = len(cv.current_result().peaks)

    cv.add_peak_button.setChecked(True)

    class _Evt:
        button = 1
        dblclick = False
        inaxes = object()  # not one of the canvas' Axes
        xdata = 0.25
        ydata = 1e-5

    window._on_canvas_click(_Evt())
    assert len(cv.current_result().peaks) == n_before  # nothing added


# --- regression: Curve Fitting / XRD still work -------------------


def test_curve_fitting_still_default_and_functional(qapp):
    panel, _cv, _ds = _make_panel()
    panel.tool_combo.setCurrentText("Curve Fitting")
    assert panel.fit_section.isVisibleTo(panel)
    assert not panel.cv_section.isVisibleTo(panel)
    panel.tool_combo.setCurrentText("XRD Peak Analysis")
    assert panel.xrd_section.isVisibleTo(panel)
    assert not panel.cv_section.isVisibleTo(panel)


# ======================================================================
# Code-review regression tests: H2, H3, M1, M2, M3
# ======================================================================


def _two_series_panel(qapp):
    """One 2D panel with two eligible CV line series ('A' and a 3x-scaled
    'B') -- so the source combo actually has something to switch between."""
    figure = GnoviFigure()
    df_b = model.build_reversible()
    df_b["Current/A"] = df_b["Current/A"] * 3.0
    ds_a = Dataset(name="A", dataframe=model.build_reversible())
    ds_b = Dataset(name="B", dataframe=df_b)
    manager = DatasetManager()
    manager.add(ds_a)
    manager.add(ds_b)
    figure.add_series(PlotSeries.line(ds_a, "Potential/V", "Current/A", label="A"))
    figure.add_series(PlotSeries.line(ds_b, "Potential/V", "Current/A", label="B"))
    panel = AnalysisPanel(figure, manager)
    panel.refresh()
    panel.tool_combo.setCurrentText("Cyclic Voltammetry")
    return panel, panel.cv_section_widget


def test_h2_source_series_switch_detaches_result_and_clears_overlay(qapp):
    panel, cv = _two_series_panel(qapp)
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.find_peaks_button.click()
    result_a = ready[-1]
    assert cv.overlay_payload().get("candidate_xy") is not None

    cv.source_combo.setCurrentIndex(1)  # user picks series B

    assert cv.current_result() is None  # working pointer detached...
    assert not cv.is_manual_peak_mode()
    payload = cv.overlay_payload() or {}
    assert "candidate_xy" not in payload  # ...so no stale markers from A over B
    assert len(ready) == 1  # the History entry for A was NOT touched / duplicated


def test_h2_history_entry_for_the_old_series_stays_selectable(qapp):
    panel, cv = _two_series_panel(qapp)
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.find_peaks_button.click()
    result_a = ready[-1]
    cv.source_combo.setCurrentIndex(1)

    cv.load_result(result_a)  # re-selected from the History list

    assert cv.current_result() is result_a
    assert cv.source_combo.currentData() == result_a.source_series_id  # source restored to A
    assert len(ready) == 1  # still no re-run


def test_h2_overlay_gated_on_source_series_id(qapp):
    panel, cv = _two_series_panel(qapp)
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.find_peaks_button.click()
    # forge a result whose series id matches nothing eligible -> no markers
    ready[-1].source_series_id = "not-a-real-series"
    payload = cv.overlay_payload() or {}
    assert "candidate_xy" not in payload
    assert "cycle_rising_xy" in payload  # cycle tint for the *current* series still shows


def test_m1_manual_click_far_outside_the_cycle_is_rejected(qapp):
    _panel, cv, _ds = _make_panel()
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.find_peaks_button.click()
    n_before = len(ready[-1].peaks)
    messages = []
    cv.status_message.connect(messages.append)

    cv.add_peak_button.setChecked(True)
    cv.add_manual_peak(5.0, 999.0)  # E = 5 V, I = 999 A -- nowhere near the curve

    assert len(cv.current_result().peaks) == n_before  # nothing added
    assert messages and "curve" in messages[-1].lower()


def test_m1_manual_click_with_the_wrong_current_at_a_valid_potential_is_rejected(qapp):
    _panel, cv, _ds = _make_panel()
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.find_peaks_button.click()
    n_before = len(ready[-1].peaks)

    cv.add_peak_button.setChecked(True)
    cv.add_manual_peak(model.EPA_TRUE, -5.0e-4)  # right potential, wildly wrong current

    assert len(cv.current_result().peaks) == n_before


def test_m1_manual_click_on_the_curve_snaps_to_the_right_sweep(qapp):
    """The same potential appears on both sweeps; the click's current must
    decide which sweep, not 'whichever sweep has this potential'."""
    _panel, cv, _ds = _make_panel()
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.find_peaks_button.click()

    cv.add_peak_button.setChecked(True)
    cv.add_manual_peak(model.EPC_TRUE, -1.0e-5)  # negative current -> the falling (cathodic) wave
    added = cv.current_result().peaks[-1]
    assert added.origin == "manual"
    assert added.sweep == "falling"
    assert added.process == PROCESS_CATHODIC


def test_m2_cv_section_minimum_width_within_the_analysis_drawer_budget(qapp):
    from gnovi_plot.gui.widgets.cv_analysis_section import CVAnalysisSection
    from gnovi_plot.gui.widgets.xrd_analysis_section import XRDAnalysisSection

    cv = CVAnalysisSection(GnoviFigure(), DatasetManager())
    xrd = XRDAnalysisSection(GnoviFigure(), DatasetManager())
    # CV must not need a wider drawer than Curve Fitting / XRD already use.
    assert cv.minimumSizeHint().width() <= xrd.minimumSizeHint().width()


def test_m3_metadata_cycle_source_handles_nan_rows_in_potential_current(qapp):
    """The metadata cycle column is grouped in the NaN-cleaned row space
    (aligned with the arrays sweep segmentation runs on), so a NaN row in
    the potential/current columns does not shift the cycle boundaries."""
    df = model.build_reversible()
    df["segment"] = (np.arange(len(df)) // (len(df) // 4)).clip(max=3) + 1
    df.loc[10, "Current/A"] = np.nan  # a hole in the data
    df.loc[2500, "Potential/V"] = np.nan
    ds = Dataset(name="cv", dataframe=df)
    manager = DatasetManager()
    manager.add(ds)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(ds, "Potential/V", "Current/A"))
    panel = AnalysisPanel(figure, manager)
    panel.refresh()
    panel.tool_combo.setCurrentText("Cyclic Voltammetry")
    cv = panel.cv_section_widget
    cv.cycle_source_combo.setCurrentIndex(cv.cycle_source_combo.findData("metadata"))
    cv.metadata_column_combo.setCurrentIndex(cv.metadata_column_combo.findText("segment"))

    assert cv.cycle_combo.count() == 4
    # Find Peaks must run on the cleaned arrays without an IndexError
    ready = []
    cv.analysis_result_ready.connect(ready.append)
    cv.find_peaks_button.click()
    assert isinstance(ready[-1], CVCycleAnalysisResult)


def test_h3_shared_overlay_boundary_clears_reference_cursor_and_both_overlays(qapp):
    """`PlotCanvas.clear_gui_only_overlays` is the single boundary every
    save/export path calls; the toolbar Save and the Export dialog both
    route through it."""
    from gnovi_plot.gui import main_window
    from gnovi_plot.gui.dialogs import export_figure_dialog
    import inspect

    canvas_src = inspect.getsource(main_window._CursorSafeNavigationToolbar.save_figure)
    assert "clear_gui_only_overlays" in canvas_src

    dialog_src = inspect.getsource(export_figure_dialog.ExportFigureDialog._hide_gui_only_overlays)
    assert "clear_gui_only_overlays" in dialog_src


def test_h3_clear_gui_only_overlays_removes_cv_artists(qapp):
    window = MainWindow()
    ds = _cv_dataset()
    window.dataset_manager.add(ds)
    window.figure_model.active_panel.add_series(PlotSeries.line(ds, "Potential/V", "Current/A"))
    window.analysis_panel.refresh()
    window.analysis_panel.tool_combo.setCurrentText("Cyclic Voltammetry")
    window._rerender()
    window.analysis_panel.cv_section_widget.find_peaks_button.click()
    window._refresh_cv_overlay()
    assert window.plot_canvas._cv_overlay_artists  # overlay is drawn

    window.plot_canvas.clear_gui_only_overlays()

    assert not window.plot_canvas._cv_overlay_artists
    assert not window.plot_canvas._analysis_overlay_artists
    assert not window.plot_canvas._cursor_artists
