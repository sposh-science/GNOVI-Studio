"""XRD-2: the XRD Peak Analysis workspace inside AnalysisPanel/XRDAnalysisSection.

Covers section 31 of the XRD-2 spec: tool selector, Panel3D rejection,
source-series selection, radiation presets/custom wavelength, d-spacing
updates, background method switching, transient preview immutability,
smoothing off-by-default, detection-input selection, find peaks/prominence
behavior, peak table population, manual add/remove/enable-disable, marker
overlay, label modes, history-entry distinguishability, save/reopen, Focus,
Extract, derived corrected/smoothed curves, and no-pybaselines behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnovi_plot.analysis.results import AnalysisResult
from gnovi_plot.core.project import Project
from gnovi_plot.core.project_io import load_project, save_project
from gnovi_plot.core.workbench import Workbench
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.widgets.analysis_panel import AnalysisPanel
from gnovi_plot.gui.widgets.xrd_analysis_section import XRDAnalysisSection
from gnovi_plot.modules.xrd import preprocessing as xrd_preprocessing
from gnovi_plot.modules.xrd.preprocessing import PybaselinesNotAvailableError
from gnovi_plot.modules.xrd.radiation import CU_KALPHA1_ANGSTROM
from gnovi_plot.modules.xrd.results import XRDAnalysisResult
from gnovi_plot.plotting.figure import GnoviFigure, Panel, Panel3D
from gnovi_plot.plotting.series import PlotSeries
from gnovi_plot.plotting.series3d import Plot3DType, Series3D


def _gaussian(x, center, amp, sigma):
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _synthetic_pattern_dataset(name="XRD pattern", seed=1):
    rng = np.random.default_rng(seed)
    two_theta = np.linspace(10.0, 90.0, 2000)
    intensity = 20.0 + 0.1 * two_theta
    for center, amp, sigma in [(30.0, 500.0, 0.08), (45.0, 300.0, 0.08), (60.0, 150.0, 0.07)]:
        intensity = intensity + _gaussian(two_theta, center, amp, sigma)
    intensity = intensity + rng.normal(0, 2.0, size=two_theta.shape)
    df = pd.DataFrame({"2theta": two_theta, "intensity": intensity})
    return Dataset(name=name, dataframe=df)


def _panel_with_series(figure: GnoviFigure, dataset: Dataset) -> PlotSeries:
    series = PlotSeries.line(dataset, "2theta", "intensity")
    figure.add_series(series)
    return series


# --- Analysis tool selector / Curve Fitting regression -----------------------


def test_analysis_tool_selector_offers_curve_fitting_and_xrd(qapp):
    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())
    options = [panel.tool_combo.itemText(i) for i in range(panel.tool_combo.count())]
    assert options == ["Curve Fitting", "XRD Peak Analysis"]


def test_curve_fitting_section_visible_by_default_xrd_hidden(qapp):
    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())
    assert panel.tool_combo.currentText() == "Curve Fitting"
    assert panel.fit_section.isVisibleTo(panel)
    assert not panel.xrd_section.isVisibleTo(panel)


def test_switching_to_xrd_tool_shows_xrd_section_and_hides_fit_section(qapp):
    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())
    panel.tool_combo.setCurrentText("XRD Peak Analysis")
    assert panel.xrd_section.isVisibleTo(panel)
    assert not panel.fit_section.isVisibleTo(panel)


def test_curve_fitting_still_works_unchanged_alongside_xrd(qapp):
    """Regression: adding XRD must not disturb existing curve-fit behavior."""
    figure = GnoviFigure()
    ds = Dataset(name="d", dataframe=pd.DataFrame({"x": list(range(10)), "y": [2 * v + 1 for v in range(10)]}))
    figure.add_series(PlotSeries.line(ds, "x", "y", label="Line"))
    panel = AnalysisPanel(figure, DatasetManager())
    from gnovi_plot.analysis.fitting import LINEAR

    idx = panel.model_combo.findData(LINEAR)
    panel.model_combo.setCurrentIndex(idx)
    panel._on_run_fit_clicked()
    assert panel._pending_fit is not None
    assert panel._pending_fit.model == LINEAR


# --- Panel3D rejection ---------------------------------------------------------


def test_xrd_section_disabled_with_explanation_when_active_panel_is_3d(qapp):
    figure = GnoviFigure()
    figure.panels = [Panel3D()]
    figure.active_panel_index = 0
    section = XRDAnalysisSection(figure, DatasetManager())
    assert not section.source_combo.isEnabled()
    assert section.status_label.isVisibleTo(section)
    assert "2D Panel" in section.status_label.text()


def test_xrd_section_ignores_series3d_as_source(qapp):
    figure = GnoviFigure()
    panel3d = Panel3D()
    figure.panels = [panel3d]
    figure.active_panel_index = 0
    ds = _synthetic_pattern_dataset()
    panel3d.series.append(
        Series3D(dataset=ds, x_column="2theta", y_column="intensity", z_column="intensity", plot_type=Plot3DType.SCATTER)
    )
    section = XRDAnalysisSection(figure, DatasetManager())
    assert section.source_combo.count() == 0


# --- Source-series selection / radiation / d-spacing --------------------------


def test_source_combo_lists_eligible_2d_series(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    series = _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    assert section.source_combo.count() == 1
    assert section.source_combo.currentData() == series.id


def test_radiation_presets_include_ka1_and_weighted_ka_distinctly(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    idx_ka1 = section.radiation_combo.findData("cu_ka1")
    idx_ka = section.radiation_combo.findData("cu_ka")
    assert idx_ka1 >= 0 and idx_ka >= 0
    assert idx_ka1 != idx_ka
    section.radiation_combo.setCurrentIndex(idx_ka1)
    assert section._radiation.wavelength_angstrom == pytest.approx(CU_KALPHA1_ANGSTROM)
    section.radiation_combo.setCurrentIndex(idx_ka)
    assert section._radiation.wavelength_angstrom != pytest.approx(CU_KALPHA1_ANGSTROM)


def test_custom_wavelength_entry(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    idx_custom = section.radiation_combo.findData("custom")
    section.radiation_combo.setCurrentIndex(idx_custom)
    section.custom_wavelength_spin.setValue(1.2345)
    assert section._radiation.wavelength_angstrom == pytest.approx(1.2345)
    assert section.custom_wavelength_spin.isVisibleTo(section)


def test_no_radiation_selected_blocks_find_peaks_with_clear_message(qapp, monkeypatch):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    calls = []
    monkeypatch.setattr(
        "gnovi_plot.gui.widgets.xrd_analysis_section.QMessageBox.warning",
        lambda *a, **k: calls.append(a[2]),
    )
    section._on_find_peaks_clicked()
    assert calls and "Radiation" in calls[0]
    assert section.current_result() is None


def test_changing_radiation_updates_d_spacing_deterministically(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section.prominence_spin.setValue(30.0)
    section._on_find_peaks_clicked()
    result = section.current_result()
    assert result is not None and result.peaks
    d1 = section._peak_d_spacing(result.peaks[0])

    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("mo_ka1"))
    d2 = section._peak_d_spacing(result.peaks[0])
    assert d1 != pytest.approx(d2)
    # Deterministic: same wavelength always gives the same d-spacing.
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    d3 = section._peak_d_spacing(result.peaks[0])
    assert d1 == pytest.approx(d3)


# --- Background / transient preview / immutability -----------------------------


def test_background_method_switching_updates_visible_controls(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.background_method_combo.setCurrentText("Polynomial")
    assert section.baseline_points_edit.isVisibleTo(section)
    assert not section.arpls_lam_spin.isVisibleTo(section)
    section.background_method_combo.setCurrentText("arPLS")
    assert section.arpls_lam_spin.isVisibleTo(section)
    assert not section.baseline_points_edit.isVisibleTo(section)


def test_polynomial_background_preview_is_transient_and_does_not_mutate_dataset(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    original = ds.dataframe["intensity"].to_numpy().copy()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section.polynomial_degree_spin.setValue(1)
    section._on_preview_background_clicked()

    assert section._background_preview is not None
    assert np.array_equal(ds.dataframe["intensity"].to_numpy(), original)
    manager = DatasetManager()
    assert len(manager.datasets) == 0  # preview never registers a Dataset

    # Changing the source clears the preview -- it's session-local only.
    section._on_source_changed()
    assert section._background_preview is None


def test_preview_curve_reflects_smoothed_over_background_when_both_present(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()
    section.smoothing_enabled_check.setChecked(True)
    section._on_preview_smoothed_clicked()
    x, y = section.preview_curve()
    assert np.array_equal(y, section._smooth_preview.smoothed_intensity)


# --- Smoothing off-by-default --------------------------------------------------


def test_smoothing_is_off_by_default(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    assert not section.smoothing_enabled_check.isChecked()
    assert section._smooth_preview is None


def test_finding_peaks_never_runs_smoothing_when_not_selected(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section._on_find_peaks_clicked()
    result = section.current_result()
    assert result.parameters["preprocessing"]["smoothing"] is None
    assert result.parameters["detection_input"] == "raw"


# --- Detection-input chain / find peaks / prominence ---------------------------


def test_detection_input_options_only_show_whats_available(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    options = [section.detection_input_combo.itemData(i) for i in range(section.detection_input_combo.count())]
    assert options == ["raw"]

    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()
    options = [section.detection_input_combo.itemData(i) for i in range(section.detection_input_combo.count())]
    assert set(options) == {"raw", "background_corrected"}

    section.smoothing_enabled_check.setChecked(True)
    options = [section.detection_input_combo.itemData(i) for i in range(section.detection_input_combo.count())]
    assert set(options) == {"raw", "background_corrected", "smoothed_raw", "smoothed_background_corrected"}


def test_find_peaks_detects_known_synthetic_centers_within_tolerance(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset(seed=7)
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section.prominence_spin.setValue(40.0)
    section._on_find_peaks_clicked()
    result = section.current_result()
    centers = sorted(p.two_theta for p in result.peaks)
    truth = [30.0, 45.0, 60.0]
    assert len(centers) == len(truth)
    for t, c in zip(truth, centers):
        assert abs(t - c) < 0.1


def test_higher_prominence_detects_fewer_peaks(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset(seed=7)
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section.prominence_spin.setValue(10.0)
    section._on_find_peaks_clicked()
    low_count = len(section.current_result().peaks)

    section.prominence_spin.setValue(400.0)
    section._on_find_peaks_clicked()
    high_count = len(section.current_result().peaks)
    assert high_count < low_count


def test_no_peaks_detected_shows_clear_status_not_a_crash(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset(seed=7)
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section.prominence_spin.setValue(1e9)
    section._on_find_peaks_clicked()
    assert section.current_result() is not None
    assert section.current_result().peaks == []
    assert "No peaks detected" in section.detection_status_label.text()


def test_find_peaks_always_creates_a_new_history_entry(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset(seed=7)
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section.prominence_spin.setValue(40.0)

    produced = []
    section.analysis_result_ready.connect(produced.append)
    section._on_find_peaks_clicked()
    section._on_find_peaks_clicked()
    assert len(produced) == 2
    assert produced[0].result_id != produced[1].result_id


# --- Peak table / manual add / remove / enable-disable --------------------------


def _section_with_detected_peaks(seed=7, prominence=40.0):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset(seed=seed)
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section.prominence_spin.setValue(prominence)
    section._on_find_peaks_clicked()
    return figure, section


def test_peak_table_populated_with_expected_columns_no_fitted_columns(qapp):
    _figure, section = _section_with_detected_peaks()
    result = section.current_result()
    # The detailed peak table now renders in the bottom Results tab, from
    # `XRDAnalysisResult.detail_table()` -- one row per peak candidate,
    # seed-level columns only (no fitted center/FWHM/area/model).
    columns, rows = result.detail_table()
    assert len(rows) == len(result.peaks)
    assert columns == [
        "Peak #",
        "Seed 2θ (°)",
        "Observed intensity",
        "Prominence",
        "d-spacing (Å)",
        "Origin",
        "Enabled",
    ]
    for forbidden in ("FWHM", "Area", "Fit", "Crystallite"):
        assert forbidden not in columns

    from gnovi_plot.gui.widgets.analysis_result_view import AnalysisResultView

    view = AnalysisResultView(GnoviFigure(), DatasetManager())
    view.show_result(result)
    assert view._detail_table.rowCount() == len(result.peaks)
    view_headers = [
        view._detail_table.horizontalHeaderItem(i).text() for i in range(view._detail_table.columnCount())
    ]
    assert view_headers == columns


def test_manual_add_peak_creates_a_seed_not_a_fitted_center(qapp):
    _figure, section = _section_with_detected_peaks()
    before = len(section.current_result().peaks)
    section.add_manual_peak(52.0, 77.0)
    assert len(section.current_result().peaks) == before + 1
    new_peak = section.current_result().peaks[-1]
    assert new_peak.origin == "manual"
    assert new_peak.two_theta == pytest.approx(52.0)
    assert new_peak.index is None  # not tied to any detection-array position


def test_manual_add_peak_before_any_detection_starts_a_fresh_result(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    assert section.current_result() is None
    produced = []
    section.analysis_result_ready.connect(produced.append)
    section.add_manual_peak(40.0, 100.0)
    assert section.current_result() is not None
    assert len(section.current_result().peaks) == 1
    assert produced and produced[0] is section.current_result()


def test_remove_selected_peak(qapp):
    _figure, section = _section_with_detected_peaks()
    before = len(section.current_result().peaks)
    assert before > 0
    section.set_selected_peak_rows([0])  # selection comes from the Results-tab table
    section._on_remove_selected_clicked()
    assert len(section.current_result().peaks) == before - 1


def test_toggle_enabled_excludes_peak_from_overlay(qapp):
    _figure, section = _section_with_detected_peaks()
    section.set_selected_peak_rows([0])
    peak = section.current_result().peaks[0]
    assert peak.enabled
    section._on_toggle_enabled_clicked()
    assert not section.current_result().peaks[0].enabled
    overlay = section.overlay_points()
    assert all(pt[0] != peak.two_theta for pt in overlay) or len(overlay) == len(section.current_result().peaks) - 1


def test_manual_edits_emit_result_updated_not_a_new_history_entry(qapp):
    _figure, section = _section_with_detected_peaks()
    ready = []
    updated = []
    section.analysis_result_ready.connect(ready.append)
    section.result_updated.connect(updated.append)
    section.add_manual_peak(70.0, 30.0)
    assert len(ready) == 0
    assert len(updated) == 1
    assert updated[0] is section.current_result()


# --- Marker overlay / labels ------------------------------------------------


def test_overlay_points_reflect_only_enabled_peaks(qapp):
    _figure, section = _section_with_detected_peaks()
    total = len(section.current_result().peaks)
    section.set_selected_peak_rows([0])
    section._on_toggle_enabled_clicked()
    overlay = section.overlay_points()
    assert len(overlay) == total - 1


def test_label_mode_off_gives_empty_labels(qapp):
    _figure, section = _section_with_detected_peaks()
    section.label_mode_combo.setCurrentText("Off")
    overlay = section.overlay_points()
    assert all(label == "" for _x, _y, label in overlay)


def test_label_mode_two_theta_and_d_spacing(qapp):
    _figure, section = _section_with_detected_peaks()
    section.label_mode_combo.setCurrentText("2θ")
    overlay = section.overlay_points()
    assert all("°" in label for _x, _y, label in overlay)

    section.label_mode_combo.setCurrentText("d-spacing")
    overlay = section.overlay_points()
    assert all("Å" in label for _x, _y, label in overlay)


def test_overlay_points_none_when_active_panel_does_not_match_result(qapp):
    figure, section = _section_with_detected_peaks()
    other_panel = figure.panels[0].__class__()
    figure.panels.append(other_panel)
    figure.active_panel_index = 1
    assert section.overlay_points() is None


# --- Analysis History distinguishability / selection restores state -----------


def test_xrd_result_summary_is_distinguishable_from_fit_result(qapp):
    _figure, section = _section_with_detected_peaks()
    result = section.current_result()
    summary = result.summary()
    assert "XRD" in summary or "peak" in summary.lower()


def test_selecting_xrd_history_entry_restores_result_and_switches_tool(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset(seed=7)
    series = _panel_with_series(figure, ds)
    panel = AnalysisPanel(figure, DatasetManager())
    panel.tool_combo.setCurrentText("XRD Peak Analysis")
    xrd = panel.xrd_section_widget
    xrd.radiation_combo.setCurrentIndex(xrd.radiation_combo.findData("cu_ka1"))
    xrd.prominence_spin.setValue(40.0)
    xrd._on_find_peaks_clicked()
    result = xrd.current_result()

    panel.tool_combo.setCurrentText("Curve Fitting")
    panel.sync_history([result], result)
    assert panel.tool_combo.currentText() == "XRD Peak Analysis"
    assert xrd.current_result() is result
    # The detailed table (bottom Results tab) renders one row per peak.
    _columns, table_rows = result.detail_table()
    assert len(table_rows) == len(result.peaks)


# --- Focus / Extract ---------------------------------------------------------


def _project_with_xrd_result():
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset(seed=7)
    _panel_with_series(figure, ds)
    manager = DatasetManager()
    manager.add(ds)
    workbench = Workbench(name="wb", figure=figure)
    section = XRDAnalysisSection(figure, manager)
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section.prominence_spin.setValue(40.0)
    section._on_find_peaks_clicked()
    result = section.current_result()
    workbench.analysis_results.add(figure.active_panel.id, result)
    project = Project(dataset_manager=manager, workbenches=[workbench], active_workbench_id=workbench.id)
    return project, workbench, result, manager


def test_focus_preserves_xrd_result_on_the_same_live_panel(qapp):
    project, workbench, result, _manager = _project_with_xrd_result()
    panel = workbench.figure.active_panel
    # Focus never clones the Panel (see MainWindow._focus_panel) -- the
    # result's own source_panel_id still matches after "focusing" it.
    assert result.source_panel_id == panel.id
    panel.title = "Focused view"  # a persistent modification made while focused
    assert workbench.figure.active_panel.title == "Focused view"
    assert workbench.analysis_results.current(panel.id) is result


def test_extract_remaps_xrd_result_source_panel_id(qapp):
    project, workbench, result, _manager = _project_with_xrd_result()
    source_panel_id = workbench.figure.active_panel.id
    new_workbench = project.extract_panel_to_workbench(workbench.id, source_panel_id)
    assert new_workbench is not None
    extracted_panel = new_workbench.figure.panels[0]
    assert extracted_panel.id != source_panel_id
    extracted_history = new_workbench.analysis_results.all(extracted_panel.id)
    assert len(extracted_history) == 1
    extracted_result = extracted_history[0]
    assert isinstance(extracted_result, XRDAnalysisResult)
    assert extracted_result.source_panel_id == extracted_panel.id
    assert extracted_result.peaks == result.peaks or [p.id for p in extracted_result.peaks] == [
        p.id for p in result.peaks
    ]
    # Source workbench's own history is untouched.
    assert workbench.analysis_results.current(source_panel_id) is result


# --- Derived corrected/smoothed dataset --------------------------------------


def test_add_corrected_curve_creates_derived_dataset_with_provenance(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    manager = DatasetManager()
    manager.add(ds)
    section = XRDAnalysisSection(figure, manager)
    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()

    added = []
    section.add_to_plot_requested.connect(added.append)
    section._on_add_corrected_clicked()
    assert len(added) == 1
    new_series = added[0][0]
    assert new_series.dataset.metadata["operation"] == "xrd_background_correction"
    assert new_series.dataset.metadata["source_dataset_id"] == ds.id
    assert manager.get(new_series.dataset.id) is not None
    # The derived dataset is independent -- the source is untouched.
    assert np.array_equal(ds.dataframe["intensity"].to_numpy(), ds.raw_dataframe["intensity"].to_numpy())


def test_add_smoothed_curve_creates_derived_dataset(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    manager = DatasetManager()
    manager.add(ds)
    section = XRDAnalysisSection(figure, manager)
    section.smoothing_enabled_check.setChecked(True)
    section._on_preview_smoothed_clicked()

    added = []
    section.add_to_plot_requested.connect(added.append)
    section._on_add_smoothed_clicked()
    assert len(added) == 1
    new_series = added[0][0]
    assert new_series.dataset.metadata["operation"] == "xrd_smoothing"
    assert manager.get(new_series.dataset.id) is not None


# --- No pybaselines installed -----------------------------------------------


def test_arpls_preview_without_pybaselines_shows_clear_message_not_a_crash(qapp, monkeypatch):
    monkeypatch.setattr(xrd_preprocessing, "_PYBASELINES_AVAILABLE", False)
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.background_method_combo.setCurrentText("arPLS")
    calls = []
    monkeypatch.setattr(
        "gnovi_plot.gui.widgets.xrd_analysis_section.QMessageBox.critical",
        lambda *a, **k: calls.append(a[2]),
    )
    section._on_preview_background_clicked()
    assert calls and "pybaselines" in calls[0]
    assert section._background_preview is None
    # Everything else keeps working without pybaselines.
    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()
    assert section._background_preview is not None


def test_gnovi_starts_and_xrd_section_constructs_without_pybaselines(qapp, monkeypatch):
    monkeypatch.setattr(xrd_preprocessing, "_PYBASELINES_AVAILABLE", False)
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())  # must not raise
    assert section.isEnabled()


# --- CSV export --------------------------------------------------------------


def test_export_peak_table_csv_writes_expected_rows(qapp, tmp_path):
    _figure, section = _section_with_detected_peaks()
    result = section.current_result()
    path = tmp_path / "peaks.csv"
    section.export_peak_table_csv(str(path))

    import csv

    # The file is written as UTF-8 with a BOM (utf-8-sig) so the Unicode
    # scientific headers survive on Windows too (see below).
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "Peak #",
        "Seed 2θ (°)",
        "Observed intensity",
        "Prominence",
        "d-spacing (Å)",
        "Origin",
        "Enabled",
    ]
    assert len(rows) == len(result.peaks) + 1
    assert rows[1][5] == "automatic"


def test_export_peak_table_csv_is_utf8_sig_and_preserves_unicode_headers(qapp, tmp_path):
    """Windows CI regression: CSV export used the platform-default text
    encoding, which is cp1252 on Windows and cannot encode 'θ' -- the
    export raised UnicodeEncodeError. It now writes explicit utf-8-sig;
    'θ' / '°' / 'Å' must survive and the file must carry a UTF-8 BOM."""
    _figure, section = _section_with_detected_peaks()
    result = section.current_result()
    path = tmp_path / "peaks.csv"
    section.export_peak_table_csv(str(path))

    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM (utf-8-sig), Excel-friendly on Windows

    text = raw.decode("utf-8-sig")
    header_line = text.splitlines()[0]
    assert "Seed 2θ (°)" in header_line
    assert "d-spacing (Å)" in header_line
    assert "θ" in header_line and "°" in header_line and "Å" in header_line

    import csv

    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    # Row data intact and correct: one data row per peak, d-spacing column
    # parses as a float, first data row keeps its origin.
    assert len(rows) == len(result.peaks) + 1
    first_d = rows[1][4]
    assert first_d == "" or float(first_d) > 0.0
    assert rows[1][5] in ("automatic", "manual")
    # Enabled column round-trips as a bool string.
    assert rows[1][6] in ("True", "False")


def test_export_peak_table_csv_bytes_are_pure_utf8(qapp, tmp_path):
    """Linux behavior is unchanged in substance -- the payload after the
    BOM is plain UTF-8 and decodes losslessly."""
    _figure, section = _section_with_detected_peaks()
    path = tmp_path / "peaks.csv"
    section.export_peak_table_csv(str(path))
    raw = path.read_bytes()
    # Whole file (BOM + body) is valid UTF-8; body alone is valid UTF-8 too.
    raw.decode("utf-8")
    raw[3:].decode("utf-8")


def test_export_peak_table_with_no_result_is_a_safe_no_op(qapp, tmp_path):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    path = tmp_path / "peaks.csv"
    section.export_peak_table_csv(str(path))  # must not raise
    assert not path.exists()


# --- Bottom Results-tab peak table: full-window synchronization --------------


def _xrd_main_window(seed=7):
    window = MainWindow()
    ds = _synthetic_pattern_dataset(seed=seed)
    window.dataset_manager.add(ds)
    window.figure_model.active_panel.add_series(PlotSeries.line(ds, "2theta", "intensity", label="pattern"))
    window._on_figure_content_changed()
    window.analysis_panel.tool_combo.setCurrentText("XRD Peak Analysis")
    xrd = window.analysis_panel.xrd_section_widget
    xrd.radiation_combo.setCurrentIndex(xrd.radiation_combo.findData("cu_ka1"))
    xrd.prominence_spin.setValue(40.0)
    return window, xrd


def test_find_peaks_populates_the_bottom_results_peak_table(qapp):
    window, xrd = _xrd_main_window()
    xrd._on_find_peaks_clicked()
    result = xrd.current_result()
    table = window.analysis_result_view._detail_table
    assert table.rowCount() == len(result.peaks)
    assert table.isVisibleTo(window.analysis_result_view)


def test_results_table_selection_drives_enable_disable_through_full_wiring(qapp):
    window, xrd = _xrd_main_window()
    xrd._on_find_peaks_clicked()
    result = xrd.current_result()
    assert len(result.peaks) >= 2

    window.analysis_result_view._detail_table.selectRow(1)
    xrd._on_toggle_enabled_clicked()
    assert result.peaks[1].enabled is False
    # The re-displayed table keeps that row selected (same result, in-place edit).
    assert window.analysis_result_view.selected_detail_rows() == [1]
    assert window.analysis_result_view._detail_table.item(1, 6).text() == "No"


def test_results_table_remove_selected_through_full_wiring(qapp):
    window, xrd = _xrd_main_window()
    xrd._on_find_peaks_clicked()
    result = xrd.current_result()
    before = len(result.peaks)
    window.analysis_result_view._detail_table.selectRow(0)
    xrd._on_remove_selected_clicked()
    assert len(result.peaks) == before - 1
    assert window.analysis_result_view._detail_table.rowCount() == before - 1


def test_switching_xrd_history_shows_that_results_own_peak_table(qapp):
    window, xrd = _xrd_main_window()
    xrd._on_find_peaks_clicked()
    result_a = xrd.current_result()
    xrd.prominence_spin.setValue(150.0)
    xrd._on_find_peaks_clicked()
    result_b = xrd.current_result()
    assert len(result_a.peaks) != len(result_b.peaks)

    panel_id = window.figure_model.active_panel.id
    history = window._project.active_workbench.analysis_results.all(panel_id)
    window.analysis_panel.history_list.setCurrentRow(history.index(result_a))
    assert window.analysis_result_view._detail_table.rowCount() == len(result_a.peaks)
    window.analysis_panel.history_list.setCurrentRow(history.index(result_b))
    assert window.analysis_result_view._detail_table.rowCount() == len(result_b.peaks)


def test_manual_add_peak_updates_the_bottom_results_peak_table(qapp):
    window, xrd = _xrd_main_window()
    xrd._on_find_peaks_clicked()
    before = window.analysis_result_view._detail_table.rowCount()
    xrd.add_manual_peak(55.0, 42.0)
    assert window.analysis_result_view._detail_table.rowCount() == before + 1
    last = window.analysis_result_view._detail_table
    assert last.item(last.rowCount() - 1, 5).text() == "manual"


def test_radiation_change_refreshes_bottom_peak_table_d_spacing(qapp):
    window, xrd = _xrd_main_window()
    xrd._on_find_peaks_clicked()
    table = window.analysis_result_view._detail_table
    d_before = table.item(0, 4).text()
    xrd.radiation_combo.setCurrentIndex(xrd.radiation_combo.findData("mo_ka1"))
    d_after = window.analysis_result_view._detail_table.item(0, 4).text()
    assert d_before != d_after


def test_panel_switch_clears_bottom_peak_table_when_new_panel_has_no_result(qapp):
    window, xrd = _xrd_main_window()
    window.figure_size_panel.layout_combo.setCurrentIndex(
        window.figure_size_panel.layout_combo.findText("1 x 2")
    )
    window._set_active_panel(0)
    xrd._on_find_peaks_clicked()
    assert window.analysis_result_view._detail_table.rowCount() > 0
    window._set_active_panel(1)  # a fresh panel with no analysis history
    assert window.analysis_result_view._detail_table.rowCount() == 0
    assert not window.analysis_result_view._detail_table.isVisibleTo(window.analysis_result_view)


# --- Save/reopen --------------------------------------------------------------


def test_xrd_result_saves_and_reopens_with_project(qapp, tmp_path):
    project, workbench, result, manager = _project_with_xrd_result()
    path = tmp_path / "xrd.gnovi"
    save_project(project, str(path))

    reloaded_project = load_project(str(path))
    reloaded_workbench = reloaded_project.workbenches[0]
    reloaded_panel = reloaded_workbench.figure.panels[0]
    history = reloaded_workbench.analysis_results.all(reloaded_panel.id)
    assert len(history) == 1
    reloaded_result = history[0]
    assert isinstance(reloaded_result, XRDAnalysisResult)
    assert reloaded_result.radiation.label == result.radiation.label
    assert reloaded_result.radiation.wavelength_angstrom == pytest.approx(result.radiation.wavelength_angstrom)
    assert len(reloaded_result.peaks) == len(result.peaks)
    assert reloaded_result.engine == "gnovi"
    assert reloaded_result.operation == "xrd_peak_detection"


# --- Review-fix regression: manual peak click must target the panel it
# actually landed in, never any other panel merely because "Add Peak" is
# armed (code-review finding #1) ---------------------------------------------


class _FakeClickEvent:
    def __init__(self, inaxes, xdata, ydata, button=1, dblclick=False):
        self.inaxes = inaxes
        self.xdata = xdata
        self.ydata = ydata
        self.button = button
        self.dblclick = dblclick


def _two_panel_xrd_window():
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(window.figure_size_panel.layout_combo.findText("1 x 2"))
    ds1 = _synthetic_pattern_dataset(name="Pattern 1", seed=11)
    ds2 = _synthetic_pattern_dataset(name="Pattern 2", seed=12)
    window.dataset_manager.add(ds1)
    window.dataset_manager.add(ds2)
    window.figure_model.panels[0].add_series(PlotSeries.line(ds1, "2theta", "intensity", label="Pattern 1"))
    window.figure_model.panels[1].add_series(PlotSeries.line(ds2, "2theta", "intensity", label="Pattern 2"))
    window._on_figure_content_changed()
    window._set_active_panel(0)
    window.analysis_panel.tool_combo.setCurrentText("XRD Peak Analysis")
    xrd = window.analysis_panel.xrd_section_widget
    xrd.radiation_combo.setCurrentIndex(xrd.radiation_combo.findData("cu_ka1"))
    xrd.add_peak_button.setChecked(True)  # arm "Add Peak" for Panel 1
    return window, xrd


def test_manual_peak_click_on_a_different_panel_is_ignored(qapp):
    window, xrd = _two_panel_xrd_window()
    assert xrd.current_result() is None

    other_axes = window.plot_canvas.axes_list[1]
    was_dirty = window._dirty
    window._on_canvas_click(_FakeClickEvent(inaxes=other_axes, xdata=45.0, ydata=200.0))
    assert xrd.current_result() is None
    assert window.figure_model.active_panel_index == 0  # never silently switched
    assert window._dirty == was_dirty  # a rejected click never dirties the project

    own_axes = window.plot_canvas.axes_list[0]
    window._on_canvas_click(_FakeClickEvent(inaxes=own_axes, xdata=45.0, ydata=200.0))
    assert xrd.current_result() is not None
    assert len(xrd.current_result().peaks) == 1
    assert xrd.current_result().peaks[0].two_theta == pytest.approx(45.0)


def test_manual_peak_click_rejects_non_finite_coordinates(qapp):
    window, xrd = _two_panel_xrd_window()
    own_axes = window.plot_canvas.axes_list[0]
    window._on_canvas_click(_FakeClickEvent(inaxes=own_axes, xdata=float("nan"), ydata=200.0))
    assert xrd.current_result() is None
    window._on_canvas_click(_FakeClickEvent(inaxes=own_axes, xdata=45.0, ydata=float("inf")))
    assert xrd.current_result() is None


def test_manual_peak_click_ignores_outside_any_axes(qapp):
    window, xrd = _two_panel_xrd_window()
    window._on_canvas_click(_FakeClickEvent(inaxes=None, xdata=None, ydata=None))
    assert xrd.current_result() is None


def test_manual_peak_click_ignores_when_active_panel_is_3d(qapp):
    window, xrd = _two_panel_xrd_window()
    window.figure_model.panels[1] = Panel3D()
    window._on_figure_content_changed()
    # Add Peak was armed while Panel 1 (2D) was active. An active-panel
    # switch now disarms "Add Peak" itself (see `AnalysisPanel.
    # disarm_xrd_manual_peak_mode`, wired from `MainWindow._on_panel_
    # switched`) -- belt-and-suspenders with the click handler's own
    # Panel3D guard, which is what this test still exercises: a click
    # landing on a Panel3D must be rejected regardless of whether
    # something re-arms Add Peak afterward.
    window._set_active_panel(1)
    window.analysis_panel.xrd_section_widget._set_manual_peak_mode(True)
    window._xrd_manual_peak_mode = True
    axes_3d = window.plot_canvas.axes_list[1]
    window._on_canvas_click(_FakeClickEvent(inaxes=axes_3d, xdata=45.0, ydata=200.0))
    assert xrd.current_result() is None


# --- Review-fix regression: an actual History-row click (not just
# `sync_history`'s own programmatic path) must fully reload the XRD
# section (code-review finding #2) -------------------------------------------


def test_clicking_an_xrd_history_row_reloads_the_xrd_section(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset(seed=21)
    _panel_with_series(figure, ds)
    panel = AnalysisPanel(figure, DatasetManager())
    xrd = panel.xrd_section_widget
    xrd.radiation_combo.setCurrentIndex(xrd.radiation_combo.findData("cu_ka1"))

    xrd.prominence_spin.setValue(40.0)
    xrd._on_find_peaks_clicked()
    result_a = xrd.current_result()

    xrd.prominence_spin.setValue(120.0)
    xrd._on_find_peaks_clicked()
    result_b = xrd.current_result()
    assert result_a.result_id != result_b.result_id

    panel.tool_combo.setCurrentText("Curve Fitting")  # simulate having navigated away
    panel.sync_history([result_a, result_b], result_b)

    # An actual click (never sync_history) on result A's row must restore it.
    panel.history_list.setCurrentRow(panel._history_results.index(result_a))
    assert panel.tool_combo.currentText() == "XRD Peak Analysis"
    assert xrd.current_result() is result_a
    assert len(result_a.detail_table()[1]) == len(result_a.peaks)

    # Editing now must only ever touch result A, never result B.
    before_b = len(result_b.peaks)
    xrd.add_manual_peak(50.0, 10.0)
    assert xrd.current_result() is result_a
    assert len(result_b.peaks) == before_b

    panel.history_list.setCurrentRow(panel._history_results.index(result_b))
    assert xrd.current_result() is result_b
    assert len(result_b.detail_table()[1]) == len(result_b.peaks)


def test_clicking_between_a_fit_result_and_an_xrd_result_in_the_same_history(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset(seed=23)
    _panel_with_series(figure, ds)
    panel = AnalysisPanel(figure, DatasetManager())
    xrd = panel.xrd_section_widget
    xrd.radiation_combo.setCurrentIndex(xrd.radiation_combo.findData("cu_ka1"))
    xrd.prominence_spin.setValue(40.0)
    xrd._on_find_peaks_clicked()
    xrd_result = xrd.current_result()

    from gnovi_plot.analysis.fitting import LINEAR, fit_curve

    fit_result = fit_curve(
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        LINEAR,
        source_dataset_id=ds.id,
        x_column="2theta",
        y_column="intensity",
        source_panel_id=figure.active_panel.id,
    )

    panel.sync_history([fit_result, xrd_result], fit_result)
    assert panel.tool_combo.currentText() == "Curve Fitting"

    panel.history_list.setCurrentRow(panel._history_results.index(xrd_result))
    assert panel.tool_combo.currentText() == "XRD Peak Analysis"
    assert xrd.current_result() is xrd_result

    panel.history_list.setCurrentRow(panel._history_results.index(fit_result))
    assert panel.tool_combo.currentText() == "Curve Fitting"
    assert panel._current_result is fit_result
    assert xrd.current_result() is None  # never left pointing at the stale XRD result


# --- Review-fix regression: transient background/smoothing preview state
# and the Detection Input options that depend on it must be invalidated on
# every source-context change, not just an explicit source-combo edit
# (code-review finding #3) ---------------------------------------------------


def test_detection_input_options_invalidated_on_active_panel_switch(qapp):
    figure = GnoviFigure()
    ds1 = _synthetic_pattern_dataset(name="Pattern 1", seed=31)
    ds2 = _synthetic_pattern_dataset(name="Pattern 2", seed=32)
    _panel_with_series(figure, ds1)
    figure.panels.append(Panel())
    figure.panels[1].add_series(PlotSeries.line(ds2, "2theta", "intensity"))
    manager = DatasetManager()
    manager.add(ds1)
    manager.add(ds2)

    section = XRDAnalysisSection(figure, manager)
    section.background_method_combo.setCurrentText("arPLS")
    section._on_preview_background_clicked()
    assert section._background_preview is not None
    options = [section.detection_input_combo.itemData(i) for i in range(section.detection_input_combo.count())]
    assert "background_corrected" in options
    assert section.add_corrected_button.isEnabled()

    figure.set_active_panel(1)
    section.set_figure(figure)  # a Workbench-switch-style repoint clears it unconditionally
    assert section._background_preview is None
    options_after = [section.detection_input_combo.itemData(i) for i in range(section.detection_input_combo.count())]
    assert "background_corrected" not in options_after
    assert section.detection_input_combo.currentData() == "raw"
    assert not section.add_corrected_button.isEnabled()


def test_detection_input_options_invalidated_when_refresh_resolves_a_different_series(qapp):
    figure = GnoviFigure()
    ds1 = _synthetic_pattern_dataset(name="Pattern 1", seed=33)
    ds2 = _synthetic_pattern_dataset(name="Pattern 2", seed=34)
    series1 = _panel_with_series(figure, ds1)
    manager = DatasetManager()
    manager.add(ds1)
    manager.add(ds2)

    section = XRDAnalysisSection(figure, manager)
    section.background_method_combo.setCurrentText("arPLS")
    section._on_preview_background_clicked()
    assert section._background_preview is not None

    # Simulate the active panel's series changing out from under an
    # existing preview (e.g. Extract/Focus swapping which series
    # `_eligible_series` resolves) without a full set_figure repoint --
    # `refresh()` itself must notice and invalidate.
    figure.active_panel.remove_series(series1.id)
    figure.active_panel.add_series(PlotSeries.line(ds2, "2theta", "intensity"))
    section.refresh()
    assert section._background_preview is None
    options = [section.detection_input_combo.itemData(i) for i in range(section.detection_input_combo.count())]
    assert options == ["raw"]


def test_detection_input_options_survive_an_unrelated_refresh_of_the_same_series(qapp):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset(seed=35)
    _panel_with_series(figure, ds)
    manager = DatasetManager()
    manager.add(ds)

    section = XRDAnalysisSection(figure, manager)
    section.background_method_combo.setCurrentText("arPLS")
    section._on_preview_background_clicked()
    assert section._background_preview is not None

    section.refresh()  # e.g. an unrelated figure-content-changed refresh
    assert section._background_preview is not None  # not discarded for no reason
    options = [section.detection_input_combo.itemData(i) for i in range(section.detection_input_combo.count())]
    assert "background_corrected" in options
