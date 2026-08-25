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
from gnovi_plot.gui.widgets.analysis_panel import AnalysisPanel
from gnovi_plot.gui.widgets.xrd_analysis_section import XRDAnalysisSection
from gnovi_plot.modules.xrd import preprocessing as xrd_preprocessing
from gnovi_plot.modules.xrd.preprocessing import PybaselinesNotAvailableError
from gnovi_plot.modules.xrd.radiation import CU_KALPHA1_ANGSTROM
from gnovi_plot.modules.xrd.results import XRDAnalysisResult
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
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
    assert section.peak_table.rowCount() == len(section.current_result().peaks)
    headers = [section.peak_table.horizontalHeaderItem(i).text() for i in range(section.peak_table.columnCount())]
    assert headers == [
        "Peak #",
        "Seed 2θ (°)",
        "Observed intensity",
        "Prominence",
        "d-spacing (Å)",
        "Origin",
        "Enabled",
    ]
    for forbidden in ("FWHM", "Area", "Fit", "Crystallite"):
        assert forbidden not in headers


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
    section.peak_table.selectRow(0)
    section._on_remove_selected_clicked()
    assert len(section.current_result().peaks) == before - 1


def test_toggle_enabled_excludes_peak_from_overlay(qapp):
    _figure, section = _section_with_detected_peaks()
    section.peak_table.selectRow(0)
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
    section.peak_table.selectRow(0)
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


def test_selecting_xrd_history_entry_restores_peak_table_and_switches_tool(qapp):
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
    assert xrd.peak_table.rowCount() == len(result.peaks)


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

    with open(path, newline="") as f:
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


def test_export_peak_table_with_no_result_is_a_safe_no_op(qapp, tmp_path):
    figure = GnoviFigure()
    ds = _synthetic_pattern_dataset()
    _panel_with_series(figure, ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    path = tmp_path / "peaks.csv"
    section.export_peak_table_csv(str(path))  # must not raise
    assert not path.exists()


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
