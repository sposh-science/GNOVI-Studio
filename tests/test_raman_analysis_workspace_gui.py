"""Raman-3: the Raman Peak Analysis workspace inside RamanAnalysisSection.

Covers: section construction, source-series selection/Panel3D rejection,
background method switching (off by default), transient preview
immutability, smoothing (off by default, validation), detection-input
selection across all four combinations, baseline-then-smoothing processing
order, find-peaks/prominence-default behavior, manual add/remove/enable-
disable, new-result-vs-update-in-place, marker overlay, label modes,
derived corrected/smoothed curves, provenance parameters, load_result, and
no-pybaselines behavior. MainWindow/AnalysisPanel integration is out of
scope here (Issue #75); this exercises RamanAnalysisSection directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.raman_analysis_section import RamanAnalysisSection
from gnovi_plot.modules.raman.peaks import ORIGIN_AUTOMATIC, ORIGIN_MANUAL
from gnovi_plot.modules.raman.results import RamanAnalysisResult
from gnovi_plot.modules.xrd import preprocessing as xrd_preprocessing
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series import PlotSeries
from gnovi_plot.plotting.series3d import Plot3DType, Series3D


def _gaussian(x, center, amp, sigma):
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _synthetic_spectrum_dataset(name="Raman spectrum", seed=1, sloped_background=False):
    rng = np.random.default_rng(seed)
    raman_shift = np.linspace(100.0, 3200.0, 2000)
    background = 20.0 + (0.05 * raman_shift if sloped_background else 0.0)
    intensity = background
    for center, amp, sigma in [(1350.0, 500.0, 4.0), (1580.0, 300.0, 4.0), (2700.0, 150.0, 5.0)]:
        intensity = intensity + _gaussian(raman_shift, center, amp, sigma)
    intensity = intensity + rng.normal(0, 2.0, size=raman_shift.shape)
    df = pd.DataFrame({"raman_shift": raman_shift, "intensity": intensity})
    return Dataset(name=name, dataframe=df)


def _panel_with_series(figure: GnoviFigure, dataset: Dataset) -> PlotSeries:
    series = PlotSeries.line(dataset, "raman_shift", "intensity")
    figure.add_series(series)
    return series


def _section_with_series(**kwargs):
    figure = GnoviFigure()
    ds = _synthetic_spectrum_dataset(**kwargs)
    _panel_with_series(figure, ds)
    manager = DatasetManager()
    manager.add(ds)
    section = RamanAnalysisSection(figure, manager)
    return figure, section, ds


# --- construction / source selection -----------------------------------


def test_section_constructs_and_enables_controls_with_an_eligible_series(qapp):
    _figure, section, _ds = _section_with_series()
    assert section.source_combo.isEnabled()
    assert section.find_peaks_button.isEnabled()
    assert not section.status_label.isVisible()


def test_no_eligible_series_disables_controls_and_shows_a_message(qapp):
    figure = GnoviFigure()
    section = RamanAnalysisSection(figure, DatasetManager())
    assert not section.source_combo.isEnabled()
    assert not section.find_peaks_button.isEnabled()
    assert section.status_label.isVisibleTo(section)


def test_panel3d_active_panel_disables_controls_with_a_clear_message(qapp):
    figure = GnoviFigure()
    figure.panels[0] = Panel3D()
    ds = _synthetic_spectrum_dataset()
    figure.panels[0].add_series(
        Series3D(dataset=ds, x_column="raman_shift", y_column="intensity", z_column="raman_shift",
                 plot_type=Plot3DType.SCATTER, label="s")
    )
    section = RamanAnalysisSection(figure, DatasetManager())
    assert not section.source_combo.isEnabled()
    assert "2D Panel" in section.status_label.text()


def test_source_combo_excludes_histogram_series(qapp):
    figure = GnoviFigure()
    ds = _synthetic_spectrum_dataset()
    figure.add_series(PlotSeries.line(ds, "raman_shift", "intensity"))
    figure.add_series(PlotSeries.histogram(ds, "raman_shift"))
    manager = DatasetManager()
    manager.add(ds)
    section = RamanAnalysisSection(figure, manager)
    assert section.source_combo.count() == 1


# --- baseline: off by default, Polynomial, arPLS ------------------------


def test_background_method_defaults_to_none(qapp):
    _figure, section, _ds = _section_with_series()
    assert section.background_method_combo.currentText() == "None"
    assert not section.arpls_lam_spin.isVisible()
    assert not section.baseline_points_edit.isVisible()


def test_polynomial_baseline_preview_does_not_mutate_source_data(qapp):
    figure, section, ds = _section_with_series(sloped_background=True)
    raw_before = ds.dataframe["intensity"].to_numpy().copy()

    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()

    assert section._background_preview is not None
    assert section.add_corrected_button.isEnabled()
    np.testing.assert_array_equal(ds.dataframe["intensity"].to_numpy(), raw_before)


def test_arpls_baseline_preview_uses_the_configured_lambda(qapp):
    _figure, section, _ds = _section_with_series(sloped_background=True)
    section.background_method_combo.setCurrentText("arPLS")
    section.arpls_lam_spin.setValue(5e4)
    section._on_preview_background_clicked()
    assert section._background_preview is not None
    assert section._background_preview.parameters["lam"] == 5e4


def test_arpls_unavailable_shows_a_raman_appropriate_message_not_a_crash(qapp, monkeypatch):
    monkeypatch.setattr(xrd_preprocessing, "_PYBASELINES_AVAILABLE", False)
    _figure, section, _ds = _section_with_series()
    section.background_method_combo.setCurrentText("arPLS")
    calls = []
    monkeypatch.setattr(
        "gnovi_plot.gui.widgets.raman_analysis_section.QMessageBox.critical",
        lambda *a, **k: calls.append(a[2]),
    )
    section._on_preview_background_clicked()
    assert calls and "pybaselines" in calls[0]
    # The message must not claim this is "XRD support" -- it's shown in a
    # Raman dialog (see the design audit's own §7 recommendation).
    assert "XRD support" not in calls[0]
    assert section._background_preview is None
    # Everything else keeps working without pybaselines.
    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()
    assert section._background_preview is not None


def test_gnovi_raman_section_constructs_without_pybaselines(qapp, monkeypatch):
    monkeypatch.setattr(xrd_preprocessing, "_PYBASELINES_AVAILABLE", False)
    _figure, section, _ds = _section_with_series()  # must not raise
    assert section.isEnabled()


# --- smoothing: off by default, validation -------------------------------


def test_smoothing_defaults_to_off(qapp):
    _figure, section, _ds = _section_with_series()
    assert not section.smoothing_enabled_check.isChecked()
    assert not section.smoothing_window_spin.isVisible()


def test_smoothing_preview_does_not_mutate_source_data(qapp):
    figure, section, ds = _section_with_series()
    raw_before = ds.dataframe["intensity"].to_numpy().copy()

    section.smoothing_enabled_check.setChecked(True)
    section._on_preview_smoothed_clicked()

    assert section._smooth_preview is not None
    np.testing.assert_array_equal(ds.dataframe["intensity"].to_numpy(), raw_before)


def test_smoothing_rejects_an_even_window_length(qapp, monkeypatch):
    _figure, section, _ds = _section_with_series()
    section.smoothing_enabled_check.setChecked(True)
    section.smoothing_window_spin.setValue(10)  # even -- invalid
    calls = []
    monkeypatch.setattr(
        "gnovi_plot.gui.widgets.raman_analysis_section.QMessageBox.critical",
        lambda *a, **k: calls.append(a[2]),
    )
    section._on_preview_smoothed_clicked()
    assert calls
    assert section._smooth_preview is None


# --- processing order: baseline before smoothing --------------------------


def test_smoothing_operates_on_the_baseline_corrected_spectrum_when_both_active(qapp):
    _figure, section, _ds = _section_with_series(sloped_background=True)
    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()
    corrected_y = section._background_preview.corrected

    section.smoothing_enabled_check.setChecked(True)
    section._on_preview_smoothed_clicked()

    # The smoothed preview should track the baseline-corrected curve's
    # own scale, not the raw (still-sloped) curve's -- confirms smoothing
    # ran on the corrected spectrum, not raw, when both are active.
    assert section._smooth_preview.smoothed_intensity.mean() == pytest.approx(corrected_y.mean(), abs=50.0)


# --- detection-input chain: all four combinations --------------------------


def test_detection_input_offers_only_raw_when_nothing_else_is_active(qapp):
    _figure, section, _ds = _section_with_series()
    options = [section.detection_input_combo.itemText(i) for i in range(section.detection_input_combo.count())]
    assert options == ["Raw"]


def test_detection_input_offers_background_corrected_once_previewed(qapp):
    _figure, section, _ds = _section_with_series(sloped_background=True)
    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()
    options = [section.detection_input_combo.itemText(i) for i in range(section.detection_input_combo.count())]
    assert options == ["Raw", "Background-corrected"]


def test_detection_input_offers_all_four_once_both_previews_exist(qapp):
    _figure, section, _ds = _section_with_series(sloped_background=True)
    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()
    section.smoothing_enabled_check.setChecked(True)
    section._on_preview_smoothed_clicked()
    options = [section.detection_input_combo.itemText(i) for i in range(section.detection_input_combo.count())]
    assert options == ["Raw", "Background-corrected", "Smoothed raw", "Smoothed background-corrected"]


@pytest.mark.parametrize(
    "choice_text",
    ["Raw", "Background-corrected", "Smoothed raw", "Smoothed background-corrected"],
)
def test_find_peaks_detects_against_the_resolved_detection_input(qapp, choice_text):
    _figure, section, _ds = _section_with_series(sloped_background=True)
    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()
    section.smoothing_enabled_check.setChecked(True)
    section._on_preview_smoothed_clicked()

    idx = section.detection_input_combo.findText(choice_text)
    section.detection_input_combo.setCurrentIndex(idx)
    section.prominence_spin.setValue(50.0)
    section.find_peaks_button.click()

    result = section.current_result()
    assert result is not None
    assert len(result.peaks) >= 1
    assert result.parameters["detection_input"] == {
        "Raw": "raw",
        "Background-corrected": "background_corrected",
        "Smoothed raw": "smoothed_raw",
        "Smoothed background-corrected": "smoothed_background_corrected",
    }[choice_text]


# --- provenance -----------------------------------------------------------


def test_provenance_records_preprocessing_and_detection_settings(qapp):
    _figure, section, _ds = _section_with_series(sloped_background=True)
    section.background_method_combo.setCurrentText("arPLS")
    section.arpls_lam_spin.setValue(2e4)
    section._on_preview_background_clicked()
    idx = section.detection_input_combo.findText("Background-corrected")
    section.detection_input_combo.setCurrentIndex(idx)
    section.prominence_spin.setValue(50.0)
    section.distance_spin.setValue(20)

    section.find_peaks_button.click()

    parameters = section.current_result().parameters
    assert parameters["detection"]["prominence"] == 50.0
    assert parameters["detection"]["distance"] == 20
    assert parameters["detection_input"] == "background_corrected"
    assert parameters["preprocessing"]["background"]["method"] == "arpls"
    assert parameters["preprocessing"]["background"]["lam"] == 2e4
    assert parameters["preprocessing"]["smoothing"] is None


def test_source_data_never_mutated_across_a_full_workflow(qapp):
    figure, section, ds = _section_with_series(sloped_background=True)
    raw_before = ds.dataframe["intensity"].to_numpy().copy()

    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()
    section.smoothing_enabled_check.setChecked(True)
    section._on_preview_smoothed_clicked()
    section.find_peaks_button.click()
    section.add_peak_button.setChecked(True)
    section.add_manual_peak(1500.0, 400.0)

    np.testing.assert_array_equal(ds.dataframe["intensity"].to_numpy(), raw_before)


# --- peak detection / Find Peaks result creation --------------------------


def test_find_peaks_creates_a_new_result_and_emits_analysis_result_ready(qapp):
    _figure, section, _ds = _section_with_series()
    received = []
    section.analysis_result_ready.connect(received.append)

    section.prominence_spin.setValue(50.0)
    section.find_peaks_button.click()

    assert len(received) == 1
    assert isinstance(received[0], RamanAnalysisResult)
    assert section.current_result() is received[0]
    assert all(p.origin == ORIGIN_AUTOMATIC for p in received[0].peaks)


def test_repeated_find_peaks_creates_a_second_distinct_result(qapp):
    _figure, section, _ds = _section_with_series()
    section.prominence_spin.setValue(50.0)
    section.find_peaks_button.click()
    first = section.current_result()
    section.find_peaks_button.click()
    second = section.current_result()
    assert first.result_id != second.result_id


# --- manual peak add/remove/enable-disable --------------------------------


def test_manual_add_peak_on_a_fresh_panel_creates_a_result_and_emits_ready(qapp):
    _figure, section, _ds = _section_with_series()
    received = []
    section.analysis_result_ready.connect(received.append)

    section.add_peak_button.setChecked(True)
    assert section.is_manual_peak_mode() is True
    section.add_manual_peak(1580.0, 450.0)

    assert section.is_manual_peak_mode() is False  # consumed by the click
    assert len(received) == 1
    assert len(section.current_result().peaks) == 1
    assert section.current_result().peaks[0].origin == ORIGIN_MANUAL


def test_manual_add_peak_on_an_existing_result_updates_in_place(qapp):
    _figure, section, _ds = _section_with_series()
    section.prominence_spin.setValue(50.0)
    section.find_peaks_button.click()
    existing = section.current_result()
    n_before = len(existing.peaks)

    updated = []
    section.result_updated.connect(updated.append)
    section.add_manual_peak(1500.0, 200.0)

    assert section.current_result() is existing  # same object, no new History entry
    assert len(existing.peaks) == n_before + 1
    assert len(updated) == 1
    assert updated[0] is existing


def test_remove_selected_deletes_peaks_and_emits_result_updated(qapp):
    _figure, section, _ds = _section_with_series()
    section.prominence_spin.setValue(50.0)
    section.find_peaks_button.click()
    n_before = len(section.current_result().peaks)
    assert n_before >= 1

    updated = []
    section.result_updated.connect(updated.append)
    section.set_selected_peak_rows([0])
    section._on_remove_selected_clicked()

    assert len(section.current_result().peaks) == n_before - 1
    assert len(updated) == 1


def test_toggle_enabled_flips_the_enabled_flag(qapp):
    _figure, section, _ds = _section_with_series()
    section.prominence_spin.setValue(50.0)
    section.find_peaks_button.click()
    peak = section.current_result().peaks[0]
    assert peak.enabled is True

    section.set_selected_peak_rows([0])
    section._on_toggle_enabled_clicked()

    assert peak.enabled is False


# --- overlay payload --------------------------------------------------------


def test_overlay_points_excludes_disabled_peaks(qapp):
    _figure, section, _ds = _section_with_series()
    section.prominence_spin.setValue(50.0)
    section.find_peaks_button.click()
    section.current_result().peaks[0].enabled = False

    points = section.overlay_points()

    assert len(points) == len(section.current_result().peaks) - 1


def test_overlay_points_label_mode_raman_shift(qapp):
    _figure, section, _ds = _section_with_series()
    section.prominence_spin.setValue(50.0)
    section.find_peaks_button.click()
    section.label_mode_combo.setCurrentText("Raman shift")

    points = section.overlay_points()

    assert all("cm" in label for _x, _y, label in points)


def test_overlay_points_none_without_a_current_result(qapp):
    _figure, section, _ds = _section_with_series()
    assert section.overlay_points() is None


def test_preview_curve_prefers_smoothed_over_background(qapp):
    _figure, section, _ds = _section_with_series(sloped_background=True)
    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()
    section.smoothing_enabled_check.setChecked(True)
    section._on_preview_smoothed_clicked()

    x, y = section.preview_curve()

    np.testing.assert_array_equal(y, section._smooth_preview.smoothed_intensity)


# --- derived series ---------------------------------------------------------


def test_add_corrected_curve_emits_a_raman_named_series(qapp):
    _figure, section, ds = _section_with_series(sloped_background=True)
    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()

    added = []
    section.add_to_plot_requested.connect(added.append)
    section._on_add_corrected_clicked()

    assert len(added) == 1
    (series,) = added[0]
    assert series.label.startswith("Raman background-corrected:")


def test_add_smoothed_curve_emits_a_raman_named_series(qapp):
    _figure, section, ds = _section_with_series()
    section.smoothing_enabled_check.setChecked(True)
    section._on_preview_smoothed_clicked()

    added = []
    section.add_to_plot_requested.connect(added.append)
    section._on_add_smoothed_clicked()

    assert len(added) == 1
    (series,) = added[0]
    assert series.label.startswith("Raman smoothed:")


# --- load_result -------------------------------------------------------------


def test_load_result_restores_source_and_detection_settings_without_a_preview(qapp):
    _figure, section, _ds = _section_with_series(sloped_background=True)
    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-30, 1970-1999")
    section._on_preview_background_clicked()
    section.prominence_spin.setValue(75.0)
    section.distance_spin.setValue(30)
    section.find_peaks_button.click()
    result = section.current_result()

    # Simulate a History selection landing elsewhere, then back.
    section._invalidate_previews()
    section.prominence_spin.setValue(0.0)
    section.distance_spin.setValue(0)

    section.load_result(result)

    assert section.current_result() is result
    assert section.prominence_spin.value() == 75.0
    assert section.distance_spin.value() == 30
    # load_result must NOT silently recompute a preview.
    assert section._background_preview is None


def test_load_result_ignores_a_non_raman_result(qapp):
    _figure, section, _ds = _section_with_series()
    section.load_result(None)
    assert section.current_result() is None


# --- prominence default ------------------------------------------------------


def test_a_fresh_source_series_gets_a_nonzero_data_dependent_prominence_default(qapp):
    _figure, section, _ds = _section_with_series()
    assert section.prominence_spin.value() > 0.0
