"""XRD-3B: the Peak Profile Fitting subsection inside XRDAnalysisSection.

Covers the researcher workflow over the frozen XRD-3A numerical engine:
peak selection, fit-window proposal/edit + transient span, model/baseline
choice, Fit Peak, strict stale invalidation, transient total-fit/baseline
overlay, the quantitative Results display, warnings, the windowed residual
range, Add/Remove Fitted Curve, history routing/load, radiation-optional
d-spacing, and the GUI-only overlay/export boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnovi_plot.analysis.fitting import LINEAR, fit_curve
from gnovi_plot.core.project_io import load_project, save_project
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.widgets.xrd_analysis_section import XRDAnalysisSection
from gnovi_plot.modules.xrd.fitting import (
    GAUSSIAN,
    LORENTZIAN,
    PSEUDO_VOIGT,
    XRDPeakFitResult,
    evaluate_baseline,
    sample_fit_curve,
)
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series import PlotSeries


def _gaussian(x, center, amp, sigma):
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _pattern_dataset(name="XRD pattern", seed=7):
    rng = np.random.default_rng(seed)
    two_theta = np.linspace(10.0, 90.0, 3000)
    intensity = 20.0 + 0.1 * two_theta
    for center, amp, sigma in [(30.0, 500.0, 0.10), (45.0, 300.0, 0.10), (60.0, 150.0, 0.09)]:
        intensity = intensity + _gaussian(two_theta, center, amp, sigma)
    intensity = intensity + rng.normal(0, 1.5, size=two_theta.shape)
    return Dataset(name=name, dataframe=pd.DataFrame({"2theta": two_theta, "intensity": intensity}))


def _bare_section(*, detect=True, radiation=True):
    figure = GnoviFigure()
    ds = _pattern_dataset()
    figure.add_series(PlotSeries.line(ds, "2theta", "intensity", label="pattern"))
    section = XRDAnalysisSection(figure, DatasetManager())
    if radiation:
        section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    if detect:
        section.prominence_spin.setValue(40.0)
        section._on_find_peaks_clicked()
    section.fitting_section.set_expanded(True)
    return figure, section


def _select_first_peak_and_fit(section, model=GAUSSIAN):
    section.fit_peak_combo.setCurrentIndex(0)
    section.fit_model_combo.setCurrentIndex(section.fit_model_combo.findData(model))
    produced = []
    section.analysis_result_ready.connect(produced.append)
    section._on_fit_peak_clicked()
    return produced[-1] if produced else None


def _xrd_window(seed=7):
    window = MainWindow()
    ds = _pattern_dataset(seed=seed)
    window.dataset_manager.add(ds)
    window.figure_model.active_panel.add_series(
        PlotSeries.line(ds, "2theta", "intensity", label="pattern")
    )
    window._on_figure_content_changed()
    window.analysis_panel.tool_combo.setCurrentText("XRD Peak Analysis")
    xrd = window.analysis_panel.xrd_section_widget
    xrd.radiation_combo.setCurrentIndex(xrd.radiation_combo.findData("cu_ka1"))
    xrd.prominence_spin.setValue(40.0)
    xrd._on_find_peaks_clicked()
    xrd.fitting_section.set_expanded(True)
    return window, xrd


# =====================================================================
# A. initial disabled state
# =====================================================================


def test_fitting_disabled_without_a_source_or_detected_peaks(qapp):
    figure = GnoviFigure()
    section = XRDAnalysisSection(figure, DatasetManager())
    assert section.fit_peak_button.isEnabled() is False
    assert section.fit_peak_combo.count() == 0

    figure2, section2 = _bare_section(detect=False)
    assert section2.fit_peak_button.isEnabled() is False
    assert section2.fit_peak_combo.count() == 0
    assert section2.add_fitted_curve_button.isEnabled() is False


def test_fitting_disabled_when_active_panel_is_3d(qapp):
    figure = GnoviFigure()
    ds = _pattern_dataset()
    figure.add_series(PlotSeries.line(ds, "2theta", "intensity"))
    section = XRDAnalysisSection(figure, DatasetManager())
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section.prominence_spin.setValue(40.0)
    section._on_find_peaks_clicked()

    figure.panels = list(figure.panels) + [Panel3D()]
    figure.active_panel_index = 1
    section.refresh()
    assert section.fit_peak_button.isEnabled() is False
    assert section.fit_overlay() == (None, None)


# =====================================================================
# B. detected peaks populate the dropdown
# =====================================================================


def test_enabled_peaks_populate_the_dropdown_with_readable_labels(qapp):
    _figure, section = _bare_section()
    result = section.current_result()
    assert section.fit_peak_combo.count() == len([p for p in result.peaks if p.enabled])
    text = section.fit_peak_combo.itemText(0)
    assert text.startswith("Peak 1 — ") and "° 2θ" in text
    assert section.fit_peak_combo.itemData(0) == result.peaks[0].id


def test_disabling_a_peak_drops_it_from_the_dropdown(qapp):
    window, xrd = _xrd_window()
    n_before = xrd.fit_peak_combo.count()
    window.analysis_result_view._detail_table.selectRow(1)
    xrd._on_toggle_enabled_clicked()
    assert xrd.fit_peak_combo.count() == n_before - 1


# =====================================================================
# C. peak selection proposes a window + transient span
# =====================================================================


def test_selecting_a_peak_proposes_a_visible_fit_window(qapp):
    _figure, section = _bare_section()
    section.fit_peak_combo.setCurrentIndex(0)
    seed_2theta = section._selected_fit_seed().two_theta
    lo, hi = section.fit_min_spin.value(), section.fit_max_spin.value()
    assert lo < seed_2theta < hi
    window, curves = section.fit_overlay()
    assert window == pytest.approx((lo, hi))
    assert curves is None  # no fit yet


def test_editing_the_window_updates_the_span_and_invalidates_the_fit(qapp):
    _figure, section = _bare_section()
    _select_first_peak_and_fit(section)
    assert section.current_fit_result() is not None
    section.fit_max_spin.setValue(section.fit_max_spin.value() + 0.5)
    assert section.current_fit_result() is None
    window, curves = section.fit_overlay()
    assert window is not None and curves is None


# =====================================================================
# D. fit-window validity
# =====================================================================


def test_invalid_fit_window_disables_fit_peak_with_a_message(qapp):
    _figure, section = _bare_section()
    section.fit_peak_combo.setCurrentIndex(0)
    section.fit_min_spin.setValue(40.0)
    section.fit_max_spin.setValue(38.0)
    assert section.fit_peak_button.isEnabled() is False
    assert "greater than min" in section.fit_status_label.text()
    section.fit_max_spin.setValue(42.0)
    assert section.fit_peak_button.isEnabled() is True


# =====================================================================
# E / F. model + baseline reach fit_xrd_peak; successful fit
# =====================================================================


@pytest.mark.parametrize("model", [GAUSSIAN, LORENTZIAN, PSEUDO_VOIGT])
def test_selected_model_reaches_the_engine(qapp, model):
    _figure, section = _bare_section()
    section.fit_baseline_combo.setCurrentIndex(section.fit_baseline_combo.findData("constant"))
    result = _select_first_peak_and_fit(section, model=model)
    assert isinstance(result, XRDPeakFitResult)
    assert result.model == model
    assert result.baseline_model == "constant"
    assert result.source_peak_id == section.current_result().peaks[0].id
    assert result.source_result_id == section.current_result().result_id


def test_successful_fit_creates_history_entry_and_shows_results_and_overlay(qapp):
    window, xrd = _xrd_window()
    panel_id = window.figure_model.active_panel.id
    history_before = len(window._project.active_workbench.analysis_results.all(panel_id))

    xrd.fit_peak_combo.setCurrentIndex(0)
    xrd._on_fit_peak_clicked()

    history = window._project.active_workbench.analysis_results.all(panel_id)
    assert len(history) == history_before + 1
    assert isinstance(history[-1], XRDPeakFitResult)
    assert window.analysis_result_view.result is history[-1]

    detail_labels = {label for label, _ in history[-1].details()}
    assert "Standard errors" in detail_labels
    assert "Center (°2θ)" in detail_labels

    fit_window, fit_curves = xrd.fit_overlay()
    assert fit_window is not None
    assert fit_curves is not None and "total_xy" in fit_curves and "baseline_xy" in fit_curves
    assert xrd.add_fitted_curve_button.isEnabled() is True


# =====================================================================
# G / H. strict stale invalidation
# =====================================================================


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.fit_model_combo.setCurrentIndex(s.fit_model_combo.findData(LORENTZIAN)),
        lambda s: s.fit_baseline_combo.setCurrentIndex(s.fit_baseline_combo.findData("none")),
        lambda s: s.fit_min_spin.setValue(s.fit_min_spin.value() - 0.3),
        lambda s: s.fit_max_spin.setValue(s.fit_max_spin.value() + 0.3),
        lambda s: s.fit_peak_combo.setCurrentIndex(1),
        lambda s: s.radiation_combo.setCurrentIndex(s.radiation_combo.findData("co_ka1")),
    ],
    ids=["model", "baseline", "window_min", "window_max", "peak", "radiation"],
)
def test_changing_any_fit_defining_input_invalidates_the_working_fit(qapp, mutate):
    _figure, section = _bare_section()
    _select_first_peak_and_fit(section)
    assert section.current_fit_result() is not None
    assert section.add_fitted_curve_button.isEnabled() is True

    mutate(section)

    assert section.current_fit_result() is None
    assert section.add_fitted_curve_button.isEnabled() is False
    _window, curves = section.fit_overlay()
    assert curves is None


def test_changing_source_series_invalidates_the_working_fit(qapp):
    figure = GnoviFigure()
    ds = _pattern_dataset()
    figure.add_series(PlotSeries.line(ds, "2theta", "intensity", label="a"))
    figure.add_series(PlotSeries.line(ds, "2theta", "intensity", label="b"))
    section = XRDAnalysisSection(figure, DatasetManager())
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section.prominence_spin.setValue(40.0)
    section.source_combo.setCurrentIndex(0)
    section._on_find_peaks_clicked()
    section.fitting_section.set_expanded(True)
    _select_first_peak_and_fit(section)
    assert section.current_fit_result() is not None

    section.source_combo.setCurrentIndex(1)
    assert section.current_fit_result() is None


def test_stale_invalidation_never_deletes_the_history_entry(qapp):
    window, xrd = _xrd_window()
    panel_id = window.figure_model.active_panel.id
    xrd.fit_peak_combo.setCurrentIndex(0)
    xrd._on_fit_peak_clicked()
    n_history = len(window._project.active_workbench.analysis_results.all(panel_id))

    xrd.fit_model_combo.setCurrentIndex(xrd.fit_model_combo.findData(LORENTZIAN))

    assert xrd.current_fit_result() is None
    assert len(window._project.active_workbench.analysis_results.all(panel_id)) == n_history


# =====================================================================
# I / J. Add / Remove Fitted Curve
# =====================================================================


def test_add_fitted_curve_adds_one_total_fit_series_through_the_normal_path(qapp):
    window, xrd = _xrd_window()
    series_before = len(window.figure_model.series)
    datasets_before = len(window.dataset_manager.datasets)

    xrd.fit_peak_combo.setCurrentIndex(0)
    xrd._on_fit_peak_clicked()
    result = xrd.current_fit_result()
    xrd.add_fitted_curve_button.click()

    assert len(window.figure_model.series) == series_before + 1
    assert len(window.dataset_manager.datasets) == datasets_before + 1
    added = window.figure_model.series[-1]
    assert added.dataset.metadata.get("result_id") == result.result_id
    assert added.label.startswith("Peak fit — ")

    x_expected, y_expected = sample_fit_curve(result)
    np.testing.assert_allclose(added.dataset.dataframe[result.x_column].to_numpy(), x_expected)
    np.testing.assert_allclose(added.dataset.dataframe[result.y_column].to_numpy(), y_expected)
    # total fit only -- not the baseline
    assert not np.allclose(added.dataset.dataframe[result.y_column].to_numpy(),
                           evaluate_baseline(result, x_expected))

    # second add is blocked (strict per-result_id toggle)
    assert xrd.add_fitted_curve_button.isEnabled() is False
    assert xrd.remove_fitted_curve_button.isEnabled() is True
    xrd.add_fitted_curve_button.click()
    assert len(window.figure_model.series) == series_before + 1

    # undo removes the fitted-curve series like any other add
    window.undo_action.trigger()
    assert len(window.figure_model.series) == series_before


def test_remove_fitted_curve_removes_only_the_matching_result_series(qapp):
    window, xrd = _xrd_window()
    # an unrelated user series on the panel
    other = PlotSeries.line(_pattern_dataset(seed=2), "2theta", "intensity", label="other")
    window._on_add_to_plot([other])
    n_after_other = len(window.figure_model.series)

    xrd.fit_peak_combo.setCurrentIndex(0)
    xrd._on_fit_peak_clicked()
    xrd.add_fitted_curve_button.click()
    assert len(window.figure_model.series) == n_after_other + 1

    xrd.remove_fitted_curve_button.click()
    assert len(window.figure_model.series) == n_after_other
    assert any(s.label == "other" for s in window.figure_model.series)
    assert xrd.add_fitted_curve_button.isEnabled() is True


# =====================================================================
# K. residual_x_range contract
# =====================================================================


def test_xrd_fit_result_residual_range_is_the_fit_window_generic_is_none(qapp):
    _figure, section = _bare_section()
    result = _select_first_peak_and_fit(section)
    assert result.residual_x_range() == result.fit_window

    generic = fit_curve([0, 1, 2, 3], [0, 1, 2, 3], LINEAR, source_dataset_id="d", x_column="x", y_column="y")
    assert generic.residual_x_range() is None


def test_residual_view_clips_to_the_fit_window(qapp):
    window, xrd = _xrd_window()
    xrd.fit_peak_combo.setCurrentIndex(0)
    xrd._on_fit_peak_clicked()
    result = xrd.current_fit_result()

    view = window.analysis_result_view
    view._on_view_residuals_clicked()
    residuals = view._residual_window._plot._axes.collections[0].get_offsets()
    xs = np.asarray(residuals)[:, 0]
    assert xs.min() >= result.fit_window[0] - 1e-9
    assert xs.max() <= result.fit_window[1] + 1e-9
    assert len(xs) < len(window.dataset_manager.datasets[0].dataframe)  # not the whole pattern


# =====================================================================
# L. history selection routing / load_fit_result
# =====================================================================


def test_selecting_a_fit_history_row_restores_the_controls_without_a_new_result(qapp):
    window, xrd = _xrd_window()
    panel_id = window.figure_model.active_panel.id

    xrd.fit_peak_combo.setCurrentIndex(0)
    xrd.fit_model_combo.setCurrentIndex(xrd.fit_model_combo.findData(PSEUDO_VOIGT))
    xrd._on_fit_peak_clicked()
    fit_result = xrd.current_fit_result()

    # run a detection so the current history row is no longer the fit
    xrd._on_find_peaks_clicked()
    assert xrd.current_fit_result() is None
    history = window._project.active_workbench.analysis_results.all(panel_id)
    n_history = len(history)

    window.analysis_panel.history_list.setCurrentRow(history.index(fit_result))

    assert window.analysis_panel.tool_combo.currentText() == "XRD Peak Analysis"
    assert xrd.current_fit_result() is fit_result
    assert xrd.fit_model_combo.currentData() == PSEUDO_VOIGT
    assert xrd.fit_min_spin.value() == pytest.approx(fit_result.fit_window[0])
    assert xrd.fit_max_spin.value() == pytest.approx(fit_result.fit_window[1])
    # no new history entry created by loading
    assert len(window._project.active_workbench.analysis_results.all(panel_id)) == n_history
    fit_window, fit_curves = xrd.fit_overlay()
    assert fit_curves is not None


# =====================================================================
# M. project reopen
# =====================================================================


def test_xrd_peak_fit_survives_project_save_and_reopen(qapp, tmp_path):
    window, xrd = _xrd_window()
    panel_id = window.figure_model.active_panel.id
    xrd.fit_peak_combo.setCurrentIndex(0)
    xrd._on_fit_peak_clicked()
    fit_result = xrd.current_fit_result()

    out = save_project(window._project, tmp_path / "fit.gnovi")
    reloaded = load_project(out)
    restored = reloaded.workbenches[0].analysis_results.all(panel_id)
    assert any(isinstance(r, XRDPeakFitResult) and r.result_id == fit_result.result_id for r in restored)
    window.close()


# =====================================================================
# N / O. errors and warnings
# =====================================================================


def test_fit_error_is_shown_in_the_status_label_not_a_modal(qapp, monkeypatch):
    import gnovi_plot.gui.widgets.xrd_analysis_section as mod

    calls = []
    monkeypatch.setattr(mod.QMessageBox, "critical", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(mod.QMessageBox, "warning", lambda *a, **k: calls.append(a))

    window, xrd = _xrd_window()
    panel_id = window.figure_model.active_panel.id
    n_history = len(window._project.active_workbench.analysis_results.all(panel_id))

    xrd.fit_peak_combo.setCurrentIndex(0)
    # a 3-point window is far below the max(2P, 10) floor -> XRDFitError
    centre = xrd._selected_fit_seed().two_theta
    xrd.fit_min_spin.setValue(centre - 0.002)
    xrd.fit_max_spin.setValue(centre + 0.002)
    xrd._on_fit_peak_clicked()

    assert calls == []
    assert "Fit failed" in xrd.fit_status_label.text()
    assert xrd.current_fit_result() is None
    assert len(window._project.active_workbench.analysis_results.all(panel_id)) == n_history


def test_all_fit_warnings_appear_as_result_rows(qapp):
    _figure, section = _bare_section()
    # a neighbouring detected peak inside the window -> a "caution" warning
    section.fit_peak_combo.setCurrentIndex(0)
    section.fit_min_spin.setValue(28.0)
    section.fit_max_spin.setValue(47.0)  # spans the 45 deg reflection too
    result = _select_first_peak_and_fit(section)
    assert result.warnings
    caution_rows = [value for label, value in result.details() if label == "Caution"]
    assert len(caution_rows) == len(result.warnings)
    assert "caution" in section.fit_status_label.text().lower()


# =====================================================================
# P. radiation optional
# =====================================================================


def test_fit_without_radiation_succeeds_with_d_spacing_unavailable(qapp):
    # Detection needs radiation (XRD-2 behaviour); fitting does not. Detect
    # with a radiation, then clear it before fitting.
    _figure, section = _bare_section()
    section.radiation_combo.setCurrentIndex(0)  # "Select radiation…" -> None
    assert section._radiation is None
    result = _select_first_peak_and_fit(section)
    assert isinstance(result, XRDPeakFitResult)
    assert result.d_spacing is None
    assert any("no radiation context" in value for _label, value in result.details())


# =====================================================================
# Q. overlay is GUI-only / never exported
# =====================================================================


def test_fit_overlay_artists_are_gui_only_and_cleared_by_the_export_boundary(qapp):
    window, xrd = _xrd_window()
    xrd.fit_peak_combo.setCurrentIndex(0)
    xrd._on_fit_peak_clicked()
    window._refresh_xrd_overlay()

    assert len(window.plot_canvas._analysis_overlay_artists) > 0
    # none of the overlay artists carry a legend label
    for artist in window.plot_canvas._analysis_overlay_artists:
        label = getattr(artist, "get_label", lambda: "")()
        assert not label or label.startswith("_")

    window.plot_canvas.clear_gui_only_overlays()
    assert window.plot_canvas._analysis_overlay_artists == []
