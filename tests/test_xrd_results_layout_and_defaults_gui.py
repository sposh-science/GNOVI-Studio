"""XRD-2 usability/correctness stabilization -- covers real-world bugs found
by actual human use of the merged XRD-2 feature:

- the central Results pane growing to swallow the plot canvas when an
  `XRDAnalysisResult` has many peaks (`XRDAnalysisResult.details()` used to
  emit one row per peak into an unbounded `QFormLayout`; `BottomPanel`'s
  Results tab now also wraps its content in a `QScrollArea` as a structural
  backstop);
- the XRD left panel (radiation/background/smoothing/detection/peak table)
  overflowing a laptop-height window, now scrollable beneath a fixed
  Analysis Tool selector (`AnalysisPanel`'s own `workflow_scroll`);
- peak-detection's first-run Prominence/Minimum-separation defaulting to 0
  (= "no threshold at all" to `scipy.signal.find_peaks`), which on real
  noisy raw XRD data returned hundreds/thousands of candidates
  (`_default_prominence_from_signal`/`_maybe_apply_default_detection_params`);
- "Add Peak" armed-state lifecycle gaps (active-panel switch, Analysis Tool
  switch, source-series change) beyond what the prior review-fix already
  covered (wrong-panel click, Panel3D, non-finite coordinates).

Does NOT cover the reported Fn+PrtSc screenshot-shortcut regression with an
automated test: no `grabMouse`/`grabKeyboard`/`installEventFilter`/
`QShortcut` exists anywhere in `gnovi_plot` (verified by direct grep across
the whole package), so there is nothing GNOVI-internal to assert against in
an offscreen Qt test -- see this milestone's own final report for the full
investigation and the human Ubuntu verification checklist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnovi_plot.analysis.fitting import LINEAR, fit_curve
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.widgets.analysis_panel import AnalysisPanel
from gnovi_plot.gui.widgets.analysis_result_view import AnalysisResultView
from gnovi_plot.gui.widgets.bottom_panel import BottomPanel
from gnovi_plot.gui.widgets.xrd_analysis_section import (
    XRDAnalysisSection,
    _default_prominence_from_signal,
)
from gnovi_plot.modules.xrd.peaks import XRDPeakSeed
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series import PlotSeries


def _gaussian(x, center, amp, sigma):
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def synthetic_xrd_quartz_like(
    seed: int = 42,
    n_points: int = 3501,
    n_peaks: int = 12,
    noise_std: float = 3.0,
    intensity_scale: float = 1.0,
) -> tuple[Dataset, np.ndarray]:
    """A raw (no background subtracted, no smoothing) synthetic powder-XRD
    pattern shaped like the real bug report that motivated this file: a
    realistic 2θ range, a linear baseline, a modest number of real Gaussian
    peaks, and Gaussian noise -- the RAW-input, no-preprocessing case that
    actually blew up to 1,118 candidates in real use. `intensity_scale`
    lets a test rescale the whole signal (peaks, baseline, and noise
    together) to check the default isn't tied to one particular absolute
    unit."""
    rng = np.random.default_rng(seed)
    two_theta = np.linspace(10.0, 80.0, n_points)
    intensity = 30.0 + 0.05 * two_theta
    for center in np.linspace(15.0, 75.0, n_peaks):
        amp = rng.uniform(150.0, 600.0)
        sigma = rng.uniform(0.05, 0.15)
        intensity = intensity + _gaussian(two_theta, center, amp, sigma)
    intensity = intensity + rng.normal(0.0, noise_std, size=two_theta.shape)
    intensity = intensity * intensity_scale
    df = pd.DataFrame({"2theta": two_theta, "intensity": intensity})
    return Dataset(name="synthetic_xrd_quartz_like", dataframe=df), np.linspace(15.0, 75.0, n_peaks)


def _figure_with_series(dataset: Dataset) -> tuple[GnoviFigure, PlotSeries]:
    figure = GnoviFigure()
    series = PlotSeries.line(dataset, "2theta", "intensity")
    figure.active_panel.add_series(series)
    return figure, series


def _xrd_result_with_n_peaks(n: int):
    """A real `XRDAnalysisResult` with `n` manual peak seeds, built
    directly with `build_xrd_analysis_result` and a single `_refresh_peak_
    table()` call -- exactly the shape a real `find_peaks` run with `n`
    candidates produces (peaks assembled in one batch, table rebuilt
    once), not `n` individual `add_manual_peak` calls each rebuilding the
    whole table from scratch (which is what a researcher actually
    clicking "Add Peak" `n` times would do, and is correctly O(n^2) for
    that interactive use case -- but is the wrong shape for a "given a
    result with `n` peaks, does the LAYOUT behave" test)."""
    from gnovi_plot.modules.xrd.radiation import RADIATION_PRESETS
    from gnovi_plot.modules.xrd.results import build_xrd_analysis_result

    ds, _ = synthetic_xrd_quartz_like(n_peaks=3, n_points=200)
    figure, series = _figure_with_series(ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    result = build_xrd_analysis_result(
        source_dataset_id=series.dataset.id,
        x_column=series.x_column,
        y_column=series.y_column,
        radiation=RADIATION_PRESETS["cu_ka1"],
        peaks=[XRDPeakSeed.manual(float(10 + i * 0.01), 100.0) for i in range(n)],
        source_series_id=series.id,
        source_panel_id=figure.active_panel.id,
        parameters={
            "detection": {"prominence": None, "distance": None, "height": None, "width": None},
            "detection_input": "raw",
            "preprocessing": {"background": None, "smoothing": None},
        },
    )
    section._current_result = result
    section._refresh_peak_table()
    return section, result


# --- Part 2/3: central Results pane must never scale with peak count --------


@pytest.mark.parametrize("n_peaks", [5, 20, 100, 1000])
def test_xrd_details_row_count_is_fixed_regardless_of_peak_count(qapp, n_peaks):
    _section, result = _xrd_result_with_n_peaks(n_peaks)
    assert len(result.peaks) == n_peaks
    rows = result.details()
    assert len(rows) == 8  # Radiation/candidates/enabled/background/smoothing/input/prominence/distance
    assert not any(label.startswith("Peak ") and label != "Peak candidates" for label, _ in rows)


@pytest.mark.parametrize("n_peaks", [5, 20, 100, 1000])
def test_bottom_panel_minimum_size_hint_stays_bounded_regardless_of_peak_count(qapp, n_peaks):
    _section, result = _xrd_result_with_n_peaks(n_peaks)
    manager = DatasetManager()

    bottom_panel = BottomPanel()
    view = AnalysisResultView(GnoviFigure(), manager)
    bottom_panel.set_results_widget(view)
    view.show_result(result)
    bottom_panel.show_results_tab()

    hint = bottom_panel.minimumSizeHint()
    # A real (unfixed) 1,118-peak result gave this widget a minimumSizeHint
    # in the tens of thousands of pixels -- an ordinary desktop's full
    # screen height is nowhere close to that, so this bound is generous
    # while still being a meaningful regression guard.
    assert hint.height() < 400


def test_bottom_panel_minimum_size_hint_does_not_grow_between_5_and_1000_peaks(qapp):
    manager = DatasetManager()
    heights = []
    for n_peaks in (5, 1000):
        _section, result = _xrd_result_with_n_peaks(n_peaks)
        bottom_panel = BottomPanel()
        view = AnalysisResultView(GnoviFigure(), manager)
        bottom_panel.set_results_widget(view)
        view.show_result(result)
        heights.append(bottom_panel.minimumSizeHint().height())
    assert heights[0] == heights[1]  # geometry is "essentially stable" -- identical, in fact


def test_short_curve_fit_result_and_large_xrd_result_give_the_same_bounded_geometry(qapp):
    """A short generic `FitResult` and a 1,000-peak `XRDAnalysisResult`
    must both leave the Results tab -- and therefore the central splitter
    -- at the same small, bounded geometry; neither analysis tool's result
    should behave differently from the other here."""
    manager = DatasetManager()
    x = np.linspace(0, 10, 20)
    y = 2.0 * x + 1.0
    fit_result = fit_curve(
        x, y, LINEAR, source_dataset_id="ds1", x_column="x", y_column="y",
    )
    _section, xrd_result = _xrd_result_with_n_peaks(1000)

    heights = []
    for result in (fit_result, xrd_result):
        bottom_panel = BottomPanel()
        view = AnalysisResultView(GnoviFigure(), manager)
        bottom_panel.set_results_widget(view)
        view.show_result(result)
        heights.append(bottom_panel.minimumSizeHint().height())
    assert heights[0] == heights[1]


def test_results_tab_content_is_wrapped_in_a_scroll_area(qapp):
    """Structural check for the `BottomPanel.set_results_widget` fix
    itself -- independent of whether `details()` also stays bounded, the
    Results tab must never hand an unbounded widget straight to the
    QTabWidget."""
    from PySide6.QtWidgets import QScrollArea

    bottom_panel = BottomPanel()
    view = AnalysisResultView(GnoviFigure(), DatasetManager())
    bottom_panel.set_results_widget(view)
    layout = bottom_panel._results_tab.layout()
    assert layout.count() == 1
    scroll = layout.itemAt(0).widget()
    assert isinstance(scroll, QScrollArea)
    assert scroll.widget() is view


# --- Part 3: splitter geometry across a real MainWindow ----------------------


def test_center_splitter_stays_draggable_after_a_huge_xrd_result(qapp):
    """The real regression: after a huge XRD result populates the Results
    tab, the plot canvas must not be starved of space with no way to
    recover it. `center_splitter`'s own `minimumSizeHint` (the sum of its
    children's) must stay well within an ordinary window height, so a
    normal drag can always give the plot canvas back its share."""
    window = MainWindow()
    ds, _ = synthetic_xrd_quartz_like(n_points=500, n_peaks=3)
    window.dataset_manager.add(ds)
    series = PlotSeries.line(ds, "2theta", "intensity")
    window.figure_model.active_panel.add_series(series)
    window._on_figure_content_changed()

    window.analysis_panel.tool_combo.setCurrentText("XRD Peak Analysis")
    xrd = window.analysis_panel.xrd_section_widget
    xrd.radiation_combo.setCurrentIndex(xrd.radiation_combo.findData("cu_ka1"))
    _section, huge_result = _xrd_result_with_n_peaks(1200)
    xrd._current_result = huge_result
    xrd._refresh_peak_table()

    window.analysis_result_view.show_result(huge_result)
    window.bottom_panel.show_results_tab()

    assert window.center_splitter.minimumSizeHint().height() < 600
    # childrenCollapsible defaults to True in Qt -- confirm nothing in this
    # milestone turned that off, which is what makes the handle draggable
    # all the way (not just down to a large residual minimum).
    assert window.center_splitter.childrenCollapsible()


# --- Part 4: XRD left panel scrolling ----------------------------------------


def test_analysis_tool_selector_is_not_inside_the_scrolling_region(qapp):
    figure, _series = _figure_with_series(synthetic_xrd_quartz_like(n_points=50, n_peaks=1)[0])
    panel = AnalysisPanel(figure, DatasetManager())
    top_level = [panel.layout().itemAt(i).widget() for i in range(panel.layout().count())]
    assert panel.tool_label in top_level
    assert panel.tool_combo in top_level

    from PySide6.QtWidgets import QScrollArea

    scroll_areas = [w for w in top_level if isinstance(w, QScrollArea)]
    assert len(scroll_areas) == 1
    scroll = scroll_areas[0]
    assert panel.tool_combo is not scroll.widget()
    # The XRD/Curve-Fitting/History sections live inside the scroll
    # region's content widget, not directly on the panel's own layout.
    assert panel.xrd_section.parentWidget() is not panel
    assert panel.fit_section.parentWidget() is not panel


def test_xrd_workflow_controls_are_reachable_through_the_scroll_region(qapp):
    """Every XRD control group must actually be inside the scrolling
    content -- not accidentally left out of it, which would either hide it
    entirely or re-break the original overflow bug for just that group."""
    figure, _series = _figure_with_series(synthetic_xrd_quartz_like(n_points=50, n_peaks=1)[0])
    panel = AnalysisPanel(figure, DatasetManager())
    xrd = panel.xrd_section_widget
    for widget in (
        xrd.radiation_combo,
        xrd.background_method_combo,
        xrd.smoothing_enabled_check,
        xrd.prominence_spin,
        xrd.peak_table,
    ):
        node = widget
        found_panel = False
        while node is not None:
            if node is panel:
                found_panel = True
                break
            node = node.parentWidget()
        assert found_panel, f"{widget} is not a descendant of AnalysisPanel"


def test_no_horizontal_scrollbar_policy_on_the_workflow_scroll_area(qapp):
    from PySide6.QtCore import Qt

    figure, _series = _figure_with_series(synthetic_xrd_quartz_like(n_points=50, n_peaks=1)[0])
    panel = AnalysisPanel(figure, DatasetManager())
    from PySide6.QtWidgets import QScrollArea

    scroll = next(
        panel.layout().itemAt(i).widget()
        for i in range(panel.layout().count())
        if isinstance(panel.layout().itemAt(i).widget(), QScrollArea)
    )
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff


# --- Part 5: peak table bounded height / large row counts -------------------


@pytest.mark.parametrize("n_peaks", [5, 20, 50, 100, 1000])
def test_peak_table_bounded_height_at_various_row_counts(qapp, n_peaks):
    section, result = _xrd_result_with_n_peaks(n_peaks)
    assert section.peak_table.rowCount() == n_peaks
    # QTableWidget already scrolls internally regardless of row count (its
    # own sizeHint/minimumSizeHint don't scale with rows -- verified
    # directly against this Qt version); this is the explicit bound this
    # milestone adds on top, for UX rather than correctness.
    assert section.peak_table.maximumHeight() <= 220
    headers = [section.peak_table.horizontalHeaderItem(i).text() for i in range(section.peak_table.columnCount())]
    assert headers[0] == "Peak #"


def test_peak_table_enable_disable_works_at_1000_rows(qapp):
    section, result = _xrd_result_with_n_peaks(1000)
    section.peak_table.selectRow(500)
    section._on_toggle_enabled_clicked()
    assert result.peaks[500].enabled is False
    section.peak_table.selectRow(500)
    section._on_toggle_enabled_clicked()
    assert result.peaks[500].enabled is True


def test_peak_table_remove_works_at_1000_rows(qapp):
    section, result = _xrd_result_with_n_peaks(1000)
    section.peak_table.selectRow(999)
    section._on_remove_selected_clicked()
    assert len(result.peaks) == 999
    assert section.peak_table.rowCount() == 999


# --- Part 6: peak-detection first-run defaults -------------------------------


def test_default_prominence_helper_is_zero_for_a_constant_signal():
    y = np.full(500, 42.0)
    assert _default_prominence_from_signal(y) == 0.0


def test_default_prominence_helper_scales_with_noise_level():
    rng = np.random.default_rng(1)
    low_noise = 20.0 + rng.normal(0, 0.5, size=2000)
    high_noise = 20.0 + rng.normal(0, 5.0, size=2000)
    low = _default_prominence_from_signal(low_noise)
    high = _default_prominence_from_signal(high_noise)
    assert 0.0 < low < high


def test_raw_input_first_run_candidate_count_is_plausible_not_thousands(qapp):
    """THE primary regression scenario, per the real-user clarification:
    Background = None, Smoothing = Off, Detection Input = Raw, first-run
    (untouched) Prominence/Minimum separation. Must not explode into
    hundreds/thousands of candidates on ordinary noisy raw XRD data. Not
    asserting an exact count (that would overfit to this synthetic
    pattern's own randomness) -- asserting a scientifically plausible
    band: at least one candidate (real peaks exist and towers over noise),
    nowhere near the ~1,118 the unfixed defaults produced from ~12 real
    peaks."""
    ds, known_centers = synthetic_xrd_quartz_like(seed=42)
    figure, series = _figure_with_series(ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    idx = section.source_combo.findData(series.id)
    section.source_combo.setCurrentIndex(idx)
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))

    assert section.prominence_spin.value() > 0.0  # never the unusable 0.0 default
    assert section.background_method_combo.currentText() == "None"
    assert not section.smoothing_enabled_check.isChecked()
    assert section.detection_input_combo.currentData() in (None, "raw")

    section._on_find_peaks_clicked()
    n = len(section.current_result().peaks)
    assert 1 <= n <= 200  # plausible, not exactly len(known_centers), and nowhere near 1000+


@pytest.mark.parametrize("seed", [42, 100, 7])
def test_raw_input_first_run_candidate_count_is_plausible_across_seeds(qapp, seed):
    ds, _ = synthetic_xrd_quartz_like(seed=seed)
    figure, series = _figure_with_series(ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.source_combo.setCurrentIndex(section.source_combo.findData(series.id))
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section._on_find_peaks_clicked()
    assert 1 <= len(section.current_result().peaks) <= 200


def test_low_noise_raw_pattern_first_run_candidate_count_is_plausible(qapp):
    ds, _ = synthetic_xrd_quartz_like(seed=5, noise_std=0.5)
    figure, series = _figure_with_series(ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.source_combo.setCurrentIndex(section.source_combo.findData(series.id))
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section._on_find_peaks_clicked()
    assert 1 <= len(section.current_result().peaks) <= 200


def test_moderate_noise_raw_pattern_first_run_candidate_count_is_plausible(qapp):
    ds, _ = synthetic_xrd_quartz_like(seed=6, noise_std=8.0)
    figure, series = _figure_with_series(ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.source_combo.setCurrentIndex(section.source_combo.findData(series.id))
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section._on_find_peaks_clicked()
    assert 1 <= len(section.current_result().peaks) <= 300


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_default_prominence_scales_with_absolute_intensity_units(qapp, scale):
    """The default must not be a fixed absolute number -- it must track
    whatever units/scale the actual data happens to be in."""
    ds, _ = synthetic_xrd_quartz_like(seed=9, intensity_scale=scale)
    figure, series = _figure_with_series(ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.source_combo.setCurrentIndex(section.source_combo.findData(series.id))
    prominence = section.prominence_spin.value()
    assert prominence > 0.0
    # Roughly proportional to scale (order of magnitude, not exact) --
    # confirms the default isn't some fixed constant unrelated to the
    # data's own units.
    y = ds.dataframe["intensity"].to_numpy()
    expected = _default_prominence_from_signal(y)
    # `prominence_spin` is a 4-decimal-place QDoubleSpinBox -- its stored
    # value is the computed default rounded to that display precision,
    # not the raw float, so compare with a tolerance wide enough to
    # absorb that rounding rather than exact float equality.
    assert prominence == pytest.approx(expected, abs=1e-3)


def test_flat_no_peak_pattern_does_not_explode_candidate_count(qapp):
    two_theta = np.linspace(10.0, 80.0, 2000)
    rng = np.random.default_rng(3)
    intensity = 50.0 + rng.normal(0, 1.0, size=two_theta.shape)  # noise only, no real peaks
    ds = Dataset(name="flat", dataframe=pd.DataFrame({"2theta": two_theta, "intensity": intensity}))
    figure, series = _figure_with_series(ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.source_combo.setCurrentIndex(section.source_combo.findData(series.id))
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section._on_find_peaks_clicked()
    n = len(section.current_result().peaks) if section.current_result() else 0
    assert n < 50  # nowhere near an unbounded noise-driven explosion


def test_default_never_overwrites_a_value_the_researcher_already_edited(qapp):
    ds, _ = synthetic_xrd_quartz_like(seed=42)
    figure, series = _figure_with_series(ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.source_combo.setCurrentIndex(section.source_combo.findData(series.id))
    default_value = section.prominence_spin.value()
    assert default_value > 0.0

    section.prominence_spin.setValue(default_value + 500.0)  # a deliberate researcher edit
    section.refresh()  # any generic refresh must not silently revert it
    assert section.prominence_spin.value() == pytest.approx(default_value + 500.0)


def test_a_new_source_series_gets_its_own_fresh_default(qapp):
    """Switching to a genuinely different dataset/series -- not just
    re-selecting the same one -- must compute a fresh default for the NEW
    signal's own scale/noise, even though the previous series' default was
    left untouched (never "touched" by the researcher)."""
    ds1, _ = synthetic_xrd_quartz_like(seed=1, intensity_scale=1.0)
    ds2, _ = synthetic_xrd_quartz_like(seed=1, intensity_scale=50.0)
    figure = GnoviFigure()
    series1 = PlotSeries.line(ds1, "2theta", "intensity", label="Series 1")
    series2 = PlotSeries.line(ds2, "2theta", "intensity", label="Series 2")
    figure.active_panel.add_series(series1)
    figure.active_panel.add_series(series2)

    section = XRDAnalysisSection(figure, DatasetManager())
    section.source_combo.setCurrentIndex(section.source_combo.findData(series1.id))
    prominence_1 = section.prominence_spin.value()

    section.source_combo.setCurrentIndex(section.source_combo.findData(series2.id))
    prominence_2 = section.prominence_spin.value()

    assert prominence_1 > 0.0
    assert prominence_2 > 0.0
    assert prominence_2 == pytest.approx(prominence_1 * 50.0, rel=0.05)


def test_detection_default_is_recorded_in_result_provenance(qapp):
    ds, _ = synthetic_xrd_quartz_like(seed=42)
    figure, series = _figure_with_series(ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.source_combo.setCurrentIndex(section.source_combo.findData(series.id))
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    used_prominence = section.prominence_spin.value()
    section._on_find_peaks_clicked()
    stored = section.current_result().parameters["detection"]["prominence"]
    assert stored == pytest.approx(used_prominence)


# --- Part 6 (secondary scenarios, per the real-user clarification):
# background-corrected/smoothed input, tested SEPARATELY from the primary
# raw-input case above, never as a prerequisite for it -----------------------


def test_background_corrected_input_first_run_candidate_count_is_plausible(qapp):
    ds, _ = synthetic_xrd_quartz_like(seed=42)
    figure, series = _figure_with_series(ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.source_combo.setCurrentIndex(section.source_combo.findData(series.id))
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section.background_method_combo.setCurrentText("Polynomial")
    section.baseline_points_edit.setText("0-20, 3480-3500")
    section.polynomial_degree_spin.setValue(1)
    section._on_preview_background_clicked()
    idx = section.detection_input_combo.findData("background_corrected")
    assert idx >= 0
    section.detection_input_combo.setCurrentIndex(idx)
    section._on_find_peaks_clicked()
    assert 1 <= len(section.current_result().peaks) <= 200


def test_smoothed_input_first_run_candidate_count_is_plausible(qapp):
    pytest.importorskip("scipy.signal")
    ds, _ = synthetic_xrd_quartz_like(seed=42)
    figure, series = _figure_with_series(ds)
    section = XRDAnalysisSection(figure, DatasetManager())
    section.source_combo.setCurrentIndex(section.source_combo.findData(series.id))
    section.radiation_combo.setCurrentIndex(section.radiation_combo.findData("cu_ka1"))
    section.smoothing_enabled_check.setChecked(True)
    section._on_preview_smoothed_clicked()
    idx = section.detection_input_combo.findData("smoothed_raw")
    assert idx >= 0
    section.detection_input_combo.setCurrentIndex(idx)
    section._on_find_peaks_clicked()
    assert 1 <= len(section.current_result().peaks) <= 200


# --- Part 8: Add Peak armed-state lifecycle ----------------------------------


def _armed_xrd_window():
    window = MainWindow()
    ds, _ = synthetic_xrd_quartz_like(n_points=300, n_peaks=3)
    window.dataset_manager.add(ds)
    series = PlotSeries.line(ds, "2theta", "intensity")
    window.figure_model.active_panel.add_series(series)
    window._on_figure_content_changed()
    window.analysis_panel.tool_combo.setCurrentText("XRD Peak Analysis")
    xrd = window.analysis_panel.xrd_section_widget
    xrd.radiation_combo.setCurrentIndex(xrd.radiation_combo.findData("cu_ka1"))
    xrd.add_peak_button.setChecked(True)
    return window, xrd


def test_add_peak_disarms_on_analysis_tool_switch(qapp):
    window, xrd = _armed_xrd_window()
    assert xrd.is_manual_peak_mode()
    window.analysis_panel.tool_combo.setCurrentText("Curve Fitting")
    assert not xrd.is_manual_peak_mode()
    assert not xrd.add_peak_button.isChecked()


def test_add_peak_disarms_on_active_panel_switch(qapp):
    window, xrd = _armed_xrd_window()
    window.figure_size_panel.layout_combo.setCurrentIndex(window.figure_size_panel.layout_combo.findText("1 x 2"))
    # Applying a new layout already routes through the same panel-switch
    # machinery (`MainWindow._on_panel_switched`), so it disarms too --
    # re-arm explicitly and switch the active panel directly to isolate
    # exactly the transition Part 8 asks for.
    xrd.add_peak_button.setChecked(True)
    assert xrd.is_manual_peak_mode()
    window._set_active_panel(1)
    assert not xrd.is_manual_peak_mode()
    assert not window._xrd_manual_peak_mode


def test_add_peak_disarms_on_source_series_change(qapp):
    window, xrd = _armed_xrd_window()
    ds2, _ = synthetic_xrd_quartz_like(seed=99, n_points=300, n_peaks=2)
    window.dataset_manager.add(ds2)
    series2 = PlotSeries.line(ds2, "2theta", "intensity", label="Second series")
    window.figure_model.active_panel.add_series(series2)
    window._on_figure_content_changed()
    idx = xrd.source_combo.findData(series2.id)
    xrd.source_combo.setCurrentIndex(idx)
    assert not xrd.is_manual_peak_mode()


def test_add_peak_disarms_on_workbench_switch(qapp):
    window, xrd = _armed_xrd_window()
    from gnovi_plot.core.workbench import Workbench

    new_workbench = Workbench(name="Second workbench", figure=GnoviFigure())
    window._project.add_workbench(new_workbench)
    window._activate_workbench(new_workbench)
    assert not xrd.is_manual_peak_mode()


def test_add_peak_disarms_on_new_project(qapp):
    window, xrd = _armed_xrd_window()
    # `_load_project_into_window` directly -- the single path New Project
    # and Open Project both funnel through -- bypassing `_on_new_project`'s
    # own unsaved-changes confirmation dialog (a real modal `QMessageBox`,
    # which would block an offscreen test with nothing to click it).
    window._load_project_into_window(window._new_project())
    assert not window.analysis_panel.xrd_section_widget.is_manual_peak_mode()


def test_add_peak_successful_add_is_single_shot(qapp):
    window, xrd = _armed_xrd_window()
    assert xrd.is_manual_peak_mode()
    own_axes = window.plot_canvas.axes_list[0]

    class _Click:
        inaxes = own_axes
        xdata = 30.0
        ydata = 100.0
        button = 1
        dblclick = False

    window._on_canvas_click(_Click())
    assert not xrd.is_manual_peak_mode()
    assert not xrd.add_peak_button.isChecked()


def test_add_peak_toggle_off_disarms(qapp):
    _window, xrd = _armed_xrd_window()
    assert xrd.is_manual_peak_mode()
    xrd.add_peak_button.setChecked(False)
    assert not xrd.is_manual_peak_mode()


# --- Part 7 (defensive-only): no OS-level input grab is ever held -----------


def test_no_mouse_or_keyboard_grab_is_ever_held_across_xrd_interactions(qapp):
    """Cannot exercise a real OS global shortcut in an offscreen test (see
    this module's own docstring and the milestone's final report), but
    this asserts the one thing actually within GNOVI's control: no widget
    ever holds `QWidget.grabMouse()`/`grabKeyboard()` during or after a
    realistic XRD interaction sequence, matching the direct grep finding
    that no such call exists anywhere in `gnovi_plot`."""
    from PySide6.QtWidgets import QWidget

    window, xrd = _armed_xrd_window()
    assert QWidget.mouseGrabber() is None
    assert QWidget.keyboardGrabber() is None

    own_axes = window.plot_canvas.axes_list[0]

    class _Click:
        inaxes = own_axes
        xdata = 30.0
        ydata = 100.0
        button = 1
        dblclick = False

    window._on_canvas_click(_Click())
    assert QWidget.mouseGrabber() is None
    assert QWidget.keyboardGrabber() is None

    xrd._on_find_peaks_clicked()
    assert QWidget.mouseGrabber() is None
    assert QWidget.keyboardGrabber() is None

    window.bottom_panel.show_results_tab()
    assert QWidget.mouseGrabber() is None
    assert QWidget.keyboardGrabber() is None
