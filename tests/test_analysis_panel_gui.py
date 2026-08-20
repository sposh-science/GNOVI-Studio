from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QMessageBox

from gnovi_plot.analysis.fitting import EXPONENTIAL, GAUSSIAN, LINEAR, POLYNOMIAL
from gnovi_plot.analysis.results import AnalysisResult
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.analysis_panel import AnalysisPanel
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries, PlotType


def _dataset(name="d", x=None, y=None):
    x = list(range(10)) if x is None else x
    y = [2 * v + 1 for v in x] if y is None else y
    return Dataset(name=name, dataframe=pd.DataFrame({"x": x, "y": y}))


def _capture_results(panel: AnalysisPanel) -> list[AnalysisResult]:
    captured: list[AnalysisResult] = []
    panel.analysis_result_ready.connect(captured.append)
    return captured


# --- Source-series population -------------------------------------------------


def test_source_combo_lists_line_and_scatter_series_in_the_active_panel(qapp):
    figure = GnoviFigure()
    ds = _dataset()
    line = PlotSeries.line(ds, "x", "y", label="Line series")
    scatter = PlotSeries.scatter(ds, "x", "y", label="Scatter series")
    figure.add_series(line)
    figure.add_series(scatter)

    panel = AnalysisPanel(figure, DatasetManager())

    labels = [panel.source_combo.itemText(i) for i in range(panel.source_combo.count())]
    assert labels == ["Line series", "Scatter series"]
    assert panel.source_combo.itemData(0) == line.id
    assert panel.source_combo.itemData(1) == scatter.id


def test_source_combo_excludes_histogram_series(qapp):
    figure = GnoviFigure()
    ds = _dataset()
    figure.add_series(PlotSeries.line(ds, "x", "y", label="Fittable"))
    figure.add_series(PlotSeries.histogram(ds, "y", label="Not fittable"))

    panel = AnalysisPanel(figure, DatasetManager())

    labels = [panel.source_combo.itemText(i) for i in range(panel.source_combo.count())]
    assert labels == ["Fittable"]


def test_source_combo_excludes_stale_series(qapp):
    figure = GnoviFigure()
    ds = _dataset()
    good = PlotSeries.line(ds, "x", "y", label="Good")
    stale = PlotSeries.line(ds, "x", "y", label="Stale one")
    stale.stale = True
    figure.add_series(good)
    figure.add_series(stale)

    panel = AnalysisPanel(figure, DatasetManager())

    labels = [panel.source_combo.itemText(i) for i in range(panel.source_combo.count())]
    assert labels == ["Good"]


def test_no_eligible_series_disables_run_fit_and_shows_status(qapp):
    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())

    assert panel.source_combo.count() == 0
    assert not panel.run_fit_button.isEnabled()
    assert panel.status_label.isVisibleTo(panel)


def test_eligible_series_enables_run_fit_and_hides_status(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))

    panel = AnalysisPanel(figure, DatasetManager())

    assert panel.run_fit_button.isEnabled()
    assert not panel.status_label.isVisibleTo(panel)


# --- Active-panel changes ------------------------------------------------


def test_refresh_reflects_the_active_panels_series_only(qapp):
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    figure.panels[0].add_series(PlotSeries.line(_dataset(), "x", "y", label="Panel 1 series"))
    figure.panels[1].add_series(PlotSeries.line(_dataset(), "x", "y", label="Panel 2 series"))
    figure.set_active_panel(0)

    panel = AnalysisPanel(figure, DatasetManager())
    assert [panel.source_combo.itemText(i) for i in range(panel.source_combo.count())] == [
        "Panel 1 series"
    ]

    figure.set_active_panel(1)
    panel.refresh()

    assert [panel.source_combo.itemText(i) for i in range(panel.source_combo.count())] == [
        "Panel 2 series"
    ]


def test_set_figure_repoints_and_reloads(qapp):
    figure_a = GnoviFigure()
    figure_a.add_series(PlotSeries.line(_dataset(), "x", "y", label="From A"))
    panel = AnalysisPanel(figure_a, DatasetManager())

    figure_b = GnoviFigure()
    figure_b.add_series(PlotSeries.line(_dataset(), "x", "y", label="From B"))
    panel.set_figure(figure_b)

    labels = [panel.source_combo.itemText(i) for i in range(panel.source_combo.count())]
    assert labels == ["From B"]


# --- Model selection / polynomial order -----------------------------------


def test_model_combo_offers_the_four_milestone_models(qapp):
    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())

    models = [panel.model_combo.itemData(i) for i in range(panel.model_combo.count())]
    assert models == [LINEAR, POLYNOMIAL, EXPONENTIAL, GAUSSIAN]


def test_polynomial_order_control_only_visible_for_polynomial_model(qapp):
    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())

    panel.model_combo.setCurrentIndex(panel.model_combo.findData(LINEAR))
    assert not panel.degree_spin.isVisibleTo(panel)

    panel.model_combo.setCurrentIndex(panel.model_combo.findData(POLYNOMIAL))
    assert panel.degree_spin.isVisibleTo(panel)

    panel.model_combo.setCurrentIndex(panel.model_combo.findData(GAUSSIAN))
    assert not panel.degree_spin.isVisibleTo(panel)


# --- Run Fit: success -------------------------------------------------------


def test_run_fit_emits_a_fit_result_for_linear_data(qapp):
    figure = GnoviFigure()
    x = list(range(20))
    y = [3.0 * v + 2.0 for v in x]
    ds = _dataset(x=x, y=y)
    series = PlotSeries.line(ds, "x", "y", label="Linear series")
    figure.add_series(series)

    panel = AnalysisPanel(figure, DatasetManager())
    results = _capture_results(panel)
    panel.model_combo.setCurrentIndex(panel.model_combo.findData(LINEAR))

    panel.run_fit_button.click()

    assert len(results) == 1
    result = results[0]
    assert result.model == LINEAR
    assert result.params["a"] == pytest.approx(3.0, abs=1e-6)
    assert result.params["b"] == pytest.approx(2.0, abs=1e-6)


def test_run_fit_uses_the_configured_polynomial_degree(qapp):
    figure = GnoviFigure()
    x = np.linspace(-5, 5, 30).tolist()
    y = [1.0 + 2.0 * v + 0.5 * v**2 for v in x]
    ds = _dataset(x=x, y=y)
    figure.add_series(PlotSeries.line(ds, "x", "y", label="Quadratic series"))

    panel = AnalysisPanel(figure, DatasetManager())
    results = _capture_results(panel)
    panel.model_combo.setCurrentIndex(panel.model_combo.findData(POLYNOMIAL))
    panel.degree_spin.setValue(2)

    panel.run_fit_button.click()

    assert len(results) == 1
    assert set(results[0].params.keys()) == {"c0", "c1", "c2"}


def test_fit_result_carries_stable_provenance_from_the_source_series(qapp):
    figure = GnoviFigure()
    ds = _dataset(name="my-dataset")
    series = PlotSeries.line(ds, "x", "y", label="Provenance series")
    figure.add_series(series)

    panel = AnalysisPanel(figure, DatasetManager())
    results = _capture_results(panel)
    panel.run_fit_button.click()

    result = results[0]
    assert result.source_dataset_id == ds.id
    assert result.source_series_id == series.id
    assert result.x_column == "x"
    assert result.y_column == "y"


# --- Run Fit: no plot side effects ------------------------------------------


def test_run_fit_creates_no_new_dataset_or_plot_series(qapp):
    figure = GnoviFigure()
    ds = _dataset()
    series = PlotSeries.line(ds, "x", "y", label="Only series")
    figure.add_series(series)

    panel = AnalysisPanel(figure, DatasetManager())
    _capture_results(panel)

    before = list(figure.series)
    panel.run_fit_button.click()
    after = list(figure.series)

    assert [s.id for s in after] == [s.id for s in before]
    assert len(after) == 1  # still just the original series -- no fit curve added


# --- Error handling ----------------------------------------------------------


def test_run_fit_with_no_selection_warns_and_emits_nothing(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warnings.append(a)))

    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())
    results = _capture_results(panel)

    # The button is disabled with no eligible series (covered by
    # test_no_eligible_series_disables_run_fit_and_shows_status); call the
    # guard directly to exercise "no selection" defensively, the same way a
    # future selection could still resolve to None.
    panel._on_run_fit_clicked()

    assert results == []
    assert len(warnings) == 1


def test_run_fit_with_insufficient_points_shows_critical_error(qapp, monkeypatch):
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a)))

    figure = GnoviFigure()
    ds = _dataset(x=[0, 1], y=[0, 1])  # too few points for any milestone model
    figure.add_series(PlotSeries.line(ds, "x", "y", label="Too short"))

    panel = AnalysisPanel(figure, DatasetManager())
    results = _capture_results(panel)
    panel.model_combo.setCurrentIndex(panel.model_combo.findData(GAUSSIAN))

    panel.run_fit_button.click()

    assert results == []
    assert len(errors) == 1
    assert "Curve Fitting" in errors[0]


def test_run_fit_with_non_numeric_data_shows_critical_error(qapp, monkeypatch):
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a)))

    figure = GnoviFigure()
    ds = Dataset(name="bad", dataframe=pd.DataFrame({"x": ["a", "b", "c"], "y": ["d", "e", "f"]}))
    figure.add_series(PlotSeries.line(ds, "x", "y", label="Non-numeric"))

    panel = AnalysisPanel(figure, DatasetManager())
    results = _capture_results(panel)

    panel.run_fit_button.click()

    assert results == []
    assert len(errors) == 1


def test_run_fit_non_convergent_model_shows_critical_error_not_crash(qapp, monkeypatch):
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a)))

    figure = GnoviFigure()
    # Flat data has no exponential curvature at all -- an inappropriate
    # model/data combination that should fail cleanly, not crash.
    ds = _dataset(x=list(range(10)), y=[5.0] * 10)
    figure.add_series(PlotSeries.line(ds, "x", "y", label="Flat"))

    panel = AnalysisPanel(figure, DatasetManager())
    results = _capture_results(panel)
    panel.model_combo.setCurrentIndex(panel.model_combo.findData(GAUSSIAN))
    panel.degree_spin.setValue(2)

    panel.run_fit_button.click()  # must not raise

    # Exactly one of "produced a result" / "showed a clean error" happened
    # -- never both, never a silent no-op, and (implicitly, since we got
    # this far) never an uncaught exception.
    assert (len(results) == 1) != (len(errors) == 1)


# --- Add Fit Curve to Plot: button enable state -------------------------


def test_add_fit_curve_button_disabled_before_any_fit(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    panel = AnalysisPanel(figure, DatasetManager())

    assert not panel.add_fit_curve_button.isEnabled()
    assert not panel.pending_fit_label.isVisibleTo(panel)


def test_add_fit_curve_button_enabled_after_successful_run_fit(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    panel = AnalysisPanel(figure, DatasetManager())

    panel.run_fit_button.click()

    assert panel.add_fit_curve_button.isEnabled()
    assert panel.pending_fit_label.isVisibleTo(panel)
    assert "linear fit" in panel.pending_fit_label.text()


def test_add_fit_curve_button_stays_disabled_after_a_failed_run_fit(qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))

    figure = GnoviFigure()
    ds = _dataset(x=[0, 1], y=[0, 1])  # too few points
    figure.add_series(PlotSeries.line(ds, "x", "y", label="Too short"))
    panel = AnalysisPanel(figure, DatasetManager())

    panel.run_fit_button.click()

    assert not panel.add_fit_curve_button.isEnabled()


# --- Invalidation on source/model change ---------------------------------


def test_pending_fit_is_invalidated_when_source_series_changes(qapp):
    figure = GnoviFigure()
    ds = _dataset()
    figure.add_series(PlotSeries.line(ds, "x", "y", label="First"))
    figure.add_series(PlotSeries.line(ds, "x", "y", label="Second"))
    panel = AnalysisPanel(figure, DatasetManager())

    panel.source_combo.setCurrentIndex(0)
    panel.run_fit_button.click()
    assert panel.add_fit_curve_button.isEnabled()

    panel.source_combo.setCurrentIndex(1)

    assert not panel.add_fit_curve_button.isEnabled()
    assert not panel.pending_fit_label.isVisibleTo(panel)


def test_pending_fit_is_invalidated_when_model_changes(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    panel = AnalysisPanel(figure, DatasetManager())

    panel.model_combo.setCurrentIndex(panel.model_combo.findData(LINEAR))
    panel.run_fit_button.click()
    assert panel.add_fit_curve_button.isEnabled()

    panel.model_combo.setCurrentIndex(panel.model_combo.findData(POLYNOMIAL))

    assert not panel.add_fit_curve_button.isEnabled()


def test_pending_fit_survives_a_refresh_that_keeps_the_same_selection(qapp):
    """refresh() rebuilding the combo without an actual selection change
    (e.g. a style edit elsewhere) must not spuriously invalidate."""
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    panel = AnalysisPanel(figure, DatasetManager())

    panel.run_fit_button.click()
    assert panel.add_fit_curve_button.isEnabled()

    panel.refresh()

    assert panel.add_fit_curve_button.isEnabled()


# --- Run Fit alone never touches the DatasetManager -----------------------


def test_run_fit_alone_creates_no_dataset_even_run_twice(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)

    panel.run_fit_button.click()
    panel.run_fit_button.click()

    assert len(manager.datasets) == 0


# --- Add Fit Curve to Plot: dataset/series creation -----------------------


def test_add_fit_curve_creates_exactly_one_derived_dataset(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)
    added = []
    panel.add_to_plot_requested.connect(added.append)

    panel.run_fit_button.click()
    assert len(manager.datasets) == 0  # Run Fit alone: still nothing

    panel.add_fit_curve_button.click()

    assert len(manager.datasets) == 1
    assert len(added) == 1


def test_derived_dataset_is_tagged_and_carries_full_provenance(qapp):
    figure = GnoviFigure()
    ds = _dataset(name="source-ds")
    series = PlotSeries.line(ds, "x", "y", label="Source series")
    figure.add_series(series)
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)

    panel.model_combo.setCurrentIndex(panel.model_combo.findData(LINEAR))
    panel.run_fit_button.click()
    panel.add_fit_curve_button.click()

    fit_dataset = manager.datasets[0]
    meta = fit_dataset.metadata

    assert meta["kind"] == "fit"
    assert meta["source_dataset_id"] == ds.id
    assert meta["source_series_id"] == series.id
    assert meta["model"] == LINEAR
    assert meta["params"]["a"] == pytest.approx(2.0, abs=1e-6)
    assert meta["params"]["b"] == pytest.approx(1.0, abs=1e-6)
    assert "param_errors" in meta  # present (may be None) -- key always exists
    assert meta["r_squared"] == pytest.approx(1.0, abs=1e-6)
    assert meta["x_column"] == "x"
    assert meta["y_column"] == "y"
    assert "x_min" in meta and "x_max" in meta
    assert meta["num_points"] > 2


def test_derived_dataset_fitted_curve_matches_evaluate_fit(qapp):
    from gnovi_plot.analysis.fitting import evaluate_fit

    figure = GnoviFigure()
    x = np.linspace(-5, 5, 30)
    y = 1.0 + 2.0 * x + 0.5 * x**2
    ds = Dataset(name="quad", dataframe=pd.DataFrame({"x": x, "y": y}))
    figure.add_series(PlotSeries.line(ds, "x", "y", label="Quadratic"))
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)

    panel.model_combo.setCurrentIndex(panel.model_combo.findData(POLYNOMIAL))
    panel.degree_spin.setValue(2)
    panel.run_fit_button.click()
    result = panel._pending_fit
    panel.add_fit_curve_button.click()

    fit_dataset = manager.datasets[0]
    fit_df = fit_dataset.dataframe
    expected_y = evaluate_fit(result, fit_df["x"].to_numpy())

    assert fit_df["y"].to_numpy() == pytest.approx(expected_y)
    assert fit_df["x"].min() == pytest.approx(x.min())
    assert fit_df["x"].max() == pytest.approx(x.max())


def test_add_fit_curve_emits_a_normal_styleable_line_series(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)
    added = []
    panel.add_to_plot_requested.connect(lambda series_list: added.extend(series_list))

    panel.run_fit_button.click()
    panel.add_fit_curve_button.click()

    assert len(added) == 1
    series = added[0]
    assert isinstance(series, PlotSeries)
    assert series.plot_type == PlotType.LINE
    assert series.dataset is manager.datasets[0]
    # Not yet styled -- color/etc. are still at PlotSeries defaults, so
    # GnoviFigure.add_series's normal auto-color-cycle applies exactly like
    # any other freshly added series (see figure.py's add_series).
    assert series.color is None
    assert series.color_is_manual is False
    assert series.visible is True


# --- Fit-time descriptive provenance snapshot ------------------------------


def test_run_fit_passes_the_live_dataset_name_and_series_label_to_fit_curve(qapp):
    figure = GnoviFigure()
    ds = _dataset(name="Ferricyanide 50 mV/s")
    series = PlotSeries.line(ds, "x", "y", label="Current vs Potential")
    figure.add_series(series)
    panel = AnalysisPanel(figure, DatasetManager())

    panel.run_fit_button.click()

    assert panel._pending_fit.source_dataset_name == "Ferricyanide 50 mV/s"
    assert panel._pending_fit.source_series_label == "Current vs Potential"


# --- Add Fit Curve to Plot: stays usable, clear feedback (corrected PR5) --


def test_add_fit_curve_button_remains_enabled_after_a_successful_add(qapp):
    """Do NOT permanently disable the button merely because the current
    fit has already been added once -- the same fit may be added again
    while it's still valid (e.g. a second copy to style differently)."""
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)
    added = []
    panel.add_to_plot_requested.connect(lambda series_list: added.extend(series_list))

    panel.run_fit_button.click()
    panel.add_fit_curve_button.click()

    assert panel.add_fit_curve_button.isEnabled()

    panel.add_fit_curve_button.click()  # adding again must work, not be a no-op

    assert len(manager.datasets) == 2
    assert len(added) == 2  # two separate add_to_plot_requested emissions


def test_added_feedback_shown_after_a_successful_add(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    panel = AnalysisPanel(figure, DatasetManager())

    panel.run_fit_button.click()
    assert not panel.added_feedback_label.isVisibleTo(panel)

    panel.add_fit_curve_button.click()

    assert panel.added_feedback_label.isVisibleTo(panel)
    assert panel.added_feedback_label.text() == "Added to plot: Fit: linear — y"


def test_added_feedback_clears_when_a_new_fit_is_run(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    panel = AnalysisPanel(figure, DatasetManager())

    panel.run_fit_button.click()
    panel.add_fit_curve_button.click()
    assert panel.added_feedback_label.isVisibleTo(panel)

    panel.run_fit_button.click()  # a fresh fit hasn't been added yet

    assert not panel.added_feedback_label.isVisibleTo(panel)


def test_meaningful_source_change_still_invalidates_after_a_successful_add(qapp):
    """Existing stale-fit invalidation on Source/Model changes must
    continue to work exactly as before, even once the current fit has
    already been added to the plot."""
    figure = GnoviFigure()
    ds = _dataset()
    figure.add_series(PlotSeries.line(ds, "x", "y", label="First"))
    figure.add_series(PlotSeries.line(ds, "x", "y", label="Second"))
    panel = AnalysisPanel(figure, DatasetManager())

    panel.source_combo.setCurrentIndex(0)
    panel.run_fit_button.click()
    panel.add_fit_curve_button.click()
    assert panel.add_fit_curve_button.isEnabled()
    assert panel.added_feedback_label.isVisibleTo(panel)

    panel.source_combo.setCurrentIndex(1)  # meaningful Source change

    assert not panel.add_fit_curve_button.isEnabled()
    assert not panel.added_feedback_label.isVisibleTo(panel)
    assert not panel.pending_fit_label.isVisibleTo(panel)


def test_meaningful_model_change_still_invalidates_after_a_successful_add(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    panel = AnalysisPanel(figure, DatasetManager())

    panel.model_combo.setCurrentIndex(panel.model_combo.findData(LINEAR))
    panel.run_fit_button.click()
    panel.add_fit_curve_button.click()
    assert panel.add_fit_curve_button.isEnabled()

    panel.model_combo.setCurrentIndex(panel.model_combo.findData(POLYNOMIAL))

    assert not panel.add_fit_curve_button.isEnabled()
    assert not panel.added_feedback_label.isVisibleTo(panel)
