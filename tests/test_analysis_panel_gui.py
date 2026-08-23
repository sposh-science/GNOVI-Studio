from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QMessageBox

from gnovi_plot.analysis.fitting import EXPONENTIAL, GAUSSIAN, LINEAR, POLYNOMIAL, fit_curve
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


def test_fit_result_carries_the_active_panels_stable_id(qapp):
    """`source_panel_id` must be the *active* panel's `Panel.id` -- never
    an index or "Panel N" display text -- so a panel-scoped history can
    key on it reliably."""
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    ds = _dataset(name="my-dataset")
    figure.set_active_panel(1)
    series = PlotSeries.line(ds, "x", "y", label="Panel 2 series")
    figure.add_series(series)
    expected_panel_id = figure.active_panel.id

    panel = AnalysisPanel(figure, DatasetManager())
    results = _capture_results(panel)
    panel.run_fit_button.click()

    assert results[0].source_panel_id == expected_panel_id
    assert results[0].source_panel_id != figure.panels[0].id


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


# --- Add / Remove Fit Curve: one FitResult.result_id -> at most one curve ---


def test_add_fit_curve_button_disables_after_a_successful_add(qapp):
    """One FitResult.result_id maps to at most one generated PlotSeries in
    its panel -- Add becomes disabled immediately after a successful add
    (this alone needs no outside help: it's computed purely from
    `_pending_fit` + the active panel's own series, see
    `_refresh_fit_curve_buttons`). Repeated clicking of Add Fit Curve is
    never the duplication mechanism (see AnalysisPanel's own class
    docstring); a deliberate second styled copy is a job for a future
    "Duplicate Series" command on the Series page instead."""
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)
    # Mirrors what MainWindow's _on_add_to_plot actually does: mutate the
    # same live figure the panel reads from -- a bare capture-only lambda
    # (the old pattern here) can never exercise the new duplicate guard,
    # which checks the *actual* panel series list.
    panel.add_to_plot_requested.connect(lambda series_list: figure.add_series(series_list[0]))

    panel.run_fit_button.click()
    assert panel.add_fit_curve_button.isEnabled()
    assert not panel.remove_fit_curve_button.isEnabled()

    panel.add_fit_curve_button.click()

    assert not panel.add_fit_curve_button.isEnabled()


def test_remove_fit_curve_button_enables_once_mainwindow_style_sync_reports_the_match(qapp):
    """Unlike Add (which falls back to self-computing from `_pending_fit`
    -- see `_add_target`), 'Remove' is deliberately driven only by
    whatever MainWindow tells this panel via `sync_history` -- a fully
    isolated AnalysisPanel (no MainWindow) must be told explicitly,
    exactly as MainWindow's own `_on_figure_content_changed` does in the
    real app (see the MainWindow-level equivalent of this test in
    test_main_window_analysis_workflow_gui.py, which needs no such
    explicit call)."""
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)
    panel.add_to_plot_requested.connect(lambda series_list: figure.add_series(series_list[0]))

    panel.run_fit_button.click()
    result = panel._pending_fit
    panel.add_fit_curve_button.click()
    assert not panel.remove_fit_curve_button.isEnabled()  # not yet told

    panel.sync_history([result], result)

    assert panel.remove_fit_curve_button.isEnabled()


def test_clicking_add_fit_curve_again_while_disabled_creates_no_second_series(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)
    panel.add_to_plot_requested.connect(lambda series_list: figure.add_series(series_list[0]))

    panel.run_fit_button.click()
    panel.add_fit_curve_button.click()
    count_after_first_add = len(figure.active_panel.series)

    panel.add_fit_curve_button.click()  # disabled -- Qt no-op
    assert len(figure.active_panel.series) == count_after_first_add

    panel._on_add_fit_curve_clicked()  # defensive, even called directly
    assert len(figure.active_panel.series) == count_after_first_add


def test_a_second_panel_instance_only_sees_the_curve_after_its_own_sync(qapp):
    """A fresh AnalysisPanel instance sharing the same figure (as if
    reloaded/newly wired) only recognizes an already-existing curve via
    an explicit sync_history call -- mirrors how MainWindow drives this
    after a project reload, with no fresh pending fit involved at all
    (see test_reopening_a_project_with_a_fit_curve_restores_disabled_
    add_enabled_remove in test_main_window_analysis_workflow_gui.py)."""
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)
    panel.add_to_plot_requested.connect(lambda series_list: figure.add_series(series_list[0]))

    panel.run_fit_button.click()
    result = panel._pending_fit
    panel.add_fit_curve_button.click()

    other_panel = AnalysisPanel(figure, manager)
    assert not other_panel.remove_fit_curve_button.isEnabled()

    other_panel.sync_history([result], result)

    assert other_panel.remove_fit_curve_button.isEnabled()


def test_removing_the_fit_curve_re_enables_add_and_disables_remove(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)
    added = []

    def _on_add(series_list):
        figure.add_series(series_list[0])
        added.extend(series_list)

    panel.add_to_plot_requested.connect(_on_add)
    removed_ids = []
    panel.remove_fit_curve_requested.connect(removed_ids.extend)

    panel.run_fit_button.click()
    result = panel._pending_fit
    panel.add_fit_curve_button.click()
    fit_series = next(s for s in figure.active_panel.series if s.dataset.metadata.get("result_id") == result.result_id)
    panel.sync_history([result], result)  # as MainWindow would, right after the add
    assert panel.remove_fit_curve_button.isEnabled()

    panel.remove_fit_curve_button.click()

    assert removed_ids == [fit_series.id]
    # AnalysisPanel only emits the request -- it never mutates the figure
    # itself (mirrors add_to_plot_requested's own contract), so simulate
    # what MainWindow's _on_remove_fit_curve does, then re-sync.
    figure.remove_series(fit_series.id)
    panel.sync_history([result], result)

    assert panel.add_fit_curve_button.isEnabled()
    assert not panel.remove_fit_curve_button.isEnabled()

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


# --- Analysis History: list population / disambiguation / selection ----------


def _fit_result(model=LINEAR, y_column="y", panel_id="panel-1", x=None, y=None, **overrides):
    x = np.linspace(0, 10, 25) if x is None else x
    y = (2.0 * x + 1.0) if y is None else y
    return fit_curve(
        x,
        y,
        model,
        source_dataset_id=overrides.pop("source_dataset_id", "dataset-1"),
        x_column="x",
        y_column=y_column,
        source_panel_id=panel_id,
        **overrides,
    )


def test_history_list_is_populated_oldest_first_with_compact_labels(qapp):
    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())
    gaussian = _fit_result(model=GAUSSIAN, y_column="gaussian_peak_y", x=np.linspace(-5, 5, 30), y=np.exp(-np.linspace(-5, 5, 30) ** 2))
    linear = _fit_result(model=LINEAR, y_column="linear_y")

    panel.sync_history([gaussian, linear], linear)

    labels = [panel.history_list.item(i).text() for i in range(panel.history_list.count())]
    assert labels == ["Gaussian fit — gaussian_peak_y", "Linear fit — linear_y"]
    assert panel.history_list.currentRow() == 1  # the current result is selected
    assert not panel.history_status_label.isVisibleTo(panel)


def test_history_list_disambiguates_repeated_model_and_column_labels(qapp):
    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())
    first = _fit_result(model=LINEAR, y_column="linear_y")
    second = _fit_result(model=LINEAR, y_column="linear_y")
    third = _fit_result(model=LINEAR, y_column="linear_y")

    panel.sync_history([first, second, third], third)

    labels = [panel.history_list.item(i).text() for i in range(panel.history_list.count())]
    assert labels == [
        "Linear fit — linear_y",
        "Linear fit — linear_y · #2",
        "Linear fit — linear_y · #3",
    ]


def test_history_list_disambiguation_does_not_change_stored_provenance(qapp):
    """The '· #N' suffix is a display-only decoration -- it must never
    leak into the underlying FitResult/its persisted fields."""
    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())
    first = _fit_result(model=LINEAR, y_column="linear_y")
    second = _fit_result(model=LINEAR, y_column="linear_y")

    panel.sync_history([first, second], second)

    assert first.y_column == "linear_y"
    assert second.y_column == "linear_y"
    assert "#" not in first.to_dict()["y_column"]


def test_empty_history_shows_the_empty_state_label(qapp):
    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())

    panel.sync_history([], None)

    assert panel.history_status_label.isVisibleTo(panel)
    assert panel.history_list.count() == 0


def test_selecting_a_history_row_emits_history_result_selected(qapp):
    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())
    gaussian = _fit_result(model=GAUSSIAN, y_column="y", x=np.linspace(-5, 5, 30), y=np.exp(-np.linspace(-5, 5, 30) ** 2))
    linear = _fit_result(model=LINEAR, y_column="y")
    panel.sync_history([gaussian, linear], linear)

    selected = []
    panel.history_result_selected.connect(selected.append)
    panel.history_list.setCurrentRow(0)  # pick the older, Gaussian entry

    assert selected == [gaussian]
    assert panel._current_result is gaussian


def test_programmatic_sync_history_never_emits_history_result_selected(qapp):
    """sync_history() is MainWindow pushing state in (panel/Workbench
    switch, undo/redo, project load) -- it must never look like a user
    click, or MainWindow would record a selection nobody made and mark
    the project dirty for a pure display sync."""
    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())
    result = _fit_result()

    selected = []
    panel.history_result_selected.connect(selected.append)
    panel.sync_history([result], result)

    assert selected == []


def test_set_figure_clears_the_history_list(qapp):
    figure = GnoviFigure()
    panel = AnalysisPanel(figure, DatasetManager())
    panel.sync_history([_fit_result()], None)
    assert panel.history_list.count() == 1

    panel.set_figure(GnoviFigure())

    assert panel.history_list.count() == 0
    assert panel.history_status_label.isVisibleTo(panel)


# --- Add Fit Curve: regenerating a historical result's curve -----------------


def test_add_fit_curve_uses_the_results_stored_curve_range_not_the_lives_series_range(qapp):
    """A FitResult selected from History (never run in this session) must
    regenerate its curve across its own stored curve_x_min/curve_x_max --
    what was actually fitted -- never the *current* live source range,
    even when they differ."""
    figure = GnoviFigure()
    live_series = PlotSeries.line(_dataset(x=list(range(0, 20)), y=[v for v in range(0, 20)]), "x", "y", label="live")
    figure.add_series(live_series)
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)
    added = []
    panel.add_to_plot_requested.connect(lambda series_list: (figure.add_series(series_list[0]), added.extend(series_list)))

    result = _fit_result(
        x=np.linspace(-5.0, -1.0, 10),
        y=2.0 * np.linspace(-5.0, -1.0, 10) + 1.0,
        source_series_id=live_series.id,
    )
    assert result.curve_x_min == pytest.approx(-5.0)
    assert result.curve_x_max == pytest.approx(-1.0)

    panel.sync_history([result], result)
    panel.add_fit_curve_button.click()

    assert len(added) == 1
    fit_series = added[0]
    x_values = fit_series.dataframe[fit_series.x_column]
    assert x_values.min() == pytest.approx(-5.0)
    assert x_values.max() == pytest.approx(-1.0)


def test_add_fit_curve_falls_back_to_live_data_range_when_the_result_has_none_stored(qapp):
    """A `FitResult` persisted before curve_x_min/curve_x_max existed
    (both `None`) must fall back to the source series' current live
    range -- see `_resolve_curve_range`."""
    figure = GnoviFigure()
    live_series = PlotSeries.line(_dataset(x=list(range(0, 8)), y=[2 * v for v in range(0, 8)]), "x", "y", label="live")
    figure.add_series(live_series)
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)
    added = []
    panel.add_to_plot_requested.connect(lambda series_list: (figure.add_series(series_list[0]), added.extend(series_list)))

    result = _fit_result(source_series_id=live_series.id)
    result.curve_x_min = None
    result.curve_x_max = None
    result.curve_num_points = None

    panel.sync_history([result], result)
    panel.add_fit_curve_button.click()

    assert len(added) == 1
    x_values = added[0].dataframe[added[0].x_column]
    assert x_values.min() == pytest.approx(0.0)
    assert x_values.max() == pytest.approx(7.0)


def test_add_fit_curve_fails_gracefully_when_no_range_is_available_at_all(qapp, monkeypatch):
    """No stored curve range AND the source dataset/series no longer
    exists -- must show a clear message and create nothing, never crash
    or fabricate a range."""
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warnings.append(a)))

    figure = GnoviFigure()
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)
    added = []
    panel.add_to_plot_requested.connect(lambda series_list: added.extend(series_list))

    result = _fit_result(source_series_id=None, source_dataset_id="does-not-exist-in-manager")
    result.curve_x_min = None
    result.curve_x_max = None
    result.curve_num_points = None

    panel.sync_history([result], result)
    panel.add_fit_curve_button.click()

    assert len(warnings) == 1
    assert "Curve Fitting" in warnings[0]
    assert added == []
    assert manager.datasets == []


# --- Precedence: explicit current result vs. session-local _pending_fit -----
#
# Regression coverage for a result with source_panel_id=None -- never added
# to any panel's PanelResultHistory, so it can never appear in the `results`
# list `sync_history` is given, yet it can still be `current` (Results is
# showing it). Add/Remove Fit Curve must still target that exact result,
# never a stale/unrelated `_pending_fit`, purely because history has nothing
# recorded for it -- source_panel_id must never be the deciding condition for
# which result the Analysis controls act on (see `sync_history`'s and
# `_add_target`'s own docstrings).


def test_explicit_current_result_with_no_source_panel_id_beats_a_stale_pending_fit(qapp):
    """(A) An explicit current result (source_panel_id=None, absent from
    `results`) must win over a different, stale `_pending_fit` -- Add/
    Remove must target the explicit current result, never the pending
    one, and Results/current must agree."""
    figure = GnoviFigure()
    live_series = PlotSeries.line(_dataset(), "x", "y", label="live")
    figure.add_series(live_series)
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)
    added = []
    panel.add_to_plot_requested.connect(lambda series_list: (figure.add_series(series_list[0]), added.extend(series_list)))

    # A stale pending fit, as if "Run Fit" had been clicked for a
    # completely different result.
    panel.model_combo.setCurrentIndex(panel.model_combo.findData(LINEAR))
    panel.run_fit_button.click()
    stale_pending = panel._pending_fit
    assert stale_pending is not None

    # An explicit current result with source_panel_id=None -- never part
    # of any panel's history (see PanelResultHistory), but still supplied
    # to the panel as what Results is showing right now.
    explicit_current = _fit_result(panel_id=None, source_series_id=live_series.id)
    assert explicit_current.result_id != stale_pending.result_id
    panel.sync_history([], explicit_current)  # empty history list -- not tracked in it

    assert panel._current_result is explicit_current
    assert panel._add_target() is explicit_current  # never the stale pending fit

    panel.add_fit_curve_button.click()

    assert len(added) == 1
    assert added[0].dataset.metadata.get("result_id") == explicit_current.result_id
    assert added[0].dataset.metadata.get("result_id") != stale_pending.result_id


def test_explicit_current_result_scopes_remove_to_itself_not_the_pending_fit(qapp):
    """(A) Remove Fit Curve, too, must target the explicit current result
    -- never a stale pending fit -- even when the pending fit's own curve
    is also present on the plot."""
    figure = GnoviFigure()
    live_series = PlotSeries.line(_dataset(), "x", "y", label="live")
    figure.add_series(live_series)
    manager = DatasetManager()
    panel = AnalysisPanel(figure, manager)
    panel.add_to_plot_requested.connect(lambda series_list: figure.add_series(series_list[0]))

    panel.model_combo.setCurrentIndex(panel.model_combo.findData(LINEAR))
    panel.run_fit_button.click()
    pending = panel._pending_fit
    panel.add_fit_curve_button.click()  # pending fit's curve is now on the plot
    pending_series_id = next(
        s.id for s in figure.active_panel.series if s.dataset.metadata.get("result_id") == pending.result_id
    )

    explicit_current = _fit_result(panel_id=None, source_series_id=live_series.id)
    panel.sync_history([], explicit_current)

    # Explicit current has no curve of its own yet -- Remove must not be
    # enabled for the (unrelated) pending fit's curve.
    assert not panel.remove_fit_curve_button.isEnabled()
    assert panel._matched_fit_curve_series_ids == []

    remaining_before = {s.id for s in figure.active_panel.series}
    assert pending_series_id in remaining_before  # untouched by the sync above


def test_no_explicit_current_result_still_supports_the_simple_run_fit_workflow(qapp):
    """(B) With no explicit current/history result supplied at all
    (fresh panel, nothing ever synced), the plain Run Fit -> Add Fit
    Curve workflow must keep working exactly as before, via the
    `_pending_fit` fallback."""
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    panel = AnalysisPanel(figure, DatasetManager())
    assert panel._current_result is None

    panel.run_fit_button.click()

    assert panel._add_target() is panel._pending_fit
    assert panel.add_fit_curve_button.isEnabled()


def test_clearing_the_explicit_current_result_does_not_expose_a_stale_selection(qapp):
    """(C) Explicitly clearing current (sync_history(..., None), e.g. an
    empty-history panel switch) must leave state coherent: no stale
    result treated as selected, History shows no row highlighted, and
    Add/Remove Fit Curve fall back cleanly (Add only via a genuinely
    still-pending fit, Remove disabled since nothing is current)."""
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    panel = AnalysisPanel(figure, DatasetManager())

    first = _fit_result(panel_id=None)
    panel.sync_history([], first)
    assert panel._current_result is first

    panel.sync_history([], None)  # explicit clear -- e.g. switched to an empty-history panel

    assert panel._current_result is None
    assert panel.history_list.currentRow() == -1
    assert not panel.remove_fit_curve_button.isEnabled()
    assert panel._matched_fit_curve_series_ids == []
    # Add falls back to whatever's genuinely still pending in this
    # session (here, nothing was ever run) -- never `first`.
    assert panel._add_target() is None
    assert not panel.add_fit_curve_button.isEnabled()
