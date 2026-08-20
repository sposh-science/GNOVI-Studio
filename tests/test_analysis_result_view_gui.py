from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
from PySide6.QtGui import QGuiApplication

from gnovi_plot.analysis.fitting import LINEAR, fit_curve
from gnovi_plot.analysis.results import AnalysisResult
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.analysis_result_view import AnalysisResultView
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries


@dataclass
class _DummyResult(AnalysisResult):
    """A second, unrelated `AnalysisResult` subclass -- stands in for a
    future tool (statistics, peak analysis, ...) so these tests can prove
    `AnalysisResultView` isn't secretly coupled to `FitResult`. Does not
    override `supports_residuals()` -- stays `False`, the base default."""

    kind: ClassVar[str] = "dummy"
    note: str = ""

    def summary(self) -> str:
        return f"dummy result: {self.note}"

    def details(self) -> list[tuple[str, str]]:
        return [("Note", self.note), ("Extra", "42")]


def _make_dummy(**overrides) -> _DummyResult:
    defaults = dict(
        source_dataset_id="dataset-1",
        source_dataset_name=None,
        source_series_id=None,
        source_series_label=None,
        x_column="x",
        y_column="y",
        row_range=None,
        note="hello",
    )
    defaults.update(overrides)
    return _DummyResult(**defaults)


def _make_dataset(name="d", x=None, y=None):
    x = list(range(10)) if x is None else x
    y = [2.0 * v + 1.0 for v in x] if y is None else y
    return Dataset(name=name, dataframe=pd.DataFrame({"x": x, "y": y}))


def _make_fit_result(**overrides):
    x = np.linspace(0, 10, 10)
    y = 2.0 * x + 1.0
    defaults = dict(
        source_dataset_id="dataset-2",
        source_series_id="series-2",
        x_column="x",
        y_column="y",
    )
    defaults.update(overrides)
    return fit_curve(x, y, LINEAR, **defaults)


def _make_view(figure=None, manager=None) -> AnalysisResultView:
    return AnalysisResultView(figure or GnoviFigure(), manager or DatasetManager())


# --- Empty state / generic rendering (unchanged behavior) -------------------


def test_starts_in_empty_state(qapp):
    view = _make_view()
    assert view.result is None
    assert view._empty_label.isVisibleTo(view)
    assert not view._content.isVisibleTo(view)


def test_show_result_hides_empty_state_and_shows_content(qapp):
    view = _make_view()
    view.show_result(_make_dummy())

    assert not view._empty_label.isVisibleTo(view)
    assert view._content.isVisibleTo(view)
    assert view.result is not None


def test_show_result_renders_summary_and_details_generically(qapp):
    view = _make_view()
    result = _make_dummy(note="peak search")
    view.show_result(result)

    assert view._summary_label.text() == result.summary()
    assert view._details_form.rowCount() == len(result.details())


def test_clear_returns_to_empty_state(qapp):
    view = _make_view()
    view.show_result(_make_dummy())
    view.clear()

    assert view.result is None
    assert view._empty_label.isVisibleTo(view)
    assert not view._content.isVisibleTo(view)


def test_showing_a_second_result_replaces_the_first_not_appends(qapp):
    view = _make_view()
    view.show_result(_make_dummy(note="first"))
    first_row_count = view._details_form.rowCount()

    view.show_result(_make_dummy(note="second"))

    assert view._details_form.rowCount() == first_row_count
    assert "second" in view._summary_label.text()
    assert "first" not in view._summary_label.text()


def test_renders_a_fit_result_without_any_fitting_specific_code_path(qapp):
    """AnalysisResultView never imports FitResult -- this just proves a
    real FitResult renders correctly through the same generic contract as
    the dummy result above."""
    view = _make_view()
    result = _make_fit_result()

    view.show_result(result)

    assert view._summary_label.text() == result.summary()
    detail_labels = [view._details_form.itemAt(i * 2).widget().text() for i in range(view._details_form.rowCount())]
    assert "Model:" in detail_labels


def test_view_module_does_not_import_fitresult():
    """Checks actual name bindings, not source text -- the module's own
    docstring legitimately *mentions* curve fitting in prose explaining
    why the view stays generic; it must not *import* the concrete type."""
    from gnovi_plot.gui.widgets import analysis_result_view

    assert not hasattr(analysis_result_view, "FitResult")
    assert not hasattr(analysis_result_view, "fit_curve")


# --- Human-readable provenance (stored snapshot -> live -> id fallback) -----


def test_uses_the_fit_time_stored_name_snapshot_first(qapp):
    """Even though nothing resolvable exists in the figure/manager passed
    to the view, the stored snapshot alone is enough."""
    view = _make_view()
    result = _make_fit_result(
        source_dataset_name="Ferricyanide 50 mV/s", source_series_label="Current vs Potential"
    )

    view.show_result(result)

    assert view._dataset_label.text() == "Dataset: Ferricyanide 50 mV/s"
    assert view._series_label.text() == "Series: Current vs Potential"


def test_falls_back_to_live_resolution_when_no_stored_snapshot(qapp):
    ds = _make_dataset(name="Ferricyanide 50 mV/s")
    series = PlotSeries.line(ds, "x", "y", label="Current vs Potential")
    figure = GnoviFigure()
    figure.add_series(series)
    manager = DatasetManager()
    manager.add(ds)
    view = _make_view(figure=figure, manager=manager)

    result = _make_fit_result(
        source_dataset_id=ds.id,
        source_series_id=series.id,
        source_dataset_name=None,  # no snapshot -- must fall back live
        source_series_label=None,
    )

    view.show_result(result)

    assert view._dataset_label.text() == "Dataset: Ferricyanide 50 mV/s"
    assert view._series_label.text() == "Series: Current vs Potential"


def test_falls_back_to_raw_id_as_a_last_resort(qapp):
    """Neither a stored snapshot nor a live match -- show the id rather
    than nothing, for an older/orphaned result."""
    view = _make_view()  # empty figure/manager -- nothing resolves live
    result = _make_fit_result(
        source_dataset_id="dataset-vanished",
        source_series_id="series-vanished",
        source_dataset_name=None,
        source_series_label=None,
    )

    view.show_result(result)

    assert view._dataset_label.text() == "Dataset: dataset-vanished"
    assert view._series_label.text() == "Series: series-vanished"


def test_series_row_hidden_when_result_has_no_series(qapp):
    view = _make_view()
    result = _make_dummy(source_series_id=None, source_series_label=None)

    view.show_result(result)

    assert not view._series_label.isVisibleTo(view)


def test_ids_are_retained_in_the_provenance_section_not_the_primary_display(qapp):
    view = _make_view()
    result = _make_fit_result(
        source_dataset_id="dataset-2",
        source_series_id="series-2",
        source_dataset_name="Ferricyanide 50 mV/s",
        source_series_label="Current vs Potential",
    )

    view.show_result(result)

    provenance_rows = [view._provenance_form.itemAt(i * 2).widget().text() for i in range(view._provenance_form.rowCount())]
    assert "Source dataset ID:" in provenance_rows
    assert "Source series ID:" in provenance_rows
    provenance_values = [
        view._provenance_form.itemAt(i * 2 + 1).widget().text() for i in range(view._provenance_form.rowCount())
    ]
    assert "dataset-2" in provenance_values
    assert "series-2" in provenance_values
    # Ids never appear in the primary Dataset/Series display lines.
    assert "dataset-2" not in view._dataset_label.text()
    assert "series-2" not in view._series_label.text()


def test_provenance_section_is_collapsed_by_default(qapp):
    view = _make_view()
    view.show_result(_make_fit_result())

    assert view._provenance_section.is_expanded() is False


# --- Residuals ---------------------------------------------------------------


def test_show_residuals_button_hidden_for_a_result_without_residual_support(qapp):
    view = _make_view()
    view.show_result(_make_dummy())

    assert not view._show_residuals_button.isVisibleTo(view)


def test_show_residuals_button_visible_and_plots_for_a_resolvable_fit(qapp):
    ds = _make_dataset()
    series = PlotSeries.line(ds, "x", "y", label="my series")
    figure = GnoviFigure()
    figure.add_series(series)
    manager = DatasetManager()
    manager.add(ds)
    view = _make_view(figure=figure, manager=manager)

    result = fit_curve(
        ds.dataframe["x"].to_numpy(),
        ds.dataframe["y"].to_numpy(),
        LINEAR,
        source_dataset_id=ds.id,
        source_series_id=series.id,
        x_column="x",
        y_column="y",
    )
    view.show_result(result)
    assert view._show_residuals_button.isVisibleTo(view)
    assert not view._residual_plot.isVisibleTo(view)  # never auto-shown

    view._show_residuals_button.setChecked(True)

    assert view._residual_plot.isVisibleTo(view)
    assert not view._residuals_unavailable_label.isVisibleTo(view)
    assert view._show_residuals_button.text() == "Hide Residuals"

    view._show_residuals_button.setChecked(False)
    assert not view._residual_plot.isVisibleTo(view)
    assert view._show_residuals_button.text() == "Show Residuals"


def test_show_residuals_reports_unavailable_when_source_cannot_be_resolved(qapp):
    view = _make_view()  # nothing registered -- source cannot resolve
    result = _make_fit_result(source_dataset_id="gone", source_series_id="also-gone")
    view.show_result(result)

    view._show_residuals_button.setChecked(True)

    assert not view._residual_plot.isVisibleTo(view)
    assert view._residuals_unavailable_label.isVisibleTo(view)


def test_residuals_reset_to_hidden_for_each_new_result(qapp):
    ds = _make_dataset()
    series = PlotSeries.line(ds, "x", "y", label="s")
    figure = GnoviFigure()
    figure.add_series(series)
    manager = DatasetManager()
    manager.add(ds)
    view = _make_view(figure=figure, manager=manager)

    result = fit_curve(
        ds.dataframe["x"].to_numpy(),
        ds.dataframe["y"].to_numpy(),
        LINEAR,
        source_dataset_id=ds.id,
        source_series_id=series.id,
        x_column="x",
        y_column="y",
    )
    view.show_result(result)
    view._show_residuals_button.setChecked(True)
    assert view._residual_plot.isVisibleTo(view)

    view.show_result(result)  # a "new" result, even if the same object

    assert not view._show_residuals_button.isChecked()
    assert not view._residual_plot.isVisibleTo(view)


# --- Copy Fit Summary ---------------------------------------------------------


def test_copy_fit_summary_writes_report_text_to_clipboard(qapp, monkeypatch):
    captured = []

    class _FakeClipboard:
        def setText(self, text):
            captured.append(text)

    monkeypatch.setattr(
        "gnovi_plot.gui.widgets.analysis_result_view.QGuiApplication.clipboard",
        staticmethod(lambda: _FakeClipboard()),
    )

    view = _make_view()
    result = _make_fit_result(
        source_dataset_name="Ferricyanide 50 mV/s", source_series_label="Current vs Potential"
    )
    view.show_result(result)

    view._copy_summary_button.click()

    assert len(captured) == 1
    text = captured[0]
    assert "Dataset: Ferricyanide 50 mV/s" in text
    assert "Series: Current vs Potential" in text
    assert "Model: linear" in text
    assert "RMSE" in text
    # UUIDs must not be prominent in the copied summary.
    assert "dataset-2" not in text
    assert "series-2" not in text


def test_copy_fit_summary_omits_name_lines_when_nothing_resolves(qapp, monkeypatch):
    """No stored snapshot and no live match -- the copy summary omits the
    Dataset/Series lines rather than showing a raw id as if it were a
    name (contrast with the on-screen display, which does show the id as
    a last resort -- see test_falls_back_to_raw_id_as_a_last_resort)."""
    captured = []

    class _FakeClipboard:
        def setText(self, text):
            captured.append(text)

    monkeypatch.setattr(
        "gnovi_plot.gui.widgets.analysis_result_view.QGuiApplication.clipboard",
        staticmethod(lambda: _FakeClipboard()),
    )

    view = _make_view()  # empty figure/manager -- nothing resolves live
    result = _make_fit_result(
        source_dataset_id="gone", source_series_id="also-gone", source_dataset_name=None, source_series_label=None
    )
    view.show_result(result)

    view._copy_summary_button.click()

    assert len(captured) == 1
    assert "Dataset:" not in captured[0]
    assert "Series:" not in captured[0]
    assert "gone" not in captured[0]
