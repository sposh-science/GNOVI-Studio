from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from gnovi_plot.analysis.fitting import LINEAR, fit_curve
from gnovi_plot.analysis.results import AnalysisResult
from gnovi_plot.gui.widgets.analysis_result_view import AnalysisResultView


@dataclass
class _DummyResult(AnalysisResult):
    """A second, unrelated `AnalysisResult` subclass -- stands in for a
    future tool (statistics, peak analysis, ...) so these tests can prove
    `AnalysisResultView` isn't secretly coupled to `FitResult`."""

    kind: ClassVar[str] = "dummy"
    note: str = ""

    def summary(self) -> str:
        return f"dummy result: {self.note}"

    def details(self) -> list[tuple[str, str]]:
        return [("Note", self.note), ("Extra", "42")]


def _make_dummy(**overrides) -> _DummyResult:
    defaults = dict(
        source_dataset_id="dataset-1",
        source_series_id=None,
        x_column="x",
        y_column="y",
        row_range=None,
        note="hello",
    )
    defaults.update(overrides)
    return _DummyResult(**defaults)


def _make_fit_result():
    import numpy as np

    x = np.linspace(0, 10, 10)
    y = 2.0 * x + 1.0
    return fit_curve(
        x,
        y,
        LINEAR,
        source_dataset_id="dataset-2",
        source_series_id="series-2",
        x_column="x",
        y_column="y",
    )


def test_starts_in_empty_state(qapp):
    view = AnalysisResultView()
    assert view.result is None
    assert view._empty_label.isVisibleTo(view)
    assert not view._content.isVisibleTo(view)


def test_show_result_hides_empty_state_and_shows_content(qapp):
    view = AnalysisResultView()
    view.show_result(_make_dummy())

    assert not view._empty_label.isVisibleTo(view)
    assert view._content.isVisibleTo(view)
    assert view.result is not None


def test_show_result_renders_summary_and_details_generically(qapp):
    view = AnalysisResultView()
    result = _make_dummy(note="peak search")
    view.show_result(result)

    assert view._summary_label.text() == result.summary()
    assert view._details_form.rowCount() == len(result.details())


def test_clear_returns_to_empty_state(qapp):
    view = AnalysisResultView()
    view.show_result(_make_dummy())
    view.clear()

    assert view.result is None
    assert view._empty_label.isVisibleTo(view)
    assert not view._content.isVisibleTo(view)


def test_showing_a_second_result_replaces_the_first_not_appends(qapp):
    view = AnalysisResultView()
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
    view = AnalysisResultView()
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
