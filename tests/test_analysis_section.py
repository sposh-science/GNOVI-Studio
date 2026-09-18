"""Issue #61: the shared `AnalysisSection` contract -- and that
`XRDAnalysisSection`/`CVAnalysisSection` still satisfy it after being
refactored onto it. Deliberately thin: this is NOT a re-run of either
section's own full behavioural test matrix (see
`test_xrd_analysis_workspace_gui.py`/`test_cv_analysis_workspace_gui.py`
for that) -- only the base class's own genuinely-shared behaviour, plus a
"the real subclasses are still wired to it" smoke check per subclass."""

from __future__ import annotations

import pandas as pd
import pytest

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.analysis_section import AnalysisSection, eligible_analysis_series
from gnovi_plot.gui.widgets.cv_analysis_section import CVAnalysisSection
from gnovi_plot.gui.widgets.xrd_analysis_section import XRDAnalysisSection
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series import PlotSeries
from gnovi_plot.plotting.series3d import Plot3DType, Series3D


def _make_dataset(name="d") -> Dataset:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return Dataset(name=name, dataframe=df)


# --- eligible_analysis_series -- the deduplicated source-series filter -----


def test_eligible_analysis_series_includes_line_and_scatter_only():
    ds = _make_dataset()
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(ds, "x", "y"))
    figure.add_series(PlotSeries.scatter(ds, "x", "y"))
    figure.add_series(PlotSeries.histogram(ds, "x"))  # no y_column -- excluded

    assert len(eligible_analysis_series(figure)) == 2


def test_eligible_analysis_series_excludes_stale_series():
    ds = _make_dataset()
    figure = GnoviFigure()
    series = PlotSeries.line(ds, "x", "y")
    figure.add_series(series)
    series.stale = True

    assert eligible_analysis_series(figure) == []


def test_eligible_analysis_series_empty_for_panel3d():
    ds = _make_dataset()
    figure = GnoviFigure()
    figure.panels[0] = Panel3D(panel_label="3D")
    figure.panels[0].add_series(
        Series3D(dataset=ds, x_column="x", y_column="y", z_column="x", plot_type=Plot3DType.SCATTER, label="s")
    )

    assert eligible_analysis_series(figure) == []


# --- AnalysisSection base: shared behaviour -----------------------------
#
# A minimal concrete subclass exercising only what the base itself
# provides -- required overrides deliberately left unimplemented except
# `_set_manual_peak_mode`, since `disarm_manual_peak_mode()`'s shared
# implementation calls it.


class _MinimalSection(AnalysisSection):
    def __init__(self, figure, manager, parent=None):
        super().__init__(figure, manager, parent)
        self.set_manual_peak_mode_calls: list[bool] = []

    def _set_manual_peak_mode(self, active: bool) -> None:
        self._manual_peak_mode = active
        self.set_manual_peak_mode_calls.append(active)


def _minimal_section(qapp) -> _MinimalSection:
    figure = GnoviFigure()
    manager = DatasetManager()
    return _MinimalSection(figure, manager)


def test_required_overrides_raise_not_implemented_when_unimplemented(qapp):
    section = _minimal_section(qapp)

    with pytest.raises(NotImplementedError):
        section.set_figure(GnoviFigure())
    with pytest.raises(NotImplementedError):
        section.refresh()
    with pytest.raises(NotImplementedError):
        section.load_result(None)
    with pytest.raises(NotImplementedError):
        section.add_manual_peak(1.0, 2.0)


def test_is_manual_peak_mode_defaults_false(qapp):
    section = _minimal_section(qapp)

    assert section.is_manual_peak_mode() is False


def test_set_manager_repoints_the_shared_manager_attribute(qapp):
    section = _minimal_section(qapp)
    new_manager = DatasetManager()

    section.set_manager(new_manager)

    assert section._manager is new_manager


def test_set_selected_peak_rows_sorts_and_deduplicates(qapp):
    section = _minimal_section(qapp)

    section.set_selected_peak_rows([3, 1, 1, 2])

    assert section._results_selected_rows == [1, 2, 3]


def test_disarm_manual_peak_mode_calls_into_the_subclass_hook(qapp):
    section = _minimal_section(qapp)
    section._set_manual_peak_mode(True)
    assert section.is_manual_peak_mode() is True

    section.disarm_manual_peak_mode()

    assert section.is_manual_peak_mode() is False
    assert section.set_manual_peak_mode_calls == [True, False]


# --- XRD/CV still satisfy the contract after the refactor -----------------


def test_xrd_analysis_section_is_an_analysis_section(qapp):
    figure = GnoviFigure()
    section = XRDAnalysisSection(figure, DatasetManager())

    assert isinstance(section, AnalysisSection)
    assert section.is_manual_peak_mode() is False  # inherited, not overridden
    section.set_manager(DatasetManager())  # inherited -- must not raise
    section.close()


def test_cv_analysis_section_is_an_analysis_section(qapp):
    figure = GnoviFigure()
    section = CVAnalysisSection(figure, DatasetManager())

    assert isinstance(section, AnalysisSection)
    assert section.is_manual_peak_mode() is False  # inherited, not overridden
    section.set_manager(DatasetManager())  # inherited -- must not raise
    section.close()
