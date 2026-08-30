"""XRDAnalysisResult / XRDPeakFitResult project persistence: save/reopen
through the real `.gnovi` container, confirming PROJECT_FORMAT_VERSION
needed no bump. Mirrors test_project_io_3d.py's own style.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gnovi_plot.core.project import Project
from gnovi_plot.core.project_io import PROJECT_FORMAT_VERSION, load_project, save_project
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.modules.xrd.fitting import (
    BASELINE_LINEAR,
    GAUSSIAN,
    XRDPeakFitResult,
    fit_xrd_peak,
    peak_component,
)
from gnovi_plot.modules.xrd.peaks import XRDPeakSeed
from gnovi_plot.modules.xrd.radiation import radiation_from_preset
from gnovi_plot.modules.xrd.results import XRDAnalysisResult, build_xrd_analysis_result


def _make_dataset() -> Dataset:
    df = pd.DataFrame({"2theta": [10.0, 20.0, 30.0, 40.0], "counts": [50.0, 800.0, 60.0, 350.0]})
    return Dataset(name="xrd pattern", dataframe=df)


def _project_with_xrd_result():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id

    result = build_xrd_analysis_result(
        source_dataset_id=dataset.id,
        source_dataset_name=dataset.name,
        x_column="2theta",
        y_column="counts",
        radiation=radiation_from_preset("cu_ka1"),
        peaks=[
            XRDPeakSeed(two_theta=20.0, intensity=800.0, origin="automatic", index=1, prominence=750.0),
            XRDPeakSeed(two_theta=40.0, intensity=350.0, origin="automatic", index=3, prominence=300.0),
        ],
        source_panel_id=panel_id,
        parameters={"detection": {"prominence": 100.0}},
    )
    workbench.analysis_results.add(panel_id, result)
    return project, dataset, panel_id, result


def test_save_reopen_preserves_an_xrd_analysis_result(tmp_path):
    project, _dataset, panel_id, result = _project_with_xrd_result()

    out_path = save_project(project, tmp_path / "xrd.gnovi")
    reloaded = load_project(out_path)

    reloaded_result = reloaded.workbenches[0].analysis_results.current(panel_id)
    assert isinstance(reloaded_result, XRDAnalysisResult)
    assert reloaded_result.result_id == result.result_id
    assert reloaded_result.radiation == result.radiation
    assert [p.two_theta for p in reloaded_result.peaks] == [p.two_theta for p in result.peaks]
    assert reloaded_result.engine == result.engine
    assert reloaded_result.engine_version == result.engine_version


def test_xrd_result_persistence_needed_no_project_format_bump():
    """XRDAnalysisResult / XRDPeakFitResult are new AnalysisResult "kind"s
    plus (for XRDAnalysisResult) the engine/engine_version/operation/
    parameters fields on AnalysisResult itself -- all additive within the
    already-polymorphic analysis_results structure (see core/project_io.py's
    own comment on PROJECT_FORMAT_VERSION), so the format version stays 3."""
    assert PROJECT_FORMAT_VERSION == 3


def _project_with_xrd_fit_result():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id

    x = np.linspace(26.0, 30.0, 1200)
    y = 40.0 + 2.0 * (x - 28.0) + peak_component(x, GAUSSIAN, 500.0, 28.0, 0.4)
    result = fit_xrd_peak(
        x, y, GAUSSIAN,
        fit_window=(26.5, 29.5),
        baseline=BASELINE_LINEAR,
        radiation=radiation_from_preset("cu_ka1"),
        source_dataset_id=dataset.id,
        source_dataset_name=dataset.name,
        x_column="2theta",
        y_column="counts",
        source_panel_id=panel_id,
        source_peak_id="seed-abc",
    )
    workbench.analysis_results.add(panel_id, result)
    return project, panel_id, result


def test_save_reopen_preserves_an_xrd_peak_fit_result(tmp_path):
    project, panel_id, result = _project_with_xrd_fit_result()

    out_path = save_project(project, tmp_path / "xrd_fit.gnovi")
    reloaded = load_project(out_path)

    back = reloaded.workbenches[0].analysis_results.current(panel_id)
    assert isinstance(back, XRDPeakFitResult)
    assert back.kind == "xrd_peak_fit"
    assert back.result_id == result.result_id
    assert back.model == result.model
    assert back.baseline_model == result.baseline_model
    assert back.fit_window == result.fit_window
    assert back.fwhm_units == result.fwhm_units
    assert back.params == result.params
    assert back.area == result.area
    assert back.area_error == result.area_error
    assert back.center_2theta == result.center_2theta
    assert back.d_spacing == result.d_spacing
    assert back.d_spacing_error == result.d_spacing_error
    assert back.radiation == result.radiation
    assert back.source_peak_id == "seed-abc"
    assert back.warnings == result.warnings
    assert back.engine == "gnovi"
    assert back.engine_version == result.engine_version


def test_a_project_saved_before_xrd_existed_still_loads(tmp_path):
    """A manifest with only a 'fit' kind result (no 'xrd_peaks' anywhere)
    -- the ordinary "older project, newer app" case -- must still load
    cleanly; XRDAnalysisResult registering itself must not disturb loading
    a project that never used it."""
    from gnovi_plot.analysis.fitting import fit_curve

    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id

    fit_result = fit_curve(
        [0, 1, 2, 3], [0, 1, 2, 3], "linear",
        source_dataset_id=dataset.id, x_column="2theta", y_column="counts",
        source_panel_id=panel_id,
    )
    workbench.analysis_results.add(panel_id, fit_result)

    out_path = save_project(project, tmp_path / "fit_only.gnovi")
    reloaded = load_project(out_path)

    reloaded_result = reloaded.workbenches[0].analysis_results.current(panel_id)
    assert reloaded_result.kind == "fit"
    assert reloaded_result.result_id == fit_result.result_id
