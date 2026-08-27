"""gnovi_plot.modules.electrochemistry.results.CVCycleAnalysisResult:
construction, the bounded/unbounded display split, polymorphic
serialization, PanelResultHistory integration, and real .gnovi project
save/reopen (confirming no PROJECT_FORMAT_VERSION bump)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from gnovi_plot.analysis.panel_results import PanelResultHistory
from gnovi_plot.analysis.results import ENGINE_GNOVI, result_from_dict
from gnovi_plot.core.app_info import __version__ as APP_VERSION
from gnovi_plot.core.project import Project
from gnovi_plot.core.project_io import PROJECT_FORMAT_VERSION, load_project, save_project
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.modules.electrochemistry.common import SWEEP_FALLING, SWEEP_RISING, segment_sweeps
from gnovi_plot.modules.electrochemistry.cv import (
    PROCESS_ANODIC,
    PROCESS_CATHODIC,
    RATIO_BASIS_CORRECTED,
    CVPeakSeed,
    couple_metrics,
    measure_peak,
)
from gnovi_plot.modules.electrochemistry.results import (
    CV_OPERATION_PEAK_ANALYSIS,
    CYCLE_CONFIDENCE_DETECTED,
    CYCLE_CONFIDENCE_MANUAL,
    CVBaselineInfo,
    CVCycleAnalysisResult,
    CVPeakResult,
    CVSweepInfo,
    build_cv_cycle_analysis_result,
    peak_result_from_seed,
)


def _triangle(cycles=1, lo=-0.2, hi=0.6, step=0.001):
    n = round((hi - lo) / step)
    rising = np.round(np.linspace(lo, hi, n + 1), 10)
    falling = rising[::-1]
    legs = [rising]
    for k in range(1, 2 * cycles):
        legs.append((falling if k % 2 else rising)[1:])
    return np.concatenate(legs)


def _reversible_result(**overrides) -> CVCycleAnalysisResult:
    e = _triangle(1)
    rising_mask = np.gradient(e) > 0
    i = 2e-7 + 1e-5 * np.exp(-((e - 0.25) / 0.03) ** 2) * rising_mask
    i -= 1e-5 * np.exp(-((e - 0.191) / 0.03) ** 2) * (~rising_mask)
    segs = segment_sweeps(e)
    rising = next(s for s in segs if s.direction == SWEEP_RISING)
    falling = next(s for s in segs if s.direction == SWEEP_FALLING)
    a = measure_peak(e, i, search=(rising.start, rising.end), process=PROCESS_ANODIC)
    c = measure_peak(e, i, search=(falling.start, falling.end), process=PROCESS_CATHODIC)
    cm = couple_metrics(a, c)
    peaks = [
        CVPeakResult(peak_id="pk-a", sweep=SWEEP_RISING, process=PROCESS_ANODIC, origin="automatic",
                     enabled=True, e_peak_v=a.potential_v, i_peak_raw_a=a.i_peak_raw_a,
                     i_peak_corrected_a=None, baseline=CVBaselineInfo.none(), prominence=8e-6),
        CVPeakResult(peak_id="pk-c", sweep=SWEEP_FALLING, process=PROCESS_CATHODIC, origin="automatic",
                     enabled=True, e_peak_v=c.potential_v, i_peak_raw_a=c.i_peak_raw_a,
                     i_peak_corrected_a=None, baseline=CVBaselineInfo.none(), prominence=7e-6),
    ]
    kwargs = dict(
        source_dataset_id="dataset-1",
        x_column="Potential/V",
        y_column="Current/A",
        sign_convention="anodic_positive",
        sweeps=segs,
        peaks=peaks,
        couple=cm,
        cycle_index=1,
        source_panel_id="panel-1",
        parameters={"detection": {"prominence": 1e-6}},
    )
    kwargs.update(overrides)
    return build_cv_cycle_analysis_result(**kwargs)


# --- construction / provenance -----------------------------------------


def test_build_uses_native_gnovi_engine_fields_and_fresh_id():
    a = _reversible_result()
    b = _reversible_result()
    assert a.engine == ENGINE_GNOVI
    assert a.engine_version == APP_VERSION
    assert a.operation == CV_OPERATION_PEAK_ANALYSIS
    assert a.kind == "cv_peaks"
    assert a.result_id != b.result_id


def test_generic_provenance_is_inherited_not_duplicated():
    result = _reversible_result()
    # fields come from AnalysisResult
    assert result.source_dataset_id == "dataset-1"
    assert result.source_panel_id == "panel-1"
    assert result.x_column == "Potential/V"
    # provenance_details() is the shared base implementation
    labels = [row[0] for row in result.provenance_details()]
    assert "Source dataset ID" in labels
    assert "Engine" in labels


def test_couple_metrics_copied_onto_result():
    result = _reversible_result()
    assert result.delta_ep_v == pytest.approx(0.059, abs=0.003)
    assert result.e_half_v == pytest.approx(0.2205, abs=0.002)
    assert result.peak_current_ratio_ipa_over_ipc is not None
    assert result.peak_current_ratio_ipc_over_ipa is not None
    assert result.peak_current_ratio_basis is not None


def test_result_without_a_couple_has_none_metrics():
    result = _reversible_result(couple=None)
    assert result.delta_ep_v is None
    assert result.e_half_v is None
    assert result.peak_current_ratio_basis is None


# --- bounded details() vs unbounded detail_table() --------------------


def test_details_is_bounded_regardless_of_peak_count():
    few = _reversible_result()
    many = build_cv_cycle_analysis_result(
        source_dataset_id="d", x_column="E", y_column="I", sign_convention="anodic_positive",
        sweeps=few.sweeps,
        peaks=[peak_result_from_seed(CVPeakSeed.manual(0.1 + 0.001 * k, 1e-6)) for k in range(2000)],
    )
    assert len(many.details()) <= len(few.details()) + 1  # no per-peak rows
    assert len(many.details()) < 20


def test_detail_table_has_one_row_per_peak_with_expected_columns():
    result = _reversible_result()
    columns, rows = result.detail_table()
    assert columns[:5] == ["Peak #", "Sweep", "Process", "Origin", "Enabled"]
    assert "I_peak raw (A)" in columns
    assert "I_peak corrected (A)" in columns
    assert len(rows) == 2
    assert rows[0][0] == "1"
    assert rows[0][2] == PROCESS_ANODIC


def test_detail_table_labels_a_missing_baseline_as_raw_extremum():
    result = _reversible_result()
    _cols, rows = result.detail_table()
    baseline_col = result.detail_table()[0].index("Baseline")
    corrected_col = result.detail_table()[0].index("I_peak corrected (A)")
    assert "raw extremum" in rows[0][baseline_col]
    assert rows[0][corrected_col] == "—"


def test_detail_table_row_count_scales_unlike_details():
    result = build_cv_cycle_analysis_result(
        source_dataset_id="d", x_column="E", y_column="I", sign_convention="anodic_positive",
        sweeps=[], peaks=[peak_result_from_seed(CVPeakSeed.manual(float(k), 1e-6)) for k in range(400)],
    )
    _cols, rows = result.detail_table()
    assert len(rows) == 400
    assert len(result.details()) < 20


def test_summary_is_human_readable():
    result = _reversible_result()
    s = result.summary()
    assert "cycle 1" in s
    assert "ΔEp" in s and "E½" in s


# --- serialization ----------------------------------------------------


def test_to_dict_is_json_safe():
    data = _reversible_result().to_dict()
    json.dumps(data)
    assert data["kind"] == "cv_peaks"
    assert data["sign_convention"] == "anodic_positive"
    assert len(data["peaks"]) == 2
    assert len(data["sweeps"]) == 2


def test_from_dict_round_trip_preserves_everything():
    original = _reversible_result()
    restored = CVCycleAnalysisResult.from_dict(original.to_dict())
    assert restored.result_id == original.result_id
    assert restored.sign_convention == original.sign_convention
    assert restored.cycle_index == original.cycle_index
    assert restored.cycle_complete == original.cycle_complete
    assert [s.direction for s in restored.sweeps] == [s.direction for s in original.sweeps]
    assert [p.peak_id for p in restored.peaks] == [p.peak_id for p in original.peaks]
    assert [p.process for p in restored.peaks] == [p.process for p in original.peaks]
    assert restored.delta_ep_v == pytest.approx(original.delta_ep_v)
    assert restored.e_half_v == pytest.approx(original.e_half_v)
    assert restored.peak_current_ratio_ipa_over_ipc == pytest.approx(
        original.peak_current_ratio_ipa_over_ipc
    )
    assert restored.peak_current_ratio_basis == original.peak_current_ratio_basis


def test_baseline_metadata_survives_round_trip():
    result = _reversible_result()
    result.peaks[0].baseline = CVBaselineInfo(
        method="linear", anchor_ranges=[(10, 40), (120, 160)], baseline_current_a=3.1e-7
    )
    result.peaks[0].i_peak_corrected_a = 9.7e-6
    restored = CVCycleAnalysisResult.from_dict(result.to_dict())
    bl = restored.peaks[0].baseline
    assert bl.method == "linear"
    assert bl.anchor_ranges == [(10, 40), (120, 160)]
    assert bl.baseline_current_a == pytest.approx(3.1e-7)
    assert restored.peaks[0].i_peak_corrected_a == pytest.approx(9.7e-6)


def test_registered_with_the_polymorphic_kind_registry():
    original = _reversible_result()
    restored = result_from_dict(original.to_dict())
    assert isinstance(restored, CVCycleAnalysisResult)
    assert restored.result_id == original.result_id


def test_from_dict_tolerates_a_project_saved_before_engine_fields_existed():
    data = _reversible_result().to_dict()
    for key in ("engine", "engine_version", "operation", "parameters"):
        del data[key]
    restored = CVCycleAnalysisResult.from_dict(data)
    assert restored.engine == ENGINE_GNOVI
    assert restored.engine_version is None
    assert restored.operation == CV_OPERATION_PEAK_ANALYSIS
    assert restored.parameters == {}


def test_deepcopy_is_independent():
    original = _reversible_result()
    cloned = copy.deepcopy(original)
    cloned.peaks[0].enabled = False
    assert original.peaks[0].enabled is True
    assert cloned.sweeps is not original.sweeps


def test_cv_sweep_info_from_segment():
    seg = segment_sweeps(_triangle(1))[0]
    info = CVSweepInfo.from_segment(seg)
    assert info.direction == seg.direction
    assert info.start == seg.start
    assert CVSweepInfo.from_dict(info.to_dict()) == info


# --- PanelResultHistory ---------------------------------------------


def test_panel_result_history_round_trip():
    history = PanelResultHistory()
    result = _reversible_result(source_panel_id="panel-1")
    history.add("panel-1", result)

    restored = PanelResultHistory.from_dict(history.to_dict())
    restored_result = restored.current("panel-1")
    assert isinstance(restored_result, CVCycleAnalysisResult)
    assert restored_result.result_id == result.result_id
    assert [p.process for p in restored_result.peaks] == [p.process for p in result.peaks]


def test_panel_result_history_copy_panel_remaps_panel_id():
    history = PanelResultHistory()
    result = _reversible_result(source_panel_id="panel-1")
    history.add("panel-1", result)
    extracted = history.copy_panel("panel-1", "panel-2")
    assert extracted.current("panel-2").source_panel_id == "panel-2"
    assert extracted.current("panel-2").result_id == result.result_id
    assert history.current("panel-1").source_panel_id == "panel-1"


# --- real project save / reopen -----------------------------------


def _project_with_cv_result():
    df = pd.DataFrame({"Potential/V": [-0.2, 0.0, 0.2, 0.4, 0.2, 0.0, -0.2],
                       "Current/A": [1e-7, 2e-7, 9e-6, 3e-7, -8e-6, -2e-7, -1e-7]})
    dataset = Dataset(name="cv run", dataframe=df)
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id
    result = _reversible_result(source_dataset_id=dataset.id, source_panel_id=panel_id)
    workbench.analysis_results.add(panel_id, result)
    return project, dataset, panel_id, result


def test_save_reopen_preserves_a_cv_analysis_result(tmp_path):
    project, _dataset, panel_id, result = _project_with_cv_result()
    reloaded = load_project(save_project(project, tmp_path / "cv.gnovi"))
    reloaded_result = reloaded.workbenches[0].analysis_results.current(panel_id)
    assert isinstance(reloaded_result, CVCycleAnalysisResult)
    assert reloaded_result.result_id == result.result_id
    assert reloaded_result.sign_convention == "anodic_positive"
    assert reloaded_result.delta_ep_v == pytest.approx(result.delta_ep_v)
    assert [p.peak_id for p in reloaded_result.peaks] == [p.peak_id for p in result.peaks]


def test_cv_result_persistence_needed_no_project_format_bump():
    assert PROJECT_FORMAT_VERSION == 3


def test_a_project_saved_before_cv_existed_still_loads(tmp_path):
    from gnovi_plot.analysis.fitting import fit_curve

    df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]})
    dataset = Dataset(name="plain", dataframe=df)
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id
    fit_result = fit_curve([0, 1, 2, 3], [0, 1, 2, 3], "linear",
                           source_dataset_id=dataset.id, x_column="x", y_column="y",
                           source_panel_id=panel_id)
    workbench.analysis_results.add(panel_id, fit_result)

    reloaded = load_project(save_project(project, tmp_path / "fit_only.gnovi"))
    reloaded_result = reloaded.workbenches[0].analysis_results.current(panel_id)
    assert reloaded_result.kind == "fit"
    assert reloaded_result.result_id == fit_result.result_id


def test_cycle_confidence_values_persist():
    for confidence in (CYCLE_CONFIDENCE_DETECTED, CYCLE_CONFIDENCE_MANUAL):
        result = _reversible_result(cycle_confidence=confidence)
        assert CVCycleAnalysisResult.from_dict(result.to_dict()).cycle_confidence == confidence
