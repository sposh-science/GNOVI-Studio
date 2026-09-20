"""modules.raman.results.RamanAnalysisResult: construction, serialization,
and polymorphic registry dispatch -- mirrors test_xrd_results.py's
structure for the equivalent XRD result, minus anything involving
radiation/d-spacing (Raman has no equivalent) and minus PanelResultHistory/
project-persistence integration (full project integration belongs to
Issue #75, not this backend-only milestone)."""

from __future__ import annotations

import copy
import json

import pytest

from gnovi_plot.analysis.results import ENGINE_GNOVI, result_from_dict
from gnovi_plot.core.app_info import __version__ as APP_VERSION
from gnovi_plot.modules.raman.peaks import RamanPeakSeed
from gnovi_plot.modules.raman.results import (
    OPERATION_PEAK_DETECTION,
    RamanAnalysisResult,
    build_raman_analysis_result,
)


def _peaks() -> list[RamanPeakSeed]:
    return [
        RamanPeakSeed(raman_shift=1350.0, intensity=800.0, origin="automatic", index=100, prominence=750.0),
        RamanPeakSeed(raman_shift=1580.0, intensity=350.0, origin="automatic", index=250, prominence=300.0),
    ]


def _build(**overrides) -> RamanAnalysisResult:
    defaults = dict(
        source_dataset_id="dataset-1",
        x_column="raman_shift",
        y_column="intensity",
        peaks=_peaks(),
        source_panel_id="panel-1",
        parameters={"detection": {"prominence": 100.0}},
    )
    defaults.update(overrides)
    return build_raman_analysis_result(**defaults)


# --- construction / engine-neutral provenance -------------------------------


def test_build_result_uses_native_gnovi_engine_fields():
    result = _build()
    assert result.engine == ENGINE_GNOVI
    assert result.engine_version == APP_VERSION
    assert result.operation == OPERATION_PEAK_DETECTION
    assert result.kind == "raman_peaks"


def test_build_result_generates_a_fresh_result_id_every_call():
    a = _build()
    b = _build()
    assert a.result_id != b.result_id


def test_summary_and_details_are_human_readable():
    result = _build()
    assert "peak candidate" in result.summary()
    labels = [row[0] for row in result.details()]
    assert "Peak candidates" in labels
    assert "Prominence" in labels
    # details() is deliberately a fixed, bounded set of rows -- never one
    # row per peak (mirrors XRDAnalysisResult.details()'s own layout-bug
    # rationale). The full per-peak view lives in detail_table() instead.
    assert not any(label.startswith("Peak 1") or label.startswith("Peak 2") for label in labels)


def test_details_does_not_assume_a_preprocessing_shape():
    # This milestone has no preprocessing GUI yet (see the module
    # docstring) -- details() must not reference a "preprocessing" key
    # that nothing produces yet, unlike XRDAnalysisResult.details(),
    # which co-evolved with its own Background/Smoothing controls.
    result = _build(parameters={})
    labels = [row[0] for row in result.details()]
    assert "Background" not in labels
    assert "Smoothing" not in labels


def test_details_row_count_does_not_scale_with_peak_count():
    few = _build()  # 2 peaks, see _build()
    many = _build(peaks=[RamanPeakSeed.manual(float(i), 100.0) for i in range(1000)])
    assert len(many.details()) == len(few.details())


# --- detailed peak table (rendered in the bottom Results tab) ----------------


def test_detail_table_has_one_row_per_peak_with_expected_columns():
    result = _build()  # 2 automatic peaks
    columns, rows = result.detail_table()
    assert columns == [
        "Peak #",
        "Raman shift (cm⁻¹)",
        "Observed intensity",
        "Prominence",
        "Origin",
        "Enabled",
    ]
    assert len(rows) == len(result.peaks)
    assert rows[0][0] == "1"
    assert rows[0][4] == "automatic"
    assert rows[0][5] == "Yes"


def test_detail_table_row_count_scales_with_peaks_unlike_details():
    result = _build(peaks=[RamanPeakSeed.manual(float(100 + i), 100.0) for i in range(500)])
    _columns, rows = result.detail_table()
    assert len(rows) == 500  # unlike details(), this IS allowed to be long
    assert len(result.details()) < 20  # the compact summary stays bounded


def test_detail_table_reflects_enabled_flag():
    result = _build()
    result.peaks[0].enabled = False
    rows = result.detail_table()[1]
    assert rows[0][5] == "No"
    assert rows[1][5] == "Yes"


def test_detail_table_never_labels_a_column_fwhm_or_linewidth():
    # Scientific requirement (Issue #72): no fitted quantity exists in
    # this milestone, so no column may claim to be one.
    columns, _rows = result_from_dict(_build().to_dict()).detail_table()
    forbidden = {"fwhm", "linewidth", "fitted width", "raman fwhm"}
    assert not any(col.strip().lower() in forbidden for col in columns)


# --- serialization -----------------------------------------------------------


def test_to_dict_is_json_safe():
    result = _build()
    data = result.to_dict()
    json.dumps(data)  # must not raise
    assert data["kind"] == "raman_peaks"
    assert len(data["peaks"]) == 2


def test_from_dict_round_trip_preserves_peaks_and_parameters():
    original = _build()
    restored = RamanAnalysisResult.from_dict(original.to_dict())

    assert restored.result_id == original.result_id
    assert [p.raman_shift for p in restored.peaks] == [p.raman_shift for p in original.peaks]
    assert [p.id for p in restored.peaks] == [p.id for p in original.peaks]
    assert [p.prominence for p in restored.peaks] == [p.prominence for p in original.peaks]
    assert [p.enabled for p in restored.peaks] == [p.enabled for p in original.peaks]
    assert restored.parameters == original.parameters
    assert restored.engine == ENGINE_GNOVI
    assert restored.operation == OPERATION_PEAK_DETECTION


def test_registered_with_the_polymorphic_kind_registry():
    original = _build()
    restored = result_from_dict(original.to_dict())
    assert isinstance(restored, RamanAnalysisResult)
    assert restored.result_id == original.result_id


def test_from_dict_tolerates_a_project_saved_before_engine_fields_existed():
    """Backward compatibility for an OLD saved project -- no
    PROJECT_FORMAT_VERSION bump was needed for engine/engine_version/
    operation/parameters (see AnalysisResult's own docstring); confirm a
    dict missing those keys still reconstructs with sensible defaults,
    mirroring the equivalent XRD test."""
    data = _build().to_dict()
    del data["engine"]
    del data["engine_version"]
    del data["operation"]
    del data["parameters"]

    restored = RamanAnalysisResult.from_dict(data)

    assert restored.engine == ENGINE_GNOVI
    assert restored.engine_version is None
    assert restored.operation == OPERATION_PEAK_DETECTION
    assert restored.parameters == {}


# --- deepcopy safety ----------------------------------------------------------


def test_deepcopy_produces_an_independent_result():
    original = _build()
    cloned = copy.deepcopy(original)

    assert cloned.peaks[0].raman_shift == original.peaks[0].raman_shift
    assert cloned.peaks is not original.peaks
    assert cloned.peaks[0] is not original.peaks[0]

    cloned.peaks[0].enabled = False
    assert original.peaks[0].enabled is True


# --- scientific-labeling guard -------------------------------------------------


def test_summary_never_claims_a_fitted_measurement():
    result = _build()
    text = result.summary().lower()
    for forbidden in ("fwhm", "linewidth", "fitted"):
        assert forbidden not in text
