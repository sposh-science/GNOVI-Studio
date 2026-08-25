"""modules.xrd.results.XRDAnalysisResult: construction, serialization,
polymorphic registry dispatch, and PanelResultHistory integration (Extract/
Focus-compatible copy, save/reopen shape)."""

from __future__ import annotations

import copy
import json

import pytest

from gnovi_plot.analysis.panel_results import PanelResultHistory
from gnovi_plot.analysis.results import ENGINE_GNOVI, result_from_dict
from gnovi_plot.core.app_info import __version__ as APP_VERSION
from gnovi_plot.modules.xrd.peaks import XRDPeakSeed
from gnovi_plot.modules.xrd.radiation import radiation_from_preset
from gnovi_plot.modules.xrd.results import (
    OPERATION_PEAK_DETECTION,
    XRDAnalysisResult,
    build_xrd_analysis_result,
)


def _peaks() -> list[XRDPeakSeed]:
    return [
        XRDPeakSeed(two_theta=28.4, intensity=800.0, origin="automatic", index=100, prominence=750.0),
        XRDPeakSeed(two_theta=47.3, intensity=350.0, origin="automatic", index=250, prominence=300.0),
    ]


def _build(**overrides) -> XRDAnalysisResult:
    defaults = dict(
        source_dataset_id="dataset-1",
        x_column="2theta",
        y_column="counts",
        radiation=radiation_from_preset("cu_ka1"),
        peaks=_peaks(),
        source_panel_id="panel-1",
        parameters={"detection": {"prominence": 100.0}},
    )
    defaults.update(overrides)
    return build_xrd_analysis_result(**defaults)


# --- construction / engine-neutral provenance -------------------------------


def test_build_result_uses_native_gnovi_engine_fields():
    result = _build()
    assert result.engine == ENGINE_GNOVI
    assert result.engine_version == APP_VERSION
    assert result.operation == OPERATION_PEAK_DETECTION
    assert result.kind == "xrd_peaks"


def test_build_result_generates_a_fresh_result_id_every_call():
    a = _build()
    b = _build()
    assert a.result_id != b.result_id


def test_summary_and_details_are_human_readable():
    result = _build()
    assert "peak candidate" in result.summary()
    labels = [row[0] for row in result.details()]
    assert "Radiation" in labels
    assert any(label.startswith("Peak 1") for label in labels)


# --- serialization -----------------------------------------------------------


def test_to_dict_is_json_safe():
    result = _build()
    data = result.to_dict()
    json.dumps(data)  # must not raise
    assert data["kind"] == "xrd_peaks"
    assert data["radiation"]["label"] == "Cu Ka1"
    assert len(data["peaks"]) == 2


def test_from_dict_round_trip_preserves_peaks_and_radiation():
    original = _build()
    restored = XRDAnalysisResult.from_dict(original.to_dict())

    assert restored.result_id == original.result_id
    assert restored.radiation == original.radiation
    assert [p.two_theta for p in restored.peaks] == [p.two_theta for p in original.peaks]
    assert restored.engine == ENGINE_GNOVI
    assert restored.operation == OPERATION_PEAK_DETECTION


def test_registered_with_the_polymorphic_kind_registry():
    original = _build()
    restored = result_from_dict(original.to_dict())
    assert isinstance(restored, XRDAnalysisResult)
    assert restored.result_id == original.result_id


def test_from_dict_tolerates_a_project_saved_before_engine_fields_existed():
    """Backward compatibility for an OLD saved project: no
    PROJECT_FORMAT_VERSION bump was needed for engine/engine_version/
    operation/parameters (see AnalysisResult's own docstring) -- confirm a
    dict missing those keys still reconstructs with sensible defaults."""
    data = _build().to_dict()
    del data["engine"]
    del data["engine_version"]
    del data["operation"]
    del data["parameters"]

    restored = XRDAnalysisResult.from_dict(data)

    assert restored.engine == ENGINE_GNOVI
    assert restored.engine_version is None
    assert restored.operation == OPERATION_PEAK_DETECTION
    assert restored.parameters == {}


# --- deepcopy safety ----------------------------------------------------------


def test_deepcopy_produces_an_independent_result():
    original = _build()
    cloned = copy.deepcopy(original)

    assert cloned.peaks[0].two_theta == original.peaks[0].two_theta
    assert cloned.peaks is not original.peaks
    assert cloned.peaks[0] is not original.peaks[0]

    cloned.peaks[0].enabled = False
    assert original.peaks[0].enabled is True


# --- PanelResultHistory integration (save/reopen shape, current selection) --


def test_panel_result_history_add_and_current():
    history = PanelResultHistory()
    result = _build(source_panel_id="panel-1")

    history.add("panel-1", result)

    assert history.current("panel-1") is result
    assert history.all("panel-1") == [result]


def test_panel_result_history_to_dict_from_dict_round_trip():
    history = PanelResultHistory()
    result = _build(source_panel_id="panel-1")
    history.add("panel-1", result)

    restored = PanelResultHistory.from_dict(history.to_dict())

    restored_result = restored.current("panel-1")
    assert isinstance(restored_result, XRDAnalysisResult)
    assert restored_result.result_id == result.result_id
    assert [p.two_theta for p in restored_result.peaks] == [p.two_theta for p in result.peaks]


# --- Extract-equivalent behavior: PanelResultHistory.copy_panel -------------


def test_copy_panel_gives_the_extracted_workbench_an_independent_history():
    """Mirrors exactly what `core.project.Project.extract_panel_to_
    workbench` does for a 2D FitResult -- XRDAnalysisResult needs no
    special-case handling in `copy_panel` at all, since it's plain
    `copy.deepcopy` plus a `source_panel_id` remap (see `PanelResultHistory
    .copy_panel`'s own docstring)."""
    history = PanelResultHistory()
    result = _build(source_panel_id="panel-1")
    history.add("panel-1", result)

    extracted = history.copy_panel("panel-1", "panel-2")

    extracted_result = extracted.current("panel-2")
    assert extracted_result.source_panel_id == "panel-2"
    assert extracted_result.result_id == result.result_id  # identity preserved, see copy_panel's docstring
    assert extracted_result is not result
    assert extracted_result.peaks is not result.peaks

    # The source workbench's own history is untouched.
    assert history.current("panel-1") is result
    assert history.current("panel-1").source_panel_id == "panel-1"


def test_copy_panel_preserves_the_current_selection_marker():
    history = PanelResultHistory()
    first = _build(source_panel_id="panel-1")
    second = _build(source_panel_id="panel-1")
    history.add("panel-1", first)
    history.add("panel-1", second)
    history.set_current("panel-1", first.result_id)

    extracted = history.copy_panel("panel-1", "panel-2")

    assert extracted.current("panel-2").result_id == first.result_id
