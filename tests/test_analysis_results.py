from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar

import pytest

from gnovi_plot.analysis.results import ENGINE_GNOVI, AnalysisResult


@dataclass
class _DummyResult(AnalysisResult):
    """Minimal concrete subclass -- exercises the base contract without
    depending on `analysis.fitting`, so these tests stay about
    `AnalysisResult` itself."""

    kind: ClassVar[str] = "dummy"
    value: float = 0.0

    def summary(self) -> str:
        return f"dummy: {self.value}"

    def details(self) -> list[tuple[str, str]]:
        return [("Value", str(self.value))]

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["value"] = self.value
        return data


def _make(**overrides) -> _DummyResult:
    defaults = dict(
        source_dataset_id="dataset-123",
        source_dataset_name="Ferricyanide 50 mV/s",
        source_series_id="series-456",
        source_series_label="Current vs Potential",
        x_column="time",
        y_column="signal",
        row_range=(2, 8),
        source_panel_id="panel-1",
        result_id="result-1",
        engine=ENGINE_GNOVI,
        engine_version="0.9.0",
        operation="dummy_op",
        parameters={},
    )
    defaults.update(overrides)
    return _DummyResult(**defaults)


def test_base_class_summary_and_details_are_not_implemented():
    result = AnalysisResult(
        source_dataset_id="dataset-123",
        source_dataset_name=None,
        source_series_id=None,
        source_series_label=None,
        x_column="x",
        y_column="y",
        row_range=None,
        source_panel_id=None,
        result_id="result-1",
        engine=ENGINE_GNOVI,
        engine_version=None,
        operation="test",
        parameters={},
    )
    with pytest.raises(NotImplementedError):
        result.summary()
    with pytest.raises(NotImplementedError):
        result.details()


def test_base_class_provenance_details_has_a_real_default_implementation():
    """Unlike summary()/details(), provenance_details() has one correct
    shared implementation on the base class -- no subclass needs to
    override it."""
    result = AnalysisResult(
        source_dataset_id="dataset-123",
        source_dataset_name="doesn't matter here -- always ids",
        source_series_id="series-456",
        source_series_label="doesn't matter here either",
        x_column="x",
        y_column="y",
        row_range=(2, 8),
        source_panel_id="panel-1",
        result_id="result-1",
        engine=ENGINE_GNOVI,
        engine_version="0.9.0",
        operation="test",
        parameters={},
    )
    rows = dict(result.provenance_details())
    assert rows["Source dataset ID"] == "dataset-123"
    assert rows["Source series ID"] == "series-456"
    assert rows["Columns"] == "x → y"
    assert rows["Row range"] == "2–8"
    assert rows["Engine"] == "gnovi 0.9.0"
    assert rows["Operation"] == "test"
    # Never names -- provenance_details() is always raw ids.
    assert "Ferricyanide" not in str(rows)


def test_base_class_provenance_details_omits_optional_rows():
    result = AnalysisResult(
        source_dataset_id="dataset-123",
        source_dataset_name=None,
        source_series_id=None,
        source_series_label=None,
        x_column="x",
        y_column="y",
        row_range=None,
        source_panel_id=None,
        result_id="result-1",
        engine=ENGINE_GNOVI,
        engine_version=None,
        operation="test",
        parameters={},
    )
    rows = dict(result.provenance_details())
    assert "Source series ID" not in rows
    assert "Row range" not in rows


def test_provenance_details_omits_engine_version_when_none():
    result = _make(engine_version=None)
    rows = dict(result.provenance_details())
    assert rows["Engine"] == "gnovi"


def test_engine_defaults_are_native_gnovi_and_round_trip_through_to_dict():
    result = _make(engine=ENGINE_GNOVI, engine_version="0.9.0", operation="dummy_op", parameters={"a": 1})
    assert result.engine == "gnovi"
    data = result.to_dict()
    assert data["engine"] == "gnovi"
    assert data["engine_version"] == "0.9.0"
    assert data["operation"] == "dummy_op"
    assert data["parameters"] == {"a": 1}
    json.dumps(data)  # parameters must stay JSON-safe


def test_engine_provenance_survives_deepcopy():
    import copy

    result = _make(parameters={"lam": 100000.0, "nested": {"x": 1}})
    cloned = copy.deepcopy(result)
    assert cloned.engine == result.engine
    assert cloned.parameters == result.parameters
    assert cloned.parameters is not result.parameters
    cloned.parameters["lam"] = -1.0
    assert result.parameters["lam"] == 100000.0  # independent copy, not shared


def test_base_class_does_not_support_residuals_by_default():
    result = _make()
    assert result.supports_residuals() is False
    with pytest.raises(NotImplementedError):
        result.compute_residuals([1, 2], [1, 2])


def test_base_kind_is_generic():
    assert AnalysisResult.kind == "analysis"


def test_subclass_kind_overrides_base():
    assert _DummyResult.kind == "dummy"
    assert _make().kind == "dummy"


def test_provenance_uses_stable_ids_as_the_authoritative_identifier():
    result = _make(source_dataset_id="dataset-123", source_series_id="series-456")
    assert result.source_dataset_id == "dataset-123"
    assert result.source_series_id == "series-456"


def test_provenance_also_carries_a_descriptive_name_snapshot():
    """Names/labels are a snapshot of what was fitted, alongside (never
    instead of) the stable ids."""
    result = _make(
        source_dataset_name="Ferricyanide 50 mV/s",
        source_series_label="Current vs Potential",
    )
    assert result.source_dataset_name == "Ferricyanide 50 mV/s"
    assert result.source_series_label == "Current vs Potential"
    # Both id and name fields coexist -- neither replaces the other.
    assert set(vars(result).keys()) == {
        "source_dataset_id",
        "source_dataset_name",
        "source_series_id",
        "source_series_label",
        "x_column",
        "y_column",
        "row_range",
        "source_panel_id",
        "result_id",
        "engine",
        "engine_version",
        "operation",
        "parameters",
        "value",
    }


def test_name_snapshot_fields_are_optional():
    """Older/degenerate results (or ones with no natural series involved)
    can have no name snapshot at all -- must not be required."""
    result = _make(source_dataset_name=None, source_series_label=None)
    assert result.source_dataset_name is None
    assert result.source_series_label is None
    assert result.source_dataset_id == "dataset-123"  # ids stay required


def test_source_series_id_is_optional():
    result = _make(source_series_id=None, source_series_label=None)
    assert result.source_series_id is None
    assert result.to_dict()["source_series_id"] is None


def test_to_dict_is_json_safe_and_includes_kind_and_names():
    result = _make(row_range=(2, 8))
    data = result.to_dict()
    assert data["kind"] == "dummy"
    assert data["source_dataset_id"] == "dataset-123"
    assert data["source_dataset_name"] == "Ferricyanide 50 mV/s"
    assert data["source_series_id"] == "series-456"
    assert data["source_series_label"] == "Current vs Potential"
    assert data["x_column"] == "time"
    assert data["y_column"] == "signal"
    # row_range comes back as a list (JSON has no tuple type), matching
    # PlotSeries.to_dict()'s own convention.
    assert data["row_range"] == [2, 8]
    assert isinstance(data["row_range"], list)
    json.dumps(data)  # must not raise


def test_to_dict_row_range_none_stays_none():
    result = _make(row_range=None)
    assert result.to_dict()["row_range"] is None


def test_source_panel_id_is_carried_and_included_in_to_dict():
    result = _make(source_panel_id="panel-1")
    assert result.source_panel_id == "panel-1"
    assert result.to_dict()["source_panel_id"] == "panel-1"


def test_source_panel_id_can_be_none():
    """A result not associated with any panel (defensive/future case) --
    must not be required to have one."""
    result = _make(source_panel_id=None)
    assert result.source_panel_id is None
    assert result.to_dict()["source_panel_id"] is None


def test_subclass_to_dict_extends_base_fields():
    result = _make(value=3.5)
    data = result.to_dict()
    assert data["value"] == 3.5
    # base fields still present alongside the subclass's own
    assert data["source_dataset_id"] == "dataset-123"


def test_report_text_is_generic_and_includes_names_when_given():
    result = _make()
    text = result.report_text(dataset_name="Ferricyanide 50 mV/s", series_label="Current vs Potential")
    assert "dummy: 0.0" in text
    assert "Dataset: Ferricyanide 50 mV/s" in text
    assert "Series: Current vs Potential" in text
    assert "Value: 0.0" in text
    # report_text() is a names-only summary -- never surfaces raw ids.
    assert "dataset-123" not in text
    assert "series-456" not in text


def test_report_text_omits_name_lines_when_not_given():
    result = _make()
    text = result.report_text()
    assert "Dataset:" not in text
    assert "Series:" not in text
    assert "dummy: 0.0" in text


# --- result_id: this result's own stable identity ----------------------------


def test_result_id_is_carried_and_included_in_to_dict():
    result = _make(result_id="my-result-id")
    assert result.result_id == "my-result-id"
    assert result.to_dict()["result_id"] == "my-result-id"


# --- polymorphic dispatch: register_result_kind / result_from_dict -----------


def test_result_from_dict_raises_for_an_unrecognized_kind():
    from gnovi_plot.analysis.results import result_from_dict

    with pytest.raises(ValueError):
        result_from_dict({"kind": "does-not-exist"})


def test_result_from_dict_raises_for_a_missing_kind():
    from gnovi_plot.analysis.results import result_from_dict

    with pytest.raises(ValueError):
        result_from_dict({})


def test_register_result_kind_makes_a_subclass_dispatchable():
    """Exercises the registry mechanism itself with a throwaway subclass
    -- `FitResult`'s own real registration is covered in test_fitting.py,
    this proves the generic decorator/dispatch machinery works for *any*
    subclass, not just the one that exists today."""
    from gnovi_plot.analysis.results import register_result_kind, result_from_dict

    @dataclass
    class _RegisteredDummy(AnalysisResult):
        kind: ClassVar[str] = "registered-dummy-for-test"
        note: str = ""

        def summary(self) -> str:
            return self.note

        def details(self) -> list[tuple[str, str]]:
            return []

        def to_dict(self) -> dict:
            data = super().to_dict()
            data["note"] = self.note
            return data

        @classmethod
        def from_dict(cls, data: dict) -> "_RegisteredDummy":
            return cls(
                source_dataset_id=data["source_dataset_id"],
                source_dataset_name=None,
                source_series_id=None,
                source_series_label=None,
                x_column=data["x_column"],
                y_column=data["y_column"],
                row_range=None,
                source_panel_id=None,
                result_id=data["result_id"],
                engine=data.get("engine", ENGINE_GNOVI),
                engine_version=data.get("engine_version"),
                operation=data.get("operation", ""),
                parameters=dict(data.get("parameters", {})),
                note=data["note"],
            )

    register_result_kind(_RegisteredDummy)
    original = _RegisteredDummy(
        source_dataset_id="d",
        source_dataset_name=None,
        source_series_id=None,
        source_series_label=None,
        x_column="x",
        y_column="y",
        row_range=None,
        source_panel_id=None,
        result_id="r1",
        engine=ENGINE_GNOVI,
        engine_version=None,
        operation="test",
        parameters={},
        note="hello",
    )

    restored = result_from_dict(original.to_dict())

    assert isinstance(restored, _RegisteredDummy)
    assert restored.result_id == "r1"
    assert restored.note == "hello"

    # A project saved before engine/operation/parameters existed has none
    # of those keys at all -- from_dict must still reconstruct cleanly
    # (see AnalysisResult's own docstring on why this needed no
    # PROJECT_FORMAT_VERSION bump).
    old_saved_dict = original.to_dict()
    del old_saved_dict["engine"]
    del old_saved_dict["engine_version"]
    del old_saved_dict["operation"]
    del old_saved_dict["parameters"]
    restored_old = result_from_dict(old_saved_dict)
    assert restored_old.engine == ENGINE_GNOVI
    assert restored_old.engine_version is None
    assert restored_old.parameters == {}
