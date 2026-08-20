from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from gnovi_plot.analysis.results import AnalysisResult


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
        source_series_id="series-456",
        x_column="time",
        y_column="signal",
        row_range=(2, 8),
    )
    defaults.update(overrides)
    return _DummyResult(**defaults)


def test_base_class_summary_and_details_are_not_implemented():
    result = AnalysisResult(
        source_dataset_id="dataset-123",
        source_series_id=None,
        x_column="x",
        y_column="y",
        row_range=None,
    )
    with pytest.raises(NotImplementedError):
        result.summary()
    with pytest.raises(NotImplementedError):
        result.details()


def test_base_kind_is_generic():
    assert AnalysisResult.kind == "analysis"


def test_subclass_kind_overrides_base():
    assert _DummyResult.kind == "dummy"
    assert _make().kind == "dummy"


def test_provenance_uses_stable_ids_not_labels():
    result = _make(source_dataset_id="dataset-123", source_series_id="series-456")
    assert result.source_dataset_id == "dataset-123"
    assert result.source_series_id == "series-456"
    # Nothing in the base class stores a display name/filename/label --
    # only the two id fields and the column/row-range provenance.
    assert set(vars(result).keys()) == {
        "source_dataset_id",
        "source_series_id",
        "x_column",
        "y_column",
        "row_range",
        "value",
    }


def test_source_series_id_is_optional():
    result = _make(source_series_id=None)
    assert result.source_series_id is None
    assert result.to_dict()["source_series_id"] is None


def test_to_dict_is_json_safe_and_includes_kind():
    import json

    result = _make(row_range=(2, 8))
    data = result.to_dict()
    assert data["kind"] == "dummy"
    assert data["source_dataset_id"] == "dataset-123"
    assert data["source_series_id"] == "series-456"
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


def test_subclass_to_dict_extends_base_fields():
    result = _make(value=3.5)
    data = result.to_dict()
    assert data["value"] == 3.5
    # base fields still present alongside the subclass's own
    assert data["source_dataset_id"] == "dataset-123"
