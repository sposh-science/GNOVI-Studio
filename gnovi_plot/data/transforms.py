from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class CalculatedColumnInfo:
    """Metadata retained for a calculated column, kept alongside the working
    DataFrame so it survives independently of the column data itself."""

    name: str
    formula: str
    source_columns: list[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class Transformation:
    """One structured entry in a Dataset's transformation history.

    `kind` is one of "calculated_column", "exclude_rows", "keep_rows", "reset".
    `detail` holds the structured payload needed to describe or, later,
    replay the operation (e.g. {"name": ..., "formula": ...} or
    {"row_positions": [...]})  -- kept as plain data (no Dataset/DataFrame
    references) so it stays cheap to store and could back a future
    project save/load or replay feature without rework.
    """

    kind: str
    description: str
    detail: dict
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def describe_row_positions(positions: list[int]) -> str:
    """Compact human-readable summary of positional row indices, collapsing
    contiguous runs into ranges, e.g. [124..131, 200] -> "124-131, 200"."""
    if not positions:
        return "(none)"
    ordered = sorted(set(positions))
    runs: list[str] = []
    start = prev = ordered[0]
    for pos in ordered[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        runs.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = pos
    runs.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(runs)
