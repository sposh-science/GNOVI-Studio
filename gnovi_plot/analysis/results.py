from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class AnalysisResult:
    """Base for the output of any analysis tool run against a Dataset's
    data -- curve fitting today (see `analysis.fitting.FitResult`), and
    later tools (statistics, peak analysis, FFT, smoothing, ...) that don't
    exist yet.

    Provenance (`source_dataset_id` / `source_series_id`) is always a
    stable internal id -- `Dataset.id` / `PlotSeries.id` -- never a
    filename, label, or display name. Those ids are resolved against
    `DatasetManager`/the active `Panel` by whatever's displaying the
    result, so renaming the source dataset later never breaks the link.
    `source_series_id` is optional: a result can be computed directly
    against a dataset's column pair with no live plotted series involved.

    `row_range` records which rows of the source were actually used
    (positional, end-exclusive, `DataFrame.iloc`-style), mirroring
    `PlotSeries.row_range`; `None` means the whole dataset.

    Pure data plus a small display contract (`summary`/`details`) any
    results view can render without knowing the concrete subclass. No Qt,
    no plotting, no project I/O.
    """

    source_dataset_id: str
    source_series_id: str | None
    x_column: str
    y_column: str
    row_range: tuple[int, int] | None

    kind: ClassVar[str] = "analysis"

    def summary(self) -> str:
        """One-line description for a compact header/list entry."""
        raise NotImplementedError

    def details(self) -> list[tuple[str, str]]:
        """Label/value rows for a full results display."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """JSON-safe representation -- base provenance fields plus `kind`.
        Subclasses extend this (see `FitResult.to_dict`) so the combined
        dict can be embedded directly in a derived `Dataset.metadata` with
        no `project.json` schema change: `metadata` is already a free-form
        dict that's serialized whole.
        """
        return {
            "kind": self.kind,
            "source_dataset_id": self.source_dataset_id,
            "source_series_id": self.source_series_id,
            "x_column": self.x_column,
            "y_column": self.y_column,
            "row_range": list(self.row_range) if self.row_range is not None else None,
        }
