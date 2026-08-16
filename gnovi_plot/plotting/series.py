from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from gnovi_plot.data.dataset import Dataset

# Future plot types (KDE, violin, box, bar, cumulative frequency) are
# intentionally not modeled yet -- only add a member when the type is
# actually implemented.


class PlotType(str, Enum):
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"


@dataclass
class PlotSeries:
    """A plotted item, independent of any live Matplotlib artist.

    LINE/SCATTER use x_column + y_column. HISTOGRAM uses only x_column (the
    numeric column whose distribution is plotted); y_column must be None.

    `row_range` optionally restricts the series to a contiguous slice of
    `dataset.dataframe` -- (start, end) positional indices, end exclusive,
    as used by `DataFrame.iloc`. This is how a segment of a dataset (e.g. a
    single cyclic-voltammetry cycle) is represented without splitting or
    copying the source Dataset: the same Dataset is shared by every
    segment's PlotSeries, and only the row selection differs. `None` means
    the whole dataset.
    """

    dataset: Dataset
    plot_type: PlotType
    label: str
    x_column: str
    y_column: str | None = None
    row_range: tuple[int, int] | None = None
    visible: bool = True
    color: str | None = None
    # True once the user has explicitly picked `color` (see
    # gui.widgets.plot_series_panel._pick_color) -- distinguishes an
    # explicit choice (never silently changed, see
    # plotting.backends.matplotlib_backend.is_low_contrast) from an
    # auto-assigned cycle color (which may be re-picked from a
    # theme-appropriate cycle, see plotting.figure.theme_color_cycle).
    color_is_manual: bool = False
    line_width: float = 1.5
    line_style: str = "-"
    marker: str = ""
    marker_size: float = 6.0
    marker_filled: bool = True
    marker_edge_width: float = 1.0
    alpha: float = 1.0
    zorder: float = 2.0
    bins: int | str = "auto"
    hist_mode: str = "frequency"  # "frequency" | "percentage" | "cumulative"
    # Plotting-only transformations -- never applied to dataset/dataframe
    # data, only to the values handed to Matplotlib at draw time.
    y_offset: float = 0.0
    normalize_to_max: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Set by GnoviFigure.invalidate_series_for_dataset() after a dataset
    # transformation invalidates this series (missing column / stale row_range).
    stale: bool = False

    def __post_init__(self) -> None:
        if not self.x_column:
            raise ValueError("PlotSeries.x_column must not be empty")

        if self.plot_type in (PlotType.LINE, PlotType.SCATTER):
            if not self.y_column:
                raise ValueError(f"{self.plot_type.value} series requires a y_column")
        elif self.plot_type == PlotType.HISTOGRAM:
            if self.y_column is not None:
                raise ValueError("Histogram series must not have a y_column")
            if not (self.bins == "auto" or (isinstance(self.bins, int) and self.bins > 0)):
                raise ValueError("Histogram bins must be 'auto' or a positive integer")
            if self.hist_mode not in ("frequency", "percentage", "cumulative"):
                raise ValueError("Histogram hist_mode must be 'frequency', 'percentage' or 'cumulative'")

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("PlotSeries.alpha must be between 0.0 and 1.0")

        if self.row_range is not None:
            start, end = self.row_range
            if not (0 <= start < end <= self.dataset.row_count):
                raise ValueError(
                    f"PlotSeries.row_range {self.row_range} is out of bounds for dataset "
                    f"with {self.dataset.row_count} rows"
                )

    @property
    def dataframe(self) -> pd.DataFrame:
        """The dataset rows this series covers.

        The full `dataset.dataframe`, or -- for a segment such as one
        detected cycle -- just its `row_range` slice. `DataFrame.iloc`
        slicing never mutates `dataset.dataframe`.
        """
        if self.row_range is None:
            return self.dataset.dataframe
        start, end = self.row_range
        return self.dataset.dataframe.iloc[start:end]

    def to_dict(self) -> dict:
        """Project-save representation: stores `dataset_id` rather than a
        nested `Dataset` (or its DataFrame) -- the caller resolves it back
        to a shared live `Dataset` on load via `from_dict`'s
        `dataset_lookup` (see `core.project_io`)."""
        return {
            "dataset_id": self.dataset.id,
            "plot_type": self.plot_type.value,
            "label": self.label,
            "x_column": self.x_column,
            "y_column": self.y_column,
            "row_range": list(self.row_range) if self.row_range is not None else None,
            "visible": self.visible,
            "color": self.color,
            "color_is_manual": self.color_is_manual,
            "line_width": self.line_width,
            "line_style": self.line_style,
            "marker": self.marker,
            "marker_size": self.marker_size,
            "marker_filled": self.marker_filled,
            "marker_edge_width": self.marker_edge_width,
            "alpha": self.alpha,
            "zorder": self.zorder,
            "bins": self.bins,
            "hist_mode": self.hist_mode,
            "y_offset": self.y_offset,
            "normalize_to_max": self.normalize_to_max,
            "id": self.id,
            "stale": self.stale,
        }

    @classmethod
    def from_dict(cls, data: dict, dataset_lookup: dict[str, Dataset]) -> "PlotSeries | None":
        """Reconstruct from `to_dict`'s output, resolving `dataset_id`
        against `dataset_lookup`. Returns None -- rather than raising -- if
        `dataset_id` isn't in `dataset_lookup`, so one stale reference in an
        otherwise-valid project doesn't block loading the rest (the caller
        is expected to log/report skipped series; see `core.project_io`)."""
        dataset = dataset_lookup.get(data["dataset_id"])
        if dataset is None:
            return None
        row_range = data.get("row_range")
        return cls(
            dataset=dataset,
            plot_type=PlotType(data["plot_type"]),
            label=data["label"],
            x_column=data["x_column"],
            y_column=data.get("y_column"),
            row_range=tuple(row_range) if row_range is not None else None,
            visible=data.get("visible", True),
            color=data.get("color"),
            color_is_manual=data.get("color_is_manual", False),
            line_width=data.get("line_width", 1.5),
            line_style=data.get("line_style", "-"),
            marker=data.get("marker", ""),
            marker_size=data.get("marker_size", 6.0),
            marker_filled=data.get("marker_filled", True),
            marker_edge_width=data.get("marker_edge_width", 1.0),
            alpha=data.get("alpha", 1.0),
            zorder=data.get("zorder", 2.0),
            bins=data.get("bins", "auto"),
            hist_mode=data.get("hist_mode", "frequency"),
            y_offset=data.get("y_offset", 0.0),
            normalize_to_max=data.get("normalize_to_max", False),
            id=data["id"],
            stale=data.get("stale", False),
        )

    @classmethod
    def line(
        cls,
        dataset: Dataset,
        x_column: str,
        y_column: str,
        label: str | None = None,
        **overrides,
    ) -> "PlotSeries":
        overrides.setdefault("marker", "")
        return cls(
            dataset=dataset,
            plot_type=PlotType.LINE,
            x_column=x_column,
            y_column=y_column,
            label=label or f"{dataset.name} — {y_column}",
            **overrides,
        )

    @classmethod
    def scatter(
        cls,
        dataset: Dataset,
        x_column: str,
        y_column: str,
        label: str | None = None,
        **overrides,
    ) -> "PlotSeries":
        overrides.setdefault("marker", "o")
        return cls(
            dataset=dataset,
            plot_type=PlotType.SCATTER,
            x_column=x_column,
            y_column=y_column,
            label=label or f"{dataset.name} — {y_column}",
            **overrides,
        )

    @classmethod
    def histogram(
        cls,
        dataset: Dataset,
        column: str,
        label: str | None = None,
        bins: int | str = "auto",
        **overrides,
    ) -> "PlotSeries":
        overrides.setdefault("alpha", 0.6)
        return cls(
            dataset=dataset,
            plot_type=PlotType.HISTOGRAM,
            x_column=column,
            y_column=None,
            label=label or f"{dataset.name} — {column}",
            bins=bins,
            **overrides,
        )
