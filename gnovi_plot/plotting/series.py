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
