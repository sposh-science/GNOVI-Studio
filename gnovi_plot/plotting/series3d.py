from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from gnovi_plot.data.dataset import Dataset


class Plot3DType(str, Enum):
    """3D plot kinds this milestone implements -- mirrors `plotting.series.
    PlotType`'s own `str, Enum` convention. No surface/mesh/wireframe kinds
    (out of scope, see `Panel3D`'s own docstring)."""

    SCATTER = "scatter"
    LINE = "line"
    LINE_MARKER = "line_marker"


@dataclass
class Series3D:
    """A plotted 3D item (scatter, line, or line+markers), independent of
    any live Matplotlib artist -- the 3D sibling of `plotting.series.
    PlotSeries`, deliberately NOT a variant of it (see `Panel3D`'s own
    docstring for why 3D got a sibling Panel type rather than new fields on
    the existing one; the same reasoning applies here: a 3D point needs a
    `z_column` no 2D series has any use for, and this class needs none of
    `PlotSeries`'s histogram/offset/z-order/stacking fields).

    `row_indices` optionally restricts this series to an explicit, ordered
    subset of `dataset.dataframe`'s row POSITIONS (`.iloc` order) -- how a
    "Group by" curve family (e.g. one `Series3D` per distinct temperature
    in a diode I-V dataset) is represented without duplicating the source
    Dataset or its DataFrame: every series in the family shares the same
    live `Dataset`, only the row selection differs. Deliberately NOT
    `PlotSeries.row_range`'s contiguous (start, end) form -- a group's rows
    are not guaranteed contiguous in source order (the dataset may
    interleave groups row-by-row rather than block-by-block), so an
    explicit position list is the only representation that can't silently
    misrepresent a group's membership. Row order within `row_indices` is
    always the dataset's own original source order (see
    `data.numeric.group_row_positions`) -- grouping never sorts by X or any
    other column, since a sweep can be genuinely non-monotonic in X by
    design (forward/reverse/cyclic scans). `None` means "the whole
    dataset" -- the same convention as `PlotSeries.row_range=None`, and
    exactly what every ungrouped (`Group by = None`) `Series3D` uses, so a
    plain single-series 3D scatter/line stores no row-selection metadata at
    all, identical to this milestone's predecessor.

    Numeric validity (is this (x, y, z) triple actually plottable) is
    intentionally NOT baked into `row_indices` at creation time and is
    instead re-checked at render time via `numeric_xyz` on `self.dataframe`
    (see `plotting.backends.matplotlib_backend._draw_series_3d`) -- the
    same "recompute at render/export time" convention every other
    GNOVI series already follows, so a later Working Data edit that makes a
    previously-valid row's value non-numeric is caught automatically rather
    than needing `row_indices` regenerated. GROUP MEMBERSHIP itself is the
    one thing captured once and never silently recomputed (see
    `data.numeric.group_row_positions`'s own docstring) -- recomputing it
    from a live, possibly-since-edited group column could otherwise
    silently produce a different partition than the one the user actually
    saw and styled.
    """

    dataset: Dataset
    x_column: str
    y_column: str
    z_column: str
    label: str = ""
    visible: bool = True
    color: str | None = None
    # True once the user has explicitly picked `color` -- same convention
    # as `PlotSeries.color_is_manual` (distinguishes an explicit choice
    # from an auto-assigned theme-cycle color; see `Panel3D.add_series`).
    color_is_manual: bool = False
    plot_type: Plot3DType = Plot3DType.SCATTER
    marker: str = "o"
    # Matplotlib `Axes.scatter`'s own `s` (marker area, points^2) default is
    # `rcParams['lines.markersize'] ** 2` -- `36.0` reproduces that exactly
    # for Matplotlib's own default `lines.markersize` of 6.0, and mirrors
    # `PlotSeries.marker_size`'s convention of being a LINEAR size the
    # renderer squares (`s=series.marker_size**2`, see
    # `plotting.backends.matplotlib_backend._draw_series`), not the area
    # itself -- so the same number means the same visual marker size in
    # both 2D scatter and 3D scatter.
    marker_size: float = 6.0
    # LINE/LINE_MARKER only -- ignored by the SCATTER render path, same
    # convention as `PlotSeries.line_style`/`.line_width`.
    line_style: str = "-"
    line_width: float = 1.5
    alpha: float = 1.0
    # Explicit row-position subset -- see this class's own docstring.
    row_indices: tuple[int, ...] | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Set by GnoviFigure.invalidate_series_for_dataset() after a dataset
    # transformation invalidates this series (missing column, or -- when
    # the row set itself changed -- a `row_indices` subset that can no
    # longer be trusted to mean what it did) -- same convention as
    # PlotSeries.stale.
    stale: bool = False

    def __post_init__(self) -> None:
        if not self.x_column:
            raise ValueError("Series3D.x_column must not be empty")
        if not self.y_column:
            raise ValueError("Series3D.y_column must not be empty")
        if not self.z_column:
            raise ValueError("Series3D.z_column must not be empty")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("Series3D.alpha must be between 0.0 and 1.0")
        if self.row_indices is not None:
            if not self.row_indices:
                raise ValueError("Series3D.row_indices must not be empty when given (use None for the whole dataset)")
            if any(not (0 <= i < self.dataset.row_count) for i in self.row_indices):
                raise ValueError(
                    f"Series3D.row_indices is out of bounds for dataset with {self.dataset.row_count} rows"
                )

    @property
    def dataframe(self) -> pd.DataFrame:
        """The dataset rows this series covers -- the full
        `dataset.dataframe`, or (for one member of a "Group by" curve
        family) just its `row_indices` subset, in the exact stored order.
        `DataFrame.iloc` indexing never mutates `dataset.dataframe`."""
        if self.row_indices is None:
            return self.dataset.dataframe
        return self.dataset.dataframe.iloc[list(self.row_indices)]

    def to_dict(self) -> dict:
        """Project-save representation: stores `dataset_id` rather than a
        nested `Dataset` -- same convention as `PlotSeries.to_dict`, the
        caller resolves it back to a shared live `Dataset` on load via
        `from_dict`'s `dataset_lookup` (see `core.project_io`). Every field
        added since this class's original (scatter-only) version is a
        plain optional key with a safe default in `from_dict` below -- no
        `PROJECT_FORMAT_VERSION` bump was needed for this (see this
        milestone's own architecture notes): an older app's `Series3D.
        from_dict` simply ignores keys it doesn't ask for and reconstructs
        a plain, unstyled scatter series exactly as it always did, rather
        than misparsing or refusing the file."""
        return {
            "dataset_id": self.dataset.id,
            "x_column": self.x_column,
            "y_column": self.y_column,
            "z_column": self.z_column,
            "label": self.label,
            "visible": self.visible,
            "color": self.color,
            "color_is_manual": self.color_is_manual,
            "plot_type": self.plot_type.value,
            "marker": self.marker,
            "marker_size": self.marker_size,
            "line_style": self.line_style,
            "line_width": self.line_width,
            "alpha": self.alpha,
            "row_indices": list(self.row_indices) if self.row_indices is not None else None,
            "id": self.id,
            "stale": self.stale,
        }

    @classmethod
    def from_dict(cls, data: dict, dataset_lookup: dict[str, Dataset]) -> "Series3D | None":
        """Reconstruct from `to_dict`'s output, resolving `dataset_id`
        against `dataset_lookup`. Returns None -- rather than raising -- if
        `dataset_id` isn't in `dataset_lookup`, so one stale reference in an
        otherwise-valid project doesn't block loading the rest -- same
        tolerance `PlotSeries.from_dict` already has."""
        dataset = dataset_lookup.get(data["dataset_id"])
        if dataset is None:
            return None
        row_indices = data.get("row_indices")
        return cls(
            dataset=dataset,
            x_column=data["x_column"],
            y_column=data["y_column"],
            z_column=data["z_column"],
            label=data.get("label", ""),
            visible=data.get("visible", True),
            color=data.get("color"),
            color_is_manual=data.get("color_is_manual", False),
            plot_type=Plot3DType(data.get("plot_type", Plot3DType.SCATTER.value)),
            marker=data.get("marker", "o"),
            marker_size=data.get("marker_size", 6.0),
            line_style=data.get("line_style", "-"),
            line_width=data.get("line_width", 1.5),
            alpha=data.get("alpha", 1.0),
            row_indices=tuple(row_indices) if row_indices is not None else None,
            id=data["id"],
            stale=data.get("stale", False),
        )
