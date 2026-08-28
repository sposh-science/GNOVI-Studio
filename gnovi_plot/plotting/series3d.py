from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pandas as pd

from gnovi_plot.data.dataset import Dataset


@dataclass
class Series3D:
    """A plotted 3D scatter item, independent of any live Matplotlib artist
    -- the 3D sibling of `plotting.series.PlotSeries`, deliberately NOT a
    variant of it (see `Panel3D`'s own docstring for why 3D got a sibling
    Panel type rather than new fields on the existing one; the same
    reasoning applies here: a 3D point needs a `z_column` no 2D series has
    any use for, and this milestone's only 3D plot kind -- scatter -- needs
    none of `PlotSeries`'s 2D-only fields, line-plot fields, or histogram
    fields).

    Deliberately minimal for this milestone: only what a genuine XYZ
    scatter needs (`dataset` + three column names, `marker`/`marker_size`/
    `color`/`alpha`, `visible`). No `row_range` (cycle-segment selection --
    not a 3D scatter concept here), no surface/mesh/volume fields (out of
    scope for this milestone, see `plotting.figure.Panel3D`'s own
    docstring), no plot-type enum (there is exactly one 3D plot kind so
    far; a mega-enum anticipating future kinds that don't exist yet would
    be speculative).
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
    alpha: float = 1.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Set by GnoviFigure.invalidate_series_for_dataset() after a dataset
    # transformation invalidates this series (missing column) -- same
    # convention as PlotSeries.stale.
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

    @property
    def dataframe(self) -> pd.DataFrame:
        """`dataset.dataframe`, unfiltered -- unlike `PlotSeries.dataframe`,
        there is no `row_range` concept for a 3D scatter series (see this
        class's own docstring), so this is always the whole working data."""
        return self.dataset.dataframe

    def to_dict(self) -> dict:
        """Project-save representation: stores `dataset_id` rather than a
        nested `Dataset` -- same convention as `PlotSeries.to_dict`, the
        caller resolves it back to a shared live `Dataset` on load via
        `from_dict`'s `dataset_lookup` (see `core.project_io`)."""
        return {
            "dataset_id": self.dataset.id,
            "x_column": self.x_column,
            "y_column": self.y_column,
            "z_column": self.z_column,
            "label": self.label,
            "visible": self.visible,
            "color": self.color,
            "color_is_manual": self.color_is_manual,
            "marker": self.marker,
            "marker_size": self.marker_size,
            "alpha": self.alpha,
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
        return cls(
            dataset=dataset,
            x_column=data["x_column"],
            y_column=data["y_column"],
            z_column=data["z_column"],
            label=data.get("label", ""),
            visible=data.get("visible", True),
            color=data.get("color"),
            color_is_manual=data.get("color_is_manual", False),
            marker=data.get("marker", "o"),
            marker_size=data.get("marker_size", 6.0),
            alpha=data.get("alpha", 1.0),
            id=data["id"],
            stale=data.get("stale", False),
        )
