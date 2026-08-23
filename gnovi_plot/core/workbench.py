from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from gnovi_plot.analysis.panel_results import PanelResultHistory
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.figure import GnoviFigure

DEFAULT_WORKBENCH_NAME = "Workbench 1"


@dataclass
class Workbench:
    """One independent plotting workspace/page within a `Project` --
    GNOVI Studio = the application, Workbench = this, Graph Library =
    reusable saved graphs, Project = datasets + Graph Library + multiple
    Workbenches (see `core.project.Project`).

    A thin, stable-identity wrapper around a `GnoviFigure`, deliberately
    mirroring how `plotting.graph.Graph` wraps a `Panel`: `Workbench` owns
    naming/identity (`id`/`name`), `GnoviFigure` stays exactly what it
    already is -- a declarative figure/rendering description, complete
    with its own `active_panel_index` (Workbench never duplicates that;
    `GnoviFigure` remains its sole owner). Datasets and the Graph Library
    are never held here -- they stay Project-level shared resources (see
    `Project.dataset_manager`/`Project.graph_library`); duplicating a
    Workbench must never duplicate a `Dataset` (see
    `plotting.graph.clone_figure_with_shared_datasets`, the one place a
    `Workbench`'s `figure` is ever cloned).

    `analysis_results` (`PanelResultHistory`, keyed by stable `Panel.id`)
    is this Workbench's own analysis-result history -- owned here rather
    than by a separate manager because it's intrinsically part of what a
    Workbench's working state is, exactly like `figure`. Two Workbenches
    (even with structurally identical, freshly-cloned panels) therefore
    always have independent histories, with no separate bookkeeping
    needed to keep that true: duplicating a Workbench constructs a new
    one with a fresh, empty `PanelResultHistory()` (see
    `Project.duplicate_workbench`), never copying the original's.

    Persisted (`to_dict()`/`from_dict()` below include it) -- a saved
    project preserves the scientific analysis record, not only the
    fit-derived `PlotSeries` it produced. A project saved before this
    existed simply has no `"analysis_results"` key, so `from_dict()`
    falls back to an empty `PanelResultHistory()`, same as New Project;
    no `PROJECT_FORMAT_VERSION` bump needed (purely additive, `.get()`-
    defaulted, same reasoning as `Panel.id`'s own backward compatibility).
    """

    name: str
    figure: GnoviFigure
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    analysis_results: PanelResultHistory = field(default_factory=PanelResultHistory)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "figure": self.figure.to_dict(),
            "analysis_results": self.analysis_results.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict, dataset_lookup: dict[str, Dataset]) -> "Workbench":
        return cls(
            id=data["id"],
            name=data.get("name", DEFAULT_WORKBENCH_NAME),
            figure=GnoviFigure.from_dict(data.get("figure", {}), dataset_lookup),
            analysis_results=PanelResultHistory.from_dict(data.get("analysis_results", {})),
        )
