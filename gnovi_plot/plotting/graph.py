from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.figure import GnoviFigure, Panel, Panel3D, panel_from_dict


@dataclass
class Graph:
    """A saved, independently reusable, project-bound editable plot
    configuration: a full `Panel` snapshot (series + every display setting),
    stored in a project's `GraphLibrary`.

    Deliberately NOT a re-bindable template -- a `Graph`'s series reference
    the project's actual live `Dataset` objects (by identity, preserved via
    `dataset_identity_memo` when cloning), never a column/role binding meant
    to be re-pointed at different data later. See `GraphLibrary` for the
    save/load-into-panel workflow that keeps a loaded copy fully independent
    of the stored `Graph`.
    """

    name: str
    panel: Panel | Panel3D
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "panel": self.panel.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict, dataset_lookup: dict[str, Dataset]) -> "Graph":
        return cls(
            id=data["id"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            modified_at=datetime.fromisoformat(data["modified_at"]),
            # `panel_from_dict` (not `Panel.from_dict`) so a saved Panel3D
            # Graph reloads as a Panel3D, never silently misparsed as a
            # `Panel` -- see `plotting.figure.panel_from_dict`'s own
            # docstring for the "kind" dispatch this relies on.
            panel=panel_from_dict(data["panel"], dataset_lookup),
        )


def dataset_identity_memo(dataset_manager) -> dict:
    """`copy.deepcopy` memo mapping `id(dataset) -> dataset` for every
    dataset currently in `dataset_manager`. Seeding a deepcopy's memo with
    this makes a `Panel`/`PlotSeries.dataset` reference short-circuit back
    to the same live `Dataset` instead of duplicating it -- the same trick
    `gui.undo_manager.snapshot_figure` uses, reused here by
    `GraphLibrary.save_panel_as_graph`/`load_graph_into_panel`.
    """
    return {id(dataset): dataset for dataset in dataset_manager.datasets}


def clone_panel_with_shared_datasets(panel: Panel | Panel3D, dataset_manager) -> Panel | Panel3D:
    """Deep-copy `panel` (series/styling/everything) while keeping every
    series' `.dataset` pointed at the same live `Dataset` instance from
    `dataset_manager` -- see `dataset_identity_memo`. Works unchanged for
    either `Panel` or `Panel3D`: nothing here is type-specific --
    `copy.deepcopy` recurses through whichever dataclass it's given, and
    reassigning `.id` afterward is plain duck typing (both types have one).

    Assigns the clone a *fresh* `.id` -- this produces a genuinely
    independent panel (Graph Library save/load-into-panel and
    `core.project.Project.extract_panel_to_workbench` are its callers
    today), never "the same panel at a different point in time" (that's
    `gui.undo_manager.snapshot_figure`'s plain `copy.deepcopy` instead,
    which deliberately preserves `id`). Without this, e.g. loading two
    different Graphs into two different panels that started from the same
    stored `Graph.panel` would leave them sharing an id -- letting one
    panel's analysis-result history appear to belong to the other."""
    cloned = copy.deepcopy(panel, dataset_identity_memo(dataset_manager))
    cloned.id = uuid.uuid4().hex
    return cloned


def clone_figure_with_shared_datasets(figure: GnoviFigure, dataset_manager) -> GnoviFigure:
    """Deep-copy `figure` (every panel, every series, every display/layout/
    typography setting) while keeping every `PlotSeries.dataset` pointed at
    the same live `Dataset` instance from `dataset_manager` -- the
    whole-figure counterpart of `clone_panel_with_shared_datasets`, same
    `dataset_identity_memo` trick. Used by `core.workbench.Workbench`
    duplication (`core.project.Project.duplicate_workbench`) to produce an
    independent working copy of an entire Workbench's Figure -- every
    `Panel.source_graph_id` (Graph Library provenance) is preserved
    automatically, since it's just another field in the copied tree.

    Every cloned panel gets a *fresh* `Panel.id`, same reasoning as
    `clone_panel_with_shared_datasets`: a duplicated Workbench must never
    share panel identity with the Workbench it was duplicated from."""
    cloned = copy.deepcopy(figure, dataset_identity_memo(dataset_manager))
    for cloned_panel in cloned.panels:
        cloned_panel.id = uuid.uuid4().hex
    return cloned
