from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.figure import Panel


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
    panel: Panel
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
            panel=Panel.from_dict(data["panel"], dataset_lookup),
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


def clone_panel_with_shared_datasets(panel: Panel, dataset_manager) -> Panel:
    """Deep-copy `panel` (series/styling/everything) while keeping every
    `PlotSeries.dataset` pointed at the same live `Dataset` instance from
    `dataset_manager` -- see `dataset_identity_memo`."""
    return copy.deepcopy(panel, dataset_identity_memo(dataset_manager))
