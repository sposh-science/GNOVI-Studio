from __future__ import annotations

from pathlib import Path

from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.graph_library import GraphLibrary

DEFAULT_PROJECT_NAME = "Untitled Project"


class Project:
    """A Gnovi Studio project (session): datasets + Graph Library + Figures
    + per-dataset transformation history, as a single reproducible unit
    saved/loaded via `core.project_io`.

    Plain in-memory container -- no Qt/GUI dependencies, no `to_dict`/
    `from_dict` of its own (see `core.project_io` for why: assembling
    `project.json` has to interleave with zip-embedded dataset CSVs, which
    doesn't fit a plain dict round-trip). `path` is `None` for a project
    that has never been saved ("Untitled").
    """

    def __init__(
        self,
        *,
        name: str = DEFAULT_PROJECT_NAME,
        dataset_manager: DatasetManager | None = None,
        graph_library: GraphLibrary | None = None,
        figures: list[GnoviFigure] | None = None,
        active_figure_index: int = 0,
        path: Path | None = None,
    ) -> None:
        self.name = name
        self.dataset_manager = dataset_manager if dataset_manager is not None else DatasetManager()
        self.graph_library = graph_library if graph_library is not None else GraphLibrary()
        self.figures = figures if figures is not None else [GnoviFigure()]
        self.active_figure_index = active_figure_index
        self.path = path

    @classmethod
    def new(cls) -> "Project":
        """A fresh, empty project -- used by "New Project"."""
        return cls()

    @property
    def active_figure(self) -> GnoviFigure:
        return self.figures[self.active_figure_index]
