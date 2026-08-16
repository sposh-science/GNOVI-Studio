from __future__ import annotations

from pathlib import Path

from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.graph import clone_figure_with_shared_datasets
from gnovi_plot.plotting.graph_library import GraphLibrary
from gnovi_plot.core.workbench import DEFAULT_WORKBENCH_NAME, Workbench

DEFAULT_PROJECT_NAME = "Untitled Project"


class Project:
    """A Gnovi Studio project (session): datasets + Graph Library + multiple
    Workbenches + per-dataset transformation history, as a single
    reproducible unit saved/loaded via `core.project_io`.

    A `Project` always holds at least one `Workbench` -- there is
    deliberately no "empty project" state with zero Workbenches, since
    `MainWindow.figure_model` (and everything downstream of it) always
    needs *some* current Figure to point at; `remove_workbench` refuses to
    drop the last one (see its docstring).

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
        workbenches: list[Workbench] | None = None,
        active_workbench_id: str | None = None,
        path: Path | None = None,
    ) -> None:
        self.name = name
        self.dataset_manager = dataset_manager if dataset_manager is not None else DatasetManager()
        self.graph_library = graph_library if graph_library is not None else GraphLibrary()
        self.workbenches = (
            workbenches if workbenches is not None else [Workbench(name=DEFAULT_WORKBENCH_NAME, figure=GnoviFigure())]
        )
        self.active_workbench_id = active_workbench_id or self.workbenches[0].id
        self.path = path

    @classmethod
    def new(cls) -> "Project":
        """A fresh, empty project (one blank Workbench) -- used by "New
        Project"."""
        return cls()

    @property
    def active_workbench(self) -> Workbench:
        """The current Workbench -- falls back to the first Workbench if
        `active_workbench_id` doesn't resolve (e.g. a corrupt/edited save),
        rather than raising, since there's always at least one."""
        return self.get_workbench(self.active_workbench_id) or self.workbenches[0]

    def get_workbench(self, workbench_id: str) -> Workbench | None:
        for workbench in self.workbenches:
            if workbench.id == workbench_id:
                return workbench
        return None

    def add_workbench(self, workbench: Workbench) -> None:
        self.workbenches.append(workbench)

    def rename_workbench(self, workbench_id: str, new_name: str) -> None:
        workbench = self.get_workbench(workbench_id)
        if workbench is not None:
            workbench.name = new_name

    def duplicate_workbench(self, workbench_id: str) -> Workbench | None:
        """Add and return an independent copy of Workbench `workbench_id`
        (new `id`, name suffixed " (Copy)"), sharing the same live
        `Dataset` objects as the original -- see
        `plotting.graph.clone_figure_with_shared_datasets`. Returns None
        (no-op) if `workbench_id` isn't in this project."""
        workbench = self.get_workbench(workbench_id)
        if workbench is None:
            return None
        copy_workbench = Workbench(
            name=f"{workbench.name} (Copy)",
            figure=clone_figure_with_shared_datasets(workbench.figure, self.dataset_manager),
        )
        self.add_workbench(copy_workbench)
        return copy_workbench

    def remove_workbench(self, workbench_id: str) -> bool:
        """Remove Workbench `workbench_id`. Refuses (returns False, no-op)
        if it's unknown, or if it's the only Workbench left -- a `Project`
        always keeps at least one (see class docstring). If the removed
        Workbench was active, `active_workbench_id` moves to the Workbench
        that was immediately before it (or the new first one), so the user
        lands somewhere sensible rather than on an arbitrary Workbench."""
        if len(self.workbenches) <= 1:
            return False
        index = next((i for i, w in enumerate(self.workbenches) if w.id == workbench_id), None)
        if index is None:
            return False
        removing_active = self.workbenches[index].id == self.active_workbench_id
        del self.workbenches[index]
        if removing_active:
            new_index = max(index - 1, 0)
            self.active_workbench_id = self.workbenches[new_index].id
        return True
