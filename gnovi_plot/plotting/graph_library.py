from __future__ import annotations

from datetime import datetime, timezone

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.graph import Graph, clone_panel_with_shared_datasets


class GraphLibrary:
    """Project-local collection of saved `Graph`s. Has no Qt/GUI
    dependencies -- mirrors `data.dataset_manager.DatasetManager`."""

    def __init__(self) -> None:
        self._graphs: dict[str, Graph] = {}

    def add(self, graph: Graph) -> None:
        self._graphs[graph.id] = graph

    def remove(self, graph_id: str) -> None:
        self._graphs.pop(graph_id, None)

    def get(self, graph_id: str) -> Graph | None:
        return self._graphs.get(graph_id)

    def rename(self, graph_id: str, new_name: str) -> None:
        graph = self._graphs.get(graph_id)
        if graph is None:
            return
        graph.name = new_name
        graph.modified_at = datetime.now(timezone.utc)

    def duplicate(self, graph_id: str, dataset_manager) -> Graph | None:
        """Add and return an independent copy of the graph `graph_id`
        (new `id`, name suffixed " (Copy)"), sharing the same live
        `Dataset` objects as the original -- see `clone_panel_with_shared_datasets`."""
        graph = self._graphs.get(graph_id)
        if graph is None:
            return None
        copy_graph = Graph(
            name=f"{graph.name} (Copy)",
            panel=clone_panel_with_shared_datasets(graph.panel, dataset_manager),
        )
        self.add(copy_graph)
        return copy_graph

    @property
    def graphs(self) -> list[Graph]:
        return list(self._graphs.values())

    def __len__(self) -> int:
        return len(self._graphs)

    def __iter__(self):
        return iter(self._graphs.values())

    def to_dict(self) -> list[dict]:
        return [graph.to_dict() for graph in self._graphs.values()]

    @classmethod
    def from_dict(cls, data: list[dict], dataset_lookup: dict[str, Dataset]) -> "GraphLibrary":
        library = cls()
        for graph_data in data:
            library.add(Graph.from_dict(graph_data, dataset_lookup))
        return library

    def save_panel_as_graph(self, figure: GnoviFigure, name: str, dataset_manager) -> Graph:
        """Copy `figure.active_panel`'s configuration (series + every
        display setting) into the library as a new `Graph`, per "Save
        Current Panel as Graph". The stored `Panel` shares the project's
        live `Dataset` objects but is otherwise fully independent of the
        live figure -- later edits to the active panel never affect it.

        Also stamps the live active panel's `source_graph_id` with the new
        `Graph.id`, so it now displays as that Graph's working copy (see
        `Panel.source_graph_id`) rather than "Unsaved graph"."""
        graph = Graph(
            name=name,
            panel=clone_panel_with_shared_datasets(figure.active_panel, dataset_manager),
        )
        self.add(graph)
        figure.active_panel.source_graph_id = graph.id
        return graph

    def load_graph_into_panel(self, graph_id: str, figure: GnoviFigure, dataset_manager) -> bool:
        """Replace `figure`'s active panel with an INDEPENDENT EDITABLE COPY
        of the stored graph's panel, per "Load Graph into Active Panel".
        Editing the now-active panel afterward never mutates the stored
        `Graph` -- it's a fresh deep copy, not a shared reference. The new
        active panel's `source_graph_id` is (re-)stamped with `graph_id`
        regardless of whatever the stored snapshot happened to carry, so it
        always correctly identifies the Graph it was just loaded from (see
        `Panel.source_graph_id`). Returns False (no-op) if `graph_id` isn't
        in the library."""
        graph = self._graphs.get(graph_id)
        if graph is None:
            return False
        loaded_panel = clone_panel_with_shared_datasets(graph.panel, dataset_manager)
        loaded_panel.source_graph_id = graph.id
        figure.panels[figure.active_panel_index] = loaded_panel
        figure._renumber_panel_labels()
        return True

    def update_graph_from_panel(self, graph_id: str, figure: GnoviFigure, dataset_manager) -> bool:
        """Replace the stored Graph's panel snapshot with the current
        active panel's state, per "Update Saved Graph" -- the only
        operation that ever lets a working copy's edits reach the Graph
        Library; `save_panel_as_graph`/`load_graph_into_panel` otherwise
        always keep the two independent (see their docstrings). Keeps the
        Graph's `id`/`name`; only `panel` and `modified_at` change. Returns
        False (no-op) if `graph_id` isn't in the library."""
        graph = self._graphs.get(graph_id)
        if graph is None:
            return False
        graph.panel = clone_panel_with_shared_datasets(figure.active_panel, dataset_manager)
        graph.modified_at = datetime.now(timezone.utc)
        return True
