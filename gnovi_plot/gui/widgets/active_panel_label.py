from __future__ import annotations

from collections.abc import Callable

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel

from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.graph_library import GraphLibrary


class ActivePanelLabel(QLabel):
    """Compact, read-only multi-line context block shown near the top of
    every left-drawer page whose controls act on a `GnoviFigure`'s active
    panel (Plot/Series/Figure/Layout/Axes):

        Active panel: Panel 2
        Graph: Ferricyanide 50 mV/s (working copy)
        Data: Ferricyanide SR-0.05

    "Active panel" is purely a display of `GnoviFigure.active_panel_index`
    (the existing single source of truth for "which panel is active", see
    `GnoviFigure.set_active_panel`), never a second one.

    "Graph" distinguishes a WORKING panel from the persistent Graph Library
    snapshot it may have originated from (see `Panel.source_graph_id`):
    "Unsaved graph" if the active panel was never saved as/loaded from a
    Graph, or if its origin Graph has since been deleted from the library;
    otherwise "<name> (working copy)" -- editing the panel further never
    implies the stored Graph changes (see `plotting.graph_library.
    GraphLibrary`'s save/load/update methods).

    "Data" is the active panel's plotted dataset provenance, derived
    directly from its `PlotSeries.dataset` references (never a separate
    stored list -- see `Panel`'s docstring): "No data", the one dataset's
    name, or "N datasets" with a tooltip listing every name.

    `get_graph_library` is a callable (re-invoked on every `refresh()`,
    like `GraphLibraryPanel`'s own `get_figure`/`get_dataset_manager`) so
    this label stays correct after Open/New Project repoints the project's
    `GraphLibrary` -- pass `None` (the default) to omit the Graph/Data
    lines entirely and show only "Active panel: Panel N", if ever needed.
    Callers refresh via `refresh(figure)` alongside their own existing
    refresh path (panel switch, layout change, series edit, graph saved/
    loaded/updated/renamed/deleted, project load/new, undo/redo) -- it
    holds no state of its own beyond the `get_graph_library` callable.
    """

    def __init__(
        self,
        figure: GnoviFigure,
        get_graph_library: Callable[[], GraphLibrary] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._get_graph_library = get_graph_library
        font = QFont(self.font())
        font.setBold(True)
        self.setFont(font)
        self.refresh(figure)

    def refresh(self, figure: GnoviFigure) -> None:
        lines = [f"Active panel: Panel {figure.active_panel_index + 1}"]
        tooltip = ""

        if self._get_graph_library is not None:
            panel = figure.active_panel
            library = self._get_graph_library()
            graph = library.get(panel.source_graph_id) if panel.source_graph_id else None
            if graph is not None:
                lines.append(f"Graph: {graph.name} (working copy)")
            else:
                lines.append("Graph: Unsaved graph")

            dataset_names: list[str] = []
            seen_dataset_ids: set[str] = set()
            for series in panel.series:
                if series.dataset.id in seen_dataset_ids:
                    continue
                seen_dataset_ids.add(series.dataset.id)
                dataset_names.append(series.dataset.name)

            if not dataset_names:
                lines.append("Data: No data")
            elif len(dataset_names) == 1:
                lines.append(f"Data: {dataset_names[0]}")
            else:
                lines.append(f"Data: {len(dataset_names)} datasets")
                tooltip = "\n".join(dataset_names)

        self.setText("\n".join(lines))
        self.setToolTip(tooltip)
