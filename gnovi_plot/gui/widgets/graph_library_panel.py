from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.graph_library import GraphLibrary

_NO_SELECTION_MESSAGE = "Select a graph first."


class GraphLibraryPanel(QWidget):
    """Graphs tab: the project-local Graph Library list plus Save Current
    Panel as Graph / Load Selected Graph into Active Panel / Rename /
    Duplicate / Delete. No thumbnails (v0.9).

    `get_figure`/`get_dataset_manager` are callables -- re-invoked on every
    action -- rather than fixed references, so this panel always acts on
    the owner's *current* active figure/dataset manager even after
    Open/New Project swaps them; only the `GraphLibrary` itself needs
    `set_library`, since Open/New Project points it at a fresh instance.

    Two signals distinguish what changed, since they need different
    handling by the owner: `graph_library_changed` (Save/Rename/Duplicate/
    Delete -- only the library's contents changed, no re-render needed) vs.
    `graph_loaded_into_panel` (Load -- the active panel's series/styling
    changed, needs the same handling as any other figure-content edit:
    re-render, refresh the Series/Properties panels, an undo checkpoint).
    """

    graph_library_changed = Signal()
    graph_loaded_into_panel = Signal()

    def __init__(
        self,
        graph_library: GraphLibrary,
        get_figure: Callable[[], GnoviFigure],
        get_dataset_manager: Callable[[], DatasetManager],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._library = graph_library
        self._get_figure = get_figure
        self._get_dataset_manager = get_dataset_manager

        self.graph_list = QListWidget()

        self.save_button = QPushButton("Save Current Panel as Graph")
        self.save_button.setProperty("primary", True)
        self.load_button = QPushButton("Load Selected Graph into Active Panel")
        self.rename_button = QPushButton("Rename Graph")
        self.duplicate_button = QPushButton("Duplicate Graph")
        self.delete_button = QPushButton("Delete Graph")

        layout = QVBoxLayout(self)
        layout.addWidget(self.graph_list)
        layout.addWidget(self.save_button)
        layout.addWidget(self.load_button)
        button_row = QHBoxLayout()
        button_row.addWidget(self.rename_button)
        button_row.addWidget(self.duplicate_button)
        button_row.addWidget(self.delete_button)
        layout.addLayout(button_row)

        self.save_button.clicked.connect(self._on_save_clicked)
        self.load_button.clicked.connect(self._on_load_clicked)
        self.rename_button.clicked.connect(self._on_rename_clicked)
        self.duplicate_button.clicked.connect(self._on_duplicate_clicked)
        self.delete_button.clicked.connect(self._on_delete_clicked)

        self._refresh_list()

    def set_library(self, graph_library: GraphLibrary) -> None:
        """Repoint this panel at a different project's `GraphLibrary` (e.g.
        after Open/New Project) and reload its list."""
        self._library = graph_library
        self._refresh_list()

    def _refresh_list(self, select_id: str | None = None) -> None:
        self.graph_list.blockSignals(True)
        self.graph_list.clear()
        target_item = None
        for graph in self._library.graphs:
            item = QListWidgetItem(graph.name)
            item.setData(Qt.UserRole, graph.id)
            self.graph_list.addItem(item)
            if select_id is not None and graph.id == select_id:
                target_item = item
        self.graph_list.blockSignals(False)
        if target_item is not None:
            self.graph_list.setCurrentItem(target_item)
        else:
            self.graph_list.setCurrentRow(-1)

    def _current_graph_id(self) -> str | None:
        item = self.graph_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def _on_save_clicked(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Current Panel as Graph", "Graph name:")
        name = name.strip()
        if not ok or not name:
            return
        graph = self._library.save_panel_as_graph(self._get_figure(), name, self._get_dataset_manager())
        self._refresh_list(select_id=graph.id)
        self.graph_library_changed.emit()

    def _on_load_clicked(self) -> None:
        graph_id = self._current_graph_id()
        if graph_id is None:
            QMessageBox.information(self, "Load Graph", _NO_SELECTION_MESSAGE)
            return
        loaded = self._library.load_graph_into_panel(
            graph_id, self._get_figure(), self._get_dataset_manager()
        )
        if loaded:
            self.graph_loaded_into_panel.emit()

    def _on_rename_clicked(self) -> None:
        graph_id = self._current_graph_id()
        if graph_id is None:
            QMessageBox.information(self, "Rename Graph", _NO_SELECTION_MESSAGE)
            return
        graph = self._library.get(graph_id)
        new_name, ok = QInputDialog.getText(self, "Rename Graph", "Graph name:", text=graph.name)
        new_name = new_name.strip()
        if not ok or not new_name:
            return
        self._library.rename(graph_id, new_name)
        self._refresh_list(select_id=graph_id)
        self.graph_library_changed.emit()

    def _on_duplicate_clicked(self) -> None:
        graph_id = self._current_graph_id()
        if graph_id is None:
            QMessageBox.information(self, "Duplicate Graph", _NO_SELECTION_MESSAGE)
            return
        copy_graph = self._library.duplicate(graph_id, self._get_dataset_manager())
        if copy_graph is not None:
            self._refresh_list(select_id=copy_graph.id)
            self.graph_library_changed.emit()

    def _on_delete_clicked(self) -> None:
        graph_id = self._current_graph_id()
        if graph_id is None:
            QMessageBox.information(self, "Delete Graph", _NO_SELECTION_MESSAGE)
            return
        self._library.remove(graph_id)
        self._refresh_list()
        self.graph_library_changed.emit()
