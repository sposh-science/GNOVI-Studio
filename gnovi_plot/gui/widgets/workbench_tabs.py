from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QMenu, QTabBar, QToolButton, QWidget

from gnovi_plot.core.workbench import Workbench


class WorkbenchTabBar(QWidget):
    """Compact Workbench switcher docked above the Workbench header:

        [ CV Comparison ] [ New Scan ] [ Publication Figure ]  [+]

    A bare `QTabBar` (never `QTabWidget`) plus a "+" `QToolButton` -- a
    `QTabWidget` would imply separate page-content per tab, which is wrong
    here: switching a Workbench swaps which Figure the *same* canvas/
    drawers/toolbar point at (see `MainWindow._switch_workbench`), it never
    swaps which widgets are visible.

    Deliberately has no per-tab close button (`setTabsClosable` stays
    False) -- too easy to mis-click for a destructive action. Rename/
    Duplicate/Delete are reached via right-click context menu here (see
    `context_menu_requested`) as a secondary path; the primary path is the
    top-level "Workbench" menu (`MainWindow._create_menu`), which fires the
    exact same signals against whichever Workbench is currently active.
    Delete is always routed through the owner for confirmation + "keep at
    least one" enforcement (`Project.remove_workbench`) -- this widget
    never deletes anything itself.

    Long names elide (`QTabBar.setElideMode` + the `max-width` QSS rule on
    `QTabBar#WorkbenchTabBar::tab`) with the full name as a native tooltip;
    many Workbenches scroll via the tab bar's built-in scroll buttons
    rather than shrinking every tab to fit.
    """

    workbench_selected = Signal(str)  # workbench id
    new_workbench_requested = Signal()
    rename_requested = Signal(str)  # workbench id
    duplicate_requested = Signal(str)  # workbench id
    delete_requested = Signal(str)  # workbench id

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkbenchTabStrip")
        self._updating = False

        self.tab_bar = QTabBar()
        self.tab_bar.setObjectName("WorkbenchTabBar")
        self.tab_bar.setExpanding(False)
        self.tab_bar.setUsesScrollButtons(True)
        self.tab_bar.setElideMode(Qt.TextElideMode.ElideRight)
        self.tab_bar.setMovable(False)
        self.tab_bar.setTabsClosable(False)
        self.tab_bar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tab_bar.currentChanged.connect(self._on_current_changed)
        self.tab_bar.customContextMenuRequested.connect(self._on_context_menu_requested)

        self.new_button = QToolButton()
        self.new_button.setObjectName("WorkbenchNewButton")
        self.new_button.setText("+")
        self.new_button.setToolTip("New Workbench")
        self.new_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.new_button.clicked.connect(self.new_workbench_requested)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(2)
        layout.addWidget(self.tab_bar)
        layout.addWidget(self.new_button)
        layout.addStretch(1)

    def set_workbenches(self, workbenches: list[Workbench], active_id: str) -> None:
        """Rebuild every tab from `workbenches`, selecting `active_id`.
        Does not emit `workbench_selected` -- this reflects state the
        caller already knows about (it just performed the switch/create/
        rename/duplicate/delete), it doesn't request a new one."""
        self._updating = True
        self.tab_bar.blockSignals(True)
        while self.tab_bar.count():
            self.tab_bar.removeTab(0)
        active_index = 0
        for workbench in workbenches:
            index = self.tab_bar.addTab(workbench.name)
            self.tab_bar.setTabData(index, workbench.id)
            self.tab_bar.setTabToolTip(index, workbench.name)
            if workbench.id == active_id:
                active_index = index
        if self.tab_bar.count():
            self.tab_bar.setCurrentIndex(active_index)
        self.tab_bar.blockSignals(False)
        self._updating = False

    def current_workbench_id(self) -> str | None:
        index = self.tab_bar.currentIndex()
        if index < 0:
            return None
        return self.tab_bar.tabData(index)

    def _on_current_changed(self, index: int) -> None:
        if self._updating or index < 0:
            return
        workbench_id = self.tab_bar.tabData(index)
        if workbench_id:
            self.workbench_selected.emit(workbench_id)

    def _on_context_menu_requested(self, pos) -> None:
        index = self.tab_bar.tabAt(pos)
        if index < 0:
            return
        workbench_id = self.tab_bar.tabData(index)

        menu = QMenu(self)
        rename_action = menu.addAction("Rename Workbench")
        duplicate_action = menu.addAction("Duplicate Workbench")
        delete_action = menu.addAction("Delete Workbench")
        delete_action.setEnabled(self.tab_bar.count() > 1)

        chosen = menu.exec(self.tab_bar.mapToGlobal(pos))
        if chosen is rename_action:
            self.rename_requested.emit(workbench_id)
        elif chosen is duplicate_action:
            self.duplicate_requested.emit(workbench_id)
        elif chosen is delete_action:
            self.delete_requested.emit(workbench_id)
