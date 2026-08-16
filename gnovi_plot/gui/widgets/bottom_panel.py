from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QTabWidget, QVBoxLayout, QWidget

_RESULTS_PLACEHOLDER_TEXT = (
    "No results yet.\n\n"
    "This tab is a reusable container for future analysis output -- curve "
    "fits, peak tables, FFT results, and other reports -- once those "
    "analyses are implemented."
)


class _QtLogHandler(logging.Handler, QObject):
    """Bridges a Python `logging.Logger` to the Messages tab. Only wired up
    for the real running app (`app.main()` calls `install_logging`) --
    never automatically inside `BottomPanel.__init__`, since many tests
    construct/destroy `MainWindow` (and therefore `BottomPanel`) instances,
    and an auto-installed handler on the shared `gnovi_plot` logger would
    outlive each destroyed widget and log against a deleted Qt object.
    """

    message_logged = Signal(str)

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.message_logged.emit(self.format(record))


class BottomPanel(QTabWidget):
    """Collapsible/resizable tabbed panel below the plot canvas.

    Tabs: Data (a table supplied by the owner, e.g. the Data Preview),
    Graphs (the project-local Graph Library, e.g.
    `gui.widgets.graph_library_panel.GraphLibraryPanel`), Transformations (a
    list supplied by the owner, e.g. Working Data history), Results (an
    inert placeholder -- the reusable home for future fit/peak/FFT output,
    deliberately empty this milestone), and Messages (a live application
    log).

    This widget only owns the tab *container*; tab content for Data/Graphs/
    Transformations is handed in by the owner via `set_data_widget`/
    `set_graphs_widget`/`set_transformations_widget` so this panel has no
    dependency on DatasetPanel/GraphLibraryPanel/DataToolsPanel internals.
    `QTabWidget` never destroys/rebuilds a tab's content widget when
    switching tabs, so widget identity (and state) survives tab switching
    and show/hide.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._data_tab = QWidget()
        QVBoxLayout(self._data_tab).setContentsMargins(0, 0, 0, 0)
        self.addTab(self._data_tab, "Data")

        self._graphs_tab = QWidget()
        QVBoxLayout(self._graphs_tab).setContentsMargins(0, 0, 0, 0)
        self.addTab(self._graphs_tab, "Graphs")

        self._transformations_tab = QWidget()
        QVBoxLayout(self._transformations_tab).setContentsMargins(0, 0, 0, 0)
        self.addTab(self._transformations_tab, "Transformations")

        results_placeholder = QLabel(_RESULTS_PLACEHOLDER_TEXT)
        results_placeholder.setAlignment(Qt.AlignCenter)
        results_placeholder.setWordWrap(True)
        self.addTab(results_placeholder, "Results")

        self.messages_view = QPlainTextEdit()
        self.messages_view.setReadOnly(True)
        self.messages_view.setMaximumBlockCount(2000)
        self.addTab(self.messages_view, "Messages")

        self._log_handler = _QtLogHandler()
        self._log_handler.message_logged.connect(self.messages_view.appendPlainText)

    def set_data_widget(self, widget: QWidget) -> None:
        """Place `widget` (e.g. the Data Preview table) into the Data tab.
        Reparents `widget` -- Qt moves it, it is not copied."""
        self._data_tab.layout().addWidget(widget)

    def set_graphs_widget(self, widget: QWidget) -> None:
        """Place `widget` (the Graph Library UI) into the Graphs tab.
        Reparents `widget` -- Qt moves it, it is not copied."""
        self._graphs_tab.layout().addWidget(widget)

    def set_transformations_widget(self, widget: QWidget) -> None:
        """Place `widget` (e.g. the Working Data transformation history)
        into the Transformations tab."""
        self._transformations_tab.layout().addWidget(widget)

    def install_logging(self, logger_name: str = "gnovi_plot") -> None:
        """Start mirroring `logger_name` into the Messages tab. Call this
        once, from the real application entry point only -- see the class
        docstring for why it's not automatic."""
        logging.getLogger(logger_name).addHandler(self._log_handler)

    def append_message(self, text: str) -> None:
        """Append a line directly, bypassing the logging bridge."""
        self.messages_view.appendPlainText(text)
