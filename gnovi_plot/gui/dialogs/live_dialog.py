from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QFrame, QScrollArea, QVBoxLayout, QWidget

_MAX_SCREEN_WIDTH_FRACTION = 0.7
_MAX_SCREEN_HEIGHT_FRACTION = 0.85


class LiveDialog(QDialog):
    """Non-modal, scrollable dialog that hosts a persistent content widget.

    The dialog wraps -- rather than rebuilds -- `content`, so it is created
    once and every subsequent menu/toolbar action just raises the same
    instance. That trivially satisfies "preserve settings when reopened" and
    "update preview live" (the content widget stays bound to the live model
    the whole time), and non-modal so it can stay open while the user keeps
    interacting with the plot.
    """

    def __init__(self, title: str, content: QWidget, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

        self._fit_within_screen(content)

    def _fit_within_screen(self, content: QWidget) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        max_width = int(available.width() * _MAX_SCREEN_WIDTH_FRACTION)
        max_height = int(available.height() * _MAX_SCREEN_HEIGHT_FRACTION)

        hint = content.sizeHint()
        width = min(max(hint.width() + 48, 380), max_width)
        height = min(max(hint.height() + 32, 320), max_height)
        self.resize(width, height)

    def show_raised(self) -> None:
        """Bring the dialog to front, reusing whatever state it already has."""
        self.show()
        self.raise_()
        self.activateWindow()
