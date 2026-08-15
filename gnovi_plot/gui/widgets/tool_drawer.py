from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QStackedWidget, QToolButton, QVBoxLayout, QWidget

STRIP_WIDTH = 64
_ICON_SIZE = 22
_ICON_COLOR = "#5b6270"


def _make_icon(kind: str) -> QIcon:
    """A small, simple monochrome glyph drawn in-process for each tool-strip
    button -- deliberately generic shapes (grid/waveform/stacked lines/dial),
    not a copy of any specific instrument's iconography, so a button is
    identifiable by shape as well as by its label (never icon-only)."""
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(_ICON_COLOR))
    pen.setWidthF(1.6)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)

    m = 3
    size = _ICON_SIZE
    if kind == "data":
        painter.drawRect(m, m, size - 2 * m, size - 2 * m)
        painter.drawLine(m, size // 2, size - m, size // 2)
        painter.drawLine(size // 2, m, size // 2, size - m)
    elif kind == "plot":
        painter.drawLine(m, size - m, m, m)
        painter.drawLine(m, size - m, size - m, size - m)
        points = [
            (m + 2, size - m - 3),
            (m + 7, size - m - 10),
            (m + 11, size - m - 6),
            (size - m - 1, m + 2),
        ]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            painter.drawLine(x1, y1, x2, y2)
    elif kind == "series":
        for y in (size // 2 - 5, size // 2, size // 2 + 5):
            painter.drawLine(m, y, size - m, y)
    elif kind == "working":
        painter.drawEllipse(m, m, size - 2 * m, size - 2 * m)
        painter.drawLine(size // 2, m + 2, size // 2, m + 6)
        painter.drawLine(size // 2, size // 2, size // 2 + 5, size // 2 - 3)
    elif kind == "figure":
        painter.drawRect(m, m, size - 2 * m, size - 2 * m)
        painter.drawLine(m, m + 6, size - m, m + 6)
    elif kind == "layout":
        painter.drawRect(m, m, size - 2 * m, size - 2 * m)
        inset = 5
        painter.drawRect(m + inset, m + inset, size - 2 * m - 2 * inset, size - 2 * m - 2 * inset)
    elif kind == "axes":
        painter.drawLine(m, size - m, m, m)
        painter.drawLine(m, size - m, size - m, size - m)
        x = m + 4
        while x < size - m:
            painter.drawLine(x, size - m, x, size - m - 2)
            x += 5
        y = m + 4
        while y < size - m:
            painter.drawLine(m, y, m + 2, y)
            y += 5
    painter.end()
    return QIcon(pixmap)


class ToolDrawer(QWidget):
    """Compact DSO-style vertical tool strip with an adjacent drawer.

    The strip holds large, always-visible, checkable buttons (icon + short
    label + tooltip -- never icon alone); the drawer is a `QStackedWidget`
    that shows exactly one registered page's controls at a time, docked
    next to the strip rather than floating over the graph. `side="right"`
    mirrors the strip to the outer edge for a right-docked instance (e.g.
    the Working Data drawer) while keeping identical behavior.

    Clicking a strip button opens its page (replacing whatever page was
    showing). Clicking the *already active* button collapses the drawer
    down to just the strip -- pages are hidden, not destroyed, so their
    state (list selection, scroll position, in-progress edits) survives.
    `collapsed_changed` lets the owner reclaim the freed width for the plot
    canvas; see `MainWindow._set_side_drawer_collapsed`.
    """

    collapsed_changed = Signal(bool)
    page_changed = Signal(str)

    def __init__(self, parent=None, side: str = "left"):
        super().__init__(parent)
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self._side = side
        self._buttons: dict[str, QToolButton] = {}
        self._pages: dict[str, QWidget] = {}
        self._active_key: str | None = None

        self._strip = QWidget()
        # Plain objectName per side rather than a `[side="..."]` dynamic-
        # property QSS selector: Qt's stylesheet engine has to evaluate
        # property-based selectors per-widget on every full re-polish
        # (e.g. QApplication::setStyleSheet, called once per MainWindow at
        # startup), which is dramatically more expensive at application
        # scope than a plain objectName match -- previously measured at
        # multiple seconds once enough widgets existed application-wide.
        self._strip.setObjectName("ToolStripLeft" if side == "left" else "ToolStripRight")
        self._strip.setFixedWidth(STRIP_WIDTH)
        self._strip_layout = QVBoxLayout(self._strip)
        self._strip_layout.setContentsMargins(4, 8, 4, 8)
        self._strip_layout.setSpacing(4)
        self._strip_layout.addStretch(1)

        self._stack = QStackedWidget()
        self._stack.setVisible(False)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        # Mirrored for the right-hand drawer -- the strip stays on the
        # outer edge (nearest the window edge) with its page docked
        # between the strip and the plot canvas, on both sides.
        if side == "left":
            outer.addWidget(self._strip)
            outer.addWidget(self._stack, 1)
        else:
            outer.addWidget(self._stack, 1)
            outer.addWidget(self._strip)

    @property
    def strip_width(self) -> int:
        return STRIP_WIDTH

    def add_page(self, key: str, label: str, tooltip: str, icon_kind: str, widget: QWidget) -> None:
        button = QToolButton()
        button.setObjectName("ToolStripButtonLeft" if self._side == "left" else "ToolStripButtonRight")
        button.setCheckable(True)
        button.setText(label)
        button.setToolTip(tooltip)
        button.setIcon(_make_icon(icon_kind))
        button.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        button.setMinimumHeight(56)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: self._on_button_clicked(key))

        # Insert before the trailing stretch so buttons stack from the top.
        self._strip_layout.insertWidget(self._strip_layout.count() - 1, button)
        self._buttons[key] = button
        self._pages[key] = widget
        self._stack.addWidget(widget)

    def _on_button_clicked(self, key: str) -> None:
        if self._active_key == key:
            self.collapse()
        else:
            self.show_page(key)

    def show_page(self, key: str) -> None:
        if key not in self._pages:
            return
        was_collapsed = self._active_key is None
        self._active_key = key
        self._stack.setCurrentWidget(self._pages[key])
        self._stack.setVisible(True)
        for button_key, button in self._buttons.items():
            button.setChecked(button_key == key)
        if was_collapsed:
            self.collapsed_changed.emit(False)
        self.page_changed.emit(key)

    def collapse(self) -> None:
        if self._active_key is None:
            return
        self._active_key = None
        self._stack.setVisible(False)
        for button in self._buttons.values():
            button.setChecked(False)
        self.collapsed_changed.emit(True)

    @property
    def active_key(self) -> str | None:
        return self._active_key

    @property
    def is_collapsed(self) -> bool:
        return self._active_key is None
