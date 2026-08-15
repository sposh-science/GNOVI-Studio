from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSizePolicy, QToolButton, QVBoxLayout, QWidget

_EXPANDED_MARK = "▾"  # ▾
_COLLAPSED_MARK = "▸"  # ▸


class CollapsibleSection(QWidget):
    """Wraps a single content widget with a header that toggles its visibility.

    Collapsing only ever changes the *content widget's* own visibility --
    it never reaches into whatever is nested inside it -- so any dynamic
    show/hide state a panel already manages internally (e.g. plot-type
    dependent controls) survives a collapse/expand cycle untouched.
    """

    toggled = Signal(bool)

    def __init__(self, title: str, content: QWidget, parent=None, expanded: bool = True):
        super().__init__(parent)
        self._title = title
        self._content = content

        self.toggle_button = QToolButton(self)
        self.toggle_button.setProperty("collapsible", True)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle_button.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self._content)

        self._set_button_text(expanded)
        self._content.setVisible(expanded)
        self.toggle_button.toggled.connect(self._on_toggled)

    def _set_button_text(self, expanded: bool) -> None:
        mark = _EXPANDED_MARK if expanded else _COLLAPSED_MARK
        self.toggle_button.setText(f"{mark} {self._title}")

    def _on_toggled(self, checked: bool) -> None:
        self._set_button_text(checked)
        self._content.setVisible(checked)
        self.toggled.emit(checked)

    def is_expanded(self) -> bool:
        return self.toggle_button.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setChecked(expanded)

    @property
    def content(self) -> QWidget:
        return self._content
