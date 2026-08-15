from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFrame, QScrollArea, QVBoxLayout, QWidget

_MAX_SCREEN_WIDTH_FRACTION = 0.7
_MAX_SCREEN_HEIGHT_FRACTION = 0.85


class LiveDialog(QDialog):
    """Non-modal, scrollable dialog that hosts a persistent content widget,
    with Apply / Reset / Cancel behavior around it.

    The dialog wraps -- rather than rebuilds -- `content`, so it is created
    once and every subsequent menu/toolbar action just raises the same
    instance. That trivially satisfies "preserve settings when reopened" and
    "update preview live" (the content widget stays bound to the live model
    the whole time, so every edit renders immediately), and non-modal so it
    can stay open while the user keeps interacting with the plot.

    Edits still apply live (as before) so the plot always reflects the
    dialog's current state -- but a per-open baseline is captured, and:
      - **Apply** re-baselines the current (already-live) state, so a
        subsequent Cancel reverts only edits made *since* this Apply.
      - **Cancel** (or closing the dialog, e.g. its window X, which Qt
        routes through `reject()`) restores that baseline and closes.
      - **Reset** restores `content` to its defaults, live, without closing.

    `content` opts in to this by implementing `capture_state()` /
    `restore_state(state)` / `reset_to_defaults()`; a `content` that
    implements none of them still gets the button row, just as a no-op
    Cancel/Reset (only Apply's "close" behavior would remain meaningful) --
    no LiveDialog caller currently omits the interface, but duck-typing here
    means adding a future non-conforming content widget fails soft, not with
    an `AttributeError` on open.
    """

    def __init__(self, title: str, content: QWidget, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self._content = content
        self._snapshot = None

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Apply | QDialogButtonBox.Reset | QDialogButtonBox.Cancel
        )
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        self.button_box.button(QDialogButtonBox.Reset).clicked.connect(self._on_reset)
        self.button_box.button(QDialogButtonBox.Cancel).clicked.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        layout.addWidget(self.button_box)

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
        height = min(max(hint.height() + 32, 320) + self.button_box.sizeHint().height(), max_height)
        self.resize(width, height)

    def show_raised(self) -> None:
        """Bring the dialog to front, reusing whatever state it already has.
        Captures a fresh Cancel baseline only when actually (re)opening --
        raising an already-open dialog leaves the current baseline alone."""
        if not self.isVisible():
            self._capture_snapshot()
        self.show()
        self.raise_()
        self.activateWindow()

    def _capture_snapshot(self) -> None:
        capture = getattr(self._content, "capture_state", None)
        self._snapshot = capture() if capture is not None else None

    def _on_apply(self) -> None:
        """Re-baseline: the current (already-live) state becomes the new
        Cancel target, so only edits made after this point are revertible."""
        self._capture_snapshot()

    def _on_reset(self) -> None:
        reset = getattr(self._content, "reset_to_defaults", None)
        if reset is not None:
            reset()

    def reject(self) -> None:
        if self._snapshot is not None:
            restore = getattr(self._content, "restore_state", None)
            if restore is not None:
                restore(self._snapshot)
        self._snapshot = None
        super().reject()
