"""Spin/combo controls that ignore an unfocused mouse-wheel scroll.

`QAbstractSpinBox`/`QComboBox` respond to a wheel event whenever the
cursor is over them, with no focus required -- inside a long, scrollable
analysis workflow (XRD Peak Analysis, Cyclic Voltammetry, and Curve
Fitting all pack many spin/combo controls into `AnalysisPanel`'s
`workflow_scroll`, see that module), a user trying to scroll the page
with the mouse wheel will, almost anywhere their cursor lands, silently
change whatever parameter control happens to be underneath it instead --
a real risk for scientific analysis inputs (peak-detection thresholds,
background parameters, fit-window bounds, ...), not just a cosmetic
annoyance.

These subclasses restore the expected interaction: click/focus a control
before the wheel adjusts it. An unfocused control's `wheelEvent` calls
`event.ignore()` -- Qt's own documented behaviour for an ignored wheel
event is to propagate it to the parent widget, so it reaches the
containing `QScrollArea` the same way it already does today for a wheel
scroll over a `QLabel`, `QGroupBox`, or empty space in these same panels
(no special-casing needed there; this is why only spin/combo controls
need this override, not every widget in the workflow). A focused control
keeps its ordinary Qt wheel behaviour unchanged.
"""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class ScrollSafeSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class ScrollSafeDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class ScrollSafeComboBox(QComboBox):
    def wheelEvent(self, event) -> None:
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)
