"""Spin/combo controls that only respond to a wheel scroll after an
explicit click, not merely by having focus.

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

A `hasFocus()` check alone is not enough: these widget classes default
to `Qt.WheelFocus`, which -- per Qt's own documentation -- means a wheel
scroll over an *unfocused* control is itself one of the built-in ways to
grant it focus, on real mouse/touchpad input (confirmed on real hardware
during #57 manual testing; not reproducible through a synthetic
`QWheelEvent` sent directly to a widget in an offscreen test, which
bypasses whatever platform-dispatch step performs that grant -- see
tests/test_scroll_safe_controls.py's own module docstring). So the wheel
event that's supposed to be rejected can itself have already made
`hasFocus()` true by the time `wheelEvent()` runs.

Fixed at the root, in two parts:

1. `setFocusPolicy(Qt.StrongFocus)` -- `StrongFocus` is `WheelFocus`
   minus exactly the bit that lets a wheel scroll grant focus on its
   own. Tab and click focus are unaffected. This alone stops a hover
   scroll from silently becoming a focus grant.
2. An explicit "click-armed" flag, set only by `focusInEvent` with
   `reason() == Qt.FocusReason.MouseFocusReason` (a genuine click) and
   cleared by `focusOutEvent` -- deliberately *not* armed by Tab/
   keyboard focus or any other reason, matching the required GNOVI 1.0
   interaction: hover, and Tab/keyboard focus, must scroll the
   surrounding panel; only an explicit mouse click on the control makes
   it wheel-adjustable, until focus leaves it again.

An unfocused-or-not-click-armed control's `wheelEvent` calls
`event.ignore()` -- Qt's own documented behaviour for an ignored wheel
event is to propagate it to the parent widget, so it reaches the
containing `QScrollArea` the same way it already does today for a wheel
scroll over a `QLabel`, `QGroupBox`, or empty space in these same panels
(no special-casing needed there; this is why only spin/combo controls
need this override, not every widget in the workflow). A click-armed
control keeps its ordinary Qt wheel behaviour unchanged.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class _WheelRequiresExplicitClick:
    """Shared behaviour, mixed into each concrete control below rather
    than duplicated three times. Not usable on its own -- always the
    first base alongside the real Qt widget class, so `super()` in each
    override chains into that widget's own implementation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)
        self._wheel_armed = False

    def focusInEvent(self, event: QFocusEvent) -> None:
        self._wheel_armed = event.reason() == Qt.FocusReason.MouseFocusReason
        super().focusInEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        self._wheel_armed = False
        super().focusOutEvent(event)

    def wheelEvent(self, event) -> None:
        if not self._wheel_armed:
            event.ignore()
            return
        super().wheelEvent(event)


class ScrollSafeSpinBox(_WheelRequiresExplicitClick, QSpinBox):
    pass


class ScrollSafeDoubleSpinBox(_WheelRequiresExplicitClick, QDoubleSpinBox):
    pass


class ScrollSafeComboBox(_WheelRequiresExplicitClick, QComboBox):
    pass
