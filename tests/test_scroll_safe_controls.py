"""Issue #56: an unfocused ScrollSafe* control must ignore a wheel scroll
(never change value/selection); a focused one keeps normal Qt wheel
behaviour.

These tests verify the value-protection contract directly -- reliably
reproducible via QApplication.sendEvent, as confirmed during the #56
audit. Whether the surrounding QScrollArea then scrolls is Qt's own
documented behaviour for an ignored wheel event (and is already relied
on today for every other widget type in this same workflow -- a QLabel
or QGroupBox already lets wheel-scroll bubble to the parent with no
special code), but a synthetic QWheelEvent sent directly to a widget
does not reliably exercise that same top-level dispatch path in an
automated test -- so that part is verified manually, not asserted here
(see the #56 PR description's manual verification steps).
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

from gnovi_plot.gui.widgets.scroll_safe_controls import (
    ScrollSafeComboBox,
    ScrollSafeDoubleSpinBox,
    ScrollSafeSpinBox,
)


def _host(widget: QWidget) -> QWidget:
    """A decoy-focused container so `widget` starts genuinely unfocused --
    a bare top-level widget can otherwise pick up default focus itself."""
    container = QWidget()
    layout = QVBoxLayout(container)
    decoy = QPushButton("decoy")
    layout.addWidget(decoy)
    layout.addWidget(widget)
    container.show()
    QCoreApplication.processEvents()
    decoy.setFocus()
    QCoreApplication.processEvents()
    return container


def _give_focus(widget: QWidget) -> None:
    widget.setFocus()
    QCoreApplication.processEvents()


def _send_wheel(widget: QWidget) -> QWheelEvent:
    event = QWheelEvent(
        QPointF(widget.rect().center()),
        QPointF(widget.mapToGlobal(widget.rect().center())),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(widget, event)
    QCoreApplication.processEvents()
    return event


def test_unfocused_double_spin_box_wheel_does_not_change_value(qapp):
    spin = ScrollSafeDoubleSpinBox()
    container = _host(spin)  # noqa: F841 -- keeps Qt ownership alive
    assert spin.hasFocus() is False

    event = _send_wheel(spin)

    assert spin.value() == 0.0
    assert event.isAccepted() is False


def test_focused_double_spin_box_wheel_changes_value_normally(qapp):
    spin = ScrollSafeDoubleSpinBox()
    container = _host(spin)  # noqa: F841 -- keeps Qt ownership alive
    _give_focus(spin)
    assert spin.hasFocus() is True

    _send_wheel(spin)

    assert spin.value() != 0.0


def test_unfocused_spin_box_wheel_does_not_change_value(qapp):
    spin = ScrollSafeSpinBox()
    container = _host(spin)  # noqa: F841 -- keeps Qt ownership alive
    assert spin.hasFocus() is False

    event = _send_wheel(spin)

    assert spin.value() == 0
    assert event.isAccepted() is False


def test_focused_spin_box_wheel_changes_value_normally(qapp):
    spin = ScrollSafeSpinBox()
    container = _host(spin)  # noqa: F841 -- keeps Qt ownership alive
    _give_focus(spin)

    _send_wheel(spin)

    assert spin.value() != 0


def test_unfocused_combo_box_wheel_does_not_change_selection(qapp):
    combo = ScrollSafeComboBox()
    combo.addItems(["First", "Second", "Third"])
    container = _host(combo)  # noqa: F841 -- keeps Qt ownership alive
    assert combo.hasFocus() is False

    event = _send_wheel(combo)

    assert combo.currentIndex() == 0
    assert event.isAccepted() is False


def test_focused_combo_box_wheel_is_not_suppressed(qapp):
    # A focused ScrollSafeComboBox must fall through to the normal Qt
    # wheelEvent, not the ignore-branch -- verified via event acceptance
    # rather than an actual currentIndex change: even a plain, unmodified
    # QComboBox does not visibly change selection from a synthetic
    # QWheelEvent in this offscreen test environment (confirmed
    # separately, unrelated to this override), so that part of Qt's combo
    # wheel behaviour isn't reliably assertable headlessly. Acceptance is:
    # unfocused -> ignored (see test_unfocused_combo_box_wheel_...),
    # focused -> accepted, same as any ordinary QComboBox.
    combo = ScrollSafeComboBox()
    combo.addItems(["First", "Second", "Third"])
    container = _host(combo)  # noqa: F841 -- keeps Qt ownership alive
    _give_focus(combo)

    event = _send_wheel(combo)

    assert event.isAccepted() is True
