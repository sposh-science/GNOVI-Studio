"""Issue #56/#57: a ScrollSafe* control only responds to a wheel scroll
after an explicit mouse click, not merely by having focus.

Real-device manual testing on PR #57 found that a `hasFocus()`-only check
is insufficient: these widget classes default to `Qt.WheelFocus`, and a
wheel scroll over an *unfocused* control is itself one of Qt's built-in
ways to grant it focus on real mouse/touchpad input -- so the very wheel
event that's supposed to be rejected can have already made `hasFocus()`
true by the time `wheelEvent()` runs. This was not reproducible through a
synthetic `QWheelEvent` sent directly to a widget via
`QApplication.sendEvent` in this offscreen test environment, which
apparently bypasses whatever platform-dispatch step performs that grant.

The fix removes the wheel-focus policy bit entirely (`Qt.StrongFocus`,
verified below) and additionally tracks an explicit "click-armed" state
via `QFocusEvent.reason()`, armed only by `Qt.FocusReason.MouseFocusReason`
and cleared on `focusOutEvent`. These tests exercise that state machine
through Qt's own real `QWidget.setFocus(reason)` API (which delivers a
genuine `QFocusEvent` with that reason to `focusInEvent`, the same as a
real click/Tab/programmatic focus change would) -- not a fabricated
event. What they do NOT and cannot prove is the original real-device
symptom itself (a genuine touchpad wheel scroll silently granting focus)
-- that remains a required manual acceptance test on real hardware
(scenarios A/B/C in the #57 PR description), not something asserted here.
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

_ALL_CLASSES = (ScrollSafeSpinBox, ScrollSafeDoubleSpinBox, ScrollSafeComboBox)


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


def _focus_with_reason(widget: QWidget, reason: Qt.FocusReason) -> None:
    widget.setFocus(reason)
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


# --- Focus policy: the wheel itself must never grant focus --------------


def test_focus_policy_excludes_wheel_focus(qapp):
    for cls in _ALL_CLASSES:
        widget = cls()
        assert widget.focusPolicy() == Qt.StrongFocus, cls.__name__
        # StrongFocus keeps Tab and Click; only the WheelFocus-only bit
        # (Qt.WheelFocus & ~Qt.StrongFocus) must be absent.
        assert not (int(widget.focusPolicy()) & (int(Qt.WheelFocus) & ~int(Qt.StrongFocus)))


# --- Unfocused / hover-only: wheel must never change value -------------


def test_unfocused_double_spin_box_wheel_does_not_change_value(qapp):
    spin = ScrollSafeDoubleSpinBox()
    container = _host(spin)  # noqa: F841 -- keeps Qt ownership alive
    assert spin.hasFocus() is False
    assert spin._wheel_armed is False

    event = _send_wheel(spin)

    assert spin.value() == 0.0
    assert event.isAccepted() is False


def test_unfocused_spin_box_wheel_does_not_change_value(qapp):
    spin = ScrollSafeSpinBox()
    container = _host(spin)  # noqa: F841 -- keeps Qt ownership alive
    assert spin.hasFocus() is False

    event = _send_wheel(spin)

    assert spin.value() == 0
    assert event.isAccepted() is False


def test_unfocused_combo_box_wheel_does_not_change_selection(qapp):
    combo = ScrollSafeComboBox()
    combo.addItems(["First", "Second", "Third"])
    container = _host(combo)  # noqa: F841 -- keeps Qt ownership alive
    assert combo.hasFocus() is False

    event = _send_wheel(combo)

    assert combo.currentIndex() == 0
    assert event.isAccepted() is False


# --- Focus-reason arming: only an explicit click arms wheel adjustment --


def test_mouse_focus_reason_arms_wheel_adjustment(qapp):
    for cls in _ALL_CLASSES:
        widget = cls()
        container = _host(widget)  # noqa: F841

        _focus_with_reason(widget, Qt.FocusReason.MouseFocusReason)

        assert widget._wheel_armed is True, cls.__name__


def test_tab_focus_reason_does_not_arm_wheel_adjustment(qapp):
    for cls in _ALL_CLASSES:
        widget = cls()
        container = _host(widget)  # noqa: F841

        _focus_with_reason(widget, Qt.FocusReason.TabFocusReason)

        assert widget.hasFocus() is True, cls.__name__  # genuinely focused...
        assert widget._wheel_armed is False, cls.__name__  # ...but not wheel-armed


def test_other_focus_reasons_do_not_arm_wheel_adjustment(qapp):
    for reason in (Qt.FocusReason.OtherFocusReason, Qt.FocusReason.ActiveWindowFocusReason):
        for cls in _ALL_CLASSES:
            widget = cls()
            container = _host(widget)  # noqa: F841

            _focus_with_reason(widget, reason)

            assert widget._wheel_armed is False, (cls.__name__, reason)


def test_focus_out_disarms_wheel_adjustment(qapp):
    for cls in _ALL_CLASSES:
        widget = cls()
        container = _host(widget)  # noqa: F841
        _focus_with_reason(widget, Qt.FocusReason.MouseFocusReason)
        assert widget._wheel_armed is True, cls.__name__

        widget.clearFocus()
        QCoreApplication.processEvents()

        assert widget.hasFocus() is False, cls.__name__
        assert widget._wheel_armed is False, cls.__name__


# --- Behavioural consequence: armed vs. merely-focused ------------------


def test_tab_focused_but_not_click_armed_double_spin_box_ignores_wheel(qapp):
    spin = ScrollSafeDoubleSpinBox()
    container = _host(spin)  # noqa: F841
    _focus_with_reason(spin, Qt.FocusReason.TabFocusReason)
    assert spin.hasFocus() is True  # genuinely focused via Tab...
    value_before = spin.value()

    event = _send_wheel(spin)

    assert spin.value() == value_before  # ...but wheel must not touch it
    assert event.isAccepted() is False


def test_click_armed_double_spin_box_retains_normal_wheel_adjustment(qapp):
    spin = ScrollSafeDoubleSpinBox()
    container = _host(spin)  # noqa: F841
    _focus_with_reason(spin, Qt.FocusReason.MouseFocusReason)

    _send_wheel(spin)

    assert spin.value() != 0.0


def test_click_armed_spin_box_retains_normal_wheel_adjustment(qapp):
    spin = ScrollSafeSpinBox()
    container = _host(spin)  # noqa: F841
    _focus_with_reason(spin, Qt.FocusReason.MouseFocusReason)

    _send_wheel(spin)

    assert spin.value() != 0


def test_click_armed_combo_box_wheel_is_not_suppressed(qapp):
    # A click-armed ScrollSafeComboBox must fall through to the normal Qt
    # wheelEvent, not the ignore-branch -- verified via event acceptance
    # rather than an actual currentIndex change: even a plain, unmodified
    # QComboBox does not visibly change selection from a synthetic
    # QWheelEvent in this offscreen test environment (confirmed
    # separately, unrelated to this override), so that part of Qt's combo
    # wheel behaviour isn't reliably assertable headlessly.
    combo = ScrollSafeComboBox()
    combo.addItems(["First", "Second", "Third"])
    container = _host(combo)  # noqa: F841
    _focus_with_reason(combo, Qt.FocusReason.MouseFocusReason)

    event = _send_wheel(combo)

    assert event.isAccepted() is True
