from PySide6.QtWidgets import QDialogButtonBox, QLabel, QWidget

from gnovi_plot.gui.dialogs.live_dialog import LiveDialog
from gnovi_plot.gui.widgets.figure_properties_panel import FigurePropertiesPanel
from gnovi_plot.gui.widgets.figure_size_panel import FigureSizePanel
from gnovi_plot.plotting.figure import GnoviFigure


def test_dialog_shows_apply_reset_and_cancel_buttons(qapp):
    figure = GnoviFigure()
    dialog = LiveDialog("Test", FigureSizePanel(figure))

    box = dialog.button_box
    assert box.button(QDialogButtonBox.Apply) is not None
    assert box.button(QDialogButtonBox.Reset) is not None
    assert box.button(QDialogButtonBox.Cancel) is not None


def test_first_show_raised_captures_a_baseline_snapshot(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    dialog = LiveDialog("Figure Size", panel)

    dialog.show_raised()

    assert dialog._snapshot is not None
    dialog.close()


def test_cancel_reverts_edits_made_since_open(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    dialog = LiveDialog("Figure Size", panel)
    dialog.show_raised()

    panel.width_spin.setValue(9.0)
    assert figure.figure_width_in != 6.4

    dialog.reject()

    assert figure.figure_width_in == 6.4


def test_closing_via_reject_is_equivalent_to_cancel(qapp):
    """QDialog routes the window's close (X) button through reject() by
    default -- this is what makes closing the dialog without an explicit
    Cancel click still discard unsaved edits."""
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    dialog = LiveDialog("Figure Size", panel)
    dialog.show_raised()
    panel.width_spin.setValue(9.0)

    dialog.close()

    assert figure.figure_width_in == 6.4


def test_apply_rebaselines_so_a_later_cancel_only_reverts_edits_after_it(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    dialog = LiveDialog("Figure Size", panel)
    dialog.show_raised()

    panel.width_spin.setValue(9.0)
    dialog._on_apply()
    panel.width_spin.setValue(3.0)
    dialog.reject()

    assert figure.figure_width_in == 9.0


def test_reset_button_restores_defaults_without_closing(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    dialog = LiveDialog("Figure Size", panel)
    dialog.show_raised()
    panel.width_spin.setValue(20.0)

    dialog._on_reset()

    assert figure.figure_width_in == GnoviFigure().figure_width_in
    assert dialog.isVisible() is True
    dialog.close()


def test_reopening_an_already_visible_dialog_does_not_recapture_the_baseline(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    dialog = LiveDialog("Figure Size", panel)
    dialog.show_raised()
    panel.width_spin.setValue(9.0)

    dialog.show_raised()  # already open -- must not re-baseline to 9.0
    dialog.reject()

    assert figure.figure_width_in == 6.4


def test_axes_dialog_cancel_reverts_the_active_panel(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)
    dialog = LiveDialog("Axes", panel)
    dialog.show_raised()

    panel.title_edit.setText("Changed")
    panel._apply_title()
    assert figure.active_panel.title == "Changed"

    dialog.reject()

    assert figure.active_panel.title == ""


def test_content_without_the_snapshot_interface_still_gets_working_buttons(qapp):
    """A LiveDialog content widget that implements none of capture_state/
    restore_state/reset_to_defaults must not crash the dialog -- Cancel and
    Reset just become no-ops for it."""
    content = QWidget()
    content.layout_hint = QLabel("plain", content)
    dialog = LiveDialog("Plain", content)

    dialog.show_raised()
    dialog._on_reset()
    dialog.reject()  # must not raise
