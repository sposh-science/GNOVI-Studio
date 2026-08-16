from PySide6.QtWidgets import QApplication, QMessageBox

from gnovi_plot.core.app_info import APP_NAME, APP_TAGLINE, VERSION_LABEL
from gnovi_plot.gui.dialogs.export_figure_dialog import ExportFigureDialog
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.styles import PlotTheme, build_stylesheet


def test_window_title_is_the_product_name(qapp):
    window = MainWindow()
    assert window.windowTitle() == f"Untitled Project — {APP_NAME}"
    window.close()


def test_about_dialog_shows_name_tagline_and_version(qapp, monkeypatch):
    captured = []
    monkeypatch.setattr(QMessageBox, "about", staticmethod(lambda *args: captured.append(args)))

    window = MainWindow()
    window._show_about()

    assert len(captured) == 1
    _parent, title, text = captured[0]
    assert APP_NAME in title
    assert APP_NAME in text
    assert APP_TAGLINE in text
    assert VERSION_LABEL in text
    window.close()


def test_plot_theme_menu_defaults_to_light(qapp):
    window = MainWindow()
    assert window._theme_actions[PlotTheme.LIGHT].isChecked()
    assert not window._theme_actions[PlotTheme.DARK].isChecked()
    window.close()


def test_selecting_dark_plot_theme_updates_state_and_settings(qapp):
    window = MainWindow()

    window._on_theme_changed(PlotTheme.DARK)

    assert window.figure_model.plot_theme == PlotTheme.DARK
    assert window._settings.value("plot_theme") == "dark"
    window.close()


def test_plot_theme_toolbar_combo_offers_only_light_and_dark(qapp):
    """No "System" option -- Plot Theme is an explicit Light/Dark choice for
    the canvas only (see gui.styles.PlotTheme)."""
    window = MainWindow()
    combo = window.toolbar_theme_combo

    assert combo.toolTip() == "Plot Theme"
    assert [combo.itemText(i) for i in range(combo.count())] == ["Light", "Dark"]
    window.close()


def test_selecting_dark_plot_theme_recolors_the_canvas_only(qapp):
    window = MainWindow()
    chrome_stylesheet_before = QApplication.instance().styleSheet()

    window._on_theme_changed(PlotTheme.DARK)

    assert window.plot_canvas.figure.get_facecolor() != (1.0, 1.0, 1.0, 1.0)
    # The application chrome's stylesheet is untouched by a Plot Theme
    # change -- it stays the single fixed light stylesheet throughout.
    assert QApplication.instance().styleSheet() == chrome_stylesheet_before == build_stylesheet()
    window.close()


def test_export_dialog_background_defaults_to_as_shown_matching_the_live_figure(qapp):
    """Export Figure is WYSIWYG: it saves the live on-screen Figure, which
    already reflects the figure's current Plot Theme (declarative
    `GnoviFigure` state, see `plotting.figure.PlotTheme`) via
    `MainWindow._rerender`'s `dark_mode` -- so "As shown" (the Background
    combo's default) alone reproduces the figure's current appearance,
    with no separate theme-tracking checkbox needed. Opening the dialog
    never mutates the figure's own theme."""
    window = MainWindow()
    window._on_theme_changed(PlotTheme.DARK)

    dark_dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    assert dark_dialog.background_combo.currentText() == "As shown"
    assert window.figure_model.plot_theme == PlotTheme.DARK  # opening the dialog never mutates it
    assert window.plot_canvas.figure.get_facecolor() != (1.0, 1.0, 1.0, 1.0)

    window._on_theme_changed(PlotTheme.LIGHT)
    light_dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    assert light_dialog.background_combo.currentText() == "As shown"
    assert window.plot_canvas.figure.get_facecolor() == (1.0, 1.0, 1.0, 1.0)

    window.close()
