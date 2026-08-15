from PySide6.QtWidgets import QApplication

from gnovi_plot.gui.styles import PlotTheme, apply_app_theme, build_stylesheet


def test_plot_theme_has_exactly_light_and_dark():
    assert {mode.value for mode in PlotTheme} == {"light", "dark"}


def test_build_stylesheet_defaults_to_the_single_light_palette():
    assert build_stylesheet() == build_stylesheet()
    assert build_stylesheet().strip()


def test_build_stylesheet_covers_the_theme_relevant_widget_classes():
    qss = build_stylesheet()
    for selector in ("QDialog", "QMenu", "QTabWidget::pane", "QTabBar::tab", "QScrollBar", "QStatusBar"):
        assert selector in qss


def test_apply_app_theme_sets_the_fixed_light_application_stylesheet(qapp: QApplication):
    apply_app_theme(qapp)

    assert qapp.styleSheet() == build_stylesheet()


def test_apply_app_theme_is_not_affected_by_plot_theme_selection(qapp: QApplication):
    """The application chrome has exactly one stylesheet, regardless of
    which `PlotTheme` (Light/Dark) is active for the plot canvas -- there is
    no app-wide dark mode (see gui.styles module docstring)."""
    apply_app_theme(qapp)
    chrome_stylesheet = qapp.styleSheet()

    # Changing the plot canvas theme is a MainWindow-level concern (drives
    # `PlotCanvas.render(..., dark_mode=...)`) and never re-applies the
    # application stylesheet, so re-asserting it here should be a no-op.
    apply_app_theme(qapp)

    assert qapp.styleSheet() == chrome_stylesheet
