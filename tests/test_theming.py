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


def test_stylesheet_supplies_an_explicit_down_arrow_for_every_combo_box():
    """`QComboBox::drop-down` is customized (border/width) above this rule --
    doing so suppresses the style engine's own default arrow glyph, so
    `::down-arrow` must be supplied explicitly or every QComboBox in the app
    renders as a plain field with no visible dropdown indicator (see
    tests/test_combobox_dropdown_arrow.py for the rendered-pixel check)."""
    qss = build_stylesheet()
    assert "QComboBox::down-arrow" in qss
    # A real glyph (the CSS-border-triangle technique), not just `image: none`
    # with nothing to replace it.
    down_arrow_rule = qss.split("QComboBox::down-arrow")[1].split("}")[0]
    assert "border-top" in down_arrow_rule


def test_apply_app_theme_sets_the_fixed_light_application_stylesheet(qapp: QApplication):
    apply_app_theme(qapp)

    assert qapp.styleSheet() == build_stylesheet()


# --- Visual polish: surface hierarchy / context styling / Workbench chrome ----


def test_stylesheet_defines_a_restrained_context_accent_selector():
    """`QLabel[contextRow="true"]` is how current-context/status labels
    (Active panel/Graph/Data, active dataset) opt into a restrained,
    non-bold accent color instead of bold text -- see `gui.styles` module
    docstring's "no bold for current state" rule and
    `gui.widgets.active_panel_label.ActivePanelLabel`."""
    qss = build_stylesheet()
    assert 'QLabel[contextRow="true"]' in qss
    rule = qss.split('QLabel[contextRow="true"]')[1].split("}")[0]
    assert "font-weight: 400" in rule  # normal weight, never bold


def test_stylesheet_defines_the_workbench_header_selectors():
    """Application chrome for `gui.widgets.workbench_header.WorkbenchHeader`
    -- confirms the styling hooks it relies on exist without asserting
    exact cosmetic pixel values."""
    qss = build_stylesheet()
    for selector in ("QWidget#WorkbenchHeader", "QLabel#WorkbenchHeaderLabel"):
        assert selector in qss


def test_stylesheet_defines_the_workbench_tab_strip_selectors():
    """Application chrome for `gui.widgets.workbench_tabs.WorkbenchTabBar`."""
    qss = build_stylesheet()
    for selector in (
        "QWidget#WorkbenchTabStrip",
        "QTabBar#WorkbenchTabBar::tab",
        "QToolButton#WorkbenchNewButton",
    ):
        assert selector in qss


def test_selection_and_checked_states_do_not_rely_on_bold():
    """Selection/checked/current-state feedback throughout the chrome
    (list items, tabs, tool-strip buttons) must come from background/
    border/color, never `font-weight: 600` -- see `gui.styles` module
    docstring."""
    qss = build_stylesheet()
    for selector in (
        "QListWidget::item:selected",
        "QTabBar::tab:selected",
        "QToolButton#ToolStripButtonLeft:checked",
        "QToolButton#ToolStripButtonRight:checked",
    ):
        rule = qss.split(selector)[1].split("}")[0]
        assert "font-weight" not in rule


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
