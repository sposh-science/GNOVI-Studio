"""Centralized QSS styling for Gnovi Studio.

Restrained, modern scientific-desktop look using PySide6/QSS only -- no
additional theme dependency. Deliberately avoids hard-coded font families so
each platform's native font is used.

Gnovi Studio's own chrome (menus, toolbars, sidebars, bottom panel, data
table, dialogs, status bar) is always styled with the single light palette
below -- it is never user-switchable. What *is* user-switchable is the
`PlotTheme` applied to the Matplotlib preview canvas only (see
`plotting.backends.matplotlib_backend.render_figure`'s `dark_mode` flag);
the two are deliberately independent so a dark plot never drags the rest of
the interface into dark mode with it. `PlotTheme` itself is declarative
`GnoviFigure` state (see `plotting.figure.PlotTheme`'s docstring for why),
re-exported here only so existing `from gnovi_plot.gui.styles import
PlotTheme` call sites keep working.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from gnovi_plot.plotting.figure import PlotTheme  # noqa: F401 -- re-exported, see module docstring

_LIGHT_PALETTE = {
    "background": "#f4f5f7",
    "panel_background": "#ffffff",
    "subtle_background": "#f8f9fb",
    "border": "#d5d8dd",
    "text": "#20242b",
    "muted_text": "#5b6270",
    "accent": "#2f6fed",
    "accent_hover": "#255ac9",
    "accent_pressed": "#1e4aa8",
    "accent_text": "#ffffff",
    "selection_bg": "#e4ecfd",
    "scrollbar": "#c7cbd3",
}

# A single stale-series indicator color -- the application chrome (where
# this labels list items, not the plot canvas) is always light, so this
# never needs a dark counterpart.
STALE_COLOR = "#e5484d"

# Non-modal low-contrast-warning text color (see
# gui.widgets.plot_series_panel) -- same "always-light chrome" reasoning as
# STALE_COLOR above.
WARNING_COLOR = "#b06000"

# GUI-only active-panel badge background (see
# gui.widgets.plot_canvas._ActivePanelBadge) -- deliberately the same
# "accent_pressed" shade already used for pressed-button state above,
# rather than the plain `accent` also used for list-item selection/focus,
# so the badge reads as its own distinct, still-on-brand signal. An opaque
# pill with white text, so it stays legible sitting over either a Light or
# Dark Plot Theme canvas background without needing its own dark variant.
ACTIVE_PANEL_BADGE_COLOR = "#1e4aa8"

_STYLESHEET_TEMPLATE = """
QWidget {{
    background-color: {background};
    color: {text};
    font-size: 10pt;
}}

QMainWindow {{
    background-color: {background};
}}

QDialog {{
    background-color: {background};
}}

QGroupBox {{
    background-color: {panel_background};
    border: 1px solid {border};
    border-radius: 6px;
    margin-top: 16px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    color: {muted_text};
    letter-spacing: 0.3px;
}}

QPushButton {{
    background-color: {panel_background};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 5px 12px;
    min-height: 22px;
}}

QPushButton:hover {{
    border-color: {accent};
}}

QPushButton:pressed {{
    background-color: {selection_bg};
}}

QPushButton:disabled {{
    color: {muted_text};
}}

QPushButton[primary="true"] {{
    background-color: {accent};
    border-color: {accent};
    color: {accent_text};
    font-weight: 600;
}}

QPushButton[primary="true"]:hover {{
    background-color: {accent_hover};
    border-color: {accent_hover};
}}

QPushButton[primary="true"]:pressed {{
    background-color: {accent_pressed};
    border-color: {accent_pressed};
}}

QPushButton[primary="true"]:disabled {{
    background-color: {subtle_background};
    border-color: {border};
    color: {muted_text};
}}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {panel_background};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 22px;
}}

QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {accent};
}}

QComboBox::drop-down {{
    border: none;
    width: 18px;
}}

/* Styling `::drop-down` above suppresses the style engine's own default
arrow glyph -- Qt then expects `::down-arrow` to supply one explicitly.
Drawn as a plain CSS-border triangle (no icon/image resource needed) so
every QComboBox in the app keeps an obvious, standard-looking dropdown
chevron rather than looking like a plain text field. */
QComboBox::down-arrow {{
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {muted_text};
    margin-right: 7px;
}}

QComboBox::down-arrow:disabled {{
    border-top-color: {border};
}}

QListWidget, QTableView {{
    background-color: {panel_background};
    border: 1px solid {border};
    border-radius: 4px;
    alternate-background-color: {subtle_background};
    gridline-color: {border};
}}

QListWidget::item {{
    padding: 4px 6px 4px 8px;
    border-radius: 3px;
    border-left: 3px solid transparent;
}}

QListWidget::item:selected {{
    background-color: {selection_bg};
    color: {text};
    border-left: 3px solid {accent};
    font-weight: 600;
}}

QTableView::item:selected {{
    background-color: {selection_bg};
    color: {text};
}}

QHeaderView::section {{
    background-color: {subtle_background};
    border: none;
    border-right: 1px solid {border};
    border-bottom: 1px solid {border};
    padding: 5px 6px;
    font-weight: 600;
    letter-spacing: 0.2px;
    color: {muted_text};
}}

QCheckBox {{
    spacing: 6px;
}}

QSplitter::handle {{
    background-color: {border};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

QToolButton[collapsible="true"] {{
    background-color: transparent;
    border: none;
    text-align: left;
    padding: 4px 2px;
    font-weight: 600;
    color: {muted_text};
}}

QToolButton[collapsible="true"]:hover {{
    color: {text};
}}

QWidget#ToolStripLeft {{
    background-color: {panel_background};
    border-right: 1px solid {border};
}}

QWidget#ToolStripRight {{
    background-color: {panel_background};
    border-left: 1px solid {border};
}}

QToolButton#ToolStripButtonLeft, QToolButton#ToolStripButtonRight {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 6px 2px;
    color: {muted_text};
    font-size: 8pt;
}}

QToolButton#ToolStripButtonLeft:hover, QToolButton#ToolStripButtonRight:hover {{
    background-color: {subtle_background};
    color: {text};
}}

QToolButton#ToolStripButtonLeft:checked {{
    background-color: {selection_bg};
    color: {text};
    border-left: 3px solid {accent};
    font-weight: 600;
}}

QToolButton#ToolStripButtonRight:checked {{
    background-color: {selection_bg};
    color: {text};
    border-right: 3px solid {accent};
    font-weight: 600;
}}

QMenuBar {{
    background-color: {panel_background};
    border-bottom: 1px solid {border};
}}

QMenuBar::item:selected {{
    background-color: {selection_bg};
}}

QMenu {{
    background-color: {panel_background};
    border: 1px solid {border};
    padding: 4px;
}}

QMenu::item {{
    padding: 4px 20px 4px 12px;
    border-radius: 3px;
}}

QMenu::item:selected {{
    background-color: {selection_bg};
    color: {text};
}}

QToolBar {{
    background-color: {panel_background};
    border-bottom: 1px solid {border};
    spacing: 4px;
}}

QStatusBar {{
    background-color: {panel_background};
    border-top: 1px solid {border};
    color: {muted_text};
}}

QTabWidget::pane {{
    background-color: {panel_background};
    border: 1px solid {border};
    border-radius: 4px;
    top: -1px;
}}

QTabBar::tab {{
    background-color: {subtle_background};
    border: 1px solid {border};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 5px 12px;
    color: {muted_text};
}}

QTabBar::tab:selected {{
    background-color: {panel_background};
    color: {text};
    font-weight: 600;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {scrollbar};
    border-radius: 5px;
    min-height: 24px;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 12px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {scrollbar};
    border-radius: 5px;
    min-width: 24px;
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
"""


def build_stylesheet(palette: dict[str, str] = _LIGHT_PALETTE) -> str:
    """Render the shared QSS template against `palette` (the app's single
    light palette by default, or a custom dict with the same keys)."""
    return _STYLESHEET_TEMPLATE.format(**palette)


_THEME_APPLIED_PROPERTY = "_gnovi_studio_theme_applied"


def apply_app_theme(app: QApplication) -> None:
    """Apply Gnovi Studio's one, fixed, light application stylesheet.

    Not user-switchable and not affected by `PlotTheme` -- see the module
    docstring for why the app chrome and the plot canvas theme are kept
    independent.

    Idempotent per `QApplication` instance (guarded by a dynamic property):
    the real app calls this exactly once (`MainWindow.__init__`, and there
    is only ever one `MainWindow`), so the guard changes nothing there --
    but `QApplication::setStyleSheet` unconditionally re-polishes every
    widget the application has ever created, including ones from windows
    that were `.close()`d but not (yet) garbage-collected. In a test
    session that constructs many `MainWindow`s against one shared,
    session-scoped `QApplication`, calling this on every construction made
    that repeated, unnecessary re-polish (of a stylesheet that never
    actually changes) the dominant cost -- measured growing from ~0.15s to
    several seconds per `MainWindow()` over just a few dozen instances.
    Since the stylesheet content is static, applying it more than once has
    no observable effect to guard against.
    """
    if app.property(_THEME_APPLIED_PROPERTY):
        return
    app.setStyleSheet(build_stylesheet())
    app.setProperty(_THEME_APPLIED_PROPERTY, True)
