"""Centralized QSS styling for Gnovi Studio.

Restrained, modern scientific-desktop look using PySide6/QSS only -- no
additional theme dependency. Deliberately avoids hard-coded font families so
each platform's native font is used.

Gnovi Studio's own chrome (menus, toolbars, sidebars, bottom panel, data
table, dialogs, status bar) is always styled with the single light palette
below -- it is never user-switchable. This is a deliberate, standing design
decision, not an oversight: the chrome stays one stable, neutral/light
professional surface regardless of which `PlotTheme` (Light/Dark) is active
on the Matplotlib preview canvas inside the Workbench (see
`plotting.backends.matplotlib_backend.render_figure`'s `dark_mode` flag) --
polishing the chrome (see "Surface hierarchy" below) means making sure it
reads cleanly and calmly next to *either* Plot Theme, never adding a second,
independent dark chrome palette. `PlotTheme` itself is declarative
`GnoviFigure` state (see `plotting.figure.PlotTheme`'s docstring for why),
re-exported here only so existing `from gnovi_plot.gui.styles import
PlotTheme` call sites keep working.

Surface hierarchy
------------------
Rather than styling widgets ad hoc, the chrome is built from a small stack
of tonal "surfaces" (each just a background + border pairing in
`_LIGHT_PALETTE`), applied consistently instead of one-off per-widget
colors:

    Surface 0 -- `bg_app`            main application background (behind
                                      everything; only visible in splitter
                                      gaps/margins).
    Surface 1 -- `bg_recessed`       drawers, bottom panel, dialogs: a step
                                      up from Surface 0, a supporting region
                                      rather than the app's own backdrop.
    Surface 2 -- `bg_raised_top`/    toolbars, section/tab headers,
                 `bg_raised_bottom`  Workbench chrome: a subtle top-to-
                                      bottom gradient between these two
                                      stops gives a gently "raised" strip
                                      without a real drop shadow (QSS has
                                      none).
    Surface 3 -- `bg_control`        buttons, inputs, lists, tables: the
                                      lightest, most "interactive" surface.
    Accent     -- `accent`*          selected/checked/engaged state,
                                      always accent-derived, never bold
                                      text (see `context_accent` below).
    Pressed    -- `bg_control_press` a slightly inset/tucked-in state for
                                      Surface 3 controls being clicked.

`context_accent` is a separate, restrained blue-gray reserved for
current-context/status text (active panel, active dataset, current graph --
see `gui.widgets.active_panel_label.ActivePanelLabel` and
`gui.widgets.dataset_panel.DatasetPanel`): normal font weight, color instead
of bold, so "what's currently active" reads calmly rather than shouting.
Never used for real warnings/errors (`STALE_COLOR`/`WARNING_COLOR` below
stay separate, deliberately more saturated, alert colors).
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from gnovi_plot.plotting.figure import PlotTheme  # noqa: F401 -- re-exported, see module docstring

_LIGHT_PALETTE = {
    # --- Surface hierarchy (see module docstring) ---------------------------
    "bg_app": "#eef0f4",
    "bg_recessed": "#f6f7fa",
    "bg_raised_top": "#ffffff",
    "bg_raised_bottom": "#f0f2f6",
    "bg_control": "#ffffff",
    "bg_control_hover": "#ffffff",
    "bg_control_press": "#e9edf6",
    "border": "#d6dae1",
    "border_strong": "#c1c6d0",
    "text": "#20242b",
    "muted_text": "#5b6270",
    "accent": "#2f6fed",
    "accent_hover": "#255ac9",
    "accent_pressed": "#1e4aa8",
    "accent_text": "#ffffff",
    "accent_soft": "#e4ecfd",
    # Restrained current-context/status color (Active panel / Graph / Data,
    # active dataset) -- see module docstring. Deliberately calmer/less
    # saturated than `accent` (which still means "selected/interactive")
    # and never fluorescent.
    "context_accent": "#3d6ea6",
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

# GUI-only Workbench header background (see
# gui.widgets.workbench_header.WorkbenchHeader) -- application chrome, kept
# out of the Matplotlib Figure entirely (a plain Qt widget docked above the
# canvas, never a scientific artist), same reasoning as the active-panel
# badge above.
WORKBENCH_HEADER_BG = "#f6f7fa"

_STYLESHEET_TEMPLATE = """
QWidget {{
    background-color: {bg_app};
    color: {text};
    font-size: 10pt;
}}

QMainWindow {{
    background-color: {bg_app};
}}

QDialog {{
    background-color: {bg_app};
}}

QGroupBox {{
    background-color: {bg_control};
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
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {bg_control}, stop:1 {bg_recessed});
    border: 1px solid {border};
    border-radius: 5px;
    padding: 5px 12px;
    min-height: 22px;
}}

QPushButton:hover {{
    background-color: {bg_control_hover};
    border-color: {accent};
}}

QPushButton:pressed {{
    background-color: {bg_control_press};
    border-color: {border_strong};
    padding-top: 6px;
    padding-bottom: 4px;
}}

QPushButton:checked {{
    background-color: {accent_soft};
    border: 1px solid {accent};
    color: {text};
}}

QPushButton:disabled {{
    background: {bg_recessed};
    color: {muted_text};
}}

QPushButton[primary="true"] {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {accent_hover}, stop:1 {accent});
    border-color: {accent_pressed};
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
    padding-top: 6px;
    padding-bottom: 4px;
}}

QPushButton[primary="true"]:disabled {{
    background: {bg_recessed};
    border-color: {border};
    color: {muted_text};
}}

QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 3px;
}}

QToolButton:hover {{
    background-color: {bg_recessed};
    border-color: {border};
}}

QToolButton:pressed {{
    background-color: {bg_control_press};
}}

QToolButton:checked {{
    background-color: {accent_soft};
    border-color: {accent};
}}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {bg_control};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 22px;
}}

QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {border_strong};
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
    background-color: {bg_control};
    border: 1px solid {border};
    border-radius: 4px;
    alternate-background-color: {bg_recessed};
    gridline-color: {border};
}}

QListWidget::item {{
    padding: 4px 6px 4px 8px;
    border-radius: 3px;
    border-left: 3px solid transparent;
}}

QListWidget::item:hover {{
    background-color: {bg_recessed};
}}

QListWidget::item:selected {{
    background-color: {accent_soft};
    color: {text};
    border-left: 3px solid {accent};
}}

QTableView::item:selected {{
    background-color: {accent_soft};
    color: {text};
}}

QHeaderView::section {{
    background-color: {bg_recessed};
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
    background-color: transparent;
    color: {text};
}}

/* Left/right DSO-style tool-strip -- Surface 2 (Workbench-adjacent chrome):
plain objectName selectors rather than a `[side="..."]` dynamic property
(see gui.widgets.tool_drawer.ToolDrawer's own comment on why -- a full
QApplication::setStyleSheet re-polish evaluates every stylesheet rule
against every widget, and property selectors cost meaningfully more than a
plain objectName match at that scale). */
QWidget#ToolStripLeft, QWidget#ToolStripRight {{
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0, stop:0 {bg_raised_top}, stop:1 {bg_raised_bottom}
    );
}}

QWidget#ToolStripLeft {{
    border-right: 1px solid {border};
}}

QWidget#ToolStripRight {{
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
    background-color: {bg_recessed};
    border-color: {border};
    color: {text};
}}

QToolButton#ToolStripButtonLeft:pressed, QToolButton#ToolStripButtonRight:pressed {{
    background-color: {bg_control_press};
}}

/* Checked/engaged tool-strip button: a restrained accent edge + soft fill
+ accent-colored label -- never bold (see module docstring's
`context_accent` note and the general "no bold for current-state" rule
followed throughout the chrome). */
QToolButton#ToolStripButtonLeft:checked {{
    background-color: {accent_soft};
    color: {accent_pressed};
    border-left: 3px solid {accent};
}}

QToolButton#ToolStripButtonRight:checked {{
    background-color: {accent_soft};
    color: {accent_pressed};
    border-right: 3px solid {accent};
}}

/* Context/status text (Active panel / Graph / Data, active-dataset
context) -- restrained accent color at normal weight instead of bold; see
module docstring. Applied via a dynamic property rather than a widget
subclass/objectName so any plain QLabel can opt in. */
QLabel[contextRow="true"] {{
    color: {context_accent};
    font-weight: 400;
}}

/* Workbench header -- see gui.widgets.workbench_header.WorkbenchHeader.
Surface 2 (Workbench-adjacent chrome), GUI-only application chrome docked
above the plot canvas, never part of the Matplotlib Figure/exports. */
QWidget#WorkbenchHeader {{
    background-color: qlineargradient(
        x1:0, y1:0, x2:0, y2:1, stop:0 {bg_raised_top}, stop:1 {bg_raised_bottom}
    );
    border-bottom: 1px solid {border};
}}

QLabel#WorkbenchHeaderLabel {{
    color: {muted_text};
    font-weight: 400;
    letter-spacing: 1px;
}}

QLabel#WorkbenchHeaderLayoutLabel {{
    color: {muted_text};
    font-weight: 400;
}}

QMenuBar {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {bg_raised_top}, stop:1 {bg_raised_bottom});
    border-bottom: 1px solid {border};
}}

QMenuBar::item {{
    padding: 4px 8px;
    background: transparent;
}}

QMenuBar::item:selected {{
    background-color: {accent_soft};
    border-radius: 3px;
}}

QMenu {{
    background-color: {bg_control};
    border: 1px solid {border};
    padding: 4px;
}}

QMenu::item {{
    padding: 4px 20px 4px 12px;
    border-radius: 3px;
}}

QMenu::item:selected {{
    background-color: {accent_soft};
    color: {text};
}}

QToolBar {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {bg_raised_top}, stop:1 {bg_raised_bottom});
    border-bottom: 1px solid {border};
    spacing: 4px;
    padding: 2px;
}}

QToolBar::separator {{
    background-color: {border};
    width: 1px;
    margin: 4px 6px;
}}

QStatusBar {{
    background-color: {bg_raised_top};
    border-top: 1px solid {border};
    color: {muted_text};
}}

QTabWidget::pane {{
    background-color: {bg_control};
    border: 1px solid {border};
    border-radius: 4px;
    top: -1px;
}}

QTabBar::tab {{
    background-color: {bg_recessed};
    border: 1px solid {border};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 5px 12px;
    color: {muted_text};
}}

QTabBar::tab:hover {{
    color: {text};
}}

QTabBar::tab:selected {{
    background-color: {bg_control};
    color: {accent_pressed};
    border-bottom: 2px solid {accent};
    margin-bottom: -1px;
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
