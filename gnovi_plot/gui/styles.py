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

import tempfile
from pathlib import Path

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

# QSpinBox/QDoubleSpinBox `::up-arrow`/`::down-arrow` image files -- see
# `_ensure_spin_arrow_icon_files` (generates them) and the matching QSS
# comment above `QSpinBox::up-arrow` (why a real image file, not an inline
# QSS technique). The OS temp dir, not a path next to this source file:
# these are tiny, fully regeneratable-on-demand assets, never user data,
# and every platform guarantees a writable temp dir exists, unlike the
# install location. Forward slashes always -- Qt's own QSS `url()` parser
# expects them even in a Windows path (backslashes are escape characters
# in QSS strings).
_SPIN_ARROW_ASSET_DIR = Path(tempfile.gettempdir()) / "gnovi_studio_assets"
_SPIN_UP_ARROW_PATH = _SPIN_ARROW_ASSET_DIR / "spin_up_arrow.png"
_SPIN_DOWN_ARROW_PATH = _SPIN_ARROW_ASSET_DIR / "spin_down_arrow.png"
_SPIN_UP_ARROW_DISABLED_PATH = _SPIN_ARROW_ASSET_DIR / "spin_up_arrow_disabled.png"
_SPIN_DOWN_ARROW_DISABLED_PATH = _SPIN_ARROW_ASSET_DIR / "spin_down_arrow_disabled.png"


def _ensure_spin_arrow_icon_files() -> None:
    """Generate the four small triangle PNGs referenced by the
    `QSpinBox::up-arrow`/`::down-arrow` QSS rules below, if they don't
    already exist on disk. Idempotent and cheap to call repeatedly (an
    `exists()` check short-circuits everything after the first call).

    Needs a `QApplication` (`QPixmap`/`QPainter` do) -- called only from
    `apply_app_theme`, which already requires one, never from
    `build_stylesheet` itself. `build_stylesheet` must stay safe to call
    before any `QApplication` exists (some tests call it directly, without
    a `qapp` fixture, to check the QSS text alone) -- it only ever
    formats the fixed path strings above into the template, regardless of
    whether the files behind them exist yet."""
    if (
        _SPIN_UP_ARROW_PATH.exists()
        and _SPIN_DOWN_ARROW_PATH.exists()
        and _SPIN_UP_ARROW_DISABLED_PATH.exists()
        and _SPIN_DOWN_ARROW_DISABLED_PATH.exists()
    ):
        return

    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QPainter, QPixmap, QPolygonF

    _SPIN_ARROW_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    up_points = [(2.0, 6.0), (10.0, 6.0), (6.0, 2.0)]
    down_points = [(2.0, 2.0), (10.0, 2.0), (6.0, 6.0)]
    for path, points, color_hex in (
        (_SPIN_UP_ARROW_PATH, up_points, _LIGHT_PALETTE["muted_text"]),
        (_SPIN_DOWN_ARROW_PATH, down_points, _LIGHT_PALETTE["muted_text"]),
        (_SPIN_UP_ARROW_DISABLED_PATH, up_points, _LIGHT_PALETTE["border"]),
        (_SPIN_DOWN_ARROW_DISABLED_PATH, down_points, _LIGHT_PALETTE["border"]),
    ):
        pixmap = QPixmap(12, 8)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color_hex))
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
        painter.end()
        pixmap.save(str(path), "PNG")

_STYLESHEET_TEMPLATE = """
/* 11pt, not the app's original 10pt -- a restrained +1pt readability pass
across all application chrome (menus/toolbars/tabs/dialogs/panels cascade
from this one rule). Deliberately NOT +2pt/a larger visual footprint, and
entirely separate from the ToolStrip's own small caption label (kept at its
existing 8pt below -- it sits in a fixed 64px-wide strip where growing the
label risks clipping) and from Matplotlib figure typography (Base/Title/
Axis/Tick/Legend size in the Figure drawer), which is user-controlled
scientific-figure state rendered by a completely separate pipeline, never
touched by this QSS. */
QWidget {{
    background-color: {bg_app};
    color: {text};
    font-size: 11pt;
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

/* Neutral lift, not accent -- accent is reserved for FOCUS/CHECKED so each
state reads as a distinct signal rather than three shades of the same blue
border (see the state-system note in the module docstring). */
QPushButton:hover {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {bg_raised_top}, stop:1 {bg_control});
    border-color: {border_strong};
}}

/* Flattened (no gradient) + a 1px content shift -- reads as physically
depressed rather than just recolored. */
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
    border-color: {border};
    color: {muted_text};
}}

/* Keyboard-focus cue, deliberately separate from :hover -- an accent
border with no fill/lift change, so tabbing through a form doesn't look
like the mouse is hovering everything. */
QPushButton:focus {{
    border-color: {accent};
}}

/* Lighter stop on top, darker on bottom -- same raised direction as the
plain QPushButton gradient above (light source from above); the previous
build had these two stops reversed, which read as faintly inset even at
rest. */
QPushButton[primary="true"] {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {accent}, stop:1 {accent_hover});
    border-color: {accent_pressed};
    color: {accent_text};
    font-weight: 600;
}}

/* Flattens to the gradient's lighter (top) stop -- brighter than resting
state, never the darker `accent_hover` alone. */
QPushButton[primary="true"]:hover {{
    background-color: {accent};
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

QPushButton[primary="true"]:focus {{
    border-color: {accent_pressed};
}}

/* Scoped to the app's own "Main" toolbar (Undo/Redo/Import Data/Save
Working Data/Export Figure -- see `MainWindow._create_toolbar`'s
`setObjectName("MainToolBar")`) -- deliberately NOT the bare `QToolButton`
selector. The Matplotlib navigation toolbar (Home/Pan/Zoom/.../Save) is
also built from un-decorated QToolButtons; painting an opaque raised
background on every QToolButton made those icons unreadable, so the tactile
chrome below is scoped to stay off Matplotlib's own toolbar entirely. Every
other QToolButton in the app (ToolStrip, WorkbenchNewButton, the
collapsible-section toggle) carries its own objectName/property selector
and so was never affected either way. */
QToolBar#MainToolBar QToolButton {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {bg_control}, stop:1 {bg_recessed});
    border: 1px solid {border};
    border-radius: 4px;
    padding: 4px;
}}

QToolBar#MainToolBar QToolButton:hover {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {bg_raised_top}, stop:1 {bg_control});
    border-color: {border_strong};
}}

QToolBar#MainToolBar QToolButton:pressed {{
    background-color: {bg_control_press};
    border-color: {border_strong};
    padding-top: 5px;
    padding-bottom: 3px;
}}

QToolBar#MainToolBar QToolButton:checked {{
    background-color: {accent_soft};
    border-color: {accent};
    color: {accent_pressed};
}}

QToolBar#MainToolBar QToolButton:disabled {{
    background-color: {bg_recessed};
    border-color: {border};
    color: {muted_text};
}}

QToolBar#MainToolBar QToolButton:focus {{
    border-color: {accent};
}}

/* Same two tones as the raised controls above, gradient direction reversed
(darker stop on top) -- a restrained "recessed field" cue that costs no new
palette tokens, less depth than a push button rather than a heavier bevel.
One shared, slightly taller `min-height` across all four types keeps
QSpinBox/QDoubleSpinBox visually matched to their neighboring QComboBox/
QLineEdit. None of the four are width-capped: every ordinary left-drawer
field (QLineEdit/QComboBox/QSpinBox/QDoubleSpinBox alike) expands to fill
its row via Qt's own default QFormLayout field-growth policy plus each
widget's own default (non-Fixed) horizontal size policy -- deliberately not
fought with a `max-width` here. A couple of specific numeric fields whose
*configured range* (not this shared rule) inflated their natural minimum
width still get an explicit, narrower `setMinimumWidth` at their own
construction site -- see `figure_properties_panel._make_limit_spin` and
`plot_series_panel`'s `offset_spin`/`offset_step_spin`. */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {bg_recessed}, stop:1 {bg_control});
    border: 1px solid {border};
    border-radius: 4px;
    padding: 4px 6px;
    min-height: 26px;
}}

/* QComboBox alone drops back to a FLAT background -- confirmed by direct
runtime reproduction (not guesswork) that a `qlineargradient` background on
QComboBox itself, specifically, breaks its native popup's item-highlight
rendering on this Qt6/Fusion/Wayland stack: the hovered/current row painted
solid black instead of the palette's Highlight color, regardless of any
QSS given to the popup view itself. QLineEdit/QSpinBox/QDoubleSpinBox have
no popup and keep the gradient above unaffected; only this one rule (later
in the cascade, so it wins over the shared gradient for QComboBox) exists
to route around the bug. */
QComboBox {{
    background-color: {bg_recessed};
}}

QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
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

/* QSpinBox/QDoubleSpinBox up/down sub-controls -- unstyled, these render as
two heavily-boxed native mini-buttons (Fusion's default), which reads as
mechanical next to the rest of the restrained chrome. The goal here is one
unified recessed field with a subtle arrow column, not two push buttons
bolted onto a text field: transparent button backgrounds (so the field's
own recessed fill shows straight through in the normal state) with only a
thin vertical rule marking where the arrow column starts, matched corner
radii so the column's outer corners still follow the field's own rounding,
and a single hairline between the two arrow halves -- deliberately the same
{border} tone as the vertical rule rather than a separate paler token (the
two already read as different weights of "subtle" purely from being a
full-height line vs. a half-width one, without needing a second color).
Hover/pressed are scoped per half (`::up-button`/`::down-button`
individually) so only the half under the cursor lights up, reusing the
exact same hover/pressed tones as every other recessed field elsewhere in
this stylesheet -- no new palette entries. */
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
    border-left: 1px solid {border};
    border-bottom: 1px solid {border};
    border-top-right-radius: 4px;
    background: transparent;
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
    border-left: 1px solid {border};
    border-bottom-right-radius: 4px;
    background: transparent;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {bg_recessed};
}}

/* Reuses the same "flatten to a darker fill" pressed language as every
other control in this stylesheet (QPushButton/QToolButton/ToolStrip). */
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
    background-color: {bg_control_press};
}}

/* Unlike `QComboBox::down-arrow` above, the plain CSS-border-triangle
technique (a 0x0 content box with only borders, one of them colored to
form a triangle) does NOT render as a triangle for QSpinBox/
QDoubleSpinBox's `::up-arrow`/`::down-arrow` on this Qt6/Fusion build --
confirmed by direct isolated reproduction across several QSS variants
(0x0 + border, explicit width/height + border, explicit subcontrol-
origin/position on the arrow itself): every one painted a solid filled
rectangle instead, never a triangle. A real image file loaded via `url()`
is the one technique that reliably rendered correctly in that same
reproduction. See `_ensure_spin_arrow_icon_files` (called from
`apply_app_theme`, generating these tiny PNGs into the OS temp dir once)
for why this is a file and not another inline technique. */
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({spin_up_arrow_path});
    width: 10px;
    height: 6px;
}}

QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{
    image: url({spin_up_arrow_disabled_path});
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({spin_down_arrow_path});
    width: 10px;
    height: 6px;
}}

QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
    image: url({spin_down_arrow_disabled_path});
}}

/* The popup list (a plain `QAbstractItemView`/`QListView` Qt builds for
every QComboBox) previously had no rule of its own here, so its hover/
current row fell back to the platform style's own highlight color -- on
this app's real Linux/Wayland/Qt6 runtime that rendered as a solid black
strip (confirmed by direct runtime reproduction, not guesswork -- see
`tests/test_theming.py`'s runtime popup-palette test). `selection-
background-color`/`selection-color` here are load-bearing on their own:
deliberately NO `::item`/`::item:hover` sub-control rules alongside them --
adding one (even just `padding`/`border-radius`, no color at all) hands
item painting over to Qt's styled-item delegate, which does NOT reliably
pick up this container's `selection-background-color` for the hovered/
current row and repaints it black again. Explicit here once, rather than
per-QComboBox, since every combo box in the app shares one popup styling
need. */
QComboBox QAbstractItemView {{
    background-color: {bg_control};
    border: 1px solid {border};
    outline: none;
    padding: 2px;
    selection-background-color: {accent_soft};
    selection-color: {text};
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

/* Inactive tool-strip buttons now read as individual raised "keys" on the
strip's own gradient chrome, rather than bare text -- the DSO-panel feel
called for in the tool-strip's own docstring, reusing the same raised
gradient/border tokens as QPushButton/QToolButton above (no one-off
colours). */
QToolButton#ToolStripButtonLeft, QToolButton#ToolStripButtonRight {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {bg_control}, stop:1 {bg_recessed});
    border: 1px solid {border};
    border-radius: 6px;
    padding: 6px 2px;
    color: {muted_text};
    font-size: 8pt;
}}

QToolButton#ToolStripButtonLeft:hover, QToolButton#ToolStripButtonRight:hover {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {bg_raised_top}, stop:1 {bg_control});
    border-color: {border_strong};
    color: {text};
}}

QToolButton#ToolStripButtonLeft:pressed, QToolButton#ToolStripButtonRight:pressed {{
    background-color: {bg_control_press};
    border-color: {border_strong};
    padding-top: 7px;
    padding-bottom: 5px;
}}

/* Checked/engaged tool-strip button: a restrained accent edge + soft fill
+ accent-colored label -- never bold (see module docstring's
`context_accent` note and the general "no bold for current-state" rule
followed throughout the chrome). Distinct from :pressed above (soft accent
fill + edge stripe that persists, vs. a transient darker flatten) so the
currently-open tool reads as latched rather than just "recently clicked". */
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

QToolButton#ToolStripButtonLeft:disabled, QToolButton#ToolStripButtonRight:disabled {{
    background-color: {bg_recessed};
    border-color: {border};
    color: {muted_text};
}}

QToolButton#ToolStripButtonLeft:focus, QToolButton#ToolStripButtonRight:focus {{
    border-color: {accent};
}}

/* Context/status text (Active panel / Graph / Data, active-dataset
context) -- restrained accent color at normal weight instead of bold; see
module docstring. Applied via a dynamic property rather than a widget
subclass/objectName so any plain QLabel can opt in. Also its own light
Surface-3 "card" (background/border/padding) rather than bare text sitting
directly on the drawer's Surface-1 background -- QLabel supports the full
box model directly via QSS, so no wrapper container widget is needed at
any of this label's five call sites (Plot/Series/Figure/Layout/Axes
drawers). No heavy border (the same restrained `border` token as every
other control), modest radius, and generous padding for breathing room;
`ActivePanelLabel` already leaves height unset and wraps long dataset
lists, so the card still grows with its content rather than clipping or
scrolling. */
QLabel[contextRow="true"] {{
    background-color: {bg_control};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 8px 10px;
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
}}

/* Workbench tab strip -- see gui.widgets.workbench_tabs.WorkbenchTabBar.
Application/workspace navigation, deliberately distinct from the bottom
panel's generic QTabBar (Graphs/Data/Transformations, styled above): an
underline-tab look rather than a pill-tab look, sitting on the same
Surface 2 raised band as the toolbar/Workbench header directly below it. */
QWidget#WorkbenchTabStrip {{
    background-color: qlineargradient(
        x1:0, y1:0, x2:0, y2:1, stop:0 {bg_raised_top}, stop:1 {bg_raised_bottom}
    );
    border-bottom: 1px solid {border};
}}

QTabBar#WorkbenchTabBar::tab {{
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 6px 14px;
    margin-right: 2px;
    color: {muted_text};
    max-width: 180px;
}}

QTabBar#WorkbenchTabBar::tab:hover {{
    background-color: {bg_recessed};
    color: {text};
}}

QTabBar#WorkbenchTabBar::tab:selected {{
    color: {accent_pressed};
    border-bottom: 2px solid {accent};
}}

QToolButton#WorkbenchNewButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    color: {muted_text};
    font-weight: 600;
    min-width: 22px;
    min-height: 22px;
    margin: 2px 4px;
}}

QToolButton#WorkbenchNewButton:hover {{
    background-color: {bg_recessed};
    border-color: {accent};
    color: {accent};
}}

QToolButton#WorkbenchNewButton:pressed {{
    background-color: {bg_control_press};
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
    background-color: {bg_raised_bottom};
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
    light palette by default, or a custom dict with the same keys). The
    four `spin_*_arrow*_path` values are supplied separately from
    `palette` (paths, not colors) so a caller's custom palette dict never
    needs those keys too -- see `_ensure_spin_arrow_icon_files` for why
    these particular QSS rules reference image files at all. Safe to call
    before those files exist (or before any `QApplication` exists, which
    generating them requires) -- this only formats the path strings into
    the template text; nothing here touches the filesystem or Qt."""
    return _STYLESHEET_TEMPLATE.format(
        spin_up_arrow_path=_SPIN_UP_ARROW_PATH.as_posix(),
        spin_down_arrow_path=_SPIN_DOWN_ARROW_PATH.as_posix(),
        spin_up_arrow_disabled_path=_SPIN_UP_ARROW_DISABLED_PATH.as_posix(),
        spin_down_arrow_disabled_path=_SPIN_DOWN_ARROW_DISABLED_PATH.as_posix(),
        **palette,
    )


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
    _ensure_spin_arrow_icon_files()
    app.setStyleSheet(build_stylesheet())
    app.setProperty(_THEME_APPLIED_PROPERTY, True)
