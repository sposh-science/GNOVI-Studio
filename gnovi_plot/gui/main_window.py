import math
from pathlib import Path

from matplotlib.backends import backend_qt
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QPointF, QRect, QRectF, QSettings, Qt
from PySide6.QtGui import (
    QActionGroup,
    QColor,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QTableView,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.analysis.segments import InvalidRowRangeError, contiguous_row_range
from gnovi_plot.core.app_info import APP_NAME, about_text
from gnovi_plot.core.project import Project
from gnovi_plot.core.project_io import ProjectIOError, load_project, save_project
from gnovi_plot.core.workbench import Workbench
from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_xy
from gnovi_plot.gui.dialogs.export_figure_dialog import ExportFigureDialog
from gnovi_plot.gui.styles import PlotTheme, apply_app_theme
from gnovi_plot.gui.undo_manager import UndoManager, snapshot_figure
from gnovi_plot.gui.widgets.active_panel_label import ActivePanelLabel
from gnovi_plot.gui.widgets.analysis_panel import AnalysisPanel
from gnovi_plot.gui.widgets.analysis_result_view import AnalysisResultView
from gnovi_plot.gui.widgets.bottom_panel import BottomPanel
from gnovi_plot.gui.widgets.data_tools_panel import DataToolsPanel
from gnovi_plot.gui.widgets.dataframe_table_model import DataFrameTableModel
from gnovi_plot.gui.widgets.dataset_panel import DatasetPanel
from gnovi_plot.gui.widgets.figure_layout_panel import FigureLayoutPanel
from gnovi_plot.gui.widgets.figure_properties_panel import FigurePropertiesPanel
from gnovi_plot.gui.widgets.figure_size_panel import LAYOUT_PRESETS, FigureSizePanel
from gnovi_plot.gui.widgets.graph_library_panel import GraphLibraryPanel
from gnovi_plot.gui.widgets.plot_canvas import PlotCanvas, ReferenceCursorMode
from gnovi_plot.gui.widgets.plot_series_panel import PlotSeriesPanel
from gnovi_plot.gui.widgets.tool_drawer import ToolDrawer
from gnovi_plot.gui.widgets.workbench_header import WorkbenchHeader
from gnovi_plot.gui.widgets.workbench_tabs import WorkbenchTabBar
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries

# Wide enough for "x = -0000.0000, y = -0000.0000" so the status-bar
# coordinate readout never changes width as digits change while the mouse
# moves over the plot -- nothing else in the toolbar/status bar shifts.
_COORD_LABEL_SAMPLE_TEXT = "x = -0000.0000, y = -0000.0000"

_UNDO_REDO_ICON_SIZE = 24
_UNDO_REDO_ICON_COLOR = "#20242b"  # matches styles._LIGHT_PALETTE["text"]; see _make_undo_redo_icon


def _arrowhead_polygon(tip: QPointF, direction: QPointF, size: float) -> QPolygonF:
    """A small filled triangle at `tip`, pointing along `direction`."""
    length = math.hypot(direction.x(), direction.y()) or 1.0
    dx, dy = direction.x() / length, direction.y() / length
    px, py = -dy, dx
    back = QPointF(tip.x() - dx * size, tip.y() - dy * size)
    left = QPointF(back.x() + px * size * 0.65, back.y() + py * size * 0.65)
    right = QPointF(back.x() - px * size * 0.65, back.y() - py * size * 0.65)
    return QPolygonF([tip, left, right])


def _make_undo_redo_icon(direction: str) -> QIcon:
    """A "reply-arrow" hook glyph -- a short vertical stub, one smooth
    quarter-circle turn, then a long horizontal shaft ending in a bold
    arrowhead -- drawn in-process via QPainter, consistent with the app's
    existing icon system (see `gui.widgets.tool_drawer._make_icon`, the
    same hand-drawn-glyph technique), never dependent on an OS/desktop icon
    theme being present (`QIcon.fromTheme` would silently return a null
    icon on many Windows/macOS/minimal-Linux setups).

    Deliberately NOT a segment of one constant-radius circle (an earlier
    build was, at various spans) -- a full arc's silhouette reads as
    "circular motion" (Refresh/Reload) regardless of how much of the circle
    is missing, especially once anti-aliased down to toolbar size. Here the
    horizontal shaft -- most of the glyph's length -- is a straight line, so
    the shape reads as a directional arrow first; the quarter-circle turn
    is only a small connecting flourish, not the dominant silhouette.

    Redo is undo's exact mirror image: every x-coordinate is reflected
    about the icon's vertical center (`size - x`), and the quarter-turn's
    sweep direction is reversed to match (`180, -90` instead of `0, 90`) --
    reflection flips handedness, so simply relabeling coordinates without
    also flipping the sweep would draw the turn on the wrong side."""
    size = _UNDO_REDO_ICON_SIZE
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(_UNDO_REDO_ICON_COLOR))
    pen.setWidthF(size * 0.11)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

    mirror = direction == "redo"
    margin = size * 0.12
    radius = size * 0.22
    stub_length = size * 0.16
    shaft_y = size * 0.33
    undo_stub_x = size - margin - 2  # undo's stub sits near the right edge
    undo_arc_center_x = undo_stub_x - radius
    arc_center_y = shaft_y + radius

    def mirror_x(x: float) -> float:
        return size - x if mirror else x

    arc_center_x = mirror_x(undo_arc_center_x)
    # Undo's arc entry (tangent vertical, where the stub attaches) is the
    # circle's east point; mirroring flips it to the west point instead.
    arc_entry_x = arc_center_x - radius if mirror else arc_center_x + radius
    stub_bottom = QPointF(mirror_x(undo_stub_x), arc_center_y + stub_length)
    stub_top = QPointF(arc_entry_x, arc_center_y)
    shaft_tip = QPointF(mirror_x(margin), shaft_y)  # arc exit -> shaft -> arrowhead tip

    path = QPainterPath()
    path.moveTo(stub_bottom)
    path.lineTo(stub_top)
    arc_rect = QRectF(arc_center_x - radius, arc_center_y - radius, 2 * radius, 2 * radius)
    if not mirror:
        path.arcTo(arc_rect, 0, 90)  # east -> north, tangent vertical -> horizontal
    else:
        path.arcTo(arc_rect, 180, -90)  # mirrored: west -> north, opposite sweep direction
    path.lineTo(shaft_tip)
    painter.setPen(pen)
    painter.strokePath(path, pen)

    arrow_direction = QPointF(1, 0) if mirror else QPointF(-1, 0)  # shaft always ends pointing horizontally
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(_UNDO_REDO_ICON_COLOR))
    painter.drawPolygon(_arrowhead_polygon(shaft_tip, arrow_direction, size * 0.22))

    painter.end()
    return QIcon(pixmap)

_PLOT_THEME_MENU_LABELS = ((PlotTheme.LIGHT, "Light"), (PlotTheme.DARK, "Dark"))

_REFERENCE_CURSOR_MENU_LABELS = (
    (ReferenceCursorMode.OFF, "Off"),
    (ReferenceCursorMode.X_LINE, "X line"),
    (ReferenceCursorMode.Y_LINE, "Y line"),
    (ReferenceCursorMode.CROSSHAIR, "Crosshair"),
)

# The View menu's "Reference Cursor" submenu title already gives the short
# labels above their context; the toolbar combo has no such title next to
# it, so its own items spell out "Cursor: " explicitly -- otherwise a lone
# "Off" in the toolbar reads as ambiguous (which "Off"?).
_REFERENCE_CURSOR_TOOLBAR_LABELS = (
    (ReferenceCursorMode.OFF, "Cursor: Off"),
    (ReferenceCursorMode.X_LINE, "Cursor: X Line"),
    (ReferenceCursorMode.Y_LINE, "Cursor: Y Line"),
    (ReferenceCursorMode.CROSSHAIR, "Cursor: Crosshair"),
)

# Fraction of the screen's available geometry the main window occupies at
# startup. Centered rather than maximized, and always derived from the
# actual screen -- never a fixed resolution.
_STARTUP_SCREEN_FRACTION = 0.92

_FALLBACK_AVAILABLE_GEOMETRY = QRect(0, 0, 1280, 800)

# Extra width reserved beyond a side drawer's widest page's own
# minimumSizeHint when computing that drawer's default expanded width: the
# vertical scrollbar (see QScrollBar:vertical width in gui.styles) plus a
# small comfortable margin so controls never sit flush against it. Shared by
# both the left (tool_drawer) and right (working_drawer) side -- see
# `_side_drawer_min_width`.
_DRAWER_SCROLLBAR_RESERVE = 12
_DRAWER_CONTENT_MARGIN = 12

# Default expanded width for each side drawer, as a fraction of the window's
# own width -- only a floor/starting point, never a hard size: each drawer's
# actual minimum is raised to whatever its widest page's content genuinely
# needs (see `_side_drawer_min_width`), and the user can always drag the
# splitter wider afterward. Deliberately different per side: the right
# Working Data drawer holds far less content than the left Data/Plot/
# Series/Figure/Layout/Axes drawer, so it needs a smaller share by default.
_LEFT_DRAWER_WIDTH_FRACTION = 0.19
_RIGHT_DRAWER_WIDTH_FRACTION = 0.16

# The narrowest the center Workbench is ever allowed to shrink to in
# `compute_drawer_widths`, even on a screen too small to also fit both
# drawers at their preferred widths -- letting it shrink further toward 0
# doesn't just look cramped, it makes Matplotlib's own coordinate
# transforms singular against a literally-zero-size canvas. Deliberately
# NOT enforced via a hard `QWidget.setMinimumSize` on the window itself:
# `MainWindow` must never demand a size larger than the actual screen (see
# `compute_initial_geometry`/`test_main_window_startup_geometry_never_
# exceeds_screen`), which a real small-screen user could violate -- so this
# is instead a soft priority `compute_drawer_widths` applies to whatever
# total width it's actually given, shrinking the two side drawers (below
# their own comfortable widths, though never asked to go below the hard
# content floor a caller passed in) rather than the center.
_MIN_WORKBENCH_WIDTH = 360


def _side_drawer_min_width(drawer, content_widgets) -> int:
    """The narrowest `drawer` (a `ToolDrawer`) can go without clipping any
    of `content_widgets` (the un-wrapped widgets placed into its pages,
    before `_wrap_scrollable`) -- their own `minimumSizeHint`, which no
    scroll area can shrink below without cutting content off, plus the
    strip and a small reserve for the scrollbar/margin. Used to raise each
    side drawer's screen-fraction default up to a real floor when
    necessary; see the two `_*_DRAWER_WIDTH_FRACTION` constants above."""
    widest_content = max(widget.minimumSizeHint().width() for widget in content_widgets)
    return drawer.strip_width + widest_content + _DRAWER_SCROLLBAR_RESERVE + _DRAWER_CONTENT_MARGIN


def compute_drawer_widths(
    total_width: int,
    left_min_width: int,
    right_min_width: int,
    *,
    left_floor_width: int = 0,
    right_floor_width: int = 0,
    left_locked_collapsed: bool = False,
    right_locked_collapsed: bool = False,
    collapse_priority: str = "right",
) -> tuple[int, int, int, bool, bool]:
    """Return `(left_width, center_width, right_width, auto_collapse_left,
    auto_collapse_right)` for the main horizontal splitter, given the
    window's total content width and each side drawer's own true minimum
    (`left_min_width`/`right_min_width`, from `_side_drawer_min_width` --
    the width below which that drawer's widest page actually clips) and
    *preferred* width (computed here as a fraction of `total_width`, never
    below the true minimum).

    `left_floor_width`/`right_floor_width` are each drawer's *collapsed*
    width (its `strip_width`) -- the only width a drawer is ever given
    below its own true minimum, and only once it's actually collapsed
    (page hidden), never while still showing content. `left_min_width`/
    `right_min_width` reflect the *platform's* text metrics for whatever's
    on that drawer's widest page (see `_side_drawer_min_width`) -- not just
    the window's own width, so two platforms can legitimately disagree on
    what that minimum is for pixel-identical content.

    `left_locked_collapsed`/`right_locked_collapsed` mark a side as already
    collapsed for a reason this function doesn't get a vote on (typically
    the user explicitly collapsed it) -- that side is pinned at its own
    `floor_width` throughout and is never a candidate for auto-collapse
    (it already is collapsed) or for growing back to fill spare room (see
    `MainWindow._reflow_side_drawers`'s docstring on why fighting a manual
    choice like that would be unwelcome).

    Priority under width pressure, in order:
      1. Both non-locked sides at their preferred width, if the Workbench
         still gets `_MIN_WORKBENCH_WIDTH` at that size.
      2. Reclaim comfort slack -- shrink each non-locked side back toward
         (never below) its own true minimum.
      3. If the Workbench still doesn't have a sensible minimum even with
         both non-locked sides already at their true minimum, auto-collapse
         one side (`collapse_priority` first) rather than shrinking either
         below what its content needs.
      4. If collapsing one side still isn't enough, auto-collapse the
         other too.
    A drawer is *never* left expanded with less than its own true minimum
    -- it's either given that minimum (or more) or it's collapsed; an
    over-shrunk-but-still-"open" drawer clips its content
    (`_wrap_scrollable`'s horizontal scrollbar is deliberately off), which
    this priority order exists specifically to avoid.

    A pure function (like `compute_initial_geometry` above) so it can be
    exercised directly at specific window widths/states in tests,
    independent of whatever screen size the test's own Qt platform plugin
    happens to report. Used both at construction and, via
    `MainWindow.resizeEvent`/`_reflow_side_drawers`, on every subsequent
    resize -- `QSplitter`'s own default behavior (proportionally rescaling
    whatever sizes were last set) has no notion of either drawer's real
    content floor, so relying on it alone across a resize can shrink an
    expanded drawer below what its page actually needs and clip it."""
    if collapse_priority not in ("left", "right"):
        raise ValueError(f"collapse_priority must be 'left' or 'right', got {collapse_priority!r}")

    # Stage A: preferred widths -- a locked-collapsed side is pinned at its
    # floor (never grows to fill spare room; see the docstring above).
    left_width = left_floor_width if left_locked_collapsed else max(
        int(total_width * _LEFT_DRAWER_WIDTH_FRACTION), left_min_width
    )
    right_width = right_floor_width if right_locked_collapsed else max(
        int(total_width * _RIGHT_DRAWER_WIDTH_FRACTION), right_min_width
    )
    center_width = total_width - left_width - right_width
    if center_width >= _MIN_WORKBENCH_WIDTH:
        return left_width, center_width, right_width, False, False

    # Stage B: reclaim comfort slack -- shrink each non-locked side toward,
    # never below, its own true minimum.
    deficit = _MIN_WORKBENCH_WIDTH - center_width
    left_slack = 0 if left_locked_collapsed else left_width - left_min_width
    right_slack = 0 if right_locked_collapsed else right_width - right_min_width
    slack = left_slack + right_slack
    if slack > 0:
        reclaimed = min(deficit, slack)
        if left_slack:
            left_width -= round(reclaimed * left_slack / slack)
        if right_slack:
            right_width -= round(reclaimed * right_slack / slack)
    center_width = total_width - left_width - right_width
    if center_width >= _MIN_WORKBENCH_WIDTH:
        return left_width, center_width, right_width, False, False

    # Stage C: both non-locked sides are now already at their own true
    # minimum -- shrinking either further would clip its content. Collapse
    # one side instead: `collapse_priority`'s side unconditionally, unless
    # it's already collapsed (user-locked), in which case there's nothing
    # further to gain there and the *other* side is tried instead. The
    # remaining (still-open) side stays exactly at the true minimum/lock
    # width Stage B already left it at -- the freed width goes entirely to
    # the Workbench, not to growing the remaining drawer.
    collapsed_side: str | None = None
    for side in (collapse_priority, "left" if collapse_priority == "right" else "right"):
        if side == "right" and not right_locked_collapsed:
            right_width = right_floor_width
            collapsed_side = "right"
            break
        if side == "left" and not left_locked_collapsed:
            left_width = left_floor_width
            collapsed_side = "left"
            break
    center_width = total_width - left_width - right_width
    if collapsed_side is not None and center_width >= _MIN_WORKBENCH_WIDTH:
        return left_width, center_width, right_width, collapsed_side == "left", collapsed_side == "right"

    # Stage D: collapsing the one available side still isn't enough --
    # collapse the other one too, if it's available (unconditionally
    # accepted as the best remaining option even if the Workbench still
    # doesn't reach its full minimum at this window width --
    # `MainWindow.center_splitter`'s own hard minimum-width is the final
    # backstop below this).
    if collapsed_side == "right" and not left_locked_collapsed:
        left_width = left_floor_width
        center_width = total_width - left_width - right_width
        return left_width, max(center_width, 0), right_width, True, True
    if collapsed_side == "left" and not right_locked_collapsed:
        right_width = right_floor_width
        center_width = total_width - left_width - right_width
        return left_width, max(center_width, 0), right_width, True, True

    # Nothing further collapsible (whatever's still open is already
    # locked, or was the only side available and didn't reach the target
    # alone) -- return the best true-minimum-respecting widths found.
    return left_width, max(center_width, 0), right_width, collapsed_side == "left", collapsed_side == "right"


def compute_initial_geometry(available: QRect, fraction: float = _STARTUP_SCREEN_FRACTION) -> QRect:
    """Return a geometry centered within `available`, scaled by `fraction`.

    Always fits inside `available` for any fraction in (0, 1], regardless
    of the screen's actual resolution.
    """
    fraction = min(max(fraction, 0.1), 1.0)
    width = max(1, int(available.width() * fraction))
    height = max(1, int(available.height() * fraction))
    x = available.x() + (available.width() - width) // 2
    y = available.y() + (available.height() - height) // 2
    return QRect(x, y, width, height)


def _wrap_scrollable(content) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setWidget(content)
    return scroll


class _CursorSafeNavigationToolbar(NavigationToolbar2QT):
    """The Matplotlib navigation toolbar's own "Save" button calls
    `self.canvas.figure.savefig(...)` directly on the live Figure --
    correct for GNOVI's WYSIWYG export goal (see `ExportFigureDialog`'s own
    docstring), but the reference cursor is a real Matplotlib artist on
    that same Figure (see `PlotCanvas.update_reference_cursor` -- unlike
    the active-panel badge, a separate Qt widget never added to the
    Figure), so saving it as-is would otherwise export whatever crosshair/
    reference line happens to be showing. Cleared immediately before
    Matplotlib's own save dialog opens; it simply reappears on the next
    mouse move over the canvas."""

    def save_figure(self, *args):
        self.canvas.clear_reference_cursor()
        super().save_figure(*args)


def _patch_mpl_icon_engine_dark_mode_detection() -> None:
    """Matplotlib's own `NavigationToolbar2QT` (Home/Back/Forward/Pan/Zoom/
    Configure Subplots/Save) recolors its icons white-on-transparent for
    "dark mode" via a private `_IconEngine._is_dark_mode()` that reads
    `self.toolbar.palette().color(self.toolbar.backgroundRole())` at *paint
    time* (see `matplotlib.backends.backend_qt._IconEngine`) -- lazily, on
    every repaint, not once at construction.

    Once `gui.styles.apply_app_theme` puts ANY stylesheet on the
    QApplication, Qt's `QStyleSheetStyle` takes over palette resolution for
    every widget it hasn't explicitly styled (this toolbar included, since
    GNOVI deliberately never styles it -- see `_create_toolbar`'s own note
    on why), and that resolution measured black (`value() == 0`) for this
    toolbar's Button role even though it visibly paints light -- confirmed
    by direct comparison against the exact same toolbar constructed without
    `apply_app_theme` applied (color `#fcfcfc`, correctly detected as
    light). Explicitly re-`setPalette`-ing the toolbar does not stick either
    -- Qt re-derives the palette from the stylesheet cascade on every
    show()/polish. `_is_dark_mode()` misreading that as dark mode is what
    produced icons the user reported as "washed out": nearly-white
    strokes recolored for a dark background, over GNOVI's actual light
    background.

    GNOVI's application chrome is architecturally always light -- never a
    user-switchable app-wide dark mode (see `gui.styles` module docstring)
    -- so unconditionally returning `False` here is not a workaround, it is
    asserting a real, permanent invariant of this application, immune to
    whatever caused the palette misdetection above. Applied once at import
    time (this module only ever runs one `NavigationToolbar2QT` instance
    per `MainWindow`, but this is a class-level patch on Matplotlib's own
    private class, so it must not be re-applied per-instance); a
    `hasattr` guard makes it a no-op rather than a hard failure if a future
    Matplotlib version renames/removes this private class."""
    icon_engine_cls = getattr(backend_qt, "_IconEngine", None)
    if icon_engine_cls is not None and hasattr(icon_engine_cls, "_is_dark_mode"):
        icon_engine_cls._is_dark_mode = lambda self: False


_patch_mpl_icon_engine_dark_mode_detection()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)

        # Explicit IniFormat rather than the platform-native
        # registry/plist backend: simpler, predictable across Linux/
        # Windows/macOS, and honors QSettings.setPath() overrides (tests
        # redirect it away from the real user config store; see
        # tests/conftest.py's `_isolated_qsettings` fixture).
        self._settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "GnoviStudio", "GnoviStudio")
        # The application chrome itself is always the one fixed light theme
        # (see gui.styles module docstring) -- only the plot canvas below is
        # user-switchable.
        apply_app_theme(QApplication.instance())
        # Plot Theme is declarative `GnoviFigure` state now (see
        # `plotting.figure.PlotTheme`), not a cached MainWindow attribute --
        # every read goes through `self.figure_model.plot_theme` so it's
        # always correct for whichever figure is currently active,
        # including after Open/New Project swaps it. QSettings only seeds
        # the *default* theme for a brand-new figure/project (see
        # `_new_project`), never overrides a loaded project's own saved
        # theme.
        try:
            self._default_new_figure_theme = PlotTheme(
                self._settings.value("plot_theme", PlotTheme.LIGHT.value)
            )
        except ValueError:
            self._default_new_figure_theme = PlotTheme.LIGHT

        screen = QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else _FALLBACK_AVAILABLE_GEOMETRY
        geometry = compute_initial_geometry(available)
        self.setGeometry(geometry)

        # `self._project` is the source of truth for everything persisted
        # (see core.project.Project / core.project_io); `dataset_manager`/
        # `figure_model` stay as MainWindow's own attribute names (used
        # throughout this file) but are just references into it, reassigned
        # together by `_load_project_into_window` on New/Open Project and by
        # `_activate_workbench` on Workbench switch/create/duplicate/delete.
        self._project = self._new_project()
        self.dataset_manager = self._project.dataset_manager

        # Undo/Redo (figure/panels/series/styling only -- see
        # gui.undo_manager for why dataset mutations are deliberately out
        # of scope here and stay tracked only in Transformation History).
        # One independent `UndoManager` + pending-snapshot per Workbench,
        # keyed by `Workbench.id` -- see `self._undo_manager`/
        # `self._pending_undo_snapshot` properties below, which always
        # resolve against `self._current_workbench_id`. This makes "Undo
        # applied to the wrong Workbench" structurally impossible: switching
        # Workbenches (`_activate_workbench`) just repoints which dict entry
        # those properties resolve to, never mixes entries, and Workbench
        # switching itself never touches either dict's *contents* (no
        # checkpoint is committed by navigation alone).
        self._undo_managers: dict[str, UndoManager] = {}
        self._pending_snapshots: dict[str, GnoviFigure] = {}
        self._current_workbench_id = self._project.active_workbench.id
        self.figure_model = self._project.active_workbench.figure
        self._dirty = False

        # `_pending_undo_snapshot` always holds a snapshot of the figure as
        # of the last committed checkpoint; see `_commit_undo_checkpoint`.
        self._undo_manager = UndoManager()
        self._pending_undo_snapshot = snapshot_figure(self.figure_model, self.dataset_manager)

        try:
            self._cursor_mode = ReferenceCursorMode(
                self._settings.value("reference_cursor", ReferenceCursorMode.OFF.value)
            )
        except ValueError:
            self._cursor_mode = ReferenceCursorMode.OFF

        self.plot_canvas = PlotCanvas(self)
        self.workbench_tab_bar = WorkbenchTabBar()
        self.workbench_header = WorkbenchHeader(self._project.active_workbench.name, self.figure_model)
        # coordinates=False: Matplotlib's built-in toolbar coordinate label
        # uses an Expanding size policy that reflows neighboring toolbar
        # content as its text width changes -- exactly the instability
        # ruled out below. The status bar's own fixed-width `coord_label`
        # (added further down) replaces it at a stable location instead.
        nav_toolbar = _CursorSafeNavigationToolbar(self.plot_canvas, self, coordinates=False)
        self.addToolBar(nav_toolbar)
        # Forces the Matplotlib toolbar and the custom "Main" toolbar (built
        # in _create_toolbar) onto separate rows unconditionally, so neither
        # ever reflows into the other at narrower widths.
        self.addToolBarBreak()

        self.coord_label = QLabel("")
        self.coord_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.coord_label.setMinimumWidth(
            self.coord_label.fontMetrics().horizontalAdvance(_COORD_LABEL_SAMPLE_TEXT)
        )
        self.statusBar().addPermanentWidget(self.coord_label)
        self.plot_canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.plot_canvas.mpl_connect("figure_leave_event", self._on_mouse_leave)
        self.plot_canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.plot_canvas.set_cursor_mode(self._cursor_mode)

        self.preview_table = QTableView()
        self.preview_model = DataFrameTableModel()
        self.preview_table.setModel(self.preview_model)
        self.preview_table.setEditTriggers(QTableView.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)

        # `lambda: self._project.graph_library` -- re-invoked on every
        # ActivePanelLabel.refresh() -- rather than a fixed reference, so
        # every page's "Graph: ..." line always resolves against the
        # *current* project's Graph Library even after Open/New Project
        # repoints `self._project` (mirrors `graph_library_panel`'s own
        # `get_figure`/`get_dataset_manager` callables below).
        get_graph_library = lambda: self._project.graph_library  # noqa: E731
        self.dataset_panel = DatasetPanel(self.dataset_manager, self.preview_table)
        self.series_panel = PlotSeriesPanel(self.figure_model, get_graph_library=get_graph_library)
        self.properties_panel = FigurePropertiesPanel(self.figure_model, get_graph_library=get_graph_library)
        self.figure_size_panel = FigureSizePanel(self.figure_model, get_graph_library=get_graph_library)
        self.figure_layout_panel = FigureLayoutPanel(self.figure_model, get_graph_library=get_graph_library)
        self.analysis_panel = AnalysisPanel(self.figure_model, self.dataset_manager)
        self.data_tools_panel = DataToolsPanel(self.preview_table)
        # `get_figure`/`get_dataset_manager` are re-invoked on every Graph
        # Library action rather than captured once, so this panel always
        # acts on the *current* project even after Open/New Project swaps
        # `self.figure_model`/`self.dataset_manager` -- see
        # `GraphLibraryPanel`'s docstring.
        self.graph_library_panel = GraphLibraryPanel(
            self._project.graph_library,
            lambda: self.figure_model,
            lambda: self.dataset_manager,
        )

        # LEFT: compact DSO-style vertical tool strip (Data / Plot / Series /
        # Figure / Layout / Axes) plus a single-page drawer next to it --
        # "what data/series do I want to plot, and how should the figure/
        # axes look?". See gui.widgets.tool_drawer.ToolDrawer. Existing panel
        # widgets are relocated into drawer pages rather than rebuilt:
        # DatasetPanel's two CollapsibleSections (Datasets, Add to Plot)
        # split across the Data and Plot pages, while PlotSeriesPanel/
        # FigureSizePanel/FigureLayoutPanel/FigurePropertiesPanel each
        # become a page whole -- their own scientific behavior/signals are
        # untouched, only relocated (they previously lived inside non-modal
        # LiveDialogs reachable from the Figure/Panels menus; those dialogs
        # are gone now that this is their one, permanently-visible home --
        # see `_open_drawer_page` below for how the menu items that used to
        # open them behave now). Only one page is ever visible at a time
        # (QStackedWidget) and clicking the active strip button collapses
        # the drawer, handing its width back to the plot canvas (see
        # `_set_side_drawer_collapsed`).
        self.dataset_panel.layout().removeWidget(self.dataset_panel.dataset_section)
        self.dataset_panel.layout().removeWidget(self.dataset_panel.plot_section)

        data_page = QWidget()
        data_page_layout = QVBoxLayout(data_page)
        data_page_layout.addWidget(self.dataset_panel.dataset_section)
        data_page_layout.addStretch(1)

        # DatasetPanel itself stays figure-agnostic (dataset/plot-column
        # concerns only, see its own docstring) -- the Plot page's "Active
        # panel" context line is composed here instead, since MainWindow is
        # what already knows about both `figure_model` and `dataset_panel`.
        self.plot_page_active_panel_label = ActivePanelLabel(self.figure_model, get_graph_library)

        plot_page = QWidget()
        plot_page_layout = QVBoxLayout(plot_page)
        plot_page_layout.addWidget(self.plot_page_active_panel_label)
        plot_page_layout.addWidget(self.dataset_panel.plot_section)
        plot_page_layout.addStretch(1)

        # Every widget that becomes a left-drawer page's content, kept so the
        # default expanded width (below) can be sized off their real
        # minimumSizeHint rather than a screen-fraction guess -- see
        # `_side_drawer_min_width`.
        left_drawer_content_widgets = [
            data_page,
            plot_page,
            self.series_panel,
            self.figure_size_panel,
            self.figure_layout_panel,
            self.properties_panel,
            self.analysis_panel,
        ]

        self.tool_drawer = ToolDrawer(side="left")
        self.tool_drawer.add_page(
            "data", "Data", "Dataset list -- import and remove datasets.", "data", _wrap_scrollable(data_page)
        )
        self.tool_drawer.add_page(
            "plot",
            "Plot",
            "Plot type, X/Y columns, plot mode and cycle controls.",
            "plot",
            _wrap_scrollable(plot_page),
        )
        self.tool_drawer.add_page(
            "series",
            "Series",
            "Plot series list and the selected series' styling.",
            "series",
            _wrap_scrollable(self.series_panel),
        )
        self.tool_drawer.add_page(
            "figure",
            "Figure",
            "Figure size/aspect ratio, publication preset, panel layout, plot theme and typography.",
            "figure",
            _wrap_scrollable(self.figure_size_panel),
        )
        self.tool_drawer.add_page(
            "layout",
            "Layout",
            "Figure margins and spacing between panels -- GNOVI Studio's route to the "
            "same controls as Matplotlib's Configure Subplots.",
            "layout",
            _wrap_scrollable(self.figure_layout_panel),
        )
        self.tool_drawer.add_page(
            "axes",
            "Axes",
            "Active panel's axis limits, ticks, spines, grid and legend.",
            "axes",
            _wrap_scrollable(self.properties_panel),
        )
        self.tool_drawer.add_page(
            "analysis",
            "Analysis",
            "Curve fitting and (as they're added) other analysis tools, run against "
            "the active panel's plotted series.",
            "analysis",
            _wrap_scrollable(self.analysis_panel),
        )
        self.tool_drawer.show_page("data")

        # RIGHT: a dedicated Working Data drawer -- "how do I derive/filter/
        # modify Working Data?", kept separate from the LEFT
        # Data/Plot/Series drawer so plotting setup and working-data
        # mutation stay conceptually (and visually) apart. Same ToolDrawer
        # architecture, mirrored (`side="right"`), with a single "Working"
        # page hosting DataToolsPanel whole -- its own signals/logic are
        # untouched, only relocated. The Transformation History list stays
        # in the bottom panel's Transformations tab (below), not here.
        self.working_drawer = ToolDrawer(side="right")
        self.working_drawer.add_page(
            "working",
            "Working",
            "Working Data actions and calculated columns.",
            "working",
            _wrap_scrollable(self.data_tools_panel),
        )
        self.working_drawer.show_page("working")

        # BOTTOM: collapsible/resizable Data / Transformations / Results /
        # Messages tabs. Results shows the most recent AnalysisResult from
        # any analysis tool (curve fitting today; nothing produces one yet
        # this milestone, so it stays in its own empty state).
        self.bottom_panel = BottomPanel()
        self.bottom_panel.set_data_widget(self.preview_table)
        self.bottom_panel.set_graphs_widget(self.graph_library_panel)
        self.bottom_panel.set_transformations_widget(self.data_tools_panel.history_group)
        self.analysis_result_view = AnalysisResultView(self.figure_model, self.dataset_manager)
        self.bottom_panel.set_results_widget(self.analysis_result_view)
        self._bottom_panel_sizes: list[int] | None = None

        # CENTER: the plot canvas stays the dominant workspace -- the
        # bottom panel starts at a modest fraction of the center column's
        # height and is fully drag-resizable/hideable without affecting the
        # figure's own configured size (on-screen canvas size never drives
        # export resolution; see plotting.backends / export.figure_export).
        # `workbench_tab_bar` (the Workbench switcher) and `workbench_header`
        # (a slim application-chrome strip naming the active Workbench) are
        # docked directly above `plot_canvas` -- wrapped together in
        # `workbench_container` purely for layout purposes; `self.plot_canvas`
        # itself is untouched (still the exact widget added to the splitter
        # in earlier milestones, just now inside one extra container), so
        # every existing `window.plot_canvas.*` call site keeps working
        # unchanged. See `gui.widgets.workbench_tabs.WorkbenchTabBar` and
        # `gui.widgets.workbench_header.WorkbenchHeader`.
        workbench_container = QWidget()
        workbench_layout = QVBoxLayout(workbench_container)
        workbench_layout.setContentsMargins(0, 0, 0, 0)
        workbench_layout.setSpacing(0)
        workbench_layout.addWidget(self.workbench_tab_bar)
        workbench_layout.addWidget(self.workbench_header)
        workbench_layout.addWidget(self.plot_canvas, 1)

        self.center_splitter = QSplitter(Qt.Vertical)
        self.center_splitter.addWidget(workbench_container)
        self.center_splitter.addWidget(self.bottom_panel)
        self.center_splitter.setStretchFactor(0, 7)
        self.center_splitter.setStretchFactor(1, 3)
        # A hard absolute floor -- unlike the two side drawers (which
        # collapse to their strip width by design, see `ToolDrawer`/
        # `_set_side_drawer_collapsed`), the Workbench has no collapsed
        # state to reconcile this with, so `QSplitter` can enforce it
        # unconditionally: on a screen too small for `compute_drawer_widths`
        # to give the Workbench its full `_MIN_WORKBENCH_WIDTH` without
        # shrinking a drawer below its own content floor (see that
        # function's own comment), this is the last-resort backstop that
        # keeps the plot canvas above a literal zero width, which Matplotlib
        # cannot render into (a singular coordinate transform), not just an
        # uncomfortably narrow one.
        self.center_splitter.setMinimumWidth(50)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self.tool_drawer)
        main_splitter.addWidget(self.center_splitter)
        main_splitter.addWidget(self.working_drawer)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)
        # Each fraction-of-window default below is only a floor -- on
        # narrower windows it can land under what that drawer's widest page
        # (content margins, spin-box arrows, checkbox/button labels, the
        # vertical scrollbar) actually needs, clipping controls at the
        # drawer's right edge (or, on the right drawer, its action buttons'
        # text). Raise it to that real minimum when necessary; wide windows
        # are unaffected since the fraction already clears it there. The
        # right drawer previously had no such floor at all -- just the flat
        # fraction -- which is what let "Exclude Selected Rows from Working
        # Data"/"Keep Selected Rows in Working Data" clip; those labels are
        # shorter now too (see `gui.widgets.data_tools_panel`), so this
        # floor only has to cover their shorter natural width, not the
        # drawer becoming uncomfortably wide to fit the old, longer ones.
        # Stored (not just a local) so `resizeEvent` below can reuse them on
        # every resize without re-walking every left-drawer page's
        # `minimumSizeHint()` each time -- neither ever changes after
        # construction (drawer pages are fixed; see `ToolDrawer.add_page`).
        self._left_drawer_min_width = _side_drawer_min_width(self.tool_drawer, left_drawer_content_widgets)
        self._right_drawer_min_width = _side_drawer_min_width(self.working_drawer, [self.data_tools_panel])
        # Neither drawer has auto-collapsed anything yet -- see
        # `_apply_auto_collapse`'s own docstring for why this state is kept
        # separate from `ToolDrawer.is_collapsed` (which alone can't
        # distinguish "the user collapsed this" from "width pressure did").
        self._left_auto_collapsed = False
        self._right_auto_collapsed = False
        self._left_auto_collapsed_from_key: str | None = None
        self._right_auto_collapsed_from_key: str | None = None
        left_width, center_width, right_width, auto_collapse_left, auto_collapse_right = compute_drawer_widths(
            geometry.width(),
            self._left_drawer_min_width,
            self._right_drawer_min_width,
            left_floor_width=self.tool_drawer.strip_width,
            right_floor_width=self.working_drawer.strip_width,
            left_locked_collapsed=self.tool_drawer.is_collapsed,
            right_locked_collapsed=self.working_drawer.is_collapsed,
        )
        self._apply_auto_collapse(auto_collapse_left, auto_collapse_right)
        main_splitter.setSizes([left_width, center_width, right_width])
        self.main_splitter = main_splitter

        self.setCentralWidget(main_splitter)
        self.center_splitter.setSizes(
            [int(geometry.height() * 0.7), int(geometry.height() * 0.3)]
        )

        # Connected only after the drawers' initial pages/sizing above so
        # constructing the window doesn't itself trigger a splitter resize.
        # Keyed by each drawer's main_splitter index (0 = left, 2 = right)
        # so collapsing/reopening one side never disturbs the other's
        # remembered width -- see `_set_side_drawer_collapsed`.
        sizes = main_splitter.sizes()
        self._drawer_open_widths: dict[int, int] = {0: sizes[0], 2: sizes[2]}
        self.tool_drawer.collapsed_changed.connect(
            lambda collapsed: self._set_side_drawer_collapsed(0, self.tool_drawer, collapsed)
        )
        self.working_drawer.collapsed_changed.connect(
            lambda collapsed: self._set_side_drawer_collapsed(2, self.working_drawer, collapsed)
        )

        self.dataset_panel.dataset_selected.connect(self._on_dataset_selected)
        self.dataset_panel.add_to_plot_requested.connect(self._on_add_to_plot)
        self.dataset_panel.clear_plot_requested.connect(self._on_clear_plot)
        self.dataset_panel.axis_preset_requested.connect(self._on_axis_preset_requested)
        self.dataset_panel.datasets_changed.connect(self._on_datasets_changed)
        self.graph_library_panel.graph_library_changed.connect(self._on_graph_library_changed)
        self.graph_library_panel.graph_loaded_into_panel.connect(self._on_graph_loaded_into_panel)
        self.series_panel.changed.connect(self._on_figure_content_changed)
        self.properties_panel.changed.connect(self._on_figure_content_changed)
        self.figure_size_panel.changed.connect(self._on_figure_content_changed)
        self.figure_layout_panel.changed.connect(self._on_figure_content_changed)
        self.figure_size_panel.panel_switched.connect(self._on_panel_switched)
        self.figure_size_panel.theme_change_requested.connect(self._on_theme_changed)
        self.data_tools_panel.transformation_applied.connect(self._on_transformation_applied)
        self.data_tools_panel.plot_selected_rows_requested.connect(self._on_plot_selected_rows)
        self.figure_size_panel.panel_labels_check.toggled.connect(self._sync_panel_labels_action)
        self.analysis_panel.analysis_result_ready.connect(self._on_analysis_result_ready)
        self.analysis_panel.add_to_plot_requested.connect(self._on_add_to_plot)
        self.analysis_panel.remove_fit_curve_requested.connect(self._on_remove_fit_curve)
        self.analysis_panel.history_result_selected.connect(self._on_history_result_selected)

        self.workbench_tab_bar.workbench_selected.connect(self._on_workbench_tab_selected)
        self.workbench_tab_bar.new_workbench_requested.connect(self._on_new_workbench_requested)
        self.workbench_tab_bar.rename_requested.connect(self._on_rename_workbench_requested)
        self.workbench_tab_bar.duplicate_requested.connect(self._on_duplicate_workbench_requested)
        self.workbench_tab_bar.delete_requested.connect(self._on_delete_workbench_requested)
        self.workbench_tab_bar.set_workbenches(self._project.workbenches, self._project.active_workbench_id)

        self._create_menu()
        self._create_toolbar()
        self._sync_window_title()

    # --- Undo/Redo per-Workbench state (see the dict comment in __init__) --

    @property
    def _undo_manager(self) -> UndoManager:
        return self._undo_managers.setdefault(self._current_workbench_id, UndoManager())

    @_undo_manager.setter
    def _undo_manager(self, value: UndoManager) -> None:
        self._undo_managers[self._current_workbench_id] = value

    @property
    def _pending_undo_snapshot(self) -> GnoviFigure:
        if self._current_workbench_id not in self._pending_snapshots:
            self._pending_snapshots[self._current_workbench_id] = snapshot_figure(
                self.figure_model, self.dataset_manager
            )
        return self._pending_snapshots[self._current_workbench_id]

    @_pending_undo_snapshot.setter
    def _pending_undo_snapshot(self, value: GnoviFigure) -> None:
        self._pending_snapshots[self._current_workbench_id] = value

    # --- Menus -----------------------------------------------------------

    def _create_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        self.new_project_action = file_menu.addAction("New Project")
        self.new_project_action.setShortcut(QKeySequence.New)
        self.new_project_action.triggered.connect(self._on_new_project)
        self.open_project_action = file_menu.addAction("Open Project…")
        self.open_project_action.setShortcut(QKeySequence.Open)
        self.open_project_action.triggered.connect(self._on_open_project)
        self.save_project_action = file_menu.addAction("Save Project")
        self.save_project_action.setShortcut(QKeySequence.Save)
        self.save_project_action.triggered.connect(self._on_save_project)
        self.save_project_as_action = file_menu.addAction("Save Project As…")
        self.save_project_as_action.setShortcut(QKeySequence.SaveAs)
        self.save_project_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        # Figure-content Undo/Redo only -- see gui.undo_manager for the
        # scoping rationale (dataset mutations keep their own Reset Working
        # Data recovery path and Transformation History, deliberately kept
        # separate from this stack). These are the SAME QAction objects the
        # toolbar's Undo/Redo buttons use (see `_create_toolbar`) -- one
        # shared action pair drives both, never two separate systems. Scoped
        # per-Workbench (see the `_undo_manager`/`_pending_undo_snapshot`
        # properties above): whichever Workbench is active when Undo/Redo
        # fires is the only one they can ever affect.
        edit_menu = self.menuBar().addMenu("&Edit")
        self.undo_action = edit_menu.addAction("Undo")
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.setToolTip("Undo (Ctrl+Z)")
        self.undo_action.triggered.connect(self._on_undo)
        self.redo_action = edit_menu.addAction("Redo")
        # Ctrl+Shift+Z is QKeySequence.Redo's own platform-standard binding
        # on Linux/macOS; Ctrl+Y is added as a widely-expected alternative
        # (e.g. the Windows convention) regardless of platform.
        self.redo_action.setShortcuts([QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")])
        self.redo_action.setToolTip("Redo (Ctrl+Shift+Z)")
        self.redo_action.triggered.connect(self._on_redo)
        self._sync_undo_redo_actions()

        data_menu = self.menuBar().addMenu("&Data")
        import_action = data_menu.addAction("Import Data…")
        import_action.triggered.connect(self._on_import_data)
        save_working_action = data_menu.addAction("Save Working Data…")
        save_working_action.triggered.connect(self._on_save_working_data)

        plot_menu = self.menuBar().addMenu("&Plot")
        add_to_plot_action = plot_menu.addAction("Add to Plot")
        add_to_plot_action.triggered.connect(lambda: self.dataset_panel.add_to_plot_button.click())
        clear_plot_action = plot_menu.addAction("Clear Plot")
        clear_plot_action.triggered.connect(lambda: self.dataset_panel.clear_plot_button.click())

        figure_menu = self.menuBar().addMenu("&Figure")
        figure_size_action = figure_menu.addAction("Figure Size && Ratio…")
        figure_size_action.triggered.connect(self._show_figure_size_dialog)
        publication_action = figure_menu.addAction("Publication Presets…")
        publication_action.triggered.connect(self._show_figure_size_dialog)
        typography_action = figure_menu.addAction("Typography…")
        typography_action.triggered.connect(self._show_figure_size_dialog)
        figure_menu.addSeparator()
        axes_action = figure_menu.addAction("Axes && Ticks…")
        axes_action.triggered.connect(self._show_axes_dialog)
        legend_action = figure_menu.addAction("Legend…")
        legend_action.triggered.connect(self._show_axes_dialog)
        figure_menu.addSeparator()
        export_action = figure_menu.addAction("Export Figure…")
        export_action.triggered.connect(self._on_export_figure)

        self.panels_menu = self.menuBar().addMenu("&Panels")
        self.layout_menu = self.panels_menu.addMenu("Layout")
        self.layout_menu.aboutToShow.connect(self._rebuild_layout_menu)
        self.active_panel_menu = self.panels_menu.addMenu("Active Panel…")
        self.active_panel_menu.aboutToShow.connect(self._rebuild_active_panel_menu)
        self.panels_menu.addSeparator()
        copy_style_action = self.panels_menu.addAction("Copy Active Panel Style to All Panels")
        copy_style_action.triggered.connect(self._on_copy_style_to_all_panels)
        self.panels_menu.addSeparator()
        self.panel_labels_action = self.panels_menu.addAction("Panel Labels On/Off")
        self.panel_labels_action.setCheckable(True)
        self.panel_labels_action.toggled.connect(self._on_toggle_panel_labels)

        # Dedicated top-level menu (not nested under File) -- these act on
        # whichever Workbench is currently active, exactly mirroring what
        # the Workbench tab strip's right-click context menu offers as a
        # secondary path (see `gui.widgets.workbench_tabs.WorkbenchTabBar`).
        self.workbench_menu = self.menuBar().addMenu("&Workbench")
        self.new_workbench_action = self.workbench_menu.addAction("New Workbench")
        self.new_workbench_action.triggered.connect(self._on_new_workbench_requested)
        self.rename_workbench_action = self.workbench_menu.addAction("Rename Workbench")
        self.rename_workbench_action.triggered.connect(
            lambda: self._on_rename_workbench_requested(self._project.active_workbench_id)
        )
        self.duplicate_workbench_action = self.workbench_menu.addAction("Duplicate Workbench")
        self.duplicate_workbench_action.triggered.connect(
            lambda: self._on_duplicate_workbench_requested(self._project.active_workbench_id)
        )
        self.delete_workbench_action = self.workbench_menu.addAction("Delete Workbench")
        self.delete_workbench_action.triggered.connect(
            lambda: self._on_delete_workbench_requested(self._project.active_workbench_id)
        )
        self.workbench_menu.aboutToShow.connect(self._sync_workbench_menu_state)
        self._sync_workbench_menu_state()

        view_menu = self.menuBar().addMenu("&View")
        # Hides/shows the entire left tool strip + drawer (both the Data /
        # Plot / Series buttons and whichever page is open) -- distinct
        # from a strip button's own collapse, which only hides the drawer
        # page and leaves the strip itself visible.
        self.toggle_controls_action = view_menu.addAction("Controls")
        self.toggle_controls_action.setCheckable(True)
        self.toggle_controls_action.setChecked(True)
        self.toggle_controls_action.toggled.connect(self._on_toggle_controls)

        # Same idea for the RIGHT Working Data drawer, independent of the
        # left one.
        self.toggle_working_data_action = view_menu.addAction("Working Data")
        self.toggle_working_data_action.setCheckable(True)
        self.toggle_working_data_action.setChecked(True)
        self.toggle_working_data_action.toggled.connect(self._on_toggle_working_data)

        self.toggle_bottom_panel_action = view_menu.addAction("Bottom Panel")
        self.toggle_bottom_panel_action.setCheckable(True)
        self.toggle_bottom_panel_action.setChecked(True)
        self.toggle_bottom_panel_action.toggled.connect(self._on_toggle_bottom_panel)

        view_menu.addSeparator()
        # "Plot Theme" recolors only the Matplotlib canvas (see
        # gui.styles.PlotTheme) -- it never touches this menu, any other
        # chrome, or dialogs.
        theme_menu = view_menu.addMenu("Plot Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        self._theme_actions: dict[PlotTheme, object] = {}
        for mode, label in _PLOT_THEME_MENU_LABELS:
            action = theme_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(mode == self.figure_model.plot_theme)
            theme_group.addAction(action)
            action.triggered.connect(lambda _checked=False, m=mode: self._on_theme_changed(m))
            self._theme_actions[mode] = action
        self._theme_action_group = theme_group

        # Reference cursor: on-screen-only crosshair/reference-line overlay
        # that follows the mouse -- never part of an export (see
        # gui.widgets.plot_canvas.PlotCanvas.update_reference_cursor).
        cursor_menu = view_menu.addMenu("Reference Cursor")
        cursor_group = QActionGroup(self)
        cursor_group.setExclusive(True)
        self._cursor_actions: dict[ReferenceCursorMode, object] = {}
        for mode, label in _REFERENCE_CURSOR_MENU_LABELS:
            action = cursor_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(mode == self._cursor_mode)
            cursor_group.addAction(action)
            action.triggered.connect(lambda _checked=False, m=mode: self._on_cursor_mode_changed(m))
            self._cursor_actions[mode] = action
        self._cursor_action_group = cursor_group

        help_menu = self.menuBar().addMenu("&Help")
        about_action = help_menu.addAction(f"About {APP_NAME}")
        about_action.triggered.connect(self._show_about)

    def _rebuild_layout_menu(self) -> None:
        self.layout_menu.clear()
        current_index = self.figure_size_panel.layout_combo.currentIndex()
        for i, (text, _dims) in enumerate(LAYOUT_PRESETS):
            action = self.layout_menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(i == current_index)
            action.triggered.connect(lambda _checked=False, index=i: self._set_layout(index))

    def _rebuild_active_panel_menu(self) -> None:
        self.active_panel_menu.clear()
        current_index = self.figure_model.active_panel_index
        for i in range(len(self.figure_model.panels)):
            action = self.active_panel_menu.addAction(f"Panel {i + 1}")
            action.setCheckable(True)
            action.setChecked(i == current_index)
            action.triggered.connect(lambda _checked=False, index=i: self._set_active_panel(index))

    def _show_about(self):
        QMessageBox.about(self, f"About {APP_NAME}", about_text())

    # --- Toolbar -----------------------------------------------------------

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main", self)
        # Scopes the raised/tactile QToolButton chrome (see gui.styles) to
        # this toolbar only, via `QToolBar#MainToolBar QToolButton` --
        # deliberately NOT the bare `QToolButton` selector, which would also
        # repaint the Matplotlib navigation toolbar's own buttons (added
        # separately, see `__init__`'s `nav_toolbar`) and previously made
        # their icons unreadable against the new opaque background.
        toolbar.setObjectName("MainToolBar")
        self.addToolBar(toolbar)

        # Undo/Redo: the SAME `self.undo_action`/`self.redo_action` QAction
        # objects the Edit menu uses (see `_create_menu`) -- `addAction`
        # adds the identical object to this toolbar, not a copy, so menu
        # and toolbar are always in sync by construction, never two
        # systems. Leading position (before Import/Save/Export) since these
        # are described as important, immediately-discoverable working
        # controls. Icons are hand-drawn (see `_make_undo_redo_icon`) so
        # they render identically regardless of OS icon theme; disabled
        # state (nothing to undo/redo) already renders muted via the
        # QToolButton:disabled QSS rule plus Qt's own auto-generated
        # disabled-icon treatment.
        self.undo_action.setIcon(_make_undo_redo_icon("undo"))
        self.redo_action.setIcon(_make_undo_redo_icon("redo"))
        toolbar.addAction(self.undo_action)
        toolbar.addAction(self.redo_action)
        toolbar.addSeparator()

        import_action = toolbar.addAction("Import Data")
        import_action.triggered.connect(self._on_import_data)

        save_working_action = toolbar.addAction("Save Working Data")
        save_working_action.triggered.connect(self._on_save_working_data)

        export_action = toolbar.addAction("Export Figure")
        export_action.triggered.connect(self._on_export_figure)

        toolbar.addSeparator()

        self.toolbar_layout_combo = QComboBox()
        self.toolbar_layout_combo.addItems([text for text, _dims in LAYOUT_PRESETS])
        self.toolbar_layout_combo.setToolTip("Panel Layout")
        self.toolbar_layout_combo.currentIndexChanged.connect(self._on_toolbar_layout_changed)
        toolbar.addWidget(self.toolbar_layout_combo)

        self.toolbar_panel_combo = QComboBox()
        self.toolbar_panel_combo.setToolTip("Active Panel")
        self.toolbar_panel_combo.currentIndexChanged.connect(self._on_toolbar_panel_changed)
        toolbar.addWidget(self.toolbar_panel_combo)

        toolbar.addSeparator()

        self.toolbar_theme_combo = QComboBox()
        for mode, label in _PLOT_THEME_MENU_LABELS:
            self.toolbar_theme_combo.addItem(label, mode)
        self.toolbar_theme_combo.setToolTip("Plot Theme")
        self.toolbar_theme_combo.currentIndexChanged.connect(self._on_toolbar_theme_changed)
        toolbar.addWidget(self.toolbar_theme_combo)

        self.toolbar_cursor_combo = QComboBox()
        for mode, label in _REFERENCE_CURSOR_TOOLBAR_LABELS:
            self.toolbar_cursor_combo.addItem(label, mode)
        self.toolbar_cursor_combo.setToolTip("Reference Cursor")
        self.toolbar_cursor_combo.currentIndexChanged.connect(self._on_toolbar_cursor_changed)
        toolbar.addWidget(self.toolbar_cursor_combo)

        self._sync_toolbar_panel_controls()
        self._sync_theme_controls()
        self._sync_cursor_controls()

    def _sync_toolbar_panel_controls(self) -> None:
        self.toolbar_layout_combo.blockSignals(True)
        self.toolbar_layout_combo.setCurrentIndex(self.figure_size_panel.layout_combo.currentIndex())
        self.toolbar_layout_combo.blockSignals(False)

        self.toolbar_panel_combo.blockSignals(True)
        self.toolbar_panel_combo.clear()
        for i in range(len(self.figure_model.panels)):
            self.toolbar_panel_combo.addItem(f"Panel {i + 1}")
        self.toolbar_panel_combo.setCurrentIndex(self.figure_model.active_panel_index)
        self.toolbar_panel_combo.blockSignals(False)

    def _on_toolbar_layout_changed(self, index: int) -> None:
        if index < 0 or index == self.figure_size_panel.layout_combo.currentIndex():
            return
        self._set_layout(index)

    def _on_toolbar_panel_changed(self, index: int) -> None:
        if index < 0 or index == self.figure_model.active_panel_index:
            return
        self._set_active_panel(index)

    def _on_toolbar_theme_changed(self, index: int) -> None:
        mode = self.toolbar_theme_combo.itemData(index)
        if mode is None or mode == self.figure_model.plot_theme:
            return
        self._on_theme_changed(mode)

    def _on_toolbar_cursor_changed(self, index: int) -> None:
        mode = self.toolbar_cursor_combo.itemData(index)
        if mode is None or mode == self._cursor_mode:
            return
        self._on_cursor_mode_changed(mode)

    # --- Shared handlers (menu, toolbar, and sidebar controls all call these) --

    def _on_import_data(self) -> None:
        self.dataset_panel.import_button.click()

    def _on_save_working_data(self) -> None:
        dataset = self.dataset_panel.current_dataset
        if dataset is None:
            QMessageBox.information(self, "Save Working Data", "Select a dataset first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Working Data", f"{dataset.name}.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            dataset.dataframe.to_csv(path, index=False)
        except OSError as exc:
            QMessageBox.critical(self, "Save Working Data", str(exc))

    def _show_figure_size_dialog(self) -> None:
        self._open_drawer_page("figure")

    def _show_axes_dialog(self) -> None:
        self._open_drawer_page("axes")

    def _open_drawer_page(self, key: str) -> None:
        """Reveal `key`'s page in the left ToolDrawer -- Figure Size/
        Typography/Axes/Legend now live there permanently rather than in
        their own dialogs, so these Figure-menu entries are just shortcuts
        to the one place those controls live."""
        if not self.tool_drawer.isVisible():
            self.toggle_controls_action.setChecked(True)
        self.tool_drawer.show_page(key)

    def _set_layout(self, index: int) -> None:
        self.figure_size_panel.layout_combo.setCurrentIndex(index)

    def _set_active_panel(self, index: int) -> None:
        self.figure_size_panel.panel_combo.setCurrentIndex(index)

    def _on_copy_style_to_all_panels(self) -> None:
        self.figure_model.copy_active_panel_style_to_all()
        self._on_figure_content_changed()

    def _on_toggle_panel_labels(self, checked: bool) -> None:
        self.figure_size_panel.panel_labels_check.setChecked(checked)

    def _sync_panel_labels_action(self, checked: bool) -> None:
        self.panel_labels_action.blockSignals(True)
        self.panel_labels_action.setChecked(checked)
        self.panel_labels_action.blockSignals(False)

    def _on_toggle_controls(self, visible: bool) -> None:
        self.tool_drawer.setVisible(visible)

    def _on_toggle_working_data(self, visible: bool) -> None:
        self.working_drawer.setVisible(visible)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reflow_side_drawers()

    def _reflow_side_drawers(self) -> None:
        """Re-derive fresh left/center/right splitter widths for the
        window's *current* content width on every resize -- not just once
        at construction (see `compute_drawer_widths`'s own docstring for
        why `QSplitter`'s default behavior, proportionally rescaling
        whatever sizes were last set, isn't enough on its own: it has no
        notion of either drawer's real content floor, so it can shrink an
        expanded drawer below what its page actually needs and clip it --
        `_wrap_scrollable`'s horizontal scrollbar is deliberately off, so
        an over-shrunk page clips rather than scrolls).

        A side the user collapsed by hand (`is_collapsed` True, but not via
        `_apply_auto_collapse`) is passed through as locked -- pinned at
        its strip width, never fought back open just because the window
        grew. A side *this* window auto-collapsed under width pressure
        (`_left_auto_collapsed`/`_right_auto_collapsed`) is passed through
        as still negotiable, so `compute_drawer_widths` is free to hand it
        back a real width (and `_apply_auto_collapse` reopens it) once
        there's room again -- see that function's own docstring.

        No-ops before `main_splitter` exists -- `QMainWindow.resizeEvent`
        can fire from `setGeometry`/`show()` earlier in `__init__`, before
        any of the widgets this reads have been constructed yet.
        """
        main_splitter = getattr(self, "main_splitter", None)
        if main_splitter is None:
            return
        total_width = main_splitter.width()
        if total_width <= 0:
            return
        left_locked = self.tool_drawer.is_collapsed and not self._left_auto_collapsed
        right_locked = self.working_drawer.is_collapsed and not self._right_auto_collapsed
        left_width, center_width, right_width, auto_collapse_left, auto_collapse_right = compute_drawer_widths(
            total_width,
            self._left_drawer_min_width,
            self._right_drawer_min_width,
            left_floor_width=self.tool_drawer.strip_width,
            right_floor_width=self.working_drawer.strip_width,
            left_locked_collapsed=left_locked,
            right_locked_collapsed=right_locked,
        )
        self._apply_auto_collapse(auto_collapse_left, auto_collapse_right)
        main_splitter.setSizes([left_width, center_width, right_width])

    def _apply_auto_collapse(self, auto_collapse_left: bool, auto_collapse_right: bool) -> None:
        """Apply `compute_drawer_widths`'s auto-collapse decision for this
        layout pass, transitioning only what actually changed since the
        last one.

        Kept as state separate from `ToolDrawer.is_collapsed` (via
        `_left_auto_collapsed`/`_right_auto_collapsed`) specifically so a
        side *this* function collapsed under width pressure can be
        reopened again once the window has room -- while a side the user
        collapsed by hand (`is_collapsed` True, `_*_auto_collapsed` False)
        is never touched here at all: `_reflow_side_drawers` always passes
        that case through as `*_locked_collapsed=True`, so
        `compute_drawer_widths` never asks to auto-collapse or reopen it in
        the first place. Restores whichever page was active before an
        auto-collapse (falling back to each drawer's default page if that
        somehow isn't available) rather than defaulting to nothing shown.
        """
        if auto_collapse_left and not self.tool_drawer.is_collapsed:
            self._left_auto_collapsed_from_key = self.tool_drawer.active_key
            self.tool_drawer.collapse()
            self._left_auto_collapsed = True
        elif not auto_collapse_left and self._left_auto_collapsed:
            self.tool_drawer.show_page(self._left_auto_collapsed_from_key or "data")
            self._left_auto_collapsed = False

        if auto_collapse_right and not self.working_drawer.is_collapsed:
            self._right_auto_collapsed_from_key = self.working_drawer.active_key
            self.working_drawer.collapse()
            self._right_auto_collapsed = True
        elif not auto_collapse_right and self._right_auto_collapsed:
            self.working_drawer.show_page(self._right_auto_collapsed_from_key or "working")
            self._right_auto_collapsed = False

    def _set_side_drawer_collapsed(self, splitter_index: int, drawer: ToolDrawer, collapsed: bool) -> None:
        """Reclaim (or return) a side drawer's width for the plot canvas
        when its tool-strip page is collapsed/reopened, remembering the
        last expanded width for *that side* so reopening restores it
        rather than a sliver -- mirrors `_on_toggle_bottom_panel`'s
        approach for the bottom panel. `splitter_index` is the drawer's own
        slot in `main_splitter` (0 = left ToolDrawer, 2 = right
        `working_drawer`); the center plot canvas is always index 1 and is
        the only slot that ever grows/shrinks in response -- collapsing one
        side never touches the other side's width.
        """
        sizes = self.main_splitter.sizes()
        strip_width = drawer.strip_width
        if collapsed:
            self._drawer_open_widths[splitter_index] = sizes[splitter_index]
            freed = max(sizes[splitter_index] - strip_width, 0)
            sizes[splitter_index] = strip_width
            sizes[1] += freed
        else:
            open_width = self._drawer_open_widths.get(splitter_index)
            if open_width is None:
                return
            delta = open_width - sizes[splitter_index]
            sizes[splitter_index] = open_width
            sizes[1] = max(sizes[1] - delta, 0)
        self.main_splitter.setSizes(sizes)

    def _on_theme_changed(self, mode: PlotTheme | str) -> None:
        """Change the active figure's Plot Theme -- declarative
        `GnoviFigure` state (see `plotting.figure.PlotTheme`), so this is a
        figure-content edit like any other: undoable, marks the project
        dirty, and persists in the saved `.gnovi` file. Only the Matplotlib
        canvas re-renders differently (see `_rerender`); the application
        chrome is never touched (see `gui.styles` module docstring).

        `mode` is normalized to `PlotTheme` here rather than assumed to
        already be one: `QComboBox.itemData()` round-trips a str-subclassed
        Enum through QVariant and hands back a plain `str` (confirmed --
        Qt's marshalling, not a display-text guess), so
        `_on_toolbar_theme_changed` below passes a bare string on every
        toolbar selection. The View menu's `QAction` closures pass a real
        `PlotTheme` (no QVariant round-trip involved), so this also covers
        that path as a no-op. Guarded against a no-op change (both paths
        already check before calling this, but an exclusive `QActionGroup`
        can still re-fire `triggered` for the already-checked action) --
        otherwise re-selecting the current theme would push a spurious undo
        checkpoint and mark a clean project dirty for nothing.
        """
        if not isinstance(mode, PlotTheme):
            mode = PlotTheme(mode)
        if mode == self.figure_model.plot_theme:
            return
        self.figure_model.plot_theme = mode
        # Last-used theme becomes the default for the *next* new
        # figure/project (see `_new_project`) -- never overrides a loaded
        # project's own saved theme.
        self._settings.setValue("plot_theme", mode.value)
        self._sync_theme_controls()
        self._on_figure_content_changed()

    def _sync_theme_controls(self) -> None:
        """Keep the View > Plot Theme menu and the toolbar Plot Theme combo
        showing the active figure's current theme, regardless of whether it
        just changed via the menu/toolbar or a different figure became
        active (Open/New Project, `_load_project_into_window`)."""
        current = self.figure_model.plot_theme
        for mode, action in self._theme_actions.items():
            action.blockSignals(True)
            action.setChecked(mode == current)
            action.blockSignals(False)

        self.toolbar_theme_combo.blockSignals(True)
        self.toolbar_theme_combo.setCurrentIndex(self.toolbar_theme_combo.findData(current))
        self.toolbar_theme_combo.blockSignals(False)

        self.figure_size_panel.set_current_theme(current)

    def _on_cursor_mode_changed(self, mode: ReferenceCursorMode | str) -> None:
        """Normalized the same way and for the same reason as
        `_on_theme_changed` above -- see its docstring."""
        if not isinstance(mode, ReferenceCursorMode):
            mode = ReferenceCursorMode(mode)
        self._cursor_mode = mode
        self._settings.setValue("reference_cursor", mode.value)
        self._sync_cursor_controls()
        self.plot_canvas.set_cursor_mode(mode)

    def _sync_cursor_controls(self) -> None:
        """Keep the View > Reference Cursor menu and the toolbar combo
        showing the same selection, regardless of which one changed it."""
        for mode, action in self._cursor_actions.items():
            action.blockSignals(True)
            action.setChecked(mode == self._cursor_mode)
            action.blockSignals(False)

        self.toolbar_cursor_combo.blockSignals(True)
        self.toolbar_cursor_combo.setCurrentIndex(self.toolbar_cursor_combo.findData(self._cursor_mode))
        self.toolbar_cursor_combo.blockSignals(False)

    def _on_toggle_bottom_panel(self, visible: bool) -> None:
        """Show/hide the bottom panel without disturbing the plot canvas's
        own configured figure layout -- only the splitter's on-screen
        allocation changes. Remembers the last size split so re-showing
        restores it instead of collapsing to a sliver."""
        if visible:
            self.bottom_panel.setVisible(True)
            if self._bottom_panel_sizes is not None:
                self.center_splitter.setSizes(self._bottom_panel_sizes)
        else:
            self._bottom_panel_sizes = self.center_splitter.sizes()
            self.bottom_panel.setVisible(False)

    def _on_analysis_result_ready(self, result) -> None:
        """An analysis tool (curve fitting today; any later tool the same
        way) produced a result -- record it in the active Workbench's own
        panel-scoped history (see `core.workbench.Workbench.
        analysis_results`) and show it immediately rather than making the
        scientist go find and open Results themselves. Reuses the Bottom
        Panel visibility toggle's own restore-last-size logic
        (`_on_toggle_bottom_panel`) instead of duplicating it, by driving
        the same action the View menu entry drives.

        `analysis_results` is persisted project state (see that field's
        own docstring) -- adding to it is a genuine content change, so
        this marks the project dirty, same as any other edit that will
        be saved. Merely *displaying* an existing result (panel switch,
        Workbench switch, undo/redo -- see `_sync_results_to_active_panel`)
        never does."""
        if result.source_panel_id is not None:
            self._project.active_workbench.analysis_results.add(result.source_panel_id, result)
            self._set_dirty(True)
        self.analysis_result_view.show_result(result)
        self.analysis_panel.sync_history(
            self._project.active_workbench.analysis_results.all(self.figure_model.active_panel.id),
            result,
        )
        if not self.toggle_bottom_panel_action.isChecked():
            self.toggle_bottom_panel_action.setChecked(True)
        self.bottom_panel.show_results_tab()

    def _on_history_result_selected(self, result) -> None:
        """The scientist picked a different entry in the Analysis
        History list (see `AnalysisPanel.history_result_selected`) --
        record it as the active panel's new current result and show it,
        without rerunning anything. `current_result_id` is persisted
        project state (see `Workbench.analysis_results`), so this marks
        the project dirty, same as `_on_analysis_result_ready`; no undo
        checkpoint (selection isn't part of the figure/series undo
        snapshot, same as any other display-only sync)."""
        self._project.active_workbench.analysis_results.set_current(
            self.figure_model.active_panel.id, result.result_id
        )
        self.analysis_result_view.show_result(result)
        self._set_dirty(True)

    def _on_mouse_move(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            self.coord_label.setText("")
            self.plot_canvas.clear_reference_cursor()
            return
        self.coord_label.setText(f"x = {event.xdata:.4g}, y = {event.ydata:.4g}")
        self.plot_canvas.update_reference_cursor(event.inaxes, event.xdata, event.ydata)

    def _on_mouse_leave(self, _event) -> None:
        self.coord_label.setText("")
        self.plot_canvas.clear_reference_cursor()

    def _on_canvas_click(self, event) -> None:
        """Clicking anywhere inside a panel's Axes makes that panel active
        -- the same effect as picking it from the toolbar/menu Active Panel
        control, just faster for a multi-panel layout. A single-panel
        layout has nothing else to activate, so this is a no-op there."""
        if event.inaxes is None:
            return
        index = self.plot_canvas.panel_index_for_axes(event.inaxes)
        if index is None or index == self.figure_model.active_panel_index:
            return
        self._set_active_panel(index)

    def _on_dataset_selected(self, dataset):
        self.preview_model.set_dataframe(dataset.dataframe if dataset is not None else None)
        self.data_tools_panel.set_dataset(dataset)

    def _on_transformation_applied(self, dataset, row_set_changed: bool) -> None:
        self._set_dirty(True)
        self.preview_model.set_dataframe(dataset.dataframe)
        self.dataset_panel.refresh_columns()
        if row_set_changed:
            self.dataset_panel.reset_manual_cycles()

        newly_stale = self.figure_model.invalidate_series_for_dataset(dataset, row_set_changed)
        if newly_stale:
            self.series_panel.refresh()
        self._rerender()

        if newly_stale:
            names = "\n".join(f"- {s.label}" for s in newly_stale)
            QMessageBox.information(
                self,
                "Plot Series Invalidated",
                f"The working data for '{dataset.name}' changed in a way that invalidates "
                f"{len(newly_stale)} plot series (a row range no longer applies, or a "
                f"column it used was removed):\n\n{names}\n\n"
                "These are no longer drawn. Remove and re-add them against the updated "
                "working data.",
            )

    def _on_plot_selected_rows(self, positions: list[int]) -> None:
        """Add a new PlotSeries scoped to the selected Data Preview rows to
        the active panel, without touching the dataset's raw or working
        data. Deliberately separate from Working Data transformations
        (Exclude/Keep Selection): no `Dataset` method is called here, so
        nothing is added to the transformation history and no existing
        series can be invalidated.
        """
        dataset = self.dataset_panel.current_dataset
        if dataset is None:
            return

        x_col = self.dataset_panel.x_combo.currentText()
        y_col = self.dataset_panel.y_combo.currentText()
        if not x_col or not y_col:
            QMessageBox.warning(self, "Plot Selected Rows", "Select X and Y columns to plot.")
            return

        try:
            row_range = contiguous_row_range(positions)
        except InvalidRowRangeError as exc:
            QMessageBox.warning(self, "Plot Selected Rows", str(exc))
            return

        start, end = row_range
        try:
            numeric_xy(dataset.dataframe.iloc[start:end], x_col, y_col)
        except (KeyError, InsufficientNumericDataError) as exc:
            QMessageBox.critical(self, "Plot Selected Rows", str(exc))
            return

        series = PlotSeries.line(
            dataset,
            x_col,
            y_col,
            label=f"{dataset.name} — rows {start}–{end - 1}",
            row_range=row_range,
        )
        self._on_add_to_plot([series])

    def _on_add_to_plot(self, series_list):
        dark_mode = self.figure_model.plot_theme == PlotTheme.DARK
        last_id = None
        for series in series_list:
            self.figure_model.add_series(series, dark_mode=dark_mode)
            last_id = series.id
        self.series_panel.refresh(select_id=last_id)
        self._on_figure_content_changed()

    def _on_remove_fit_curve(self, series_ids: list[str]) -> None:
        """"Remove Fit Curve from Plot" -- removes only the derived
        PlotSeries (see `AnalysisPanel.remove_fit_curve_requested`),
        through the exact same removal path `PlotSeriesPanel`'s own
        delete action uses, so it participates in Undo/Redo and dirty-
        marking identically to any other series removal. Never touches
        `FitResult`/history/Results/residuals -- those are untouched by
        what's plotted (see `Workbench.analysis_results`)."""
        for series_id in series_ids:
            self.figure_model.remove_series(series_id)
        self.series_panel.refresh()
        self._on_figure_content_changed()

    def _on_clear_plot(self):
        self.figure_model.clear_series()
        self.series_panel.refresh()
        self._on_figure_content_changed()

    def _on_axis_preset_requested(self, preset: dict) -> None:
        panel = self.figure_model.active_panel
        panel.xlabel = preset.get("xlabel", panel.xlabel)
        panel.ylabel = preset.get("ylabel", panel.ylabel)
        self.properties_panel.refresh()
        self._on_figure_content_changed()

    def _on_panel_switched(self):
        self.series_panel.refresh()
        self.properties_panel.refresh()
        self.figure_layout_panel.refresh()
        self.analysis_panel.refresh()
        self._refresh_active_panel_context()
        self._sync_toolbar_panel_controls()
        self._sync_results_to_active_panel()
        self._rerender()

    def _sync_results_to_active_panel(self) -> None:
        """Restore Results to whatever the *active* Workbench's *active*
        panel's own analysis history says it should show -- never rerun
        an analysis, never create a new result, purely display
        restoration (see `core.workbench.Workbench.analysis_results`).

        Call after anything that can change which panel/Workbench is
        active: panel switch, layout change (both fire `panel_switched`,
        see `_on_panel_switched`), Workbench switch/create/duplicate/
        delete and New/Open Project (all funnel through
        `_activate_workbench`), and undo/redo (`_restore_figure_snapshot`,
        which can change `active_panel_index`/the panel set). Also prunes
        the active Workbench's history to the panels that currently exist
        in its figure -- cheap and idempotent, so running it on every
        call needs no separate "did a layout change actually happen"
        signal (see `Workbench.analysis_results.prune_to`)."""
        workbench = self._project.active_workbench
        workbench.analysis_results.prune_to({p.id for p in self.figure_model.panels})
        panel_id = self.figure_model.active_panel.id
        result = workbench.analysis_results.current(panel_id)
        if result is not None:
            self.analysis_result_view.show_result(result)
        else:
            self.analysis_result_view.clear()
        self.analysis_panel.sync_history(workbench.analysis_results.all(panel_id), result)

    def _refresh_active_panel_context(self) -> None:
        """Refresh every page's "Active panel / Graph / Data" context line
        (see `gui.widgets.active_panel_label.ActivePanelLabel`), the
        Workbench header's name/panel-layout readout (see
        `gui.widgets.workbench_header.WorkbenchHeader`), plus the Graph
        Library's Update Saved Graph enabled state -- call whenever the
        active panel's identity, its origin Graph, its plotted series/
        datasets, the panel layout, or the active Workbench's name may have
        changed (panel switch, any figure-content edit, graph saved/
        loaded/renamed/duplicated/deleted/updated, undo/redo, project load,
        Workbench switch/rename)."""
        self.plot_page_active_panel_label.refresh(self.figure_model)
        self.series_panel.active_panel_label.refresh(self.figure_model)
        self.figure_size_panel.active_panel_label.refresh(self.figure_model)
        self.figure_layout_panel.active_panel_label.refresh(self.figure_model)
        self.properties_panel.active_panel_label.refresh(self.figure_model)
        self.workbench_header.refresh(self._project.active_workbench.name, self.figure_model)
        self.graph_library_panel.sync_active_panel_state()

    def _on_export_figure(self):
        dialog = ExportFigureDialog(self.figure_model, self.plot_canvas, self)
        dialog.exec()

    def _rerender(self):
        dark_mode = self.figure_model.plot_theme == PlotTheme.DARK
        self.plot_canvas.render(self.figure_model, dark_mode=dark_mode)
        active_axes = self.plot_canvas.active_axes(self.figure_model)
        self.properties_panel.sync_axes_limits(active_axes.get_xlim(), active_axes.get_ylim())
        self.series_panel.update_contrast_warnings(dark_mode)

    # --- Undo/Redo (figure content only -- see gui.undo_manager) -----------

    def _on_figure_content_changed(self) -> None:
        """The single entry point every figure-content mutation (series
        add/remove/style, panel display settings, layout, typography, grid)
        routes through instead of calling `_rerender()` directly -- pure
        navigation (switching the active panel, toggling the Plot Theme)
        must NOT call this, or it would show up as a spurious undo step."""
        self._commit_undo_checkpoint()
        self._set_dirty(True)
        self._rerender()
        self._refresh_active_panel_context()
        self.analysis_panel.refresh()
        # Any series add/remove (via Analysis's own Add/Remove Fit Curve,
        # or a plain delete on the Series page) can change whether a fit
        # curve exists for the active panel's current result -- recompute
        # Add/Remove Fit Curve state fresh rather than trusting whatever
        # it was before this change.
        self._sync_results_to_active_panel()

    def _commit_undo_checkpoint(self) -> None:
        """Push the snapshot captured just before this change (i.e. the
        state a following Undo should restore) and re-baseline
        `_pending_undo_snapshot` to the new current state, ready for the
        *next* checkpoint. Always pushes -- the mutation that just happened
        already changed the model, and figuring out whether it was a no-op
        would mean `==`-comparing snapshots that hold a Dataset/DataFrame,
        which raises (pandas' truth-value-of-a-DataFrame ambiguity)."""
        self._undo_manager.push(self._pending_undo_snapshot)
        self._pending_undo_snapshot = snapshot_figure(self.figure_model, self.dataset_manager)
        self._sync_undo_redo_actions()

    def _on_undo(self) -> None:
        previous = self._undo_manager.undo(self._pending_undo_snapshot)
        if previous is None:
            return
        self._pending_undo_snapshot = previous
        self._restore_figure_snapshot(previous)

    def _on_redo(self) -> None:
        nxt = self._undo_manager.redo(self._pending_undo_snapshot)
        if nxt is None:
            return
        self._pending_undo_snapshot = nxt
        self._restore_figure_snapshot(nxt)

    def _restore_figure_snapshot(self, snapshot: GnoviFigure) -> None:
        """Copy `snapshot`'s data onto the live `self.figure_model` in
        place, rather than replacing the object -- `series_panel`/
        `properties_panel`/`figure_size_panel` all hold a reference to this
        exact GnoviFigure instance from construction, so swapping it out
        would leave them silently pointing at stale state."""
        live = self.figure_model
        live.plot_theme = snapshot.plot_theme
        live.panels = snapshot.panels
        live.layout = snapshot.layout
        live.active_panel_index = min(snapshot.active_panel_index, len(live.panels) - 1)
        live.panel_labels_visible = snapshot.panel_labels_visible
        live.figure_width_in = snapshot.figure_width_in
        live.figure_height_in = snapshot.figure_height_in
        live.aspect_preset = snapshot.aspect_preset
        live.lock_aspect_ratio = snapshot.lock_aspect_ratio
        live.panel_aspect_preset = snapshot.panel_aspect_preset
        live.font_family = snapshot.font_family
        live.base_font_size = snapshot.base_font_size
        live.title_font_size = snapshot.title_font_size
        live.axis_label_font_size = snapshot.axis_label_font_size
        live.tick_label_font_size = snapshot.tick_label_font_size
        live.legend_font_size = snapshot.legend_font_size
        live.grid_linestyle = snapshot.grid_linestyle
        live.grid_linewidth = snapshot.grid_linewidth
        live.grid_alpha = snapshot.grid_alpha
        live.grid_color = snapshot.grid_color
        live.margin_left = snapshot.margin_left
        live.margin_right = snapshot.margin_right
        live.margin_bottom = snapshot.margin_bottom
        live.margin_top = snapshot.margin_top
        live.panel_wspace = snapshot.panel_wspace
        live.panel_hspace = snapshot.panel_hspace

        self.series_panel.refresh()
        self.properties_panel.refresh()
        self.figure_size_panel.refresh()
        self.figure_layout_panel.refresh()
        self.analysis_panel.refresh()
        self._refresh_active_panel_context()
        self._sync_toolbar_panel_controls()
        self._sync_theme_controls()
        self._sync_undo_redo_actions()
        self._sync_results_to_active_panel()  # display sync only -- not itself an undo/redo step
        self._set_dirty(True)
        self._rerender()

    def _sync_undo_redo_actions(self) -> None:
        self.undo_action.setEnabled(self._undo_manager.can_undo)
        self.redo_action.setEnabled(self._undo_manager.can_redo)

    # --- Project persistence (New/Open/Save, dirty-state) -------------------

    def _on_datasets_changed(self) -> None:
        """Dataset import/remove is a project-content change but isn't
        routed through `_on_figure_content_changed` (it doesn't touch the
        figure) or `_on_transformation_applied` (it isn't a Working Data
        transformation on an existing Dataset) -- so it needs its own dirty
        hook, fed by `DatasetPanel.datasets_changed`."""
        self._set_dirty(True)

    def _on_graph_library_changed(self) -> None:
        """Save/Rename/Duplicate/Delete/Update Saved Graph -- only the
        Graph Library's contents changed, not the figure, so no undo
        checkpoint/re-render is needed, just marking the project dirty and
        refreshing the "Graph: ..." context line (Save/Update/Rename/Delete
        can all change what it should show for the active panel) and the
        Update Saved Graph button state."""
        self._set_dirty(True)
        self._refresh_active_panel_context()

    def _on_graph_loaded_into_panel(self) -> None:
        """Load Selected Graph into Active Panel replaced the active
        panel's series/styling -- handled exactly like any other
        figure-content edit (undo checkpoint, dirty, re-render, and the
        Series/Properties panels must reload since the active panel's
        content changed under them)."""
        self.series_panel.refresh()
        self.properties_panel.refresh()
        self._on_figure_content_changed()

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self._sync_window_title()

    def _sync_window_title(self) -> None:
        marker = "*" if self._dirty else ""
        self.setWindowTitle(f"{self._project.name}{marker} — {APP_NAME}")

    def _confirm_discard_unsaved(self) -> bool:
        """True if it's safe to proceed (discard/replace the current
        project) -- either it's not dirty, or the user chose Save/Discard.
        Shared by New Project, Open Project, and `closeEvent`."""
        if not self._dirty:
            return True
        response = QMessageBox.warning(
            self,
            APP_NAME,
            f"'{self._project.name}' has unsaved changes. Save before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if response == QMessageBox.Cancel:
            return False
        if response == QMessageBox.Save:
            return self._on_save_project()
        return True

    def _new_project(self) -> Project:
        """A fresh, empty `Project` (one blank Workbench) -- like
        `Project.new()`, except its default figure's Plot Theme is seeded
        from QSettings' last-used value (see `__init__`) rather than always
        `PlotTheme.LIGHT`. Used for the app's initial project and "New
        Project"; never for Open Project, whose loaded figure's own saved
        theme always governs."""
        project = Project.new()
        project.active_workbench.figure.plot_theme = self._default_new_figure_theme
        return project

    def _on_new_project(self) -> None:
        if not self._confirm_discard_unsaved():
            return
        self._load_project_into_window(self._new_project())

    def _on_open_project(self) -> None:
        if not self._confirm_discard_unsaved():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Gnovi Project (*.gnovi)")
        if not path:
            return
        try:
            project = load_project(path)
        except ProjectIOError as exc:
            QMessageBox.critical(self, "Open Project", str(exc))
            return
        self._load_project_into_window(project)

    def _on_save_project(self) -> bool:
        """Returns True only on an actual successful save -- `False` for a
        cancelled or failed Save As -- so `_confirm_discard_unsaved` never
        treats "the user didn't actually save" as safe to proceed with a
        destructive action (Close/New Project/Open Project)."""
        if self._project.path is None:
            return self._on_save_project_as()
        return self._save_project_to(self._project.path)

    def _on_save_project_as(self) -> bool:
        """Returns False (not saved) if the user cancels the file picker --
        see `_on_save_project`'s docstring for why this must never be
        treated as success."""
        default_name = f"{self._project.name}.gnovi"
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", default_name, "Gnovi Project (*.gnovi)")
        if not path:
            return False
        if not path.lower().endswith(".gnovi"):
            path += ".gnovi"
        return self._save_project_to(path)

    def _save_project_to(self, path) -> bool:
        try:
            save_project(self._project, path)
        except OSError as exc:
            QMessageBox.critical(self, "Save Project", str(exc))
            return False
        self._project.name = Path(path).stem
        self._set_dirty(False)
        return True

    def _activate_workbench(self, workbench: Workbench) -> None:
        """Point every figure-dependent GUI widget at `workbench.figure` and
        swap in `workbench`'s own per-Workbench Undo/Redo history -- the one
        shared retargeting path `_on_workbench_tab_selected`/
        `_on_new_workbench_requested`/`_on_duplicate_workbench_requested`/
        `_on_delete_workbench_requested` (Workbench switch/create/duplicate/
        delete) and `_load_project_into_window` (New/Open Project) all
        funnel through, so there is exactly one place that knows how to
        retarget the GUI to a different Figure. Never marks the project
        dirty and never commits an undo checkpoint itself -- purely
        navigation/retargeting (see each caller for what it does around
        this, e.g. `_load_project_into_window` also resets the per-project
        dataset manager/Graph Library and the dirty flag, none of which
        belongs here since a plain Workbench switch must touch neither)."""
        self._current_workbench_id = workbench.id
        self.figure_model = workbench.figure

        self.series_panel.set_figure(self.figure_model)
        self.properties_panel.set_figure(self.figure_model)
        self.figure_size_panel.set_figure(self.figure_model)
        self.figure_layout_panel.set_figure(self.figure_model)
        self.analysis_panel.set_figure(self.figure_model)
        self.analysis_result_view.set_figure(self.figure_model)

        self._sync_undo_redo_actions()
        self._refresh_active_panel_context()
        self._sync_toolbar_panel_controls()
        self._sync_theme_controls()  # the newly-active Workbench may have a different Plot Theme
        self._sync_results_to_active_panel()  # never leak a previous Workbench's result into this one
        self._rerender()

    def _load_project_into_window(self, project: Project) -> None:
        """The single path New Project and Open Project both funnel
        through: repoint every widget that caches the dataset manager/Graph
        Library/active Figure (see `_activate_workbench` for the Figure
        side), reset every Workbench's Undo/Redo to a fresh per-Workbench
        stack (loading a project must not let the previous project's undo
        history restore its content -- decision: Undo/Redo is scoped to
        the WORKBENCH, not the project on disk), and reset the dirty flag."""
        self._project = project
        self.dataset_manager = project.dataset_manager

        self.dataset_panel.set_manager(self.dataset_manager)
        self.analysis_panel.set_manager(self.dataset_manager)
        self.analysis_result_view.set_manager(self.dataset_manager)
        self.graph_library_panel.set_library(project.graph_library)

        self._undo_managers = {}
        self._pending_snapshots = {}
        self._activate_workbench(project.active_workbench)

        self.workbench_tab_bar.set_workbenches(project.workbenches, project.active_workbench_id)
        self._sync_workbench_menu_state()

        self.preview_model.set_dataframe(None)
        self.data_tools_panel.set_dataset(None)
        self._set_dirty(False)

    # --- Workbenches (switch/create/rename/duplicate/delete) ---------------

    def _sync_workbench_menu_state(self) -> None:
        """Delete Workbench must be disabled whenever exactly one Workbench
        remains -- a `Project` always keeps at least one (see
        `Project.remove_workbench`)."""
        self.delete_workbench_action.setEnabled(len(self._project.workbenches) > 1)

    def _on_workbench_tab_selected(self, workbench_id: str) -> None:
        """Pure navigation -- see `_activate_workbench`'s docstring: never
        marks the project dirty, never commits an undo checkpoint, and the
        Workbench being left behind keeps its own Undo/Redo history exactly
        as it was, ready to resume the moment the user switches back."""
        if workbench_id == self._current_workbench_id:
            return
        workbench = self._project.get_workbench(workbench_id)
        if workbench is None:
            return
        self._project.active_workbench_id = workbench_id
        self._activate_workbench(workbench)

    def _on_new_workbench_requested(self) -> None:
        name = f"Workbench {len(self._project.workbenches) + 1}"
        workbench = Workbench(name=name, figure=GnoviFigure())
        self._project.add_workbench(workbench)
        self._project.active_workbench_id = workbench.id
        self._activate_workbench(workbench)
        self.workbench_tab_bar.set_workbenches(self._project.workbenches, self._project.active_workbench_id)
        self._sync_workbench_menu_state()
        self._set_dirty(True)

    def _on_rename_workbench_requested(self, workbench_id: str) -> None:
        workbench = self._project.get_workbench(workbench_id)
        if workbench is None:
            return
        new_name, ok = QInputDialog.getText(self, "Rename Workbench", "Workbench name:", text=workbench.name)
        new_name = new_name.strip()
        if not ok or not new_name:
            return
        self._project.rename_workbench(workbench_id, new_name)
        self.workbench_tab_bar.set_workbenches(self._project.workbenches, self._project.active_workbench_id)
        self._refresh_active_panel_context()  # the Workbench header shows the new name if it's the active one
        self._set_dirty(True)

    def _on_duplicate_workbench_requested(self, workbench_id: str) -> None:
        copy_workbench = self._project.duplicate_workbench(workbench_id)
        if copy_workbench is None:
            return
        self._project.active_workbench_id = copy_workbench.id
        self._activate_workbench(copy_workbench)
        self.workbench_tab_bar.set_workbenches(self._project.workbenches, self._project.active_workbench_id)
        self._sync_workbench_menu_state()
        self._set_dirty(True)

    def _on_delete_workbench_requested(self, workbench_id: str) -> None:
        # Note: `self.delete_workbench_action` itself is also kept in sync
        # eagerly (see the `_sync_workbench_menu_state()` calls in the
        # other three handlers here, not just `aboutToShow`) so a disabled
        # action can never be `.trigger()`-ed at all -- this guard is the
        # second, always-correct line of defense (e.g. for the tab strip's
        # own context-menu path, which never goes through this QAction).
        workbench = self._project.get_workbench(workbench_id)
        if workbench is None or len(self._project.workbenches) <= 1:
            return
        response = QMessageBox.warning(
            self,
            "Delete Workbench",
            f'Delete Workbench "{workbench.name}"? This cannot be undone.',
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if response != QMessageBox.Yes:
            return
        removed = self._project.remove_workbench(workbench_id)
        if not removed:
            return
        # Drop the deleted Workbench's runtime Undo/Redo state -- it can
        # never be switched back to.
        self._undo_managers.pop(workbench_id, None)
        self._pending_snapshots.pop(workbench_id, None)
        self._activate_workbench(self._project.active_workbench)
        self.workbench_tab_bar.set_workbenches(self._project.workbenches, self._project.active_workbench_id)
        self._sync_workbench_menu_state()
        self._set_dirty(True)

    def closeEvent(self, event) -> None:
        if self._confirm_discard_unsaved():
            event.accept()
        else:
            event.ignore()
