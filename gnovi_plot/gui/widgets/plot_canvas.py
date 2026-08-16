from __future__ import annotations

from enum import Enum

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from gnovi_plot.gui.styles import ACTIVE_PANEL_BADGE_COLOR
from gnovi_plot.plotting.backends.matplotlib_backend import (
    apply_figure_layout,
    fit_panel_legends_to_axes,
    render_figure,
)
from gnovi_plot.plotting.figure import GnoviFigure

# Interactive-only chrome, kept out of scientific output two different
# ways: the active-panel badge is a separate Qt widget never added to the
# Matplotlib Figure at all (see `_ActivePanelBadge`), so it structurally
# cannot appear in any export. The reference cursor below, in contrast, IS
# drawn as real Matplotlib `Line2D` artists directly on the live Figure
# (see `update_reference_cursor`) -- since that same live Figure is now
# GNOVI Export Figure's own WYSIWYG source (see the class docstring), any
# caller saving it must explicitly call `clear_reference_cursor()` first
# (see `gui.dialogs.export_figure_dialog.ExportFigureDialog
# ._hide_gui_only_overlays` and `gui.main_window._CursorSafeNavigationToolbar`).
_CURSOR_COLOR = "#8a8f99"
_ACTIVE_PANEL_BADGE_MARGIN_PX = 6


class ReferenceCursorMode(str, Enum):
    """On-screen-only crosshair/reference-line overlay that follows the
    mouse, independent of the fixed-width status-bar coordinate readout
    (see MainWindow._on_mouse_move)."""

    OFF = "off"
    X_LINE = "x"
    Y_LINE = "y"
    CROSSHAIR = "crosshair"


class _ActivePanelBadge(QLabel):
    """GUI-only "which panel is active" indicator -- a plain Qt child
    widget floating above the canvas, entirely separate from the
    Matplotlib Figure/Axes. Never a spine/border highlight: scientific
    axes styling (spine color/width, exactly as the user configured it in
    Axes settings) is never touched to indicate selection. Since this is
    a Qt widget drawn by Qt's own compositor on top of the canvas -- not a
    Matplotlib artist -- it structurally cannot appear in `ax.figure`'s own
    rendering, and therefore never appears in Matplotlib's toolbar "Save",
    nor in any PNG/TIFF/SVG/PDF export (those all render a fresh, separate
    `Figure` -- see `export.figure_export` -- that this widget was never
    part of to begin with)."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            f"background-color: {ACTIVE_PANEL_BADGE_COLOR}; color: white; "
            "font-weight: 600; font-size: 8pt; padding: 2px 7px; border-radius: 8px;"
        )
        self.hide()


class PlotCanvas(FigureCanvasQTAgg):
    """Interactive multi-panel Matplotlib canvas embedded in the GUI.

    Keeps the Matplotlib Axes grid in sync with `GnoviFigure.layout`, but
    delegates the actual drawing to `plotting.backends.matplotlib_backend`
    so the exact same code path renders on-screen and into exported files.
    `self.figure` -- this canvas's own live Matplotlib Figure -- is also
    the WYSIWYG export source: `gui.dialogs.export_figure_dialog` saves it
    directly via `export.figure_export.export_live_figure` for "Complete
    Figure"/"Active Panel" export, exactly like the Matplotlib navigation
    toolbar's own "Save" button does, rather than reconstructing a second
    Figure. (`export.figure_export.export_figure` still builds its own
    fresh, correctly-sized Figure for headless/scripted use -- a
    `GnoviFigure` model with no live canvas involved at all.)

    When `GnoviFigure.lock_aspect_ratio` is set, the canvas stays fully
    responsive to workspace resizing but letterboxes/pillarboxes the actual
    subplot grid within it so the configured `figure_width_in` /
    `figure_height_in` aspect ratio is always what's previewed, rather than
    stretching to fill whatever shape the splitter happens to be -- see
    `_letterbox_rect`.

    After every render (and on every resize), also runs
    `matplotlib_backend.fit_panel_legends_to_axes` -- a preview-only pass
    that shrinks a panel's on-screen legend (padding, then font size, down
    to a readability floor) if it would otherwise overflow into a
    neighboring panel, WITHOUT touching the configured
    `Panel.legend_fontsize`/`GnoviFigure.legend_font_size`. Since GNOVI's
    own Export Figure now saves this exact live Figure, an exported legend
    reflects whatever's genuinely on screen at export time (true WYSIWYG,
    matching the Matplotlib toolbar's own Save) -- the headless
    `export.figure_export.export_figure` path (no live canvas) never
    shrinks anything, always using the exact configured size.

    Which panel is active is shown by `_ActivePanelBadge`, a plain Qt
    overlay widget positioned over the active Axes -- never a spine/border
    color change on the Axes themselves (scientific axes styling and GUI
    selection state stay completely separate). Being a Qt widget, not a
    Matplotlib artist, it structurally can't appear in Matplotlib's
    toolbar "Save" or any export.
    """

    def __init__(self, parent=None):
        figure = Figure()
        super().__init__(figure)
        self.setParent(parent)
        self._layout: tuple[int, int] | None = None
        self.axes_list: list = []
        self._last_figure: GnoviFigure | None = None
        self._last_dark_mode: bool = False
        self._cursor_mode: ReferenceCursorMode = ReferenceCursorMode.OFF
        self._cursor_artists: list = []
        self._active_panel_badge = _ActivePanelBadge(self)
        self._ensure_layout((1, 1))

    def _ensure_layout(self, layout: tuple[int, int]) -> None:
        if layout == self._layout:
            return
        rows, cols = layout
        self.figure.clear()
        self.axes_list = list(self.figure.subplots(rows, cols, squeeze=False).flat)
        self._layout = layout

    @property
    def axes(self):
        """Backward-compatible alias for the first panel's Axes."""
        return self.axes_list[0]

    def active_axes(self, figure: GnoviFigure):
        """The Axes for `figure`'s currently active panel."""
        return self.axes_list[figure.active_panel_index]

    def panel_index_for_axes(self, ax) -> int | None:
        """The panel index `ax` belongs to (e.g. from a click event's
        `event.inaxes`), or None if `ax` isn't one of ours."""
        try:
            return self.axes_list.index(ax)
        except ValueError:
            return None

    def render(self, figure: GnoviFigure, *, dark_mode: bool = False) -> None:
        """Fully redraw every panel from `figure`. Reading a series' data
        never mutates the underlying Dataset.dataframe (see
        plotting.backends.matplotlib_backend). `dark_mode` only affects this
        on-screen preview -- exported figures resolve their own background
        independently (see `export.figure_export`)."""
        self._ensure_layout(figure.layout)
        render_figure(self.axes_list, figure, dark_mode=dark_mode)
        # render_panel() calls ax.cla() on every panel above, which already
        # discarded any previously-drawn reference-cursor artists -- drop
        # our stale references too (without calling .remove() on them: cla()
        # already did, and doing so again raises).
        self._cursor_artists = []
        self._last_figure = figure
        self._last_dark_mode = dark_mode
        self._apply_layout(figure)
        self.draw()  # synchronous: fit_panel_legends_to_axes/badge positioning need real extents
        fit_panel_legends_to_axes(self.axes_list, figure, dark_mode=dark_mode)
        self._update_active_panel_badge(figure)
        self.draw_idle()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Keep an aspect-locked preview correctly letterboxed, any panel's
        # on-screen legend correctly fitted, and the active-panel badge
        # correctly positioned, as the workspace resizes (splitter drags,
        # drawer collapse/expand, window resize) -- without waiting for the
        # next content change.
        if self._last_figure is not None:
            self._apply_layout(self._last_figure)
            self.draw()
            fit_panel_legends_to_axes(self.axes_list, self._last_figure, dark_mode=self._last_dark_mode)
            self._update_active_panel_badge(self._last_figure)
            self.draw_idle()

    def _apply_layout(self, figure: GnoviFigure) -> None:
        apply_figure_layout(self.figure, figure, rect=self._letterbox_rect(figure))

    def _letterbox_rect(self, figure: GnoviFigure) -> tuple[float, float, float, float]:
        """The (left, bottom, right, top) figure-fraction box the subplot
        grid's margins (see `apply_figure_layout`) should be expressed
        within. Full-bleed (0, 0, 1, 1) --
        today's fully-responsive stretch-to-fill behavior -- unless the
        figure's aspect ratio is locked, in which case it shrinks to the
        centered sub-rectangle matching `figure_width_in`/`figure_height_in`
        for the canvas's current on-screen pixel size."""
        if not figure.lock_aspect_ratio or figure.figure_width_in <= 0 or figure.figure_height_in <= 0:
            return (0.0, 0.0, 1.0, 1.0)

        canvas_w = max(self.width(), 1)
        canvas_h = max(self.height(), 1)
        target_ratio = figure.figure_width_in / figure.figure_height_in
        canvas_ratio = canvas_w / canvas_h

        if canvas_ratio > target_ratio:
            # Canvas wider than the target -- pillarbox (margins on the sides).
            content_w_frac = target_ratio / canvas_ratio
            content_h_frac = 1.0
        else:
            # Canvas taller than the target -- letterbox (margins top/bottom).
            content_w_frac = 1.0
            content_h_frac = canvas_ratio / target_ratio

        left = (1.0 - content_w_frac) / 2
        bottom = (1.0 - content_h_frac) / 2
        return (left, bottom, left + content_w_frac, bottom + content_h_frac)

    def _update_active_panel_badge(self, figure: GnoviFigure) -> None:
        """Move/relabel the GUI-only active-panel badge (see
        `_ActivePanelBadge`) to sit just OUTSIDE the active panel's Axes,
        above its top-right corner -- so "click a panel to make it active"
        (see `MainWindow._on_canvas_click`) has visible feedback without
        ever occupying scientific plotting space or touching the Axes/
        spines themselves. Shown for every layout, including a single panel
        (1x1), per the badge's own purpose: "which panel is active" stays
        meaningful even with just one. Recomputed from the active Axes'
        actual current on-screen position on every call, so it tracks any
        cause of that position changing (resize, drawer/bottom-panel
        resize, layout change, Figure/Panel Aspect Ratio change, margin/
        spacing change) automatically -- see `render()`/`resizeEvent()`,
        the only two callers.
        """
        index = figure.active_panel_index
        if not 0 <= index < len(self.axes_list):
            self._active_panel_badge.hide()
            return
        ax = self.axes_list[index]
        renderer = self.get_renderer()
        if renderer is None:
            self._active_panel_badge.hide()
            return
        ax_bbox = ax.get_window_extent(renderer=renderer)

        self._active_panel_badge.setText(f"P{index + 1} · ACTIVE")
        self._active_panel_badge.adjustSize()
        badge_w = self._active_panel_badge.width()
        badge_h = self._active_panel_badge.height()

        # Matplotlib's window extent is bottom-left-origin (y grows
        # upward); Qt widget coordinates are top-left-origin (y grows
        # downward). Right-aligned with the panel's own right edge, resting
        # just above its top edge.
        x = int(ax_bbox.x1) - badge_w
        y = int(self.height() - ax_bbox.y1) - badge_h - _ACTIVE_PANEL_BADGE_MARGIN_PX
        # Clamped to stay fully within the canvas widget -- this is a
        # GUI-only overlay, so it must never be allowed to reshuffle panel
        # geometry (margins/spacing) just to make room for itself; if a
        # panel sits flush against the figure's own edge, the badge simply
        # sits as close as it can rather than pushing anything.
        x = max(0, min(x, self.width() - badge_w))
        y = max(0, y)
        self._active_panel_badge.move(x, y)
        self._active_panel_badge.raise_()
        self._active_panel_badge.show()

    # --- Reference cursor (Off / X line / Y line / Crosshair) --------------

    def set_cursor_mode(self, mode: ReferenceCursorMode) -> None:
        self._cursor_mode = mode
        self._clear_cursor_artists()
        self.draw_idle()

    def update_reference_cursor(self, ax, xdata: float | None, ydata: float | None) -> None:
        """Redraw the reference-cursor overlay at `(xdata, ydata)` in `ax`."""
        self._clear_cursor_artists()
        if self._cursor_mode == ReferenceCursorMode.OFF:
            return
        if self._cursor_mode in (ReferenceCursorMode.X_LINE, ReferenceCursorMode.CROSSHAIR) and xdata is not None:
            self._cursor_artists.append(
                ax.axvline(xdata, color=_CURSOR_COLOR, linewidth=0.8, linestyle="--", zorder=1000)
            )
        if self._cursor_mode in (ReferenceCursorMode.Y_LINE, ReferenceCursorMode.CROSSHAIR) and ydata is not None:
            self._cursor_artists.append(
                ax.axhline(ydata, color=_CURSOR_COLOR, linewidth=0.8, linestyle="--", zorder=1000)
            )
        self.draw_idle()

    def clear_reference_cursor(self) -> None:
        if self._cursor_artists:
            self._clear_cursor_artists()
            self.draw_idle()

    def _clear_cursor_artists(self) -> None:
        for artist in self._cursor_artists:
            artist.remove()
        self._cursor_artists = []
