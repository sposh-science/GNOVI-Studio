from __future__ import annotations

from enum import Enum

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gnovi_plot.plotting.backends.matplotlib_backend import apply_figure_layout, render_figure
from gnovi_plot.plotting.figure import GnoviFigure

# Interactive-only chrome -- deliberately never part of
# `matplotlib_backend.render_figure`/`render_panel` (the shared on-screen +
# export drawing code), so neither the active-panel highlight nor the
# reference cursor ever appears in an exported file.
_ACTIVE_PANEL_COLOR = "#2f6fed"
_ACTIVE_PANEL_LINEWIDTH = 2.2
_CURSOR_COLOR = "#8a8f99"


class ReferenceCursorMode(str, Enum):
    """On-screen-only crosshair/reference-line overlay that follows the
    mouse, independent of the fixed-width status-bar coordinate readout
    (see MainWindow._on_mouse_move)."""

    OFF = "off"
    X_LINE = "x"
    Y_LINE = "y"
    CROSSHAIR = "crosshair"


class PlotCanvas(FigureCanvasQTAgg):
    """Interactive multi-panel Matplotlib canvas embedded in the GUI.

    Keeps the Matplotlib Axes grid in sync with `GnoviFigure.layout`, but
    delegates the actual drawing to `plotting.backends.matplotlib_backend`
    so the exact same code path renders on-screen and into exported files.
    On-screen canvas pixel size never determines export resolution -- export
    always builds its own correctly-sized Figure (see `export.figure_export`).

    When `GnoviFigure.lock_aspect_ratio` is set, the canvas stays fully
    responsive to workspace resizing but letterboxes/pillarboxes the actual
    subplot grid within it so the configured `figure_width_in` /
    `figure_height_in` aspect ratio is always what's previewed, rather than
    stretching to fill whatever shape the splitter happens to be -- see
    `_letterbox_rect`.
    """

    def __init__(self, parent=None):
        figure = Figure()
        super().__init__(figure)
        self.setParent(parent)
        self._layout: tuple[int, int] | None = None
        self.axes_list: list = []
        self._last_figure: GnoviFigure | None = None
        self._cursor_mode: ReferenceCursorMode = ReferenceCursorMode.OFF
        self._cursor_artists: list = []
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
        if len(self.axes_list) > 1:
            self._highlight_active_panel(figure.active_panel_index)
        self._last_figure = figure
        self._apply_layout(figure)
        self.draw_idle()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Keep an aspect-locked preview correctly letterboxed as the
        # workspace resizes (splitter drags, window resize) without waiting
        # for the next content change -- cheap (no re-render of series data).
        if self._last_figure is not None:
            self._apply_layout(self._last_figure)
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

    def _highlight_active_panel(self, active_index: int) -> None:
        """A subtle accent-colored border around whichever panel is active,
        so "click a panel to make it active" (see MainWindow._on_canvas_click)
        has visible feedback in a multi-panel layout. Single-panel layouts
        skip this entirely -- there's nothing to distinguish."""
        if not 0 <= active_index < len(self.axes_list):
            return
        for spine in self.axes_list[active_index].spines.values():
            if spine.get_visible():
                spine.set_edgecolor(_ACTIVE_PANEL_COLOR)
                spine.set_linewidth(max(spine.get_linewidth(), _ACTIVE_PANEL_LINEWIDTH))

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
