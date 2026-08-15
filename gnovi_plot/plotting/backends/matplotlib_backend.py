from __future__ import annotations

from collections.abc import Sequence

import matplotlib
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure as MplFigure
from matplotlib.ticker import MultipleLocator

from gnovi_plot.data.numeric import numeric_column, numeric_xy
from gnovi_plot.plotting.figure import GnoviFigure, Panel
from gnovi_plot.plotting.series import PlotSeries, PlotType

"""Pure-Matplotlib rendering for a GnoviFigure -- no Qt/PySide6 dependency.

This is the single place that turns a declarative GnoviFigure/Panel/
PlotSeries description into Matplotlib draw calls, shared by the
interactive `gui.widgets.plot_canvas.PlotCanvas` and the headless
`export.figure_export`, so on-screen and exported figures always come from
the exact same drawing code.

`dark_mode` only recolors chrome (figure/axes background, spines, ticks,
labels, grid, legend frame) -- never `PlotSeries.color`, which stays
explicit so series remain equally visible in both themes. It defaults to
`False` everywhere, including in `export.figure_export`, so the on-screen
GUI theme never silently changes what gets published -- a dark export is
only ever an explicit, separate choice made in the Export Figure dialog.
Kept Qt-free on purpose: this module has its own small dark palette rather
than importing `gui.styles`, so the rendering core never depends on PySide6.
Both palettes below are applied explicitly and unconditionally (not just
the dark one, relying on Matplotlib's own defaults) so a light re-render
after a dark one is always a full, reliable reset rather than leftover
dark-mode state.
"""

# Independent light/dark chrome palettes -- deliberately not imported from
# gui.styles (which depends on PySide6) so this module stays Qt-free; the
# dark values are chosen to read consistently alongside that QSS theme
# without coupling to it. Light values match Matplotlib's own defaults
# (explicit rather than implicit, so switching themes is always reversible).
_LIGHT_CHROME = {
    "figure_bg": "white",
    "axes_bg": "white",
    "text": "black",
    "grid": "#b0b0b0",
    "spine": "black",
    "legend_bg": "white",
    "legend_edge": "#cccccc",
}
_DARK_CHROME = {
    "figure_bg": "#1c1d22",
    "axes_bg": "#25262d",
    "text": "#e7e9ee",
    "grid": "#454854",
    "spine": "#5a5d68",
    "legend_bg": "#25262d",
    "legend_edge": "#5a5d68",
}


def render_figure(axes_list: Sequence[Axes], figure: GnoviFigure, *, dark_mode: bool = False) -> None:
    """Fully redraw `axes_list` (one Matplotlib Axes per `figure.panels`
    entry, same order) from `figure`."""
    figure_bg = _DARK_CHROME["figure_bg"] if dark_mode else _LIGHT_CHROME["figure_bg"]
    rc = {"font.family": figure.font_family} if figure.font_family else {}
    with matplotlib.rc_context(rc):
        for ax, panel in zip(axes_list, figure.panels):
            render_panel(ax, panel, figure, dark_mode=dark_mode)
            if ax.figure is not None:
                ax.figure.set_facecolor(figure_bg)


def apply_figure_layout(
    mpl_figure,
    figure: GnoviFigure,
    *,
    rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> None:
    """Apply `figure`'s stored outer margins and inter-panel spacing
    (`margin_left`/`margin_right`/`margin_bottom`/`margin_top`/
    `panel_wspace`/`panel_hspace`) to `mpl_figure` via `subplots_adjust` --
    the single place both the interactive preview
    (`gui.widgets.plot_canvas.PlotCanvas`) and every exported format
    (`export.figure_export`) position panels within the figure, so on-screen
    and exported layout can never diverge. Deliberately never calls
    Matplotlib's automatic `tight_layout()` -- that only runs as an
    explicit, one-off action (see `compute_tight_layout` below) so a
    figure's layout is fully determined by these six stored numbers, not by
    the rendering environment's font metrics.

    `rect` is the (left, bottom, right, top) figure-fraction box the
    margins are expressed within -- full-bleed (0, 0, 1, 1) unless the
    caller is letterboxing/pillarboxing an aspect-locked preview into a
    differently-shaped canvas (see `PlotCanvas._letterbox_rect`).
    """
    rect_left, rect_bottom, rect_right, rect_top = rect
    width_frac = rect_right - rect_left
    height_frac = rect_top - rect_bottom
    mpl_figure.subplots_adjust(
        left=rect_left + figure.margin_left * width_frac,
        right=rect_left + figure.margin_right * width_frac,
        bottom=rect_bottom + figure.margin_bottom * height_frac,
        top=rect_bottom + figure.margin_top * height_frac,
        wspace=figure.panel_wspace,
        hspace=figure.panel_hspace,
    )


def compute_tight_layout(figure: GnoviFigure) -> dict[str, float]:
    """Compute a one-off "Tight Layout" set of margins/spacing for `figure`
    -- Matplotlib's automatic layout engine run once against a fresh,
    offscreen Figure sized exactly like `figure`'s configured export size
    (`figure_width_in`/`figure_height_in`), so the result matches what
    `export.figure_export` would produce. Returns a dict keyed by the six
    `GnoviFigure` margin/spacing field names; the caller (see
    `gui.widgets.figure_layout_panel`) writes them back onto the figure
    itself -- this never mutates `figure` and is never invoked implicitly
    by a normal render, only by the explicit "Tight Layout" action.
    """
    rows, cols = figure.layout
    mpl_figure = MplFigure(figsize=(figure.figure_width_in, figure.figure_height_in))
    axes_list = list(mpl_figure.subplots(rows, cols, squeeze=False).flat)
    render_figure(axes_list, figure)
    mpl_figure.tight_layout()
    subplot_params = mpl_figure.subplotpars
    return {
        "margin_left": subplot_params.left,
        "margin_right": subplot_params.right,
        "margin_bottom": subplot_params.bottom,
        "margin_top": subplot_params.top,
        "panel_wspace": subplot_params.wspace,
        "panel_hspace": subplot_params.hspace,
    }


class MatplotlibBackend:
    """Thin class wrapper satisfying `plotting.backends.base.PlotBackend`."""

    def render_figure(self, axes_list: Sequence[Axes], figure: GnoviFigure) -> None:
        render_figure(axes_list, figure)


def render_panel(ax: Axes, panel: Panel, figure: GnoviFigure | None = None, *, dark_mode: bool = False) -> None:
    """Fully redraw a single Axes from a Panel. `figure` supplies typography
    fallbacks (used whenever a Panel-level font-size override is None) and
    the shared panel-label visibility flag; it may be omitted for a bare
    single-panel render.
    """
    ax.cla()

    for series in panel.series:
        if series.visible and not series.stale:
            _draw_series(ax, series)

    title_size = panel.title_size or (figure.title_font_size if figure else None)
    label_size = panel.axis_label_size or (figure.axis_label_font_size if figure else None)
    tick_size = panel.tick_label_size or (figure.tick_label_font_size if figure else None)

    ax.set_title(panel.title, fontsize=title_size)
    ax.set_xlabel(panel.xlabel, fontsize=label_size)
    ax.set_ylabel(panel.ylabel, fontsize=label_size)

    ax.set_xscale(panel.xscale)
    ax.set_yscale(panel.yscale)

    if panel.xlim is not None:
        ax.set_xlim(*panel.xlim)
    if panel.ylim is not None:
        ax.set_ylim(*panel.ylim)
    if panel.xlim is None and panel.ylim is None:
        ax.autoscale(enable=True, axis="both")
    elif panel.xlim is None:
        ax.autoscale(enable=True, axis="x")
    elif panel.ylim is None:
        ax.autoscale(enable=True, axis="y")

    if panel.invert_x and ax.get_xlim()[0] < ax.get_xlim()[1]:
        ax.invert_xaxis()
    if panel.invert_y and ax.get_ylim()[0] < ax.get_ylim()[1]:
        ax.invert_yaxis()

    ax.tick_params(
        axis="both",
        which="major",
        direction=panel.tick_direction,
        labelsize=tick_size,
        length=panel.major_tick_length,
        width=panel.major_tick_width,
    )
    if panel.minor_ticks:
        ax.minorticks_on()
    else:
        ax.minorticks_off()
    ax.tick_params(
        axis="both",
        which="minor",
        direction=panel.tick_direction,
        length=panel.minor_tick_length,
        width=panel.minor_tick_width,
    )

    if panel.major_tick_spacing_x:
        ax.xaxis.set_major_locator(MultipleLocator(panel.major_tick_spacing_x))
    if panel.major_tick_spacing_y:
        ax.yaxis.set_major_locator(MultipleLocator(panel.major_tick_spacing_y))
    if panel.minor_ticks and panel.minor_tick_spacing_x:
        ax.xaxis.set_minor_locator(MultipleLocator(panel.minor_tick_spacing_x))
    if panel.minor_ticks and panel.minor_tick_spacing_y:
        ax.yaxis.set_minor_locator(MultipleLocator(panel.minor_tick_spacing_y))

    # ticklabel_format(style="scientific") only works with Matplotlib's
    # ScalarFormatter -- a log-scale axis uses LogFormatterSciNotation
    # instead and raises AttributeError if asked for it, so this is skipped
    # rather than applied unconditionally on every render (root cause of a
    # crash previously reported as an "Unexpected Error" dialog). Scientific
    # notation is a linear-axis-only concept; a log axis already displays
    # its own power-of-ten style.
    if panel.scientific_notation_x and panel.xscale == "linear":
        ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
    if panel.scientific_notation_y and panel.yscale == "linear":
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    for side, visible in (
        ("top", panel.spine_top),
        ("bottom", panel.spine_bottom),
        ("left", panel.spine_left),
        ("right", panel.spine_right),
    ):
        spine = ax.spines[side]
        spine.set_visible(visible)
        spine.set_linewidth(panel.spine_linewidth)

    if panel.grid:
        theme_grid_color = _DARK_CHROME["grid"] if dark_mode else _LIGHT_CHROME["grid"]
        grid_color = (figure.grid_color if figure and figure.grid_color else None) or theme_grid_color
        grid_linestyle = figure.grid_linestyle if figure else "--"
        grid_linewidth = figure.grid_linewidth if figure else 0.8
        grid_alpha = figure.grid_alpha if figure else 0.6
        ax.grid(
            True,
            which=panel.grid_which,
            color=grid_color,
            linestyle=grid_linestyle,
            linewidth=grid_linewidth,
            alpha=grid_alpha,
        )
    else:
        ax.grid(False)

    if panel.legend_visible:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            legend_fontsize = panel.legend_fontsize or (figure.legend_font_size if figure else None)
            legend_kwargs = dict(
                ncols=max(1, panel.legend_ncol),
                frameon=panel.legend_frameon,
                fontsize=legend_fontsize,
                title=panel.legend_title or None,
            )
            # "outside right"/"outside bottom" aren't real Matplotlib `loc`
            # values -- they're placed via `bbox_to_anchor` instead, anchored
            # to an in-bounds `loc` corner so the legend box sits just
            # outside the axes rather than on top of the plotted data.
            # `bbox_inches="tight"` (the default export/preview setting) and
            # the canvas's own `tight_layout` call already account for
            # artists like this that extend past the axes, so it isn't
            # clipped.
            if panel.legend_loc == "outside right":
                legend_kwargs.update(loc="center left", bbox_to_anchor=(1.02, 0.5))
            elif panel.legend_loc == "outside bottom":
                legend_kwargs.update(loc="upper center", bbox_to_anchor=(0.5, -0.15))
            else:
                legend_kwargs["loc"] = panel.legend_loc
            ax.legend(**legend_kwargs)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    if panel.panel_label and figure is not None and figure.panel_labels_visible:
        ax.text(
            0.02,
            0.98,
            panel.panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

    _apply_chrome(ax, dark_mode)


def _apply_chrome(ax: Axes, dark_mode: bool) -> None:
    """Apply the light or dark chrome palette to `ax`. Runs last, after
    every other panel setting has been applied, and only touches colors --
    never visibility/position/scale -- so it can't fight the user's own
    spine, grid, or legend configuration. Applied unconditionally (light
    included) so a theme switch is always a complete, reliable reset."""
    chrome = _DARK_CHROME if dark_mode else _LIGHT_CHROME
    ax.set_facecolor(chrome["axes_bg"])
    ax.title.set_color(chrome["text"])
    ax.xaxis.label.set_color(chrome["text"])
    ax.yaxis.label.set_color(chrome["text"])
    ax.tick_params(axis="both", colors=chrome["text"])
    for spine in ax.spines.values():
        spine.set_color(chrome["spine"])

    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor(chrome["legend_bg"])
        legend.get_frame().set_edgecolor(chrome["legend_edge"])
        for text in legend.get_texts():
            text.set_color(chrome["text"])
        if legend.get_title() is not None:
            legend.get_title().set_color(chrome["text"])


def _series_xy(series: PlotSeries):
    """(x, y) with plotting-only transforms (normalize-to-max, vertical
    offset) applied -- `series.dataframe`/`series.dataset.dataframe` are
    never mutated; only the values handed to Matplotlib are adjusted."""
    x, y = numeric_xy(series.dataframe, series.x_column, series.y_column)
    if series.normalize_to_max:
        peak = y.abs().max()
        if peak:
            y = y / peak
    if series.y_offset:
        y = y + series.y_offset
    return x, y


def _draw_series(ax: Axes, series: PlotSeries) -> None:
    if series.plot_type == PlotType.LINE:
        x, y = _series_xy(series)
        ax.plot(
            x,
            y,
            label=series.label,
            color=series.color,
            linewidth=series.line_width,
            linestyle=series.line_style,
            marker=series.marker or "",
            markersize=series.marker_size,
            markerfacecolor=series.color if series.marker_filled else "none",
            markeredgecolor=series.color,
            markeredgewidth=series.marker_edge_width,
            alpha=series.alpha,
            zorder=series.zorder,
        )
    elif series.plot_type == PlotType.SCATTER:
        x, y = _series_xy(series)
        ax.scatter(
            x,
            y,
            label=series.label,
            facecolors=series.color if series.marker_filled else "none",
            edgecolors=series.color,
            linewidths=series.marker_edge_width,
            marker=series.marker or "o",
            s=series.marker_size**2,
            alpha=series.alpha,
            zorder=series.zorder,
        )
    elif series.plot_type == PlotType.HISTOGRAM:
        values = numeric_column(series.dataframe, series.x_column)
        hist_kwargs = dict(
            bins=series.bins,
            label=series.label,
            color=series.color,
            alpha=series.alpha,
            zorder=series.zorder,
        )
        if series.hist_mode == "percentage":
            hist_kwargs["weights"] = [100.0 / len(values)] * len(values)
        elif series.hist_mode == "cumulative":
            hist_kwargs["cumulative"] = True
        ax.hist(values, **hist_kwargs)


# --- Theme-aware contrast checking (manual series colors only) -------------
#
# Never used to change a color -- only to warn (see
# gui.widgets.plot_series_panel.PlotSeriesPanel.update_contrast_warnings) or
# to compute a fresh replacement for an explicit "Optimize Colors for Theme"
# click. Auto-assigned colors don't need this: they're already picked from a
# theme-appropriate cycle at assignment time (see `plotting.figure.
# theme_color_cycle`).


def _relative_luminance(color: str) -> float:
    """WCAG relative luminance (0.0-1.0) of `color`."""
    def _linearize(channel: float) -> float:
        return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

    r, g, b = (_linearize(c) for c in to_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(color_a: str, color_b: str) -> float:
    """WCAG contrast ratio between two Matplotlib-recognized colors, from
    1.0 (identical) to 21.0 (black vs. white)."""
    luminance_a = _relative_luminance(color_a)
    luminance_b = _relative_luminance(color_b)
    lighter, darker = max(luminance_a, luminance_b), min(luminance_a, luminance_b)
    return (lighter + 0.05) / (darker + 0.05)


# WCAG 2.1's minimum contrast for graphical objects/UI components (3:1) --
# looser than the 4.5:1 body-text threshold, which fits a plotted
# line/marker better than a block of text.
LOW_CONTRAST_THRESHOLD = 3.0


def is_low_contrast(color: str, dark_mode: bool) -> bool:
    """Whether `color` would be hard to see against the current Plot
    Theme's axes background."""
    background = _DARK_CHROME["axes_bg"] if dark_mode else _LIGHT_CHROME["axes_bg"]
    try:
        return contrast_ratio(color, background) < LOW_CONTRAST_THRESHOLD
    except ValueError:
        return False
