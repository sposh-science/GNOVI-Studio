from __future__ import annotations

from collections.abc import Sequence

import matplotlib
from matplotlib.axes import Axes
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
"""


def render_figure(axes_list: Sequence[Axes], figure: GnoviFigure) -> None:
    """Fully redraw `axes_list` (one Matplotlib Axes per `figure.panels`
    entry, same order) from `figure`."""
    rc = {"font.family": figure.font_family} if figure.font_family else {}
    with matplotlib.rc_context(rc):
        for ax, panel in zip(axes_list, figure.panels):
            render_panel(ax, panel, figure)


class MatplotlibBackend:
    """Thin class wrapper satisfying `plotting.backends.base.PlotBackend`."""

    def render_figure(self, axes_list: Sequence[Axes], figure: GnoviFigure) -> None:
        render_figure(axes_list, figure)


def render_panel(ax: Axes, panel: Panel, figure: GnoviFigure | None = None) -> None:
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

    ax.tick_params(axis="both", direction=panel.tick_direction, labelsize=tick_size)
    if panel.minor_ticks:
        ax.minorticks_on()
    else:
        ax.minorticks_off()

    if panel.major_tick_spacing_x:
        ax.xaxis.set_major_locator(MultipleLocator(panel.major_tick_spacing_x))
    if panel.major_tick_spacing_y:
        ax.yaxis.set_major_locator(MultipleLocator(panel.major_tick_spacing_y))
    if panel.minor_ticks and panel.minor_tick_spacing_x:
        ax.xaxis.set_minor_locator(MultipleLocator(panel.minor_tick_spacing_x))
    if panel.minor_ticks and panel.minor_tick_spacing_y:
        ax.yaxis.set_minor_locator(MultipleLocator(panel.minor_tick_spacing_y))

    if panel.scientific_notation_x:
        ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
    if panel.scientific_notation_y:
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

    ax.grid(panel.grid, which=panel.grid_which)

    if panel.legend_visible:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            legend_fontsize = panel.legend_fontsize or (figure.legend_font_size if figure else None)
            ax.legend(
                loc=panel.legend_loc,
                ncols=max(1, panel.legend_ncol),
                frameon=panel.legend_frameon,
                fontsize=legend_fontsize,
                title=panel.legend_title or None,
            )
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
