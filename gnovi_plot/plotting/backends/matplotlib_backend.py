from __future__ import annotations

from collections.abc import Sequence

import matplotlib
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.figure import Figure as MplFigure
from matplotlib.ticker import MultipleLocator

from gnovi_plot.data.numeric import numeric_column, numeric_xy, numeric_xyz
from gnovi_plot.plotting.figure import GnoviFigure, Panel, Panel3D
from gnovi_plot.plotting.series import PlotSeries, PlotType
from gnovi_plot.plotting.series3d import Plot3DType, Series3D
from gnovi_plot.plotting.units import panel_box_aspect

# mpl_toolkits.mplot3d must be imported once for Matplotlib to register the
# "3d" projection with add_subplot/add_axes -- never used directly by name
# below (Axes3D objects are created via `projection="3d"`, not this class),
# but the import's side effect is required.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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


def render_panel_with_figure_background(
    ax: Axes, panel: Panel | Panel3D, figure: GnoviFigure, *, dark_mode: bool = False
) -> None:
    """Render `panel` into `ax` (via `render_panel`/`render_panel_3d`,
    dispatched on `panel`'s type -- `ax` must already be the matching
    projection, see `build_projection_aware_axes`) and apply `figure`'s own
    background chrome to `ax`'s parent Matplotlib Figure -- the shared
    per-panel body `render_figure`'s loop below uses. Also called directly
    (never through that loop, which assumes `len(axes_list) ==
    len(figure.panels)`) by `gui.widgets.plot_canvas.PlotCanvas.render`'s
    Focus-mode path, where exactly one Axes renders one Panel regardless
    of how many panels `figure` actually has -- so the single-panel
    rendering logic (2D or 3D) and this figure-background step never need
    a second implementation for that case. This one dispatch point is also
    what gives Focus mode, Extract, and both export paths 3D support for
    free: none of them render panels any other way.
    """
    figure_bg = _DARK_CHROME["figure_bg"] if dark_mode else _LIGHT_CHROME["figure_bg"]
    rc = {"font.family": figure.font_family} if figure.font_family else {}
    with matplotlib.rc_context(rc):
        if isinstance(panel, Panel3D):
            render_panel_3d(ax, panel, figure, dark_mode=dark_mode)
        else:
            render_panel(ax, panel, figure, dark_mode=dark_mode)
        if ax.figure is not None:
            ax.figure.set_facecolor(figure_bg)


def render_figure(axes_list: Sequence[Axes], figure: GnoviFigure, *, dark_mode: bool = False) -> None:
    """Fully redraw `axes_list` (one Matplotlib Axes per `figure.panels`
    entry, same order) from `figure`."""
    for ax, panel in zip(axes_list, figure.panels):
        render_panel_with_figure_background(ax, panel, figure, dark_mode=dark_mode)


def build_projection_aware_axes(mpl_figure, rows: int, cols: int, panels: Sequence[Panel | Panel3D]) -> list[Axes]:
    """One Axes per `(rows, cols)` grid cell, each created with
    `projection="3d"` exactly where `panels[i]` is a `Panel3D`, an ordinary
    2D Axes otherwise -- the projection choice is made PER CELL, never for
    the whole Figure, so a mixed 2D/3D layout is possible at all. This is
    the one place every Axes grid in this app gets built from a
    `GnoviFigure`'s panels: `gui.widgets.plot_canvas.PlotCanvas.
    _ensure_layout` (the interactive canvas), `export.figure_export.
    export_figure` (headless export), and `compute_tight_layout` below all
    call this, so a mixed Figure's on-screen preview and every exported
    format build an identical Axes grid, and Focus/Extract/Export/full-
    Figure-export never need separate 3D-aware axes-creation logic of
    their own.

    A plain `mpl_figure.subplots(rows, cols, squeeze=False)` (what this
    replaces) cannot do this -- it creates a uniformly-2D grid in one call,
    with no per-cell projection control; `Axes3D` also cannot be produced
    by reconfiguring an existing 2D Axes after creation (`projection` is
    only a creation-time argument to `add_subplot`), so this loop -- not a
    post-hoc conversion step -- is the only way to build a mixed grid.
    """
    return [
        mpl_figure.add_subplot(rows, cols, i + 1, projection="3d" if isinstance(panel, Panel3D) else None)
        for i, panel in enumerate(panels)
    ]


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
    axes_list = build_projection_aware_axes(mpl_figure, rows, cols, figure.panels)
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


# "outside right"/"outside bottom" aren't real Matplotlib `loc` values --
# used by both `_legend_kwargs` below and `fit_panel_legends_to_axes`
# (which skips them: they're deliberately placed outside the axes, so
# preview-only fitting must never pull them back inside).
_LEGEND_OUTSIDE_LOCS = {"outside right", "outside bottom"}


def _legend_kwargs(panel: Panel, figure: GnoviFigure | None, *, fontsize: float | None = None) -> dict:
    """Build the kwargs `ax.legend(...)` is called with for `panel` -- the
    single place this is constructed, shared by `render_panel` (used by
    both the interactive preview and `export.figure_export`, unconditionally
    at the configured size) and the preview-only `fit_panel_legends_to_axes`
    below, so a fitted/shrunk preview legend is otherwise byte-identical to
    what an unconstrained render (or export) would produce -- same loc/
    bbox_to_anchor/ncols/frameon/title logic either way. `fontsize`
    overrides the configured `panel.legend_fontsize`/`figure.legend_font_size`
    when given -- used only by the preview-only fit pass, which never writes
    either of those back.
    """
    resolved_fontsize = (
        fontsize if fontsize is not None else panel.legend_fontsize or (figure.legend_font_size if figure else None)
    )
    kwargs = dict(
        ncols=max(1, panel.legend_ncol),
        frameon=panel.legend_frameon,
        fontsize=resolved_fontsize,
        title=panel.legend_title or None,
    )
    # Placed via `bbox_to_anchor` instead of a real Matplotlib `loc`,
    # anchored to an in-bounds `loc` corner so the legend box sits just
    # outside the axes rather than on top of the plotted data.
    # `bbox_inches="tight"` (the default export/preview setting) and the
    # canvas's own `tight_layout` call already account for artists like
    # this that extend past the axes, so it isn't clipped.
    if panel.legend_loc == "outside right":
        kwargs.update(loc="center left", bbox_to_anchor=(1.02, 0.5))
    elif panel.legend_loc == "outside bottom":
        kwargs.update(loc="upper center", bbox_to_anchor=(0.5, -0.15))
    else:
        kwargs["loc"] = panel.legend_loc
    return kwargs


def render_panel(ax: Axes, panel: Panel, figure: GnoviFigure | None = None, *, dark_mode: bool = False) -> None:
    """Fully redraw a single Axes from a Panel. `figure` supplies typography
    fallbacks (used whenever a Panel-level font-size override is None) and
    the shared panel-label visibility flag; it may be omitted for a bare
    single-panel render.
    """
    ax.cla()

    # Panel Aspect Ratio: the physical shape of this Axes' box only -- see
    # `GnoviFigure.panel_aspect_preset`'s docstring. `set_box_aspect` (not
    # `set_aspect("equal")`) so this never forces equal X/Y data-unit
    # scaling; it's fully independent of xlim/ylim/autoscale below. `None`
    # ("Auto") clears any box-aspect constraint, restoring Matplotlib's
    # ordinary layout behavior.
    ax.set_box_aspect(panel_box_aspect(figure.panel_aspect_preset) if figure else None)

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
            ax.legend(**_legend_kwargs(panel, figure))
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


def render_panel_3d(ax, panel: Panel3D, figure: GnoviFigure | None = None, *, dark_mode: bool = False) -> None:
    """Fully redraw a single `Axes3D` from a `Panel3D` -- the 3D sibling of
    `render_panel` above, kept as a separate function rather than a large
    conditional block inside it: the two operate on fundamentally
    different Matplotlib APIs (a third axis, `view_init` instead of
    `xlim`/`ylim` autoscale semantics, pane-based background instead of
    `set_facecolor`/spines -- see `_apply_chrome_3d`), so a shared body
    would mostly be `if`/`else` branches with no actual logic in common
    beyond the few lines duplicated here (title/label font-size fallback,
    the same pattern `render_panel` uses).

    `ax` must already be a genuine `Axes3D` (`projection="3d"` at creation
    -- see `build_projection_aware_axes`); this function never creates or
    converts an Axes itself.
    """
    ax.cla()

    for series in panel.series:
        if series.visible and not series.stale:
            _draw_series_3d(ax, series)

    title_size = figure.title_font_size if figure else None
    label_size = figure.axis_label_font_size if figure else None
    tick_size = figure.tick_label_font_size if figure else None

    ax.set_title(panel.title, fontsize=title_size)
    ax.set_xlabel(panel.x_label, fontsize=label_size)
    ax.set_ylabel(panel.y_label, fontsize=label_size)
    ax.set_zlabel(panel.z_label, fontsize=label_size)

    if panel.xlim is not None:
        ax.set_xlim(*panel.xlim)
    if panel.ylim is not None:
        ax.set_ylim(*panel.ylim)
    if panel.zlim is not None:
        ax.set_zlim(*panel.zlim)

    # Scientific/data aspect -- see `Panel3D.aspect_mode`'s own docstring
    # for why this is `set_aspect` (data-unit scaling), never
    # `set_box_aspect` (physical panel shape, an entirely separate,
    # untouched concern). `adjustable="datalim"` changes the x/y/z data
    # limits to achieve the requested aspect rather than warping the
    # panel's own box shape -- "auto" is a real, valid `set_aspect` value
    # too (not just the absence of a call), so this is always called
    # unconditionally for a uniform code path.
    ax.set_aspect(panel.aspect_mode, adjustable="datalim")

    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    if panel.major_tick_spacing_x:
        ax.xaxis.set_major_locator(MultipleLocator(panel.major_tick_spacing_x))
    if panel.major_tick_spacing_y:
        ax.yaxis.set_major_locator(MultipleLocator(panel.major_tick_spacing_y))
    if panel.major_tick_spacing_z:
        ax.zaxis.set_major_locator(MultipleLocator(panel.major_tick_spacing_z))
    if panel.minor_tick_spacing_x:
        ax.xaxis.set_minor_locator(MultipleLocator(panel.minor_tick_spacing_x))
    if panel.minor_tick_spacing_y:
        ax.yaxis.set_minor_locator(MultipleLocator(panel.minor_tick_spacing_y))
    if panel.minor_tick_spacing_z:
        ax.zaxis.set_minor_locator(MultipleLocator(panel.minor_tick_spacing_z))

    ax.grid(panel.grid)
    _apply_3d_grid_style(ax, panel, dark_mode)

    # The deterministic default/exported camera -- see `Panel3D`'s own
    # docstring for why only elev/azim are modeled, and why interactive
    # mouse rotation (Matplotlib's own built-in 3D navigation, active on
    # any live Axes3D with no GNOVI code involved) never writes back here:
    # every render starts from this same stored view, exactly like a 2D
    # Panel's stored xlim/ylim (or "auto" if never explicitly set).
    ax.view_init(elev=panel.elevation, azim=panel.azimuth)

    # Deliberately minimal (see `Panel3D`'s own docstring): `loc`/`ncols`/
    # `frameon` + figure-level font size, no title/outside-bbox/per-panel
    # font-size-override customization -- `Axes3D` doesn't override
    # `legend()` (confirmed directly against the installed Matplotlib
    # version), so this is the exact same public `Axes.legend(...)` call
    # `render_panel`'s own 2D legend block uses, just a smaller kwarg
    # subset. `ax.get_legend_handles_labels()` only ever contains artists
    # from series actually drawn above (visible, non-stale), so a hidden/
    # stale series never gets a legend entry, the same "only what's
    # actually rendered" semantics `render_panel`'s own legend block
    # already has.
    if panel.legend_visible:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                loc=panel.legend_loc,
                ncols=max(1, panel.legend_ncol),
                frameon=panel.legend_frameon,
                fontsize=figure.legend_font_size if figure else None,
            )
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    if panel.panel_label and figure is not None and figure.panel_labels_visible:
        ax.text2D(
            0.02,
            0.98,
            panel.panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

    _apply_chrome_3d(ax, panel, dark_mode)


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


def _apply_chrome_3d(ax, panel: Panel3D, dark_mode: bool) -> None:
    """3D counterpart of `_apply_chrome` -- same palette dicts (never a
    duplicated theme definition, just a separate application), but a
    genuinely different Matplotlib API surface: `Axes3D` has three pane
    backgrounds (`xaxis.pane`/`yaxis.pane`/`zaxis.pane`, the 3D equivalent
    of a 2D Axes' single `set_facecolor`) and a third axis/label, and its
    `.spines` (present, inherited from the 2D Axes base) aren't the
    meaningful visual boundary a 3D view actually shows -- the panes are,
    so those are what get colored/outlined here instead of spines. Legend
    theming (frame/text colors) mirrors `_apply_chrome`'s own block
    exactly, now that `Panel3D` supports a legend.

    Pane visibility/color/alpha are `panel`-driven (see `Panel3D.
    pane_visible`/`.pane_color`/`.pane_alpha`'s own docstring) via
    `Axis.pane`'s fully public, stable `Polygon` API (`set_visible`/
    `set_facecolor`/`set_alpha`) -- no private-API concerns here, unlike
    grid (see `_apply_3d_grid_style`). `pane_color=None` falls back to the
    current Plot Theme's background exactly like before this field
    existed."""
    chrome = _DARK_CHROME if dark_mode else _LIGHT_CHROME
    ax.set_facecolor(chrome["axes_bg"])
    ax.title.set_color(chrome["text"])
    pane_color = panel.pane_color or chrome["axes_bg"]
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(chrome["text"])
        axis.pane.set_visible(panel.pane_visible)
        axis.pane.set_facecolor(pane_color)
        axis.pane.set_alpha(panel.pane_alpha)
        axis.pane.set_edgecolor(chrome["spine"])
    ax.tick_params(axis="both", colors=chrome["text"])

    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor(chrome["legend_bg"])
        legend.get_frame().set_edgecolor(chrome["legend_edge"])
        for text in legend.get_texts():
            text.set_color(chrome["text"])
        if legend.get_title() is not None:
            legend.get_title().set_color(chrome["text"])


def _apply_3d_grid_style(ax, panel: Panel3D, dark_mode: bool) -> None:
    """Apply `Panel3D`'s grid style (linestyle/linewidth/alpha/color) to a
    live `Axes3D`.

    WHY THIS TOUCHES A PRIVATE MATPLOTLIB API: verified directly against
    the installed Matplotlib version (see this milestone's own
    architecture-inspection notes) that there is no public way to do this.
    `Axes3D.grid(visible, **kwargs)` is overridden to ignore every style
    kwarg entirely (its own docstring: "does not behave the same as
    `.axes.Axes.grid`, but is intended to eventually support that
    behavior"). The seemingly-public `ax.get_xgridlines()`/`get_ygridlines`/
    `get_zgridlines()` return real, fully-settable `Line2D` objects, but
    styling them has ZERO effect on the actual rendered/exported grid --
    confirmed experimentally (set + re-export, before and after an explicit
    `draw()`, style never appeared in the output). The ONLY mechanism that
    actually controls rendered 3D grid line appearance in this Matplotlib
    version is each axis's private `_axinfo["grid"]` dict
    (`mpl_toolkits.mplot3d.axis3d.Axis.draw` reads `_axinfo["grid"]
    ["color"/"linewidth"/"linestyle"]` directly every render) -- confirmed
    experimentally the opposite way too (mutating it and re-exporting DID
    change the rendered output).

    CONTAINMENT: this is the ONE function in all of GNOVI that touches
    `_axinfo` -- every other 3D grid/pane/legend/aspect/tick control in
    this milestone uses fully public Matplotlib API. If a future
    Matplotlib release restructures `_axinfo`, this is the only place that
    needs to change.

    Alpha must be baked into the RGBA color tuple (`to_rgba(color, alpha)`)
    rather than passed as a separate `alpha` key -- confirmed by reading
    `axis3d.py`'s own draw code: it calls `set_color`/`set_linewidth`/
    `set_linestyle` on the grid line collection, never `set_alpha`, so a
    bare `alpha` dict key is silently ignored.

    GRACEFUL FALLBACK: wrapped in a broad `except` -- if `_axinfo`'s shape
    changes in a future Matplotlib release (or an invalid color string
    somehow reaches here), this silently leaves whatever grid Matplotlib's
    own `ax.grid(panel.grid)` (called by the caller, unconditionally, via
    the public API) already produced -- never crashes the render, never
    corrupts `panel` (this function only ever READS it), and never leaves
    a partially-styled axis (X/Y/Z are looped together; if one axis's
    dict write fails they all abort before any partial mutation matters,
    since each axis's dict is independent of the others' success)."""
    if not panel.grid:
        return
    theme_grid_color = _DARK_CHROME["grid"] if dark_mode else _LIGHT_CHROME["grid"]
    color = panel.grid_color or theme_grid_color
    try:
        rgba = to_rgba(color, panel.grid_alpha)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"].update(
                color=rgba,
                linewidth=panel.grid_linewidth,
                linestyle=panel.grid_linestyle,
            )
    except (AttributeError, KeyError, TypeError, ValueError):
        pass


# --- Preview-only legend fitting (screen only -- never export) -------------
#
# `export.figure_export` renders through `render_figure`/`render_panel`
# alone and never calls anything below -- exported legends always use the
# exact configured `Panel.legend_fontsize`/`GnoviFigure.legend_font_size`,
# unconditionally. `gui.widgets.plot_canvas.PlotCanvas` is the only caller,
# after every `render_figure()` call and on every resize -- see its
# docstring. Nothing here ever writes to `Panel.legend_fontsize`/
# `GnoviFigure.legend_font_size` (the stored, reproducible model); it only
# mutates the already-rendered Matplotlib `Legend` artist in place.

_LEGEND_FIT_MIN_FONTSIZE = 6.0  # readability floor, in points
_LEGEND_FIT_CONVERGENCE_PT = 0.25  # binary-search stop threshold, in points
_LEGEND_FIT_TOLERANCE_PX = 2.0  # containment slack, in display pixels
# Tightened padding/handle spacing tried before (and kept alongside) any
# font-size reduction -- "reduce legend padding/handle spacing if useful
# before shrinking too aggressively".
_LEGEND_FIT_TIGHT_SPACING = dict(
    handlelength=1.2,
    handletextpad=0.4,
    labelspacing=0.3,
    borderaxespad=0.3,
    borderpad=0.3,
    columnspacing=0.8,
)


def fit_panel_legends_to_axes(
    axes_list: Sequence[Axes], figure: GnoviFigure, *, dark_mode: bool = False, panels: Sequence[Panel] | None = None
) -> None:
    """Preview-only post-process: for each panel whose legend overflows its
    own Axes' bounding box, shrink it (tightened padding first, then font
    size down to a ~6pt floor) until it fits, or leave it at the floor if it
    still doesn't -- so a legend never spills into a neighboring panel in a
    crowded multi-panel layout. A legend already fitting at the configured
    size (fresh from `render_panel`, or a previously-shrunk one whose panel
    grew back large enough) is restored/left at that exact configured size --
    never held smaller than necessary. Skips "outside right"/"outside
    bottom" legends, deliberately placed outside the axes. Requires
    `axes_list` to already be attached to a live, drawable canvas.

    `panels` defaults to `figure.panels` (`zip`ped against `axes_list`,
    same order, the normal multi-panel case) -- pass it explicitly when
    `axes_list` doesn't correspond 1:1 with `figure.panels` in that order,
    e.g. `gui.widgets.plot_canvas.PlotCanvas.render`'s Focus-mode path,
    where a single Axes shows one arbitrary Panel from the middle of
    `figure.panels`. This also means Focus mode's lone panel is evaluated
    only against its own Axes' bounding box -- never shrunk to avoid
    neighboring panels that aren't being rendered at all.
    """
    if not axes_list:
        return
    canvas = axes_list[0].figure.canvas
    if canvas is None:
        return
    canvas.draw()
    renderer = canvas.get_renderer()
    if renderer is None:
        return

    for ax, panel in zip(axes_list, panels if panels is not None else figure.panels):
        if not panel.legend_visible or panel.legend_loc in _LEGEND_OUTSIDE_LOCS:
            continue
        legend = ax.get_legend()
        if legend is None:
            continue
        # A legend already shrunk by a previous fit pass (flagged below)
        # must be re-evaluated from the *configured* size first -- otherwise
        # a panel that grows back larger would never restore its legend to
        # the configured size, since a smaller-than-needed legend never
        # "overflows". A pristine legend (fresh from `render_panel`, or
        # already restored to the configured size by a previous pass that
        # found it fit) needs no such reset.
        if getattr(legend, "_gnovi_legend_fit_shrunk", False):
            _reset_panel_legend_to_configured(ax, panel, figure, dark_mode=dark_mode)
            canvas.draw()
            renderer = canvas.get_renderer()
        if not _legend_overflows_axes(ax, renderer, tolerance_px=_LEGEND_FIT_TOLERANCE_PX):
            continue
        _shrink_panel_legend_to_fit(ax, panel, figure, canvas, dark_mode=dark_mode)


def _legend_overflows_axes(ax: Axes, renderer, *, tolerance_px: float) -> bool:
    legend = ax.get_legend()
    if legend is None:
        return False
    ax_bbox = ax.get_window_extent(renderer=renderer)
    legend_bbox = legend.get_window_extent(renderer=renderer)
    return (
        legend_bbox.x0 < ax_bbox.x0 - tolerance_px
        or legend_bbox.x1 > ax_bbox.x1 + tolerance_px
        or legend_bbox.y0 < ax_bbox.y0 - tolerance_px
        or legend_bbox.y1 > ax_bbox.y1 + tolerance_px
    )


def _reset_panel_legend_to_configured(ax: Axes, panel: Panel, figure: GnoviFigure, *, dark_mode: bool) -> None:
    """Recreate `ax`'s legend exactly as `render_panel` would (configured
    font size, normal spacing, no `_gnovi_legend_fit_shrunk` flag) --
    undoes a previous fit-pass shrink so the panel growing back larger can
    be correctly re-evaluated from the true baseline."""
    existing = ax.get_legend()
    if existing is not None:
        existing.remove()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, **_legend_kwargs(panel, figure))
    _apply_chrome(ax, dark_mode)


def _shrink_panel_legend_to_fit(ax: Axes, panel: Panel, figure: GnoviFigure, canvas, *, dark_mode: bool) -> None:
    base_fontsize = panel.legend_fontsize or (figure.legend_font_size if figure else None) or 10.0

    def _apply(fontsize: float) -> None:
        existing = ax.get_legend()
        if existing is not None:
            existing.remove()
        handles, labels = ax.get_legend_handles_labels()
        kwargs = _legend_kwargs(panel, figure, fontsize=fontsize)
        kwargs.update(_LEGEND_FIT_TIGHT_SPACING)
        legend = ax.legend(handles, labels, **kwargs)
        legend._gnovi_legend_fit_shrunk = True  # see fit_panel_legends_to_axes
        _apply_chrome(ax, dark_mode)  # re-color the recreated legend artist

    def _fits(fontsize: float) -> bool:
        _apply(fontsize)
        canvas.draw()
        return not _legend_overflows_axes(ax, canvas.get_renderer(), tolerance_px=_LEGEND_FIT_TOLERANCE_PX)

    # Step 1: tightened spacing alone, still at the configured font size.
    if _fits(base_fontsize):
        return

    # Step 2: binary search the largest fontsize (>= the floor) that fits,
    # with tightened spacing throughout.
    low, high = _LEGEND_FIT_MIN_FONTSIZE, base_fontsize
    if not _fits(low):
        return  # doesn't fit even at the floor -- best-effort, leave it there
    while high - low > _LEGEND_FIT_CONVERGENCE_PT:
        mid = (low + high) / 2
        if _fits(mid):
            low = mid
        else:
            high = mid
    _apply(low)


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


def _draw_series_3d(ax, series: Series3D) -> None:
    """Draw one `Series3D` -- SCATTER via `Axes3D.scatter` (unconnected
    points), LINE/LINE_MARKER via `Axes3D.plot` (connects `series.
    dataframe`'s rows in their stored order -- `series.row_indices`, or the
    dataset's own row order when unset, see `Series3D`'s own docstring;
    never sorted by X, so a non-monotonic sweep renders exactly as
    recorded, and grouped-family members drawn as separate `Series3D`
    objects never connect across each other). No surface/trisurf -- the
    only 3D plot kinds this milestone implements.

    `marker_size` is squared for `ax.scatter`'s `s` for the same reason
    `_draw_series`'s 2D SCATTER case squares `PlotSeries.marker_size`:
    `Axes.scatter`'s `s` is marker AREA (points^2), not a linear size, so
    squaring here is what makes the same stored number mean the same
    visual marker size in both 2D and 3D scatter. `ax.plot`'s `markersize`
    is already linear, so LINE/LINE_MARKER pass `series.marker_size`
    through unsquared -- exactly like `_draw_series`'s own LINE/SCATTER
    cases already do for 2D.
    """
    x, y, z = numeric_xyz(series.dataframe, series.x_column, series.y_column, series.z_column)
    if series.plot_type == Plot3DType.SCATTER:
        ax.scatter(
            x,
            y,
            z,
            label=series.label,
            color=series.color,
            marker=series.marker or "o",
            s=series.marker_size**2,
            alpha=series.alpha,
        )
    else:
        marker = series.marker if series.plot_type == Plot3DType.LINE_MARKER else ""
        ax.plot(
            x,
            y,
            z,
            label=series.label,
            color=series.color,
            linestyle=series.line_style,
            linewidth=series.line_width,
            marker=marker or None,
            markersize=series.marker_size,
            alpha=series.alpha,
        )


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
