from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from gnovi_plot.plotting.backends.matplotlib_backend import apply_figure_layout, build_projection_aware_axes, render_figure
from gnovi_plot.plotting.figure import GnoviFigure, Panel, Panel3D, PlotTheme
from gnovi_plot.plotting.graph import clone_panel_with_shared_datasets

RASTER_FORMATS = ("png", "tiff")
VECTOR_FORMATS = ("svg", "pdf")
SUPPORTED_FORMATS = RASTER_FORMATS + VECTOR_FORMATS


class ExportError(Exception):
    """Raised for invalid export parameters (unsupported format, non-positive
    size/DPI)."""


def export_figure(
    figure: GnoviFigure,
    path,
    *,
    fmt: str | None = None,
    dpi: int = 300,
    transparent: bool = False,
    facecolor: str | None = None,
    tight_bbox: bool = False,
    pad_inches: float = 0.1,
    dark_mode: bool | None = None,
) -> Path | None:
    """Render `figure` to `path` as PNG/TIFF (raster) or SVG/PDF (vector).

    Always builds a fresh, offscreen `matplotlib.figure.Figure` sized from
    `figure.figure_width_in` / `figure.figure_height_in` -- never from any
    on-screen canvas widget -- so on-screen display size never determines
    export quality. Raster formats honor `dpi`; vector formats stay
    resolution-independent (text/lines are preserved as vector data; `dpi`
    there only affects any embedded raster elements, which this app doesn't
    currently produce). `fmt` defaults to `path`'s extension.

    `tight_bbox` defaults to False -- matching both the on-screen preview
    (`gui.widgets.plot_canvas.PlotCanvas`, which always shows the figure's
    full configured margins as visible space around the axes, never
    cropped) and Matplotlib's own `savefig.bbox` rcParam default
    ('standard', i.e. not tight -- exactly what the Matplotlib navigation
    toolbar's own "Save" action produces). `bbox_inches="tight"` crops the
    saved image to the tight bounding box of actually-rendered content,
    discarding the configured blank margin around it; since every
    typography size (`legend_font_size`, `tick_label_font_size`, etc.) is
    an absolute point size, removing that margin makes the exact same
    unchanged text occupy a visibly larger fraction of the (now smaller)
    image frame -- legend, ticks, and scientific notation all appear
    "oversized" together, uniformly, even though not one font size or DPI
    value actually changed. This was the root cause of GNOVI Export Figure
    looking disproportionate next to the on-screen preview and Matplotlib's
    quick-Save (which was never tight-cropped) -- not a font-size, DPI, or
    duplicate-scaling bug. `tight_bbox=True` remains available (e.g. the
    Export Figure dialog's own "Tight bounding box" checkbox) for anyone
    who explicitly wants whitespace trimmed for publication -- it just must
    never be silently on by default.

    `dark_mode` defaults to `figure.plot_theme` (declarative figure state,
    see `plotting.figure.PlotTheme`) when left unspecified -- an export
    reproduces the figure's own configured appearance by default, exactly
    like a generated script calling this function would with no theme
    knowledge of its own. Pass `dark_mode` explicitly (as the Export Figure
    dialog's "Dark background" checkbox does) to override it for one export
    without changing the figure's stored theme.

    `facecolor=None` (the default) uses Matplotlib's own 'auto'
    `savefig.facecolor` -- i.e. whatever `render_figure` already set as
    this figure's background for `dark_mode`, unchanged. Pass an explicit
    color (e.g. `"white"`) to force a specific background regardless of
    `dark_mode`/theme -- mirrors `export_live_figure`'s own `facecolor`
    param, kept symmetric so a caller building a transient `GnoviFigure`
    (see `build_panel_export_figure`/`export_panel`) can offer the exact
    same "As shown / Opaque / Transparent" background choice the Export
    Figure dialog already offers for a live Figure.

    `path` accepts a real file path (`str`/`Path`) or an in-memory buffer
    (e.g. `io.BytesIO`, as `gui.dialogs.export_figure_dialog.
    ExportFigureDialog`'s own Panel-export live preview uses) -- mirrors
    `export_live_figure`'s own `path` handling exactly; parent directories
    are only created for a real path, and the return value is `None` for a
    buffer.
    """
    is_real_path = isinstance(path, (str, Path))
    if is_real_path:
        path = Path(path)
        fmt = (fmt or path.suffix.lstrip(".")).lower()
    else:
        fmt = (fmt or "png").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ExportError(
            f"Unsupported export format: {fmt!r} (supported: {', '.join(SUPPORTED_FORMATS)})"
        )
    if dpi <= 0:
        raise ExportError("DPI must be positive")
    if figure.figure_width_in <= 0 or figure.figure_height_in <= 0:
        raise ExportError("Figure width/height must be positive")
    if dark_mode is None:
        dark_mode = figure.plot_theme == PlotTheme.DARK

    rows, cols = figure.layout
    mpl_figure = Figure(figsize=(figure.figure_width_in, figure.figure_height_in), dpi=dpi)
    axes_list = build_projection_aware_axes(mpl_figure, rows, cols, figure.panels)
    render_figure(axes_list, figure, dark_mode=dark_mode)
    # Same stored margins/spacing as the on-screen preview (see
    # `gui.widgets.plot_canvas.PlotCanvas._apply_layout`) -- never
    # Matplotlib's automatic `tight_layout()`, so export and preview can't
    # diverge and a figure's layout stays fully reproducible from its own
    # stored state. Full-bleed rect: export has no letterboxing concept.
    apply_figure_layout(mpl_figure, figure)

    save_kwargs: dict = dict(format=fmt, dpi=dpi, transparent=transparent)
    if facecolor is not None:
        save_kwargs["facecolor"] = facecolor
    if tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = pad_inches

    if is_real_path:
        path.parent.mkdir(parents=True, exist_ok=True)
    mpl_figure.savefig(path, **save_kwargs)
    return path if is_real_path else None


# Figure-level fields copied verbatim from the source figure onto a
# transient single-Panel export figure (see `build_panel_export_figure`):
# every setting that affects how a Panel's own content/typography/grid
# renders. Deliberately excludes `aspect_preset`/`lock_aspect_ratio` (GUI
# on-screen letterboxing only -- `export_figure`/`apply_figure_layout`
# never read them, see `plotting.figure.GnoviFigure.__init__`'s own
# comments) and `figure_width_in`/`figure_height_in` (computed fresh per
# panel, see `_panel_layout_size_in`, never copied from the source's own
# whole-Figure page size).
_PANEL_EXPORT_FIGURE_LEVEL_FIELDS = (
    "plot_theme",
    "panel_aspect_preset",
    "font_family",
    "base_font_size",
    "title_font_size",
    "axis_label_font_size",
    "tick_label_font_size",
    "legend_font_size",
    "grid_linestyle",
    "grid_linewidth",
    "grid_alpha",
    "grid_color",
    "margin_left",
    "margin_right",
    "margin_bottom",
    "margin_top",
    "panel_wspace",
    "panel_hspace",
)


def _panel_layout_size_in(figure: GnoviFigure, panel_index: int) -> tuple[float, float]:
    """The physical `(width_in, height_in)` Matplotlib's own GridSpec
    allocates to `figure.panels[panel_index]`'s Axes box when the WHOLE
    `figure` is rendered at its own configured `figure_width_in`/
    `figure_height_in` -- i.e. exactly the plotting-area size that Panel
    occupies in place today, accounting for `margin_left/right/bottom/top`
    and `panel_wspace`/`panel_hspace` however GNOVI currently defines them,
    without re-deriving GridSpec's own math by hand (fragile, easy to drift
    from Matplotlib's actual behavior if either ever changes independently).

    Built from a throwaway, unrendered `matplotlib.figure.Figure` --
    `subplots()` + `apply_figure_layout()` (the exact same layout call
    `export_figure`/the on-screen preview both already use) is enough to
    populate `Axes.get_position()` deterministically; no `draw()`/renderer/
    font metrics involved, so this can never vary with the runtime
    environment and stays reproducible from `figure`'s own stored state
    alone. This is the box GridSpec *allocates*, before any
    `panel_aspect_preset` box-aspect shrink -- that shrink is a separate,
    already-preserved concern (see `build_panel_export_figure`'s docstring).
    """
    rows, cols = figure.layout
    mpl_figure = Figure(figsize=(figure.figure_width_in, figure.figure_height_in))
    # `build_projection_aware_axes` (not a plain `subplots()` call) because
    # `get_position()` is NOT identical for a 2D vs. 3D Axes at the same
    # grid cell -- Matplotlib reserves extra vertical padding for an
    # `Axes3D`'s perspective by design (confirmed experimentally while
    # building this milestone). Measuring the REAL per-panel projection via
    # the same helper every other Axes-grid construction in this app uses
    # is what keeps this correct by construction rather than by an
    # assumption about Matplotlib internals.
    axes_list = build_projection_aware_axes(mpl_figure, rows, cols, figure.panels)
    apply_figure_layout(mpl_figure, figure)
    box = axes_list[panel_index].get_position()
    return box.width * figure.figure_width_in, box.height * figure.figure_height_in


def build_panel_export_figure(figure: GnoviFigure, panel: Panel | Panel3D, dataset_manager) -> GnoviFigure:
    """A transient, independent 1x1 `GnoviFigure` for exporting exactly
    `panel` alone -- built fresh on every call, never inserted into any
    `Project`/`Workbench`, never given analysis history, never serialized,
    and discarded by the caller once export finishes (see `export_panel`,
    its only caller). `figure`/`panel`/`dataset_manager` are read only,
    never mutated.

    Panel content: `plotting.graph.clone_panel_with_shared_datasets` --
    the exact same clone helper `Project.extract_panel_to_workbench`
    already uses -- deep-copies `panel` (series/styling/everything) while
    keeping every `PlotSeries.dataset` pointed at the same live `Dataset`
    instance (so fit-derived series/`Dataset.metadata["result_id"]`
    linkage renders correctly, untouched). The clone gets a *fresh*
    `Panel.id`; this is fine and expected -- the transient figure is
    thrown away right after export, never persisted, so there's no
    original id worth preserving here (contrast with a future "Apply back"
    operation, which would need to).

    Figure-level settings: `_PANEL_EXPORT_FIGURE_LEVEL_FIELDS` copied
    verbatim from `figure` (theme, typography, grid appearance, margins,
    panel-aspect preset) -- so the exported Panel's fonts/grid/aspect
    render exactly as they do in place, not reset to `GnoviFigure()`
    defaults (the gap `Project.extract_panel_to_workbench` currently has,
    which only copies `plot_theme`). `panel_labels_visible` is
    deliberately forced to `False` regardless of the source figure's own
    setting: a Panel's `panel_label` ("(a)", "(b)", ...) is figure
    COMPOSITION metadata -- auto-assigned by `GnoviFigure._renumber_panel_
    labels` from the panel's *position* in a multi-panel figure, and only
    ever drawn because of a Figure-level toggle -- not intrinsic Panel
    content. Showing "(b)" on an image that is now the entire document,
    with no sibling panels for it to distinguish itself from, would
    misrepresent the exported file as a fragment of something it no
    longer is.

    Page size: `figure_width_in`/`figure_height_in` are NOT copied from
    `figure` -- exporting Panel 2 of a 1x3 Figure must not stretch it to
    the whole Figure's page size. Instead, derived from
    `_panel_layout_size_in` (the panel's own allocated axes-box size in
    place) scaled back out to a full single-panel page using `figure`'s
    own `margin_left/right/bottom/top` fractions -- exactly what those
    fractions already mean for a 1x1 figure (the outer margin reserved for
    axis labels/ticks/title around ITS OWN axes box), so reapplying them
    to the transient 1x1 figure is consistent with their existing
    semantics, not a new convention invented for export.
    """
    panel_index = figure.panels.index(panel)
    width_in, height_in = _panel_layout_size_in(figure, panel_index)

    margin_width_frac = figure.margin_right - figure.margin_left
    margin_height_frac = figure.margin_top - figure.margin_bottom
    export_width_in = width_in / margin_width_frac if margin_width_frac > 0 else width_in
    export_height_in = height_in / margin_height_frac if margin_height_frac > 0 else height_in

    cloned_panel = clone_panel_with_shared_datasets(panel, dataset_manager)
    figure_level_kwargs = {name: getattr(figure, name) for name in _PANEL_EXPORT_FIGURE_LEVEL_FIELDS}

    return GnoviFigure(
        panels=[cloned_panel],
        layout=(1, 1),
        panel_labels_visible=False,
        figure_width_in=export_width_in,
        figure_height_in=export_height_in,
        **figure_level_kwargs,
    )


def export_panel(
    figure: GnoviFigure,
    panel: Panel | Panel3D,
    dataset_manager,
    path,
    *,
    fmt: str | None = None,
    dpi: int = 300,
    transparent: bool = False,
    facecolor: str | None = None,
    tight_bbox: bool = False,
    pad_inches: float = 0.1,
    dark_mode: bool | None = None,
) -> Path | None:
    """Export `panel` alone -- never `figure`'s other panels -- as a
    standalone publication-quality PNG/TIFF/SVG/PDF file, via a transient
    1x1 `GnoviFigure` (see `build_panel_export_figure`) passed straight
    through `export_figure` above, so it goes through the exact same
    `render_figure`/`apply_figure_layout`/`savefig` pipeline as any other
    export -- vector formats stay genuinely vector, raster formats stay
    DPI-aware, all parameters here mean exactly what they mean for
    `export_figure`, including accepting a real path or an in-memory
    buffer for `path` (see `export_figure`'s own docstring). `figure`/
    `panel`/`dataset_manager` are read only; the transient figure is
    discarded once this returns -- no Project, Workbench, Dataset, or
    analysis-history state is created or mutated.
    """
    export_figure_model = build_panel_export_figure(figure, panel, dataset_manager)
    return export_figure(
        export_figure_model,
        path,
        fmt=fmt,
        dpi=dpi,
        transparent=transparent,
        facecolor=facecolor,
        tight_bbox=tight_bbox,
        pad_inches=pad_inches,
        dark_mode=dark_mode,
    )


def export_live_figure(
    mpl_figure,
    path,
    *,
    fmt: str | None = None,
    dpi: int = 300,
    transparent: bool = False,
    facecolor: str | None = None,
    bbox_inches=None,
    pad_inches: float = 0.1,
) -> Path | None:
    """Save `mpl_figure` -- an already-rendered, LIVE Matplotlib Figure,
    typically `gui.widgets.plot_canvas.PlotCanvas.figure` -- exactly as it
    currently is, via one `savefig()` call. Never re-renders anything (no
    `render_figure`/`apply_figure_layout` call here): this is the GUI
    Export Figure dialog's WYSIWYG export path (see
    `gui.dialogs.export_figure_dialog`), saving the identical Axes/artists
    already on screen -- axes positions, legends, typography, scientific
    notation, panel geometry, margins, spacing, Figure/Panel Aspect Ratio,
    series styles, grid, labels, titles, all exactly as currently rendered
    -- so it's structurally guaranteed pixel/vector-identical to what
    Matplotlib's own toolbar "Save" would produce for the same dpi/bbox/
    background choice, since both ultimately call `savefig()` on the same
    Figure object.

    Contrast with `export_figure` above, which stays the deterministic,
    GUI-independent path (used by tests and any headless/scripted caller):
    given only a `GnoviFigure` model, it builds a fresh `Figure` sized at
    the configured `figure_width_in`/`figure_height_in`, reproducible
    regardless of any on-screen window/canvas state. Both ultimately go
    through the exact same `render_figure`/`render_panel`/
    `apply_figure_layout` drawing code (`plotting.backends.
    matplotlib_backend`) -- this function just skips that step because the
    live Figure it's given has already been through it.

    `facecolor=None` uses Matplotlib's own 'auto' `savefig.facecolor`
    default -- i.e. whatever the figure's own current background already
    is ("As shown" in the Export Figure dialog); pass an explicit color
    (e.g. `"white"`) to force a specific background regardless of the
    figure's current Plot Theme. `bbox_inches=None` saves the full Figure;
    pass `"tight"` or an explicit `matplotlib.transforms.Bbox` (in inches)
    to crop -- e.g. the Export Figure dialog's "Active Panel" scope passes
    a Bbox matching just the active Axes' own rendered extent.

    `path` accepts a real file path (`str`/`Path`) or an in-memory
    buffer (e.g. `io.BytesIO`, as the Export Figure dialog's own live
    preview uses) -- parent directories are only created for a real path,
    and the return value is `None` for a buffer (nothing meaningful to
    report back).
    """
    is_real_path = isinstance(path, (str, Path))
    if is_real_path:
        path = Path(path)
        fmt = (fmt or path.suffix.lstrip(".")).lower()
    else:
        fmt = (fmt or "png").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ExportError(
            f"Unsupported export format: {fmt!r} (supported: {', '.join(SUPPORTED_FORMATS)})"
        )
    if dpi <= 0:
        raise ExportError("DPI must be positive")

    save_kwargs: dict = dict(format=fmt, dpi=dpi, transparent=transparent)
    if facecolor is not None:
        save_kwargs["facecolor"] = facecolor
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
        save_kwargs["pad_inches"] = pad_inches

    if is_real_path:
        path.parent.mkdir(parents=True, exist_ok=True)
    mpl_figure.savefig(path, **save_kwargs)
    return path if is_real_path else None
