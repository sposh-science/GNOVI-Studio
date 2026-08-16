from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from gnovi_plot.plotting.backends.matplotlib_backend import apply_figure_layout, render_figure
from gnovi_plot.plotting.figure import GnoviFigure, PlotTheme

RASTER_FORMATS = ("png", "tiff")
VECTOR_FORMATS = ("svg", "pdf")
SUPPORTED_FORMATS = RASTER_FORMATS + VECTOR_FORMATS


class ExportError(Exception):
    """Raised for invalid export parameters (unsupported format, non-positive
    size/DPI)."""


def export_figure(
    figure: GnoviFigure,
    path: str | Path,
    *,
    fmt: str | None = None,
    dpi: int = 300,
    transparent: bool = False,
    tight_bbox: bool = False,
    pad_inches: float = 0.1,
    dark_mode: bool | None = None,
) -> Path:
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
    """
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
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
    axes_list = list(mpl_figure.subplots(rows, cols, squeeze=False).flat)
    render_figure(axes_list, figure, dark_mode=dark_mode)
    # Same stored margins/spacing as the on-screen preview (see
    # `gui.widgets.plot_canvas.PlotCanvas._apply_layout`) -- never
    # Matplotlib's automatic `tight_layout()`, so export and preview can't
    # diverge and a figure's layout stays fully reproducible from its own
    # stored state. Full-bleed rect: export has no letterboxing concept.
    apply_figure_layout(mpl_figure, figure)

    save_kwargs: dict = dict(format=fmt, dpi=dpi, transparent=transparent)
    if tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = pad_inches

    path.parent.mkdir(parents=True, exist_ok=True)
    mpl_figure.savefig(path, **save_kwargs)
    return path


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
