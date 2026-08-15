from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from gnovi_plot.plotting.backends.matplotlib_backend import apply_figure_layout, render_figure
from gnovi_plot.plotting.figure import GnoviFigure

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
    tight_bbox: bool = True,
    pad_inches: float = 0.1,
    dark_mode: bool = False,
) -> Path:
    """Render `figure` to `path` as PNG/TIFF (raster) or SVG/PDF (vector).

    Always builds a fresh, offscreen `matplotlib.figure.Figure` sized from
    `figure.figure_width_in` / `figure.figure_height_in` -- never from any
    on-screen canvas widget -- so on-screen display size never determines
    export quality. Raster formats honor `dpi`; vector formats stay
    resolution-independent (text/lines are preserved as vector data; `dpi`
    there only affects any embedded raster elements, which this app doesn't
    currently produce). `fmt` defaults to `path`'s extension.

    `dark_mode` defaults to `False` (publication-light) regardless of the
    GUI's current theme -- it is never inferred from the app, only from
    whatever the caller (the Export Figure dialog's explicit "Dark
    background" checkbox) passes in.
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
