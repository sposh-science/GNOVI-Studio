from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.series import PlotSeries

_logger = logging.getLogger("gnovi_plot")


class PlotTheme(str, Enum):
    """A figure's own rendering theme for the Matplotlib canvas/export --
    declarative `GnoviFigure` state (see `GnoviFigure.plot_theme`), not an
    application-wide preference: two different figures/projects may
    legitimately want Light vs. Dark, and a figure must reopen looking
    exactly as it was saved. Canonically defined here (the plotting engine)
    rather than in `gui.styles`, which only re-exports it for existing
    importers -- this module has no Qt dependency, so `PlotTheme` stays
    usable from a plain script (see `export.figure_export`) without pulling
    in PySide6. Never the application chrome's own theme (see
    `gui.styles` module docstring)."""

    LIGHT = "light"
    DARK = "dark"


# Matplotlib's default "tab10" cycle, reproduced as plain hex strings so this
# module has no Matplotlib/rendering dependency of its own. Used to
# auto-assign colors for the Light plot theme.
_DEFAULT_COLOR_CYCLE_LIGHT = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# A brighter/higher-saturation cycle for the Dark plot theme -- tab10's
# darker members (e.g. the navy blue, brick red) read poorly against the
# dark canvas background (see gui.styles/matplotlib_backend's dark chrome),
# so auto-assigned series use this cycle instead whenever a series is added
# while Dark is the active Plot Theme.
_DEFAULT_COLOR_CYCLE_DARK = [
    "#8ab4f8",
    "#ffb74d",
    "#81c995",
    "#f28b82",
    "#c58af9",
    "#d7a486",
    "#f6a8d0",
    "#b0b4bd",
    "#e1e05a",
    "#7fd8e8",
]


def theme_color_cycle(dark_mode: bool) -> list[str]:
    """The default auto-assigned series color cycle for the given Plot
    Theme. Only ever used to pick a color -- for a fresh auto-assignment
    (`Panel.add_series`) or an explicit "Optimize Colors for Theme" action
    (`gui.widgets.plot_series_panel`) -- never to silently reinterpret an
    already-assigned color when the theme changes later; see
    `PlotSeries.color_is_manual`.
    """
    return _DEFAULT_COLOR_CYCLE_DARK if dark_mode else _DEFAULT_COLOR_CYCLE_LIGHT

_PANEL_LABEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Display-style fields eligible for "Copy Active Panel Style to All Panels":
# scale/ticks/spines/grid/legend formatting, deliberately excluding
# title/xlabel/ylabel/xlim/ylim/series, which are per-panel content rather
# than style and would be destructive to overwrite in bulk.
_PANEL_STYLE_FIELDS = [
    "xscale",
    "yscale",
    "invert_x",
    "invert_y",
    "grid",
    "grid_which",
    "legend_visible",
    "legend_loc",
    "legend_ncol",
    "legend_frameon",
    "legend_fontsize",
    "legend_title",
    "tick_direction",
    "minor_ticks",
    "major_tick_spacing_x",
    "major_tick_spacing_y",
    "minor_tick_spacing_x",
    "minor_tick_spacing_y",
    "major_tick_length",
    "major_tick_width",
    "minor_tick_length",
    "minor_tick_width",
    "tick_label_size",
    "axis_label_size",
    "title_size",
    "scientific_notation_x",
    "scientific_notation_y",
    "spine_top",
    "spine_bottom",
    "spine_left",
    "spine_right",
    "spine_linewidth",
]


@dataclass
class Panel:
    """Declarative description of a single subplot: its series plus every
    axes-level display setting (labels, limits, scale, ticks, spines, grid,
    legend). Backend-agnostic -- rendering it is a separate concern (see
    `plotting.backends.matplotlib_backend`).

    This is exactly what `GnoviFigure` used to be before multi-panel support:
    a `GnoviFigure` with a single default `Panel` behaves identically to the
    old single-axes GnoviFigure, which is what keeps existing callers and
    tests working unchanged.
    """

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    xscale: str = "linear"  # "linear" | "log"
    yscale: str = "linear"
    invert_x: bool = False
    invert_y: bool = False

    grid: bool = False
    grid_which: str = "major"  # "major" | "minor" | "both"

    legend_visible: bool = True
    legend_loc: str = "best"
    legend_ncol: int = 1
    legend_frameon: bool = True
    legend_fontsize: float | None = None
    legend_title: str = ""

    tick_direction: str = "out"  # "in" | "out" | "inout"
    minor_ticks: bool = False
    major_tick_spacing_x: float | None = None
    major_tick_spacing_y: float | None = None
    minor_tick_spacing_x: float | None = None
    minor_tick_spacing_y: float | None = None
    # Matplotlib's own rcParam defaults (xtick.major.size/width,
    # xtick.minor.size/width), reproduced explicitly so a panel's tick
    # geometry is always part of its own declarative state rather than
    # inherited implicitly from whatever rcParams happen to be active.
    major_tick_length: float = 3.5
    major_tick_width: float = 0.8
    minor_tick_length: float = 2.0
    minor_tick_width: float = 0.6
    tick_label_size: float | None = None
    axis_label_size: float | None = None
    title_size: float | None = None
    scientific_notation_x: bool = False
    scientific_notation_y: bool = False

    spine_top: bool = True
    spine_bottom: bool = True
    spine_left: bool = True
    spine_right: bool = True
    spine_linewidth: float = 1.0

    # Auto-assigned by GnoviFigure whenever the panel grid changes; only
    # drawn when GnoviFigure.panel_labels_visible is True.
    panel_label: str = ""

    series: list[PlotSeries] = field(default_factory=list)
    _next_color_index: int = field(default=0, repr=False)

    def add_series(self, series: PlotSeries, *, dark_mode: bool = False) -> None:
        if series.color is None:
            cycle = theme_color_cycle(dark_mode)
            series.color = cycle[self._next_color_index % len(cycle)]
            self._next_color_index += 1
        self.series.append(series)

    def remove_series(self, series_id: str) -> None:
        self.series = [s for s in self.series if s.id != series_id]

    def clear_series(self) -> None:
        self.series = []
        self._next_color_index = 0

    def get_series(self, series_id: str) -> PlotSeries | None:
        for s in self.series:
            if s.id == series_id:
                return s
        return None

    def reset_limits(self) -> None:
        self.xlim = None
        self.ylim = None

    def invalidate_series_for_dataset(self, dataset: Dataset, row_set_changed: bool) -> list[PlotSeries]:
        """Mark series referencing `dataset` stale after a transformation.

        A series is marked stale if a column it uses no longer exists in the
        dataset's working data (e.g. a calculated column dropped by
        `reset_working_data()`), or -- when `row_set_changed` is True, i.e.
        the transformation was `exclude_rows`/`keep_rows`/`reset_working_data`
        -- if it has a `row_range` at all. Row ranges are invalidated
        unconditionally rather than only when out of bounds, because a
        row-count/order change can silently shift what a still-in-bounds
        range actually contains (e.g. a manual or detected cycle); guessing
        that it's still correct isn't safe. Already-stale series and series
        for other datasets are left untouched. Returns the series newly
        marked stale.
        """
        newly_stale = []
        for series in self.series:
            if series.dataset.id != dataset.id or series.stale:
                continue
            missing_column = series.x_column not in dataset.columns or (
                series.y_column is not None and series.y_column not in dataset.columns
            )
            row_range_invalid = row_set_changed and series.row_range is not None
            if missing_column or row_range_invalid:
                series.stale = True
                newly_stale.append(series)
        return newly_stale

    def to_dict(self) -> dict:
        """Project-save representation. `series` entries store `dataset_id`
        rather than a nested `Dataset` (see `PlotSeries.to_dict`)."""
        return {
            "title": self.title,
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "xlim": list(self.xlim) if self.xlim is not None else None,
            "ylim": list(self.ylim) if self.ylim is not None else None,
            "xscale": self.xscale,
            "yscale": self.yscale,
            "invert_x": self.invert_x,
            "invert_y": self.invert_y,
            "grid": self.grid,
            "grid_which": self.grid_which,
            "legend_visible": self.legend_visible,
            "legend_loc": self.legend_loc,
            "legend_ncol": self.legend_ncol,
            "legend_frameon": self.legend_frameon,
            "legend_fontsize": self.legend_fontsize,
            "legend_title": self.legend_title,
            "tick_direction": self.tick_direction,
            "minor_ticks": self.minor_ticks,
            "major_tick_spacing_x": self.major_tick_spacing_x,
            "major_tick_spacing_y": self.major_tick_spacing_y,
            "minor_tick_spacing_x": self.minor_tick_spacing_x,
            "minor_tick_spacing_y": self.minor_tick_spacing_y,
            "major_tick_length": self.major_tick_length,
            "major_tick_width": self.major_tick_width,
            "minor_tick_length": self.minor_tick_length,
            "minor_tick_width": self.minor_tick_width,
            "tick_label_size": self.tick_label_size,
            "axis_label_size": self.axis_label_size,
            "title_size": self.title_size,
            "scientific_notation_x": self.scientific_notation_x,
            "scientific_notation_y": self.scientific_notation_y,
            "spine_top": self.spine_top,
            "spine_bottom": self.spine_bottom,
            "spine_left": self.spine_left,
            "spine_right": self.spine_right,
            "spine_linewidth": self.spine_linewidth,
            "panel_label": self.panel_label,
            "next_color_index": self._next_color_index,
            "series": [s.to_dict() for s in self.series],
        }

    @classmethod
    def from_dict(cls, data: dict, dataset_lookup: dict[str, "Dataset"]) -> "Panel":
        """Reconstruct from `to_dict`'s output. `dataset_lookup` resolves
        each series' `dataset_id` (see `PlotSeries.from_dict`); a series
        whose id isn't in the lookup is silently dropped, not an error --
        see `core.project_io.load_project` for why."""
        xlim = data.get("xlim")
        ylim = data.get("ylim")
        panel = cls(
            title=data.get("title", ""),
            xlabel=data.get("xlabel", ""),
            ylabel=data.get("ylabel", ""),
            xlim=tuple(xlim) if xlim is not None else None,
            ylim=tuple(ylim) if ylim is not None else None,
            xscale=data.get("xscale", "linear"),
            yscale=data.get("yscale", "linear"),
            invert_x=data.get("invert_x", False),
            invert_y=data.get("invert_y", False),
            grid=data.get("grid", False),
            grid_which=data.get("grid_which", "major"),
            legend_visible=data.get("legend_visible", True),
            legend_loc=data.get("legend_loc", "best"),
            legend_ncol=data.get("legend_ncol", 1),
            legend_frameon=data.get("legend_frameon", True),
            legend_fontsize=data.get("legend_fontsize"),
            legend_title=data.get("legend_title", ""),
            tick_direction=data.get("tick_direction", "out"),
            minor_ticks=data.get("minor_ticks", False),
            major_tick_spacing_x=data.get("major_tick_spacing_x"),
            major_tick_spacing_y=data.get("major_tick_spacing_y"),
            minor_tick_spacing_x=data.get("minor_tick_spacing_x"),
            minor_tick_spacing_y=data.get("minor_tick_spacing_y"),
            major_tick_length=data.get("major_tick_length", 3.5),
            major_tick_width=data.get("major_tick_width", 0.8),
            minor_tick_length=data.get("minor_tick_length", 2.0),
            minor_tick_width=data.get("minor_tick_width", 0.6),
            tick_label_size=data.get("tick_label_size"),
            axis_label_size=data.get("axis_label_size"),
            title_size=data.get("title_size"),
            scientific_notation_x=data.get("scientific_notation_x", False),
            scientific_notation_y=data.get("scientific_notation_y", False),
            spine_top=data.get("spine_top", True),
            spine_bottom=data.get("spine_bottom", True),
            spine_left=data.get("spine_left", True),
            spine_right=data.get("spine_right", True),
            spine_linewidth=data.get("spine_linewidth", 1.0),
            panel_label=data.get("panel_label", ""),
        )
        for series_data in data.get("series", []):
            series = PlotSeries.from_dict(series_data, dataset_lookup)
            if series is not None:
                panel.series.append(series)
            else:
                _logger.warning(
                    "Dropped plot series '%s' on project load: its dataset "
                    "(id %s) is missing from the project.",
                    series_data.get("label", "?"),
                    series_data.get("dataset_id", "?"),
                )
        panel._next_color_index = data.get("next_color_index", 0)
        return panel


class GnoviFigure:
    """Declarative description of a publication figure/page: one or more
    `Panel`s (subplots) laid out on a single Matplotlib Figure, plus
    page-level state (figure size/DPI-relevant dimensions, typography
    defaults, panel layout).

    A default `GnoviFigure()` has exactly one panel and behaves exactly like
    the pre-Milestone-5 single-axes GnoviFigure: `title`, `xlabel`, `ylabel`,
    `xlim`, `ylim`, `grid`, `legend_visible`, `legend_loc` and `series` are
    properties delegating to the active panel, and `add_series`/
    `remove_series`/`clear_series`/`get_series`/`reset_limits`/
    `invalidate_series_for_dataset` behave exactly as before. Multi-panel
    code should use `figure.panels`, `figure.active_panel`, and
    `figure.set_layout()`/`figure.set_active_panel()` directly.
    """

    def __init__(
        self,
        *,
        name: str = "Figure 1",
        plot_theme: PlotTheme = PlotTheme.LIGHT,
        panels: list[Panel] | None = None,
        layout: tuple[int, int] = (1, 1),
        active_panel_index: int = 0,
        panel_labels_visible: bool = False,
        figure_width_in: float = 6.4,
        figure_height_in: float = 4.8,
        aspect_preset: str = "Auto / Fit workspace",
        lock_aspect_ratio: bool = False,
        panel_aspect_preset: str = "Auto",
        font_family: str | None = None,
        base_font_size: float = 10.0,
        title_font_size: float = 12.0,
        axis_label_font_size: float = 10.0,
        tick_label_font_size: float = 9.0,
        legend_font_size: float = 9.0,
        grid_linestyle: str = "--",
        grid_linewidth: float = 0.8,
        grid_alpha: float = 0.6,
        grid_color: str | None = None,
        margin_left: float = 0.125,
        margin_right: float = 0.9,
        margin_bottom: float = 0.11,
        margin_top: float = 0.88,
        panel_wspace: float = 0.2,
        panel_hspace: float = 0.2,
        **panel_kwargs,
    ) -> None:
        self.name = name
        # Declarative figure state, not an app-wide preference -- see
        # `PlotTheme`'s docstring. `MainWindow` reads/writes this directly
        # (via `figure_model.plot_theme`) rather than caching its own copy,
        # so it's always correct across Open/New Project swapping which
        # `GnoviFigure` is "current" (see `gui.main_window._sync_theme_controls`).
        self.plot_theme = plot_theme
        self.panels: list[Panel] = panels if panels is not None else [Panel(**panel_kwargs)]
        self.layout = layout
        self.active_panel_index = active_panel_index
        self.panel_labels_visible = panel_labels_visible

        self.figure_width_in = figure_width_in
        self.figure_height_in = figure_height_in
        self.aspect_preset = aspect_preset
        self.lock_aspect_ratio = lock_aspect_ratio
        # Panel Aspect Ratio -- the physical shape of each individual Axes
        # box (e.g. "1:1" = square graph boxes), completely independent of
        # `aspect_preset`/`lock_aspect_ratio` above (the OUTER figure/page
        # shape) and never a per-Panel property: it's figure/layout-level,
        # applying uniformly to every panel in this figure -- see
        # `plotting.units.panel_box_aspect` and
        # `plotting.backends.matplotlib_backend.render_panel`, which is the
        # only place it's actually applied (via `Axes.set_box_aspect`,
        # deliberately never `Axes.set_aspect("equal")` -- this never
        # changes numeric X/Y data scaling).
        self.panel_aspect_preset = panel_aspect_preset

        self.font_family = font_family
        self.base_font_size = base_font_size
        self.title_font_size = title_font_size
        self.axis_label_font_size = axis_label_font_size
        self.tick_label_font_size = tick_label_font_size
        self.legend_font_size = legend_font_size

        # "Universal" Grid Appearance: unlike `Panel.grid`/`grid_which`
        # (per-panel on/off + major/minor, since panels may reasonably
        # differ on whether a grid is shown at all), the grid's *look* is a
        # figure-wide publication-consistency setting -- every panel that
        # has its grid on renders it with this same style/width/opacity/
        # color. `grid_color=None` means "use the current Plot Theme's
        # default grid color" (see matplotlib_backend._grid_color).
        self.grid_linestyle = grid_linestyle
        self.grid_linewidth = grid_linewidth
        self.grid_alpha = grid_alpha
        self.grid_color = grid_color

        # Outer margins + inter-panel spacing (Matplotlib's
        # `figure.subplots_adjust(left/right/bottom/top/wspace/hspace)`
        # parameters), stored explicitly on the figure rather than
        # recomputed by an automatic layout engine on every render -- see
        # `gui.widgets.figure_layout_panel` and
        # `plotting.backends.matplotlib_backend.apply_figure_layout`. This
        # keeps a figure's on-screen preview and every exported format
        # (PNG/TIFF/SVG/PDF) driven by the exact same stored numbers, and
        # keeps a session's saved state enough to reproduce its layout
        # deterministically (Matplotlib's own automatic "tight" layout
        # depends on the rendering environment's font metrics, which isn't
        # reproducible from stored parameters alone). Defaults match
        # Matplotlib's own rcParams (`figure.subplot.*`), so a fresh
        # GnoviFigure renders identically to plain Matplotlib without any
        # layout call. `apply_tight_layout()` computes a good one-off
        # starting point on demand; it never runs implicitly.
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.margin_bottom = margin_bottom
        self.margin_top = margin_top
        self.panel_wspace = panel_wspace
        self.panel_hspace = panel_hspace

        self._renumber_panel_labels()

    # --- Multi-panel management ---------------------------------------------

    @property
    def active_panel(self) -> Panel:
        return self.panels[self.active_panel_index]

    def set_active_panel(self, index: int) -> None:
        if not 0 <= index < len(self.panels):
            raise ValueError(f"Panel index {index} out of range for {len(self.panels)} panel(s)")
        self.active_panel_index = index

    def set_layout(self, rows: int, cols: int) -> None:
        """Resize the panel grid to `rows * cols`, preserving existing
        panels by position: growing appends fresh `Panel`s, shrinking drops
        the trailing ones. Panel content is never migrated between grid
        positions -- arbitrary drag-and-drop relayout is out of scope for
        this milestone.
        """
        count = rows * cols
        if count < 1:
            raise ValueError("Layout must have at least one panel")

        if count > len(self.panels):
            self.panels.extend(Panel() for _ in range(count - len(self.panels)))
        elif count < len(self.panels):
            self.panels = self.panels[:count]

        self.layout = (rows, cols)
        self.active_panel_index = min(self.active_panel_index, count - 1)
        self._renumber_panel_labels()

    def _renumber_panel_labels(self) -> None:
        for i, panel in enumerate(self.panels):
            panel.panel_label = f"({_PANEL_LABEL_LETTERS[i % len(_PANEL_LABEL_LETTERS)]})"

    # --- Backward-compatible single-panel passthroughs (active panel) ------

    @property
    def title(self) -> str:
        return self.active_panel.title

    @title.setter
    def title(self, value: str) -> None:
        self.active_panel.title = value

    @property
    def xlabel(self) -> str:
        return self.active_panel.xlabel

    @xlabel.setter
    def xlabel(self, value: str) -> None:
        self.active_panel.xlabel = value

    @property
    def ylabel(self) -> str:
        return self.active_panel.ylabel

    @ylabel.setter
    def ylabel(self, value: str) -> None:
        self.active_panel.ylabel = value

    @property
    def xlim(self) -> tuple[float, float] | None:
        return self.active_panel.xlim

    @xlim.setter
    def xlim(self, value: tuple[float, float] | None) -> None:
        self.active_panel.xlim = value

    @property
    def ylim(self) -> tuple[float, float] | None:
        return self.active_panel.ylim

    @ylim.setter
    def ylim(self, value: tuple[float, float] | None) -> None:
        self.active_panel.ylim = value

    @property
    def grid(self) -> bool:
        return self.active_panel.grid

    @grid.setter
    def grid(self, value: bool) -> None:
        self.active_panel.grid = value

    @property
    def legend_visible(self) -> bool:
        return self.active_panel.legend_visible

    @legend_visible.setter
    def legend_visible(self, value: bool) -> None:
        self.active_panel.legend_visible = value

    @property
    def legend_loc(self) -> str:
        return self.active_panel.legend_loc

    @legend_loc.setter
    def legend_loc(self, value: str) -> None:
        self.active_panel.legend_loc = value

    @property
    def series(self) -> list[PlotSeries]:
        return self.active_panel.series

    @series.setter
    def series(self, value: list[PlotSeries]) -> None:
        self.active_panel.series = value

    def add_series(self, series: PlotSeries, *, dark_mode: bool = False) -> None:
        self.active_panel.add_series(series, dark_mode=dark_mode)

    def remove_series(self, series_id: str) -> None:
        for panel in self.panels:
            panel.remove_series(series_id)

    def clear_series(self) -> None:
        self.active_panel.clear_series()

    def get_series(self, series_id: str) -> PlotSeries | None:
        for panel in self.panels:
            found = panel.get_series(series_id)
            if found is not None:
                return found
        return None

    def reset_limits(self) -> None:
        self.active_panel.reset_limits()

    def invalidate_series_for_dataset(self, dataset: Dataset, row_set_changed: bool) -> list[PlotSeries]:
        newly_stale: list[PlotSeries] = []
        for panel in self.panels:
            newly_stale.extend(panel.invalidate_series_for_dataset(dataset, row_set_changed))
        return newly_stale

    def copy_active_panel_style_to_all(self) -> None:
        """Copy the active panel's display-style fields (scale, ticks,
        spines, grid, legend) onto every other panel, leaving each panel's
        title/labels/limits/series untouched."""
        source = self.active_panel
        for panel in self.panels:
            if panel is source:
                continue
            for field_name in _PANEL_STYLE_FIELDS:
                setattr(panel, field_name, getattr(source, field_name))

    def to_dict(self) -> dict:
        """Project-save representation; see `Panel.to_dict` for how panels
        (and their series) are serialized within `panels`."""
        return {
            "name": self.name,
            "plot_theme": self.plot_theme.value,
            "layout": list(self.layout),
            "active_panel_index": self.active_panel_index,
            "panel_labels_visible": self.panel_labels_visible,
            "figure_width_in": self.figure_width_in,
            "figure_height_in": self.figure_height_in,
            "aspect_preset": self.aspect_preset,
            "lock_aspect_ratio": self.lock_aspect_ratio,
            "panel_aspect_preset": self.panel_aspect_preset,
            "font_family": self.font_family,
            "base_font_size": self.base_font_size,
            "title_font_size": self.title_font_size,
            "axis_label_font_size": self.axis_label_font_size,
            "tick_label_font_size": self.tick_label_font_size,
            "legend_font_size": self.legend_font_size,
            "grid_linestyle": self.grid_linestyle,
            "grid_linewidth": self.grid_linewidth,
            "grid_alpha": self.grid_alpha,
            "grid_color": self.grid_color,
            "margin_left": self.margin_left,
            "margin_right": self.margin_right,
            "margin_bottom": self.margin_bottom,
            "margin_top": self.margin_top,
            "panel_wspace": self.panel_wspace,
            "panel_hspace": self.panel_hspace,
            "panels": [p.to_dict() for p in self.panels],
        }

    @classmethod
    def from_dict(cls, data: dict, dataset_lookup: dict[str, Dataset]) -> "GnoviFigure":
        """Reconstruct from `to_dict`'s output. An empty/missing `panels`
        list falls back to a fresh default single Panel (`panels=None`)
        rather than an empty `GnoviFigure.panels`, which `active_panel`
        can't index into."""
        panels = [Panel.from_dict(p, dataset_lookup) for p in data.get("panels", [])]
        try:
            plot_theme = PlotTheme(data.get("plot_theme", PlotTheme.LIGHT.value))
        except ValueError:
            plot_theme = PlotTheme.LIGHT
        figure = cls(
            name=data.get("name", "Figure 1"),
            plot_theme=plot_theme,
            panels=panels or None,
            layout=tuple(data.get("layout", (1, 1))),
            active_panel_index=data.get("active_panel_index", 0),
            panel_labels_visible=data.get("panel_labels_visible", False),
            figure_width_in=data.get("figure_width_in", 6.4),
            figure_height_in=data.get("figure_height_in", 4.8),
            aspect_preset=data.get("aspect_preset", "Auto / Fit workspace"),
            lock_aspect_ratio=data.get("lock_aspect_ratio", False),
            panel_aspect_preset=data.get("panel_aspect_preset", "Auto"),
            font_family=data.get("font_family"),
            base_font_size=data.get("base_font_size", 10.0),
            title_font_size=data.get("title_font_size", 12.0),
            axis_label_font_size=data.get("axis_label_font_size", 10.0),
            tick_label_font_size=data.get("tick_label_font_size", 9.0),
            legend_font_size=data.get("legend_font_size", 9.0),
            grid_linestyle=data.get("grid_linestyle", "--"),
            grid_linewidth=data.get("grid_linewidth", 0.8),
            grid_alpha=data.get("grid_alpha", 0.6),
            grid_color=data.get("grid_color"),
            margin_left=data.get("margin_left", 0.125),
            margin_right=data.get("margin_right", 0.9),
            margin_bottom=data.get("margin_bottom", 0.11),
            margin_top=data.get("margin_top", 0.88),
            panel_wspace=data.get("panel_wspace", 0.2),
            panel_hspace=data.get("panel_hspace", 0.2),
        )
        figure.active_panel_index = min(
            max(figure.active_panel_index, 0), len(figure.panels) - 1
        )
        return figure
