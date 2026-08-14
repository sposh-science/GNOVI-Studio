from __future__ import annotations

from dataclasses import dataclass, field

from gnovi_plot.plotting.series import PlotSeries

# Matplotlib's default "tab10" cycle, reproduced as plain hex strings so this
# module has no Matplotlib/rendering dependency of its own.
_DEFAULT_COLOR_CYCLE = [
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


@dataclass
class GnoviFigure:
    """Declarative description of a plot: its series plus figure-level state.

    Backend-agnostic -- rendering it (e.g. onto a Matplotlib PlotCanvas) is a
    separate concern.
    """

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    grid: bool = False
    legend_visible: bool = True
    legend_loc: str = "best"
    series: list[PlotSeries] = field(default_factory=list)
    _next_color_index: int = field(default=0, repr=False)

    def add_series(self, series: PlotSeries) -> None:
        if series.color is None:
            series.color = _DEFAULT_COLOR_CYCLE[self._next_color_index % len(_DEFAULT_COLOR_CYCLE)]
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
