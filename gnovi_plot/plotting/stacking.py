from __future__ import annotations

from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_xy
from gnovi_plot.plotting.figure import Panel
from gnovi_plot.plotting.series import PlotSeries, PlotType

_STACKABLE_TYPES = (PlotType.LINE, PlotType.SCATTER)


def stackable_series(panel: Panel) -> list[PlotSeries]:
    """LINE/SCATTER series on `panel`, in display order -- the ones a
    stacked/offset plot mode applies to. Histograms are excluded: an offset
    on bar heights doesn't mean the same thing as an offset on a curve.
    """
    return [s for s in panel.series if s.plot_type in _STACKABLE_TYPES]


def suggest_offset_step(panel: Panel) -> float:
    """A step large enough that stacked curves in `panel` won't visually
    overlap: the largest single series' y-range, so worst case neighbours
    just touch. Returns 1.0 if there is no numeric data to measure yet.
    """
    max_range = 0.0
    for series in stackable_series(panel):
        try:
            _, y = numeric_xy(series.dataframe, series.x_column, series.y_column)
        except (InsufficientNumericDataError, KeyError):
            continue
        if len(y):
            max_range = max(max_range, float(y.max() - y.min()))
    return max_range or 1.0


def auto_stack_offsets(panel: Panel, step: float | None = None) -> float:
    """Assign sequential y_offset values (0, step, 2*step, ...) to every
    stackable series on `panel`, in list order. Plotting-only: this never
    touches any Dataset or PlotSeries.dataframe. Returns the step used.
    """
    resolved_step = step if step else suggest_offset_step(panel)
    for i, series in enumerate(stackable_series(panel)):
        series.y_offset = i * resolved_step
    return resolved_step


def reset_offsets(panel: Panel) -> None:
    for series in stackable_series(panel):
        series.y_offset = 0.0
