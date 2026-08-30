from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from gnovi_plot.analysis.results import ResidualData
from gnovi_plot.gui.widgets.residual_plot import ResidualPlotWidget

_DEFAULT_SIZE = (560, 380)


class ResidualWindow(QWidget):
    """A dedicated, non-modal, independently resizable top-level window for
    residual diagnostics -- observed minus fitted vs. source x -- hosting a
    single `ResidualPlotWidget`.

    Deliberately independent of `GnoviFigure`/`Panel`/`PlotSeries`/the
    Workbench export path, same as `ResidualPlotWidget` itself: this is
    analysis-result visualization only.

    Meant to be created lazily by its owner (see
    `gui.widgets.analysis_result_view.AnalysisResultView`) and reused for
    its owner's whole lifetime: `show_residuals()` updates the title/plot
    in place and (re)shows/raises the *same* window rather than a new
    instance being created per fit, so repeated "View Residuals..." clicks
    or successive fits never accumulate windows.

    `WA_DeleteOnClose` is deliberately left unset -- the platform close
    button just hides a plain `QWidget` (the normal Qt default), it does
    not destroy it, so this instance survives to be shown again later.
    Passing `parent` with the `Qt.Window` flag makes this a true top-level
    window (independent, non-modal, normal min/max/close chrome) while
    still tying its Qt object lifetime to `parent` for cleanup.
    """

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._plot = ResidualPlotWidget()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

        self.resize(*_DEFAULT_SIZE)

    def show_residuals(self, residual_data: ResidualData, *, x_label: str, y_label: str, title: str) -> None:
        """Update this window for `residual_data`/`title`, then show, raise,
        and activate it -- always the same instance, never a new window."""
        self.setWindowTitle(title)
        self._plot.plot_residuals(residual_data, x_label=x_label, y_label=y_label)
        self.show()
        self.raise_()
        self.activateWindow()
