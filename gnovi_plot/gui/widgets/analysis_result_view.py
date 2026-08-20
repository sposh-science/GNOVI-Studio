from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget

from gnovi_plot.analysis.results import AnalysisResult

_EMPTY_STATE_TEXT = (
    "No results yet.\n\n"
    "Run an analysis -- curve fitting, and (as they're added) statistics, "
    "peak analysis, FFT, smoothing, or domain-specific tools -- to see its "
    "output here."
)


class AnalysisResultView(QWidget):
    """Read-only display for the most recently produced `AnalysisResult`.

    Renders any `AnalysisResult` through its `summary()`/`details()`
    contract only -- never imports or references `FitResult` or any other
    concrete subclass. Adding a new analysis tool (statistics, peak
    analysis, FFT, smoothing, a domain-specific module) only ever means
    giving it its own `AnalysisResult` subclass; this view does not change.

    Shows a single result at a time (the most recent), plus its own empty
    state when nothing has been shown yet -- there is no history list in
    this milestone.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._result: AnalysisResult | None = None

        self._empty_label = QLabel(_EMPTY_STATE_TEXT)
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setWordWrap(True)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet("font-weight: 600;")

        self._details_widget = QWidget()
        self._details_form = QFormLayout(self._details_widget)
        self._details_form.setContentsMargins(0, 0, 0, 0)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self._summary_label)
        content_layout.addWidget(self._details_widget)
        content_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._empty_label)
        layout.addWidget(self._content)

        self.clear()

    @property
    def result(self) -> AnalysisResult | None:
        """The result currently shown, or `None` in the empty state."""
        return self._result

    def show_result(self, result: AnalysisResult) -> None:
        """Display `result`, replacing whatever was shown before."""
        self._result = result
        self._summary_label.setText(result.summary())
        self._rebuild_details(result.details())
        self._empty_label.setVisible(False)
        self._content.setVisible(True)

    def clear(self) -> None:
        """Return to the empty state -- no result shown."""
        self._result = None
        self._summary_label.clear()
        self._rebuild_details([])
        self._empty_label.setVisible(True)
        self._content.setVisible(False)

    def _rebuild_details(self, rows: list[tuple[str, str]]) -> None:
        while self._details_form.rowCount():
            self._details_form.removeRow(0)
        for label, value in rows:
            self._details_form.addRow(f"{label}:", QLabel(value))
