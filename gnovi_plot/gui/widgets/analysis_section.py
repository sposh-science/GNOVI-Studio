"""Shared base for GNOVI's scientific-analysis-tool sections -- the
widgets `AnalysisPanel` mounts one per analysis tool (`XRDAnalysisSection`,
`CVAnalysisSection`, and future domain modules -- see Issue #61).

`XRDAnalysisSection` and `CVAnalysisSection` independently converged on the
same signal set and several of the same lifecycle/interaction methods --
this module formalizes exactly the parts that already agree, nothing more
(see Issue #61's own read-only architecture audit for the full comparison).
Deliberately left OUT of this contract, because the two real
implementations have not actually converged on them yet:

- Canvas-overlay rendering: XRD exposes three separate accessors
  (`overlay_points()`, `preview_curve()`, `fit_overlay()`); CV exposes one
  `overlay_payload()` dict. Forcing a shared shape (or even a shared,
  loosely-typed hook nothing yet calls polymorphically) from two
  disagreeing examples risks baking in the wrong one -- left for a later
  issue, once a third real implementation exists to check both against.
- `add_to_plot_requested`/`remove_fit_curve_requested` -- both sections
  declare *a* derived-curve-add signal, but CV's is presently unused
  ("plumbed for CV-2B; unused in CV-2A" -- see `CVAnalysisSection`'s own
  comment) and XRD's has no CV counterpart for removal at all. Declared
  by whichever subclass actually needs it today, not forced into the
  shared contract just because the names happen to match.
- Any MainWindow-side dispatch -- out of scope for this issue.

Every method declared here with `raise NotImplementedError` is a required
override for a concrete subclass, documented the same way
`analysis.results.AnalysisResult` already documents `summary()`/
`details()` as a required contract without `abc.ABC` -- kept consistent
with that existing convention rather than introducing a second one.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from gnovi_plot.analysis.results import AnalysisResult
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series import PlotSeries


def eligible_analysis_series(figure: GnoviFigure) -> list[PlotSeries]:
    """Line/scatter series in the active panel usable as an analysis
    source -- excludes histograms (no `y_column` -- there is no curve to
    analyze), stale series (a missing column or invalid `row_range`), and
    anything that isn't a `PlotSeries`. Empty when the active panel is a
    `Panel3D`: checked explicitly for clarity, though `isinstance(s,
    PlotSeries)` alone already guarantees this too -- a `Panel3D`'s own
    `.series` holds only `Series3D` items, which happen to have a
    non-None `y_column` of their own but are never a `PlotSeries`.

    The one shared source-series filter every analysis tool (Curve
    Fitting, XRD, CV) needs -- previously three independent, near-
    identical copies (`AnalysisPanel._eligible_series`,
    `XRDAnalysisSection._eligible_series`,
    `CVAnalysisSection.eligible_cv_series`; see Issue #61's own audit)."""
    if isinstance(figure.active_panel, Panel3D):
        return []
    return [s for s in figure.series if isinstance(s, PlotSeries) and s.y_column is not None and not s.stale]


class AnalysisSection(QWidget):
    """Base for one scientific-analysis tool's controls, hosted by
    `AnalysisPanel` inside its own `CollapsibleSection` (see that class's
    own docstring). A concrete subclass owns exactly one tool's UI and
    domain logic (source-series selection, running the analysis, manual
    peak/candidate curation, transient previews) and reports results
    through the signals below -- `AnalysisPanel` never imports or checks
    for a specific subclass, only this shared surface plus whatever
    subclass-specific proxy methods it still calls directly today (e.g.
    `overlay_points()`/`fit_overlay()` on XRD, `overlay_payload()` on CV --
    see the module docstring for why those stay subclass-specific).
    """

    #: A fresh analysis run produced a brand-new result -- a new Analysis
    #: History entry (the "Run Fit"/"Find Peaks" convention every current
    #: section shares).
    analysis_result_ready = Signal(AnalysisResult)
    #: An in-place edit to the *current* result (manual peak add/remove/
    #: enable, a setting that only affects display/derivation) -- dirty +
    #: redisplay, never a new History entry.
    result_updated = Signal(AnalysisResult)
    #: This section's live, GUI-only canvas overlay (peak markers, a
    #: preview curve, a fit window, ...) may have changed -- MainWindow
    #: re-pulls whatever subclass-specific accessor applies and redraws;
    #: never carries the data itself.
    overlay_changed = Signal()
    #: True right after manual-peak/candidate-add mode is armed -- the
    #: next canvas click inside the active panel should add a manual seed
    #: instead of activating/focusing a panel. False when disarmed or
    #: after a click is consumed.
    manual_peak_mode_changed = Signal(bool)
    #: A short researcher-facing status string (e.g. "Added to plot: ...")
    #: -- purely informational, no state change.
    status_message = Signal(str)

    def __init__(self, figure: GnoviFigure, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)
        self._figure = figure
        self._manager = dataset_manager
        self._manual_peak_mode = False
        # Peak/candidate rows currently selected in the bottom Results-tab
        # detail table -- pushed in by MainWindow via `set_selected_peak_
        # rows` whenever that table's selection changes; Remove Selected/
        # Enable-Disable act on exactly this.
        self._results_selected_rows: list[int] = []

    # --- required overrides: wiring from AnalysisPanel/MainWindow ----------
    #
    # Each of these has real, differing logic in every current subclass
    # (e.g. XRD's `set_figure` also clears its own peak-profile-fit state;
    # CV's clears its own cycle/sweep state) -- formalized here as a
    # documented contract, not merged into one shared implementation.

    def set_figure(self, figure: GnoviFigure) -> None:
        """Repoint at a different `GnoviFigure` (a Workbench switch or a
        New/Open Project) and reload -- also clears whatever working
        state (previews, a pending fit/detection) no longer applies once
        `figure` doesn't match what it was computed against."""
        raise NotImplementedError

    def refresh(self) -> None:
        """Rebuild the source-series list and enable/disable state from
        the current figure. Called after any figure-content change or
        active-panel switch, not just a source-series edit."""
        raise NotImplementedError

    def load_result(self, result: AnalysisResult | None) -> None:
        """Restore `result` (if it belongs to this section's own result
        type; `None` or any other type clears this section's working
        result) as the working state, without rerunning any analysis --
        called whenever the shared Analysis History selection changes."""
        raise NotImplementedError

    def add_manual_peak(self, x: float, y: float) -> None:
        """Called by MainWindow after a canvas click while
        `is_manual_peak_mode()` is armed, with the click's data
        coordinates (`event.xdata`, `event.ydata`) -- `x`/`y` are
        deliberately generic here; a subclass's own override may name its
        parameters for what they actually mean (e.g. `two_theta,
        intensity` or `potential_v, current_a`)."""
        raise NotImplementedError

    def _set_manual_peak_mode(self, active: bool) -> None:
        """Arm/disarm manual-peak-add mode and update this section's own
        button/status-text side effects -- these differ enough between
        current subclasses (e.g. a redundant-call guard and an extra
        status message) that this stays a required override.
        `disarm_manual_peak_mode()` below is the one shared PUBLIC entry
        point every external caller actually uses."""
        raise NotImplementedError

    # --- shared: identical across every current subclass --------------------

    def set_manager(self, dataset_manager: DatasetManager) -> None:
        """Repoint at a different `DatasetManager` (Open/New Project only
        -- datasets are project-scoped, not figure-scoped)."""
        self._manager = dataset_manager

    def is_manual_peak_mode(self) -> bool:
        return self._manual_peak_mode

    def set_selected_peak_rows(self, rows: list[int]) -> None:
        """Record which peak/candidate rows are selected in the bottom
        Results-tab detail table -- pushed in by MainWindow whenever that
        table's selection changes."""
        self._results_selected_rows = sorted({int(r) for r in rows})

    def disarm_manual_peak_mode(self) -> None:
        """Publicly disarm manual-peak-add mode (a no-op if it wasn't
        armed) -- called from every context change where an already-armed
        click target stops making sense: a source-series/Workbench/
        project change, an active-panel switch, or switching away from
        this tool in `AnalysisPanel`. A successful `add_manual_peak` or
        the researcher toggling the button off both already disarm
        directly; this method exists for every OTHER exit path, so none
        of them can leave a stale armed state (and its checked button/
        status text) behind."""
        self._set_manual_peak_mode(False)
