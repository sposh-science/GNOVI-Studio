from __future__ import annotations

import logging
from dataclasses import dataclass, field

from gnovi_plot.analysis.results import AnalysisResult, result_from_dict

_logger = logging.getLogger("gnovi_plot")


@dataclass
class PanelResultHistory:
    """Per-panel analysis-result history for one `core.workbench.Workbench`
    -- keyed exclusively by stable `Panel.id` (never a panel index or
    "Panel N" display text, both positional and unstable across layout
    changes; never Python object identity either).

    Owned directly by `Workbench` (see `Workbench.analysis_results`)
    rather than a separate manager: this state is intrinsically part of
    what a Workbench's own working analysis history is, so two different
    Workbenches (even with structurally identical panels) never share or
    cross-contaminate history.

    Each panel may accumulate multiple results (one per completed fit,
    oldest first) -- running a new fit always appends, never overwrites
    or discards an earlier one. Alongside the ordered list, each panel
    also tracks its own *current* result -- the one Results/History-list
    selection/Add-Remove Fit Curve should act on. Adding a result always
    makes it current (`add()`); a scientist can also explicitly pick an
    older one back to current (`set_current()`), e.g. from the Analysis
    page's History list -- see `gui.widgets.analysis_panel.AnalysisPanel`.
    Selecting an older result never reorders or removes anything: `all()`
    still returns the full history, oldest first, regardless of which
    entry is current.

    Plain Python, no Qt. Persisted as part of `Workbench.to_dict()`/
    `from_dict()` -- see `to_dict()`/`from_dict()` below and this
    module's `result_from_dict` import for how a stored entry's `kind`
    reconstructs the correct concrete `AnalysisResult` subclass without
    this class (or `Workbench`/`core.project_io`) ever hard-coding
    `FitResult` specifically.
    """

    _by_panel: dict[str, list[AnalysisResult]] = field(default_factory=dict)
    _current: dict[str, str] = field(default_factory=dict)  # panel_id -> result_id

    def add(self, panel_id: str, result: AnalysisResult) -> None:
        """Append `result` to `panel_id`'s history -- it becomes that
        panel's `current()` result, overriding any earlier explicit
        selection (a genuinely new result is always the most relevant
        one to show). Never overwrites or removes any earlier result for
        this or any other panel."""
        self._by_panel.setdefault(panel_id, []).append(result)
        self._current[panel_id] = result.result_id

    def set_current(self, panel_id: str, result_id: str) -> None:
        """Explicitly select `result_id` (must already be in `panel_id`'s
        history) as that panel's `current()` result -- e.g. the scientist
        picking an older entry from the Analysis History list. Never
        reorders or mutates the history itself. Raises `ValueError` if
        `result_id` doesn't belong to `panel_id`'s history; a defensive
        backstop only -- the GUI only ever passes an id it just read out
        of that same panel's own `all(panel_id)`."""
        if not any(r.result_id == result_id for r in self._by_panel.get(panel_id, [])):
            raise ValueError(f"Result {result_id!r} is not in panel {panel_id!r}'s history")
        self._current[panel_id] = result_id

    def current(self, panel_id: str) -> AnalysisResult | None:
        """`panel_id`'s currently selected result, or `None` if that panel
        has no history at all. Prefers the explicit `_current` marker;
        falls back to the most recently added entry if no marker is set
        (a project saved before `set_current` existed) or the marker no
        longer matches anything in the history (defensive)."""
        history = self._by_panel.get(panel_id)
        if not history:
            return None
        target = self._current.get(panel_id)
        return next((r for r in history if r.result_id == target), history[-1])

    def all(self, panel_id: str) -> list[AnalysisResult]:
        """`panel_id`'s full history, oldest first -- a new list each
        call, so callers can never accidentally mutate internal state
        through the returned list."""
        return list(self._by_panel.get(panel_id, []))

    def clear_panel(self, panel_id: str) -> None:
        """Drop all history (and the current-selection marker) for
        `panel_id`. A no-op if it has none."""
        self._by_panel.pop(panel_id, None)
        self._current.pop(panel_id, None)

    def prune_to(self, valid_panel_ids: set[str]) -> None:
        """Drop every panel's history (and current-selection marker)
        whose id is not in `valid_panel_ids` -- call after a layout
        change removes panels, so a later panel reusing a coincidentally-
        freed id (it never will, in practice -- ids are `uuid.uuid4().hex`
        -- but this is the correctness backstop) can never inherit an
        orphaned panel's history. Histories for panels still present are
        left untouched."""
        for stale_id in set(self._by_panel) - valid_panel_ids:
            self._by_panel.pop(stale_id, None)
            self._current.pop(stale_id, None)

    def to_dict(self) -> dict:
        """Project-save representation: `{panel_id: {"history": [...],
        "current_result_id": ...}}`, history order preserved (oldest
        first). Residual arrays are never part of this -- `AnalysisResult.
        to_dict()` never includes them (see `analysis.fitting.
        compute_residuals`'s own docstring)."""
        return {
            panel_id: {
                "history": [result.to_dict() for result in results],
                "current_result_id": self._current.get(panel_id),
            }
            for panel_id, results in self._by_panel.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PanelResultHistory":
        """Reconstruct from `to_dict()`'s output. Also accepts the older,
        pre-selection shape (`{panel_id: [result_dict, ...]}`, no explicit
        current marker) for a project saved before `set_current` existed
        -- each panel's value is a plain `list` in that shape, a `dict`
        (with `"history"`/`"current_result_id"`) in the current one, so
        the two are distinguished per-panel by that type alone. Either
        way, `add()`'s own "latest becomes current" behavior already
        gives the right default (most recent result) when no valid
        `current_result_id` is available -- so an old-shape entry, or a
        `current_result_id` that doesn't match anything (e.g. it names a
        result of an unknown `kind`, dropped below), naturally falls back
        to the latest, preserving pre-selection behavior exactly.

        An entry whose `kind` isn't recognized (e.g. saved by a newer app
        version with an analysis tool this one doesn't have) is logged
        and skipped, not fatal -- one unreconstructable result shouldn't
        sink an otherwise valid project (same tolerance
        `core.project_io.load_project` already gives a stale `PlotSeries`
        reference)."""
        history = cls()
        for panel_id, entry in data.items():
            entries = entry if isinstance(entry, list) else entry.get("history", [])
            current_result_id = None if isinstance(entry, list) else entry.get("current_result_id")

            for result_data in entries:
                try:
                    history.add(panel_id, result_from_dict(result_data))
                except ValueError:
                    _logger.warning(
                        "Dropped an analysis result of unknown kind %r for panel %s on project load.",
                        result_data.get("kind"),
                        panel_id,
                    )

            if current_result_id is not None:
                try:
                    history.set_current(panel_id, current_result_id)
                except ValueError:
                    pass  # names a dropped/unknown result -- keep add()'s latest-result default
        return history
