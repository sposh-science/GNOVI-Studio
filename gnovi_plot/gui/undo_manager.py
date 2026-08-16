from __future__ import annotations

import copy

from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.graph import dataset_identity_memo

# Bound on how many undo steps are kept -- snapshots are cheap (see
# `snapshot_figure`) but unbounded growth over a long session is still
# wasteful.
_DEFAULT_MAX_ENTRIES = 50


class UndoManager:
    """Bounded undo/redo stack of full-figure snapshots for the plotting
    layer (panels/series/styling/layout) only.

    Deliberately scoped to the FIGURE, not the Dataset: dataset mutations
    (Calculated Column, Keep/Exclude Selected Rows, Reset Working Data)
    already have their own recovery path -- Reset Working Data -- and are
    tracked for the user in the Transformation History list (see
    `gui.widgets.data_tools_panel`/`bottom_panel`), which this deliberately
    stays independent from: Undo/Redo here never touches a Dataset's
    `dataframe`, and the Transformation History never gains "undo" entries
    for plot edits. Folding dataset content into the same stack would also
    mean deep-copying a DataFrame on every edit, which `snapshot_figure`
    specifically avoids.
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES):
        self._undo: list[GnoviFigure] = []
        self._redo: list[GnoviFigure] = []
        self._max_entries = max_entries

    def push(self, snapshot: GnoviFigure) -> None:
        """Record `snapshot` as a state an undo can return to, and drop the
        redo stack -- a fresh edit always invalidates any old "future"."""
        self._undo.append(snapshot)
        if len(self._undo) > self._max_entries:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, current: GnoviFigure) -> GnoviFigure | None:
        """Pop and return the most recent undo snapshot, pushing `current`
        (the state being left) onto the redo stack. None if nothing to undo."""
        if not self._undo:
            return None
        self._redo.append(current)
        return self._undo.pop()

    def redo(self, current: GnoviFigure) -> GnoviFigure | None:
        """Pop and return the most recent redo snapshot, pushing `current`
        back onto the undo stack. None if nothing to redo."""
        if not self._redo:
            return None
        self._undo.append(current)
        return self._redo.pop()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()


def snapshot_figure(figure: GnoviFigure, dataset_manager: DatasetManager) -> GnoviFigure:
    """Deep-copy `figure` (Panels/PlotSeries/layout/typography/grid -- every
    attribute) for the undo stack, but reuse every live `Dataset` by
    identity rather than copying it: `copy.deepcopy`'s memo is pre-seeded
    with `id(dataset) -> dataset` for each dataset currently in
    `dataset_manager`, so a `PlotSeries.dataset` reference short-circuits
    straight back to the same object instead of recursing into (and
    duplicating) its DataFrame. This keeps snapshotting cheap regardless of
    dataset size, and keeps every snapshot's series pointed at the one live
    Dataset -- exactly like the figure being snapshotted, undo/redo never
    diverges a series from the dataset's current working data.
    """
    return copy.deepcopy(figure, dataset_identity_memo(dataset_manager))
