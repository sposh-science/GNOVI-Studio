import pandas as pd

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.undo_manager import UndoManager, snapshot_figure
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return Dataset(name=name, dataframe=df)


# --- snapshot_figure -------------------------------------------------------


def test_snapshot_is_a_distinct_object_with_equal_content():
    manager = DatasetManager()
    dataset = _make_dataset()
    manager.add(dataset)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))
    figure.active_panel.title = "Original"

    snapshot = snapshot_figure(figure, manager)

    assert snapshot is not figure
    assert snapshot.panels[0] is not figure.panels[0]
    assert snapshot.panels[0].series[0] is not figure.panels[0].series[0]
    assert snapshot.active_panel.title == "Original"


def test_snapshot_reuses_the_live_dataset_object_by_identity():
    """The whole point of pre-seeding the deepcopy memo: a snapshot's
    PlotSeries must reference the exact same Dataset the live figure uses,
    never a copy -- otherwise undo/redo would silently diverge a series
    from the dataset's current working data, and every snapshot would deep-
    copy the DataFrame (expensive, and pointless for undo correctness)."""
    manager = DatasetManager()
    dataset = _make_dataset()
    manager.add(dataset)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))

    snapshot = snapshot_figure(figure, manager)

    assert snapshot.panels[0].series[0].dataset is dataset


def test_mutating_the_live_figure_after_snapshotting_does_not_affect_the_snapshot():
    manager = DatasetManager()
    dataset = _make_dataset()
    manager.add(dataset)
    figure = GnoviFigure()
    figure.active_panel.title = "Before"

    snapshot = snapshot_figure(figure, manager)
    figure.active_panel.title = "After"

    assert snapshot.active_panel.title == "Before"
    assert figure.active_panel.title == "After"


# --- UndoManager -------------------------------------------------------------


def test_fresh_manager_cannot_undo_or_redo():
    manager = UndoManager()
    assert manager.can_undo is False
    assert manager.can_redo is False


def test_push_then_undo_returns_the_pushed_snapshot():
    manager = UndoManager()
    manager.push("state-0")

    result = manager.undo("state-1")

    assert result == "state-0"
    assert manager.can_undo is False
    assert manager.can_redo is True


def test_undo_with_nothing_pushed_returns_none():
    manager = UndoManager()
    assert manager.undo("current") is None


def test_undo_then_redo_round_trips():
    manager = UndoManager()
    manager.push("state-0")

    previous = manager.undo("state-1")
    restored = manager.redo(previous)

    assert previous == "state-0"
    assert restored == "state-1"
    assert manager.can_redo is False


def test_redo_with_nothing_undone_returns_none():
    manager = UndoManager()
    manager.push("state-0")
    assert manager.redo("current") is None


def test_a_new_push_clears_the_redo_stack():
    manager = UndoManager()
    manager.push("state-0")
    manager.undo("state-1")
    assert manager.can_redo is True

    manager.push("state-2")

    assert manager.can_redo is False


def test_bounded_history_drops_the_oldest_entry():
    manager = UndoManager(max_entries=3)
    for i in range(5):
        manager.push(f"state-{i}")

    # Only the 3 most recent pushes survive -- undoing 3 times exhausts it.
    seen = []
    current = "current"
    for _ in range(3):
        current = manager.undo(current)
        seen.append(current)
    assert manager.can_undo is False
    assert seen == ["state-4", "state-3", "state-2"]


def test_clear_empties_both_stacks():
    manager = UndoManager()
    manager.push("state-0")
    manager.undo("state-1")

    manager.clear()

    assert manager.can_undo is False
    assert manager.can_redo is False
