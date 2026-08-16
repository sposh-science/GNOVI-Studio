"""`WorkbenchTabBar` -- the compact Workbench switcher docked above the
Workbench header (see `gui.widgets.workbench_tabs`). Widget-level tests
only; `tests/test_main_window_project.py` covers it wired through the real
`MainWindow`.
"""

from PySide6.QtCore import QPoint

from gnovi_plot.core.workbench import Workbench
from gnovi_plot.gui.widgets.workbench_tabs import WorkbenchTabBar
from gnovi_plot.plotting.figure import GnoviFigure


def _workbenches(*names):
    return [Workbench(name=name, figure=GnoviFigure()) for name in names]


class _FakeAction:
    """Stand-in for a real `QAction` -- just enough surface
    (`setEnabled`/identity) for `WorkbenchTabBar._on_context_menu_requested`
    to run against without a real, event-loop-driven `QMenu.exec()`."""

    def __init__(self, text):
        self.text = text
        self.enabled = True

    def setEnabled(self, value):
        self.enabled = value


def _fake_menu_returning(chosen_text: str):
    """A `QMenu` stand-in whose `exec()` immediately "clicks" the action
    with `chosen_text` (or None, if `chosen_text` is None -- Cancel)."""

    class _FakeMenu:
        def __init__(self, *a, **k):
            self.actions = {}

        def addAction(self, text):
            action = _FakeAction(text)
            self.actions[text] = action
            return action

        def exec(self, *a, **k):
            return self.actions.get(chosen_text) if chosen_text is not None else None

    return _FakeMenu


# --- Populating tabs -----------------------------------------------------------


def test_set_workbenches_creates_one_tab_per_workbench(qapp):
    bar = WorkbenchTabBar()
    workbenches = _workbenches("CV Comparison", "New Scan", "Publication Figure")

    bar.set_workbenches(workbenches, workbenches[0].id)

    assert bar.tab_bar.count() == 3
    assert [bar.tab_bar.tabText(i) for i in range(3)] == [
        "CV Comparison",
        "New Scan",
        "Publication Figure",
    ]


def test_set_workbenches_selects_the_given_active_id(qapp):
    bar = WorkbenchTabBar()
    workbenches = _workbenches("A", "B", "C")

    bar.set_workbenches(workbenches, workbenches[1].id)

    assert bar.tab_bar.currentIndex() == 1
    assert bar.current_workbench_id() == workbenches[1].id


def test_set_workbenches_does_not_emit_workbench_selected(qapp):
    """Reflecting state the caller already knows about is not the same as
    requesting a switch -- rebuilding the tabs must never re-trigger the
    switch handler."""
    bar = WorkbenchTabBar()
    workbenches = _workbenches("A", "B")
    received = []
    bar.workbench_selected.connect(received.append)

    bar.set_workbenches(workbenches, workbenches[1].id)

    assert received == []


def test_long_names_are_exposed_via_tooltip(qapp):
    bar = WorkbenchTabBar()
    long_name = "Ferricyanide Potassium Hexacyanoferrate Scan Rate Comparison"
    workbenches = _workbenches(long_name)

    bar.set_workbenches(workbenches, workbenches[0].id)

    assert bar.tab_bar.tabToolTip(0) == long_name


def test_tab_bar_never_has_a_native_close_button(qapp):
    """Close-by-misclick is exactly what this design avoids -- Rename/
    Duplicate/Delete are reached via context menu or the Workbench menu."""
    bar = WorkbenchTabBar()
    assert bar.tab_bar.tabsClosable() is False


# --- Switching -------------------------------------------------------------


def test_clicking_a_different_tab_emits_workbench_selected(qapp):
    bar = WorkbenchTabBar()
    workbenches = _workbenches("A", "B")
    bar.set_workbenches(workbenches, workbenches[0].id)
    received = []
    bar.workbench_selected.connect(received.append)

    bar.tab_bar.setCurrentIndex(1)

    assert received == [workbenches[1].id]


def test_reselecting_the_same_tab_does_not_emit_again(qapp):
    bar = WorkbenchTabBar()
    workbenches = _workbenches("A", "B")
    bar.set_workbenches(workbenches, workbenches[0].id)
    received = []
    bar.workbench_selected.connect(received.append)

    bar.tab_bar.setCurrentIndex(0)  # already index 0 -- Qt itself won't re-fire currentChanged

    assert received == []


# --- New Workbench button ---------------------------------------------------


def test_new_button_click_emits_new_workbench_requested(qapp):
    bar = WorkbenchTabBar()
    received = []
    bar.new_workbench_requested.connect(lambda: received.append(True))

    bar.new_button.click()

    assert received == [True]


# --- Context menu: Rename / Duplicate / Delete ------------------------------


def test_context_menu_rename_emits_rename_requested(qapp, monkeypatch):
    bar = WorkbenchTabBar()
    workbenches = _workbenches("A", "B")
    bar.set_workbenches(workbenches, workbenches[0].id)
    received = []
    bar.rename_requested.connect(received.append)
    monkeypatch.setattr(
        "gnovi_plot.gui.widgets.workbench_tabs.QMenu", _fake_menu_returning("Rename Workbench")
    )

    bar._on_context_menu_requested(QPoint(5, 5))

    assert received == [workbenches[0].id]


def test_context_menu_duplicate_emits_duplicate_requested(qapp, monkeypatch):
    bar = WorkbenchTabBar()
    workbenches = _workbenches("A", "B")
    bar.set_workbenches(workbenches, workbenches[0].id)
    received = []
    bar.duplicate_requested.connect(received.append)
    monkeypatch.setattr(
        "gnovi_plot.gui.widgets.workbench_tabs.QMenu", _fake_menu_returning("Duplicate Workbench")
    )

    bar._on_context_menu_requested(QPoint(5, 5))

    assert received == [workbenches[0].id]


def test_context_menu_delete_emits_delete_requested(qapp, monkeypatch):
    bar = WorkbenchTabBar()
    workbenches = _workbenches("A", "B")
    bar.set_workbenches(workbenches, workbenches[0].id)
    received = []
    bar.delete_requested.connect(received.append)
    monkeypatch.setattr(
        "gnovi_plot.gui.widgets.workbench_tabs.QMenu", _fake_menu_returning("Delete Workbench")
    )

    bar._on_context_menu_requested(QPoint(5, 5))

    assert received == [workbenches[0].id]


def test_context_menu_cancel_emits_nothing(qapp, monkeypatch):
    bar = WorkbenchTabBar()
    workbenches = _workbenches("A", "B")
    bar.set_workbenches(workbenches, workbenches[0].id)
    received = []
    bar.rename_requested.connect(received.append)
    bar.duplicate_requested.connect(received.append)
    bar.delete_requested.connect(received.append)
    monkeypatch.setattr("gnovi_plot.gui.widgets.workbench_tabs.QMenu", _fake_menu_returning(None))

    bar._on_context_menu_requested(QPoint(5, 5))

    assert received == []


def test_context_menu_delete_is_disabled_with_only_one_workbench(qapp, monkeypatch):
    bar = WorkbenchTabBar()
    workbenches = _workbenches("Only One")
    bar.set_workbenches(workbenches, workbenches[0].id)
    captured = {}

    class _FakeMenu:
        def __init__(self, *a, **k):
            pass

        def addAction(self, text):
            action = _FakeAction(text)
            captured[text] = action
            return action

        def exec(self, *a, **k):
            return None

    monkeypatch.setattr("gnovi_plot.gui.widgets.workbench_tabs.QMenu", _FakeMenu)

    bar._on_context_menu_requested(QPoint(5, 5))

    assert captured["Delete Workbench"].enabled is False


def test_context_menu_at_an_empty_area_does_not_raise(qapp):
    bar = WorkbenchTabBar()
    # No workbenches at all -- tabAt() returns -1 everywhere.
    bar._on_context_menu_requested(QPoint(5, 5))  # must not raise
