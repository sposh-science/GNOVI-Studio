"""Panel right-click context menu -- "Extract Panel to New Workbench" (see
`MainWindow._on_canvas_context_menu`/`_show_panel_context_menu`).

The context menu is deliberately UI plumbing only: every action it offers
invokes the exact same handler the Panels menu already uses (here,
`MainWindow._on_extract_panel_requested`, unchanged from
`test_extract_panel_to_workbench_gui.py`'s own coverage of that handler and
`core.project.Project.extract_panel_to_workbench` beneath it) -- these tests
exist to prove the *targeting* and *reuse* behavior, not to re-prove
extraction semantics already covered there.
"""

from gnovi_plot.gui.main_window import MainWindow


class _FakeAction:
    def __init__(self, text):
        self.text = text


def _fake_menu(chosen_text=None, exec_calls=None):
    """A `QMenu` stand-in whose `exec()` immediately "clicks" the action
    with `chosen_text` (or None, if `chosen_text` is None -- Cancel), and
    optionally records every `exec()` call -- same shape as
    `test_workbench_tabs.py`'s `_fake_menu_returning`, extended with an
    `exec_calls` sink so a test can assert a menu was actually opened even
    when it wants nothing chosen from it."""

    class _FakeMenu:
        def __init__(self, *_a, **_k):
            self.actions = {}

        def addSeparator(self):
            pass

        def addAction(self, text):
            action = _FakeAction(text)
            self.actions[text] = action
            return action

        def exec(self, *_a, **_k):
            if exec_calls is not None:
                exec_calls.append(True)
            return self.actions.get(chosen_text) if chosen_text is not None else None

    return _FakeMenu


class _FakeContextMenuEvent:
    """Minimal stand-in for a Matplotlib `button_press_event` -- only the
    attributes `_on_canvas_context_menu`/`_canvas_event_global_pos` actually
    read. `button=3` is Matplotlib's right-click button code (also what
    `matplotlib.backend_bases.MouseButton.RIGHT` compares equal to, since
    it's an `IntEnum`)."""

    def __init__(self, inaxes=None, button=3, x=0, y=0):
        self.inaxes = inaxes
        self.button = button
        self.x = x
        self.y = y


def _make_3_panel_window():
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(4)  # "1 x 3"
    return window


# --- Targeting: right-click resolves and activates the clicked Panel -----------


def test_right_click_on_inactive_panel_targets_that_panel(qapp, monkeypatch):
    window = _make_3_panel_window()
    assert window.figure_model.active_panel_index == 0
    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu(None))
    clicked_axes = window.plot_canvas.axes_list[2]

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=clicked_axes))

    assert window.figure_model.active_panel_index == 2
    window.close()


def test_right_click_on_active_panel_still_opens_context_menu(qapp, monkeypatch):
    """A plain left-click's "already active -- nothing to do" early return
    must not suppress the menu: right-clicking the already-active Panel
    still has to open the menu, even though there's no Panel to activate."""
    window = _make_3_panel_window()
    exec_calls = []
    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu(None, exec_calls=exec_calls))
    active_axes = window.plot_canvas.axes_list[window.figure_model.active_panel_index]

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=active_axes))

    assert window.figure_model.active_panel_index == 0  # unchanged
    assert exec_calls == [True]  # but the menu still opened
    window.close()


def test_left_click_does_not_open_a_context_menu(qapp, monkeypatch):
    window = _make_3_panel_window()
    exec_calls = []
    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu(None, exec_calls=exec_calls))
    clicked_axes = window.plot_canvas.axes_list[1]

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=clicked_axes, button=1))

    assert exec_calls == []
    window.close()


def test_right_click_outside_any_axes_is_a_no_op(qapp, monkeypatch):
    window = _make_3_panel_window()
    exec_calls = []
    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu(None, exec_calls=exec_calls))

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=None))  # must not raise

    assert exec_calls == []
    window.close()


# --- Extract action: reuses the existing extraction workflow -------------------


def test_context_extract_invokes_the_existing_extract_handler(qapp, monkeypatch):
    """Proves the context menu never gained its own extraction domain
    logic: it must go through the exact same `MainWindow.
    _on_extract_panel_requested` bound method the Panels menu action is
    wired to, not a parallel implementation."""
    window = _make_3_panel_window()
    calls = []
    original = window._on_extract_panel_requested

    def _spy():
        calls.append(True)
        original()

    window._on_extract_panel_requested = _spy
    monkeypatch.setattr(
        "gnovi_plot.gui.main_window.QMenu", _fake_menu("Extract Panel to New Workbench")
    )
    clicked_axes = window.plot_canvas.axes_list[2]

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=clicked_axes))

    assert calls == [True]
    window.close()


def test_context_extract_calls_the_same_domain_operation_exactly_once(qapp, monkeypatch):
    """No duplicate extraction domain path: `Project.
    extract_panel_to_workbench` -- the one domain operation both entry
    points must funnel through -- is called exactly once, with the
    Workbench/Panel ids the click resolved to."""
    from gnovi_plot.core.project import Project

    window = _make_3_panel_window()
    calls = []
    original_extract = Project.extract_panel_to_workbench

    def _spy(self, workbench_id, panel_id):
        calls.append((workbench_id, panel_id))
        return original_extract(self, workbench_id, panel_id)

    monkeypatch.setattr(Project, "extract_panel_to_workbench", _spy)
    monkeypatch.setattr(
        "gnovi_plot.gui.main_window.QMenu", _fake_menu("Extract Panel to New Workbench")
    )
    source_workbench_id = window._project.active_workbench_id
    target_panel_id = window.figure_model.panels[2].id
    clicked_axes = window.plot_canvas.axes_list[2]

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=clicked_axes))

    assert calls == [(source_workbench_id, target_panel_id)]
    window.close()


def test_context_extract_produces_the_same_kind_of_result_as_the_menu_action(qapp, monkeypatch):
    """Both entry points must leave the Project in the same shape: a new,
    independent 1x1 Workbench holding a clone of the targeted Panel, active,
    with the source Workbench's own layout/panels untouched."""
    menu_window = _make_3_panel_window()
    menu_window.toolbar_panel_combo.setCurrentIndex(2)
    menu_window._on_extract_panel_requested()

    context_window = _make_3_panel_window()
    monkeypatch.setattr(
        "gnovi_plot.gui.main_window.QMenu", _fake_menu("Extract Panel to New Workbench")
    )
    clicked_axes = context_window.plot_canvas.axes_list[2]
    context_window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=clicked_axes))

    for window in (menu_window, context_window):
        assert len(window._project.workbenches) == 2
        assert window.figure_model.layout == (1, 1)
        assert window._project.active_workbench_id == window._project.workbenches[-1].id
        source = window._project.workbenches[0]
        assert source.figure.layout == (1, 3)
        window.close()


def test_source_workbench_remains_unchanged_after_context_extract(qapp, monkeypatch):
    window = _make_3_panel_window()
    original_panel_ids = [p.id for p in window.figure_model.panels]
    monkeypatch.setattr(
        "gnovi_plot.gui.main_window.QMenu", _fake_menu("Extract Panel to New Workbench")
    )
    source_workbench_id = window._project.active_workbench_id
    clicked_axes = window.plot_canvas.axes_list[1]

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=clicked_axes))

    source = window._project.get_workbench(source_workbench_id)
    assert source.figure.layout == (1, 3)
    assert [p.id for p in source.figure.panels] == original_panel_ids
    window.close()


def test_right_clicking_different_panels_extracts_the_actually_clicked_panel(qapp, monkeypatch):
    """Panel 1 is active; right-clicking Panel 3 must extract Panel 3, not
    the previously-active Panel 1 -- the scenario called out explicitly in
    the feature's targeting requirement."""
    window = _make_3_panel_window()
    assert window.figure_model.active_panel_index == 0
    panel_3_id = window.figure_model.panels[2].id
    monkeypatch.setattr(
        "gnovi_plot.gui.main_window.QMenu", _fake_menu("Extract Panel to New Workbench")
    )
    clicked_axes = window.plot_canvas.axes_list[2]

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=clicked_axes))

    extracted = window._project.workbenches[-1]
    assert extracted.figure.panels[0].id != panel_3_id  # fresh id, per clone_panel_with_shared_datasets
    source = window._project.workbenches[0]
    assert source.figure.panels[2].id == panel_3_id  # the ORIGINAL Panel 3 is untouched
    window.close()


def test_context_menu_cancel_does_not_extract(qapp, monkeypatch):
    window = _make_3_panel_window()
    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu(None))
    clicked_axes = window.plot_canvas.axes_list[1]

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=clicked_axes))

    assert len(window._project.workbenches) == 1
    window.close()
