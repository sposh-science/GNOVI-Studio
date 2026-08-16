import pandas as pd
from PySide6.QtWidgets import QTableView

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.dataset_panel import DatasetPanel

# --- Plot page dataset selector -------------------------------------------------
#
# `active_dataset_combo` on the Plot page directly selects the current
# dataset -- kept bidirectionally synchronized with `dataset_list`'s
# selection (Data page) so there is exactly one "current dataset" state,
# never a second, independent one. See
# `DatasetPanel._sync_dataset_combo`/`_on_dataset_combo_changed`.


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return Dataset(name=name, dataframe=df)


def _make_panel(*datasets):
    manager = DatasetManager()
    for dataset in datasets:
        manager.add(dataset)
    preview_table = QTableView()
    panel = DatasetPanel(manager, preview_table)
    return panel, manager


def _combo_dataset_ids(panel) -> list[str | None]:
    combo = panel.active_dataset_combo
    return [combo.itemData(i) for i in range(combo.count())]


def test_combo_shows_only_the_placeholder_when_no_dataset_exists(qapp):
    panel, _manager = _make_panel()

    assert panel.active_dataset_combo.count() == 1
    assert panel.active_dataset_combo.currentText() == "(no dataset)"
    panel.close()


def test_combo_lists_every_imported_dataset(qapp):
    d1 = _make_dataset("First")
    d2 = _make_dataset("Second")
    panel, _manager = _make_panel(d1, d2)

    panel._refresh_list(select_id=d1.id)

    assert set(_combo_dataset_ids(panel)) == {d1.id, d2.id}
    panel.close()


def test_selecting_in_data_page_list_updates_the_plot_page_combo(qapp):
    d1 = _make_dataset("First")
    d2 = _make_dataset("Second")
    panel, _manager = _make_panel(d1, d2)

    panel._refresh_list(select_id=d2.id)

    assert panel.active_dataset_combo.currentData() == d2.id
    assert panel.active_dataset_combo.currentText() == "Second"
    panel.close()


def test_selecting_in_the_plot_page_combo_updates_the_data_page_list(qapp):
    d1 = _make_dataset("First")
    d2 = _make_dataset("Second")
    panel, _manager = _make_panel(d1, d2)
    panel._refresh_list(select_id=d1.id)

    index = panel.active_dataset_combo.findData(d2.id)
    panel.active_dataset_combo.setCurrentIndex(index)

    assert panel.current_dataset is d2
    assert panel.dataset_list.currentItem().text() == "Second"
    panel.close()


def test_combo_selection_drives_exactly_what_add_to_plot_uses(qapp):
    d1 = _make_dataset("First")
    d2 = _make_dataset("Second")
    panel, _manager = _make_panel(d1, d2)
    panel._refresh_list(select_id=d1.id)

    index = panel.active_dataset_combo.findData(d2.id)
    panel.active_dataset_combo.setCurrentIndex(index)

    captured = []
    panel.add_to_plot_requested.connect(captured.append)
    panel.x_combo.setCurrentText("x")
    panel.y_combo.setCurrentText("y")
    panel._on_add_to_plot_clicked()

    assert len(captured) == 1
    assert captured[0][0].dataset is d2
    panel.close()


def test_removing_the_selected_dataset_updates_the_combo(qapp):
    dataset = _make_dataset("Only One")
    panel, manager = _make_panel(dataset)
    panel._refresh_list(select_id=dataset.id)
    assert panel.active_dataset_combo.currentData() == dataset.id

    manager.remove(dataset.id)
    panel._refresh_list()

    assert panel.active_dataset_combo.count() == 1
    assert panel.active_dataset_combo.currentText() == "(no dataset)"
    panel.close()


def test_removing_a_non_selected_dataset_drops_it_from_the_combo(qapp):
    d1 = _make_dataset("First")
    d2 = _make_dataset("Second")
    panel, manager = _make_panel(d1, d2)
    panel._refresh_list(select_id=d1.id)

    manager.remove(d2.id)
    panel._refresh_list(select_id=d1.id)

    assert set(_combo_dataset_ids(panel)) == {d1.id}
    assert panel.active_dataset_combo.currentData() == d1.id
    panel.close()


def test_set_manager_clears_the_combo(qapp):
    """Repointing at a fresh DatasetManager (Open/New Project) must not
    leave a previous project's datasets in the combo."""
    dataset = _make_dataset("Old Project Dataset")
    panel, _manager = _make_panel(dataset)
    panel._refresh_list(select_id=dataset.id)
    assert panel.active_dataset_combo.count() == 1

    panel.set_manager(DatasetManager())

    assert panel.active_dataset_combo.count() == 1
    assert panel.active_dataset_combo.currentText() == "(no dataset)"
    panel.close()


def test_set_manager_populates_the_combo_from_the_new_manager(qapp):
    old_dataset = _make_dataset("Old")
    panel, _old_manager = _make_panel(old_dataset)
    panel._refresh_list(select_id=old_dataset.id)

    new_dataset = _make_dataset("New")
    new_manager = DatasetManager()
    new_manager.add(new_dataset)
    panel.set_manager(new_manager)

    assert new_dataset.id in _combo_dataset_ids(panel)
    assert old_dataset.id not in _combo_dataset_ids(panel)
    panel.close()


def test_long_dataset_name_is_elided_in_the_combo_but_full_in_the_tooltip(qapp):
    long_name = "S1 (Ferricyanide Potassium Hexacyanoferrate) Scan Rate 0.05 V per second"
    dataset = _make_dataset(long_name)
    panel, _manager = _make_panel(dataset)

    panel._refresh_list(select_id=dataset.id)

    assert len(panel.active_dataset_combo.currentText()) < len(long_name)
    assert panel.active_dataset_combo.currentText().endswith("…")
    assert panel.active_dataset_combo.toolTip() == long_name
    panel.close()
