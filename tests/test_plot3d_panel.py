"""`Plot3DPanel` -- the "3D" sidebar page's own widget-level behavior
(dataset/column selection, validation, series-list summary, signal
emission). Creation-safety decisions (empty vs. populated 2D panel,
append-to-existing-Panel3D) are NOT this panel's concern -- see
`gui.main_window.MainWindow._on_add_3d_series_requested` and
`test_3d_scatter_gui.py` for those, since they need `GnoviFigure.
active_panel`'s type/content, which only the owner resolves.
"""

import pandas as pd

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.plot3d_panel import Plot3DPanel
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series3d import Series3D


def _make_dataset(name="mat"):
    df = pd.DataFrame(
        {"temperature": [300.0, 350.0, 400.0], "composition": [0.1, 0.15, 0.2], "conductivity": [2.4, 2.9, 3.5]}
    )
    return Dataset(name=name, dataframe=df)


def _make_panel(*datasets, figure=None):
    manager = DatasetManager()
    for dataset in datasets:
        manager.add(dataset)
    figure = figure if figure is not None else GnoviFigure()
    panel = Plot3DPanel(manager, figure)
    return panel, manager, figure


# --- Dataset/column selection --------------------------------------------------------


def test_dataset_combo_lists_every_dataset(qapp):
    d1, d2 = _make_dataset("First"), _make_dataset("Second")
    panel, _manager, _figure = _make_panel(d1, d2)

    ids = {panel.dataset_combo.itemData(i) for i in range(panel.dataset_combo.count())}

    assert ids == {d1.id, d2.id}
    panel.close()


def test_selecting_a_dataset_populates_xyz_column_combos(qapp):
    dataset = _make_dataset()
    panel, _manager, _figure = _make_panel(dataset)

    columns = {panel.x_combo.itemText(i) for i in range(panel.x_combo.count())}

    assert columns == {"temperature", "composition", "conductivity"}
    panel.close()


def test_set_manager_refreshes_the_dataset_combo(qapp):
    panel, _manager, _figure = _make_panel()
    assert panel.dataset_combo.count() == 0

    new_manager = DatasetManager()
    new_manager.add(_make_dataset())
    panel.set_manager(new_manager)

    assert panel.dataset_combo.count() == 1
    panel.close()


# --- Add 3D Series: validation, emitted Series3D --------------------------------------


def test_add_clicked_with_valid_selection_emits_a_series3d(qapp):
    dataset = _make_dataset()
    panel, _manager, _figure = _make_panel(dataset)
    panel.x_combo.setCurrentText("temperature")
    panel.y_combo.setCurrentText("composition")
    panel.z_combo.setCurrentText("conductivity")
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()

    assert len(emitted) == 1
    series = emitted[0]
    assert isinstance(series, Series3D)
    assert series.dataset is dataset
    assert series.x_column == "temperature"
    assert series.y_column == "composition"
    assert series.z_column == "conductivity"
    assert series.color is None  # assigned later by Panel3D.add_series
    panel.close()


def test_add_clicked_with_no_dataset_shows_an_error_and_emits_nothing(qapp):
    panel, _manager, _figure = _make_panel()
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()

    assert emitted == []
    assert panel.error_label.text() != ""
    panel.close()


def test_add_clicked_with_non_numeric_columns_shows_a_controlled_error(qapp):
    bad = Dataset(name="bad", dataframe=pd.DataFrame({"a": ["p", "q"], "b": ["p", "q"], "c": ["p", "q"]}))
    panel, _manager, _figure = _make_panel(bad)
    panel.x_combo.setCurrentText("a")
    panel.y_combo.setCurrentText("b")
    panel.z_combo.setCurrentText("c")
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()  # must not raise

    assert emitted == []
    assert panel.error_label.text() != ""
    panel.close()


def test_plot_type_combo_offers_only_scatter_this_milestone(qapp):
    panel, _manager, _figure = _make_panel(_make_dataset())

    options = [panel.plot_type_combo.itemText(i) for i in range(panel.plot_type_combo.count())]

    assert options == ["Scatter"]
    panel.close()


# --- Clear 3D Plot signal --------------------------------------------------------------


def test_clear_button_emits_clear_3d_plot_requested(qapp):
    dataset = _make_dataset()
    panel3d = Panel3D()
    panel3d.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    figure = GnoviFigure(panels=[panel3d])
    panel, _manager, _figure = _make_panel(dataset, figure=figure)
    emitted = []
    panel.clear_3d_plot_requested.connect(lambda: emitted.append(True))

    assert panel.clear_button.isEnabled() is True  # non-empty Panel3D -- see `refresh`
    panel.clear_button.click()

    assert emitted == [True]
    panel.close()


# --- 3D series list summary, reflecting the active panel ------------------------------


def test_refresh_shows_no_series_and_disables_clear_for_a_2d_active_panel(qapp):
    figure = GnoviFigure()
    panel, _manager, _figure = _make_panel(_make_dataset(), figure=figure)

    assert panel.series_list.count() == 0
    assert panel.clear_button.isEnabled() is False
    panel.close()


def test_refresh_lists_the_active_panel3ds_series_and_enables_clear(qapp):
    dataset = _make_dataset()
    panel3d = Panel3D()
    panel3d.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", label="mat"))
    figure = GnoviFigure(panels=[panel3d])
    panel, _manager, _figure = _make_panel(dataset, figure=figure)

    assert panel.series_list.count() == 1
    assert panel.series_list.item(0).text() == "mat"
    assert panel.clear_button.isEnabled() is True
    panel.close()


def test_set_figure_repoints_and_refreshes(qapp):
    dataset = _make_dataset()
    panel, _manager, _figure = _make_panel(dataset)
    assert panel.series_list.count() == 0

    panel3d = Panel3D()
    panel3d.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    new_figure = GnoviFigure(panels=[panel3d])
    panel.set_figure(new_figure)

    assert panel.series_list.count() == 1
    panel.close()
