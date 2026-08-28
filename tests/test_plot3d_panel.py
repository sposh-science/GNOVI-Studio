"""`Plot3DPanel` -- the "3D" sidebar page's own widget-level behavior
(dataset/column selection, plot type, Group by, validation, "Clear 3D
Plot" state, signal emission). This page's own read-only series-list
summary was removed (see PR "Sidebar Navigation & 2D/3D Workflow Polish")
once the adaptive Series page's `series3d_list` was confirmed to fully
cover selecting/renaming/styling/removing/clearing every `Series3D` this
page creates -- see `test_sidebar_navigation.py` for the tests covering
that removal and the Series page's own coverage. Creation-safety decisions
(empty vs. populated 2D panel, append-to-existing-Panel3D) are NOT this
panel's concern -- see `gui.main_window.MainWindow.
_on_add_3d_series_requested` and `test_3d_scatter_gui.py` for those, since
they need `GnoviFigure.active_panel`'s type/content, which only the owner
resolves.
"""

import pandas as pd

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.plot3d_panel import Plot3DPanel
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series3d import Plot3DType, Series3D


def _make_dataset(name="mat"):
    df = pd.DataFrame(
        {"temperature": [300.0, 350.0, 400.0], "composition": [0.1, 0.15, 0.2], "conductivity": [2.4, 2.9, 3.5]}
    )
    return Dataset(name=name, dataframe=df)


def _make_diode_dataset(name="diode"):
    """Voltage/Temperature/Current -- the milestone's own worked example:
    two temperatures, 3 rows each, interleaved (not block-sorted) so
    grouping tests genuinely exercise non-contiguous row selection."""
    df = pd.DataFrame(
        {
            "Voltage_V": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
            "Temperature_C": [25.0, 35.0, 25.0, 35.0, 25.0, 35.0],
            "Current_mA": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        }
    )
    return Dataset(name=name, dataframe=df)


def _make_panel(*datasets, figure=None):
    manager = DatasetManager()
    for dataset in datasets:
        manager.add(dataset)
    figure = figure if figure is not None else GnoviFigure()
    panel = Plot3DPanel(manager, figure)
    return panel, manager, figure


def _fill_xyz(panel, x, y, z):
    panel.x_combo.setCurrentText(x)
    panel.y_combo.setCurrentText(y)
    panel.z_combo.setCurrentText(z)


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


def test_group_by_combo_lists_none_and_every_column(qapp):
    dataset = _make_dataset()
    panel, _manager, _figure = _make_panel(dataset)

    labels = [panel.group_by_combo.itemText(i) for i in range(panel.group_by_combo.count())]

    assert labels[0] == "None"
    assert set(labels[1:]) == {"temperature", "composition", "conductivity"}
    panel.close()


def test_set_manager_refreshes_the_dataset_combo(qapp):
    panel, _manager, _figure = _make_panel()
    assert panel.dataset_combo.count() == 0

    new_manager = DatasetManager()
    new_manager.add(_make_dataset())
    panel.set_manager(new_manager)

    assert panel.dataset_combo.count() == 1
    panel.close()


# --- Plot type combo ------------------------------------------------------------------


def test_plot_type_combo_offers_scatter_line_and_line_markers(qapp):
    panel, _manager, _figure = _make_panel(_make_dataset())

    options = [panel.plot_type_combo.itemText(i) for i in range(panel.plot_type_combo.count())]

    assert options == ["Scatter", "Line", "Line + Markers"]
    panel.close()


# --- Add 3D Series: validation, emitted Series3D list ---------------------------------


def test_add_clicked_with_valid_selection_emits_one_series3d_when_ungrouped(qapp):
    dataset = _make_dataset()
    panel, _manager, _figure = _make_panel(dataset)
    _fill_xyz(panel, "temperature", "composition", "conductivity")
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()

    assert len(emitted) == 1
    series_list = emitted[0]
    assert len(series_list) == 1
    series = series_list[0]
    assert isinstance(series, Series3D)
    assert series.dataset is dataset
    assert series.x_column == "temperature"
    assert series.y_column == "composition"
    assert series.z_column == "conductivity"
    assert series.row_indices is None  # ungrouped -- whole dataset, matches pre-grouping behavior
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
    _fill_xyz(panel, "a", "b", "c")
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()  # must not raise

    assert emitted == []
    assert panel.error_label.text() != ""
    panel.close()


def test_add_clicked_default_plot_type_is_scatter_with_a_real_marker(qapp):
    dataset = _make_dataset()
    panel, _manager, _figure = _make_panel(dataset)
    _fill_xyz(panel, "temperature", "composition", "conductivity")
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()

    series = emitted[0][0]
    assert series.plot_type == Plot3DType.SCATTER
    assert series.marker == "o"
    panel.close()


def test_add_clicked_line_plot_type_has_no_marker(qapp):
    dataset = _make_dataset()
    panel, _manager, _figure = _make_panel(dataset)
    _fill_xyz(panel, "temperature", "composition", "conductivity")
    panel.plot_type_combo.setCurrentIndex(panel.plot_type_combo.findData(Plot3DType.LINE))
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()

    series = emitted[0][0]
    assert series.plot_type == Plot3DType.LINE
    assert series.marker == ""
    panel.close()


def test_add_clicked_line_marker_plot_type_has_a_real_marker(qapp):
    dataset = _make_dataset()
    panel, _manager, _figure = _make_panel(dataset)
    _fill_xyz(panel, "temperature", "composition", "conductivity")
    panel.plot_type_combo.setCurrentIndex(panel.plot_type_combo.findData(Plot3DType.LINE_MARKER))
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()

    series = emitted[0][0]
    assert series.plot_type == Plot3DType.LINE_MARKER
    assert series.marker == "o"
    panel.close()


def test_add_clicked_plot_type_is_a_genuine_enum_member_not_just_equal_to_one(qapp):
    """Regression test: `QComboBox.currentData()` round-trips a
    str-subclassed Enum through QVariant and can hand back a plain `str`
    that merely `==`-compares equal to the right `Plot3DType` member (so a
    test using only `==` would pass even if this were broken) -- the real
    failure only shows up in `Series3D.to_dict()`, which calls `.value` and
    crashes on a plain string. Caught originally via manual GUI validation,
    not by the (insufficiently strict) tests above."""
    dataset = _make_dataset()
    panel, _manager, _figure = _make_panel(dataset)
    _fill_xyz(panel, "temperature", "composition", "conductivity")
    panel.plot_type_combo.setCurrentIndex(panel.plot_type_combo.findData(Plot3DType.LINE_MARKER))
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()

    series = emitted[0][0]
    assert isinstance(series.plot_type, Plot3DType)
    series.to_dict()  # must not raise AttributeError
    panel.close()


# --- Group by: emits multiple Series3D, correct membership/order ----------------------


def test_group_by_creates_one_series3d_per_distinct_group_value(qapp):
    dataset = _make_diode_dataset()
    panel, _manager, _figure = _make_panel(dataset)
    _fill_xyz(panel, "Voltage_V", "Temperature_C", "Current_mA")
    panel.group_by_combo.setCurrentIndex(panel.group_by_combo.findData("Temperature_C"))
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()

    series_list = emitted[0]
    assert len(series_list) == 2
    assert {s.label for s in series_list} == {"25", "35"}
    panel.close()


def test_group_by_each_series_references_only_its_own_rows_in_source_order(qapp):
    dataset = _make_diode_dataset()
    panel, _manager, _figure = _make_panel(dataset)
    _fill_xyz(panel, "Voltage_V", "Temperature_C", "Current_mA")
    panel.group_by_combo.setCurrentIndex(panel.group_by_combo.findData("Temperature_C"))
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()

    by_label = {s.label: s for s in emitted[0]}
    # Rows 0,2,4 are 25C; rows 1,3,5 are 35C -- interleaved in the source.
    assert by_label["25"].row_indices == (0, 2, 4)
    assert by_label["35"].row_indices == (1, 3, 5)
    panel.close()


def test_group_by_shares_the_same_dataset_identity_across_the_family(qapp):
    dataset = _make_diode_dataset()
    panel, _manager, _figure = _make_panel(dataset)
    _fill_xyz(panel, "Voltage_V", "Temperature_C", "Current_mA")
    panel.group_by_combo.setCurrentIndex(panel.group_by_combo.findData("Temperature_C"))
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()

    series_list = emitted[0]
    assert all(s.dataset is dataset for s in series_list)
    panel.close()


def test_group_by_none_ignores_group_by_combo_selection(qapp):
    dataset = _make_diode_dataset()
    panel, _manager, _figure = _make_panel(dataset)
    _fill_xyz(panel, "Voltage_V", "Temperature_C", "Current_mA")
    # group_by_combo left at its default "None" -- explicit for clarity.
    assert panel.group_by_combo.currentText() == "None"
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()

    series_list = emitted[0]
    assert len(series_list) == 1
    assert series_list[0].row_indices is None
    panel.close()


def test_group_by_a_string_categorical_column_works(qapp):
    dataset = Dataset(
        name="cat",
        dataframe=pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 2.0, 3.0, 4.0],
                "z": [1.0, 2.0, 3.0, 4.0],
                "material": ["Si", "Ge", "Si", "Ge"],
            }
        ),
    )
    panel, _manager, _figure = _make_panel(dataset)
    _fill_xyz(panel, "x", "y", "z")
    panel.group_by_combo.setCurrentIndex(panel.group_by_combo.findData("material"))
    emitted = []
    panel.add_3d_series_requested.connect(emitted.append)

    panel.add_button.click()

    series_list = emitted[0]
    assert {s.label for s in series_list} == {"Si", "Ge"}
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


# --- "Clear 3D Plot" state, reflecting the active panel -----------------------------
#
# This page's own read-only 3D series list was removed (see the module
# docstring) -- what `refresh()` still owns is `active_panel_label` and
# `clear_button`'s enabled state, both covered below.


def test_refresh_disables_clear_for_a_2d_active_panel(qapp):
    figure = GnoviFigure()
    panel, _manager, _figure = _make_panel(_make_dataset(), figure=figure)

    assert panel.clear_button.isEnabled() is False
    panel.close()


def test_refresh_enables_clear_for_a_populated_panel3d(qapp):
    dataset = _make_dataset()
    panel3d = Panel3D()
    panel3d.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", label="mat"))
    figure = GnoviFigure(panels=[panel3d])
    panel, _manager, _figure = _make_panel(dataset, figure=figure)

    assert panel.clear_button.isEnabled() is True
    panel.close()


def test_refresh_disables_clear_for_an_empty_panel3d(qapp):
    """An empty `Panel3D` (no series yet) -- still nothing to clear."""
    figure = GnoviFigure(panels=[Panel3D()])
    panel, _manager, _figure = _make_panel(_make_dataset(), figure=figure)

    assert panel.clear_button.isEnabled() is False
    panel.close()


def test_set_figure_repoints_and_refreshes_clear_button_state(qapp):
    dataset = _make_dataset()
    panel, _manager, _figure = _make_panel(dataset)
    assert panel.clear_button.isEnabled() is False

    panel3d = Panel3D()
    panel3d.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    new_figure = GnoviFigure(panels=[panel3d])
    panel.set_figure(new_figure)

    assert panel.clear_button.isEnabled() is True
    panel.close()
