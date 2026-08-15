import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QItemSelection, QItemSelectionModel
from PySide6.QtWidgets import QMessageBox

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.widgets.figure_size_panel import LAYOUT_PRESETS

_ROW_COUNT = 4800
_LAYOUT_2X2_INDEX = next(i for i, (text, _dims) in enumerate(LAYOUT_PRESETS) if text == "2 x 2")


def _make_large_dataset(name="big"):
    x = np.arange(_ROW_COUNT, dtype=float)
    df = pd.DataFrame({"x": x, "y": x * 2.0})
    return Dataset(name=name, dataframe=df)


def _select_dataset(window: MainWindow, dataset: Dataset) -> None:
    window.dataset_manager.add(dataset)
    window.dataset_panel._refresh_list(select_id=dataset.id)


def _select_preview_rows(window: MainWindow, start: int, end_inclusive: int) -> None:
    model = window.preview_model
    selection_model = window.preview_table.selectionModel()
    top_left = model.index(start, 0)
    bottom_right = model.index(end_inclusive, model.columnCount() - 1)
    selection_model.select(QItemSelection(top_left, bottom_right), QItemSelectionModel.ClearAndSelect)


@pytest.fixture
def no_popups(monkeypatch):
    """Fail loudly (instead of hanging on a modal exec()) if any QMessageBox
    is shown during the test."""
    calls = []

    def _record(kind):
        def _inner(*_args, **_kwargs):
            calls.append(kind)
            return QMessageBox.Ok

        return _inner

    monkeypatch.setattr(QMessageBox, "information", staticmethod(_record("information")))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_record("warning")))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_record("critical")))
    return calls


def test_plot_selected_rows_multi_panel_workflow(qapp, no_popups):
    window = MainWindow()
    dataset = _make_large_dataset()
    _select_dataset(window, dataset)

    # Calculated columns for panels 2 and 3.
    dataset.add_calculated_column("y2", "x * 2")
    dataset.add_calculated_column("y3", "x + 1")
    window.dataset_panel.refresh_columns()

    # Give the figure 4 panels up front.
    window.figure_size_panel.layout_combo.setCurrentIndex(_LAYOUT_2X2_INDEX)
    assert len(window.figure_model.panels) == 4

    # Panel 1: full dataset.
    window.figure_size_panel.panel_combo.setCurrentIndex(0)
    window.dataset_panel.x_combo.setCurrentText("x")
    window.dataset_panel.y_combo.setCurrentText("y")
    window.dataset_panel.add_to_plot_button.click()

    # Panel 2: calculated-column plot.
    window.figure_size_panel.panel_combo.setCurrentIndex(1)
    window.dataset_panel.y_combo.setCurrentText("y2")
    window.dataset_panel.add_to_plot_button.click()

    # Panel 3: another combination.
    window.figure_size_panel.panel_combo.setCurrentIndex(2)
    window.dataset_panel.y_combo.setCurrentText("y3")
    window.dataset_panel.add_to_plot_button.click()

    panel_1_series_before = list(window.figure_model.panels[0].series)
    panel_2_series_before = list(window.figure_model.panels[1].series)
    panel_3_series_before = list(window.figure_model.panels[2].series)
    assert len(panel_1_series_before) == 1
    assert len(panel_2_series_before) == 1
    assert len(panel_3_series_before) == 1

    raw_row_count_before = dataset.raw_row_count
    working_row_count_before = dataset.row_count
    transformation_count_before = len(dataset.transformations)
    raw_df_before = dataset.raw_dataframe.copy()
    working_df_before = dataset.dataframe.copy()

    # Select rows 45-54 in the Data Preview, then make Panel 4 active.
    _select_preview_rows(window, 45, 54)
    window.figure_size_panel.panel_combo.setCurrentIndex(3)

    # "Plot Selected Rows".
    window.dataset_panel.x_combo.setCurrentText("x")
    window.dataset_panel.y_combo.setCurrentText("y")
    window.data_tools_panel._on_plot_selected_rows_clicked()

    # Panels 1-3 unchanged.
    assert window.figure_model.panels[0].series == panel_1_series_before
    assert window.figure_model.panels[1].series == panel_2_series_before
    assert window.figure_model.panels[2].series == panel_3_series_before
    assert all(not s.stale for s in panel_1_series_before + panel_2_series_before + panel_3_series_before)

    # Panel 4 has exactly the new selection-scoped series.
    panel_4_series = window.figure_model.panels[3].series
    assert len(panel_4_series) == 1
    new_series = panel_4_series[0]
    assert new_series.row_range == (45, 55)
    assert new_series.x_column == "x"
    assert new_series.y_column == "y"
    assert len(new_series.dataframe) == 10
    assert new_series.label == f"{dataset.name} — rows 45–54"

    # Working Data / raw data / history completely untouched.
    assert dataset.row_count == working_row_count_before == _ROW_COUNT
    assert dataset.raw_row_count == raw_row_count_before == _ROW_COUNT
    assert len(dataset.transformations) == transformation_count_before
    pd.testing.assert_frame_equal(dataset.raw_dataframe, raw_df_before)
    pd.testing.assert_frame_equal(dataset.dataframe, working_df_before)

    # No invalidation / warning / error popups were shown.
    assert no_popups == []

    window.close()


def test_plot_selected_rows_requires_at_least_two_rows(qapp, no_popups):
    window = MainWindow()
    dataset = _make_large_dataset()
    _select_dataset(window, dataset)

    _select_preview_rows(window, 10, 10)  # single row
    window.data_tools_panel._on_plot_selected_rows_clicked()

    assert window.figure_model.active_panel.series == []
    assert no_popups == ["warning"]

    window.close()


def test_plot_selected_rows_requires_contiguous_selection(qapp, no_popups):
    window = MainWindow()
    dataset = _make_large_dataset()
    _select_dataset(window, dataset)

    model = window.preview_model
    selection_model = window.preview_table.selectionModel()
    selection = QItemSelection(model.index(5, 0), model.index(5, model.columnCount() - 1))
    selection.merge(
        QItemSelection(model.index(9, 0), model.index(9, model.columnCount() - 1)),
        QItemSelectionModel.Select,
    )
    selection_model.select(selection, QItemSelectionModel.ClearAndSelect)

    window.dataset_panel.x_combo.setCurrentText("x")
    window.dataset_panel.y_combo.setCurrentText("y")
    window.data_tools_panel._on_plot_selected_rows_clicked()

    assert window.figure_model.active_panel.series == []
    assert no_popups == ["warning"]

    window.close()
