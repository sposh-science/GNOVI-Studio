import pandas as pd
import pytest
from PySide6.QtCore import QItemSelection, QItemSelectionModel
from PySide6.QtWidgets import QMessageBox, QTableView

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.widgets.data_tools_panel import _ROW_OPERATION_CONFIRMATION, DataToolsPanel
from gnovi_plot.gui.widgets.dataframe_table_model import DataFrameTableModel


def _make_dataset(name="d", rows=10):
    df = pd.DataFrame({"x": list(range(rows)), "y": [float(i) for i in range(rows)]})
    return Dataset(name=name, dataframe=df)


def _make_panel_with_dataset(dataset):
    table = QTableView()
    model = DataFrameTableModel()
    model.set_dataframe(dataset.dataframe)
    table.setModel(model)
    panel = DataToolsPanel(table)
    panel.set_dataset(dataset)
    return panel, table


def _select_rows(table, start, end_inclusive):
    model = table.model()
    selection_model = table.selectionModel()
    top_left = model.index(start, 0)
    bottom_right = model.index(end_inclusive, model.columnCount() - 1)
    selection_model.select(QItemSelection(top_left, bottom_right), QItemSelectionModel.ClearAndSelect)


# --- Button labels ------------------------------------------------------------


def test_keep_and_exclude_buttons_have_explicit_working_data_labels(qapp):
    dataset = _make_dataset()
    panel, _table = _make_panel_with_dataset(dataset)

    assert panel.keep_button.text() == "Keep Selected Rows in Working Data"
    assert panel.exclude_button.text() == "Exclude Selected Rows from Working Data"


# --- Confirmation dialog -------------------------------------------------------


@pytest.fixture
def confirm_yes(monkeypatch):
    calls = []

    def _question(*args, **kwargs):
        calls.append(args[2] if len(args) > 2 else kwargs.get("text"))
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_question))
    return calls


@pytest.fixture
def confirm_no(monkeypatch):
    calls = []

    def _question(*args, **kwargs):
        calls.append(args[2] if len(args) > 2 else kwargs.get("text"))
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_question))
    return calls


def test_keep_selection_asks_for_confirmation_with_the_required_text(qapp, confirm_yes):
    dataset = _make_dataset()
    panel, table = _make_panel_with_dataset(dataset)
    _select_rows(table, 0, 2)

    panel.keep_button.click()

    assert confirm_yes == [_ROW_OPERATION_CONFIRMATION]
    assert dataset.row_count == 3  # operation proceeded after "Yes"


def test_exclude_selection_asks_for_confirmation_with_the_required_text(qapp, confirm_yes):
    dataset = _make_dataset(rows=10)
    panel, table = _make_panel_with_dataset(dataset)
    _select_rows(table, 0, 2)

    panel.exclude_button.click()

    assert confirm_yes == [_ROW_OPERATION_CONFIRMATION]
    assert dataset.row_count == 7  # operation proceeded after "Yes"


def test_declining_keep_confirmation_leaves_working_data_untouched(qapp, confirm_no):
    dataset = _make_dataset(rows=10)
    panel, table = _make_panel_with_dataset(dataset)
    _select_rows(table, 0, 2)
    row_count_before = dataset.row_count
    transformations_before = len(dataset.transformations)

    panel.keep_button.click()

    assert dataset.row_count == row_count_before
    assert len(dataset.transformations) == transformations_before


def test_declining_exclude_confirmation_leaves_working_data_untouched(qapp, confirm_no):
    dataset = _make_dataset(rows=10)
    panel, table = _make_panel_with_dataset(dataset)
    _select_rows(table, 0, 2)
    row_count_before = dataset.row_count
    transformations_before = len(dataset.transformations)

    panel.exclude_button.click()

    assert dataset.row_count == row_count_before
    assert len(dataset.transformations) == transformations_before


def test_declining_confirmation_does_not_emit_transformation_applied(qapp, confirm_no):
    dataset = _make_dataset(rows=10)
    panel, table = _make_panel_with_dataset(dataset)
    _select_rows(table, 0, 2)

    emitted = []
    panel.transformation_applied.connect(lambda *args: emitted.append(args))
    panel.keep_button.click()

    assert emitted == []


# --- Plot Selected Rows stays unaffected/non-destructive -----------------------


def test_plot_selected_rows_never_asks_for_confirmation(qapp, confirm_yes):
    """Plot Selected Rows is not a Working Data operation and must not go
    through the Keep/Exclude confirmation prompt."""
    dataset = _make_dataset(rows=10)
    panel, table = _make_panel_with_dataset(dataset)
    _select_rows(table, 0, 3)

    requested = []
    panel.plot_selected_rows_requested.connect(lambda positions: requested.append(positions))
    panel.plot_selected_rows_button.click()

    assert confirm_yes == []
    assert requested == [[0, 1, 2, 3]]


def test_plot_selected_rows_does_not_mutate_working_data_or_history(qapp, confirm_yes):
    dataset = _make_dataset(rows=10)
    panel, table = _make_panel_with_dataset(dataset)
    _select_rows(table, 0, 3)
    row_count_before = dataset.row_count
    transformations_before = len(dataset.transformations)

    panel.plot_selected_rows_button.click()

    assert dataset.row_count == row_count_before
    assert len(dataset.transformations) == transformations_before
