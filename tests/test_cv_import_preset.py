"""The "CV" plot preset and its lightweight potential/current column
matcher in gui.widgets.dataset_panel."""

from __future__ import annotations

import pandas as pd
import pytest
from PySide6.QtWidgets import QTableView

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.dataset_panel import DatasetPanel, _match_cv_columns


def _panel(manager: DatasetManager, dataset_id: str) -> DatasetPanel:
    panel = DatasetPanel(manager, QTableView())
    panel._refresh_list(select_id=dataset_id)
    return panel


@pytest.mark.parametrize(
    "columns, expected",
    [
        (["Potential/V", "Current/A"], (0, 1)),
        (["time/s", "Ewe/V", "<I>/mA"], (1, 2)),
        (["Time", "WE(1).Potential (V)", "WE(1).Current (A)"], (1, 2)),
        (["Voltage", "Current"], (0, 1)),
        (["a", "b", "c"], None),  # no match -> caller keeps its default
        (["Potential/V", "Potential/V"], None),  # same column both sides
    ],
)
def test_match_cv_columns(columns, expected):
    assert _match_cv_columns(columns) == expected


def test_cv_preset_preselects_columns_and_forces_line(qapp):
    df = pd.DataFrame({"index": [0, 1, 2], "Ewe/V": [-0.2, 0.0, 0.2], "I/mA": [1.0, 2.0, 1.0]})
    manager = DatasetManager()
    ds = Dataset(name="cv", dataframe=df)
    manager.add(ds)
    panel = _panel(manager, ds.id)

    combo = panel.plot_preset_combo
    panel.plot_preset_combo.setCurrentIndex(combo.findData("cv"))

    assert panel.x_combo.currentText() == "Ewe/V"
    assert panel.y_combo.currentText() == "I/mA"
    assert panel._current_plot_type().value == "line"
    # the "plot by cycles" mode stays available for CV, unlike XRD
    assert panel.plot_mode_combo.isEnabled()


def test_cv_preset_emits_axis_labels_on_add_to_plot(qapp):
    df = pd.DataFrame({"Potential/V": [-0.2, 0.0, 0.2], "Current/A": [1e-6, 2e-6, 1e-6]})
    manager = DatasetManager()
    ds = Dataset(name="cv", dataframe=df)
    manager.add(ds)
    panel = _panel(manager, ds.id)
    panel.plot_preset_combo.setCurrentIndex(panel.plot_preset_combo.findData("cv"))

    presets = []
    panel.axis_preset_requested.connect(presets.append)
    panel._on_add_to_plot_clicked()

    assert presets and presets[-1] == {"xlabel": "Potential (V)", "ylabel": "Current (A)"}
