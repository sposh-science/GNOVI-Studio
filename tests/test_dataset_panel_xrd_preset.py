import pandas as pd
from PySide6.QtWidgets import QTableView

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.dataset_panel import DatasetPanel
from gnovi_plot.plotting.series import PlotType


def _make_dataset(name="xrd"):
    df = pd.DataFrame({"2theta": [10.0, 20.0, 30.0], "counts": [100.0, 500.0, 150.0]})
    return Dataset(name=name, dataframe=df)


def _make_panel():
    manager = DatasetManager()
    dataset = _make_dataset()
    manager.add(dataset)
    preview_table = QTableView()
    panel = DatasetPanel(manager, preview_table)
    panel._refresh_list(select_id=dataset.id)
    panel.x_combo.setCurrentText("2theta")
    panel.y_combo.setCurrentText("counts")
    return panel, dataset


def test_xrd_preset_forces_line_plot_type_and_disables_the_combo(qapp):
    panel, _dataset = _make_panel()

    xrd_index = panel.plot_preset_combo.findData("xrd")
    panel.plot_preset_combo.setCurrentIndex(xrd_index)

    assert panel.plot_type_combo.currentData() == PlotType.LINE
    assert panel.plot_type_combo.isEnabled() is False
    panel.close()


def test_xrd_preset_emits_axis_defaults_on_add_to_plot(qapp):
    panel, _dataset = _make_panel()
    received = []
    panel.axis_preset_requested.connect(received.append)
    panel.add_to_plot_requested.connect(lambda _series: None)

    xrd_index = panel.plot_preset_combo.findData("xrd")
    panel.plot_preset_combo.setCurrentIndex(xrd_index)
    panel._on_add_to_plot_clicked()

    assert len(received) == 1
    assert received[0] == {"xlabel": "2θ (°)", "ylabel": "Intensity (a.u.)"}
    panel.close()


def test_none_preset_does_not_emit_axis_defaults(qapp):
    panel, _dataset = _make_panel()
    received = []
    panel.axis_preset_requested.connect(received.append)
    panel.add_to_plot_requested.connect(lambda _series: None)

    panel._on_add_to_plot_clicked()

    assert received == []
    panel.close()


def test_xrd_preset_series_has_no_marker(qapp):
    panel, _dataset = _make_panel()
    captured = []
    panel.add_to_plot_requested.connect(captured.append)

    xrd_index = panel.plot_preset_combo.findData("xrd")
    panel.plot_preset_combo.setCurrentIndex(xrd_index)
    panel._on_add_to_plot_clicked()

    assert len(captured) == 1
    series_list = captured[0]
    assert len(series_list) == 1
    assert series_list[0].plot_type == PlotType.LINE
    assert series_list[0].marker == ""
    panel.close()
