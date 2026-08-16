"""Preview-only legend fitting, driven through the real `MainWindow`/
`PlotCanvas` (real Qt resize events) -- see `tests/test_legend_fit.py` for
the pure backend-level coverage. Confirms the requirement that resizing the
window, collapsing/expanding a drawer, or resizing the bottom panel all
re-evaluate legend fit, and that none of this ever marks the project dirty.
"""

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.widgets.figure_size_panel import LAYOUT_PRESETS
from gnovi_plot.plotting.series import PlotSeries

_LONG_LABELS = [
    "Extremely long legend label describing series number one in detail",
    "Extremely long legend label describing series number two in detail",
    "Extremely long legend label describing series number three in detail",
]


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    return Dataset(name=name, dataframe=df)


def _process():
    QApplication.instance().processEvents()


def _crowd_every_panel(window) -> None:
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    for panel in window.figure_model.panels:
        for label in _LONG_LABELS:
            panel.add_series(PlotSeries.line(dataset, "x", "y", label=label))
        panel.legend_visible = True
    window._rerender()


def _fontsize(ax) -> float | None:
    legend = ax.get_legend()
    return legend.get_texts()[0].get_fontsize() if legend is not None else None


def _set_layout(window, text: str) -> None:
    index = next(i for i, (t, _dims) in enumerate(LAYOUT_PRESETS) if t == text)
    window.figure_size_panel.layout_combo.setCurrentIndex(index)


# --- Headless startup smoke test -------------------------------------------------


def test_headless_startup_smoke(qapp):
    window = MainWindow()
    window.show()
    _process()
    assert window.isVisible()
    window.close()


# --- 2x2 / 2x3 legend-preview smoke tests -----------------------------------------


def test_2x2_legend_preview_smoke(qapp):
    window = MainWindow()
    window.show()
    window.resize(700, 500)  # small overall window -> small panels in 2x2
    _process()
    _set_layout(window, "2 x 2")
    _crowd_every_panel(window)
    _process()

    configured = window.figure_model.legend_font_size
    for ax in window.plot_canvas.axes_list:
        size = _fontsize(ax)
        assert size is not None
        assert 6.0 <= size <= configured
    window.close()


def test_2x3_legend_preview_smoke(qapp):
    window = MainWindow()
    window.show()
    window.resize(800, 500)
    _process()
    _set_layout(window, "2 x 3")
    _crowd_every_panel(window)
    _process()

    configured = window.figure_model.legend_font_size
    for ax in window.plot_canvas.axes_list:
        size = _fontsize(ax)
        assert size is not None
        assert 6.0 <= size <= configured
    window.close()


# --- Model preserved / project not dirtied ----------------------------------------


def test_legend_preview_fitting_never_changes_the_configured_model_value(qapp):
    window = MainWindow()
    window.show()
    window.resize(600, 400)
    _process()
    _set_layout(window, "2 x 2")
    _crowd_every_panel(window)
    _process()

    assert window.figure_model.legend_font_size == 9.0
    assert all(panel.legend_fontsize is None for panel in window.figure_model.panels)


def test_legend_preview_fitting_does_not_mark_the_project_dirty(qapp):
    window = MainWindow()
    window.show()
    window.resize(1200, 900)
    _process()
    _set_layout(window, "2 x 2")
    _crowd_every_panel(window)
    _process()
    window._set_dirty(False)

    window.resize(400, 300)  # shrink hard -> forces legend shrinking via resizeEvent
    _process()

    assert window._dirty is False
    window.close()


# --- Re-evaluated on every kind of workspace reshaping -----------------------------


def test_window_resize_re_evaluates_legend_fit(qapp):
    window = MainWindow()
    window.show()
    window.resize(2600, 1800)  # plenty of room -> full configured size
    _process()
    _set_layout(window, "2 x 2")
    _crowd_every_panel(window)
    _process()
    large_size = _fontsize(window.plot_canvas.axes_list[0])
    assert large_size == pytest.approx(window.figure_model.legend_font_size)

    window.resize(500, 400)
    _process()
    small_size = _fontsize(window.plot_canvas.axes_list[0])

    assert small_size < large_size

    window.resize(2600, 1800)
    _process()
    restored_size = _fontsize(window.plot_canvas.axes_list[0])
    assert restored_size == pytest.approx(large_size)
    window.close()


def test_collapsing_the_left_drawer_re_evaluates_legend_fit(qapp):
    window = MainWindow()
    window.show()
    window.resize(700, 500)
    _process()
    _set_layout(window, "2 x 2")
    _crowd_every_panel(window)
    _process()
    before = _fontsize(window.plot_canvas.axes_list[0])

    window.tool_drawer._buttons["data"].click()  # collapse -> canvas gets wider
    _process()

    after = _fontsize(window.plot_canvas.axes_list[0])
    assert after >= before  # more room -> never a smaller effective size
    window.close()


def test_resizing_the_bottom_panel_re_evaluates_legend_fit(qapp):
    window = MainWindow()
    window.show()
    window.resize(700, 900)
    _process()
    _set_layout(window, "2 x 2")
    _crowd_every_panel(window)
    _process()

    total = sum(window.center_splitter.sizes())
    window.center_splitter.setSizes([int(total * 0.3), int(total * 0.7)])  # shrink the canvas
    _process()

    for ax in window.plot_canvas.axes_list:
        size = _fontsize(ax)
        assert size is not None and size >= 6.0
    window.close()


# --- Project open/new dataset synchronization smoke test ---------------------------


def test_project_open_new_dataset_synchronization_smoke(qapp, monkeypatch, tmp_path):
    window = MainWindow()
    d1 = _make_dataset("Alpha")
    d2 = _make_dataset("Beta")
    window.dataset_manager.add(d1)
    window.dataset_manager.add(d2)
    window.dataset_panel._refresh_list(select_id=d1.id)
    assert window.dataset_panel.active_dataset_combo.currentData() == d1.id

    index = window.dataset_panel.active_dataset_combo.findData(d2.id)
    window.dataset_panel.active_dataset_combo.setCurrentIndex(index)
    assert window.dataset_panel.current_dataset is d2

    out_path = tmp_path / "sync_smoke.gnovi"
    monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.save_project_as_action.trigger()

    window.new_project_action.trigger()
    assert window.dataset_panel.active_dataset_combo.currentText() == "(no dataset)"

    monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getOpenFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.open_project_action.trigger()

    assert len(window.dataset_manager) == 2
    assert window.dataset_panel.active_dataset_combo.count() == 3  # placeholder + 2 datasets
    window.close()
