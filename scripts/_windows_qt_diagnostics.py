"""Temporary diagnostic script for the PR #2 Windows CI investigation.

Not part of the application -- prints Qt platform/screen/DPI/font state and
reproduces the exact window-size scenarios used by the five failing Windows
tests (badge, WYSIWYG export x2, aspect-ratio lock, legend fit), so the
actual achieved geometry can be compared against what each test requests.
Meant to be removed once the Windows CI failures are diagnosed.
"""

from __future__ import annotations

import os

import PySide6
from PySide6.QtCore import __version__ as qt_version
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)
print("QT_QPA_PLATFORM env var:", os.environ.get("QT_QPA_PLATFORM"))
print("Actual platform in use (QGuiApplication.platformName()):", QGuiApplication.platformName())
print("PySide6 version:", PySide6.__version__)
print("Qt version:", qt_version)

screen = app.primaryScreen()
print()
print("=" * 70)
print("SCREEN")
print("=" * 70)
print("geometry:", screen.geometry())
print("availableGeometry:", screen.availableGeometry())
print("devicePixelRatio:", screen.devicePixelRatio())
print("logicalDotsPerInch:", screen.logicalDotsPerInch())
print("physicalDotsPerInch:", screen.physicalDotsPerInch())

font = app.font()
print()
print("=" * 70)
print("FONT")
print("=" * 70)
print("family:", font.family())
print("pointSizeF:", font.pointSizeF())

from gnovi_plot.data.dataset import Dataset  # noqa: E402
from gnovi_plot.gui.main_window import MainWindow  # noqa: E402
from gnovi_plot.gui.widgets.figure_size_panel import LAYOUT_PRESETS  # noqa: E402
from gnovi_plot.plotting.series import PlotSeries  # noqa: E402
import pandas as pd  # noqa: E402


def _process():
    QApplication.instance().processEvents()


def _dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    return Dataset(name=name, dataframe=df)


def _report(label, window, requested_w, requested_h):
    _process()
    canvas = window.plot_canvas
    fig = canvas.figure
    fig_w_in, fig_h_in = fig.get_size_inches()
    dpi = fig.get_dpi()
    print(f"--- {label} ---")
    print(f"  requested resize: {requested_w} x {requested_h}")
    print(f"  MainWindow.size(): {window.size().width()} x {window.size().height()}")
    print(f"  MainWindow.frameGeometry(): {window.frameGeometry()}")
    print(f"  PlotCanvas.size(): {canvas.size().width()} x {canvas.size().height()}")
    print(f"  PlotCanvas.figure.get_size_inches(): {fig_w_in:.4f} x {fig_h_in:.4f}")
    print(f"  PlotCanvas.figure.dpi: {dpi}")
    print(f"  -> implied savefig pixel size at this dpi: {fig_w_in * dpi:.1f} x {fig_h_in * dpi:.1f}")


print()
print("=" * 70)
print("WINDOW/CANVAS GEOMETRY SCENARIOS")
print("=" * 70)

# Scenario A: same as tests/test_export_wysiwyg_parity.py::_build_window
w = MainWindow()
w.show()
w.resize(1300, 850)
_process()
index = next(i for i, (t, _d) in enumerate(LAYOUT_PRESETS) if t == "2 x 2")
w.figure_size_panel.layout_combo.setCurrentIndex(index)
ds = _dataset()
w.dataset_manager.add(ds)
for panel in w.figure_model.panels:
    panel.add_series(PlotSeries.line(ds, "x", "y", label="S1 (Ferricyanide) SR-0.1 -- Current/A"))
    panel.legend_visible = True
    panel.scientific_notation_y = True
w._rerender()
_report("export_wysiwyg_parity scenario (2x2, resize 1300x850)", w, 1300, 850)

# Scenario B: same as tests/test_active_panel_badge.py
w2 = MainWindow()
w2.show()
w2.resize(1400, 900)
_report("active_panel_badge scenario (resize 1400x900, before layout change)", w2, 1400, 900)

# Scenario C: same as tests/test_figure_aspect_ratio_gui.py
w3 = MainWindow()
w3.show()
w3.resize(1600, 900)
_report("figure_aspect_ratio scenario (resize 1600x900)", w3, 1600, 900)

# Scenario D: same as tests/test_legend_fit_gui.py
w4 = MainWindow()
w4.show()
w4.resize(2600, 1800)
_report("legend_fit scenario (resize 2600x1800)", w4, 2600, 1800)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
