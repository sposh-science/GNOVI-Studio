from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt

from gnovi_plot.analysis.fitting import ResidualData
from gnovi_plot.gui.widgets.residual_window import ResidualWindow


def _residual_data():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    observed = np.array([1.0, 2.5, 2.9, 4.4])
    fitted = np.array([1.0, 2.0, 3.0, 4.0])
    return ResidualData(x=x, observed=observed, fitted=fitted, residuals=observed - fitted)


def test_window_constructs_as_a_non_modal_top_level_window(qapp):
    window = ResidualWindow()

    assert window.isWindow()
    assert window.windowModality() == Qt.NonModal


def test_window_is_independently_resizable_with_a_sensible_initial_size(qapp):
    window = ResidualWindow()

    assert window.width() > 100
    assert window.height() > 100

    window.resize(800, 600)
    assert window.size().width() == 800
    assert window.size().height() == 600


def test_show_residuals_sets_title_and_forwards_data_to_the_plot(qapp):
    window = ResidualWindow()
    data = _residual_data()

    window.show_residuals(
        data,
        x_label="Potential (V)",
        y_label="Residual (Current)",
        title="Residuals — linear fit — my series",
    )

    assert window.windowTitle() == "Residuals — linear fit — my series"
    assert window._plot._axes.get_xlabel() == "Potential (V)"
    assert window._plot._axes.get_ylabel() == "Residual (Current)"
    collections = window._plot._axes.collections
    assert len(collections) == 1
    assert collections[0].get_offsets().shape[0] == len(data.x)


def test_show_residuals_makes_the_window_visible(qapp):
    window = ResidualWindow()
    window.show_residuals(_residual_data(), x_label="x", y_label="Residual", title="Residuals")

    assert window.isVisible()


def test_closing_does_not_destroy_the_window_so_it_can_be_reshown(qapp):
    """WA_DeleteOnClose must be left unset -- closing (the platform close
    button, or `.close()`) hides a plain top-level QWidget rather than
    destroying it, so the same instance can be reused."""
    window = ResidualWindow()
    window.show_residuals(_residual_data(), x_label="x", y_label="Residual", title="Residuals")
    assert window.isVisible()

    window.close()

    assert not window.isVisible()
    assert not window.testAttribute(Qt.WA_DeleteOnClose)

    window.show_residuals(_residual_data(), x_label="x", y_label="Residual", title="Residuals again")
    assert window.isVisible()
    assert window.windowTitle() == "Residuals again"


def test_repeated_show_residuals_calls_do_not_accumulate_plot_artists(qapp):
    window = ResidualWindow()
    window.show_residuals(_residual_data(), x_label="x", y_label="Residual", title="a")
    window.show_residuals(_residual_data(), x_label="x", y_label="Residual", title="b")

    assert len(window._plot._axes.collections) == 1


def test_module_has_no_gnovi_figure_panel_series_or_workbench_dependency():
    from gnovi_plot.gui.widgets import residual_window

    for name in ("GnoviFigure", "Panel", "PlotSeries", "Workbench"):
        assert not hasattr(residual_window, name)
