from __future__ import annotations

import numpy as np

from gnovi_plot.analysis.results import ResidualData
from gnovi_plot.gui.widgets.residual_plot import ResidualPlotWidget


def _residual_data():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    observed = np.array([1.0, 2.5, 2.9, 4.4])
    fitted = np.array([1.0, 2.0, 3.0, 4.0])
    return ResidualData(x=x, observed=observed, fitted=fitted, residuals=observed - fitted)


def test_widget_constructs_and_draws_without_error(qapp):
    widget = ResidualPlotWidget()
    widget.plot_residuals(_residual_data())
    # No exception -- the offscreen canvas rendered something.


def test_plot_uses_discrete_points_not_a_connected_line(qapp):
    widget = ResidualPlotWidget()
    data = _residual_data()
    widget.plot_residuals(data)

    collections = widget._axes.collections  # scatter() adds a PathCollection
    lines = widget._axes.get_lines()  # excludes the zero-reference axhline below

    assert len(collections) == 1
    assert collections[0].get_offsets().shape[0] == len(data.x)
    # The only Line2D artist is the zero-reference line, never a
    # residuals-connecting line.
    assert len(lines) == 1


def test_zero_reference_line_is_present_at_y_equals_zero(qapp):
    widget = ResidualPlotWidget()
    widget.plot_residuals(_residual_data())

    lines = widget._axes.get_lines()
    assert len(lines) == 1
    y_data = lines[0].get_ydata()
    assert all(y == 0 for y in y_data)


def test_axis_labels_use_the_supplied_column_names(qapp):
    widget = ResidualPlotWidget()
    widget.plot_residuals(_residual_data(), x_label="Potential (V)", y_label="Residual (Current)")

    assert widget._axes.get_xlabel() == "Potential (V)"
    assert widget._axes.get_ylabel() == "Residual (Current)"


def test_plot_residuals_can_be_called_repeatedly(qapp):
    widget = ResidualPlotWidget()
    widget.plot_residuals(_residual_data())
    widget.plot_residuals(_residual_data())  # must not accumulate/crash

    assert len(widget._axes.collections) == 1


def test_clear_removes_drawn_artists(qapp):
    widget = ResidualPlotWidget()
    widget.plot_residuals(_residual_data())
    widget.clear()

    assert len(widget._axes.collections) == 0
    assert len(widget._axes.get_lines()) == 0
