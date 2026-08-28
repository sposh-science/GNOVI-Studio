"""3D scatter rendering (`plotting.backends.matplotlib_backend.
render_panel_3d`/`build_projection_aware_axes`) -- Qt-free, backend-level,
mirroring `test_matplotlib_backend.py`'s/`test_legend_fit.py`'s style
(`FigureCanvasAgg` directly).
"""

import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.backends.matplotlib_backend import (
    build_projection_aware_axes,
    render_figure,
    render_panel_3d,
)
from gnovi_plot.plotting.figure import GnoviFigure, Panel, Panel3D, PlotTheme
from gnovi_plot.plotting.series import PlotSeries
from gnovi_plot.plotting.series3d import Series3D


def _make_dataset(name="mat"):
    df = pd.DataFrame(
        {"temperature": [300.0, 350.0, 400.0, 300.0], "composition": [0.1, 0.1, 0.1, 0.2], "conductivity": [2.4, 2.9, 3.5, 3.1]}
    )
    return Dataset(name=name, dataframe=df)


def _panel3d_with_series(dataset, **panel_kwargs):
    panel = Panel3D(**panel_kwargs)
    panel.add_series(
        Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity")
    )
    return panel


def _canvas_and_axes(figure: GnoviFigure, figsize=(6.4, 4.8)):
    mpl_figure = Figure(figsize=figsize)
    FigureCanvasAgg(mpl_figure)
    rows, cols = figure.layout
    axes_list = build_projection_aware_axes(mpl_figure, rows, cols, figure.panels)
    return mpl_figure, axes_list


# --- Projection-aware Axes creation -----------------------------------------------


def test_pure_2d_layout_still_creates_ordinary_axes():
    figure = GnoviFigure()
    figure.set_layout(1, 3)
    _mpl_figure, axes_list = _canvas_and_axes(figure)
    assert all(not isinstance(ax, Axes3D) for ax in axes_list)


def test_panel3d_creates_axes3d():
    dataset = _make_dataset()
    figure = GnoviFigure(panels=[_panel3d_with_series(dataset)])
    _mpl_figure, axes_list = _canvas_and_axes(figure)
    assert isinstance(axes_list[0], Axes3D)


def test_mixed_2d_3d_layout_creates_projection_per_cell():
    dataset = _make_dataset()
    figure = GnoviFigure(
        panels=[Panel(title="2D-1"), _panel3d_with_series(dataset, title="3D"), Panel(title="2D-2")],
        layout=(1, 3),
    )
    _mpl_figure, axes_list = _canvas_and_axes(figure)
    assert [isinstance(ax, Axes3D) for ax in axes_list] == [False, True, False]


def test_converting_one_panel_to_3d_at_the_same_position_rebuilds_only_that_axes():
    """`build_projection_aware_axes` is re-called (a fresh Figure each
    time here for isolation) -- this test locks in that the function is
    driven entirely by `panels`, never a stale cached shape, by building
    twice from figures that differ only in one cell's type."""
    dataset = _make_dataset()
    figure_all_2d = GnoviFigure(panels=[Panel(), Panel(), Panel()], layout=(1, 3))
    figure_mixed = GnoviFigure(panels=[Panel(), _panel3d_with_series(dataset), Panel()], layout=(1, 3))

    _f1, axes_2d = _canvas_and_axes(figure_all_2d)
    _f2, axes_mixed = _canvas_and_axes(figure_mixed)
    assert [isinstance(ax, Axes3D) for ax in axes_2d] == [False, False, False]
    assert [isinstance(ax, Axes3D) for ax in axes_mixed] == [False, True, False]


# --- 3D scatter content: point count, title/labels, marker, visibility -----------


def test_3d_scatter_renders_the_correct_point_count():
    dataset = _make_dataset()
    panel = _panel3d_with_series(dataset)
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")

    render_panel_3d(ax, panel)

    collections = ax.collections
    assert len(collections) == 1
    assert collections[0].get_offsets().shape[0] == 4 or len(collections[0]._offsets3d[0]) == 4


def test_title_and_xyz_labels_render():
    dataset = _make_dataset()
    panel = _panel3d_with_series(dataset, title="Conductivity", x_label="T (K)", y_label="Composition", z_label="σ (S/cm)")
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")

    render_panel_3d(ax, panel)

    assert ax.get_title() == "Conductivity"
    assert ax.get_xlabel() == "T (K)"
    assert ax.get_ylabel() == "Composition"
    assert ax.get_zlabel() == "σ (S/cm)"


def test_marker_styling_renders():
    dataset = _make_dataset()
    panel = Panel3D()
    panel.add_series(
        Series3D(
            dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
            marker="s", marker_size=10.0, color="#ff00ff", alpha=0.5,
        )
    )
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")

    render_panel_3d(ax, panel)

    collection = ax.collections[0]
    assert collection.get_alpha() == 0.5


def test_series_visibility_is_respected():
    dataset = _make_dataset()
    panel = _panel3d_with_series(dataset)
    panel.series[0].visible = False
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")

    render_panel_3d(ax, panel)

    assert len(ax.collections) == 0


def test_stale_series_is_not_drawn():
    dataset = _make_dataset()
    panel = _panel3d_with_series(dataset)
    panel.series[0].stale = True
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")

    render_panel_3d(ax, panel)

    assert len(ax.collections) == 0


def test_grid_toggle_does_not_raise():
    dataset = _make_dataset()
    panel = _panel3d_with_series(dataset)
    panel.grid = False
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)  # must not raise


def test_camera_elevation_and_azimuth_are_applied():
    dataset = _make_dataset()
    panel = _panel3d_with_series(dataset, elevation=15.0, azimuth=100.0)
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")

    render_panel_3d(ax, panel)

    assert ax.elev == pytest.approx(15.0)
    assert ax.azim == pytest.approx(100.0)


def test_xyz_limits_are_applied_when_set():
    dataset = _make_dataset()
    panel = _panel3d_with_series(dataset, xlim=(0.0, 500.0), ylim=(0.0, 1.0), zlim=(0.0, 5.0))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")

    render_panel_3d(ax, panel)

    assert ax.get_xlim() == (0.0, 500.0)
    assert ax.get_ylim() == (0.0, 1.0)
    assert ax.get_zlim() == (0.0, 5.0)


# --- Theme / typography -----------------------------------------------------------


def test_theme_background_is_respected_light_vs_dark():
    dataset = _make_dataset()
    panel = _panel3d_with_series(dataset)
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax_light = mpl_figure.add_subplot(1, 2, 1, projection="3d")
    ax_dark = mpl_figure.add_subplot(1, 2, 2, projection="3d")

    render_panel_3d(ax_light, panel, dark_mode=False)
    render_panel_3d(ax_dark, panel, dark_mode=True)

    assert ax_light.get_facecolor() != ax_dark.get_facecolor()


def test_typography_font_sizes_from_figure_are_applied():
    dataset = _make_dataset()
    panel = _panel3d_with_series(dataset, title="T", x_label="X", y_label="Y", z_label="Z")
    figure = GnoviFigure(panels=[panel], title_font_size=22.0, axis_label_font_size=16.0)
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")

    render_panel_3d(ax, panel, figure, dark_mode=False)

    assert ax.title.get_fontsize() == pytest.approx(22.0)
    assert ax.xaxis.label.get_fontsize() == pytest.approx(16.0)


def test_render_figure_dispatches_correctly_for_a_mixed_layout():
    """`render_figure`'s existing loop (unchanged code, see
    `render_panel_with_figure_background`) must render BOTH a 2D line and
    a 3D scatter correctly in one pass when given a mixed Figure."""
    dataset = _make_dataset()
    panel_2d = Panel(title="2D")
    panel_2d.add_series(PlotSeries.line(dataset, "temperature", "conductivity"))
    panel_3d = _panel3d_with_series(dataset, title="3D")
    figure = GnoviFigure(panels=[panel_2d, panel_3d], layout=(1, 2))

    mpl_figure, axes_list = _canvas_and_axes(figure)
    render_figure(axes_list, figure)  # must not raise

    assert axes_list[0].get_title() == "2D"
    assert len(axes_list[0].lines) == 1
    assert axes_list[1].get_title() == "3D"
    assert len(axes_list[1].collections) == 1


def test_export_theme_dark_mode_produces_a_dark_3d_background():
    dataset = _make_dataset()
    panel = _panel3d_with_series(dataset)
    figure = GnoviFigure(panels=[panel], plot_theme=PlotTheme.DARK)
    mpl_figure, axes_list = _canvas_and_axes(figure)

    render_figure(axes_list, figure, dark_mode=True)

    r, g, b, _a = axes_list[0].get_facecolor()
    assert (r, g, b) != (1.0, 1.0, 1.0)
