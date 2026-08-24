"""3D scatter rendering (`plotting.backends.matplotlib_backend.
render_panel_3d`/`build_projection_aware_axes`) -- Qt-free, backend-level,
mirroring `test_matplotlib_backend.py`'s/`test_legend_fit.py`'s style
(`FigureCanvasAgg` directly).
"""

import io

import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.backends.matplotlib_backend import (
    _apply_3d_grid_style,
    build_projection_aware_axes,
    render_figure,
    render_panel_3d,
)
from gnovi_plot.plotting.figure import GnoviFigure, Panel, Panel3D, PlotTheme
from gnovi_plot.plotting.series import PlotSeries
from gnovi_plot.plotting.series3d import Plot3DType, Series3D


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


# --- Plot types: Scatter / Line / Line + Markers -----------------------------------


def _diode_dataset():
    df = pd.DataFrame(
        {
            "Voltage_V": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
            "Temperature_C": [25.0, 35.0, 25.0, 35.0, 25.0, 35.0],
            "Current_mA": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        }
    )
    return Dataset(name="diode", dataframe=df)


def _render_single_series(series: Series3D, panel_kwargs=None) -> "Axes3D":
    panel = Panel3D(**(panel_kwargs or {}))
    panel.add_series(series)
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)
    return ax


def test_scatter_plot_type_renders_points_only_no_line():
    dataset = _make_dataset()
    series = Series3D(
        dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
        plot_type=Plot3DType.SCATTER,
    )
    ax = _render_single_series(series)
    assert len(ax.collections) == 1  # Path3DCollection from scatter
    assert len(ax.lines) == 0


def test_line_plot_type_renders_a_connected_path_without_markers():
    dataset = _make_dataset()
    series = Series3D(
        dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
        plot_type=Plot3DType.LINE, marker="",
    )
    ax = _render_single_series(series)
    assert len(ax.lines) == 1
    assert len(ax.collections) == 0
    assert ax.lines[0].get_marker() in ("None", "none", None)


def test_line_marker_plot_type_renders_both_line_and_markers():
    dataset = _make_dataset()
    series = Series3D(
        dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
        plot_type=Plot3DType.LINE_MARKER, marker="o",
    )
    ax = _render_single_series(series)
    assert len(ax.lines) == 1
    assert ax.lines[0].get_marker() == "o"


def test_line_style_is_applied():
    dataset = _make_dataset()
    series = Series3D(
        dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
        plot_type=Plot3DType.LINE, line_style="--",
    )
    ax = _render_single_series(series)
    assert ax.lines[0].get_linestyle() == "--"


def test_line_width_is_applied():
    dataset = _make_dataset()
    series = Series3D(
        dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
        plot_type=Plot3DType.LINE, line_width=3.5,
    )
    ax = _render_single_series(series)
    assert ax.lines[0].get_linewidth() == pytest.approx(3.5)


def test_line_marker_size_is_applied():
    dataset = _make_dataset()
    series = Series3D(
        dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
        plot_type=Plot3DType.LINE_MARKER, marker="o", marker_size=12.0,
    )
    ax = _render_single_series(series)
    assert ax.lines[0].get_markersize() == pytest.approx(12.0)


def test_series_visibility_hides_it_regardless_of_plot_type():
    dataset = _make_dataset()
    series = Series3D(
        dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
        plot_type=Plot3DType.LINE, visible=False,
    )
    ax = _render_single_series(series)
    assert len(ax.lines) == 0
    assert len(ax.collections) == 0


def test_line_transparency_is_applied():
    dataset = _make_dataset()
    series = Series3D(
        dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
        plot_type=Plot3DType.LINE, alpha=0.3,
    )
    ax = _render_single_series(series)
    assert ax.lines[0].get_alpha() == pytest.approx(0.3)


def test_grouped_family_renders_no_cross_group_connection():
    """Each group is a genuinely separate `Series3D`/artist -- connecting
    across groups is structurally impossible, not just avoided by
    convention."""
    dataset = _diode_dataset()
    panel = Panel3D()
    panel.add_series(
        Series3D(
            dataset=dataset, x_column="Voltage_V", y_column="Temperature_C", z_column="Current_mA",
            plot_type=Plot3DType.LINE, label="25", row_indices=(0, 2, 4),
        )
    )
    panel.add_series(
        Series3D(
            dataset=dataset, x_column="Voltage_V", y_column="Temperature_C", z_column="Current_mA",
            plot_type=Plot3DType.LINE, label="35", row_indices=(1, 3, 5),
        )
    )
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    assert len(ax.lines) == 2  # two independent line artists, never merged
    xs_25 = ax.lines[0].get_data_3d()[0]
    xs_35 = ax.lines[1].get_data_3d()[0]
    assert list(xs_25) == [0.1, 0.2, 0.3]
    assert list(xs_35) == [0.1, 0.2, 0.3]


# --- Automatic color cycle -----------------------------------------------------------


def test_grouped_series_receive_distinct_deterministic_theme_cycle_colors():
    dataset = _diode_dataset()
    panel = Panel3D()
    s1 = Series3D(dataset=dataset, x_column="Voltage_V", y_column="Temperature_C", z_column="Current_mA", row_indices=(0, 2, 4))
    s2 = Series3D(dataset=dataset, x_column="Voltage_V", y_column="Temperature_C", z_column="Current_mA", row_indices=(1, 3, 5))
    panel.add_series(s1, dark_mode=False)
    panel.add_series(s2, dark_mode=False)

    assert s1.color != s2.color
    assert s1.color is not None and s2.color is not None

    # Deterministic: re-doing the same sequence from a fresh panel gives
    # the exact same colors in the exact same order.
    panel_again = Panel3D()
    s1b = Series3D(dataset=dataset, x_column="Voltage_V", y_column="Temperature_C", z_column="Current_mA", row_indices=(0, 2, 4))
    s2b = Series3D(dataset=dataset, x_column="Voltage_V", y_column="Temperature_C", z_column="Current_mA", row_indices=(1, 3, 5))
    panel_again.add_series(s1b, dark_mode=False)
    panel_again.add_series(s2b, dark_mode=False)
    assert (s1.color, s2.color) == (s1b.color, s2b.color)


def test_manual_color_override_is_never_reassigned_by_add_series():
    dataset = _make_dataset()
    panel = Panel3D()
    series = Series3D(
        dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", color="#ff00ff",
    )
    panel.add_series(series)
    assert series.color == "#ff00ff"


# --- Legend ----------------------------------------------------------------------


def test_legend_contains_one_entry_per_visible_grouped_series():
    dataset = _diode_dataset()
    panel = Panel3D(legend_visible=True)
    panel.add_series(
        Series3D(dataset=dataset, x_column="Voltage_V", y_column="Temperature_C", z_column="Current_mA", label="25", row_indices=(0, 2, 4))
    )
    panel.add_series(
        Series3D(dataset=dataset, x_column="Voltage_V", y_column="Temperature_C", z_column="Current_mA", label="35", row_indices=(1, 3, 5))
    )
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert set(labels) == {"25", "35"}


def test_legend_excludes_hidden_series():
    dataset = _diode_dataset()
    panel = Panel3D(legend_visible=True)
    panel.add_series(
        Series3D(dataset=dataset, x_column="Voltage_V", y_column="Temperature_C", z_column="Current_mA", label="25", row_indices=(0, 2, 4))
    )
    panel.add_series(
        Series3D(
            dataset=dataset, x_column="Voltage_V", y_column="Temperature_C", z_column="Current_mA", label="35",
            row_indices=(1, 3, 5), visible=False,
        )
    )
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    legend = ax.get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["25"]


def test_legend_visibility_toggle_removes_the_legend():
    dataset = _make_dataset()
    panel = Panel3D(legend_visible=False)
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", label="s1"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    assert ax.get_legend() is None


def test_legend_location_is_applied():
    dataset = _make_dataset()
    panel = Panel3D(legend_visible=True, legend_loc="upper right")
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", label="s1"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    legend = ax.get_legend()
    assert legend is not None
    assert legend._get_loc() == 1  # Matplotlib's internal code for "upper right"


def test_legend_labels_match_series_labels_exactly():
    dataset = _make_dataset()
    panel = Panel3D(legend_visible=True)
    panel.add_series(
        Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", label="Custom Label")
    )
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["Custom Label"]


# --- Grid style (private-API-backed, see _apply_3d_grid_style's own docstring) ---


def test_grid_style_linestyle_and_width_are_applied():
    dataset = _make_dataset()
    panel = Panel3D(grid_linestyle=":", grid_linewidth=3.0)
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        assert axis._axinfo["grid"]["linestyle"] == ":"
        assert axis._axinfo["grid"]["linewidth"] == 3.0


def test_grid_alpha_is_baked_into_the_rgba_color_not_a_bare_alpha_key():
    """Confirmed via direct Matplotlib source inspection: `axis3d.py`'s own
    draw code never reads a bare `alpha` dict key, only `color`/
    `linewidth`/`linestyle` -- alpha MUST be part of the RGBA color tuple."""
    dataset = _make_dataset()
    panel = Panel3D(grid_color="#00ff00", grid_alpha=0.4)
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    color = ax.xaxis._axinfo["grid"]["color"]
    assert color == pytest.approx((0.0, 1.0, 0.0, 0.4))


def test_grid_color_none_uses_theme_appropriate_default():
    dataset = _make_dataset()
    panel_light = Panel3D(grid_color=None)
    panel_light.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax_light = mpl_figure.add_subplot(1, 2, 1, projection="3d")
    ax_dark = mpl_figure.add_subplot(1, 2, 2, projection="3d")
    render_panel_3d(ax_light, panel_light, dark_mode=False)
    render_panel_3d(ax_dark, panel_light, dark_mode=True)

    assert ax_light.xaxis._axinfo["grid"]["color"] != ax_dark.xaxis._axinfo["grid"]["color"]


def test_grid_style_exported_svg_reflects_manual_color():
    """End-to-end confirmation (not just the internal dict) -- the styled
    color must actually reach the exported/rendered output, exactly as
    verified experimentally during this milestone's own architecture
    inspection."""
    dataset = _make_dataset()
    panel = Panel3D(grid_color="#123abc", grid_alpha=1.0)
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)
    buf = io.BytesIO()
    mpl_figure.savefig(buf, format="svg")
    content = buf.getvalue().decode("utf-8").lower()
    assert "123abc" in content


def test_grid_style_gracefully_falls_back_if_axinfo_is_unavailable(monkeypatch):
    """The one function in GNOVI that touches `_axinfo` must never crash a
    render if a future Matplotlib release restructures it -- simulated
    here by making the private dict inaccessible."""
    dataset = _make_dataset()
    panel = Panel3D(grid_color="#ff0000")
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")

    class _BrokenAxis:
        def __getattr__(self, _name):
            raise AttributeError("_axinfo restructured")

    monkeypatch.setattr(ax, "xaxis", _BrokenAxis())
    _apply_3d_grid_style(ax, panel, dark_mode=False)  # must not raise


def test_grid_style_disabled_when_grid_is_off():
    dataset = _make_dataset()
    panel = Panel3D(grid=False, grid_color="#ff0000")
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)  # must not raise, grid style is simply skipped


# --- Panes (public Axis.pane API) -------------------------------------------------


def test_pane_visibility_is_applied():
    dataset = _make_dataset()
    panel = Panel3D(pane_visible=False)
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        assert axis.pane.get_visible() is False


def test_pane_manual_color_is_applied():
    dataset = _make_dataset()
    panel = Panel3D(pane_color="#123456")
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    from matplotlib.colors import to_rgba

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        assert axis.pane.get_facecolor() == pytest.approx(to_rgba("#123456"))


def test_pane_color_none_follows_theme():
    dataset = _make_dataset()
    panel = Panel3D(pane_color=None)
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax_light = mpl_figure.add_subplot(1, 2, 1, projection="3d")
    ax_dark = mpl_figure.add_subplot(1, 2, 2, projection="3d")
    render_panel_3d(ax_light, panel, dark_mode=False)
    render_panel_3d(ax_dark, panel, dark_mode=True)

    assert ax_light.xaxis.pane.get_facecolor() != ax_dark.xaxis.pane.get_facecolor()


def test_pane_alpha_is_applied():
    dataset = _make_dataset()
    panel = Panel3D(pane_alpha=0.3)
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        assert axis.pane.get_alpha() == pytest.approx(0.3)


# --- Legend columns / frame ------------------------------------------------------


def test_legend_ncol_is_applied_with_enough_series():
    dataset = _make_dataset()
    panel = Panel3D(legend_visible=True, legend_ncol=3)
    for i in range(6):
        panel.add_series(
            Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", label=f"s{i}")
        )
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    legend = ax.get_legend()
    assert legend._ncols == 3


def test_legend_frame_off_is_applied():
    dataset = _make_dataset()
    panel = Panel3D(legend_visible=True, legend_frameon=False)
    panel.add_series(
        Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", label="s1")
    )
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    legend = ax.get_legend()
    assert legend.get_frame_on() is False


def test_legend_supports_many_grouped_curves_with_multiple_columns():
    """The grouped-diode-family scenario at realistic scale -- 10 groups,
    3 legend columns, must render without error and produce all 10
    entries."""
    dataset = _make_dataset()
    panel = Panel3D(legend_visible=True, legend_ncol=3)
    for i in range(10):
        panel.add_series(
            Series3D(
                dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
                label=f"{20 + i * 10}",
            )
        )
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)  # must not raise

    legend = ax.get_legend()
    assert len(legend.get_texts()) == 10
    assert legend._ncols == 3


# --- Scientific aspect (set_aspect, never set_box_aspect) ------------------------


def test_aspect_auto_is_applied():
    dataset = _make_dataset()
    panel = Panel3D(aspect_mode="auto")
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    assert ax.get_aspect() == "auto"


def test_aspect_equal_is_applied():
    dataset = _make_dataset()
    panel = Panel3D(aspect_mode="equal")
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    assert ax.get_aspect() == "equal"


# --- Tick locators (public per-axis API) ------------------------------------------


def test_major_tick_spacing_applied_per_axis():
    dataset = _make_dataset()
    panel = Panel3D(major_tick_spacing_x=25.0, major_tick_spacing_y=0.05)
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    xticks = ax.get_xticks()
    diffs = [round(b - a, 6) for a, b in zip(xticks, xticks[1:])]
    assert all(d == pytest.approx(25.0) for d in diffs)


def test_minor_tick_spacing_applied_per_axis():
    dataset = _make_dataset()
    panel = Panel3D(minor_tick_spacing_x=10.0)
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)

    minor_ticks = ax.get_xticks(minor=True)
    assert len(minor_ticks) > 0


def test_no_tick_spacing_set_uses_automatic_placement():
    dataset = _make_dataset()
    panel = Panel3D()
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.add_subplot(1, 1, 1, projection="3d")
    render_panel_3d(ax, panel)  # must not raise
    assert len(ax.get_xticks(minor=True)) == 0
