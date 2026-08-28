"""`Panel3D`/`Series3D` -- the renderer-independent 3D scatter data model
(see `plotting.figure.Panel3D`, `plotting.series3d.Series3D`). No
Matplotlib import anywhere in the model layer; rendering is entirely
`plotting.backends.matplotlib_backend`'s concern (see
`test_matplotlib_backend_3d.py`).
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_xyz
from gnovi_plot.plotting.figure import GnoviFigure, Panel, Panel3D, panel_from_dict
from gnovi_plot.plotting.series3d import Plot3DType, Series3D


def _make_dataset(name="mat"):
    df = pd.DataFrame(
        {"temperature": [300.0, 350.0, 400.0, 300.0], "composition": [0.1, 0.1, 0.1, 0.2], "conductivity": [2.4, 2.9, 3.5, 3.1]}
    )
    return Dataset(name=name, dataframe=df)


# --- Model: construction, identity, matplotlib-free ------------------------------


def test_panel3d_construction():
    panel = Panel3D(title="Conductivity surface", x_label="T", y_label="Composition", z_label="Conductivity")
    assert panel.title == "Conductivity surface"
    assert panel.series == []


def test_panel3d_has_a_stable_unique_id():
    a, b = Panel3D(), Panel3D()
    assert a.id and b.id
    assert a.id != b.id


@pytest.mark.parametrize("module_path", ["gnovi_plot/plotting/figure.py", "gnovi_plot/plotting/series3d.py"])
def test_domain_model_files_import_no_matplotlib(module_path):
    """The absolute rule this milestone is built on: the scientific/project
    model stays renderer-independent so a future renderer can be added
    without rewriting persisted state. Checked via the actual `import`
    statements (AST-parsed, not a substring search of the file -- which
    would false-positive on the word "Matplotlib" appearing in prose
    docstrings, e.g. "renders as a Matplotlib Axes3D")."""
    tree = ast.parse((Path(__file__).parent.parent / module_path).read_text())
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert "matplotlib" not in imported_roots
    assert "mpl_toolkits" not in imported_roots


# --- 3D series: Dataset reference, X/Y/Z persistence, identity -------------------


def test_series3d_references_the_dataset_correctly():
    dataset = _make_dataset()
    series = Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity")
    assert series.dataset is dataset


def test_series3d_xyz_columns_persist_through_to_dict_from_dict():
    dataset = _make_dataset()
    series = Series3D(
        dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
        marker="s", marker_size=9.0, color="#123456", alpha=0.7,
    )
    data = series.to_dict()
    restored = Series3D.from_dict(data, {dataset.id: dataset})
    assert restored.x_column == "temperature"
    assert restored.y_column == "composition"
    assert restored.z_column == "conductivity"
    assert restored.marker == "s"
    assert restored.marker_size == 9.0
    assert restored.color == "#123456"
    assert restored.alpha == 0.7
    assert restored.id == series.id


def test_series3d_from_dict_returns_none_for_an_unresolvable_dataset():
    series = Series3D(dataset=_make_dataset(), x_column="temperature", y_column="composition", z_column="conductivity")
    assert Series3D.from_dict(series.to_dict(), {}) is None


# --- 3D series: plot_type/line_style/line_width/row_indices ----------------------


def test_series3d_plot_type_defaults_to_scatter():
    series = Series3D(dataset=_make_dataset(), x_column="temperature", y_column="composition", z_column="conductivity")
    assert series.plot_type == Plot3DType.SCATTER


def test_series3d_plot_type_line_style_width_persist_through_to_dict_from_dict():
    dataset = _make_dataset()
    series = Series3D(
        dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
        plot_type=Plot3DType.LINE_MARKER, line_style="--", line_width=2.5,
    )
    restored = Series3D.from_dict(series.to_dict(), {dataset.id: dataset})
    assert restored.plot_type == Plot3DType.LINE_MARKER
    assert restored.line_style == "--"
    assert restored.line_width == 2.5


def test_series3d_from_dict_defaults_plot_type_for_a_pre_grouping_dict():
    """A dict saved before this milestone has no "plot_type"/"line_style"/
    "line_width"/"row_indices" keys at all -- must still load as a plain
    scatter, matching the milestone's own backward-compatibility decision
    (no PROJECT_FORMAT_VERSION bump was needed)."""
    dataset = _make_dataset()
    legacy_dict = Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity").to_dict()
    for key in ("plot_type", "line_style", "line_width", "row_indices"):
        del legacy_dict[key]
    restored = Series3D.from_dict(legacy_dict, {dataset.id: dataset})
    assert restored.plot_type == Plot3DType.SCATTER
    assert restored.line_style == "-"
    assert restored.line_width == 1.5
    assert restored.row_indices is None


def test_series3d_row_indices_selects_the_correct_dataframe_subset():
    dataset = _make_dataset()  # temperature: [300.0, 350.0, 400.0, 300.0]
    series = Series3D(
        dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
        row_indices=(0, 3),
    )
    assert list(series.dataframe["temperature"]) == [300.0, 300.0]
    assert len(series.dataframe) == 2


def test_series3d_row_indices_none_means_the_whole_dataset():
    dataset = _make_dataset()
    series = Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity")
    assert series.dataframe is dataset.dataframe


def test_series3d_row_indices_persist_through_to_dict_from_dict():
    dataset = _make_dataset()
    series = Series3D(
        dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
        row_indices=(0, 2, 3),
    )
    restored = Series3D.from_dict(series.to_dict(), {dataset.id: dataset})
    assert restored.row_indices == (0, 2, 3)


def test_series3d_row_indices_out_of_bounds_raises():
    dataset = _make_dataset()  # 4 rows
    with pytest.raises(ValueError):
        Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", row_indices=(0, 99))


def test_series3d_row_indices_empty_tuple_raises():
    dataset = _make_dataset()
    with pytest.raises(ValueError):
        Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", row_indices=())


def test_series3d_no_dataset_duplication_shares_the_live_dataframe_object():
    """Grouped series never copy the source data -- `.dataframe` is always
    derived (via `.iloc`) from the SAME live `dataset.dataframe`, never a
    stored/duplicated copy."""
    dataset = _make_dataset()
    series_a = Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", row_indices=(0, 1))
    series_b = Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", row_indices=(2, 3))
    assert series_a.dataset is dataset
    assert series_b.dataset is dataset
    assert series_a.dataset is series_b.dataset


# --- Panel3D: invalidate_series_for_dataset with row_indices ---------------------


def test_invalidate_series_for_dataset_marks_row_indices_series_stale_on_row_set_change():
    dataset = _make_dataset()
    panel = Panel3D()
    series = Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", row_indices=(0, 1))
    panel.add_series(series)

    newly_stale = panel.invalidate_series_for_dataset(dataset, row_set_changed=True)

    assert series in newly_stale
    assert series.stale is True


def test_invalidate_series_for_dataset_leaves_ungrouped_series_untouched_on_row_set_change():
    dataset = _make_dataset()
    panel = Panel3D()
    series = Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity")
    panel.add_series(series)

    newly_stale = panel.invalidate_series_for_dataset(dataset, row_set_changed=True)

    assert newly_stale == []
    assert series.stale is False


def test_invalidate_series_for_dataset_ignores_row_indices_when_row_set_unchanged():
    dataset = _make_dataset()
    panel = Panel3D()
    series = Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity", row_indices=(0, 1))
    panel.add_series(series)

    newly_stale = panel.invalidate_series_for_dataset(dataset, row_set_changed=False)

    assert newly_stale == []
    assert series.stale is False


# --- Panel3D: legend fields --------------------------------------------------------


def test_panel3d_legend_defaults():
    panel = Panel3D()
    assert panel.legend_visible is True
    assert panel.legend_loc == "best"


def test_panel3d_legend_fields_persist_through_to_dict_from_dict():
    panel = Panel3D(legend_visible=False, legend_loc="upper right")
    restored = Panel3D.from_dict(panel.to_dict(), {})
    assert restored.legend_visible is False
    assert restored.legend_loc == "upper right"


def test_panel3d_legend_defaults_for_a_pre_legend_dict():
    """A dict saved before this milestone has no legend keys at all."""
    legacy_dict = Panel3D().to_dict()
    del legacy_dict["legend_visible"]
    del legacy_dict["legend_loc"]
    restored = Panel3D.from_dict(legacy_dict, {})
    assert restored.legend_visible is True
    assert restored.legend_loc == "best"


# --- Mixed GnoviFigure: Panel and Panel3D coexist ---------------------------------


def test_mixed_gnovi_figure_accepts_panel_and_panel3d():
    dataset = _make_dataset()
    panel_2d = Panel(title="2D")
    panel_3d = Panel3D(title="3D")
    panel_3d.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    figure = GnoviFigure(panels=[panel_2d, panel_3d], layout=(1, 2))
    assert [type(p).__name__ for p in figure.panels] == ["Panel", "Panel3D"]


def test_dataset_identity_is_shared_not_duplicated_in_a_panel3d():
    dataset = _make_dataset()
    panel = Panel3D()
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    assert panel.series[0].dataset is dataset


def test_existing_pure_2d_gnovi_figure_behavior_is_unchanged():
    figure = GnoviFigure()
    assert isinstance(figure.active_panel, Panel)
    figure.set_layout(1, 3)
    assert all(isinstance(p, Panel) for p in figure.panels)
    assert figure.layout == (1, 3)


def test_panel_from_dict_dispatches_on_kind():
    dataset = _make_dataset()
    panel_3d = Panel3D(title="3D")
    panel_3d.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    restored_3d = panel_from_dict(panel_3d.to_dict(), {dataset.id: dataset})
    assert isinstance(restored_3d, Panel3D)

    panel_2d = Panel(title="2D")
    restored_2d = panel_from_dict(panel_2d.to_dict(), {})
    assert isinstance(restored_2d, Panel)


def test_panel_from_dict_defaults_a_missing_kind_to_2d():
    """A project saved before Panel3D existed has plain Panel dicts with
    no "kind" key at all -- must still resolve to Panel, not raise."""
    legacy_panel_dict = Panel(title="Old").to_dict()
    del legacy_panel_dict["kind"]
    restored = panel_from_dict(legacy_panel_dict, {})
    assert isinstance(restored, Panel)


# --- Numeric XYZ extraction: row alignment, NaN handling -------------------------


def _xyz_dataframe():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0, "bad", 6.0],
            "y": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "z": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
        }
    )


def test_xyz_extraction_preserves_row_alignment():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0], "z": [100.0, 200.0, 300.0]})
    x, y, z = numeric_xyz(df, "x", "y", "z")
    assert list(x) == [1.0, 2.0, 3.0]
    assert list(y) == [10.0, 20.0, 30.0]
    assert list(z) == [100.0, 200.0, 300.0]


def test_rows_with_invalid_x_are_excluded_as_a_whole_row():
    df = _xyz_dataframe()
    x, y, z = numeric_xyz(df, "x", "y", "z", min_points=1)
    # Row index 2 has NaN x -- excluded entirely, including its (valid) y/z.
    assert 3.0 not in y.to_numpy()  # y value that belonged to the dropped x=NaN row


def test_rows_with_invalid_y_are_excluded_as_a_whole_row():
    df = _xyz_dataframe()
    x, y, z = numeric_xyz(df, "x", "y", "z", min_points=1)
    # Row index 1 has NaN y -- its x=2.0 must not survive either.
    assert 2.0 not in x.to_numpy()


def test_rows_with_invalid_z_are_excluded_as_a_whole_row():
    df = _xyz_dataframe()
    x, y, z = numeric_xyz(df, "x", "y", "z", min_points=1)
    # Row index 3 has NaN z -- its x=4.0 must not survive either.
    assert 4.0 not in x.to_numpy()


def test_mixed_numeric_and_non_numeric_input_is_handled_safely():
    df = _xyz_dataframe()
    x, y, z = numeric_xyz(df, "x", "y", "z", min_points=1)
    # Only rows 0 (1.0/1.0/1.0) and 5 (6.0/6.0/6.0) are valid across all
    # three columns -- rows 1/2/3/4 each have one NaN or non-numeric value.
    assert list(x) == [1.0, 6.0]
    assert list(y) == [1.0, 6.0]
    assert list(z) == [1.0, 6.0]
    assert len(x) == len(y) == len(z)


def test_unusable_xyz_input_raises_a_controlled_error():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1.0, 2.0], "z": [1.0, 2.0]})
    with pytest.raises(InsufficientNumericDataError):
        numeric_xyz(df, "x", "y", "z")


# --- Graph Library: Panel3D support ------------------------------------------------


def test_graph_library_saves_and_reloads_a_panel3d():
    from gnovi_plot.data.dataset_manager import DatasetManager
    from gnovi_plot.plotting.graph_library import GraphLibrary

    dataset = _make_dataset()
    manager = DatasetManager()
    manager.add(dataset)
    panel = Panel3D(title="3D")
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    figure = GnoviFigure(panels=[panel])

    library = GraphLibrary()
    graph = library.save_panel_as_graph(figure, "3D Graph", manager)

    assert isinstance(graph.panel, Panel3D)
    reloaded_library = GraphLibrary.from_dict(library.to_dict(), {dataset.id: dataset})
    assert isinstance(reloaded_library.get(graph.id).panel, Panel3D)


def test_graph_library_load_graph_into_panel_restores_a_panel3d():
    from gnovi_plot.data.dataset_manager import DatasetManager
    from gnovi_plot.plotting.graph_library import GraphLibrary

    dataset = _make_dataset()
    manager = DatasetManager()
    manager.add(dataset)
    panel = Panel3D(title="3D")
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    source_figure = GnoviFigure(panels=[panel])

    library = GraphLibrary()
    graph = library.save_panel_as_graph(source_figure, "3D Graph", manager)

    target_figure = GnoviFigure()  # a plain 2D default Figure
    loaded = library.load_graph_into_panel(graph.id, target_figure, manager)

    assert loaded is True
    assert isinstance(target_figure.active_panel, Panel3D)
    assert target_figure.active_panel.id != panel.id  # independent copy


def test_graph_library_preserves_a_grouped_curve_family():
    from gnovi_plot.data.dataset_manager import DatasetManager
    from gnovi_plot.plotting.graph_library import GraphLibrary

    dataset = _make_dataset()
    manager = DatasetManager()
    manager.add(dataset)
    panel = Panel3D(title="Grouped", legend_visible=True, legend_loc="lower left")
    panel.add_series(
        Series3D(
            dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
            label="A", plot_type=Plot3DType.LINE, row_indices=(0, 1),
        )
    )
    panel.add_series(
        Series3D(
            dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
            label="B", plot_type=Plot3DType.SCATTER, row_indices=(2, 3),
        )
    )
    source_figure = GnoviFigure(panels=[panel])

    library = GraphLibrary()
    graph = library.save_panel_as_graph(source_figure, "Grouped Graph", manager)
    reloaded_library = GraphLibrary.from_dict(library.to_dict(), {dataset.id: dataset})
    reloaded_panel = reloaded_library.get(graph.id).panel

    assert isinstance(reloaded_panel, Panel3D)
    assert len(reloaded_panel.series) == 2
    by_label = {s.label: s for s in reloaded_panel.series}
    assert by_label["A"].row_indices == (0, 1)
    assert by_label["A"].plot_type == Plot3DType.LINE
    assert by_label["B"].row_indices == (2, 3)
    assert reloaded_panel.legend_visible is True
    assert reloaded_panel.legend_loc == "lower left"

    target_figure = GnoviFigure()
    loaded = reloaded_library.load_graph_into_panel(graph.id, target_figure, manager)
    assert loaded is True
    assert len(target_figure.active_panel.series) == 2
    assert all(s.dataset is dataset for s in target_figure.active_panel.series)


# --- Panel3D: publication-polish fields (grid style/panes/legend/aspect/ticks) ---


def test_panel3d_polish_field_defaults_reproduce_current_rendering():
    """Defaults must exactly match Matplotlib's own unstyled `Axes3D`
    appearance (confirmed directly against the installed Matplotlib
    version -- see `matplotlib_backend._apply_3d_grid_style`'s own
    docstring), so a project saved before these fields existed renders
    identically after loading."""
    panel = Panel3D()
    assert panel.grid_linestyle == "-"
    assert panel.grid_linewidth == 0.8
    assert panel.grid_alpha == 1.0
    assert panel.grid_color is None
    assert panel.pane_visible is True
    assert panel.pane_color is None
    assert panel.pane_alpha == 1.0
    assert panel.legend_ncol == 1
    assert panel.legend_frameon is True
    assert panel.aspect_mode == "auto"
    assert panel.major_tick_spacing_x is None
    assert panel.major_tick_spacing_y is None
    assert panel.major_tick_spacing_z is None
    assert panel.minor_tick_spacing_x is None
    assert panel.minor_tick_spacing_y is None
    assert panel.minor_tick_spacing_z is None


def test_panel3d_polish_fields_persist_through_to_dict_from_dict():
    panel = Panel3D(
        grid_linestyle="--", grid_linewidth=2.5, grid_alpha=0.4, grid_color="#ff0000",
        pane_visible=False, pane_color="#00ff00", pane_alpha=0.6,
        legend_ncol=3, legend_frameon=False,
        aspect_mode="equal",
        major_tick_spacing_x=1.0, major_tick_spacing_y=2.0, major_tick_spacing_z=3.0,
        minor_tick_spacing_x=0.1, minor_tick_spacing_y=0.2, minor_tick_spacing_z=0.3,
    )
    restored = Panel3D.from_dict(panel.to_dict(), {})
    assert restored.grid_linestyle == "--"
    assert restored.grid_linewidth == 2.5
    assert restored.grid_alpha == 0.4
    assert restored.grid_color == "#ff0000"
    assert restored.pane_visible is False
    assert restored.pane_color == "#00ff00"
    assert restored.pane_alpha == 0.6
    assert restored.legend_ncol == 3
    assert restored.legend_frameon is False
    assert restored.aspect_mode == "equal"
    assert restored.major_tick_spacing_x == 1.0
    assert restored.major_tick_spacing_y == 2.0
    assert restored.major_tick_spacing_z == 3.0
    assert restored.minor_tick_spacing_x == 0.1
    assert restored.minor_tick_spacing_y == 0.2
    assert restored.minor_tick_spacing_z == 0.3


def test_panel3d_polish_fields_default_for_a_pre_polish_dict():
    """A dict saved before this milestone has none of these keys at all --
    must still load with defaults that reproduce prior (unstyled)
    rendering, confirming no `PROJECT_FORMAT_VERSION` bump was needed."""
    legacy_dict = Panel3D().to_dict()
    for key in (
        "grid_linestyle", "grid_linewidth", "grid_alpha", "grid_color",
        "pane_visible", "pane_color", "pane_alpha",
        "legend_ncol", "legend_frameon", "aspect_mode",
        "major_tick_spacing_x", "major_tick_spacing_y", "major_tick_spacing_z",
        "minor_tick_spacing_x", "minor_tick_spacing_y", "minor_tick_spacing_z",
    ):
        del legacy_dict[key]
    restored = Panel3D.from_dict(legacy_dict, {})
    assert restored.grid_linestyle == "-"
    assert restored.grid_alpha == 1.0
    assert restored.pane_visible is True
    assert restored.legend_ncol == 1
    assert restored.aspect_mode == "auto"
    assert restored.major_tick_spacing_x is None


def test_panel3d_polish_fields_survive_deepcopy():
    """Undo/redo relies on plain `copy.deepcopy` (see `gui.undo_manager.
    snapshot_figure`) -- confirm the new fields round-trip through it like
    any other dataclass field (no custom `__deepcopy__`/`__reduce__` on
    `Panel3D` that could silently drop them)."""
    import copy

    panel = Panel3D(grid_color="#123456", aspect_mode="equal", legend_ncol=4)
    cloned = copy.deepcopy(panel)
    assert cloned.grid_color == "#123456"
    assert cloned.aspect_mode == "equal"
    assert cloned.legend_ncol == 4
    assert cloned is not panel


def test_graph_library_preserves_publication_polish_styling():
    from gnovi_plot.data.dataset_manager import DatasetManager
    from gnovi_plot.plotting.graph_library import GraphLibrary

    dataset = _make_dataset()
    manager = DatasetManager()
    manager.add(dataset)
    panel = Panel3D(
        title="Polished", grid_color="#ff00ff", pane_visible=False, legend_ncol=3,
        aspect_mode="equal", major_tick_spacing_x=1.0,
    )
    panel.add_series(Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity"))
    source_figure = GnoviFigure(panels=[panel])

    library = GraphLibrary()
    graph = library.save_panel_as_graph(source_figure, "Polished Graph", manager)
    reloaded_library = GraphLibrary.from_dict(library.to_dict(), {dataset.id: dataset})
    reloaded_panel = reloaded_library.get(graph.id).panel

    assert reloaded_panel.grid_color == "#ff00ff"
    assert reloaded_panel.pane_visible is False
    assert reloaded_panel.legend_ncol == 3
    assert reloaded_panel.aspect_mode == "equal"
    assert reloaded_panel.major_tick_spacing_x == 1.0

    target_figure = GnoviFigure()
    loaded = reloaded_library.load_graph_into_panel(graph.id, target_figure, manager)
    assert loaded is True
    assert target_figure.active_panel.aspect_mode == "equal"
    assert target_figure.active_panel.legend_ncol == 3
