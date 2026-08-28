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
from gnovi_plot.plotting.series3d import Series3D


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
