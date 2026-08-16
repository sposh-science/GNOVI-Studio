import inspect
import json
import zipfile

import pandas as pd
import pandas.testing as pdt
import pytest

from gnovi_plot.analysis.cycles import detect_cycles
from gnovi_plot.core import project_io
from gnovi_plot.core.project import Project
from gnovi_plot.core.project_io import (
    PROJECT_FORMAT_VERSION,
    CorruptProjectError,
    ProjectIOError,
    UnsupportedProjectVersionError,
    load_project,
    save_project,
)
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.figure import GnoviFigure, PlotTheme
from gnovi_plot.plotting.series import PlotSeries


def _simple_dataset(name="sample", source_path=None):
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    return Dataset(name=name, dataframe=df, source_path=source_path)


def _basic_project(source_path=None):
    dataset = _simple_dataset(source_path=source_path)
    project = Project.new()
    project.dataset_manager.add(dataset)
    project.figures[0].add_series(PlotSeries.line(dataset, "x", "y"))
    return project, dataset


# --- Basic save/load round trip -----------------------------------------------


def test_save_project_writes_a_gnovi_file(tmp_path):
    project, _dataset = _basic_project()
    out_path = save_project(project, tmp_path / "proj.gnovi")
    assert out_path.exists()
    assert project.path == out_path


def test_load_project_round_trips_dataset_series_and_id(tmp_path):
    project, dataset = _basic_project()
    out_path = save_project(project, tmp_path / "proj.gnovi")

    loaded = load_project(out_path)

    assert len(loaded.dataset_manager.datasets) == 1
    loaded_dataset = loaded.dataset_manager.datasets[0]
    assert loaded_dataset.id == dataset.id
    pdt.assert_frame_equal(loaded_dataset.dataframe, dataset.dataframe)
    assert len(loaded.figures) == 1
    assert len(loaded.figures[0].series) == 1
    assert loaded.figures[0].series[0].dataset is loaded_dataset


def test_save_project_as_gives_the_project_a_path(tmp_path):
    project, _dataset = _basic_project()
    assert project.path is None
    save_project(project, tmp_path / "proj.gnovi")
    assert project.path == tmp_path / "proj.gnovi"


# --- Plot Theme: a project must reopen looking exactly as it was saved --------


def test_light_theme_project_round_trips_and_reopens_light(tmp_path):
    project, _dataset = _basic_project()
    project.figures[0].plot_theme = PlotTheme.LIGHT

    out_path = save_project(project, tmp_path / "light.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    assert manifest["figures"][0]["plot_theme"] == "light"

    loaded = load_project(out_path)
    assert loaded.figures[0].plot_theme == PlotTheme.LIGHT


def test_dark_theme_project_round_trips_and_reopens_dark(tmp_path):
    project, _dataset = _basic_project()
    project.figures[0].plot_theme = PlotTheme.DARK

    out_path = save_project(project, tmp_path / "dark.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    assert manifest["figures"][0]["plot_theme"] == "dark"

    loaded = load_project(out_path)
    assert loaded.figures[0].plot_theme == PlotTheme.DARK


def test_saved_theme_is_restored_after_reopen_even_when_it_differs_from_the_default(tmp_path):
    """A project saved with a non-default (Dark) theme must come back Dark
    on reopen -- the point of this whole correction: a project/figure must
    reopen looking exactly as it was saved, independent of whatever the
    QSettings default-for-new-figures happens to be at open time."""
    project, dataset = _basic_project()
    assert project.figures[0].plot_theme == PlotTheme.LIGHT  # the ordinary default
    project.figures[0].plot_theme = PlotTheme.DARK
    out_path = save_project(project, tmp_path / "explicit_dark.gnovi")

    loaded = load_project(out_path)

    assert loaded.figures[0].plot_theme == PlotTheme.DARK
    # Nothing else about the round trip regressed alongside the theme fix.
    assert loaded.dataset_manager.datasets[0].id == dataset.id
    assert len(loaded.figures[0].series) == 1


# --- Figure Aspect Ratio vs. Panel Aspect Ratio: independent round trip -------


def test_figure_and_panel_aspect_ratio_round_trip_independently(tmp_path):
    """The exact scenario from the task: Figure aspect = 16:9, Panel aspect
    = 1:1, Layout = 2x3 -- save, reopen, all three must be restored exactly,
    never inferred from the current canvas."""
    project, _dataset = _basic_project()
    figure = project.figures[0]
    figure.set_layout(2, 3)
    figure.figure_width_in = 16.0
    figure.figure_height_in = 9.0
    figure.aspect_preset = "16:9"
    figure.lock_aspect_ratio = True
    figure.panel_aspect_preset = "1:1"
    out_path = save_project(project, tmp_path / "figure_vs_panel_aspect.gnovi")

    loaded = load_project(out_path)
    loaded_figure = loaded.figures[0]

    assert loaded_figure.layout == (2, 3)
    assert loaded_figure.aspect_preset == "16:9"
    assert loaded_figure.lock_aspect_ratio is True
    assert loaded_figure.panel_aspect_preset == "1:1"


def test_panel_aspect_ratio_persists_independently_of_figure_aspect_ratio(tmp_path):
    """Changing one must never affect the other, including across a save/
    reopen round trip."""
    project, _dataset = _basic_project()
    figure = project.figures[0]
    figure.aspect_preset = "4:3"
    figure.lock_aspect_ratio = True
    figure.panel_aspect_preset = "16:9"
    out_path = save_project(project, tmp_path / "independent_aspects.gnovi")

    loaded = load_project(out_path)

    assert loaded.figures[0].aspect_preset == "4:3"
    assert loaded.figures[0].panel_aspect_preset == "16:9"


def test_panel_aspect_ratio_defaults_to_auto_for_a_project_saved_before_the_feature_existed(tmp_path):
    """An old `.gnovi` manifest with no `panel_aspect_preset` key at all
    (pre-existing project) must load cleanly with Panel Aspect Ratio
    defaulting to "Auto", not raise."""
    project, _dataset = _basic_project()
    out_path = save_project(project, tmp_path / "old_format.gnovi")

    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    del manifest["figures"][0]["panel_aspect_preset"]
    stripped_path = tmp_path / "old_format_stripped.gnovi"
    with zipfile.ZipFile(out_path) as zf_in, zipfile.ZipFile(stripped_path, "w") as zf_out:
        for name in zf_in.namelist():
            if name == "project.json":
                zf_out.writestr(name, json.dumps(manifest))
            else:
                zf_out.writestr(name, zf_in.read(name))

    loaded = load_project(stripped_path)

    assert loaded.figures[0].panel_aspect_preset == "Auto"


# --- project_format_version ---------------------------------------------------


def test_saved_manifest_contains_the_current_format_version(tmp_path):
    project, _dataset = _basic_project()
    out_path = save_project(project, tmp_path / "proj.gnovi")

    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    assert manifest["project_format_version"] == PROJECT_FORMAT_VERSION == 1
    # Independent of the app's own version string.
    assert "app_version" in manifest


# --- Malformed files -----------------------------------------------------------


def test_malformed_zip_is_rejected(tmp_path):
    bad = tmp_path / "bad.gnovi"
    bad.write_bytes(b"this is not a zip file at all")
    with pytest.raises(CorruptProjectError):
        load_project(bad)


def test_zip_missing_manifest_is_rejected(tmp_path):
    path = tmp_path / "no_manifest.gnovi"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("datasets/x/raw.csv", "x\n1\n")
    with pytest.raises(CorruptProjectError):
        load_project(path)


def test_manifest_with_malformed_json_is_rejected(tmp_path):
    path = tmp_path / "bad_json.gnovi"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("project.json", "{not valid json,,,")
    with pytest.raises(CorruptProjectError):
        load_project(path)


def test_manifest_missing_format_version_is_rejected(tmp_path):
    path = tmp_path / "no_version.gnovi"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("project.json", json.dumps({"datasets": [], "figures": []}))
    with pytest.raises(CorruptProjectError):
        load_project(path)


def test_unsupported_future_format_version_is_rejected(tmp_path):
    path = tmp_path / "future.gnovi"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "project.json",
            json.dumps(
                {
                    "project_format_version": PROJECT_FORMAT_VERSION + 1,
                    "datasets": [],
                    "graph_library": [],
                    "figures": [],
                }
            ),
        )
    with pytest.raises(UnsupportedProjectVersionError):
        load_project(path)


def test_manifest_referencing_a_missing_dataset_csv_is_rejected(tmp_path):
    project, dataset = _basic_project()
    out_path = save_project(project, tmp_path / "proj.gnovi")

    # Rewrite the zip with the dataset's raw.csv removed but the manifest
    # still referencing it.
    with zipfile.ZipFile(out_path) as zf:
        manifest_bytes = zf.read("project.json")
        working_csv = zf.read(f"datasets/{dataset.id}/working.csv")
    stripped_path = tmp_path / "stripped.gnovi"
    with zipfile.ZipFile(stripped_path, "w") as zf:
        zf.writestr("project.json", manifest_bytes)
        zf.writestr(f"datasets/{dataset.id}/working.csv", working_csv)
        # raw.csv deliberately omitted

    with pytest.raises(CorruptProjectError):
        load_project(stripped_path)


def test_a_failed_load_never_mutates_or_returns_a_partial_project(tmp_path):
    path = tmp_path / "bad.gnovi"
    path.write_bytes(b"not a zip")
    try:
        load_project(path)
    except ProjectIOError:
        pass
    # Nothing to assert on the caller's live state here beyond "it raised
    # cleanly" -- `load_project` never returns in the failure path, so
    # there is no partially-built Project for a caller to accidentally use.
    # (See tests/test_main_window_project.py for the MainWindow-level
    # guarantee that its *own* current project is untouched on failure.)


# --- No pickle / executable payload --------------------------------------------


def test_project_io_module_never_uses_pickle():
    source = inspect.getsource(project_io)
    assert "import pickle" not in source
    assert "pickle.load" not in source
    assert "pickle.dump" not in source


def test_saved_manifest_is_plain_json_not_a_pickle_stream(tmp_path):
    project, _dataset = _basic_project()
    out_path = save_project(project, tmp_path / "proj.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        # json.loads succeeding on the raw bytes (no pickle.loads anywhere
        # in the read path) is itself the guarantee: pickle streams are not
        # valid JSON text.
        manifest = json.loads(zf.read("project.json").decode("utf-8"))
        assert isinstance(manifest, dict)
        names = zf.namelist()
    # Every member is either the manifest or a plain-text CSV -- nothing
    # else (no .pkl, no compiled/executable payload).
    assert all(name == "project.json" or name.endswith(".csv") for name in names)


# --- Portability: project survives the original source file moving/gone -------


def test_project_reopens_after_the_original_source_csv_is_deleted(tmp_path):
    src_csv = tmp_path / "source.csv"
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    df.to_csv(src_csv, index=False)

    dataset = Dataset(name="sample", dataframe=pd.read_csv(src_csv), source_path=str(src_csv))
    project = Project.new()
    project.dataset_manager.add(dataset)
    out_path = save_project(project, tmp_path / "proj.gnovi")

    src_csv.unlink()
    assert not src_csv.exists()

    loaded = load_project(out_path)
    loaded_dataset = loaded.dataset_manager.datasets[0]
    pdt.assert_frame_equal(loaded_dataset.dataframe, df)
    assert loaded_dataset.source_path == str(src_csv)  # provenance only, not re-read


def test_project_reopens_after_the_original_source_csv_moves(tmp_path):
    src_dir = tmp_path / "originals"
    src_dir.mkdir()
    src_csv = src_dir / "source.csv"
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    df.to_csv(src_csv, index=False)

    dataset = Dataset(name="sample", dataframe=pd.read_csv(src_csv), source_path=str(src_csv))
    project = Project.new()
    project.dataset_manager.add(dataset)
    out_path = save_project(project, tmp_path / "proj.gnovi")

    moved_csv = tmp_path / "moved.csv"
    src_csv.rename(moved_csv)

    loaded = load_project(out_path)
    pdt.assert_frame_equal(loaded.dataset_manager.datasets[0].dataframe, df)


def test_embedded_working_data_is_independent_of_the_raw_dataframe_after_reopen(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    dataset = Dataset(name="sample", dataframe=df)
    dataset.exclude_rows([0])
    project = Project.new()
    project.dataset_manager.add(dataset)
    out_path = save_project(project, tmp_path / "proj.gnovi")

    loaded = load_project(out_path)
    loaded_dataset = loaded.dataset_manager.datasets[0]

    assert loaded_dataset.raw_row_count == 4
    assert loaded_dataset.row_count == 3
    pdt.assert_frame_equal(loaded_dataset.raw_dataframe.reset_index(drop=True), df.reset_index(drop=True))


# --- The big representative round-trip integration test -----------------------


def _cv_dataset(name):
    leg = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0]
    x = leg + leg[1:] + leg[1:]
    y = [float(i) for i in range(len(x))]
    df = pd.DataFrame({"Potential/V": x, "Current/A": y})
    return Dataset(name=name, dataframe=df)


def _build_representative_project():
    """A project exercising every category called out for the round-trip
    integration test: 2 datasets (one with a calculated column, a Working
    Data transformation, and multi-cycle row ranges), >=10 saved Graphs, a
    2x2 Figure with 4 of those Graphs loaded into its panels, a
    selected-row series, cycle-row-range series, mixed colors/styles, and a
    custom figure ratio/margins/spacing/grid/legend."""
    ds1 = _cv_dataset("cv-sample")
    ds1.add_calculated_column("Power/W", "[Potential/V] * [Current/A]")
    ds1.exclude_rows([0])  # Working Data transformation

    ds2 = _simple_dataset("aux-sample")

    project = Project.new()
    project.dataset_manager.add(ds1)
    project.dataset_manager.add(ds2)

    # 10 saved Graphs, each a small independent panel referencing the live
    # datasets, with distinct styling so post-load identity is checkable.
    scratch_figure = GnoviFigure()
    for i in range(1, 11):
        scratch_figure.active_panel.title = f"Graph {i}"
        scratch_figure.active_panel.clear_series()
        color = f"#{i:02x}{i:02x}{i:02x}"
        scratch_figure.add_series(
            PlotSeries.line(ds1, "Potential/V", "Current/A", label=f"g{i}", color=color)
        )
        project.graph_library.save_panel_as_graph(scratch_figure, f"Graph {i}", project.dataset_manager)

    graphs_by_name = {g.name: g for g in project.graph_library.graphs}
    assert len(graphs_by_name) == 10

    # The current Figure: 2x2, custom ratio/margins/spacing/grid/legend.
    figure = project.figures[0]
    figure.figure_width_in = 9.0
    figure.figure_height_in = 6.0
    figure.margin_left = 0.18
    figure.margin_right = 0.82
    figure.panel_wspace = 0.4
    figure.panel_hspace = 0.4
    figure.set_layout(2, 2)

    # Load Graphs 2, 5, 7, 10 into the four panels.
    for panel_index, graph_number in enumerate([2, 5, 7, 10]):
        graph = graphs_by_name[f"Graph {graph_number}"]
        figure.set_active_panel(panel_index)
        loaded = project.graph_library.load_graph_into_panel(graph.id, figure, project.dataset_manager)
        assert loaded

    for panel in figure.panels:
        panel.grid = True
        panel.grid_which = "both"
        panel.legend_visible = True
        panel.legend_loc = "lower left"

    # A selected-row series (contiguous row_range, e.g. from a Data Preview
    # selection) added directly to panel 0, alongside its loaded Graph 2.
    figure.set_active_panel(0)
    figure.add_series(
        PlotSeries.line(
            ds2, "x", "y", label="selection", row_range=(1, 3), color="#ff8800", line_style=":"
        )
    )

    # Cycle row ranges on panel 1, alongside its loaded Graph 5.
    cycles = detect_cycles(ds1.dataframe, "Potential/V")
    assert len(cycles) == 3
    figure.set_active_panel(1)
    for i, row_range in enumerate(cycles):
        figure.add_series(
            PlotSeries.line(
                ds1,
                "Potential/V",
                "Current/A",
                label=f"cycle {i + 1}",
                row_range=row_range,
                marker="o" if i % 2 else "",
            )
        )

    return project, ds1, ds2, graphs_by_name


def test_representative_project_round_trip(tmp_path):
    project, ds1, ds2, graphs_by_name = _build_representative_project()
    figure = project.figures[0]

    out_path = save_project(project, tmp_path / "big.gnovi")

    # "Destroy/recreate model": drop every Python reference to the
    # in-memory project before reloading, so the loaded project can't be
    # accidentally sharing objects with the one that was saved.
    del project, figure
    import gc

    gc.collect()

    loaded = load_project(out_path)

    # --- datasets: count, stable ids, raw/working/calculated/history -------
    assert len(loaded.dataset_manager.datasets) == 2
    loaded_ds1 = loaded.dataset_manager.get(ds1.id)
    loaded_ds2 = loaded.dataset_manager.get(ds2.id)
    assert loaded_ds1 is not None and loaded_ds2 is not None
    pdt.assert_frame_equal(loaded_ds1.dataframe.reset_index(drop=True), ds1.dataframe.reset_index(drop=True))
    pdt.assert_frame_equal(
        loaded_ds1.raw_dataframe.reset_index(drop=True), ds1.raw_dataframe.reset_index(drop=True)
    )
    assert loaded_ds1.row_count == ds1.row_count
    assert loaded_ds1.raw_row_count == ds1.raw_row_count
    assert set(loaded_ds1.calculated_columns.keys()) == {"Power/W"}
    assert loaded_ds1.calculated_columns["Power/W"].formula == "[Potential/V] * [Current/A]"
    assert [t.kind for t in loaded_ds1.transformations] == ["calculated_column", "exclude_rows"]

    # --- graph library: 10 graphs, names, and dataset-reference resolution -
    assert len(loaded.graph_library.graphs) == 10
    loaded_names = {g.name for g in loaded.graph_library.graphs}
    assert loaded_names == {f"Graph {i}" for i in range(1, 11)}
    for g in loaded.graph_library.graphs:
        for series in g.panel.series:
            assert series.dataset is loaded_ds1  # shared reference, not a duplicate

    # --- figure: layout, ratio, margins/spacing, grid/legend ---------------
    loaded_figure = loaded.figures[0]
    assert loaded_figure.layout == (2, 2)
    assert len(loaded_figure.panels) == 4
    assert loaded_figure.figure_width_in == 9.0
    assert loaded_figure.figure_height_in == 6.0
    assert loaded_figure.margin_left == 0.18
    assert loaded_figure.margin_right == 0.82
    assert loaded_figure.panel_wspace == 0.4
    assert loaded_figure.panel_hspace == 0.4
    for panel in loaded_figure.panels:
        assert panel.grid is True
        assert panel.grid_which == "both"
        assert panel.legend_visible is True
        assert panel.legend_loc == "lower left"

    # --- panel content: loaded Graphs 2/5/7/10 landed in the right panels --
    panel0_titles = {s.label for s in loaded_figure.panels[0].series}
    assert "g2" in panel0_titles
    assert "selection" in panel0_titles  # selected-row series alongside it
    selection_series = next(s for s in loaded_figure.panels[0].series if s.label == "selection")
    assert selection_series.row_range == (1, 3)
    assert selection_series.dataset is loaded_ds2
    assert selection_series.color == "#ff8800"
    assert selection_series.line_style == ":"

    panel1_labels = {s.label for s in loaded_figure.panels[1].series}
    assert "g5" in panel1_labels
    cycle_series = [s for s in loaded_figure.panels[1].series if s.label.startswith("cycle")]
    assert len(cycle_series) == 3
    assert {s.row_range for s in cycle_series} == set(detect_cycles(ds1.dataframe, "Potential/V"))
    assert all(s.dataset is loaded_ds1 for s in cycle_series)

    assert "g7" in {s.label for s in loaded_figure.panels[2].series}
    assert "g10" in {s.label for s in loaded_figure.panels[3].series}

    # --- shared Dataset reference behavior: every series across every
    # panel/graph that references ds1 points at the exact same reloaded
    # Dataset object, never a duplicate. ---------------------------------
    all_ds1_series = [
        s
        for panel in loaded_figure.panels
        for s in panel.series
        if s.dataset.id == ds1.id
    ] + [s for g in loaded.graph_library.graphs for s in g.panel.series if s.dataset.id == ds1.id]
    assert all(s.dataset is loaded_ds1 for s in all_ds1_series)
    assert len(loaded.dataset_manager.datasets) == 2  # never silently duplicated
