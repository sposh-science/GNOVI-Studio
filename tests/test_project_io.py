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
    _migrate_v1_to_v2,
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
    project.workbenches[0].figure.add_series(PlotSeries.line(dataset, "x", "y"))
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
    assert len(loaded.workbenches) == 1
    assert loaded.workbenches[0].name == "Workbench 1"
    assert len(loaded.workbenches[0].figure.series) == 1
    assert loaded.workbenches[0].figure.series[0].dataset is loaded_dataset


def test_save_project_as_gives_the_project_a_path(tmp_path):
    project, _dataset = _basic_project()
    assert project.path is None
    save_project(project, tmp_path / "proj.gnovi")
    assert project.path == tmp_path / "proj.gnovi"


# --- Plot Theme: a project must reopen looking exactly as it was saved --------


def test_light_theme_project_round_trips_and_reopens_light(tmp_path):
    project, _dataset = _basic_project()
    project.workbenches[0].figure.plot_theme = PlotTheme.LIGHT

    out_path = save_project(project, tmp_path / "light.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    assert manifest["workbenches"][0]["figure"]["plot_theme"] == "light"

    loaded = load_project(out_path)
    assert loaded.workbenches[0].figure.plot_theme == PlotTheme.LIGHT


def test_dark_theme_project_round_trips_and_reopens_dark(tmp_path):
    project, _dataset = _basic_project()
    project.workbenches[0].figure.plot_theme = PlotTheme.DARK

    out_path = save_project(project, tmp_path / "dark.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    assert manifest["workbenches"][0]["figure"]["plot_theme"] == "dark"

    loaded = load_project(out_path)
    assert loaded.workbenches[0].figure.plot_theme == PlotTheme.DARK


def test_saved_theme_is_restored_after_reopen_even_when_it_differs_from_the_default(tmp_path):
    """A project saved with a non-default (Dark) theme must come back Dark
    on reopen -- the point of this whole correction: a project/figure must
    reopen looking exactly as it was saved, independent of whatever the
    QSettings default-for-new-figures happens to be at open time."""
    project, dataset = _basic_project()
    assert project.workbenches[0].figure.plot_theme == PlotTheme.LIGHT  # the ordinary default
    project.workbenches[0].figure.plot_theme = PlotTheme.DARK
    out_path = save_project(project, tmp_path / "explicit_dark.gnovi")

    loaded = load_project(out_path)

    assert loaded.workbenches[0].figure.plot_theme == PlotTheme.DARK
    # Nothing else about the round trip regressed alongside the theme fix.
    assert loaded.dataset_manager.datasets[0].id == dataset.id
    assert len(loaded.workbenches[0].figure.series) == 1


# --- Figure Aspect Ratio vs. Panel Aspect Ratio: independent round trip -------


def test_figure_and_panel_aspect_ratio_round_trip_independently(tmp_path):
    """The exact scenario from the task: Figure aspect = 16:9, Panel aspect
    = 1:1, Layout = 2x3 -- save, reopen, all three must be restored exactly,
    never inferred from the current canvas."""
    project, _dataset = _basic_project()
    figure = project.workbenches[0].figure
    figure.set_layout(2, 3)
    figure.figure_width_in = 16.0
    figure.figure_height_in = 9.0
    figure.aspect_preset = "16:9"
    figure.lock_aspect_ratio = True
    figure.panel_aspect_preset = "1:1"
    out_path = save_project(project, tmp_path / "figure_vs_panel_aspect.gnovi")

    loaded = load_project(out_path)
    loaded_figure = loaded.workbenches[0].figure

    assert loaded_figure.layout == (2, 3)
    assert loaded_figure.aspect_preset == "16:9"
    assert loaded_figure.lock_aspect_ratio is True
    assert loaded_figure.panel_aspect_preset == "1:1"


def test_panel_aspect_ratio_persists_independently_of_figure_aspect_ratio(tmp_path):
    """Changing one must never affect the other, including across a save/
    reopen round trip."""
    project, _dataset = _basic_project()
    figure = project.workbenches[0].figure
    figure.aspect_preset = "4:3"
    figure.lock_aspect_ratio = True
    figure.panel_aspect_preset = "16:9"
    out_path = save_project(project, tmp_path / "independent_aspects.gnovi")

    loaded = load_project(out_path)

    assert loaded.workbenches[0].figure.aspect_preset == "4:3"
    assert loaded.workbenches[0].figure.panel_aspect_preset == "16:9"


def test_panel_aspect_ratio_defaults_to_auto_for_a_project_saved_before_the_feature_existed(tmp_path):
    """An old `.gnovi` manifest with no `panel_aspect_preset` key at all
    (pre-existing project) must load cleanly with Panel Aspect Ratio
    defaulting to "Auto", not raise."""
    project, _dataset = _basic_project()
    out_path = save_project(project, tmp_path / "old_format.gnovi")

    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    del manifest["workbenches"][0]["figure"]["panel_aspect_preset"]
    stripped_path = tmp_path / "old_format_stripped.gnovi"
    with zipfile.ZipFile(out_path) as zf_in, zipfile.ZipFile(stripped_path, "w") as zf_out:
        for name in zf_in.namelist():
            if name == "project.json":
                zf_out.writestr(name, json.dumps(manifest))
            else:
                zf_out.writestr(name, zf_in.read(name))

    loaded = load_project(stripped_path)

    assert loaded.workbenches[0].figure.panel_aspect_preset == "Auto"


# --- project_format_version ---------------------------------------------------


def test_saved_manifest_contains_the_current_format_version(tmp_path):
    project, _dataset = _basic_project()
    out_path = save_project(project, tmp_path / "proj.gnovi")

    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    assert manifest["project_format_version"] == PROJECT_FORMAT_VERSION == 2
    # Independent of the app's own version string.
    assert "app_version" in manifest


# --- Fit-derived Dataset metadata (PR 5: descriptive provenance snapshot + --
# --- RMSE/RSS/n_points, on top of PR 4's model/params/r_squared) -----------


def test_fit_derived_dataset_round_trips_with_no_format_version_bump(tmp_path):
    """A derived fit Dataset's metadata is ordinary free-form
    Dataset.metadata -- round-trips through save/load exactly like any
    other metadata, with no project_format_version change."""
    from gnovi_plot.analysis.fitting import LINEAR, fit_curve, sample_fit_curve

    source = _simple_dataset(name="source")
    x = source.dataframe["x"].to_numpy(dtype=float)
    y = source.dataframe["y"].to_numpy(dtype=float)
    result = fit_curve(
        x,
        y,
        LINEAR,
        source_dataset_id=source.id,
        source_dataset_name=source.name,
        x_column="x",
        y_column="y",
    )
    x_smooth, y_smooth = sample_fit_curve(result, float(x.min()), float(x.max()), num_points=20)
    metadata = result.to_dict()
    metadata["x_min"] = float(x.min())
    metadata["x_max"] = float(x.max())
    metadata["num_points"] = len(x_smooth)
    fit_dataset = Dataset(
        name="Fit: linear",
        dataframe=pd.DataFrame({"x": x_smooth, "y": y_smooth}),
        metadata=metadata,
    )

    project = Project.new()
    project.dataset_manager.add(source)
    project.dataset_manager.add(fit_dataset)
    project.workbenches[0].figure.add_series(PlotSeries.line(source, "x", "y"))
    project.workbenches[0].figure.add_series(PlotSeries.line(fit_dataset, "x", "y"))

    out_path = save_project(project, tmp_path / "proj.gnovi")

    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    assert manifest["project_format_version"] == PROJECT_FORMAT_VERSION == 2

    reloaded = load_project(out_path)
    reloaded_fit = next(d for d in reloaded.dataset_manager.datasets if d.metadata.get("kind") == "fit")

    assert reloaded_fit.metadata["source_dataset_name"] == "source"
    assert reloaded_fit.metadata["rmse"] == pytest.approx(metadata["rmse"])
    assert reloaded_fit.metadata["residual_sum_of_squares"] == pytest.approx(metadata["residual_sum_of_squares"])
    assert reloaded_fit.metadata["n_points"] == metadata["n_points"]
    pdt.assert_frame_equal(
        reloaded_fit.dataframe.reset_index(drop=True), fit_dataset.dataframe.reset_index(drop=True)
    )


def test_old_style_fit_metadata_without_new_fields_still_loads(tmp_path):
    """A `.gnovi` saved before PR 5 has fit metadata missing
    source_dataset_name/source_series_label/rmse/residual_sum_of_squares/
    n_points entirely -- the free-form metadata dict simply lacks those
    keys. Loading must not choke on their absence."""
    old_style_metadata = {
        "kind": "fit",
        "source_dataset_id": "some-id",
        "source_series_id": None,
        "x_column": "x",
        "y_column": "y",
        "row_range": None,
        "model": "linear",
        "params": {"a": 2.0, "b": 1.0},
        "param_errors": None,
        "r_squared": 0.99,
        "formula": "y = a·x + b",
        "x_min": 0.0,
        "x_max": 10.0,
        "num_points": 20,
        # No source_dataset_name/source_series_label/rmse/
        # residual_sum_of_squares/n_points -- pre-PR5 shape.
    }
    fit_dataset = Dataset(
        name="Fit: linear",
        dataframe=pd.DataFrame({"x": [0.0, 5.0, 10.0], "y": [1.0, 11.0, 21.0]}),
        metadata=old_style_metadata,
    )
    project = Project.new()
    project.dataset_manager.add(fit_dataset)

    out_path = save_project(project, tmp_path / "proj.gnovi")
    reloaded = load_project(out_path)

    reloaded_fit = reloaded.dataset_manager.datasets[0]
    assert reloaded_fit.metadata["kind"] == "fit"
    assert reloaded_fit.metadata.get("rmse") is None  # simply absent, no crash
    assert reloaded_fit.metadata.get("source_dataset_name") is None


# --- v1 -> v2 migration: figures/active_figure_index -> workbenches -----------


def _rewrite_manifest(zip_path, tmp_path, new_manifest: dict, *, out_name: str = "rewritten.gnovi"):
    """Copy `zip_path` to a new file with its `project.json` replaced by
    `new_manifest`, keeping every other member (dataset CSVs) byte-for-byte
    -- how every "hand a load_project a manifest I control" test in this
    file builds its fixture, from the malformed-file tests through the
    migration tests below."""
    out_path = tmp_path / out_name
    with zipfile.ZipFile(zip_path) as zf_in, zipfile.ZipFile(out_path, "w") as zf_out:
        for name in zf_in.namelist():
            if name == "project.json":
                zf_out.writestr(name, json.dumps(new_manifest))
            else:
                zf_out.writestr(name, zf_in.read(name))
    return out_path


def test_migrate_v1_to_v2_is_a_pure_dict_transform():
    """Exercises `_migrate_v1_to_v2` directly against a literal, hand-written
    v1-shaped manifest -- independent of whatever the current `save_project`
    happens to produce, so this keeps testing the real migration a genuinely
    old `.gnovi` file goes through even if `_build_manifest`'s own shape
    drifts further in the future."""
    figure_a = GnoviFigure(name="A").to_dict()
    figure_b = GnoviFigure(name="B").to_dict()
    v1_manifest = {
        "project_format_version": 1,
        "app_version": "0.1.0",
        "project_name": "Old Project",
        "datasets": [],
        "graph_library": [],
        "figures": [figure_a, figure_b],
        "active_figure_index": 1,
        "results": [],
    }

    migrated = _migrate_v1_to_v2(v1_manifest)

    assert migrated["project_format_version"] == 2
    assert len(migrated["workbenches"]) == 2
    assert [w["name"] for w in migrated["workbenches"]] == ["Workbench 1", "Workbench 2"]
    assert migrated["workbenches"][0]["figure"] == figure_a
    assert migrated["workbenches"][1]["figure"] == figure_b
    assert migrated["active_workbench_id"] == migrated["workbenches"][1]["id"]
    assert migrated["workbenches"][0]["id"] != migrated["workbenches"][1]["id"]


def test_migrate_v1_to_v2_with_zero_figures_produces_one_default_workbench():
    v1_manifest = {
        "project_format_version": 1,
        "datasets": [],
        "graph_library": [],
        "figures": [],
        "active_figure_index": 0,
    }

    migrated = _migrate_v1_to_v2(v1_manifest)

    assert len(migrated["workbenches"]) == 1
    assert migrated["active_workbench_id"] == migrated["workbenches"][0]["id"]


def test_migrate_v1_to_v2_clamps_an_out_of_range_active_figure_index():
    v1_manifest = {
        "project_format_version": 1,
        "datasets": [],
        "graph_library": [],
        "figures": [GnoviFigure().to_dict()],
        "active_figure_index": 99,
    }

    migrated = _migrate_v1_to_v2(v1_manifest)

    assert migrated["active_workbench_id"] == migrated["workbenches"][0]["id"]


def test_opening_a_v1_project_migrates_to_one_named_workbench_preserving_content(tmp_path):
    """The end-to-end path: a real v1 `.gnovi` file (with an embedded
    dataset and a series) opens as a Workbench, all scientific content
    intact -- not just the pure dict transform in isolation above."""
    project, dataset = _basic_project()
    out_path = save_project(project, tmp_path / "v2_source.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        v2_manifest = json.loads(zf.read("project.json"))

    v1_manifest = {
        "project_format_version": 1,
        "app_version": v2_manifest["app_version"],
        "project_name": v2_manifest["project_name"],
        "saved_at": v2_manifest["saved_at"],
        "datasets": v2_manifest["datasets"],
        "graph_library": v2_manifest["graph_library"],
        "figures": [v2_manifest["workbenches"][0]["figure"]],
        "active_figure_index": 0,
        "results": [],
    }
    v1_path = _rewrite_manifest(out_path, tmp_path, v1_manifest, out_name="v1_real.gnovi")

    loaded = load_project(v1_path)

    assert len(loaded.workbenches) == 1
    assert loaded.workbenches[0].name == "Workbench 1"
    assert loaded.active_workbench_id == loaded.workbenches[0].id
    assert len(loaded.workbenches[0].figure.series) == 1
    loaded_series = loaded.workbenches[0].figure.series[0]
    assert loaded_series.dataset.id == dataset.id
    assert loaded_series.dataset.name == dataset.name
    assert len(loaded.dataset_manager.datasets) == 1


def test_opening_a_v1_project_with_multiple_figures_migrates_to_multiple_named_workbenches(tmp_path):
    project = Project.new()
    project.workbenches[0].figure.active_panel.title = "First"
    from gnovi_plot.core.workbench import Workbench

    project.add_workbench(Workbench(name="ignored-by-v1", figure=GnoviFigure()))
    project.workbenches[1].figure.active_panel.title = "Second"
    project.workbenches[1].figure.set_layout(1, 2)
    out_path = save_project(project, tmp_path / "v2_source.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        v2_manifest = json.loads(zf.read("project.json"))

    v1_manifest = {
        "project_format_version": 1,
        "app_version": v2_manifest["app_version"],
        "project_name": v2_manifest["project_name"],
        "datasets": [],
        "graph_library": [],
        "figures": [w["figure"] for w in v2_manifest["workbenches"]],
        "active_figure_index": 1,
        "results": [],
    }
    v1_path = _rewrite_manifest(out_path, tmp_path, v1_manifest, out_name="v1_multi.gnovi")

    loaded = load_project(v1_path)

    assert len(loaded.workbenches) == 2
    assert [w.name for w in loaded.workbenches] == ["Workbench 1", "Workbench 2"]
    assert loaded.workbenches[0].figure.active_panel.title == "First"
    assert loaded.workbenches[1].figure.active_panel.title == "Second"
    assert loaded.workbenches[1].figure.layout == (1, 2)
    # active_figure_index (1) migrated to point at the second Workbench.
    assert loaded.active_workbench_id == loaded.workbenches[1].id


def test_reopening_a_migrated_v1_project_and_resaving_writes_v2(tmp_path):
    project, dataset = _basic_project()
    out_path = save_project(project, tmp_path / "v2_source.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        v2_manifest = json.loads(zf.read("project.json"))
    v1_manifest = {
        "project_format_version": 1,
        "app_version": v2_manifest["app_version"],
        "project_name": v2_manifest["project_name"],
        "datasets": v2_manifest["datasets"],
        "graph_library": v2_manifest["graph_library"],
        "figures": [v2_manifest["workbenches"][0]["figure"]],
        "active_figure_index": 0,
        "results": [],
    }
    v1_path = _rewrite_manifest(out_path, tmp_path, v1_manifest, out_name="v1.gnovi")

    loaded = load_project(v1_path)
    resave_path = tmp_path / "resaved.gnovi"
    save_project(loaded, resave_path)

    with zipfile.ZipFile(resave_path) as zf:
        resaved_manifest = json.loads(zf.read("project.json"))
    assert resaved_manifest["project_format_version"] == 2
    assert "workbenches" in resaved_manifest
    assert "figures" not in resaved_manifest

    reloaded = load_project(resave_path)
    assert len(reloaded.workbenches) == 1
    assert len(reloaded.workbenches[0].figure.series) == 1


# --- v2 manifests missing/misreferencing workbenches -- fall back cleanly -----


def test_v2_manifest_with_no_workbenches_key_falls_back_to_one_default_workbench(tmp_path):
    project, _dataset = _basic_project()
    out_path = save_project(project, tmp_path / "proj.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    del manifest["workbenches"]
    del manifest["active_workbench_id"]
    stripped_path = _rewrite_manifest(out_path, tmp_path, manifest, out_name="no_workbenches.gnovi")

    loaded = load_project(stripped_path)

    assert len(loaded.workbenches) == 1
    assert loaded.active_workbench_id == loaded.workbenches[0].id


def test_active_workbench_id_not_matching_any_workbench_falls_back_to_first(tmp_path):
    project, _dataset = _basic_project()
    out_path = save_project(project, tmp_path / "proj.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    manifest["active_workbench_id"] = "does-not-exist"
    stripped_path = _rewrite_manifest(out_path, tmp_path, manifest, out_name="bad_active_id.gnovi")

    loaded = load_project(stripped_path)

    assert loaded.active_workbench_id == loaded.workbenches[0].id


# --- Panel.id: stable identity through save/load -------------------------------


def test_panel_id_round_trips_through_save_and_load(tmp_path):
    project, _dataset = _basic_project()
    original_id = project.workbenches[0].figure.active_panel.id

    out_path = save_project(project, tmp_path / "proj.gnovi")
    loaded = load_project(out_path)

    assert loaded.workbenches[0].figure.active_panel.id == original_id


def test_loading_a_project_with_no_panel_id_generates_one(tmp_path):
    """A `.gnovi` saved before `Panel.id` existed has no `"id"` key in its
    panel dicts at all -- loading it must not crash, and must not leave
    the reconstructed panel with a falsy/missing id. No
    `PROJECT_FORMAT_VERSION` bump needed for this -- see `Panel.from_dict`."""
    project, _dataset = _basic_project()
    out_path = save_project(project, tmp_path / "proj.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    del manifest["workbenches"][0]["figure"]["panels"][0]["id"]
    stripped_path = _rewrite_manifest(out_path, tmp_path, manifest, out_name="no_panel_id.gnovi")

    loaded = load_project(stripped_path)

    panel = loaded.workbenches[0].figure.active_panel
    assert panel.id
    assert isinstance(panel.id, str)
    # The rest of that panel's content loaded fine alongside the missing id.
    assert len(loaded.workbenches[0].figure.series) == 1


def test_panels_missing_ids_in_the_same_project_each_get_a_different_generated_id(tmp_path):
    project, _dataset = _basic_project()
    project.workbenches[0].figure.set_layout(1, 2)
    out_path = save_project(project, tmp_path / "proj.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    for panel_data in manifest["workbenches"][0]["figure"]["panels"]:
        del panel_data["id"]
    stripped_path = _rewrite_manifest(out_path, tmp_path, manifest, out_name="no_panel_ids.gnovi")

    loaded = load_project(stripped_path)

    ids = [p.id for p in loaded.workbenches[0].figure.panels]
    assert len(ids) == 2
    assert len(set(ids)) == 2  # no collision between the two generated ids


def test_no_duplicate_panel_ids_across_a_multi_panel_multi_workbench_project(tmp_path):
    project, _dataset = _basic_project()
    project.workbenches[0].figure.set_layout(2, 2)
    from gnovi_plot.core.workbench import Workbench

    second_figure = GnoviFigure()
    second_figure.set_layout(1, 2)
    project.add_workbench(Workbench(name="Second", figure=second_figure))

    out_path = save_project(project, tmp_path / "proj.gnovi")
    loaded = load_project(out_path)

    all_ids = [p.id for w in loaded.workbenches for p in w.figure.panels]
    assert len(all_ids) == 6
    assert len(set(all_ids)) == 6


def test_existing_gnovi_file_saved_before_panel_id_still_loads_end_to_end(tmp_path):
    """A realistic old-style file: format version 2, panels with no `id`
    key at all (as every `.gnovi` saved before this change would be) --
    the whole project must still load with all content intact, not just
    the one touched panel."""
    project, dataset = _basic_project()
    project.workbenches[0].figure.active_panel.title = "Legacy Panel"
    out_path = save_project(project, tmp_path / "proj.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    assert manifest["project_format_version"] == PROJECT_FORMAT_VERSION
    del manifest["workbenches"][0]["figure"]["panels"][0]["id"]
    legacy_path = _rewrite_manifest(out_path, tmp_path, manifest, out_name="legacy.gnovi")

    loaded = load_project(legacy_path)

    assert loaded.workbenches[0].figure.active_panel.title == "Legacy Panel"
    assert loaded.workbenches[0].figure.active_panel.id
    assert loaded.dataset_manager.datasets[0].id == dataset.id
    assert len(loaded.workbenches[0].figure.series) == 1


# --- Analysis history: persistence through save/reopen -------------------------


def test_analysis_history_round_trips_across_save_and_reopen(tmp_path):
    """The exact required workflow: 3 panels, Linear fit on panel 1,
    Gaussian fit on panel 2, nothing on panel 3 -- save, reopen. Both
    fits must come back as real `FitResult` objects reconstructed from
    the persisted history (never re-derived from the fit-derived
    Dataset), with correct type/model/parameters/statistics/provenance;
    panel 3 stays empty; panel-switch-style lookup behaves identically to
    before saving; residuals recompute correctly from the restored
    FitResult plus the live (reloaded) source data."""
    from gnovi_plot.analysis.fitting import GAUSSIAN, LINEAR, FitResult, fit_curve
    from gnovi_plot.data.numeric import numeric_xy

    dataset_a = Dataset(
        name="linear-data",
        dataframe=pd.DataFrame({"x": list(range(10)), "y": [2.0 * i + 1.0 for i in range(10)]}),
    )
    dataset_b = Dataset(
        name="gaussian-data",
        dataframe=pd.DataFrame(
            {
                "x": [float(i) for i in range(20)],
                "y": [10.0 * 2.71828 ** (-((i - 10.0) ** 2) / (2 * 3.0**2)) for i in range(20)],
            }
        ),
    )
    project = Project.new()
    project.dataset_manager.add(dataset_a)
    project.dataset_manager.add(dataset_b)
    workbench = project.workbenches[0]
    workbench.figure.set_layout(1, 3)
    panel_1_id = workbench.figure.panels[0].id
    panel_2_id = workbench.figure.panels[1].id
    panel_3_id = workbench.figure.panels[2].id

    workbench.figure.set_active_panel(0)
    series_a = PlotSeries.line(dataset_a, "x", "y")
    workbench.figure.add_series(series_a)
    x_a, y_a = numeric_xy(series_a.dataframe, "x", "y")
    result_a = fit_curve(
        x_a.to_numpy(),
        y_a.to_numpy(),
        LINEAR,
        source_dataset_id=dataset_a.id,
        source_dataset_name=dataset_a.name,
        source_series_id=series_a.id,
        source_series_label=series_a.label,
        x_column="x",
        y_column="y",
        source_panel_id=panel_1_id,
    )
    workbench.analysis_results.add(panel_1_id, result_a)

    workbench.figure.set_active_panel(1)
    series_b = PlotSeries.line(dataset_b, "x", "y")
    workbench.figure.add_series(series_b)
    x_b, y_b = numeric_xy(series_b.dataframe, "x", "y")
    result_b = fit_curve(
        x_b.to_numpy(),
        y_b.to_numpy(),
        GAUSSIAN,
        source_dataset_id=dataset_b.id,
        source_dataset_name=dataset_b.name,
        source_series_id=series_b.id,
        source_series_label=series_b.label,
        x_column="x",
        y_column="y",
        source_panel_id=panel_2_id,
    )
    workbench.analysis_results.add(panel_2_id, result_b)
    # Panel 3: deliberately no analysis.

    out_path = save_project(project, tmp_path / "proj.gnovi")
    loaded = load_project(out_path)

    loaded_workbench = loaded.workbenches[0]
    loaded_result_a = loaded_workbench.analysis_results.current(panel_1_id)
    loaded_result_b = loaded_workbench.analysis_results.current(panel_2_id)

    # Panel 1: correct type/model/parameters/statistics/provenance.
    assert isinstance(loaded_result_a, FitResult)
    assert loaded_result_a.model == LINEAR
    assert loaded_result_a.params == pytest.approx(result_a.params)
    assert loaded_result_a.param_errors == result_a.param_errors
    assert loaded_result_a.r_squared == pytest.approx(result_a.r_squared)
    assert loaded_result_a.adjusted_r_squared() == pytest.approx(result_a.adjusted_r_squared())
    assert loaded_result_a.rmse == pytest.approx(result_a.rmse)
    assert loaded_result_a.residual_sum_of_squares == pytest.approx(result_a.residual_sum_of_squares)
    assert loaded_result_a.n_points == result_a.n_points
    assert loaded_result_a.source_dataset_id == dataset_a.id
    assert loaded_result_a.source_dataset_name == "linear-data"
    assert loaded_result_a.source_series_id == series_a.id
    assert loaded_result_a.source_series_label == series_a.label
    assert loaded_result_a.source_panel_id == panel_1_id
    assert loaded_result_a.result_id == result_a.result_id

    # Panel 2: a different model, independently correct.
    assert isinstance(loaded_result_b, FitResult)
    assert loaded_result_b.model == GAUSSIAN
    assert loaded_result_b.result_id == result_b.result_id
    assert loaded_result_b.source_panel_id == panel_2_id

    # Panel 3: Results stays empty.
    assert loaded_workbench.analysis_results.current(panel_3_id) is None

    # Switching through all 3 panels behaves identically to before saving
    # (the same `current(panel_id)` lookup MainWindow's
    # `_sync_results_to_active_panel` uses for real panel-switch restore).
    for panel_id, expected_id in [
        (panel_1_id, result_a.result_id),
        (panel_2_id, result_b.result_id),
        (panel_3_id, None),
    ]:
        current = loaded_workbench.analysis_results.current(panel_id)
        assert (current.result_id if current is not None else None) == expected_id

    # Residuals recompute correctly from the restored FitResult + live
    # (reloaded) source data -- nothing about residuals was ever persisted.
    loaded_series_a = loaded_workbench.figure.get_series(series_a.id)
    x_live, y_live = numeric_xy(loaded_series_a.dataframe, "x", "y")
    restored_residuals = loaded_result_a.compute_residuals(x_live.to_numpy(), y_live.to_numpy())
    original_residuals = result_a.compute_residuals(x_a.to_numpy(), y_a.to_numpy())
    assert restored_residuals.residuals == pytest.approx(original_residuals.residuals)


def test_analysis_history_persistence_works_with_project_io_alone(tmp_path):
    """The polymorphic dispatch registry (see `analysis.results.
    register_result_kind`) must be populated by `project_io.py`'s own
    import alone -- proves `load_project` correctly reconstructs a
    persisted FitResult in a *fresh process* that never imports anything
    from the GUI layer (`gui.widgets.analysis_panel`, which also imports
    `analysis.fitting`, and could otherwise be silently doing the
    registration instead of `project_io.py`). A same-process assertion
    like "`gnovi_plot.gui...` not in `sys.modules`" would be meaningless
    here -- pytest's own collection has almost certainly already imported
    the GUI layer via other test files by the time this test runs."""
    import subprocess
    import sys

    dataset = _simple_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id
    from gnovi_plot.analysis.fitting import LINEAR, fit_curve

    result = fit_curve(
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        LINEAR,
        source_dataset_id=dataset.id,
        x_column="x",
        y_column="y",
        source_panel_id=panel_id,
    )
    workbench.analysis_results.add(panel_id, result)
    out_path = save_project(project, tmp_path / "proj.gnovi")

    script = f"""
import sys
assert "gnovi_plot.gui" not in sys.modules
from gnovi_plot.core.project_io import load_project
loaded = load_project(r{out_path.as_posix()!r})
restored = loaded.workbenches[0].analysis_results.current({panel_id!r})
assert restored is not None, "analysis result was not restored"
assert restored.result_id == {result.result_id!r}
assert type(restored).__name__ == "FitResult"
print("OK")
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_analysis_history_key_absent_in_a_pre_existing_gnovi_file_loads_empty(tmp_path):
    """An old `.gnovi` saved before analysis-history persistence existed
    has no `"analysis_results"` key in any workbench dict at all --
    loading it must not crash, and every Workbench simply gets empty
    history, same as New Project. No `PROJECT_FORMAT_VERSION` bump
    needed -- see `Workbench.to_dict()`'s own docstring."""
    project, _dataset = _basic_project()
    out_path = save_project(project, tmp_path / "proj.gnovi")
    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    assert manifest["project_format_version"] == PROJECT_FORMAT_VERSION
    del manifest["workbenches"][0]["analysis_results"]
    legacy_path = _rewrite_manifest(out_path, tmp_path, manifest, out_name="no_analysis_results.gnovi")

    loaded = load_project(legacy_path)

    panel_id = loaded.workbenches[0].figure.active_panel.id
    assert loaded.workbenches[0].analysis_results.current(panel_id) is None
    # The rest of that workbench's content loaded fine alongside the
    # missing key.
    assert len(loaded.workbenches[0].figure.series) == 1


# --- Multi-workbench v2 round trip ---------------------------------------------


def test_multiple_workbenches_round_trip_independently(tmp_path):
    dataset = _simple_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    project.workbenches[0].name = "CV Comparison"
    project.workbenches[0].figure.set_layout(2, 2)
    project.workbenches[0].figure.add_series(PlotSeries.line(dataset, "x", "y", label="a"))

    from gnovi_plot.core.workbench import Workbench

    second = Workbench(name="New Scan", figure=GnoviFigure())
    second.figure.add_series(PlotSeries.line(dataset, "x", "y", label="b"))
    project.add_workbench(second)
    project.active_workbench_id = second.id

    out_path = save_project(project, tmp_path / "multi.gnovi")
    loaded = load_project(out_path)

    assert len(loaded.workbenches) == 2
    names = {w.name for w in loaded.workbenches}
    assert names == {"CV Comparison", "New Scan"}
    loaded_first = next(w for w in loaded.workbenches if w.name == "CV Comparison")
    loaded_second = next(w for w in loaded.workbenches if w.name == "New Scan")
    assert loaded_first.figure.layout == (2, 2)
    assert loaded_first.figure.series[0].label == "a"
    assert loaded_second.figure.series[0].label == "b"
    assert loaded.active_workbench_id == loaded_second.id
    # Shared dataset, not duplicated.
    assert loaded_first.figure.series[0].dataset is loaded_second.figure.series[0].dataset
    assert len(loaded.dataset_manager.datasets) == 1


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
        zf.writestr("project.json", json.dumps({"datasets": [], "workbenches": []}))
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
                    "workbenches": [],
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
    figure = project.workbenches[0].figure
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
    figure = project.workbenches[0].figure
    expected_source_graph_ids = [graphs_by_name[f"Graph {n}"].id for n in (2, 5, 7, 10)]

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

    # --- workbenches: exactly one, named --------------------------------
    assert len(loaded.workbenches) == 1
    assert loaded.workbenches[0].name == "Workbench 1"
    assert loaded.active_workbench_id == loaded.workbenches[0].id

    # --- figure: layout, ratio, margins/spacing, grid/legend ---------------
    loaded_figure = loaded.workbenches[0].figure
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

    # --- Graph identity/provenance context survives the round trip ---------
    # Each panel's `source_graph_id` (see `Panel.source_graph_id`) must
    # still resolve to its correct origin Graph after reload, so
    # `ActivePanelLabel`/`GraphLibraryPanel.sync_active_panel_state` show
    # the right "Graph: ... (working copy)" text and Update Saved Graph
    # enablement without any further wiring.
    assert [p.source_graph_id for p in loaded_figure.panels] == expected_source_graph_ids
    for panel, graph_id in zip(loaded_figure.panels, expected_source_graph_ids):
        assert loaded.graph_library.get(panel.source_graph_id) is not None
        assert panel.source_graph_id == graph_id

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
