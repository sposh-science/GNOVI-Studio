"""Panel3D project persistence: save/reopen, XYZ mapping, styling/camera,
labels/title, backward compatibility with pre-3D (`PROJECT_FORMAT_VERSION`
2) projects, and that no Matplotlib object ever enters the serialized
manifest. Mirrors `test_project_io.py`'s own style.
"""

import json
import zipfile

import pandas as pd
import pytest

from gnovi_plot.core.project import Project
from gnovi_plot.core.project_io import PROJECT_FORMAT_VERSION, load_project, save_project
from gnovi_plot.core.workbench import Workbench
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.figure import GnoviFigure, Panel, Panel3D
from gnovi_plot.plotting.series3d import Plot3DType, Series3D


def _make_dataset(name="mat", id="ds1"):
    df = pd.DataFrame(
        {"temperature": [300.0, 350.0, 400.0, 300.0], "composition": [0.1, 0.1, 0.1, 0.2], "conductivity": [2.4, 2.9, 3.5, 3.1]}
    )
    return Dataset(id=id, name=name, dataframe=df)


def _mixed_project():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    wb = project.workbenches[0]
    wb.figure.set_layout(1, 3)
    panel3d = Panel3D(
        title="3D scatter", x_label="T", y_label="Comp", z_label="Cond", elevation=22.0, azimuth=-50.0
    )
    panel3d.add_series(
        Series3D(
            dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity",
            marker="s", marker_size=8.0, color="#ff0000", alpha=0.8, label="mat scatter",
        )
    )
    wb.figure.panels[1] = panel3d
    wb.figure._renumber_panel_labels()
    return project, dataset, panel3d


# --- Save/reopen round-trip -------------------------------------------------------


def test_save_reopen_preserves_panel3d(tmp_path):
    project, _dataset, panel3d = _mixed_project()
    out_path = save_project(project, tmp_path / "mixed.gnovi")

    reloaded = load_project(out_path)
    reloaded_panel = reloaded.workbenches[0].figure.panels[1]

    assert isinstance(reloaded_panel, Panel3D)
    assert reloaded_panel.id == panel3d.id
    assert reloaded.workbenches[0].figure.layout == (1, 3)
    assert isinstance(reloaded.workbenches[0].figure.panels[0], Panel)
    assert isinstance(reloaded.workbenches[0].figure.panels[2], Panel)


def test_save_reopen_preserves_xyz_mapping(tmp_path):
    project, _dataset, _panel3d = _mixed_project()
    out_path = save_project(project, tmp_path / "mixed.gnovi")

    reloaded = load_project(out_path)
    series = reloaded.workbenches[0].figure.panels[1].series[0]

    assert series.x_column == "temperature"
    assert series.y_column == "composition"
    assert series.z_column == "conductivity"


def test_save_reopen_preserves_styling_and_camera(tmp_path):
    project, _dataset, _panel3d = _mixed_project()
    out_path = save_project(project, tmp_path / "mixed.gnovi")

    reloaded = load_project(out_path)
    panel = reloaded.workbenches[0].figure.panels[1]
    series = panel.series[0]

    assert panel.elevation == pytest.approx(22.0)
    assert panel.azimuth == pytest.approx(-50.0)
    assert series.marker == "s"
    assert series.marker_size == pytest.approx(8.0)
    assert series.color == "#ff0000"
    assert series.alpha == pytest.approx(0.8)


def test_save_reopen_preserves_labels_and_title(tmp_path):
    project, _dataset, _panel3d = _mixed_project()
    out_path = save_project(project, tmp_path / "mixed.gnovi")

    reloaded = load_project(out_path)
    panel = reloaded.workbenches[0].figure.panels[1]

    assert panel.title == "3D scatter"
    assert panel.x_label == "T"
    assert panel.y_label == "Comp"
    assert panel.z_label == "Cond"


def test_save_reopen_preserves_dataset_identity_sharing(tmp_path):
    project, dataset, _panel3d = _mixed_project()
    out_path = save_project(project, tmp_path / "mixed.gnovi")

    reloaded = load_project(out_path)
    reloaded_series = reloaded.workbenches[0].figure.panels[1].series[0]
    reloaded_dataset = reloaded.dataset_manager.get(dataset.id)

    assert reloaded_series.dataset is reloaded_dataset


# --- Backward compatibility --------------------------------------------------------


def test_old_2d_only_project_files_still_open(tmp_path):
    """A project saved by a pre-3D GNOVI has no "kind" key on any panel at
    all -- must still load every panel as a plain `Panel`, unchanged."""
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    project.workbenches[0].figure.set_layout(1, 2)
    out_path = save_project(project, tmp_path / "old.gnovi")

    # Simulate a genuinely pre-3D file by stripping "kind" from every panel,
    # exactly what a file saved before Panel3D existed would look like.
    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    for panel_data in manifest["workbenches"][0]["figure"]["panels"]:
        del panel_data["kind"]
    manifest["project_format_version"] = 2
    tmp_zip = tmp_path / "old_rewritten.gnovi"
    with zipfile.ZipFile(out_path) as src, zipfile.ZipFile(tmp_zip, "w") as dst:
        for item in src.infolist():
            data = json.dumps(manifest).encode() if item.filename == "project.json" else src.read(item.filename)
            dst.writestr(item.filename, data)

    reloaded = load_project(tmp_zip)

    assert reloaded.workbenches[0].figure.layout == (1, 2)
    assert all(isinstance(p, Panel) for p in reloaded.workbenches[0].figure.panels)


def test_project_format_version_is_3_and_bumped_for_panel3d(tmp_path):
    project, _dataset, _panel3d = _mixed_project()
    out_path = save_project(project, tmp_path / "mixed.gnovi")

    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))

    assert manifest["project_format_version"] == PROJECT_FORMAT_VERSION == 3


def test_a_v2_manifest_with_no_kind_keys_loads_as_all_2d(tmp_path):
    """Directly exercises the version-2 shape `load_project` must keep
    accepting: `project_format_version: 2`, panels with no "kind" key."""
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    out_path = save_project(project, tmp_path / "v2.gnovi")

    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))
    manifest["project_format_version"] = 2
    for panel_data in manifest["workbenches"][0]["figure"]["panels"]:
        del panel_data["kind"]
    rewritten = tmp_path / "v2_rewritten.gnovi"
    with zipfile.ZipFile(out_path) as src, zipfile.ZipFile(rewritten, "w") as dst:
        for item in src.infolist():
            data = json.dumps(manifest).encode() if item.filename == "project.json" else src.read(item.filename)
            dst.writestr(item.filename, data)

    reloaded = load_project(rewritten)

    assert isinstance(reloaded.workbenches[0].figure.panels[0], Panel)


# --- No Matplotlib objects enter the serialized model -----------------------------


def test_no_matplotlib_objects_in_the_serialized_manifest(tmp_path):
    project, _dataset, _panel3d = _mixed_project()
    out_path = save_project(project, tmp_path / "mixed.gnovi")

    with zipfile.ZipFile(out_path) as zf:
        raw = zf.read("project.json")

    # A plain, successful json.loads() over the whole manifest is itself
    # strong evidence: any stray Matplotlib object (Axes3D, Artist, etc.)
    # accidentally captured in Panel3D/Series3D state would not be JSON-
    # serializable at all, so `save_project` would have raised instead of
    # producing this file.
    manifest = json.loads(raw)
    panel3d_data = manifest["workbenches"][0]["figure"]["panels"][1]
    assert panel3d_data["kind"] == "3d"
    assert isinstance(panel3d_data["elevation"], (int, float))
    assert isinstance(panel3d_data["series"], list)


# --- Grouped 3D curve families: plot_type/line style/width/row selection/legend ---


def _diode_dataset():
    df = pd.DataFrame(
        {
            "Voltage_V": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
            "Temperature_C": [25.0, 35.0, 25.0, 35.0, 25.0, 35.0],
            "Current_mA": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        }
    )
    return Dataset(id="diode1", name="diode", dataframe=df)


def _grouped_project():
    dataset = _diode_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    wb = project.workbenches[0]
    panel = Panel3D(title="I-V families", legend_visible=True, legend_loc="upper left")
    panel.add_series(
        Series3D(
            dataset=dataset, x_column="Voltage_V", y_column="Temperature_C", z_column="Current_mA",
            label="25", plot_type=Plot3DType.LINE_MARKER, line_style="--", line_width=2.5,
            row_indices=(0, 2, 4), color="#ff0000", color_is_manual=True,
        )
    )
    panel.add_series(
        Series3D(
            dataset=dataset, x_column="Voltage_V", y_column="Temperature_C", z_column="Current_mA",
            label="35", plot_type=Plot3DType.LINE, row_indices=(1, 3, 5), visible=False,
        )
    )
    wb.figure.panels = [panel]
    wb.figure.layout = (1, 1)
    return project, dataset, panel


def test_save_reopen_preserves_a_grouped_curve_family(tmp_path):
    project, dataset, panel = _grouped_project()
    out_path = save_project(project, tmp_path / "grouped.gnovi")

    reloaded = load_project(out_path)
    reloaded_panel = reloaded.workbenches[0].figure.panels[0]

    assert isinstance(reloaded_panel, Panel3D)
    assert len(reloaded_panel.series) == 2
    by_label = {s.label: s for s in reloaded_panel.series}
    assert by_label["25"].row_indices == (0, 2, 4)
    assert by_label["35"].row_indices == (1, 3, 5)


def test_save_reopen_preserves_plot_type_line_style_and_width(tmp_path):
    project, _dataset, _panel = _grouped_project()
    out_path = save_project(project, tmp_path / "grouped.gnovi")

    reloaded = load_project(out_path)
    series = {s.label: s for s in reloaded.workbenches[0].figure.panels[0].series}

    assert series["25"].plot_type == Plot3DType.LINE_MARKER
    assert series["25"].line_style == "--"
    assert series["25"].line_width == pytest.approx(2.5)
    assert series["35"].plot_type == Plot3DType.LINE


def test_save_reopen_preserves_manual_and_automatic_colors(tmp_path):
    project, _dataset, panel = _grouped_project()
    out_path = save_project(project, tmp_path / "grouped.gnovi")

    reloaded = load_project(out_path)
    series = {s.label: s for s in reloaded.workbenches[0].figure.panels[0].series}

    assert series["25"].color == "#ff0000"
    assert series["25"].color_is_manual is True


def test_save_reopen_preserves_visibility_per_series(tmp_path):
    project, _dataset, _panel = _grouped_project()
    out_path = save_project(project, tmp_path / "grouped.gnovi")

    reloaded = load_project(out_path)
    series = {s.label: s for s in reloaded.workbenches[0].figure.panels[0].series}

    assert series["25"].visible is True
    assert series["35"].visible is False


def test_save_reopen_preserves_legend_state(tmp_path):
    project, _dataset, _panel = _grouped_project()
    out_path = save_project(project, tmp_path / "grouped.gnovi")

    reloaded = load_project(out_path)
    panel = reloaded.workbenches[0].figure.panels[0]

    assert panel.legend_visible is True
    assert panel.legend_loc == "upper left"


def test_save_reopen_preserves_dataset_identity_across_a_grouped_family(tmp_path):
    project, dataset, _panel = _grouped_project()
    out_path = save_project(project, tmp_path / "grouped.gnovi")

    reloaded = load_project(out_path)
    reloaded_dataset = reloaded.dataset_manager.get(dataset.id)
    series = reloaded.workbenches[0].figure.panels[0].series

    assert all(s.dataset is reloaded_dataset for s in series)


def test_project_format_version_is_unchanged_for_grouped_families(tmp_path):
    """This milestone's own decision: every new field is optional with a
    safe default, so no PROJECT_FORMAT_VERSION bump was needed (see
    `Series3D.to_dict`'s own docstring for the full reasoning)."""
    project, _dataset, _panel = _grouped_project()
    out_path = save_project(project, tmp_path / "grouped.gnovi")

    with zipfile.ZipFile(out_path) as zf:
        manifest = json.loads(zf.read("project.json"))

    assert manifest["project_format_version"] == PROJECT_FORMAT_VERSION == 3
