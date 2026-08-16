from __future__ import annotations

import io
import json
import logging
import os
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gnovi_plot.core.app_info import __version__ as APP_VERSION
from gnovi_plot.core.project import Project
from gnovi_plot.core.workbench import DEFAULT_WORKBENCH_NAME, Workbench
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.graph_library import GraphLibrary

_logger = logging.getLogger("gnovi_plot")

# The `.gnovi` container's own schema version -- independent of
# `core.app_info.__version__` (the *application's* version). Bump this only
# when `project.json`'s structure changes in a way old code can't read.
#
# v1 -> v2: a single flat `figures`/`active_figure_index` list became named,
# independently-switchable `workbenches`/`active_workbench_id` -- see
# `_migrate_v1_to_v2` for the explicit migration `load_project` runs on an
# older file rather than rejecting it.
PROJECT_FORMAT_VERSION = 2

_MANIFEST_NAME = "project.json"
_DATASETS_DIR = "datasets"


class ProjectIOError(Exception):
    """Base for every error `save_project`/`load_project` can raise."""


class CorruptProjectError(ProjectIOError):
    """The file isn't a valid `.gnovi` project (not a zip, missing/invalid
    manifest, or a referenced dataset file is missing from the archive)."""


class UnsupportedProjectVersionError(ProjectIOError):
    """The file's `project_format_version` is newer than this app version
    of Gnovi Studio knows how to read."""


def _dataset_dir(dataset_id: str) -> str:
    return f"{_DATASETS_DIR}/{dataset_id}"


def save_project(project: Project, path: str | Path) -> Path:
    """Write `project` to `path` as a versioned ZIP `.gnovi` container (see
    module docstring-level comment on `PROJECT_FORMAT_VERSION`): one
    `project.json` manifest plus, per dataset, a `raw.csv`/`working.csv`
    pair -- see `load_project` for the exact layout. Never uses `pickle`.

    Written to a temp file in the same directory and then swapped onto
    `path` with `os.replace` -- atomic, so an interrupted/crashed write can
    never corrupt an existing project file at `path`.
    """
    path = Path(path)
    manifest = _build_manifest(project)

    tmp_path = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_MANIFEST_NAME, json.dumps(manifest, indent=2))
        for dataset in project.dataset_manager.datasets:
            dir_name = _dataset_dir(dataset.id)
            zf.writestr(f"{dir_name}/raw.csv", dataset.raw_dataframe.to_csv(index=False))
            zf.writestr(f"{dir_name}/working.csv", dataset.dataframe.to_csv(index=False))
    os.replace(tmp_path, path)

    project.path = path
    return path


def _build_manifest(project: Project) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "project_format_version": PROJECT_FORMAT_VERSION,
        "app_version": APP_VERSION,
        "project_name": project.name,
        "saved_at": now,
        "datasets": [
            {
                **dataset.to_dict(),
                "raw_data_file": f"{_dataset_dir(dataset.id)}/raw.csv",
                "working_data_file": f"{_dataset_dir(dataset.id)}/working.csv",
            }
            for dataset in project.dataset_manager.datasets
        ],
        "graph_library": project.graph_library.to_dict(),
        "workbenches": [workbench.to_dict() for workbench in project.workbenches],
        "active_workbench_id": project.active_workbench_id,
        "results": [],
    }


def load_project(path: str | Path) -> Project:
    """Read `path` back into a fully-built `Project`, or raise a
    `ProjectIOError` subclass -- never returns/mutates a partially-loaded
    project, so a malformed file leaves the caller's current project
    untouched (see `gui.main_window`, which only swaps state after this
    returns successfully).

    A `PlotSeries` whose `dataset_id` doesn't resolve against the loaded
    datasets is skipped (logged, not raised) -- one stale reference
    shouldn't block recovering an otherwise-valid project. A dataset whose
    `raw.csv`/`working.csv` is missing from the archive IS fatal (that
    dataset can't be reconstructed at all).

    A format-1 manifest (single `figures`/`active_figure_index` list) is
    transparently migrated to format 2 (`workbenches`/`active_workbench_id`)
    via `_migrate_v1_to_v2` before the rest of this function ever sees it --
    an old project is never rejected just for predating multiple Workbenches.
    """
    path = Path(path)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            _check_format_version(manifest)
            if manifest["project_format_version"] == 1:
                manifest = _migrate_v1_to_v2(manifest)
            dataset_manager, dataset_lookup = _load_datasets(zf, manifest)
    except zipfile.BadZipFile as exc:
        raise CorruptProjectError(f"'{path.name}' is not a valid Gnovi Studio project file.") from exc

    graph_library = GraphLibrary.from_dict(manifest.get("graph_library", []), dataset_lookup)
    workbenches = [Workbench.from_dict(w, dataset_lookup) for w in manifest.get("workbenches", [])]
    if not workbenches:
        workbenches = [Workbench(name=DEFAULT_WORKBENCH_NAME, figure=GnoviFigure())]
    active_workbench_id = manifest.get("active_workbench_id")
    if active_workbench_id not in {w.id for w in workbenches}:
        active_workbench_id = workbenches[0].id

    return Project(
        name=manifest.get("project_name", "Untitled Project"),
        dataset_manager=dataset_manager,
        graph_library=graph_library,
        workbenches=workbenches,
        active_workbench_id=active_workbench_id,
        path=path,
    )


def _migrate_v1_to_v2(manifest: dict) -> dict:
    """Format 1 (a single flat `figures` list + `active_figure_index`) ->
    format 2 (named, independently-switchable `workbenches` +
    `active_workbench_id`; see `core.workbench.Workbench`). Each old Figure
    becomes its own Workbench, numbered in original order ("Workbench 1",
    "Workbench 2", ...), preserving every Figure's content and which one
    was active. A pure dict -> dict transform (no I/O), so it's testable
    against a literal format-1 fixture independent of whatever the current
    `save_project` happens to write -- the real migration path a genuinely
    old `.gnovi` file goes through.
    """
    figures = manifest.get("figures", [])
    workbenches = [
        {"id": uuid.uuid4().hex, "name": f"Workbench {i + 1}", "figure": figure_data}
        for i, figure_data in enumerate(figures)
    ]
    if not workbenches:
        workbenches = [{"id": uuid.uuid4().hex, "name": DEFAULT_WORKBENCH_NAME, "figure": {}}]

    active_figure_index = manifest.get("active_figure_index", 0)
    active_figure_index = min(max(active_figure_index, 0), len(workbenches) - 1)

    manifest["workbenches"] = workbenches
    manifest["active_workbench_id"] = workbenches[active_figure_index]["id"]
    manifest["project_format_version"] = 2
    return manifest


def _read_manifest(zf: zipfile.ZipFile) -> dict:
    try:
        raw = zf.read(_MANIFEST_NAME)
    except KeyError as exc:
        raise CorruptProjectError("Project file is missing its project.json manifest.") from exc
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorruptProjectError("Project file's manifest is not valid JSON.") from exc
    if "project_format_version" not in manifest:
        raise CorruptProjectError("Project file's manifest is missing project_format_version.")
    return manifest


def _check_format_version(manifest: dict) -> None:
    version = manifest["project_format_version"]
    if not isinstance(version, int) or version < 1:
        raise CorruptProjectError(f"Project file has an invalid format version: {version!r}.")
    if version > PROJECT_FORMAT_VERSION:
        raise UnsupportedProjectVersionError(
            f"This project was saved by a newer version of Gnovi Studio "
            f"(format version {version}, this app supports up to {PROJECT_FORMAT_VERSION}) "
            "and can't be opened here."
        )


def _load_datasets(zf: zipfile.ZipFile, manifest: dict) -> tuple[DatasetManager, dict[str, Dataset]]:
    manager = DatasetManager()
    for entry in manifest.get("datasets", []):
        raw_df = _read_csv_member(zf, entry.get("raw_data_file"), entry.get("name", entry.get("id")))
        working_df = _read_csv_member(
            zf, entry.get("working_data_file"), entry.get("name", entry.get("id"))
        )
        dataset = Dataset.from_project_data(entry, raw_dataframe=raw_df, working_dataframe=working_df)
        manager.add(dataset)
    dataset_lookup = {dataset.id: dataset for dataset in manager.datasets}
    return manager, dataset_lookup


def _read_csv_member(zf: zipfile.ZipFile, member_name: str | None, dataset_label: str) -> pd.DataFrame:
    if not member_name:
        raise CorruptProjectError(f"Dataset '{dataset_label}' is missing a data file reference.")
    try:
        raw_bytes = zf.read(member_name)
    except KeyError as exc:
        raise CorruptProjectError(
            f"Dataset '{dataset_label}' references '{member_name}', which is missing from the project file."
        ) from exc
    return pd.read_csv(io.BytesIO(raw_bytes))
