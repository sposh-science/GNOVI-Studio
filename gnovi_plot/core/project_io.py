from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gnovi_plot.core.app_info import __version__ as APP_VERSION
from gnovi_plot.core.project import Project
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.graph_library import GraphLibrary

_logger = logging.getLogger("gnovi_plot")

# The `.gnovi` container's own schema version -- independent of
# `core.app_info.__version__` (the *application's* version). Bump this only
# when `project.json`'s structure changes in a way old code can't read.
PROJECT_FORMAT_VERSION = 1

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
        "figures": [figure.to_dict() for figure in project.figures],
        "active_figure_index": project.active_figure_index,
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
    """
    path = Path(path)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            _check_format_version(manifest)
            dataset_manager, dataset_lookup = _load_datasets(zf, manifest)
    except zipfile.BadZipFile as exc:
        raise CorruptProjectError(f"'{path.name}' is not a valid Gnovi Studio project file.") from exc

    graph_library = GraphLibrary.from_dict(manifest.get("graph_library", []), dataset_lookup)
    figures = [GnoviFigure.from_dict(f, dataset_lookup) for f in manifest.get("figures", [])]
    if not figures:
        figures = [GnoviFigure()]
    active_figure_index = manifest.get("active_figure_index", 0)
    active_figure_index = min(max(active_figure_index, 0), len(figures) - 1)

    return Project(
        name=manifest.get("project_name", "Untitled Project"),
        dataset_manager=dataset_manager,
        graph_library=graph_library,
        figures=figures,
        active_figure_index=active_figure_index,
        path=path,
    )


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
