import pandas as pd

from gnovi_plot.core.project import Project
from gnovi_plot.core.project_io import PROJECT_FORMAT_VERSION
from gnovi_plot.core.workbench import Workbench
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.figure import GnoviFigure


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return Dataset(name=name, dataframe=df)


def test_project_new_is_empty_with_a_single_default_workbench():
    project = Project.new()
    assert project.name == "Untitled Project"
    assert len(project.dataset_manager) == 0
    assert len(project.graph_library) == 0
    assert len(project.workbenches) == 1
    assert project.workbenches[0].name == "Workbench 1"
    assert project.active_workbench_id == project.workbenches[0].id
    assert project.path is None


def test_project_active_workbench_property_follows_the_id():
    workbench_a = Workbench(name="A", figure=GnoviFigure())
    workbench_b = Workbench(name="B", figure=GnoviFigure())
    project = Project(workbenches=[workbench_a, workbench_b], active_workbench_id=workbench_b.id)
    assert project.active_workbench is workbench_b

    project.active_workbench_id = workbench_a.id
    assert project.active_workbench is workbench_a


def test_project_dataset_ids_are_stable_and_unique_across_multiple_datasets():
    project = Project.new()
    a = _make_dataset("a")
    b = _make_dataset("b")
    project.dataset_manager.add(a)
    project.dataset_manager.add(b)

    assert a.id != b.id
    assert project.dataset_manager.get(a.id) is a
    assert project.dataset_manager.get(b.id) is b
    # id is stable -- looking it up again returns the exact same identity,
    # not a fresh/regenerated one.
    assert project.dataset_manager.get(a.id) is project.dataset_manager.get(a.id)


def test_project_format_version_is_a_positive_integer_independent_of_app_version():
    from gnovi_plot.core.app_info import __version__ as app_version

    assert isinstance(PROJECT_FORMAT_VERSION, int)
    assert PROJECT_FORMAT_VERSION >= 1
    # Not coupled to the app's own version string/scheme.
    assert str(PROJECT_FORMAT_VERSION) != app_version
