import pandas as pd

from gnovi_plot.core.project import Project
from gnovi_plot.core.workbench import Workbench
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return Dataset(name=name, dataframe=df)


def _manager_with(*datasets):
    manager = DatasetManager()
    for dataset in datasets:
        manager.add(dataset)
    return manager


# --- Workbench: identity, wraps a GnoviFigure --------------------------------


def test_workbench_holds_a_stable_id_and_the_given_figure():
    figure = GnoviFigure()
    workbench = Workbench(name="CV Comparison", figure=figure)

    assert workbench.name == "CV Comparison"
    assert workbench.figure is figure
    assert workbench.id


def test_two_workbenches_get_distinct_ids():
    a = Workbench(name="A", figure=GnoviFigure())
    b = Workbench(name="B", figure=GnoviFigure())
    assert a.id != b.id


def test_fresh_workbench_has_empty_analysis_result_history():
    workbench = Workbench(name="A", figure=GnoviFigure())
    assert workbench.analysis_results.current(workbench.figure.active_panel.id) is None
    assert workbench.analysis_results.all(workbench.figure.active_panel.id) == []


def test_two_fresh_workbenches_get_independent_analysis_result_history_objects():
    a = Workbench(name="A", figure=GnoviFigure())
    b = Workbench(name="B", figure=GnoviFigure())
    assert a.analysis_results is not b.analysis_results


def test_workbench_does_not_duplicate_active_panel_state():
    """GnoviFigure.active_panel_index remains the sole owner of active-panel
    state -- Workbench never shadows it."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(Workbench)}
    assert "active_panel_index" not in field_names
    assert "active_panel" not in field_names


def test_workbench_to_dict_from_dict_round_trip_preserves_identity_and_content():
    dataset = _make_dataset()
    figure = GnoviFigure()
    figure.active_panel.title = "Overlay"
    figure.add_series(PlotSeries.line(dataset, "x", "y", color="#ff0000"))
    workbench = Workbench(name="CV Comparison", figure=figure)

    data = workbench.to_dict()
    restored = Workbench.from_dict(data, {dataset.id: dataset})

    assert restored.id == workbench.id
    assert restored.name == "CV Comparison"
    assert restored.figure.active_panel.title == "Overlay"
    assert restored.figure.series[0].dataset is dataset
    assert restored.figure.series[0].color == "#ff0000"


def test_workbench_from_dict_missing_name_falls_back_to_a_default():
    data = {"id": "abc", "figure": GnoviFigure().to_dict()}
    restored = Workbench.from_dict(data, {})
    assert restored.name


# --- Workbench: analysis_results persistence ----------------------------------


def test_workbench_to_dict_from_dict_round_trips_analysis_results():
    from gnovi_plot.analysis.fitting import LINEAR, POLYNOMIAL, fit_curve

    dataset = _make_dataset()
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    panel_1_id = figure.panels[0].id
    panel_2_id = figure.panels[1].id
    workbench = Workbench(name="W", figure=figure)

    linear = fit_curve(
        [1.0, 2.0, 3.0], [2.0, 4.0, 6.0], LINEAR, source_dataset_id=dataset.id, x_column="x", y_column="y",
        source_panel_id=panel_1_id,
    )
    polynomial = fit_curve(
        [1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0], POLYNOMIAL,
        source_dataset_id=dataset.id, x_column="x", y_column="y", source_panel_id=panel_2_id,
    )
    workbench.analysis_results.add(panel_1_id, linear)
    workbench.analysis_results.add(panel_2_id, polynomial)

    data = workbench.to_dict()
    restored = Workbench.from_dict(data, {dataset.id: dataset})

    restored_linear = restored.analysis_results.current(panel_1_id)
    restored_polynomial = restored.analysis_results.current(panel_2_id)
    assert restored_linear.model == LINEAR
    assert restored_linear.result_id == linear.result_id
    assert restored_polynomial.model == POLYNOMIAL
    assert restored_polynomial.result_id == polynomial.result_id


def test_workbench_from_dict_with_no_analysis_results_key_gets_empty_history():
    """A project saved before analysis-result persistence existed has no
    `"analysis_results"` key at all -- must load with empty history, same
    as New Project, not raise."""
    figure = GnoviFigure()
    data = {"id": "abc", "name": "Old Project Workbench", "figure": figure.to_dict()}

    restored = Workbench.from_dict(data, {})

    assert restored.analysis_results.current(restored.figure.active_panel.id) is None


# --- Project: Workbench collection management --------------------------------


def test_new_project_has_exactly_one_workbench():
    project = Project.new()
    assert len(project.workbenches) == 1
    assert project.active_workbench_id == project.workbenches[0].id


def test_get_workbench_returns_none_for_unknown_id():
    project = Project.new()
    assert project.get_workbench("does-not-exist") is None


def test_add_workbench_appends_and_does_not_change_active():
    project = Project.new()
    original_active = project.active_workbench_id
    new_workbench = Workbench(name="New Scan", figure=GnoviFigure())

    project.add_workbench(new_workbench)

    assert len(project.workbenches) == 2
    assert project.get_workbench(new_workbench.id) is new_workbench
    assert project.active_workbench_id == original_active  # adding never switches


def test_rename_workbench():
    project = Project.new()
    workbench_id = project.workbenches[0].id

    project.rename_workbench(workbench_id, "CV Scan Rates")

    assert project.get_workbench(workbench_id).name == "CV Scan Rates"


def test_rename_unknown_workbench_is_a_no_op():
    project = Project.new()
    project.rename_workbench("does-not-exist", "New Name")  # must not raise


# --- Project: duplication -- shared datasets, independent everything else ----


def test_duplicate_workbench_produces_an_independent_copy_with_a_new_id():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    original = project.workbenches[0]
    original.figure.active_panel.title = "Base"
    original.figure.add_series(PlotSeries.line(dataset, "x", "y"))
    original.name = "Base Workbench"

    copy_workbench = project.duplicate_workbench(original.id)

    assert copy_workbench is not None
    assert copy_workbench.id != original.id
    assert copy_workbench.name == "Base Workbench (Copy)"
    assert copy_workbench.figure is not original.figure
    assert copy_workbench.figure.active_panel.title == "Base"
    assert len(project.workbenches) == 2


def test_duplicate_workbench_shares_datasets_but_not_figure_panel_series_state():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    original = project.workbenches[0]
    original.figure.add_series(PlotSeries.line(dataset, "x", "y", color="#111111"))

    copy_workbench = project.duplicate_workbench(original.id)

    # Shared live Dataset -- never duplicated.
    assert copy_workbench.figure.series[0].dataset is dataset
    assert len(project.dataset_manager) == 1

    # But editing the copy's series/panel never touches the original.
    copy_workbench.figure.series[0].color = "#999999"
    copy_workbench.figure.active_panel.title = "Edited Copy"
    copy_workbench.figure.add_series(PlotSeries.line(dataset, "x", "y"))

    assert original.figure.series[0].color == "#111111"
    assert original.figure.active_panel.title == ""
    assert len(original.figure.series) == 1


def test_duplicate_workbench_gives_the_cloned_panel_a_new_id():
    """A duplicated Workbench must never share `Panel.id` with the
    Workbench it was duplicated from -- otherwise a fit run on one would
    silently appear to belong to the other's structurally-identical panel
    (see `plotting.graph.clone_figure_with_shared_datasets`)."""
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    original = project.workbenches[0]
    original_panel_id = original.figure.active_panel.id

    copy_workbench = project.duplicate_workbench(original.id)

    assert copy_workbench.figure.active_panel.id != original_panel_id


def test_duplicate_workbench_does_not_inherit_the_originals_analysis_history():
    """Runtime-only in this milestone: a duplicate must start with an
    empty PanelResultHistory, never a copy of the original's -- even
    though the panels are structurally identical, they have fresh ids
    (see `test_duplicate_workbench_gives_the_cloned_panel_a_new_id`), so
    there is nothing to correctly remap anyway."""
    from gnovi_plot.analysis.results import AnalysisResult

    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    original = project.workbenches[0]
    original_panel_id = original.figure.active_panel.id
    result = AnalysisResult(
        source_dataset_id=dataset.id,
        source_dataset_name=None,
        source_series_id=None,
        source_series_label=None,
        x_column="x",
        y_column="y",
        row_range=None,
        source_panel_id=original_panel_id,
        result_id="result-1",
    )
    original.analysis_results.add(original_panel_id, result)

    copy_workbench = project.duplicate_workbench(original.id)

    assert copy_workbench.analysis_results.current(copy_workbench.figure.active_panel.id) is None
    assert copy_workbench.analysis_results.all(copy_workbench.figure.active_panel.id) == []
    # The original's own history is untouched by the duplication.
    assert original.analysis_results.current(original_panel_id) is result


def test_duplicate_workbench_preserves_panel_source_graph_id():
    """Graph-origin metadata (Panel.source_graph_id) survives duplication --
    the duplicate still identifies the same origin Graph as the original."""
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    original = project.workbenches[0]
    original.figure.active_panel.source_graph_id = "graph-123"

    copy_workbench = project.duplicate_workbench(original.id)

    assert copy_workbench.figure.active_panel.source_graph_id == "graph-123"


def test_duplicate_unknown_workbench_returns_none():
    project = Project.new()
    assert project.duplicate_workbench("does-not-exist") is None


# --- Project: removal ---------------------------------------------------------


def test_remove_workbench_removes_it():
    project = Project.new()
    second = Workbench(name="Second", figure=GnoviFigure())
    project.add_workbench(second)

    removed = project.remove_workbench(second.id)

    assert removed is True
    assert project.get_workbench(second.id) is None
    assert len(project.workbenches) == 1


def test_remove_unknown_workbench_is_a_no_op():
    project = Project.new()
    assert project.remove_workbench("does-not-exist") is False
    assert len(project.workbenches) == 1


def test_remove_workbench_refuses_to_drop_the_last_one():
    project = Project.new()
    only_id = project.workbenches[0].id

    removed = project.remove_workbench(only_id)

    assert removed is False
    assert len(project.workbenches) == 1
    assert project.workbenches[0].id == only_id


def test_datasets_and_graph_library_survive_workbench_removal():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    project.graph_library.save_panel_as_graph(project.workbenches[0].figure, "G1", project.dataset_manager)
    second = Workbench(name="Second", figure=GnoviFigure())
    project.add_workbench(second)

    project.remove_workbench(second.id)

    assert len(project.dataset_manager) == 1
    assert len(project.graph_library) == 1


def test_removing_the_active_workbench_retargets_to_a_sensible_sibling():
    project = Project.new()
    first_id = project.workbenches[0].id
    second = Workbench(name="Second", figure=GnoviFigure())
    third = Workbench(name="Third", figure=GnoviFigure())
    project.add_workbench(second)
    project.add_workbench(third)
    project.active_workbench_id = second.id

    project.remove_workbench(second.id)

    # Lands on the workbench immediately before the removed one.
    assert project.active_workbench_id == first_id
    assert [w.id for w in project.workbenches] == [first_id, third.id]


def test_removing_a_non_active_workbench_does_not_change_active_id():
    project = Project.new()
    first_id = project.workbenches[0].id
    second = Workbench(name="Second", figure=GnoviFigure())
    project.add_workbench(second)

    project.remove_workbench(second.id)

    assert project.active_workbench_id == first_id
