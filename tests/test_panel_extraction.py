"""`Project.extract_panel_to_workbench` -- "Panels -> Extract Active Panel to
New Workbench": copies exactly one Panel into a brand new, independent 1x1
Workbench, carrying that panel's own analysis history along with it
(remapped onto the extracted Panel's fresh id), while keeping every Dataset
shared by identity -- see `plotting.graph.clone_panel_with_shared_datasets`
and `analysis.panel_results.PanelResultHistory.copy_panel`.

Mirrors `test_workbench.py`'s "Project: duplication" section in style, since
this is `duplicate_workbench`'s closest sibling.
"""

import pandas as pd

from gnovi_plot.analysis.fitting import LINEAR, POLYNOMIAL, fit_curve
from gnovi_plot.core.project import Project
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.figure import PlotTheme
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 5.0, 6.0, 7.0]})
    return Dataset(name=name, dataframe=df)


# --- Basic extraction: identity + Workbench/Panel shape -----------------------


def test_extract_panel_from_1x3_creates_a_1x1_workbench_and_leaves_source_unchanged():
    project = Project.new()
    workbench = project.workbenches[0]
    workbench.figure.set_layout(1, 3)
    workbench.figure.set_active_panel(1)
    original_panel_ids = [p.id for p in workbench.figure.panels]

    extracted = project.extract_panel_to_workbench(workbench.id, workbench.figure.panels[1].id)

    assert extracted is not None
    assert extracted.figure.layout == (1, 1)
    assert len(extracted.figure.panels) == 1
    # Source Workbench is completely untouched.
    assert workbench.figure.layout == (1, 3)
    assert [p.id for p in workbench.figure.panels] == original_panel_ids
    assert len(project.workbenches) == 2


def test_extracted_workbench_gets_a_fresh_id():
    project = Project.new()
    workbench = project.workbenches[0]

    extracted = project.extract_panel_to_workbench(workbench.id, workbench.figure.active_panel.id)

    assert extracted.id != workbench.id


def test_extracted_panel_gets_a_fresh_id():
    project = Project.new()
    workbench = project.workbenches[0]
    original_panel_id = workbench.figure.active_panel.id

    extracted = project.extract_panel_to_workbench(workbench.id, original_panel_id)

    assert extracted.figure.active_panel.id != original_panel_id


def test_extract_unknown_workbench_returns_none():
    project = Project.new()
    panel_id = project.workbenches[0].figure.active_panel.id
    assert project.extract_panel_to_workbench("does-not-exist", panel_id) is None


def test_extract_unknown_panel_returns_none():
    project = Project.new()
    workbench = project.workbenches[0]
    assert project.extract_panel_to_workbench(workbench.id, "does-not-exist") is None
    assert len(project.workbenches) == 1  # no-op, nothing added


# --- Visual state / series ------------------------------------------------------


def test_extracted_panel_preserves_visual_state_except_identity():
    project = Project.new()
    workbench = project.workbenches[0]
    panel = workbench.figure.active_panel
    panel.title = "Ferricyanide CV"
    panel.xlabel = "E / V"
    panel.ylabel = "i / mA"
    panel.xlim = (-0.2, 0.6)
    panel.grid = True
    panel.legend_loc = "upper left"
    original_id = panel.id

    extracted = project.extract_panel_to_workbench(workbench.id, panel.id)
    extracted_panel = extracted.figure.active_panel

    assert extracted_panel.id != original_id
    assert extracted_panel.title == "Ferricyanide CV"
    assert extracted_panel.xlabel == "E / V"
    assert extracted_panel.ylabel == "i / mA"
    assert extracted_panel.xlim == (-0.2, 0.6)
    assert extracted_panel.grid is True
    assert extracted_panel.legend_loc == "upper left"


def test_extracted_figure_copies_the_source_figures_plot_theme():
    project = Project.new()
    workbench = project.workbenches[0]
    workbench.figure.plot_theme = PlotTheme.DARK

    extracted = project.extract_panel_to_workbench(workbench.id, workbench.figure.active_panel.id)

    assert extracted.figure.plot_theme == PlotTheme.DARK


def test_extracted_panel_preserves_every_plot_series():
    dataset_a = _make_dataset("a")
    dataset_b = _make_dataset("b")
    project = Project.new()
    project.dataset_manager.add(dataset_a)
    project.dataset_manager.add(dataset_b)
    workbench = project.workbenches[0]
    workbench.figure.add_series(PlotSeries.line(dataset_a, "x", "y", color="#111111"))
    workbench.figure.add_series(PlotSeries.scatter(dataset_b, "x", "y", color="#222222"))

    extracted = project.extract_panel_to_workbench(workbench.id, workbench.figure.active_panel.id)

    assert len(extracted.figure.series) == 2
    assert {s.color for s in extracted.figure.series} == {"#111111", "#222222"}


def test_extracted_series_share_the_exact_same_dataset_instance_as_the_source():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    workbench.figure.add_series(PlotSeries.line(dataset, "x", "y"))
    source_series = workbench.figure.series[0]

    extracted = project.extract_panel_to_workbench(workbench.id, workbench.figure.active_panel.id)
    extracted_series = extracted.figure.series[0]

    assert extracted_series.dataset is source_series.dataset
    assert len(project.dataset_manager) == 1  # no duplicate Dataset registered


def test_editing_the_extracted_panel_never_touches_the_source_panel():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    workbench.figure.add_series(PlotSeries.line(dataset, "x", "y", color="#111111"))

    extracted = project.extract_panel_to_workbench(workbench.id, workbench.figure.active_panel.id)
    extracted.figure.series[0].color = "#999999"
    extracted.figure.active_panel.title = "Edited Extraction"
    extracted.figure.add_series(PlotSeries.line(dataset, "x", "y"))

    assert workbench.figure.series[0].color == "#111111"
    assert workbench.figure.active_panel.title == ""
    assert len(workbench.figure.series) == 1


# --- Naming ---------------------------------------------------------------------


def test_extracted_workbench_name_uses_source_name_and_1_based_panel_number():
    project = Project.new()
    workbench = project.workbenches[0]
    workbench.name = "Ferricyanide Analysis"
    workbench.figure.set_layout(1, 3)

    extracted = project.extract_panel_to_workbench(workbench.id, workbench.figure.panels[1].id)

    assert extracted.name == "Ferricyanide Analysis — Panel 2"


# --- Analysis history: independent copy, order, current selection, remap ------


def _linear(dataset, panel_id):
    return fit_curve(
        [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], LINEAR,
        source_dataset_id=dataset.id, x_column="x", y_column="y", source_panel_id=panel_id,
    )


def _polynomial(dataset, panel_id):
    return fit_curve(
        [1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0], POLYNOMIAL,
        source_dataset_id=dataset.id, x_column="x", y_column="y", source_panel_id=panel_id,
    )


def test_extraction_with_no_analysis_history_produces_an_empty_history():
    project = Project.new()
    workbench = project.workbenches[0]

    extracted = project.extract_panel_to_workbench(workbench.id, workbench.figure.active_panel.id)

    assert extracted.analysis_results.current(extracted.figure.active_panel.id) is None
    assert extracted.analysis_results.all(extracted.figure.active_panel.id) == []


def test_extracted_workbench_gets_an_independent_history_object():
    project = Project.new()
    workbench = project.workbenches[0]

    extracted = project.extract_panel_to_workbench(workbench.id, workbench.figure.active_panel.id)

    assert extracted.analysis_results is not workbench.analysis_results


def test_extracted_history_copies_every_result_preserving_order():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id
    linear = _linear(dataset, panel_id)
    polynomial = _polynomial(dataset, panel_id)
    workbench.analysis_results.add(panel_id, linear)
    workbench.analysis_results.add(panel_id, polynomial)

    extracted = project.extract_panel_to_workbench(workbench.id, panel_id)
    extracted_panel_id = extracted.figure.active_panel.id

    assert [r.result_id for r in extracted.analysis_results.all(extracted_panel_id)] == [
        linear.result_id,
        polynomial.result_id,
    ]


def test_extracted_history_preserves_the_explicitly_selected_current_result():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id
    linear = _linear(dataset, panel_id)
    polynomial = _polynomial(dataset, panel_id)
    workbench.analysis_results.add(panel_id, linear)
    workbench.analysis_results.add(panel_id, polynomial)
    workbench.analysis_results.set_current(panel_id, linear.result_id)  # explicitly select the older one

    extracted = project.extract_panel_to_workbench(workbench.id, panel_id)
    extracted_panel_id = extracted.figure.active_panel.id

    assert extracted.analysis_results.current(extracted_panel_id).result_id == linear.result_id


def test_extracted_history_remaps_source_panel_id_to_the_new_panel():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id
    linear = _linear(dataset, panel_id)
    workbench.analysis_results.add(panel_id, linear)

    extracted = project.extract_panel_to_workbench(workbench.id, panel_id)
    extracted_panel_id = extracted.figure.active_panel.id

    copied = extracted.analysis_results.current(extracted_panel_id)
    assert copied.source_panel_id == extracted_panel_id
    assert copied.source_panel_id != panel_id


def test_extracted_history_preserves_result_id_unchanged():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id
    linear = _linear(dataset, panel_id)
    workbench.analysis_results.add(panel_id, linear)

    extracted = project.extract_panel_to_workbench(workbench.id, panel_id)
    extracted_panel_id = extracted.figure.active_panel.id

    assert extracted.analysis_results.current(extracted_panel_id).result_id == linear.result_id


def test_copied_results_are_independent_objects_not_shared_with_the_source():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id
    linear = _linear(dataset, panel_id)
    workbench.analysis_results.add(panel_id, linear)

    extracted = project.extract_panel_to_workbench(workbench.id, panel_id)
    extracted_panel_id = extracted.figure.active_panel.id
    copied = extracted.analysis_results.current(extracted_panel_id)

    assert copied is not linear
    assert copied.result_id == linear.result_id  # same identity, different object


def test_mutating_the_extracted_history_never_affects_the_source_history():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id
    linear = _linear(dataset, panel_id)
    workbench.analysis_results.add(panel_id, linear)

    extracted = project.extract_panel_to_workbench(workbench.id, panel_id)
    extracted_panel_id = extracted.figure.active_panel.id
    extra = _polynomial(dataset, extracted_panel_id)
    extracted.analysis_results.add(extracted_panel_id, extra)

    assert len(extracted.analysis_results.all(extracted_panel_id)) == 2
    assert len(workbench.analysis_results.all(panel_id)) == 1  # source untouched


# --- Fit-derived Dataset metadata linkage --------------------------------------


def test_fit_derived_dataset_metadata_result_id_still_matches_the_copied_fitresult():
    """Reproduces "Add Fit Curve to Plot"'s own linkage (see
    `gui.widgets.analysis_panel.AnalysisPanel._on_add_fit_curve_clicked`):
    a derived Dataset stamped with `metadata["result_id"]`, plotted as a
    PlotSeries. After extraction, the shared Dataset's metadata is
    untouched and still matches the copied FitResult's own `result_id`."""
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id
    linear = _linear(dataset, panel_id)
    workbench.analysis_results.add(panel_id, linear)

    fit_dataset = Dataset(
        name="Fit: linear",
        dataframe=pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 4.0]}),
        metadata=linear.to_dict(),
    )
    project.dataset_manager.add(fit_dataset)
    workbench.figure.add_series(PlotSeries.line(fit_dataset, "x", "y"))

    extracted = project.extract_panel_to_workbench(workbench.id, panel_id)
    extracted_panel_id = extracted.figure.active_panel.id

    extracted_fit_series = next(
        s for s in extracted.figure.series if s.dataset.metadata.get("result_id") == linear.result_id
    )
    copied_result = extracted.analysis_results.current(extracted_panel_id)
    assert extracted_fit_series.dataset.metadata["result_id"] == copied_result.result_id
    assert extracted_fit_series.dataset is fit_dataset  # shared, not duplicated


# --- 1x1 source / double extraction --------------------------------------------


def test_extraction_from_a_1x1_source_workbench_works():
    project = Project.new()
    workbench = project.workbenches[0]
    assert workbench.figure.layout == (1, 1)

    extracted = project.extract_panel_to_workbench(workbench.id, workbench.figure.active_panel.id)

    assert extracted is not None
    assert extracted.id != workbench.id
    assert extracted.figure.active_panel.id != workbench.figure.active_panel.id


def test_extracting_the_same_panel_twice_produces_two_independent_workbenches():
    dataset = _make_dataset()
    project = Project.new()
    project.dataset_manager.add(dataset)
    workbench = project.workbenches[0]
    panel_id = workbench.figure.active_panel.id
    linear = _linear(dataset, panel_id)
    workbench.analysis_results.add(panel_id, linear)

    first = project.extract_panel_to_workbench(workbench.id, panel_id)
    second = project.extract_panel_to_workbench(workbench.id, panel_id)

    assert first.id != second.id
    assert first.figure.active_panel.id != second.figure.active_panel.id
    assert len(project.workbenches) == 3

    # No cross-talk: mutating one extracted copy leaves the other (and the
    # source) untouched.
    first_panel_id = first.figure.active_panel.id
    second_panel_id = second.figure.active_panel.id
    extra = _polynomial(dataset, first_panel_id)
    first.analysis_results.add(first_panel_id, extra)

    assert len(first.analysis_results.all(first_panel_id)) == 2
    assert len(second.analysis_results.all(second_panel_id)) == 1
    assert len(workbench.analysis_results.all(panel_id)) == 1
