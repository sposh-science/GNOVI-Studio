"""End-to-end "Run Fit -> Add Fit Curve to Plot" workflow, driven through
the real `MainWindow` -- the fit curve must join the plot through the exact
same path (undo, styling, dataset registration) any other series does. See
`test_analysis_panel_gui.py` for isolated widget-level coverage of the
Dataset/metadata construction itself.
"""

import pandas as pd

from gnovi_plot.analysis.fitting import LINEAR
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    x = list(range(20))
    y = [3.0 * v + 2.0 for v in x]
    return Dataset(name=name, dataframe=pd.DataFrame({"x": x, "y": y}))


def test_add_fit_curve_to_plot_adds_a_normal_series_and_registers_a_dataset(qapp):
    window = MainWindow()
    window._on_add_to_plot([PlotSeries.line(_make_dataset(), "x", "y", label="Original curve")])
    before_dataset_count = len(window.dataset_manager.datasets)
    before_series_count = len(window.figure_model.series)

    window.tool_drawer._buttons["analysis"].click()
    window.analysis_panel.model_combo.setCurrentIndex(window.analysis_panel.model_combo.findData(LINEAR))
    window.analysis_panel.run_fit_button.click()
    assert len(window.dataset_manager.datasets) == before_dataset_count  # Run Fit alone: no dataset yet

    window.analysis_panel.add_fit_curve_button.click()

    assert len(window.dataset_manager.datasets) == before_dataset_count + 1
    assert len(window.figure_model.series) == before_series_count + 1

    fit_dataset = next(d for d in window.dataset_manager.datasets if d.metadata.get("kind") == "fit")
    assert fit_dataset.metadata["model"] == LINEAR

    new_series = window.figure_model.series[-1]
    assert new_series.dataset is fit_dataset
    window.close()


def test_added_fit_series_is_visible_in_the_series_panel_and_auto_colored(qapp):
    window = MainWindow()
    window._on_add_to_plot([PlotSeries.line(_make_dataset(), "x", "y", label="Original curve")])

    window.tool_drawer._buttons["analysis"].click()
    window.analysis_panel.run_fit_button.click()
    window.analysis_panel.add_fit_curve_button.click()

    # series_panel is refreshed by the same _on_add_to_plot path every other
    # "add a series" action uses -- the fit curve is a completely ordinary
    # entry there, not a special case.
    labels = [
        window.series_panel.series_list.item(i).text()
        for i in range(window.series_panel.series_list.count())
    ]
    assert any("Fit: linear" in label for label in labels)

    new_series = window.figure_model.series[-1]
    assert new_series.color is not None  # auto-assigned from the theme cycle
    window.close()


def test_add_fit_curve_undo_redo_behaves_like_any_other_series_add(qapp):
    window = MainWindow()
    window._on_add_to_plot([PlotSeries.line(_make_dataset(), "x", "y", label="Original curve")])
    window.tool_drawer._buttons["analysis"].click()
    window.analysis_panel.run_fit_button.click()
    window.analysis_panel.add_fit_curve_button.click()

    assert len(window.figure_model.series) == 2

    window._on_undo()
    assert len(window.figure_model.series) == 1
    assert all(s.label != "Fit: linear — y" for s in window.figure_model.series)

    window._on_redo()
    assert len(window.figure_model.series) == 2
    assert any("Fit: linear" in s.label for s in window.figure_model.series)
    window.close()


def test_undo_removes_the_fit_series_but_the_derived_dataset_stays_registered(qapp):
    """Undo/Redo is scoped to the FIGURE only (see gui.undo_manager's own
    docstring) -- exactly like a calculated column surviving a plot-edit
    undo, the derived fit Dataset is not part of that stack. Documented
    behavior, not a bug: Reset/removal of the dataset itself is a separate,
    explicit action a user would take on the Data page."""
    window = MainWindow()
    window._on_add_to_plot([PlotSeries.line(_make_dataset(), "x", "y", label="Original curve")])
    window.tool_drawer._buttons["analysis"].click()
    window.analysis_panel.run_fit_button.click()
    window.analysis_panel.add_fit_curve_button.click()
    dataset_count_after_add = len(window.dataset_manager.datasets)

    window._on_undo()

    assert len(window.dataset_manager.datasets) == dataset_count_after_add
    window.close()


def test_add_fit_curve_disables_after_a_successful_add_and_blocks_a_second_add(qapp):
    """One FitResult.result_id maps to at most one generated PlotSeries in
    its panel -- "Add Fit Curve to Plot" disables itself after a
    successful add and a second click creates no second curve, driven
    end-to-end through the real _on_add_to_plot path this time (contrast
    with the isolated-panel version of this test in
    test_analysis_panel_gui.py). See test_main_window_analysis_workflow_
    gui.py for the full Add/Remove Fit Curve regression coverage."""
    window = MainWindow()
    window._on_add_to_plot([PlotSeries.line(_make_dataset(), "x", "y", label="Original curve")])
    window.tool_drawer._buttons["analysis"].click()
    window.analysis_panel.run_fit_button.click()

    window.analysis_panel.add_fit_curve_button.click()
    assert not window.analysis_panel.add_fit_curve_button.isEnabled()
    assert len(window.figure_model.series) == 2

    window.analysis_panel.add_fit_curve_button.click()  # disabled -- Qt no-op

    assert len(window.figure_model.series) == 2
    assert len(window.dataset_manager.datasets) == 1  # no second derived Dataset
    window.close()


def test_added_to_plot_feedback_is_shown_after_a_successful_add(qapp):
    window = MainWindow()
    window._on_add_to_plot([PlotSeries.line(_make_dataset(), "x", "y", label="Original curve")])
    window.tool_drawer._buttons["analysis"].click()
    window.analysis_panel.run_fit_button.click()

    window.analysis_panel.add_fit_curve_button.click()

    assert window.analysis_panel.added_feedback_label.isVisibleTo(window.analysis_panel)
    assert "Added to plot: Fit: linear — y" == window.analysis_panel.added_feedback_label.text()
    window.close()
