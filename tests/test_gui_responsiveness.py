import pandas as pd
from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLabel, QTableView, QVBoxLayout, QWidget

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.main_window import MainWindow, compute_initial_geometry
from gnovi_plot.gui.widgets.collapsible_section import CollapsibleSection
from gnovi_plot.gui.widgets.dataset_panel import DatasetPanel
from gnovi_plot.plotting.series import PlotType


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return Dataset(name=name, dataframe=df)


# --- compute_initial_geometry -------------------------------------------------


def test_initial_geometry_fits_within_available_for_small_screen():
    available = QRect(0, 0, 1366, 768)
    geometry = compute_initial_geometry(available)

    assert geometry.left() >= available.left()
    assert geometry.top() >= available.top()
    assert geometry.right() <= available.right()
    assert geometry.bottom() <= available.bottom()


def test_initial_geometry_fits_within_available_for_large_screen():
    available = QRect(0, 0, 3840, 2160)
    geometry = compute_initial_geometry(available)

    assert geometry.width() <= available.width()
    assert geometry.height() <= available.height()
    assert geometry.right() <= available.right()
    assert geometry.bottom() <= available.bottom()


def test_initial_geometry_is_centered_and_not_fullscreen():
    available = QRect(0, 0, 1920, 1080)
    geometry = compute_initial_geometry(available)

    assert geometry.width() < available.width()
    assert geometry.height() < available.height()
    assert 0.9 <= geometry.width() / available.width() <= 0.95
    assert 0.9 <= geometry.height() / available.height() <= 0.95
    left_margin = geometry.left() - available.left()
    right_margin = available.right() - geometry.right()
    assert abs(left_margin - right_margin) <= 1


def test_initial_geometry_respects_nonzero_screen_origin():
    available = QRect(100, 50, 1366, 768)
    geometry = compute_initial_geometry(available)

    assert geometry.left() >= available.left()
    assert geometry.top() >= available.top()
    assert geometry.right() <= available.right()
    assert geometry.bottom() <= available.bottom()


def test_main_window_startup_geometry_never_exceeds_screen(qapp):
    window = MainWindow()
    screen = QGuiApplication.primaryScreen()
    available = screen.availableGeometry()

    assert window.geometry().right() <= available.right()
    assert window.geometry().bottom() <= available.bottom()
    assert window.geometry().left() >= available.left()
    assert window.geometry().top() >= available.top()


# --- View menu toggles ---------------------------------------------------------


def test_view_data_preview_action_toggles_right_pane(qapp):
    window = MainWindow()
    window.show()

    assert window.right_scroll.isVisible() is True
    window.toggle_preview_action.setChecked(False)
    assert window.right_scroll.isVisible() is False
    window.toggle_preview_action.setChecked(True)
    assert window.right_scroll.isVisible() is True
    window.close()


def test_view_controls_action_toggles_left_pane(qapp):
    window = MainWindow()
    window.show()

    assert window.left_scroll.isVisible() is True
    window.toggle_controls_action.setChecked(False)
    assert window.left_scroll.isVisible() is False
    window.toggle_controls_action.setChecked(True)
    assert window.left_scroll.isVisible() is True
    window.close()


# --- CollapsibleSection ---------------------------------------------------------


def test_collapsible_section_starts_expanded_by_default(qapp):
    content = QLabel("hello")
    section = CollapsibleSection("Title", content)
    section.show()

    assert section.is_expanded() is True
    assert content.isVisible() is True
    section.close()


def test_collapsible_section_collapse_hides_content_without_destroying_it(qapp):
    content = QWidget()
    layout = QVBoxLayout(content)
    inner_label = QLabel("inner")
    layout.addWidget(inner_label)
    section = CollapsibleSection("Title", content)
    section.show()

    section.set_expanded(False)

    assert section.is_expanded() is False
    assert content.isVisible() is False
    # widget still exists and is still the same object -- not destroyed
    assert section.content is content
    assert inner_label.parent() is content
    section.close()


def test_collapsible_section_expand_restores_content(qapp):
    content = QLabel("hello")
    section = CollapsibleSection("Title", content, expanded=False)
    section.show()

    assert content.isVisible() is False
    section.set_expanded(True)
    assert content.isVisible() is True
    section.close()


def test_collapsible_section_preserves_internal_dynamic_visibility(qapp):
    """Collapsing/expanding a section must not clobber visibility state a
    child widget manages itself (e.g. plot-type dependent controls in
    DatasetPanel)."""
    manager = DatasetManager()
    dataset = _make_dataset()
    manager.add(dataset)
    preview_table = QTableView()
    panel = DatasetPanel(manager, preview_table)
    panel.show()

    hist_index = panel.plot_type_combo.findData(PlotType.HISTOGRAM)
    panel.plot_type_combo.setCurrentIndex(hist_index)
    assert panel.y_combo.isVisible() is False

    panel.plot_section.set_expanded(False)
    panel.plot_section.set_expanded(True)

    # Y column selector must still be hidden -- collapsing/expanding the
    # section must not have reset it to visible.
    assert panel.y_combo.isVisible() is False
    # Widgets that were never plot-type-hidden remain reachable/visible.
    assert panel.import_button.isVisible() is True
    panel.close()
