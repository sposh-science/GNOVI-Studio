from __future__ import annotations

import dataclasses
from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.gui.widgets.active_panel_label import ActivePanelLabel
from gnovi_plot.plotting.figure import GnoviFigure, Panel, Panel3D
from gnovi_plot.plotting.graph_library import GraphLibrary

# Panel dataclass fields excluded from Apply/Cancel/Reset snapshots: `series`
# and `_next_color_index` are content, not display settings (Cancel/Reset
# here must never drop plotted series), `panel_label` is auto-managed by
# GnoviFigure, never user-edited, `source_graph_id` is Graph Library
# provenance (see `Panel.source_graph_id`), never a display setting either --
# Cancel/Reset here must never make a working copy forget which saved Graph
# it originated from -- and `id` is stable panel identity (see `Panel.id`):
# Reset Panel to Defaults must never reassign it, or a panel's own
# analysis-result history would be silently orphaned by an unrelated
# "reset formatting" action.
_PANEL_SNAPSHOT_EXCLUDED_FIELDS = {"series", "_next_color_index", "panel_label", "source_graph_id", "id"}

_LEGEND_LOCATIONS = [
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
    # Placed via `bbox_to_anchor` rather than a real Matplotlib `loc` --
    # see plotting.backends.matplotlib_backend.render_panel's legend block.
    "outside right",
    "outside bottom",
]

# A trimmed copy of `_LEGEND_LOCATIONS` for `Panel3D`'s own minimal legend
# (see that class's own docstring) -- excludes "outside right"/"outside
# bottom", which need the `bbox_to_anchor` placement logic `_legend_kwargs`
# implements for 2D; the 3D legend renderer deliberately doesn't reproduce
# that (see `matplotlib_backend.render_panel_3d`'s own legend block).
_LEGEND_LOCATIONS_3D = [
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]

_SCALE_OPTIONS = [("Linear", "linear"), ("Log", "log")]
_TICK_DIRECTIONS = [("Out", "out"), ("In", "in"), ("In & out", "inout")]
_GRID_WHICH_OPTIONS = [("Major", "major"), ("Minor", "minor"), ("Both", "both")]
_GRID_STYLE_OPTIONS = [
    ("Solid", "-"),
    ("Dashed", "--"),
    ("Dotted", ":"),
    ("Dash-dot", "-."),
]

_LIMIT_RANGE = 1e12
_TICK_SPACING_RANGE = 1e9
_TICK_GEOMETRY_RANGE = 100.0
_DEFAULT_GRID_COLOR = "#b0b0b0"
_DEFAULT_PANE_COLOR = "#ffffff"

_ASPECT_OPTIONS = [("Auto", "auto"), ("Equal", "equal")]

# Figure-wide (not per-panel) grid-appearance fields this panel also edits --
# see the "Grid" QGroupBox below. Unlike the axis/tick/spine/legend fields
# above, these live on GnoviFigure itself (every panel with its grid on
# shares one consistent style), so they're captured/restored/reset
# separately from the Panel snapshot in `capture_state`/`restore_state`/
# `reset_to_defaults`.
_FIGURE_GRID_FIELDS = ["grid_linestyle", "grid_linewidth", "grid_alpha", "grid_color"]

# mplot3d's own `Axes3D.view_init()` defaults -- see `Panel3D`'s own
# docstring. "Reset View" restores exactly these, read off a fresh
# `Panel3D()` rather than hardcoded here, so it can never drift from the
# dataclass's own defaults.
_PANEL3D_DEFAULTS = Panel3D()


class FigurePropertiesPanel(QWidget):
    """Title/axis/tick/spine/grid/legend controls for a GnoviFigure's
    currently active Panel, or (when the active panel is a `Panel3D`)
    title/X-Y-Z labels/limits/grid/camera controls for it instead.

    Every 2D field here reads/writes `figure.active_panel` when it's a
    `Panel`; every 3D field reads/writes it when it's a `Panel3D`. Switching
    the active panel (multi-panel layouts) just requires calling
    `refresh()` again -- it picks the right internal page AND reloads the
    widgets for the newly active panel's actual type.

    Internally a `QStackedWidget` of two pages, mirroring
    `plot_series_panel.PlotSeriesPanel`'s own adaptive-page pattern (see
    that class's docstring for why): every EXISTING widget attribute this
    class exposed before 3D support (`title_edit`, `x_min_spin`, etc.)
    stays a direct attribute of this outer object, unmoved -- only which
    container widget currently parents it changed.

    `changed` is emitted after every mutation so the owner can re-render.
    """

    changed = Signal()
    # "Set Current View" needs the LIVE rendered Axes3D's current elev/azim
    # -- something only `gui.widgets.plot_canvas.PlotCanvas` holds, not this
    # panel (which only ever reads/writes the declarative `Panel3D` model) --
    # so committing it is delegated to the owner (`MainWindow`) via this
    # signal, exactly like `set_figure`/`refresh` keep this panel itself
    # model-only. See `MainWindow._on_set_current_3d_view_requested`.
    set_current_view_requested = Signal()

    def __init__(
        self,
        figure: GnoviFigure,
        get_graph_library: Callable[[], GraphLibrary] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._figure = figure
        self._updating = False

        self.active_panel_label = ActivePanelLabel(figure, get_graph_library)

        self._stack = QStackedWidget()
        self._page_2d = self._build_2d_page()
        self._page_3d = self._build_3d_page()
        self._stack.addWidget(self._page_2d)
        self._stack.addWidget(self._page_3d)

        layout = QVBoxLayout(self)
        layout.addWidget(self.active_panel_label)
        layout.addWidget(self._stack)
        layout.addStretch(1)

        self.refresh()

    # --- 2D page (Panel) -----------------------------------------------------

    def _build_2d_page(self) -> QWidget:
        # --- Labels ---
        self.title_edit = QLineEdit()
        self.xlabel_edit = QLineEdit()
        self.ylabel_edit = QLineEdit()

        # --- Limits & scale ---
        self.x_manual_check = QCheckBox("Manual X limits")
        self.x_min_spin = self._make_limit_spin()
        self.x_max_spin = self._make_limit_spin()
        self.x_scale_combo = self._make_option_combo(_SCALE_OPTIONS)
        self.x_invert_check = QCheckBox("Invert X axis")

        self.y_manual_check = QCheckBox("Manual Y limits")
        self.y_min_spin = self._make_limit_spin()
        self.y_max_spin = self._make_limit_spin()
        self.y_scale_combo = self._make_option_combo(_SCALE_OPTIONS)
        self.y_invert_check = QCheckBox("Invert Y axis")

        self.reset_limits_button = QPushButton("Reset / Auto Limits")

        # --- Ticks ---
        self.tick_direction_combo = self._make_option_combo(_TICK_DIRECTIONS)
        self.minor_ticks_check = QCheckBox("Minor ticks")
        self.major_spacing_x_spin = self._make_spacing_spin()
        self.major_spacing_y_spin = self._make_spacing_spin()
        self.minor_spacing_x_spin = self._make_spacing_spin()
        self.minor_spacing_y_spin = self._make_spacing_spin()
        self.major_tick_length_spin = self._make_tick_geometry_spin()
        self.major_tick_width_spin = self._make_tick_geometry_spin()
        self.minor_tick_length_spin = self._make_tick_geometry_spin()
        self.minor_tick_width_spin = self._make_tick_geometry_spin()
        self.sci_notation_x_check = QCheckBox("Scientific notation (X)")
        self.sci_notation_y_check = QCheckBox("Scientific notation (Y)")

        # --- Spines ---
        self.spine_top_check = QCheckBox("Top")
        self.spine_bottom_check = QCheckBox("Bottom")
        self.spine_left_check = QCheckBox("Left")
        self.spine_right_check = QCheckBox("Right")
        self.spine_width_spin = QDoubleSpinBox()
        self.spine_width_spin.setRange(0.1, 10.0)
        self.spine_width_spin.setSingleStep(0.1)

        # --- Grid (single authoritative location for every grid control --
        # on/off and "which" are per-panel, style/width/opacity/color are
        # figure-wide; see `_FIGURE_GRID_FIELDS` above) ---
        self.grid_check = QCheckBox("Show grid")
        self.grid_which_combo = self._make_option_combo(_GRID_WHICH_OPTIONS)
        self.grid_style_combo = QComboBox()
        for text, code in _GRID_STYLE_OPTIONS:
            self.grid_style_combo.addItem(text, code)
        self.grid_width_spin = QDoubleSpinBox()
        self.grid_width_spin.setRange(0.1, 10.0)
        self.grid_width_spin.setSingleStep(0.1)
        self.grid_alpha_spin = QDoubleSpinBox()
        self.grid_alpha_spin.setRange(0.05, 1.0)
        self.grid_alpha_spin.setSingleStep(0.05)
        self.grid_custom_color_check = QCheckBox("Custom grid color")
        self.grid_color_button = QPushButton()
        self.grid_color_button.setFixedWidth(48)

        # --- Legend ---
        self.legend_check = QCheckBox("Show legend")
        self.legend_loc_combo = QComboBox()
        self.legend_loc_combo.addItems(_LEGEND_LOCATIONS)
        # The longest Matplotlib location strings ("outside bottom" etc.)
        # were driving this combo's own natural minimumSizeHint -- see the
        # matching note on `figure_size_panel.font_family_combo` for the
        # same pattern.
        self.legend_loc_combo.setMinimumWidth(90)
        self.legend_ncol_spin = QSpinBox()
        self.legend_ncol_spin.setRange(1, 10)
        self.legend_frame_check = QCheckBox("Legend frame")
        self.legend_title_edit = QLineEdit()

        labels_group = QGroupBox("Labels")
        labels_form = QFormLayout(labels_group)
        labels_form.addRow("Title", self.title_edit)
        labels_form.addRow("X label", self.xlabel_edit)
        labels_form.addRow("Y label", self.ylabel_edit)

        limits_group = QGroupBox("Axis Limits & Scale")
        limits_layout = QVBoxLayout(limits_group)
        limits_layout.addWidget(self.x_manual_check)
        x_row = QHBoxLayout()
        x_row.addWidget(self.x_min_spin)
        x_row.addWidget(self.x_max_spin)
        limits_layout.addLayout(x_row)
        x_scale_row = QHBoxLayout()
        x_scale_row.addWidget(self.x_scale_combo)
        x_scale_row.addWidget(self.x_invert_check)
        limits_layout.addLayout(x_scale_row)
        limits_layout.addWidget(self.y_manual_check)
        y_row = QHBoxLayout()
        y_row.addWidget(self.y_min_spin)
        y_row.addWidget(self.y_max_spin)
        limits_layout.addLayout(y_row)
        y_scale_row = QHBoxLayout()
        y_scale_row.addWidget(self.y_scale_combo)
        y_scale_row.addWidget(self.y_invert_check)
        limits_layout.addLayout(y_scale_row)
        limits_layout.addWidget(self.reset_limits_button)

        ticks_group = QGroupBox("Ticks")
        ticks_form = QFormLayout(ticks_group)
        ticks_form.addRow("Direction", self.tick_direction_combo)
        # "(0 = auto)" dropped from these four labels -- redundant with
        # `_make_spacing_spin`'s own `setSpecialValueText("Auto")` below,
        # which already shows "Auto" directly in the field itself at 0; the
        # longest label in the left drawer, driving its default width more
        # than any single field ever did (see gui.main_window's
        # `_side_drawer_min_width`).
        ticks_form.addRow("Major spacing X", self.major_spacing_x_spin)
        ticks_form.addRow("Major spacing Y", self.major_spacing_y_spin)
        ticks_form.addRow("Major length", self.major_tick_length_spin)
        ticks_form.addRow("Major width", self.major_tick_width_spin)
        ticks_form.addRow(self.minor_ticks_check)
        ticks_form.addRow("Minor spacing X", self.minor_spacing_x_spin)
        ticks_form.addRow("Minor spacing Y", self.minor_spacing_y_spin)
        ticks_form.addRow("Minor length", self.minor_tick_length_spin)
        ticks_form.addRow("Minor width", self.minor_tick_width_spin)
        ticks_form.addRow(self.sci_notation_x_check)
        ticks_form.addRow(self.sci_notation_y_check)

        spines_group = QGroupBox("Spines")
        spines_layout = QVBoxLayout(spines_group)
        spines_row = QHBoxLayout()
        spines_row.addWidget(self.spine_top_check)
        spines_row.addWidget(self.spine_bottom_check)
        spines_row.addWidget(self.spine_left_check)
        spines_row.addWidget(self.spine_right_check)
        spines_layout.addLayout(spines_row)
        spine_width_form = QFormLayout()
        spine_width_form.addRow("Line width", self.spine_width_spin)
        spines_layout.addLayout(spine_width_form)

        grid_group = QGroupBox("Grid")
        grid_form = QFormLayout(grid_group)
        grid_form.addRow(self.grid_check)
        grid_form.addRow("Grid lines", self.grid_which_combo)
        grid_form.addRow("Style", self.grid_style_combo)
        grid_form.addRow("Line width", self.grid_width_spin)
        grid_form.addRow("Grid opacity", self.grid_alpha_spin)
        grid_form.addRow(self.grid_custom_color_check)
        grid_form.addRow("Color", self.grid_color_button)

        legend_group = QGroupBox("Legend")
        legend_form = QFormLayout(legend_group)
        legend_form.addRow(self.legend_check)
        legend_form.addRow("Legend location", self.legend_loc_combo)
        legend_form.addRow("Legend columns", self.legend_ncol_spin)
        legend_form.addRow(self.legend_frame_check)
        legend_form.addRow("Legend title", self.legend_title_edit)

        self.reset_all_button = QPushButton("Reset to Defaults")

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(labels_group)
        layout.addWidget(limits_group)
        layout.addWidget(ticks_group)
        layout.addWidget(spines_group)
        layout.addWidget(grid_group)
        layout.addWidget(legend_group)
        layout.addWidget(self.reset_all_button)
        layout.addStretch(1)

        self.title_edit.editingFinished.connect(self._apply_title)
        self.xlabel_edit.editingFinished.connect(self._apply_xlabel)
        self.ylabel_edit.editingFinished.connect(self._apply_ylabel)
        self.x_manual_check.toggled.connect(self._apply_x_manual)
        self.x_min_spin.valueChanged.connect(self._apply_x_limits)
        self.x_max_spin.valueChanged.connect(self._apply_x_limits)
        self.x_scale_combo.currentIndexChanged.connect(self._apply_x_scale)
        self.x_invert_check.toggled.connect(self._apply_x_invert)
        self.y_manual_check.toggled.connect(self._apply_y_manual)
        self.y_min_spin.valueChanged.connect(self._apply_y_limits)
        self.y_max_spin.valueChanged.connect(self._apply_y_limits)
        self.y_scale_combo.currentIndexChanged.connect(self._apply_y_scale)
        self.y_invert_check.toggled.connect(self._apply_y_invert)
        self.reset_limits_button.clicked.connect(self._on_reset_limits)

        self.tick_direction_combo.currentIndexChanged.connect(self._apply_tick_direction)
        self.minor_ticks_check.toggled.connect(self._apply_minor_ticks)
        self.major_spacing_x_spin.valueChanged.connect(self._apply_major_spacing_x)
        self.major_spacing_y_spin.valueChanged.connect(self._apply_major_spacing_y)
        self.minor_spacing_x_spin.valueChanged.connect(self._apply_minor_spacing_x)
        self.minor_spacing_y_spin.valueChanged.connect(self._apply_minor_spacing_y)
        self.major_tick_length_spin.valueChanged.connect(self._apply_major_tick_length)
        self.major_tick_width_spin.valueChanged.connect(self._apply_major_tick_width)
        self.minor_tick_length_spin.valueChanged.connect(self._apply_minor_tick_length)
        self.minor_tick_width_spin.valueChanged.connect(self._apply_minor_tick_width)
        self.sci_notation_x_check.toggled.connect(self._apply_sci_x)
        self.sci_notation_y_check.toggled.connect(self._apply_sci_y)

        self.spine_top_check.toggled.connect(self._apply_spines)
        self.spine_bottom_check.toggled.connect(self._apply_spines)
        self.spine_left_check.toggled.connect(self._apply_spines)
        self.spine_right_check.toggled.connect(self._apply_spines)
        self.spine_width_spin.valueChanged.connect(self._apply_spines)

        self.grid_check.toggled.connect(self._apply_grid)
        self.grid_which_combo.currentIndexChanged.connect(self._apply_grid_which)
        self.grid_style_combo.currentIndexChanged.connect(self._apply_grid_style)
        self.grid_width_spin.valueChanged.connect(self._apply_grid_width)
        self.grid_alpha_spin.valueChanged.connect(self._apply_grid_alpha)
        self.grid_custom_color_check.toggled.connect(self._apply_grid_custom_color_toggled)
        self.grid_color_button.clicked.connect(self._pick_grid_color)
        self.legend_check.toggled.connect(self._apply_legend_visible)
        self.legend_loc_combo.currentTextChanged.connect(self._apply_legend_loc)
        self.legend_ncol_spin.valueChanged.connect(self._apply_legend_ncol)
        self.legend_frame_check.toggled.connect(self._apply_legend_frame)
        self.legend_title_edit.editingFinished.connect(self._apply_legend_title)
        self.reset_all_button.clicked.connect(self.reset_to_defaults)

        return page

    # --- 3D page (Panel3D) ----------------------------------------------------

    def _build_3d_page(self) -> QWidget:
        """`Panel3D`'s field set (see that class's own docstring): title/
        X-Y-Z labels, X/Y/Z limits, scientific (data) aspect, X/Y/Z tick
        spacing, grid style, panes, legend (visible/location/columns/
        frame), and the camera -- no tick width/length/direction, no
        per-axis grid visibility/major-minor distinction, no axis-line
        color, none of which is reliable or has a real API surface in
        this Matplotlib version (see `Panel3D`'s own docstring and
        `matplotlib_backend._apply_3d_grid_style`'s docstring for exactly
        what was verified and why each is excluded)."""
        self.d3_title_edit = QLineEdit()
        self.d3_xlabel_edit = QLineEdit()
        self.d3_ylabel_edit = QLineEdit()
        self.d3_zlabel_edit = QLineEdit()

        self.d3_x_manual_check = QCheckBox("Manual X limits")
        self.d3_x_min_spin = self._make_limit_spin()
        self.d3_x_max_spin = self._make_limit_spin()
        self.d3_y_manual_check = QCheckBox("Manual Y limits")
        self.d3_y_min_spin = self._make_limit_spin()
        self.d3_y_max_spin = self._make_limit_spin()
        self.d3_z_manual_check = QCheckBox("Manual Z limits")
        self.d3_z_min_spin = self._make_limit_spin()
        self.d3_z_max_spin = self._make_limit_spin()
        self.d3_reset_limits_button = QPushButton("Reset / Auto Limits")

        # --- Scientific Aspect -- data/coordinate scaling, NOT panel
        # shape (`set_box_aspect`, untouched) and NOT camera (`view_init`,
        # unaffected) -- see `Panel3D.aspect_mode`'s own docstring.
        self.d3_aspect_combo = self._make_option_combo(_ASPECT_OPTIONS)

        # --- Ticks -- major/minor spacing only, per axis; no width/length/
        # direction (verified unreliable/non-functional for Axes3D in this
        # Matplotlib version -- see `Panel3D`'s own docstring).
        self.d3_major_spacing_x_spin = self._make_spacing_spin()
        self.d3_major_spacing_y_spin = self._make_spacing_spin()
        self.d3_major_spacing_z_spin = self._make_spacing_spin()
        self.d3_minor_spacing_x_spin = self._make_spacing_spin()
        self.d3_minor_spacing_y_spin = self._make_spacing_spin()
        self.d3_minor_spacing_z_spin = self._make_spacing_spin()

        # --- Grid -- style is applied via a single isolated, private-API
        # -backed helper (`matplotlib_backend._apply_3d_grid_style`); see
        # that function's own docstring for why and how it fails
        # gracefully. Same widget pattern as the 2D Grid group.
        self.d3_grid_check = QCheckBox("Show grid")
        self.d3_grid_style_combo = QComboBox()
        for text, code in _GRID_STYLE_OPTIONS:
            self.d3_grid_style_combo.addItem(text, code)
        self.d3_grid_width_spin = QDoubleSpinBox()
        self.d3_grid_width_spin.setRange(0.1, 10.0)
        self.d3_grid_width_spin.setSingleStep(0.1)
        self.d3_grid_alpha_spin = QDoubleSpinBox()
        self.d3_grid_alpha_spin.setRange(0.05, 1.0)
        self.d3_grid_alpha_spin.setSingleStep(0.05)
        self.d3_grid_custom_color_check = QCheckBox("Custom grid color")
        self.d3_grid_color_button = QPushButton()
        self.d3_grid_color_button.setFixedWidth(48)

        # --- Panes -- fully public `Axis.pane` (`Polygon`) API, see
        # `Panel3D.pane_visible`/`.pane_color`/`.pane_alpha`'s own
        # docstring. No axis-line color in this PR (lower value, shares
        # grid's private-API risk profile -- deferred).
        self.d3_pane_visible_check = QCheckBox("Show panes")
        self.d3_pane_custom_color_check = QCheckBox("Custom pane color")
        self.d3_pane_color_button = QPushButton()
        self.d3_pane_color_button.setFixedWidth(48)
        self.d3_pane_alpha_spin = QDoubleSpinBox()
        self.d3_pane_alpha_spin.setRange(0.05, 1.0)
        self.d3_pane_alpha_spin.setSingleStep(0.05)

        # --- Legend -- adds columns/frame over the prior milestone's
        # visible/location only; still no title/per-panel font-size
        # override/outside-placement (see `Panel3D`'s own docstring).
        self.d3_legend_check = QCheckBox("Show legend")
        self.d3_legend_loc_combo = QComboBox()
        self.d3_legend_loc_combo.addItems(_LEGEND_LOCATIONS_3D)
        self.d3_legend_ncol_spin = QSpinBox()
        self.d3_legend_ncol_spin.setRange(1, 10)
        self.d3_legend_frame_check = QCheckBox("Legend frame")

        self.d3_elevation_spin = QDoubleSpinBox()
        self.d3_elevation_spin.setRange(-180.0, 180.0)
        self.d3_azimuth_spin = QDoubleSpinBox()
        self.d3_azimuth_spin.setRange(-360.0, 360.0)
        self.d3_set_current_view_button = QPushButton("Set Current View")
        self.d3_reset_view_button = QPushButton("Reset View")

        labels_group = QGroupBox("Labels")
        labels_form = QFormLayout(labels_group)
        labels_form.addRow("Title", self.d3_title_edit)
        labels_form.addRow("X label", self.d3_xlabel_edit)
        labels_form.addRow("Y label", self.d3_ylabel_edit)
        labels_form.addRow("Z label", self.d3_zlabel_edit)

        limits_group = QGroupBox("Limits")
        limits_layout = QVBoxLayout(limits_group)
        limits_layout.addWidget(self.d3_x_manual_check)
        x_row = QHBoxLayout()
        x_row.addWidget(self.d3_x_min_spin)
        x_row.addWidget(self.d3_x_max_spin)
        limits_layout.addLayout(x_row)
        limits_layout.addWidget(self.d3_y_manual_check)
        y_row = QHBoxLayout()
        y_row.addWidget(self.d3_y_min_spin)
        y_row.addWidget(self.d3_y_max_spin)
        limits_layout.addLayout(y_row)
        limits_layout.addWidget(self.d3_z_manual_check)
        z_row = QHBoxLayout()
        z_row.addWidget(self.d3_z_min_spin)
        z_row.addWidget(self.d3_z_max_spin)
        limits_layout.addLayout(z_row)
        limits_layout.addWidget(self.d3_reset_limits_button)

        aspect_group = QGroupBox("Scientific Aspect")
        aspect_form = QFormLayout(aspect_group)
        aspect_form.addRow("Aspect", self.d3_aspect_combo)

        ticks_group = QGroupBox("Ticks")
        ticks_form = QFormLayout(ticks_group)
        ticks_form.addRow("Major spacing X", self.d3_major_spacing_x_spin)
        ticks_form.addRow("Major spacing Y", self.d3_major_spacing_y_spin)
        ticks_form.addRow("Major spacing Z", self.d3_major_spacing_z_spin)
        ticks_form.addRow("Minor spacing X", self.d3_minor_spacing_x_spin)
        ticks_form.addRow("Minor spacing Y", self.d3_minor_spacing_y_spin)
        ticks_form.addRow("Minor spacing Z", self.d3_minor_spacing_z_spin)

        grid_group = QGroupBox("Grid")
        grid_form = QFormLayout(grid_group)
        grid_form.addRow(self.d3_grid_check)
        grid_form.addRow("Style", self.d3_grid_style_combo)
        grid_form.addRow("Line width", self.d3_grid_width_spin)
        grid_form.addRow("Grid opacity", self.d3_grid_alpha_spin)
        grid_form.addRow(self.d3_grid_custom_color_check)
        grid_form.addRow("Color", self.d3_grid_color_button)

        panes_group = QGroupBox("Panes")
        panes_form = QFormLayout(panes_group)
        panes_form.addRow(self.d3_pane_visible_check)
        panes_form.addRow(self.d3_pane_custom_color_check)
        panes_form.addRow("Color", self.d3_pane_color_button)
        panes_form.addRow("Pane opacity", self.d3_pane_alpha_spin)

        legend_group = QGroupBox("Legend")
        legend_form = QFormLayout(legend_group)
        legend_form.addRow(self.d3_legend_check)
        legend_form.addRow("Legend location", self.d3_legend_loc_combo)
        legend_form.addRow("Legend columns", self.d3_legend_ncol_spin)
        legend_form.addRow(self.d3_legend_frame_check)

        view_group = QGroupBox("View / Camera")
        view_form = QFormLayout(view_group)
        view_form.addRow("Elevation", self.d3_elevation_spin)
        view_form.addRow("Azimuth", self.d3_azimuth_spin)
        view_buttons = QHBoxLayout()
        view_buttons.addWidget(self.d3_set_current_view_button)
        view_buttons.addWidget(self.d3_reset_view_button)
        view_form.addRow(view_buttons)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(labels_group)
        layout.addWidget(limits_group)
        layout.addWidget(aspect_group)
        layout.addWidget(ticks_group)
        layout.addWidget(grid_group)
        layout.addWidget(panes_group)
        layout.addWidget(legend_group)
        layout.addWidget(view_group)
        layout.addStretch(1)

        self.d3_title_edit.editingFinished.connect(self._apply_3d_title)
        self.d3_xlabel_edit.editingFinished.connect(self._apply_3d_xlabel)
        self.d3_ylabel_edit.editingFinished.connect(self._apply_3d_ylabel)
        self.d3_zlabel_edit.editingFinished.connect(self._apply_3d_zlabel)
        self.d3_x_manual_check.toggled.connect(self._apply_3d_x_manual)
        self.d3_x_min_spin.valueChanged.connect(self._apply_3d_x_limits)
        self.d3_x_max_spin.valueChanged.connect(self._apply_3d_x_limits)
        self.d3_y_manual_check.toggled.connect(self._apply_3d_y_manual)
        self.d3_y_min_spin.valueChanged.connect(self._apply_3d_y_limits)
        self.d3_y_max_spin.valueChanged.connect(self._apply_3d_y_limits)
        self.d3_z_manual_check.toggled.connect(self._apply_3d_z_manual)
        self.d3_z_min_spin.valueChanged.connect(self._apply_3d_z_limits)
        self.d3_z_max_spin.valueChanged.connect(self._apply_3d_z_limits)
        self.d3_reset_limits_button.clicked.connect(self._on_reset_3d_limits)
        self.d3_aspect_combo.currentIndexChanged.connect(self._apply_3d_aspect)
        self.d3_major_spacing_x_spin.valueChanged.connect(self._apply_3d_major_spacing_x)
        self.d3_major_spacing_y_spin.valueChanged.connect(self._apply_3d_major_spacing_y)
        self.d3_major_spacing_z_spin.valueChanged.connect(self._apply_3d_major_spacing_z)
        self.d3_minor_spacing_x_spin.valueChanged.connect(self._apply_3d_minor_spacing_x)
        self.d3_minor_spacing_y_spin.valueChanged.connect(self._apply_3d_minor_spacing_y)
        self.d3_minor_spacing_z_spin.valueChanged.connect(self._apply_3d_minor_spacing_z)
        self.d3_grid_check.toggled.connect(self._apply_3d_grid)
        self.d3_grid_style_combo.currentIndexChanged.connect(self._apply_3d_grid_style)
        self.d3_grid_width_spin.valueChanged.connect(self._apply_3d_grid_width)
        self.d3_grid_alpha_spin.valueChanged.connect(self._apply_3d_grid_alpha)
        self.d3_grid_custom_color_check.toggled.connect(self._apply_3d_grid_custom_color_toggled)
        self.d3_grid_color_button.clicked.connect(self._pick_3d_grid_color)
        self.d3_pane_visible_check.toggled.connect(self._apply_3d_pane_visible)
        self.d3_pane_custom_color_check.toggled.connect(self._apply_3d_pane_custom_color_toggled)
        self.d3_pane_color_button.clicked.connect(self._pick_3d_pane_color)
        self.d3_pane_alpha_spin.valueChanged.connect(self._apply_3d_pane_alpha)
        self.d3_legend_check.toggled.connect(self._apply_3d_legend_visible)
        self.d3_legend_loc_combo.currentTextChanged.connect(self._apply_3d_legend_loc)
        self.d3_legend_ncol_spin.valueChanged.connect(self._apply_3d_legend_ncol)
        self.d3_legend_frame_check.toggled.connect(self._apply_3d_legend_frame)
        self.d3_elevation_spin.valueChanged.connect(self._apply_3d_elevation)
        self.d3_azimuth_spin.valueChanged.connect(self._apply_3d_azimuth)
        self.d3_set_current_view_button.clicked.connect(self.set_current_view_requested)
        self.d3_reset_view_button.clicked.connect(self._on_reset_3d_view)

        return page

    @staticmethod
    def _make_limit_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-_LIMIT_RANGE, _LIMIT_RANGE)
        spin.setDecimals(6)
        spin.setEnabled(False)
        # Qt's own automatic minimumSizeHint scales with the *widest value
        # the configured range could show* (see the QSpinBox/QDoubleSpinBox
        # max-width comment in gui.styles) -- for `±_LIMIT_RANGE` (1e12) at
        # 6 decimals that is a ~20-character string, even though real axis
        # limits are almost always short. Two of these sit side by side in
        # one row (X/Y min+max), so this one field's inflated natural
        # minimum alone was the single largest driver of the whole left
        # drawer's default width. A typed value far past this width still
        # scrolls within the field normally; this only overrides the
        # unrealistic auto-sized floor, not what the field can hold.
        spin.setMinimumWidth(80)
        return spin

    @staticmethod
    def _make_spacing_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, _TICK_SPACING_RANGE)
        spin.setDecimals(6)
        spin.setSpecialValueText("Auto")
        # Same "configured range inflated the natural minimum" pattern as
        # `_make_limit_spin` above (`_TICK_SPACING_RANGE` is 1e9 at 6
        # decimals -- a ~16-character string) -- four of these
        # (major/minor spacing x/y) sit in the Axes drawer page, so this
        # alone was a real contributor to that page's default width.
        spin.setMinimumWidth(80)
        return spin

    @staticmethod
    def _make_tick_geometry_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, _TICK_GEOMETRY_RANGE)
        spin.setSingleStep(0.5)
        spin.setDecimals(2)
        return spin

    @staticmethod
    def _make_option_combo(options: list[tuple[str, str]]) -> QComboBox:
        combo = QComboBox()
        for text, code in options:
            combo.addItem(text, code)
        return combo

    @property
    def _panel(self):
        return self._figure.active_panel

    def set_figure(self, figure: GnoviFigure) -> None:
        """Repoint this panel at a different `GnoviFigure` (e.g. after
        Open/New Project swaps the active figure) and reload from it."""
        self._figure = figure
        self.refresh()

    def refresh(self) -> None:
        """Reload every field from `figure.active_panel`. Call this after
        switching the active panel, in addition to after construction."""
        self._sync_from_figure()

    def _sync_from_figure(self) -> None:
        panel = self._panel
        self.active_panel_label.refresh(self._figure)
        if isinstance(panel, Panel3D):
            self._stack.setCurrentWidget(self._page_3d)
            self._sync_3d_from_figure(panel)
        else:
            self._stack.setCurrentWidget(self._page_2d)
            self._sync_2d_from_figure(panel)

    def _sync_2d_from_figure(self, panel: Panel) -> None:
        self._updating = True
        self.title_edit.setText(panel.title)
        self.xlabel_edit.setText(panel.xlabel)
        self.ylabel_edit.setText(panel.ylabel)

        self.x_manual_check.setChecked(panel.xlim is not None)
        self.x_min_spin.setEnabled(panel.xlim is not None)
        self.x_max_spin.setEnabled(panel.xlim is not None)
        if panel.xlim is not None:
            self.x_min_spin.setValue(panel.xlim[0])
            self.x_max_spin.setValue(panel.xlim[1])
        self.x_scale_combo.setCurrentIndex(max(self.x_scale_combo.findData(panel.xscale), 0))
        self.x_invert_check.setChecked(panel.invert_x)

        self.y_manual_check.setChecked(panel.ylim is not None)
        self.y_min_spin.setEnabled(panel.ylim is not None)
        self.y_max_spin.setEnabled(panel.ylim is not None)
        if panel.ylim is not None:
            self.y_min_spin.setValue(panel.ylim[0])
            self.y_max_spin.setValue(panel.ylim[1])
        self.y_scale_combo.setCurrentIndex(max(self.y_scale_combo.findData(panel.yscale), 0))
        self.y_invert_check.setChecked(panel.invert_y)

        self.tick_direction_combo.setCurrentIndex(
            max(self.tick_direction_combo.findData(panel.tick_direction), 0)
        )
        self.minor_ticks_check.setChecked(panel.minor_ticks)
        self.major_spacing_x_spin.setValue(panel.major_tick_spacing_x or 0.0)
        self.major_spacing_y_spin.setValue(panel.major_tick_spacing_y or 0.0)
        self.minor_spacing_x_spin.setValue(panel.minor_tick_spacing_x or 0.0)
        self.minor_spacing_y_spin.setValue(panel.minor_tick_spacing_y or 0.0)
        self.major_tick_length_spin.setValue(panel.major_tick_length)
        self.major_tick_width_spin.setValue(panel.major_tick_width)
        self.minor_tick_length_spin.setValue(panel.minor_tick_length)
        self.minor_tick_width_spin.setValue(panel.minor_tick_width)
        self.sci_notation_x_check.setChecked(panel.scientific_notation_x)
        self.sci_notation_y_check.setChecked(panel.scientific_notation_y)

        self.spine_top_check.setChecked(panel.spine_top)
        self.spine_bottom_check.setChecked(panel.spine_bottom)
        self.spine_left_check.setChecked(panel.spine_left)
        self.spine_right_check.setChecked(panel.spine_right)
        self.spine_width_spin.setValue(panel.spine_linewidth)

        self.grid_check.setChecked(panel.grid)
        self.grid_which_combo.setCurrentIndex(max(self.grid_which_combo.findData(panel.grid_which), 0))
        self.grid_style_combo.setCurrentIndex(
            max(self.grid_style_combo.findData(self._figure.grid_linestyle), 0)
        )
        self.grid_width_spin.setValue(self._figure.grid_linewidth)
        self.grid_alpha_spin.setValue(self._figure.grid_alpha)
        has_custom_grid_color = self._figure.grid_color is not None
        self.grid_custom_color_check.setChecked(has_custom_grid_color)
        self.grid_color_button.setEnabled(has_custom_grid_color)
        self._set_grid_color_swatch(self._figure.grid_color or _DEFAULT_GRID_COLOR)
        self.legend_check.setChecked(panel.legend_visible)
        self.legend_loc_combo.setCurrentText(panel.legend_loc)
        self.legend_ncol_spin.setValue(panel.legend_ncol)
        self.legend_frame_check.setChecked(panel.legend_frameon)
        self.legend_title_edit.setText(panel.legend_title)
        self._updating = False

    def sync_axes_limits(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        """Reflect the canvas' live (auto-computed) limits into the disabled
        spin boxes, so they show sensible values before the user goes
        manual. 2D panels only -- a `Panel3D`'s Z limit has no 2D-canvas
        equivalent to read live values from, and `MainWindow._rerender`
        only ever calls this with a 2D Axes' own xlim/ylim in the first
        place."""
        if not isinstance(self._panel, Panel):
            return
        panel = self._panel
        self._updating = True
        if panel.xlim is None:
            self.x_min_spin.setValue(xlim[0])
            self.x_max_spin.setValue(xlim[1])
        if panel.ylim is None:
            self.y_min_spin.setValue(ylim[0])
            self.y_max_spin.setValue(ylim[1])
        self._updating = False

    def _apply_title(self) -> None:
        if self._updating:
            return
        self._panel.title = self.title_edit.text()
        self.changed.emit()

    def _apply_xlabel(self) -> None:
        if self._updating:
            return
        self._panel.xlabel = self.xlabel_edit.text()
        self.changed.emit()

    def _apply_ylabel(self) -> None:
        if self._updating:
            return
        self._panel.ylabel = self.ylabel_edit.text()
        self.changed.emit()

    def _apply_x_manual(self, checked: bool) -> None:
        self.x_min_spin.setEnabled(checked)
        self.x_max_spin.setEnabled(checked)
        if self._updating:
            return
        self._panel.xlim = (self.x_min_spin.value(), self.x_max_spin.value()) if checked else None
        self.changed.emit()

    def _apply_x_limits(self, _value: float) -> None:
        if self._updating or not self.x_manual_check.isChecked():
            return
        self._panel.xlim = (self.x_min_spin.value(), self.x_max_spin.value())
        self.changed.emit()

    def _apply_x_scale(self, _index: int) -> None:
        if self._updating:
            return
        self._panel.xscale = self.x_scale_combo.currentData()
        self.changed.emit()

    def _apply_x_invert(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.invert_x = checked
        self.changed.emit()

    def _apply_y_manual(self, checked: bool) -> None:
        self.y_min_spin.setEnabled(checked)
        self.y_max_spin.setEnabled(checked)
        if self._updating:
            return
        self._panel.ylim = (self.y_min_spin.value(), self.y_max_spin.value()) if checked else None
        self.changed.emit()

    def _apply_y_limits(self, _value: float) -> None:
        if self._updating or not self.y_manual_check.isChecked():
            return
        self._panel.ylim = (self.y_min_spin.value(), self.y_max_spin.value())
        self.changed.emit()

    def _apply_y_scale(self, _index: int) -> None:
        if self._updating:
            return
        self._panel.yscale = self.y_scale_combo.currentData()
        self.changed.emit()

    def _apply_y_invert(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.invert_y = checked
        self.changed.emit()

    def _on_reset_limits(self) -> None:
        self._panel.reset_limits()
        self._sync_from_figure()
        self.changed.emit()

    def _apply_tick_direction(self, _index: int) -> None:
        if self._updating:
            return
        self._panel.tick_direction = self.tick_direction_combo.currentData()
        self.changed.emit()

    def _apply_minor_ticks(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.minor_ticks = checked
        self.changed.emit()

    def _apply_major_spacing_x(self, value: float) -> None:
        if self._updating:
            return
        self._panel.major_tick_spacing_x = value or None
        self.changed.emit()

    def _apply_major_spacing_y(self, value: float) -> None:
        if self._updating:
            return
        self._panel.major_tick_spacing_y = value or None
        self.changed.emit()

    def _apply_minor_spacing_x(self, value: float) -> None:
        if self._updating:
            return
        self._panel.minor_tick_spacing_x = value or None
        self.changed.emit()

    def _apply_minor_spacing_y(self, value: float) -> None:
        if self._updating:
            return
        self._panel.minor_tick_spacing_y = value or None
        self.changed.emit()

    def _apply_major_tick_length(self, value: float) -> None:
        if self._updating:
            return
        self._panel.major_tick_length = value
        self.changed.emit()

    def _apply_major_tick_width(self, value: float) -> None:
        if self._updating:
            return
        self._panel.major_tick_width = value
        self.changed.emit()

    def _apply_minor_tick_length(self, value: float) -> None:
        if self._updating:
            return
        self._panel.minor_tick_length = value
        self.changed.emit()

    def _apply_minor_tick_width(self, value: float) -> None:
        if self._updating:
            return
        self._panel.minor_tick_width = value
        self.changed.emit()

    def _apply_sci_x(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.scientific_notation_x = checked
        self.changed.emit()

    def _apply_sci_y(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.scientific_notation_y = checked
        self.changed.emit()

    def _apply_spines(self, *_args) -> None:
        if self._updating:
            return
        self._panel.spine_top = self.spine_top_check.isChecked()
        self._panel.spine_bottom = self.spine_bottom_check.isChecked()
        self._panel.spine_left = self.spine_left_check.isChecked()
        self._panel.spine_right = self.spine_right_check.isChecked()
        self._panel.spine_linewidth = self.spine_width_spin.value()
        self.changed.emit()

    def _apply_grid(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.grid = checked
        self.changed.emit()

    def _apply_grid_which(self, _index: int) -> None:
        if self._updating:
            return
        self._panel.grid_which = self.grid_which_combo.currentData()
        self.changed.emit()

    # --- Grid Appearance (figure-wide -- see `_FIGURE_GRID_FIELDS`) --------

    def _set_grid_color_swatch(self, color: str) -> None:
        self.grid_color_button.setStyleSheet(f"background-color: {color};")

    def _set_color_swatch(self, button: QPushButton, color: str) -> None:
        """Generic version of `_set_grid_color_swatch` -- used by the 3D
        page's grid/pane color pickers (two independent swatches, unlike
        2D's single figure-wide grid color button)."""
        button.setStyleSheet(f"background-color: {color};")

    def _apply_grid_style(self, _index: int) -> None:
        if self._updating:
            return
        self._figure.grid_linestyle = self.grid_style_combo.currentData()
        self.changed.emit()

    def _apply_grid_width(self, value: float) -> None:
        if self._updating:
            return
        self._figure.grid_linewidth = value
        self.changed.emit()

    def _apply_grid_alpha(self, value: float) -> None:
        if self._updating:
            return
        self._figure.grid_alpha = value
        self.changed.emit()

    def _apply_grid_custom_color_toggled(self, checked: bool) -> None:
        self.grid_color_button.setEnabled(checked)
        if self._updating:
            return
        if checked:
            self._figure.grid_color = self.grid_color_button.property("_picked_color") or _DEFAULT_GRID_COLOR
        else:
            self._figure.grid_color = None
        self._set_grid_color_swatch(self._figure.grid_color or _DEFAULT_GRID_COLOR)
        self.changed.emit()

    def _pick_grid_color(self) -> None:
        initial = QColor(self._figure.grid_color or _DEFAULT_GRID_COLOR)
        color = QColorDialog.getColor(initial, self, "Grid Color")
        if not color.isValid():
            return
        self._figure.grid_color = color.name()
        self.grid_color_button.setProperty("_picked_color", color.name())
        self._set_grid_color_swatch(color.name())
        self.changed.emit()

    def _apply_legend_visible(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.legend_visible = checked
        self.changed.emit()

    def _apply_legend_loc(self, text: str) -> None:
        if self._updating or not text:
            return
        self._panel.legend_loc = text
        self.changed.emit()

    def _apply_legend_ncol(self, value: int) -> None:
        if self._updating:
            return
        self._panel.legend_ncol = value
        self.changed.emit()

    def _apply_legend_frame(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.legend_frameon = checked
        self.changed.emit()

    def _apply_legend_title(self) -> None:
        if self._updating:
            return
        self._panel.legend_title = self.legend_title_edit.text()
        self.changed.emit()

    # --- Apply / Cancel / Reset (see gui.dialogs.live_dialog.LiveDialog) ---
    #
    # Snapshots bind to the Panel object that was active when the dialog
    # opened (not "whatever is active now") -- so if the user switches the
    # active panel elsewhere while this non-modal dialog stays open, Cancel
    # still reverts the panel it was actually opened for, and Reset (which
    # intentionally acts on whatever panel is showing right now) can't be
    # confused with it. 2D-panel-only: reachable in practice only via
    # `reset_all_button`, itself only visible/clickable on the 2D page.

    def capture_state(self) -> tuple[Panel, dict, dict]:
        panel = self._panel
        panel_values = {
            f.name: getattr(panel, f.name)
            for f in dataclasses.fields(panel)
            if f.name not in _PANEL_SNAPSHOT_EXCLUDED_FIELDS
        }
        grid_values = {name: getattr(self._figure, name) for name in _FIGURE_GRID_FIELDS}
        return panel, panel_values, grid_values

    def restore_state(self, state: tuple[Panel, dict, dict]) -> None:
        panel, panel_values, grid_values = state
        for name, value in panel_values.items():
            setattr(panel, name, value)
        for name, value in grid_values.items():
            setattr(self._figure, name, value)
        self._sync_from_figure()
        self.changed.emit()

    def reset_to_defaults(self) -> None:
        """Reset the currently active panel's display settings (not its
        series) to Panel()'s defaults, and the figure-wide Grid Appearance
        fields to a fresh GnoviFigure()'s defaults."""
        defaults = Panel()
        panel = self._panel
        for f in dataclasses.fields(panel):
            if f.name in _PANEL_SNAPSHOT_EXCLUDED_FIELDS:
                continue
            setattr(panel, f.name, getattr(defaults, f.name))
        defaults_figure = GnoviFigure()
        for name in _FIGURE_GRID_FIELDS:
            setattr(self._figure, name, getattr(defaults_figure, name))
        self._sync_from_figure()
        self.changed.emit()

    # --- 3D page behavior ------------------------------------------------------

    def _sync_3d_from_figure(self, panel: Panel3D) -> None:
        self._updating = True
        self.d3_title_edit.setText(panel.title)
        self.d3_xlabel_edit.setText(panel.x_label)
        self.d3_ylabel_edit.setText(panel.y_label)
        self.d3_zlabel_edit.setText(panel.z_label)

        self.d3_x_manual_check.setChecked(panel.xlim is not None)
        self.d3_x_min_spin.setEnabled(panel.xlim is not None)
        self.d3_x_max_spin.setEnabled(panel.xlim is not None)
        if panel.xlim is not None:
            self.d3_x_min_spin.setValue(panel.xlim[0])
            self.d3_x_max_spin.setValue(panel.xlim[1])

        self.d3_y_manual_check.setChecked(panel.ylim is not None)
        self.d3_y_min_spin.setEnabled(panel.ylim is not None)
        self.d3_y_max_spin.setEnabled(panel.ylim is not None)
        if panel.ylim is not None:
            self.d3_y_min_spin.setValue(panel.ylim[0])
            self.d3_y_max_spin.setValue(panel.ylim[1])

        self.d3_z_manual_check.setChecked(panel.zlim is not None)
        self.d3_z_min_spin.setEnabled(panel.zlim is not None)
        self.d3_z_max_spin.setEnabled(panel.zlim is not None)
        if panel.zlim is not None:
            self.d3_z_min_spin.setValue(panel.zlim[0])
            self.d3_z_max_spin.setValue(panel.zlim[1])

        self.d3_aspect_combo.setCurrentIndex(max(self.d3_aspect_combo.findData(panel.aspect_mode), 0))

        self.d3_major_spacing_x_spin.setValue(panel.major_tick_spacing_x or 0.0)
        self.d3_major_spacing_y_spin.setValue(panel.major_tick_spacing_y or 0.0)
        self.d3_major_spacing_z_spin.setValue(panel.major_tick_spacing_z or 0.0)
        self.d3_minor_spacing_x_spin.setValue(panel.minor_tick_spacing_x or 0.0)
        self.d3_minor_spacing_y_spin.setValue(panel.minor_tick_spacing_y or 0.0)
        self.d3_minor_spacing_z_spin.setValue(panel.minor_tick_spacing_z or 0.0)

        self.d3_grid_check.setChecked(panel.grid)
        self.d3_grid_style_combo.setCurrentIndex(max(self.d3_grid_style_combo.findData(panel.grid_linestyle), 0))
        self.d3_grid_width_spin.setValue(panel.grid_linewidth)
        self.d3_grid_alpha_spin.setValue(panel.grid_alpha)
        has_custom_grid_color = panel.grid_color is not None
        self.d3_grid_custom_color_check.setChecked(has_custom_grid_color)
        self.d3_grid_color_button.setEnabled(has_custom_grid_color)
        self._set_color_swatch(self.d3_grid_color_button, panel.grid_color or _DEFAULT_GRID_COLOR)

        self.d3_pane_visible_check.setChecked(panel.pane_visible)
        has_custom_pane_color = panel.pane_color is not None
        self.d3_pane_custom_color_check.setChecked(has_custom_pane_color)
        self.d3_pane_color_button.setEnabled(has_custom_pane_color)
        self._set_color_swatch(self.d3_pane_color_button, panel.pane_color or _DEFAULT_PANE_COLOR)
        self.d3_pane_alpha_spin.setValue(panel.pane_alpha)

        self.d3_legend_check.setChecked(panel.legend_visible)
        self.d3_legend_loc_combo.setCurrentText(panel.legend_loc)
        self.d3_legend_ncol_spin.setValue(panel.legend_ncol)
        self.d3_legend_frame_check.setChecked(panel.legend_frameon)

        self.d3_elevation_spin.setValue(panel.elevation)
        self.d3_azimuth_spin.setValue(panel.azimuth)
        self._updating = False

    def _apply_3d_title(self) -> None:
        if self._updating:
            return
        self._panel.title = self.d3_title_edit.text()
        self.changed.emit()

    def _apply_3d_xlabel(self) -> None:
        if self._updating:
            return
        self._panel.x_label = self.d3_xlabel_edit.text()
        self.changed.emit()

    def _apply_3d_ylabel(self) -> None:
        if self._updating:
            return
        self._panel.y_label = self.d3_ylabel_edit.text()
        self.changed.emit()

    def _apply_3d_zlabel(self) -> None:
        if self._updating:
            return
        self._panel.z_label = self.d3_zlabel_edit.text()
        self.changed.emit()

    def _apply_3d_x_manual(self, checked: bool) -> None:
        self.d3_x_min_spin.setEnabled(checked)
        self.d3_x_max_spin.setEnabled(checked)
        if self._updating:
            return
        self._panel.xlim = (self.d3_x_min_spin.value(), self.d3_x_max_spin.value()) if checked else None
        self.changed.emit()

    def _apply_3d_x_limits(self, _value: float) -> None:
        if self._updating or not self.d3_x_manual_check.isChecked():
            return
        self._panel.xlim = (self.d3_x_min_spin.value(), self.d3_x_max_spin.value())
        self.changed.emit()

    def _apply_3d_y_manual(self, checked: bool) -> None:
        self.d3_y_min_spin.setEnabled(checked)
        self.d3_y_max_spin.setEnabled(checked)
        if self._updating:
            return
        self._panel.ylim = (self.d3_y_min_spin.value(), self.d3_y_max_spin.value()) if checked else None
        self.changed.emit()

    def _apply_3d_y_limits(self, _value: float) -> None:
        if self._updating or not self.d3_y_manual_check.isChecked():
            return
        self._panel.ylim = (self.d3_y_min_spin.value(), self.d3_y_max_spin.value())
        self.changed.emit()

    def _apply_3d_z_manual(self, checked: bool) -> None:
        self.d3_z_min_spin.setEnabled(checked)
        self.d3_z_max_spin.setEnabled(checked)
        if self._updating:
            return
        self._panel.zlim = (self.d3_z_min_spin.value(), self.d3_z_max_spin.value()) if checked else None
        self.changed.emit()

    def _apply_3d_z_limits(self, _value: float) -> None:
        if self._updating or not self.d3_z_manual_check.isChecked():
            return
        self._panel.zlim = (self.d3_z_min_spin.value(), self.d3_z_max_spin.value())
        self.changed.emit()

    def _on_reset_3d_limits(self) -> None:
        self._panel.reset_limits()
        self._sync_from_figure()
        self.changed.emit()

    def _apply_3d_aspect(self, _index: int) -> None:
        if self._updating:
            return
        self._panel.aspect_mode = self.d3_aspect_combo.currentData()
        self.changed.emit()

    def _apply_3d_major_spacing_x(self, value: float) -> None:
        if self._updating:
            return
        self._panel.major_tick_spacing_x = value or None
        self.changed.emit()

    def _apply_3d_major_spacing_y(self, value: float) -> None:
        if self._updating:
            return
        self._panel.major_tick_spacing_y = value or None
        self.changed.emit()

    def _apply_3d_major_spacing_z(self, value: float) -> None:
        if self._updating:
            return
        self._panel.major_tick_spacing_z = value or None
        self.changed.emit()

    def _apply_3d_minor_spacing_x(self, value: float) -> None:
        if self._updating:
            return
        self._panel.minor_tick_spacing_x = value or None
        self.changed.emit()

    def _apply_3d_minor_spacing_y(self, value: float) -> None:
        if self._updating:
            return
        self._panel.minor_tick_spacing_y = value or None
        self.changed.emit()

    def _apply_3d_minor_spacing_z(self, value: float) -> None:
        if self._updating:
            return
        self._panel.minor_tick_spacing_z = value or None
        self.changed.emit()

    def _apply_3d_grid(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.grid = checked
        self.changed.emit()

    def _apply_3d_grid_style(self, _index: int) -> None:
        if self._updating:
            return
        self._panel.grid_linestyle = self.d3_grid_style_combo.currentData()
        self.changed.emit()

    def _apply_3d_grid_width(self, value: float) -> None:
        if self._updating:
            return
        self._panel.grid_linewidth = value
        self.changed.emit()

    def _apply_3d_grid_alpha(self, value: float) -> None:
        if self._updating:
            return
        self._panel.grid_alpha = value
        self.changed.emit()

    def _apply_3d_grid_custom_color_toggled(self, checked: bool) -> None:
        self.d3_grid_color_button.setEnabled(checked)
        if self._updating:
            return
        if checked:
            self._panel.grid_color = self.d3_grid_color_button.property("_picked_color") or _DEFAULT_GRID_COLOR
        else:
            self._panel.grid_color = None
        self._set_color_swatch(self.d3_grid_color_button, self._panel.grid_color or _DEFAULT_GRID_COLOR)
        self.changed.emit()

    def _pick_3d_grid_color(self) -> None:
        initial = QColor(self._panel.grid_color or _DEFAULT_GRID_COLOR)
        color = QColorDialog.getColor(initial, self, "3D Grid Color")
        if not color.isValid():
            return
        self._panel.grid_color = color.name()
        self.d3_grid_color_button.setProperty("_picked_color", color.name())
        self._set_color_swatch(self.d3_grid_color_button, color.name())
        self.changed.emit()

    def _apply_3d_pane_visible(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.pane_visible = checked
        self.changed.emit()

    def _apply_3d_pane_custom_color_toggled(self, checked: bool) -> None:
        self.d3_pane_color_button.setEnabled(checked)
        if self._updating:
            return
        if checked:
            self._panel.pane_color = self.d3_pane_color_button.property("_picked_color") or _DEFAULT_PANE_COLOR
        else:
            self._panel.pane_color = None
        self._set_color_swatch(self.d3_pane_color_button, self._panel.pane_color or _DEFAULT_PANE_COLOR)
        self.changed.emit()

    def _pick_3d_pane_color(self) -> None:
        initial = QColor(self._panel.pane_color or _DEFAULT_PANE_COLOR)
        color = QColorDialog.getColor(initial, self, "3D Pane Color")
        if not color.isValid():
            return
        self._panel.pane_color = color.name()
        self.d3_pane_color_button.setProperty("_picked_color", color.name())
        self._set_color_swatch(self.d3_pane_color_button, color.name())
        self.changed.emit()

    def _apply_3d_pane_alpha(self, value: float) -> None:
        if self._updating:
            return
        self._panel.pane_alpha = value
        self.changed.emit()

    def _apply_3d_legend_visible(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.legend_visible = checked
        self.changed.emit()

    def _apply_3d_legend_loc(self, text: str) -> None:
        if self._updating or not text:
            return
        self._panel.legend_loc = text
        self.changed.emit()

    def _apply_3d_legend_ncol(self, value: int) -> None:
        if self._updating:
            return
        self._panel.legend_ncol = value
        self.changed.emit()

    def _apply_3d_legend_frame(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.legend_frameon = checked
        self.changed.emit()

    def _apply_3d_elevation(self, value: float) -> None:
        if self._updating:
            return
        self._panel.elevation = value
        self.changed.emit()

    def _apply_3d_azimuth(self, value: float) -> None:
        if self._updating:
            return
        self._panel.azimuth = value
        self.changed.emit()

    def _on_reset_3d_view(self) -> None:
        """Restore the camera to `Panel3D()`'s own defaults -- a normal,
        immediate, undoable model edit (via `changed`), same convention as
        `_on_reset_limits`. Unlike "Set Current View" this needs no live
        canvas access, so it's fully self-contained here."""
        if self._updating:
            return
        self._panel.elevation = _PANEL3D_DEFAULTS.elevation
        self._panel.azimuth = _PANEL3D_DEFAULTS.azimuth
        self._sync_from_figure()
        self.changed.emit()
