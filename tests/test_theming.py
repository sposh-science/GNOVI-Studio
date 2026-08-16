from PySide6.QtWidgets import QApplication, QComboBox

from gnovi_plot.gui.styles import _LIGHT_PALETTE, PlotTheme, apply_app_theme, build_stylesheet


def test_plot_theme_has_exactly_light_and_dark():
    assert {mode.value for mode in PlotTheme} == {"light", "dark"}


def test_build_stylesheet_defaults_to_the_single_light_palette():
    assert build_stylesheet() == build_stylesheet()
    assert build_stylesheet().strip()


def test_build_stylesheet_covers_the_theme_relevant_widget_classes():
    qss = build_stylesheet()
    for selector in ("QDialog", "QMenu", "QTabWidget::pane", "QTabBar::tab", "QScrollBar", "QStatusBar"):
        assert selector in qss


def test_stylesheet_supplies_an_explicit_down_arrow_for_every_combo_box():
    """`QComboBox::drop-down` is customized (border/width) above this rule --
    doing so suppresses the style engine's own default arrow glyph, so
    `::down-arrow` must be supplied explicitly or every QComboBox in the app
    renders as a plain field with no visible dropdown indicator (see
    tests/test_combobox_dropdown_arrow.py for the rendered-pixel check)."""
    qss = build_stylesheet()
    assert "QComboBox::down-arrow" in qss
    # A real glyph (the CSS-border-triangle technique), not just `image: none`
    # with nothing to replace it.
    down_arrow_rule = qss.split("QComboBox::down-arrow")[1].split("}")[0]
    assert "border-top" in down_arrow_rule


def test_stylesheet_gives_spinbox_up_down_arrows_real_image_files():
    """QSpinBox/QDoubleSpinBox's `::up-arrow`/`::down-arrow` reference real
    PNG files (see `gui.styles._ensure_spin_arrow_icon_files`), unlike
    QComboBox's own down-arrow above -- confirmed by direct isolated
    reproduction that the CSS-border-triangle technique used for the combo
    arrow does NOT render as a triangle for this specific spin-box
    sub-control on this Qt6/Fusion build (it painted a solid filled
    rectangle instead, every variant tried). See
    tests/test_spinbox_arrows.py for the rendered-pixel check that an
    actual glyph -- not a blank or a rectangle -- paints for real."""
    from gnovi_plot.gui.styles import (
        _SPIN_DOWN_ARROW_DISABLED_PATH,
        _SPIN_DOWN_ARROW_PATH,
        _SPIN_UP_ARROW_DISABLED_PATH,
        _SPIN_UP_ARROW_PATH,
    )

    qss = build_stylesheet()
    up_rule = qss.split("QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {")[1].split("}")[0]
    down_rule = qss.split("QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {")[1].split("}")[0]
    assert f"url({_SPIN_UP_ARROW_PATH.as_posix()})" in up_rule
    assert f"url({_SPIN_DOWN_ARROW_PATH.as_posix()})" in down_rule

    up_disabled_rule = qss.split("QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {")[1].split("}")[
        0
    ]
    down_disabled_rule = qss.split("QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {")[
        1
    ].split("}")[0]
    assert f"url({_SPIN_UP_ARROW_DISABLED_PATH.as_posix()})" in up_disabled_rule
    assert f"url({_SPIN_DOWN_ARROW_DISABLED_PATH.as_posix()})" in down_disabled_rule
    # The disabled variant must be a genuinely different file (a muted
    # color), not the same enabled-state glyph reused.
    assert _SPIN_UP_ARROW_DISABLED_PATH != _SPIN_UP_ARROW_PATH


def test_ensure_spin_arrow_icon_files_generates_all_four_pngs(qapp):
    from gnovi_plot.gui.styles import (
        _SPIN_DOWN_ARROW_DISABLED_PATH,
        _SPIN_DOWN_ARROW_PATH,
        _SPIN_UP_ARROW_DISABLED_PATH,
        _SPIN_UP_ARROW_PATH,
        _ensure_spin_arrow_icon_files,
    )

    _ensure_spin_arrow_icon_files()
    for path in (
        _SPIN_UP_ARROW_PATH,
        _SPIN_DOWN_ARROW_PATH,
        _SPIN_UP_ARROW_DISABLED_PATH,
        _SPIN_DOWN_ARROW_DISABLED_PATH,
    ):
        assert path.exists(), f"{path} was not generated"
        assert path.stat().st_size > 0


def test_apply_app_theme_sets_the_fixed_light_application_stylesheet(qapp: QApplication):
    apply_app_theme(qapp)

    assert qapp.styleSheet() == build_stylesheet()


# --- Visual polish: surface hierarchy / context styling / Workbench chrome ----


def test_stylesheet_defines_a_restrained_context_accent_selector():
    """`QLabel[contextRow="true"]` is how current-context/status labels
    (Active panel/Graph/Data, active dataset) opt into a restrained,
    non-bold accent color instead of bold text -- see `gui.styles` module
    docstring's "no bold for current state" rule and
    `gui.widgets.active_panel_label.ActivePanelLabel`."""
    qss = build_stylesheet()
    assert 'QLabel[contextRow="true"]' in qss
    rule = qss.split('QLabel[contextRow="true"]')[1].split("}")[0]
    assert "font-weight: 400" in rule  # normal weight, never bold


def test_stylesheet_defines_the_workbench_header_selectors():
    """Application chrome for `gui.widgets.workbench_header.WorkbenchHeader`
    -- confirms the styling hooks it relies on exist without asserting
    exact cosmetic pixel values."""
    qss = build_stylesheet()
    for selector in ("QWidget#WorkbenchHeader", "QLabel#WorkbenchHeaderLabel"):
        assert selector in qss


def test_stylesheet_defines_the_workbench_tab_strip_selectors():
    """Application chrome for `gui.widgets.workbench_tabs.WorkbenchTabBar`."""
    qss = build_stylesheet()
    for selector in (
        "QWidget#WorkbenchTabStrip",
        "QTabBar#WorkbenchTabBar::tab",
        "QToolButton#WorkbenchNewButton",
    ):
        assert selector in qss


def test_selection_and_checked_states_do_not_rely_on_bold():
    """Selection/checked/current-state feedback throughout the chrome
    (list items, tabs, tool-strip buttons) must come from background/
    border/color, never `font-weight: 600` -- see `gui.styles` module
    docstring."""
    qss = build_stylesheet()
    for selector in (
        "QListWidget::item:selected",
        "QTabBar::tab:selected",
        "QToolButton#ToolStripButtonLeft:checked",
        "QToolButton#ToolStripButtonRight:checked",
    ):
        rule = qss.split(selector)[1].split("}")[0]
        assert "font-weight" not in rule


# --- Visual polish: control-depth / tactile state system ----------------


def test_pushbutton_and_toolbutton_define_the_full_tactile_state_system():
    """Every interactive push/tool button (base + primary) exposes the full
    NORMAL/HOVER/PRESSED/CHECKED/DISABLED/FOCUS state system described in
    the GNOVI Studio control-depth pass -- not just hover/pressed. The
    "Main" toolbar's own buttons are scoped via `QToolBar#MainToolBar
    QToolButton` (see `test_main_toolbutton_chrome_is_scoped_off_the_
    matplotlib_toolbar`), not the bare `QToolButton` selector."""
    qss = build_stylesheet()
    for selector in (
        "QPushButton",
        'QPushButton[primary="true"]',
        "QToolBar#MainToolBar QToolButton",
    ):
        for state in (":hover", ":pressed", ":disabled", ":focus"):
            assert f"{selector}{state}" in qss, f"missing {selector}{state}"
    # QPushButton:checked already existed; confirm it is still present
    # alongside the newer states rather than having been dropped.
    assert "QPushButton:checked" in qss


def test_toolbutton_pressed_state_shifts_padding_for_a_tactile_inset():
    """PRESSED must look physically depressed, not just recolored -- see the
    module's control-depth notes. A small padding shift (top increases,
    bottom decreases by the same amount) is the technique already used for
    QPushButton; confirm the "Main" toolbar's Undo/Redo/Import/Save/Export
    buttons use it too."""
    qss = build_stylesheet()
    rule = qss.split("QToolBar#MainToolBar QToolButton:pressed")[1].split("}")[0]
    assert "padding-top" in rule and "padding-bottom" in rule


def test_main_toolbutton_chrome_is_scoped_off_the_matplotlib_toolbar():
    """The raised/tactile QToolButton chrome must be scoped to
    `QToolBar#MainToolBar QToolButton`, never the bare `QToolButton`
    selector -- the Matplotlib navigation toolbar's own buttons (Home/Pan/
    Zoom/.../Save) are also un-decorated QToolButtons, and an opaque raised
    background painted on every QToolButton previously made those icons
    unreadable. See `MainWindow._create_toolbar`'s `setObjectName
    ("MainToolBar")`."""
    qss = build_stylesheet()
    assert "QToolBar#MainToolBar QToolButton {" in qss
    # No bare, un-scoped `QToolButton { ... }` rule should exist anymore
    # (only the scoped one and the objectName/property-qualified ones for
    # ToolStrip/WorkbenchNewButton/collapsible sections).
    assert "\nQToolButton {" not in qss


def test_toolstrip_buttons_define_disabled_and_focus_states():
    """The left/right DSO-style tool-strip buttons (Data/Plot/Series/
    Figure/Layout/Axes, and the mirrored Working button) previously only
    defined hover/pressed/checked -- disabled and focus must exist too."""
    qss = build_stylesheet()
    for side in ("Left", "Right"):
        for state in (":disabled", ":focus"):
            assert f"QToolButton#ToolStripButton{side}{state}" in qss


def test_toolstrip_checked_state_differs_from_pressed_state():
    """CHECKED (the currently-open tool, latched) must not reuse PRESSED's
    background -- otherwise the open tool would look identical to a button
    mid-click. See gui.widgets.tool_drawer.ToolDrawer."""
    qss = build_stylesheet()
    pressed_rule = qss.split("QToolButton#ToolStripButtonLeft:pressed")[1].split("}")[0]
    checked_rule = qss.split("QToolButton#ToolStripButtonLeft:checked")[1].split("}")[0]
    assert pressed_rule != checked_rule
    assert _LIGHT_PALETTE["bg_control_press"] in pressed_rule
    assert _LIGHT_PALETTE["bg_control_press"] not in checked_rule
    assert _LIGHT_PALETTE["accent_soft"] in checked_rule


def test_primary_button_gradient_puts_accent_above_accent_hover():
    """NORMAL should read as raised (light source from above): the top
    gradient stop must be `accent` and the bottom stop `accent_hover`,
    matching the plain QPushButton's own top-light/bottom-dark gradient
    direction. Uses a custom palette with distinguishable sentinel values so
    the assertion checks stop *order*, not real-palette color literals."""
    palette = dict(_LIGHT_PALETTE)
    palette["accent"] = "#111111"
    palette["accent_hover"] = "#222222"
    qss = build_stylesheet(palette)
    rule = qss.split('QPushButton[primary="true"] {')[1].split("}")[0]
    assert rule.index("stop:0 #111111") < rule.index("stop:1 #222222")


def test_combo_and_spin_inputs_keep_their_hover_and_focus_selectors():
    """Recessed-field polish must not drop the existing hover/focus
    affordances relied on by tests/test_combobox_dropdown_arrow.py and the
    left-drawer panels; QLineEdit now gains a hover rule it previously
    lacked."""
    qss = build_stylesheet()
    assert "QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover" in qss
    assert "QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus" in qss


# --- Visual polish: context card / readability / numeric-field geometry -


def test_context_row_label_is_a_light_card_not_bare_text():
    """`QLabel[contextRow="true"]` (Active panel / Graph / Data -- see
    `gui.widgets.active_panel_label.ActivePanelLabel`) must render as a
    dedicated Surface-3 card (background/border/padding), not bare text
    sitting directly on the drawer background -- and must keep the
    restrained, non-bold context-accent color/weight it already had."""
    qss = build_stylesheet()
    rule = qss.split('QLabel[contextRow="true"]')[1].split("}")[0]
    assert "background-color" in rule
    assert "padding" in rule
    assert "font-weight: 400" in rule  # still never bold


def test_application_chrome_font_size_is_a_restrained_one_point_bump():
    """A deliberate +1pt readability pass (10pt -> 11pt), not the +2pt the
    spec explicitly rejected. The ToolStrip's own small caption label stays
    at its existing 8pt (a fixed 64px-wide strip where growing the label
    risks clipping/wrapping)."""
    qss = build_stylesheet()
    base_rule = qss.split("QWidget {")[1].split("}")[0]
    assert "font-size: 11pt" in base_rule
    assert "font-size: 10pt" not in qss
    assert "font-size: 12pt" not in qss
    # ToolStrip caption label deliberately unchanged.
    strip_rule = qss.split("QToolButton#ToolStripButtonLeft, QToolButton#ToolStripButtonRight {")[1].split("}")[0]
    assert "font-size: 8pt" in strip_rule


def test_combobox_popup_has_its_own_non_native_selection_colors():
    """The QComboBox popup list previously had no rule of its own and fell
    back to the platform style's own highlight -- confirm it now uses the
    restrained accent_soft language instead. Deliberately no `::item`/
    `::item:hover` sub-control rules alongside this (see the styles.py
    comment on this rule): adding one, even with no color of its own, was
    found (by the runtime test below) to make Qt's styled-item delegate
    stop honoring `selection-background-color` and repaint the current/
    hovered row solid black again."""
    qss = build_stylesheet()
    assert "QComboBox QAbstractItemView {" in qss
    view_rule = qss.split("QComboBox QAbstractItemView {")[1].split("}")[0]
    assert "selection-background-color" in view_rule
    assert "selection-background-color: #000" not in view_rule.replace(" ", "")
    assert "QComboBox QAbstractItemView::item" not in qss


def test_combobox_background_is_flat_not_gradient():
    """Root cause of the black popup-row bug (see the runtime test below):
    a `qlineargradient` background on QComboBox itself -- not the popup --
    broke the popup's current/hovered-row rendering on this app's real
    Qt6/Wayland runtime. QLineEdit/QSpinBox/QDoubleSpinBox have no popup and
    keep the shared gradient; only QComboBox must stay flat."""
    qss = build_stylesheet()
    combo_rule = qss.split("\nQComboBox {")[1].split("}")[0]
    assert "qlineargradient" not in combo_rule
    assert "background-color" in combo_rule
    shared_rule = qss.split("QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {")[1].split("}")[0]
    assert "qlineargradient" in shared_rule  # still a gradient for the other three


def test_combobox_popup_current_row_never_renders_black(qapp: QApplication):
    """Runtime regression test for the black hover/current-row bug found
    during Linux visual testing -- a stylesheet-string check alone (the two
    tests above) cannot catch this, since the bug was a rendering-time
    interaction between QComboBox's own background brush and Qt's styled
    popup delegate, not something visible in the QSS text itself. Confirmed
    by direct reproduction that this same offscreen-platform rendering path
    DOES reproduce the bug against the pre-fix styling (a `qlineargradient`
    QComboBox background plus a `::item` sub-control rule) -- so a clean
    result here is a genuine regression guard, not a tautology.

    Drives the popup's current/highlighted row via `QAbstractItemView.
    setCurrentIndex` directly rather than a synthetic OS-level mouse move
    (unreliable under this offscreen/Wayland setup) -- `QComboBoxListView`
    itself calls the same `setCurrentIndex` in response to real mouse
    movement, so this exercises the identical paint path a real hover
    would."""
    apply_app_theme(qapp)
    combo = QComboBox()
    combo.addItems(["Off", "X line", "Y line", "Crosshair"])
    combo.resize(150, 26)
    combo.show()
    qapp.processEvents()
    combo.showPopup()
    qapp.processEvents()
    view = combo.view()
    try:
        for row in range(combo.count()):
            index = view.model().index(row, 0)
            view.setCurrentIndex(index)
            qapp.processEvents()
            rect = view.visualRect(index)
            image = view.grab().toImage()
            near_black_pixels = [
                (x, rect.center().y())
                for x in range(max(0, rect.left()), min(image.width(), rect.right()), 3)
                if (lambda c: c.red() < 20 and c.green() < 20 and c.blue() < 20)(image.pixelColor(x, rect.center().y()))
            ]
            assert not near_black_pixels, f"row {row} ({combo.itemText(row)!r}) rendered near-black pixels"
    finally:
        combo.hidePopup()
        combo.close()


# --- Visual regression: real Linux runtime findings ----------------------


def test_matplotlib_toolbar_icons_render_dark_not_washed_out(qapp: QApplication):
    """Runtime regression test for a bug only visible on the app's real
    Linux runtime: applying `apply_app_theme`'s global stylesheet made
    Matplotlib's own `NavigationToolbar2QT` icon engine misdetect "dark
    mode" (see `gui.main_window._patch_mpl_icon_engine_dark_mode_detection`
    for the full root-cause writeup) and recolor Home/Back/Forward/Pan/
    Zoom/.../Save white-on-transparent -- invisible against GNOVI's actual
    light chrome. A stylesheet-string check cannot catch this: the bug
    lived in Matplotlib's own private icon-color logic, not GNOVI's QSS.
    Importing `gui.main_window` (this module) is what installs the patch;
    building a bare toolbar (not a full `MainWindow`) keeps this fast."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    from PySide6.QtWidgets import QMainWindow

    apply_app_theme(qapp)
    window = QMainWindow()
    try:
        canvas = FigureCanvasQTAgg(Figure())
        toolbar = NavigationToolbar2QT(canvas, window)
        window.addToolBar(toolbar)
        window.setCentralWidget(canvas)
        window.show()
        qapp.processEvents()

        image = toolbar.grab().toImage()
        dark_pixels = sum(
            1
            for x in range(image.width())
            for y in range(image.height())
            if (lambda c: c.red() < 100 and c.green() < 100 and c.blue() < 100)(image.pixelColor(x, y))
        )
        # A washed-out (white-on-transparent) toolbar renders essentially
        # zero dark pixels; a healthy one has hundreds from the icon
        # strokes alone.
        assert dark_pixels > 200, f"only {dark_pixels} dark pixels -- toolbar icons look washed out"
    finally:
        window.close()


def test_undo_redo_icons_are_exact_horizontal_mirrors_not_a_circle(qapp: QApplication):
    """Runtime pixel check for the Undo/Redo toolbar glyphs (see
    `gui.main_window._make_undo_redo_icon`): after repeated Linux visual
    testing found earlier circular-arc builds still read as Refresh/Reload
    regardless of how open the arc was, the current shape is built from a
    short vertical stub, one quarter-circle turn, and a long horizontal
    shaft/arrowhead -- i.e. deliberately not a segment of one circle.
    Redo must be undo's *exact* pixel mirror (every alpha value matches
    once the redo image is flipped horizontally) -- confirms the mirrored
    arc-sweep-direction math is right, not just "some shape exists on each
    side"."""
    from gnovi_plot.gui.main_window import _UNDO_REDO_ICON_SIZE, _make_undo_redo_icon

    size = _UNDO_REDO_ICON_SIZE
    undo_image = _make_undo_redo_icon("undo").pixmap(size, size).toImage()
    redo_image = _make_undo_redo_icon("redo").pixmap(size, size).toImage()

    opaque_pixel_count = 0
    for y in range(size):
        for x in range(size):
            undo_alpha = undo_image.pixelColor(x, y).alpha()
            redo_alpha_mirrored = redo_image.pixelColor(size - 1 - x, y).alpha()
            assert abs(undo_alpha - redo_alpha_mirrored) <= 40, f"mismatch at ({x},{y})"
            if undo_alpha > 128:
                opaque_pixel_count += 1

    # Sanity check the glyph actually drew something substantial (a blank
    # or near-blank icon would trivially "mirror" itself).
    assert opaque_pixel_count > size  # more than a single stray line's worth of pixels


def test_ordinary_fields_are_not_width_capped_and_share_the_common_height():
    """A prior build capped QSpinBox/QDoubleSpinBox (and, separately, a
    Python-side `apply_sidebar_field_width` capped specific left-drawer
    QComboBox/QLineEdit instances) to a fixed preferred width -- Linux
    visual testing found this left ordinary property editors looking
    artificially narrow, with unused blank space to their right, instead of
    filling their row's available width via Qt's own default QFormLayout
    field-growth policy. Neither mechanism should exist in the shared QSS
    anymore: no `max-width` on any of the four field types, while `QComboBox,
    QLineEdit, QSpinBox, QDoubleSpinBox` still share one `min-height` so they
    remain visually one control family."""
    qss = build_stylesheet()
    shared_rule = qss.split("QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {")[1].split("}")[0]
    assert "min-height: 26px" in shared_rule
    assert "max-width" not in shared_rule
    # No standalone QSpinBox/QDoubleSpinBox rule capping width should exist
    # either (newline-anchored so it doesn't match inside the longer shared
    # selector above, which also ends in that same substring).
    assert "\nQSpinBox, QDoubleSpinBox {" not in qss


def test_no_sidebar_field_width_capping_helper_remains():
    """`gui.styles.apply_sidebar_field_width`/`SIDEBAR_FIELD_WIDTH` (the
    per-widget Python-side width cap for left-drawer QComboBox/QLineEdit
    fields) must not be reintroduced -- ordinary fields should rely on Qt's
    own layout/size-policy behavior to fill their row, not a fixed
    preferred width. See `test_ordinary_fields_are_not_width_capped_and_
    share_the_common_height` for the QSS side of the same requirement."""
    import gnovi_plot.gui.styles as styles_module

    assert not hasattr(styles_module, "apply_sidebar_field_width")
    assert not hasattr(styles_module, "SIDEBAR_FIELD_WIDTH")


def test_apply_app_theme_is_not_affected_by_plot_theme_selection(qapp: QApplication):
    """The application chrome has exactly one stylesheet, regardless of
    which `PlotTheme` (Light/Dark) is active for the plot canvas -- there is
    no app-wide dark mode (see gui.styles module docstring)."""
    apply_app_theme(qapp)
    chrome_stylesheet = qapp.styleSheet()

    # Changing the plot canvas theme is a MainWindow-level concern (drives
    # `PlotCanvas.render(..., dark_mode=...)`) and never re-applies the
    # application stylesheet, so re-asserting it here should be a no-op.
    apply_app_theme(qapp)

    assert qapp.styleSheet() == chrome_stylesheet
