"""Rendered-pixel regression coverage for the QComboBox dropdown chevron
(see `gui.styles`'s `QComboBox::down-arrow` rule). Confirms an actual glyph
paints inside the drop-down button area -- not just that the QSS text
mentions `down-arrow` (see `tests/test_theming.py` for that weaker check) --
for both a plain QComboBox (the global fix) and the Plot page's dataset
selector specifically (item 2's original complaint).
"""

import pandas as pd
from PySide6.QtWidgets import QComboBox, QPushButton, QVBoxLayout, QWidget

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.styles import apply_app_theme
from gnovi_plot.gui.widgets.dataset_panel import DatasetPanel

# The QSS reserves this much width for the drop-down button
# (`QComboBox::drop-down { width: 18px; }`).
_DROP_DOWN_WIDTH = 18


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return Dataset(name=name, dataframe=df)


def _grab_pixels(widget) -> list[list[bool]]:
    """Rendered `widget` as a grid of bools: True where the pixel is
    meaningfully darker than the field's white background."""
    pix = widget.grab()
    img = pix.toImage()
    w, h = img.width(), img.height()
    grid = []
    for y in range(h):
        row = []
        for x in range(w):
            c = img.pixelColor(x, y)
            row.append(c.red() < 240 or c.green() < 240 or c.blue() < 240)
        grid.append(row)
    return grid


def _has_interior_glyph(grid, *, region_width: int) -> bool:
    """True if some pixel strictly inside the drop-down button's own
    interior (excluding the 1px field border) is non-background -- i.e. an
    actual arrow glyph, not just the field's border tracing the button's
    edges."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    found = False
    for y in range(2, h - 2):
        for x in range(w - region_width + 2, w - 2):
            if grid[y][x]:
                found = True
    return found


def _prepare(widget, other_focus_target, *, width=240, height=80):
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.addWidget(widget)
    layout.addWidget(other_focus_target)
    container.resize(width, height)
    container.show()
    other_focus_target.setFocus()  # keep the combo itself unfocused (no accent border)
    return container


def test_plain_combo_box_shows_a_visible_dropdown_glyph(qapp):
    apply_app_theme(qapp)
    combo = QComboBox()
    combo.addItems(["copper Sulphate S4-SR 0.05", "b", "c"])
    other = QPushButton("other")
    container = _prepare(combo, other)
    qapp.processEvents()

    grid = _grab_pixels(combo)

    assert _has_interior_glyph(grid, region_width=_DROP_DOWN_WIDTH)
    container.close()


def test_dataset_combo_shows_a_visible_dropdown_glyph(qapp):
    apply_app_theme(qapp)
    manager = DatasetManager()
    manager.add(_make_dataset("copper Sulphate S4-SR 0.05"))
    from PySide6.QtWidgets import QTableView

    panel = DatasetPanel(manager, QTableView())
    panel._refresh_list(select_id=manager.datasets[0].id)
    other = QPushButton("other")
    container = _prepare(panel.active_dataset_combo, other)
    qapp.processEvents()

    grid = _grab_pixels(panel.active_dataset_combo)

    assert _has_interior_glyph(grid, region_width=_DROP_DOWN_WIDTH)
    container.close()
    panel.close()


def test_glyph_stays_confined_to_the_drop_down_button_region(qapp):
    """The arrow must never bleed into the text-display area to the left of
    the drop-down button -- confirmed here by checking the left two-thirds
    of the field's interior is untouched (background only, aside from the
    field's own top/bottom border)."""
    apply_app_theme(qapp)
    combo = QComboBox()
    combo.addItem("")  # blank text -- isolates the arrow glyph from text pixels
    other = QPushButton("other")
    container = _prepare(combo, other)
    qapp.processEvents()

    grid = _grab_pixels(combo)
    h = len(grid)
    w = len(grid[0])
    text_area_right_edge = w - _DROP_DOWN_WIDTH - 4  # a small buffer before the button
    interior_rows = range(3, h - 3)
    left_interior_is_clean = all(not grid[y][x] for y in interior_rows for x in range(2, text_area_right_edge))

    assert left_interior_is_clean
    container.close()


def test_dropdown_glyph_visible_at_a_narrow_drawer_width(qapp):
    apply_app_theme(qapp)
    combo = QComboBox()
    combo.addItem("copper Sulphate S4-SR 0.05")
    other = QPushButton("other")
    container = _prepare(combo, other, width=140)
    combo.resize(120, 26)
    qapp.processEvents()

    grid = _grab_pixels(combo)

    assert _has_interior_glyph(grid, region_width=_DROP_DOWN_WIDTH)
    container.close()
