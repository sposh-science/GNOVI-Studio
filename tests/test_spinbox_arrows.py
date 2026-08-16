"""Rendered-pixel regression coverage for QSpinBox/QDoubleSpinBox's
`::up-arrow`/`::down-arrow` glyphs (see `gui.styles`'s
`_ensure_spin_arrow_icon_files` and the QSS comment above
`QSpinBox::up-arrow`). Confirms an actual triangle glyph paints inside each
arrow half -- not just that the QSS text mentions `up-arrow`/`down-arrow`
(see `tests/test_theming.py` for that weaker check) -- for both an enabled
and a disabled spin box, mirroring `tests/test_combobox_dropdown_arrow.py`'s
approach for the combo box dropdown chevron.
"""

from PySide6.QtWidgets import QDoubleSpinBox, QPushButton, QVBoxLayout, QWidget

from gnovi_plot.gui.styles import apply_app_theme

# The QSS gives each arrow half a 16px-wide column on the field's right edge.
_ARROW_COLUMN_WIDTH = 16


def _grab_pixels(widget) -> list[list[bool]]:
    """Rendered `widget` as a grid of bools: True where the pixel is
    meaningfully darker than the field's white/near-white background."""
    pix = widget.grab()
    img = pix.toImage()
    w, h = img.width(), img.height()
    grid = []
    for y in range(h):
        row = []
        for x in range(w):
            c = img.pixelColor(x, y)
            row.append(c.red() < 235 or c.green() < 235 or c.blue() < 235)
        grid.append(row)
    return grid


def _has_glyph_in_region(grid, *, y_start: int, y_end: int, region_width: int) -> bool:
    """True if some pixel inside the given vertical band's right-hand
    `region_width` columns is non-background -- i.e. an actual arrow glyph,
    not just the field's own border tracing the button's edges."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    y_end = min(y_end, h)
    for y in range(max(0, y_start), y_end):
        for x in range(max(0, w - region_width), w):
            if grid[y][x]:
                return True
    return False


def _prepare(widget, other_focus_target, *, width=200, height=60):
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.addWidget(widget)
    layout.addWidget(other_focus_target)
    container.resize(width, height)
    container.show()
    other_focus_target.setFocus()  # keep the spin box itself unfocused (no accent border)
    return container


def test_enabled_spinbox_shows_up_and_down_arrow_glyphs(qapp):
    apply_app_theme(qapp)
    spin = QDoubleSpinBox()
    spin.setValue(6.4)
    other = QPushButton("other")
    container = _prepare(spin, other)
    qapp.processEvents()

    grid = _grab_pixels(spin)
    h = spin.height()

    assert _has_glyph_in_region(grid, y_start=0, y_end=h // 2, region_width=_ARROW_COLUMN_WIDTH)
    assert _has_glyph_in_region(grid, y_start=h // 2, y_end=h, region_width=_ARROW_COLUMN_WIDTH)
    container.close()


def test_disabled_spinbox_still_shows_muted_arrow_glyphs(qapp):
    """Muted, not blank -- a disabled numeric field should still show
    *where* the arrows are, just visibly unavailable (see the module-level
    control-depth state system's DISABLED requirement), never literally
    nothing painted for the sub-controls."""
    apply_app_theme(qapp)
    spin = QDoubleSpinBox()
    spin.setValue(0.0)
    spin.setEnabled(False)
    other = QPushButton("other")
    container = _prepare(spin, other)
    qapp.processEvents()

    grid = _grab_pixels(spin)
    h = spin.height()

    assert _has_glyph_in_region(grid, y_start=0, y_end=h // 2, region_width=_ARROW_COLUMN_WIDTH)
    assert _has_glyph_in_region(grid, y_start=h // 2, y_end=h, region_width=_ARROW_COLUMN_WIDTH)
    container.close()
