from __future__ import annotations

import io
from pathlib import Path

from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from gnovi_plot.export.figure_export import ExportError, export_figure
from gnovi_plot.plotting.backends.matplotlib_backend import apply_figure_layout, render_figure
from gnovi_plot.plotting.figure import GnoviFigure

_RASTER_FORMATS = ["PNG", "TIFF"]
_VECTOR_FORMATS = ["SVG", "PDF"]
_DPI_PRESETS = ["150", "300", "600", "1200", "Custom"]
_DEFAULT_DPI = 300

# The preview always renders at a fixed, modest DPI for speed -- it exists
# to show layout/background/theme, never as a stand-in for judging actual
# export sharpness (which is governed by the dialog's own DPI setting and
# shown numerically, not visually, since a screen can't really show a
# difference between e.g. 300 and 1200 DPI at preview size anyway).
_PREVIEW_DPI = 100
_PREVIEW_MAX_WIDTH = 380
_PREVIEW_MAX_HEIGHT = 220
_CHECKER_SIZE = 8
_CHECKER_LIGHT = QColor("#e4e6ea")
_CHECKER_DARK = QColor("#c9cdd4")


class ExportFigureDialog(QDialog):
    """Collects export settings and calls `export.figure_export` -- no
    rendering/export logic lives here. Figure size/DPI-relevant dimensions
    come from the figure's own configured size (Figure Size panel), not
    from any on-screen widget.
    """

    def __init__(self, figure: GnoviFigure, parent=None):
        super().__init__(parent)
        self._figure = figure
        self.setWindowTitle("Export Figure")
        self.setModal(True)

        self.format_combo = QComboBox()
        self.format_combo.addItems(_RASTER_FORMATS + _VECTOR_FORMATS)

        self.dpi_preset_combo = QComboBox()
        self.dpi_preset_combo.addItems(_DPI_PRESETS)
        self.dpi_preset_combo.setCurrentText(str(_DEFAULT_DPI))

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(1, 4800)
        self.dpi_spin.setValue(_DEFAULT_DPI)
        self.dpi_spin.setEnabled(False)

        self.transparent_check = QCheckBox("Transparent background")
        # Unchecked (publication-light) by default and independent of the
        # app's current GUI theme -- a dark export is only ever this
        # explicit, separate choice, never inherited from View > Theme.
        self.dark_background_check = QCheckBox("Dark background")
        self.tight_bbox_check = QCheckBox("Tight bounding box")
        self.tight_bbox_check.setChecked(True)

        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0.0, 5.0)
        self.padding_spin.setSingleStep(0.05)
        self.padding_spin.setValue(0.1)
        self.padding_spin.setSuffix(" in")

        self.size_label = QLabel()
        self._update_size_label()

        self.path_edit = QLineEdit()
        self.browse_button = QPushButton("Browse…")

        form = QFormLayout()
        form.addRow("Format", self.format_combo)
        form.addRow("DPI preset", self.dpi_preset_combo)
        form.addRow("Custom DPI", self.dpi_spin)
        form.addRow(self.transparent_check)
        form.addRow(self.dark_background_check)
        form.addRow(self.tight_bbox_check)
        form.addRow("Padding", self.padding_spin)
        form.addRow("Figure size", self.size_label)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.browse_button)
        form.addRow("Save to", path_row)

        # Export Preview: renders through the exact same `render_figure`
        # code path as the on-screen canvas and the real export (see
        # `_refresh_preview`), with the dialog's OWN current
        # transparent/dark-background/tight-bbox/padding settings applied
        # -- so what's shown here is what actually gets written to disk,
        # not just the live on-screen Plot Theme preview.
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFrameShape(QFrame.StyledPanel)
        self.preview_label.setMinimumSize(_PREVIEW_MAX_WIDTH, _PREVIEW_MAX_HEIGHT)
        self.preview_label.setScaledContents(False)

        content_row = QHBoxLayout()
        content_row.addLayout(form, 1)
        content_row.addWidget(self.preview_label, 1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(content_row)
        layout.addWidget(self.button_box)

        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self.dpi_preset_combo.currentTextChanged.connect(self._on_dpi_preset_changed)
        self.tight_bbox_check.toggled.connect(self.padding_spin.setEnabled)
        self.tight_bbox_check.toggled.connect(self._refresh_preview)
        self.transparent_check.toggled.connect(self._refresh_preview)
        self.dark_background_check.toggled.connect(self._refresh_preview)
        self.padding_spin.valueChanged.connect(self._refresh_preview)
        self.browse_button.clicked.connect(self._on_browse)

        self._on_format_changed(self.format_combo.currentText())
        self._refresh_preview()
        # Small, fixed-purpose dialog -- comfortably within a 1366x768 screen.
        self.resize(760, 420)

    def _update_size_label(self) -> None:
        self.size_label.setText(
            f"{self._figure.figure_width_in:.2f} in × {self._figure.figure_height_in:.2f} in "
            "(set in Figure Size)"
        )

    def _on_format_changed(self, fmt: str) -> None:
        is_raster = fmt in _RASTER_FORMATS
        self.dpi_preset_combo.setEnabled(is_raster)
        self.dpi_spin.setEnabled(is_raster and self.dpi_preset_combo.currentText() == "Custom")
        self._sync_path_extension(fmt.lower())
        # The preview always rasterizes to PNG regardless of export format
        # (see `_render_preview_pixmap`), so switching between e.g. PNG and
        # SVG doesn't change it -- only refresh if the widget already exists
        # (this can fire from `__init__` before `preview_label` is built).
        if hasattr(self, "preview_label"):
            self._refresh_preview()

    def _sync_path_extension(self, ext: str) -> None:
        text = self.path_edit.text()
        if not text:
            return
        self.path_edit.setText(str(Path(text).with_suffix(f".{ext}")))

    def _on_dpi_preset_changed(self, text: str) -> None:
        if text == "Custom":
            self.dpi_spin.setEnabled(True)
            return
        self.dpi_spin.setEnabled(False)
        self.dpi_spin.setValue(int(text))

    def _on_browse(self) -> None:
        fmt = self.format_combo.currentText().lower()
        path, _filter = QFileDialog.getSaveFileName(self, "Export Figure", f"figure.{fmt}", f"*.{fmt}")
        if path:
            self.path_edit.setText(path)

    def _on_accept(self) -> None:
        if not self.path_edit.text():
            QMessageBox.warning(self, "Export Figure", "Choose a file to save to.")
            return
        try:
            export_figure(
                self._figure,
                self.path_edit.text(),
                fmt=self.format_combo.currentText().lower(),
                dpi=self.dpi_spin.value(),
                transparent=self.transparent_check.isChecked(),
                tight_bbox=self.tight_bbox_check.isChecked(),
                pad_inches=self.padding_spin.value(),
                dark_mode=self.dark_background_check.isChecked(),
            )
        except (ExportError, OSError) as exc:
            QMessageBox.critical(self, "Export Figure", str(exc))
            return
        self.accept()

    # --- Export Preview ------------------------------------------------------

    def _refresh_preview(self) -> None:
        """Render a small PNG through the exact same `render_figure` path
        `export.figure_export.export_figure` uses, with this dialog's
        current transparent/dark-background/tight-bbox/padding settings
        applied, and show it. Best-effort: a broken preview must never
        block choosing settings or exporting (the real export still runs
        its own typed error handling in `_on_accept`), so failures here are
        shown as text in the preview area rather than raised -- rendering
        an arbitrary user-configured figure can fail in ways beyond this
        dialog's control (e.g. a bad custom font).
        """
        if self._figure.figure_width_in <= 0 or self._figure.figure_height_in <= 0:
            self.preview_label.setText("Preview unavailable: figure size must be positive.")
            return
        try:
            pixmap = self._render_preview_pixmap()
        except Exception as exc:  # noqa: BLE001 -- best-effort preview, see docstring
            self.preview_label.setText(f"Preview unavailable:\n{exc}")
            return
        self.preview_label.setPixmap(pixmap)

    def _render_preview_pixmap(self) -> QPixmap:
        rows, cols = self._figure.layout
        mpl_figure = Figure(
            figsize=(self._figure.figure_width_in, self._figure.figure_height_in), dpi=_PREVIEW_DPI
        )
        axes_list = list(mpl_figure.subplots(rows, cols, squeeze=False).flat)
        render_figure(axes_list, self._figure, dark_mode=self.dark_background_check.isChecked())
        # Same stored margins/spacing `export_figure` applies -- see
        # `plotting.backends.matplotlib_backend.apply_figure_layout` -- so
        # this preview always matches what the real export will produce.
        apply_figure_layout(mpl_figure, self._figure)

        save_kwargs: dict = dict(format="png", dpi=_PREVIEW_DPI, transparent=self.transparent_check.isChecked())
        if self.tight_bbox_check.isChecked():
            save_kwargs["bbox_inches"] = "tight"
            save_kwargs["pad_inches"] = self.padding_spin.value()

        buffer = io.BytesIO()
        mpl_figure.savefig(buffer, **save_kwargs)

        rendered = QPixmap()
        rendered.loadFromData(buffer.getvalue(), "PNG")
        rendered = rendered.scaled(
            _PREVIEW_MAX_WIDTH, _PREVIEW_MAX_HEIGHT, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        if not self.transparent_check.isChecked():
            return rendered

        # Composite over a checkerboard so a transparent export is visibly
        # distinguishable from a plain white one -- a transparent PNG
        # otherwise just looks identical to a white-background PNG here.
        canvas = QPixmap(rendered.size())
        canvas.fill(Qt.transparent)
        painter = QPainter(canvas)
        painter.drawTiledPixmap(canvas.rect(), _checkerboard_tile())
        painter.drawPixmap(0, 0, rendered)
        painter.end()
        return canvas


def _checkerboard_tile() -> QPixmap:
    tile = QPixmap(_CHECKER_SIZE * 2, _CHECKER_SIZE * 2)
    painter = QPainter(tile)
    painter.fillRect(0, 0, _CHECKER_SIZE * 2, _CHECKER_SIZE * 2, _CHECKER_LIGHT)
    painter.fillRect(0, 0, _CHECKER_SIZE, _CHECKER_SIZE, _CHECKER_DARK)
    painter.fillRect(_CHECKER_SIZE, _CHECKER_SIZE, _CHECKER_SIZE, _CHECKER_SIZE, _CHECKER_DARK)
    painter.end()
    return tile
