from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from gnovi_plot.export.figure_export import ExportError, export_figure
from gnovi_plot.plotting.figure import GnoviFigure

_RASTER_FORMATS = ["PNG", "TIFF"]
_VECTOR_FORMATS = ["SVG", "PDF"]
_DPI_PRESETS = ["150", "300", "600", "1200", "Custom"]
_DEFAULT_DPI = 300


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
        form.addRow(self.tight_bbox_check)
        form.addRow("Padding", self.padding_spin)
        form.addRow("Figure size", self.size_label)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.browse_button)
        form.addRow("Save to", path_row)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.button_box)

        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self.dpi_preset_combo.currentTextChanged.connect(self._on_dpi_preset_changed)
        self.tight_bbox_check.toggled.connect(self.padding_spin.setEnabled)
        self.browse_button.clicked.connect(self._on_browse)

        self._on_format_changed(self.format_combo.currentText())
        # Small, fixed-purpose dialog -- comfortably within a 1366x768 screen.
        self.resize(440, 380)

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
            )
        except (ExportError, OSError) as exc:
            QMessageBox.critical(self, "Export Figure", str(exc))
            return
        self.accept()
