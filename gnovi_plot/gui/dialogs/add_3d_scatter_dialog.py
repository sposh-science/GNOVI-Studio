from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.plotting.figure import Panel3D

# Same marker codes `gui.widgets.plot_series_panel` offers for 2D series,
# minus "None" -- a 3D scatter point with no marker at all would render
# nothing, unlike a 2D line plot (where marker=None just means "no marker
# dots on the line", the line itself still visible).
_MARKER_OPTIONS = [
    ("Circle", "o"),
    ("Square", "s"),
    ("Triangle", "^"),
    ("Diamond", "D"),
    ("Plus", "+"),
    ("Cross", "x"),
    ("Star", "*"),
]


class Add3DScatterDialog(QDialog):
    """Collects a Dataset + X/Y/Z column selection, plus the small set of
    properties this milestone supports (title, X/Y/Z labels, marker,
    marker size), for creating -- or editing -- a 3D scatter `Panel3D`.

    One dialog for both create and edit (see `MainWindow._on_add_3d_
    scatter_requested`): passing `existing_panel` pre-fills every field
    from its current state (its dataset/columns from its one series, if
    any) rather than starting blank, so re-invoking "Add 3D Scatter…" on
    an already-3D active Panel reads as "edit this Panel's properties",
    not "start over". This dialog only ever collects values into public
    attributes on accept (`self.dataset`/`self.x_column`/etc., same
    convention as `CalculatedColumnDialog`) -- it never touches the
    Project/Figure itself; the caller decides whether to build a fresh
    `Panel3D` or mutate an existing one in place.
    """

    def __init__(self, dataset_manager: DatasetManager, existing_panel: Panel3D | None = None, parent=None):
        super().__init__(parent)
        self._dataset_manager = dataset_manager
        self.setWindowTitle("3D Scatter Panel" if existing_panel is not None else "Add 3D Scatter")
        self.resize(420, 320)

        self.dataset_combo = QComboBox()
        for dataset in dataset_manager.datasets:
            self.dataset_combo.addItem(dataset.name, dataset.id)

        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.z_combo = QComboBox()

        self.title_edit = QLineEdit()
        self.x_label_edit = QLineEdit()
        self.y_label_edit = QLineEdit()
        self.z_label_edit = QLineEdit()

        self.marker_combo = QComboBox()
        for text, code in _MARKER_OPTIONS:
            self.marker_combo.addItem(text, code)

        self.marker_size_spin = QDoubleSpinBox()
        self.marker_size_spin.setRange(1.0, 72.0)
        self.marker_size_spin.setValue(6.0)

        form = QFormLayout()
        form.addRow("Dataset", self.dataset_combo)
        form.addRow("X", self.x_combo)
        form.addRow("Y", self.y_combo)
        form.addRow("Z", self.z_combo)
        form.addRow("Title", self.title_edit)
        form.addRow("X label", self.x_label_edit)
        form.addRow("Y label", self.y_label_edit)
        form.addRow("Z label", self.z_label_edit)
        form.addRow("Marker", self.marker_combo)
        form.addRow("Marker size", self.marker_size_spin)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #c0392b;")
        self.error_label.setVisible(False)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addWidget(self.button_box)

        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)

        # Results collected on accept -- see class docstring.
        self.dataset = None
        self.x_column: str = ""
        self.y_column: str = ""
        self.z_column: str = ""
        self.title: str = ""
        self.x_label: str = ""
        self.y_label: str = ""
        self.z_label: str = ""
        self.marker: str = "o"
        self.marker_size: float = 6.0

        if dataset_manager.datasets:
            self._on_dataset_changed(self.dataset_combo.currentIndex())
        if existing_panel is not None:
            self._prefill(existing_panel)

        self.button_box.button(QDialogButtonBox.Ok).setEnabled(bool(dataset_manager.datasets))
        if not dataset_manager.datasets:
            self.error_label.setText("Import a Dataset with numeric X/Y/Z columns first.")
            self.error_label.setVisible(True)

    def _on_dataset_changed(self, index: int) -> None:
        dataset_id = self.dataset_combo.itemData(index)
        dataset = self._dataset_manager.get(dataset_id) if dataset_id else None
        columns = dataset.columns if dataset is not None else []
        for combo in (self.x_combo, self.y_combo, self.z_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(columns)
            if current in columns:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def _prefill(self, panel: Panel3D) -> None:
        self.title_edit.setText(panel.title)
        self.x_label_edit.setText(panel.x_label)
        self.y_label_edit.setText(panel.y_label)
        self.z_label_edit.setText(panel.z_label)
        if not panel.series:
            return
        series = panel.series[0]
        dataset_index = self.dataset_combo.findData(series.dataset.id)
        if dataset_index >= 0:
            self.dataset_combo.setCurrentIndex(dataset_index)
            self._on_dataset_changed(dataset_index)
        self.x_combo.setCurrentText(series.x_column)
        self.y_combo.setCurrentText(series.y_column)
        self.z_combo.setCurrentText(series.z_column)
        self.marker_combo.setCurrentIndex(max(self.marker_combo.findData(series.marker), 0))
        self.marker_size_spin.setValue(series.marker_size)

    def _on_accept(self) -> None:
        dataset_id = self.dataset_combo.currentData()
        dataset = self._dataset_manager.get(dataset_id) if dataset_id else None
        x, y, z = self.x_combo.currentText(), self.y_combo.currentText(), self.z_combo.currentText()
        if dataset is None or not x or not y or not z:
            self.error_label.setText("Choose a Dataset and X/Y/Z columns.")
            self.error_label.setVisible(True)
            return
        self.dataset = dataset
        self.x_column, self.y_column, self.z_column = x, y, z
        self.title = self.title_edit.text()
        self.x_label = self.x_label_edit.text()
        self.y_label = self.y_label_edit.text()
        self.z_label = self.z_label_edit.text()
        self.marker = self.marker_combo.currentData()
        self.marker_size = self.marker_size_spin.value()
        self.accept()
