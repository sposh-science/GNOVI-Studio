from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)

from gnovi_plot.equations.evaluator import evaluate_formula
from gnovi_plot.equations.parser import FormulaError

_PREVIEW_ROWS = 5


class CalculatedColumnDialog(QDialog):
    """Collects a new column name and formula, with a live preview against
    the dataset's current working data before the column is actually
    created. Column names that aren't valid bare identifiers (e.g.
    `Current/A`) are inserted as `[Current/A]` bracket references.
    """

    def __init__(self, dataframe: pd.DataFrame, existing_columns: list[str], parent=None):
        super().__init__(parent)
        self._dataframe = dataframe
        self._existing_columns = existing_columns
        self.name: str = ""
        self.formula: str = ""

        self.setWindowTitle("Calculated Column")
        self.resize(480, 420)

        self.name_edit = QLineEdit()
        self.formula_edit = QLineEdit()
        self.formula_edit.setPlaceholderText("e.g. [Current/A] * 1000")

        self.column_list = QListWidget()
        self.column_list.addItems(list(dataframe.columns))
        self.insert_button = QPushButton("Insert Column Reference")

        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("New column name", self.name_edit)
        form.addRow("Formula", self.formula_edit)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Available columns (select, then insert into formula):"))
        layout.addWidget(self.column_list)
        layout.addWidget(self.insert_button)
        layout.addWidget(QLabel("Preview (first rows of the working data):"))
        layout.addWidget(self.preview_label)
        layout.addWidget(self.button_box)

        self.name_edit.textChanged.connect(self._update_preview)
        self.formula_edit.textChanged.connect(self._update_preview)
        self.insert_button.clicked.connect(self._insert_column_reference)

        self._update_preview()

    def _insert_column_reference(self) -> None:
        item = self.column_list.currentItem()
        if item is None:
            return
        name = item.text()
        token = name if name.isidentifier() else f"[{name}]"
        self.formula_edit.insert(token)
        self.formula_edit.setFocus()

    def _update_preview(self) -> None:
        formula = self.formula_edit.text().strip()
        name = self.name_edit.text().strip()

        if not formula:
            self.preview_label.setText("Enter a formula to see a preview.")
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
            return

        if not name:
            self.preview_label.setText("Enter a column name.")
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
            return

        if name in self._existing_columns:
            self.preview_label.setText(f"Column '{name}' already exists -- choose a different name.")
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
            return

        try:
            values, source_columns = evaluate_formula(self._dataframe, formula)
        except FormulaError as exc:
            self.preview_label.setText(f"Error: {exc}")
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
            return

        preview_values = ", ".join(f"{v:.6g}" for v in values.head(_PREVIEW_ROWS))
        self.preview_label.setText(
            f"Uses: {', '.join(source_columns) or '(no columns)'}\n"
            f"First values: {preview_values}"
        )
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def _on_accept(self) -> None:
        self.name = self.name_edit.text().strip()
        self.formula = self.formula_edit.text().strip()
        self.accept()
