"""Centralized QSS styling for GNOVI PLOT.

Restrained, modern scientific-desktop look using PySide6/QSS only -- no
additional theme dependency. Deliberately avoids hard-coded font families so
each platform's native font is used.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

_BACKGROUND = "#f4f5f7"
_PANEL_BACKGROUND = "#ffffff"
_BORDER = "#d5d8dd"
_TEXT = "#20242b"
_MUTED_TEXT = "#5b6270"
_ACCENT = "#2f6fed"
_ACCENT_HOVER = "#255ac9"
_ACCENT_PRESSED = "#1e4aa8"
_SELECTION_BG = "#e4ecfd"

MAIN_STYLESHEET = f"""
QWidget {{
    background-color: {_BACKGROUND};
    color: {_TEXT};
    font-size: 10pt;
}}

QMainWindow {{
    background-color: {_BACKGROUND};
}}

QGroupBox {{
    background-color: {_PANEL_BACKGROUND};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    color: {_MUTED_TEXT};
}}

QPushButton {{
    background-color: {_PANEL_BACKGROUND};
    border: 1px solid {_BORDER};
    border-radius: 4px;
    padding: 5px 12px;
}}

QPushButton:hover {{
    border-color: {_ACCENT};
}}

QPushButton:pressed {{
    background-color: {_SELECTION_BG};
}}

QPushButton:disabled {{
    color: {_MUTED_TEXT};
}}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {_PANEL_BACKGROUND};
    border: 1px solid {_BORDER};
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 22px;
}}

QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {_ACCENT};
}}

QComboBox::drop-down {{
    border: none;
    width: 18px;
}}

QListWidget, QTableView {{
    background-color: {_PANEL_BACKGROUND};
    border: 1px solid {_BORDER};
    border-radius: 4px;
    alternate-background-color: #fafbfc;
}}

QListWidget::item {{
    padding: 4px 6px;
    border-radius: 3px;
}}

QListWidget::item:selected {{
    background-color: {_SELECTION_BG};
    color: {_TEXT};
}}

QHeaderView::section {{
    background-color: {_PANEL_BACKGROUND};
    border: none;
    border-bottom: 1px solid {_BORDER};
    padding: 4px;
    font-weight: 600;
    color: {_MUTED_TEXT};
}}

QCheckBox {{
    spacing: 6px;
}}

QSplitter::handle {{
    background-color: {_BORDER};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

QMenuBar {{
    background-color: {_PANEL_BACKGROUND};
    border-bottom: 1px solid {_BORDER};
}}

QToolBar {{
    background-color: {_PANEL_BACKGROUND};
    border-bottom: 1px solid {_BORDER};
    spacing: 4px;
}}
"""


def apply_style(app: QApplication) -> None:
    app.setStyleSheet(MAIN_STYLESHEET)
