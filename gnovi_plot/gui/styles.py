"""Centralized QSS styling for GNOVI PLOT.

Restrained, modern scientific-desktop look using PySide6/QSS only -- no
additional theme dependency. Deliberately avoids hard-coded font families so
each platform's native font is used.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

_BACKGROUND = "#f4f5f7"
_PANEL_BACKGROUND = "#ffffff"
_SUBTLE_BACKGROUND = "#f8f9fb"
_BORDER = "#d5d8dd"
_TEXT = "#20242b"
_MUTED_TEXT = "#5b6270"
_ACCENT = "#2f6fed"
_ACCENT_HOVER = "#255ac9"
_ACCENT_PRESSED = "#1e4aa8"
_ACCENT_TEXT = "#ffffff"
_SELECTION_BG = "#e4ecfd"
_STALE = "#b3261e"

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
    margin-top: 16px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    color: {_MUTED_TEXT};
    letter-spacing: 0.3px;
}}

QPushButton {{
    background-color: {_PANEL_BACKGROUND};
    border: 1px solid {_BORDER};
    border-radius: 4px;
    padding: 5px 12px;
    min-height: 22px;
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

QPushButton[primary="true"] {{
    background-color: {_ACCENT};
    border-color: {_ACCENT};
    color: {_ACCENT_TEXT};
    font-weight: 600;
}}

QPushButton[primary="true"]:hover {{
    background-color: {_ACCENT_HOVER};
    border-color: {_ACCENT_HOVER};
}}

QPushButton[primary="true"]:pressed {{
    background-color: {_ACCENT_PRESSED};
    border-color: {_ACCENT_PRESSED};
}}

QPushButton[primary="true"]:disabled {{
    background-color: {_SUBTLE_BACKGROUND};
    border-color: {_BORDER};
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
    alternate-background-color: {_SUBTLE_BACKGROUND};
    gridline-color: {_BORDER};
}}

QListWidget::item {{
    padding: 4px 6px 4px 8px;
    border-radius: 3px;
    border-left: 3px solid transparent;
}}

QListWidget::item:selected {{
    background-color: {_SELECTION_BG};
    color: {_TEXT};
    border-left: 3px solid {_ACCENT};
    font-weight: 600;
}}

QTableView::item:selected {{
    background-color: {_SELECTION_BG};
    color: {_TEXT};
}}

QHeaderView::section {{
    background-color: {_SUBTLE_BACKGROUND};
    border: none;
    border-right: 1px solid {_BORDER};
    border-bottom: 1px solid {_BORDER};
    padding: 5px 6px;
    font-weight: 600;
    letter-spacing: 0.2px;
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

QToolButton[collapsible="true"] {{
    background-color: transparent;
    border: none;
    text-align: left;
    padding: 4px 2px;
    font-weight: 600;
    color: {_MUTED_TEXT};
}}

QToolButton[collapsible="true"]:hover {{
    color: {_TEXT};
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


STALE_COLOR = _STALE


def apply_style(app: QApplication) -> None:
    app.setStyleSheet(MAIN_STYLESHEET)
