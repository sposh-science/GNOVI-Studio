import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path):
    """MainWindow persists the theme choice via `QSettings("GnoviStudio",
    "GnoviStudio")`. Without this, every test that constructs a MainWindow
    would read/write the developer's real OS-level settings store. Redirect
    to a throwaway per-test directory instead, so the suite never touches
    real user config and tests can't leak theme state into each other."""
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path))
    yield
