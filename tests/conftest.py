import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _auto_discard_unsaved_project(monkeypatch):
    """MainWindow.closeEvent shows a modal Save/Discard/Cancel QMessageBox
    when the project is dirty (see gui.main_window._confirm_discard_unsaved).
    Without this, any test that mutates project state and then calls
    `window.close()` -- dozens do, e.g. test_gui_responsiveness.py -- would
    hang forever offscreen waiting for a button click that never comes.
    Auto-answer Discard by default so `close()` behaves like it did before
    project persistence existed; a test targeting the prompt itself
    re-patches `QMessageBox.warning` within its own body to override this."""
    monkeypatch.setattr(
        "gnovi_plot.gui.main_window.QMessageBox.warning",
        lambda *args, **kwargs: QMessageBox.Discard,
    )


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
