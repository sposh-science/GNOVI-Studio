import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def gui_widget(qapp):
    """Own the transient top-level widgets a test builds, so their Qt (C++)
    lifetime ends at this fixture's teardown -- the test's own ownership
    boundary -- rather than whenever the test function's local reference
    happens to be released.

    Matplotlib's Qt backend arms a one-shot `FigureCanvasQTAgg._draw_idle`
    timer on every draw. If the widget (and thus its canvas) is freed before
    the event loop next spins, that callback fires against an already-deleted
    C++ object: a `RuntimeError` that Python 3.13 escalates to a fatal
    `SystemError` in whichever test next pumps the loop. Spinning the loop
    once here while the canvas is still alive lets the pending draw finish;
    `deleteLater()` then tears the widget down deterministically.

    Usage: `widget = gui_widget(ResidualPlotWidget())`.
    """
    owned = []

    def own(widget):
        owned.append(widget)
        return widget

    yield own
    qapp.processEvents()
    for widget in owned:
        widget.deleteLater()
    qapp.processEvents()


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


@pytest.fixture(autouse=True)
def _drain_qt_between_tests():
    """Keep asynchronous Qt/Matplotlib work from one test out of the next.

    The `qapp` QApplication is session-scoped, so anything queued on its
    event loop outlives the test that queued it. In particular Matplotlib's
    Qt backend schedules a one-shot `FigureCanvasQTAgg._draw_idle()` timer,
    which keeps its (now unreferenced) canvas alive; if that callback later
    fires against a canvas whose C++ half has been torn down it raises
    `RuntimeError`. Python 3.12 masks it (Matplotlib swallows it); Python
    3.13 re-raises it as an uncatchable `SystemError` inside whichever test
    happens to pump the loop next.

    A closed `QMainWindow` / bare `QWidget` also stays owned by the
    QApplication (it is not a Python reference, so `gc` cannot reclaim it):
    over a full run the session would otherwise hold thousands of hidden
    windows, which both leaks memory and slows every later Qt call.

    So after every test: pump the loop (fire queued callbacks while their
    objects are still alive), schedule every leftover top-level widget for
    deletion, run those deletions, then collect -- leaving the next test an
    empty event loop and no lingering widgets. `deleteLater()` (not
    `close()`) is used so no `closeEvent` handler runs here.
    """
    yield
    app = QApplication.instance()
    if app is None:
        return
    app.processEvents()
    app.processEvents()
    for widget in app.topLevelWidgets():
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()
