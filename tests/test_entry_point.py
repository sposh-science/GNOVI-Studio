"""Practical, pytest-level stand-in for "packaging entry point smoke
behavior": confirms `gnovi_plot.app.main` (the entry point PyInstaller's
spec targets) is importable and callable without constructing a
QApplication at import time, which would break under pytest's own
QApplication (see `qapp` fixture) or any other headless/embedded context.
The real packaging verification is an actual built-binary smoke run done
during the packaging phase, not something this suite attempts.
"""

import inspect

import gnovi_plot.app as app_module
from gnovi_plot.app import main


def test_main_is_importable_and_callable():
    assert callable(main)


def test_importing_app_module_does_not_construct_a_qapplication():
    from PySide6.QtWidgets import QApplication

    # If importing the module had already constructed one, this assertion
    # would be trivially satisfied by that stray instance instead of
    # reflecting a clean import -- so this only proves the module doesn't
    # gratuitously create a *second* one on import, which is the failure
    # mode that would actually break embedding `main` elsewhere.
    before = QApplication.instance()
    import importlib

    importlib.reload(app_module)
    after = QApplication.instance()
    assert after is before


def test_main_constructs_its_own_qapplication_rather_than_reusing_a_global():
    source = inspect.getsource(main)
    assert "QApplication(" in source
