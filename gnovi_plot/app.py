import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from gnovi_plot.gui.main_window import MainWindow

_logger = logging.getLogger("gnovi_plot")


def _install_excepthook() -> None:
    """Safety net for genuinely unanticipated exceptions: log the full
    traceback (terminal, via the `gnovi_plot` logger) and show one generic
    dialog instead of letting Qt's default hook silently print-and-continue
    or leave the app in an inconsistent state. The many targeted `except`
    blocks throughout the GUI remain the primary, precise error UX -- this
    only catches what they don't.
    """

    def _handle(exc_type, exc_value, exc_traceback):
        _logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        QMessageBox.critical(
            None,
            "Unexpected Error",
            "Gnovi Studio hit an unexpected error and this action could not "
            "complete. Details were logged to the terminal.",
        )

    sys.excepthook = _handle


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    _install_excepthook()
    window = MainWindow()
    window.show()
    return app.exec()
