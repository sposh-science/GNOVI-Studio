import sys

from PySide6.QtWidgets import QApplication

from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.styles import apply_style


def main():
    app = QApplication(sys.argv)
    apply_style(app)
    window = MainWindow()
    window.show()
    return app.exec()
