import sys

from PySide6.QtWidgets import QApplication

from gnovi_plot.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
