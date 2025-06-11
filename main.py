import sys
from PyQt6.QtWidgets import QApplication
from audiogui import FFTApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FFTApp()
    win.resize(800, 400)
    win.show()
    sys.exit(app.exec())
