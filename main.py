import sys
import os

os.environ["QT_LOGGING_RULES"] = "*.debug=false"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from gui import InferenceGUI


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    window = InferenceGUI()
    window.show()

    window._refresh_checkpoints()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()