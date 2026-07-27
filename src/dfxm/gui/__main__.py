"""Entry point:  python -m dfxm.gui"""

from __future__ import annotations

import sys

from PySide6 import QtGui, QtWidgets

from .main_window import APP_NAME, MainWindow, logo_icon


def main() -> int:
    # Make Windows use our own icon in the taskbar (not the python.exe icon).
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("dfxm.opticalc")
    except Exception:
        pass

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(logo_icon())
    # Explicit point-size base font avoids pixel-font -1 pointSize warnings.
    app.setFont(QtGui.QFont("Segoe UI", 9))
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
