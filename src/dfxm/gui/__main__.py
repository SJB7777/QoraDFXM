"""Entry point:  python -m dfxm.gui  /  dfxm-gui  /  dfxm gui

The GUI is one of two front-ends over the same Core engine; the other is
:mod:`dfxm.cli`. A frozen single-file build ships both, so ``--cli`` here
routes straight to the command line (see ``gui.cli_bridge._launcher``).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(files=None) -> int:
    argv = list(sys.argv[1:]) if files is None else [str(f) for f in files]

    if argv and argv[0] == "--cli":
        from ..cli import main as cli_main

        return cli_main(argv[1:])

    from PySide6 import QtGui, QtWidgets

    from .main_window import APP_NAME, MainWindow, logo_icon

    # Make Windows use our own icon in the taskbar (not the python.exe icon).
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("dfxm.opticalc")
    except (AttributeError, OSError):
        pass  # not Windows, or the shell refused — cosmetic only

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(logo_icon())
    # Explicit point-size base font avoids pixel-font -1 pointSize warnings.
    app.setFont(QtGui.QFont("Segoe UI", 9))
    win = MainWindow()
    win.show()
    for path in argv:
        win.open_path(Path(path))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
