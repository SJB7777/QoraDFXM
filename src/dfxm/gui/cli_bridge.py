"""Run CLI jobs from the GUI.

The GUI builds a recipe interactively; batch work then goes back out through
the *same* command line a user could have typed (:mod:`dfxm.cli`), spawned as a
child process. Reasons for a subprocess rather than an in-process call:

* a long batch never blocks the event loop, and can be cancelled;
* a crash in a job (bad file, out-of-memory) does not take the app down;
* the exact command is printable — reproducible outside the GUI.

    job = CliJob(self)
    job.line.connect(self.log)
    job.finished_ok.connect(lambda code: ...)
    job.start(argv_for_dataset(doc.ds, points_file=pts, out="results.csv"))
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtCore

# Re-exported so GUI code has one import for "build a job and run it".
from ..cli.spec import argv_for_dataset, argv_for_fit

__all__ = ["CliJob", "argv_for_dataset", "argv_for_fit", "command_line"]


def _launcher() -> list[str]:
    """How to invoke the CLI: the interpreter, or the frozen exe itself."""
    if getattr(sys, "frozen", False):  # Nuitka / PyInstaller build
        return [sys.executable, "--cli"]
    return [sys.executable, "-m", "dfxm.cli"]


def command_line(argv: list[str]) -> str:
    """The job as a copy-pasteable one-liner (for logs / 'show command')."""
    parts = ["dfxm", *argv]
    return " ".join(f'"{p}"' if " " in p else p for p in parts)


class CliJob(QtCore.QObject):
    """One ``dfxm ...`` child process, streamed back as signals."""

    line = QtCore.Signal(str)  # one line of stdout/stderr
    finished_ok = QtCore.Signal(int)  # exit code (0 == success)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._proc = QtCore.QProcess(self)
        self._proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._drain)
        self._proc.finished.connect(lambda code, _st: self.finished_ok.emit(code))

    def start(self, argv: list[str], cwd: Path | str | None = None) -> None:
        launcher = _launcher()
        if cwd:
            self._proc.setWorkingDirectory(str(cwd))
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")  # Korean labels on Windows consoles
        self._proc.setProcessEnvironment(env)
        self.line.emit(f"$ {command_line(argv)}")
        self._proc.start(launcher[0], [*launcher[1:], *argv])

    def cancel(self) -> None:
        if self._proc.state() != QtCore.QProcess.NotRunning:
            self._proc.kill()

    def is_running(self) -> bool:
        return self._proc.state() != QtCore.QProcess.NotRunning

    def _drain(self) -> None:
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        for raw in data.splitlines():
            if raw.strip():
                self.line.emit(raw.rstrip())
