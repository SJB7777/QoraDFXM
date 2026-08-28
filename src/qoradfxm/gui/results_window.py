"""Standalone, resizable window for the Master results table.

The table is large, so it lives in its own top-level window rather than a
cramped bottom dock. It renders the SAME MasterTableModel as everywhere else,
so edits / fits stay in sync live. Row-click still switches the active shot.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


class DropTableView(QtWidgets.QTableView):
    """QTableView that accepts file drops (from the sidebar or the OS).

    Emits :attr:`filesDropped` with the dropped local paths so the host can
    auto-create rows with basic info (filename, etc.).
    """

    filesDropped = QtCore.Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)

    @staticmethod
    def _paths(e) -> list[Path]:
        md = e.mimeData()
        if not md.hasUrls():
            return []
        return [Path(u.toLocalFile()) for u in md.urls() if u.isLocalFile()]

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e) -> None:
        paths = self._paths(e)
        if paths:
            e.acceptProposedAction()
            self.filesDropped.emit(paths)
        else:
            super().dropEvent(e)


_SPREADSHEET_QSS = """
QTableView {
    background: #ffffff;
    alternate-background-color: #f5f8fc;
    gridline-color: #d5dbe2;
    color: #1f2328;
    selection-background-color: #cfe3ff;
    selection-color: #1f2328;
    outline: 0;
    font-size: 9pt;
}
QTableView::item { padding: 2px 6px; border: none; }
QTableView::item:selected { background: #cfe3ff; }
QHeaderView::section {
    background: #eef1f5;
    color: #3a3f45;
    padding: 5px 8px;
    border: none;
    border-right: 1px solid #d5dbe2;
    border-bottom: 1px solid #c3cbd4;
    font-weight: 600;
}
QHeaderView::section:hover { background: #e3e8ef; }
QTableView QTableCornerButton::section {
    background: #eef1f5;
    border: none;
    border-right: 1px solid #d5dbe2;
    border-bottom: 1px solid #c3cbd4;
}
"""


def _style_spreadsheet(table: QtWidgets.QTableView) -> None:
    """Give the table a clean, commercial-spreadsheet (Excel-like) look."""
    table.setStyleSheet(_SPREADSHEET_QSS)
    table.setShowGrid(True)
    table.setAlternatingRowColors(True)
    table.setCornerButtonEnabled(True)
    table.verticalHeader().setDefaultSectionSize(24)
    table.verticalHeader().setStyleSheet(
        "QHeaderView::section { background:#eef1f5; color:#7a828b;"
        " border:none; border-right:1px solid #c3cbd4;"
        " border-bottom:1px solid #d5dbe2; padding:0 6px; }"
    )
    table.horizontalHeader().setHighlightSections(False)
    table.setEditTriggers(
        QtWidgets.QAbstractItemView.DoubleClicked
        | QtWidgets.QAbstractItemView.EditKeyPressed
    )


class ResultsWindow(QtWidgets.QMainWindow):
    def __init__(
        self, table: QtWidgets.QTableView, actions, settings, parent=None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("결과 표 — Master Table")
        self.setWindowFlag(QtCore.Qt.Window, True)
        self._settings = settings

        tb = QtWidgets.QToolBar("데이터")
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        tb.setIconSize(QtCore.QSize(18, 18))
        for act in actions:
            if act is None:
                tb.addSeparator()
            else:
                tb.addAction(act)
        self.addToolBar(tb)

        _style_spreadsheet(table)
        self.setCentralWidget(table)
        self._status = self.statusBar()

        if (geo := settings.value("results_win_geo")) is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(960, 620)

    def show_raise(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self._settings.setValue("results_win_geo", self.saveGeometry())
        super().closeEvent(e)
