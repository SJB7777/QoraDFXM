"""Qt adapters that expose Core data objects to the UI. View-layer only.

``MasterTableModel`` is a QAbstractTableModel backed by a Core
:class:`~dfxm.core.results.ResultsFrame` (the real data). The QTableView renders
this; all data lives in Core.
"""

from __future__ import annotations

import json

import pandas as pd
from PySide6 import QtCore, QtGui

from ..core.results import MASTER_COLUMNS, ResultsFrame

_FLOAT_COLS = {
    "center_x",
    "center_y",
    "major_axis",
    "minor_axis",
    "angle_deg",
    "fit_error",
}

# Friendly, intuitive Korean headers (internal keys stay English for CSV/Core).
COLUMN_LABELS = {
    "shot_id": "샷 이름",
    "status": "상태",
    "center_x": "중심 X",
    "center_y": "중심 Y",
    "major_axis": "장축",
    "minor_axis": "단축",
    "angle_deg": "각도 (°)",
    "fit_error": "맞춤 오차",
    "points_json": "점 데이터",
    "bg_applied": "배경차감",
    "log_scale": "로그",
}


class MasterTableModel(QtCore.QAbstractTableModel):
    """Qt view-model over a Core ResultsFrame."""

    def __init__(self, results: ResultsFrame | None = None, parent=None) -> None:
        super().__init__(parent)
        self._results = results or ResultsFrame()

    # --- Core access -------------------------------------------------------
    @property
    def results(self) -> ResultsFrame:
        return self._results

    @property
    def df(self):
        return self._results.df

    def set_results(self, results: ResultsFrame) -> None:
        self.beginResetModel()
        self._results = results
        self.endResetModel()

    # --- Qt interface ------------------------------------------------------
    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._results)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(MASTER_COLUMNS)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal:
            key = MASTER_COLUMNS[section]
            if role == QtCore.Qt.DisplayRole:
                return COLUMN_LABELS.get(key, key)
            if role == QtCore.Qt.ToolTipRole:
                return key  # internal/CSV name
            return None
        if role == QtCore.Qt.DisplayRole:
            return section + 1
        return None

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        col = MASTER_COLUMNS[index.column()]
        val = self._results.df.iat[index.row(), index.column()]

        # Dim excluded rows so they read as "skip me".
        if role == QtCore.Qt.ForegroundRole:
            status = str(self._results.get_cell(index.row(), "status") or "")
            if status == "EXCLUDE":
                return QtGui.QBrush(QtGui.QColor("#6b7280"))
            return None

        if role == QtCore.Qt.EditRole:
            if pd.isna(val):
                return ""
            if col in ("bg_applied", "log_scale"):
                return "Y" if bool(val) else "N"
            return str(val)

        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.ToolTipRole):
            if pd.isna(val):
                return ""
            if col in _FLOAT_COLS:
                return f"{float(val):.3f}"
            if col in ("bg_applied", "log_scale"):
                return "Y" if bool(val) else "N"
            if col == "points_json" and role == QtCore.Qt.DisplayRole:
                try:
                    return f"{len(json.loads(val))} pts"
                except Exception:
                    return str(val)
            return str(val)
        return None

    # --- editing (manual data cleanup) -------------------------------------
    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        return (
            QtCore.Qt.ItemIsEnabled
            | QtCore.Qt.ItemIsSelectable
            | QtCore.Qt.ItemIsEditable
        )

    def setData(self, index, value, role=QtCore.Qt.EditRole) -> bool:
        if role != QtCore.Qt.EditRole or not index.isValid():
            return False
        col = MASTER_COLUMNS[index.column()]
        text = str(value).strip()
        try:
            if col in _FLOAT_COLS:
                coerced = float(text) if text else float("nan")
            elif col in ("bg_applied", "log_scale"):
                coerced = text.lower() in ("y", "true", "1", "yes")
            else:
                coerced = text
        except ValueError:
            return False  # reject non-numeric input in a float cell
        self._results.update_cell(index.row(), col, coerced)
        self.dataChanged.emit(index, index, [QtCore.Qt.DisplayRole])
        return True

    def data_for_edit(self, index):
        return self._results.df.iat[index.row(), index.column()]

    # --- mutation (wraps Core, emits Qt signals) ---------------------------
    def add_row(self, row: dict) -> int:
        r = len(self._results)
        self.beginInsertRows(QtCore.QModelIndex(), r, r)
        self._results.add_row(row)
        self.endInsertRows()
        return r

    def upsert_row(self, row: dict) -> tuple[int, bool]:
        """Update the shot_id's row in place (dataChanged) or append it."""
        r = self._results.index_of(row.get("shot_id"))
        if r is None:
            return self.add_row(row), True
        self._results.upsert_row(row)
        left = self.index(r, 0)
        right = self.index(r, len(MASTER_COLUMNS) - 1)
        self.dataChanged.emit(left, right, [QtCore.Qt.DisplayRole])
        return r, False

    def index_of(self, shot_id: str) -> int | None:
        return self._results.index_of(shot_id)

    def shot_id_at(self, r: int):
        return self._results.get_cell(r, "shot_id")

    def status_at(self, r: int):
        return self._results.get_cell(r, "status")

    def remove_row(self, r: int) -> None:
        if not 0 <= r < len(self._results):
            return
        self.beginRemoveRows(QtCore.QModelIndex(), r, r)
        self._results.remove_row(r)
        self.endRemoveRows()

    def set_status(self, r: int, status: str) -> None:
        if not 0 <= r < len(self._results):
            return
        self._results.set_status(r, status)
        idx = self.index(r, MASTER_COLUMNS.index("status"))
        self.dataChanged.emit(idx, idx, [QtCore.Qt.DisplayRole])

    def clear(self) -> None:
        self.beginResetModel()
        self._results.clear()
        self.endResetModel()

    def to_csv(self, path: str) -> None:
        self._results.to_csv(path)
