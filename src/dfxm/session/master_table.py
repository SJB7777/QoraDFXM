"""Master results table — the single source of truth for ellipse-fit results.

A pandas DataFrame holds every fit row; the on-screen QTableView is only a
*view* onto this frame (spec requirement 4).  Phase 1 fills it 1-way via the
[표에 추가] button and exports to CSV; later phases sync rows <-> shots and
persist the frame into the SQLite `.dfxm_proj` session file.
"""

from __future__ import annotations

import json

import pandas as pd
from PySide6 import QtCore

# Column order IS the schema. Keep in sync with any CSV / SQLite round-trip.
MASTER_COLUMNS = [
    "shot_id",
    "status",
    "center_x",
    "center_y",
    "major_axis",
    "minor_axis",
    "angle_deg",
    "fit_error",
    "points_json",
    "bg_applied",
    "log_scale",
]

# Numeric columns get fixed-precision display; everything else is shown as-is.
_FLOAT_COLS = {
    "center_x",
    "center_y",
    "major_axis",
    "minor_axis",
    "angle_deg",
    "fit_error",
}

_DTYPES = {
    "shot_id": "object",
    "status": "object",
    "center_x": "float64",
    "center_y": "float64",
    "major_axis": "float64",
    "minor_axis": "float64",
    "angle_deg": "float64",
    "fit_error": "float64",
    "points_json": "object",
    "bg_applied": "bool",
    "log_scale": "bool",
}


def _empty_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=_DTYPES[c]) for c in MASTER_COLUMNS})
    return df[MASTER_COLUMNS]


class MasterTableModel(QtCore.QAbstractTableModel):
    """Qt model wrapping the Master DataFrame. QTableView renders this."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._df: pd.DataFrame = _empty_frame()

    # --- pandas access -----------------------------------------------------
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def set_frame(self, df: pd.DataFrame) -> None:
        """Replace the whole frame (used by CSV/SQLite load)."""
        self.beginResetModel()
        for c in MASTER_COLUMNS:
            if c not in df.columns:
                df[c] = pd.Series(dtype=_DTYPES[c])
        self._df = df[MASTER_COLUMNS].reset_index(drop=True)
        self.endResetModel()

    # --- Qt model interface ------------------------------------------------
    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(MASTER_COLUMNS)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return MASTER_COLUMNS[section]
        return section + 1

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        col = MASTER_COLUMNS[index.column()]
        val = self._df.iat[index.row(), index.column()]

        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.ToolTipRole):
            if pd.isna(val):
                return ""
            if col in _FLOAT_COLS:
                return f"{float(val):.3f}"
            if col in ("bg_applied", "log_scale"):
                return "Y" if bool(val) else "N"
            if col == "points_json" and role == QtCore.Qt.DisplayRole:
                # keep the cell compact; full JSON is in the tooltip
                try:
                    return f"{len(json.loads(val))} pts"
                except Exception:
                    return str(val)
            return str(val)
        return None

    # --- mutation ----------------------------------------------------------
    def add_row(self, row: dict) -> int:
        """Append one fit result. Missing keys become NaN/None. Returns row idx."""
        r = len(self._df)
        self.beginInsertRows(QtCore.QModelIndex(), r, r)
        self._df.loc[r, MASTER_COLUMNS] = [row.get(c) for c in MASTER_COLUMNS]
        self._df = self._df[MASTER_COLUMNS]
        self.endInsertRows()
        return r

    def remove_row(self, r: int) -> None:
        if not 0 <= r < len(self._df):
            return
        self.beginRemoveRows(QtCore.QModelIndex(), r, r)
        self._df = self._df.drop(self._df.index[r]).reset_index(drop=True)
        self.endRemoveRows()

    def set_status(self, r: int, status: str) -> None:
        if not 0 <= r < len(self._df):
            return
        self._df.iat[r, MASTER_COLUMNS.index("status")] = status
        idx = self.index(r, MASTER_COLUMNS.index("status"))
        self.dataChanged.emit(idx, idx, [QtCore.Qt.DisplayRole])

    def clear(self) -> None:
        self.beginResetModel()
        self._df = _empty_frame()
        self.endResetModel()

    # --- export ------------------------------------------------------------
    def to_csv(self, path: str) -> None:
        self._df.to_csv(path, index=False, encoding="utf-8-sig")
