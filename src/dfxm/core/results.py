"""Fit results + the Master results frame (pandas). No Qt here.

``ResultsFrame`` is the single source of truth for fit results (spec req. 4).
The GUI's QTableView is a *view* onto it via a thin Qt adapter in
``dfxm/gui/models.py``; the CLI writes it straight to CSV.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import fitting

# Column order IS the schema. Kept in sync with CSV / SQLite round-trips.
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


@dataclass
class FitResult:
    """One ellipse fit: geometry + quality + the points it came from."""

    center_x: float
    center_y: float
    major_axis: float
    minor_axis: float
    angle_deg: float
    fit_error: float
    points: list = field(default_factory=list)
    geom: dict = field(default_factory=dict)

    @classmethod
    def from_points(cls, points) -> FitResult:
        """Fit an ellipse to picked (x, y) points and build a result."""
        pts = np.asarray(points, dtype=float)
        coeffs = fitting.fit_ellipse(pts[:, 0], pts[:, 1])
        geom = fitting.conic_to_geometry(coeffs)
        err = fitting.rms_error(coeffs, pts[:, 0], pts[:, 1])
        return cls(
            center_x=geom["center_x"],
            center_y=geom["center_y"],
            major_axis=geom["major_diameter"],
            minor_axis=geom["minor_diameter"],
            angle_deg=geom["angle_major_from_x_deg"],
            fit_error=err,
            points=pts.tolist(),
            geom=geom,
        )

    def to_row(
        self, shot_id="", status="OK", bg_applied=False, log_scale=False
    ) -> dict:
        return {
            "shot_id": shot_id,
            "status": status,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "major_axis": self.major_axis,
            "minor_axis": self.minor_axis,
            "angle_deg": self.angle_deg,
            "fit_error": self.fit_error,
            "points_json": json.dumps(self.points),
            "bg_applied": bool(bg_applied),
            "log_scale": bool(log_scale),
        }

    def to_dict(self) -> dict:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "major_axis": self.major_axis,
            "minor_axis": self.minor_axis,
            "angle_deg": self.angle_deg,
            "fit_error": self.fit_error,
            "points": self.points,
            "geom": self.geom,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FitResult:
        return cls(
            center_x=d["center_x"],
            center_y=d["center_y"],
            major_axis=d["major_axis"],
            minor_axis=d["minor_axis"],
            angle_deg=d["angle_deg"],
            fit_error=d["fit_error"],
            points=d.get("points", []),
            geom=d.get("geom", {}),
        )


def empty_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=_DTYPES[c]) for c in MASTER_COLUMNS})
    return df[MASTER_COLUMNS]


class ResultsFrame:
    """Pure pandas wrapper around the Master DataFrame. No Qt."""

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        self._df = empty_frame() if df is None else self._coerce(df)

    @staticmethod
    def _coerce(df: pd.DataFrame) -> pd.DataFrame:
        for c in MASTER_COLUMNS:
            if c not in df.columns:
                df[c] = pd.Series(dtype=_DTYPES[c])
        return df[MASTER_COLUMNS].reset_index(drop=True)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def add_row(self, row: dict) -> int:
        r = len(self._df)
        self._df.loc[r, MASTER_COLUMNS] = [row.get(c) for c in MASTER_COLUMNS]
        self._df = self._df[MASTER_COLUMNS]
        return r

    def index_of(self, shot_id: str) -> int | None:
        """Row index for a shot_id (Phase 2: 1:1 shot ↔ row), or None."""
        hits = self._df.index[self._df["shot_id"] == shot_id].tolist()
        return int(hits[0]) if hits else None

    def upsert_row(self, row: dict) -> tuple[int, bool]:
        """Update the row with this shot_id in place, or append it.

        Returns (row_index, created). Preserves an existing 'status' unless the
        incoming row carries one (so an EXCLUDE flag survives a re-fit)."""
        r = self.index_of(row.get("shot_id"))
        if r is None:
            return self.add_row(row), True
        for c in MASTER_COLUMNS:
            if c in row and row[c] is not None:
                self._df.iat[r, MASTER_COLUMNS.index(c)] = row[c]
        return r, False

    def update_cell(self, r: int, col: str, value) -> None:
        if 0 <= r < len(self._df) and col in MASTER_COLUMNS:
            self._df.iat[r, MASTER_COLUMNS.index(col)] = value

    def get_cell(self, r: int, col: str):
        if 0 <= r < len(self._df) and col in MASTER_COLUMNS:
            return self._df.iat[r, MASTER_COLUMNS.index(col)]
        return None

    def remove_row(self, r: int) -> None:
        if 0 <= r < len(self._df):
            self._df = self._df.drop(self._df.index[r]).reset_index(drop=True)

    def set_status(self, r: int, status: str) -> None:
        if 0 <= r < len(self._df):
            self._df.iat[r, MASTER_COLUMNS.index("status")] = status

    def clear(self) -> None:
        self._df = empty_frame()

    def to_csv(self, path: str) -> None:
        self._df.to_csv(path, index=False, encoding="utf-8-sig")

    @classmethod
    def from_csv(cls, path: str) -> ResultsFrame:
        return cls(pd.read_csv(path))
