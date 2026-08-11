"""Analysis ROI objects (Phase B-1).

A small class hierarchy that turns the previously single-instance ellipse / line
overlays into uniquely-identified, managed objects — the data model the ParaView
style object tree (later phases) will bind to.

``AnalysisROI`` is a metadata mixin (uuid, name, type, visible, locked) that is
combined with a concrete pyqtgraph ROI (``EllipseROI`` / ``LineSegmentROI``,
both ``pg.ROI`` subclasses).  Concrete init calls the pyqtgraph base explicitly
(pyqtgraph ROIs do not use cooperative ``super().__init__``), then attaches the
analysis metadata and wires ``sigRegionChanged`` so the object keeps its own
geometry in sync while the user drags it.
"""

from __future__ import annotations

import uuid

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

# Distinct, easy-to-tell-apart pens (app accent = cyan for hover/selected).
ELLIPSE_PEN = pg.mkPen("#ff375f", width=2)
ELLIPSE_HOVER_PEN = pg.mkPen("#00e5ff", width=2)
ELLIPSE_HANDLE_PEN = pg.mkPen("#ff9db1", width=1.5)

LINE_PEN = pg.mkPen("#facc15", width=2)
LINE_HOVER_PEN = pg.mkPen("#00e5ff", width=2)

RECT_PEN = pg.mkPen("#4ade80", width=2)
RECT_HOVER_PEN = pg.mkPen("#00e5ff", width=2)


class AnalysisROI:
    """Metadata + shared behaviour mixed into a concrete pyqtgraph ROI."""

    def _init_analysis(self, name: str, roi_type: str) -> None:
        self.roi_id: str = uuid.uuid4().hex[:8]
        self.name: str = name
        self.roi_type: str = roi_type  # "ellipse" | "line"
        self._visible: bool = True
        self._locked: bool = False
        # Keep internal geometry current while the user drags the ROI.
        self.sigRegionChanged.connect(self._on_region_changed)

    # --- visibility / lock -------------------------------------------------
    @property
    def visible(self) -> bool:
        return self._visible

    def set_visible(self, on: bool) -> None:
        self._visible = bool(on)
        self.setVisible(bool(on))

    @property
    def locked(self) -> bool:
        return self._locked

    def set_locked(self, on: bool) -> None:
        """Locked ROIs cannot be moved / resized and hide their handles."""
        self._locked = bool(on)
        self.translatable = not on
        self.resizable = not on
        self.rotateAllowed = not on
        for h in self.getHandles():
            h.setVisible(not on)
        self.update()

    # --- data ---------------------------------------------------------------
    def _on_region_changed(self, *_) -> None:
        self.update_data()

    def update_data(self) -> None:  # overridden by subclasses
        pass

    def rename(self, name: str) -> None:
        self.name = name

    def to_dict(self) -> dict:
        return {
            "roi_id": self.roi_id,
            "name": self.name,
            "roi_type": self.roi_type,
            "visible": self._visible,
            "locked": self._locked,
        }


class EllipseFitROI(pg.EllipseROI, AnalysisROI):
    """Editable ellipse initialised from a least-squares fit result."""

    def __init__(self, geometry: dict, points=None, name: str = "Ellipse", **kw):
        a = float(geometry["semi_major_axis"])
        b = float(geometry["semi_minor_axis"])
        cx = float(geometry["center_x"])
        cy = float(geometry["center_y"])
        angle = float(geometry.get("angle_major_from_x_deg", 0.0))

        # EllipseROI takes the (unrotated) bounding-box origin + size.
        pg.EllipseROI.__init__(
            self,
            [cx - a, cy - b],
            [2 * a, 2 * b],
            pen=ELLIPSE_PEN,
            hoverPen=ELLIPSE_HOVER_PEN,
            handlePen=ELLIPSE_HANDLE_PEN,
            rotatable=True,
            **kw,
        )
        self.setAngle(angle, center=[0.5, 0.5])  # rotate about the ellipse centre

        self._init_analysis(name, "ellipse")
        self.geometry: dict = dict(geometry)
        pts = [] if points is None else list(points)
        self.fit_points: list[tuple[float, float]] = [
            (float(x), float(y)) for x, y in pts
        ]
        self.update_data()

    def update_data(self) -> None:
        st = self.getState()
        sx, sy = float(st["size"][0]), float(st["size"][1])
        angle = float(st["angle"])
        # Centre in parent (data) coords — correct under any rotation.
        c = self.mapToParent(QtCore.QPointF(sx / 2.0, sy / 2.0))
        self.geometry.update(
            {
                "center_x": c.x(),
                "center_y": c.y(),
                "semi_major_axis": sx / 2.0,
                "semi_minor_axis": sy / 2.0,
                "angle_major_from_x_deg": angle,
                "major_diameter": sx,
                "minor_diameter": sy,
            }
        )


class LineProfileROI(pg.LineSegmentROI, AnalysisROI):
    """Two-point line used to sample an intensity profile."""

    def __init__(self, positions, name: str = "Line", **kw):
        pg.LineSegmentROI.__init__(
            self, positions, pen=LINE_PEN, hoverPen=LINE_HOVER_PEN, **kw
        )
        self._init_analysis(name, "line")
        self.p1: tuple[float, float] = (float(positions[0][0]), float(positions[0][1]))
        self.p2: tuple[float, float] = (float(positions[1][0]), float(positions[1][1]))
        self.update_data()

    def update_data(self) -> None:
        handles = self.getHandles()
        if len(handles) >= 2:
            a = self.mapToParent(handles[0].pos())
            b = self.mapToParent(handles[1].pos())
            self.p1 = (a.x(), a.y())
            self.p2 = (b.x(), b.y())

    def length(self) -> float:
        return float(np.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]))


class RectRegionROI(pg.RectROI, AnalysisROI):
    """Rectangular region-of-interest (crop / analysis area)."""

    def __init__(self, pos, size, name: str = "Region", **kw):
        pg.RectROI.__init__(
            self, pos, size, pen=RECT_PEN, hoverPen=RECT_HOVER_PEN, **kw
        )
        self.addScaleHandle([0, 0], [1, 1])  # extra corner handle
        self._init_analysis(name, "rect")
        self.bounds = (float(pos[0]), float(pos[1]), float(size[0]), float(size[1]))
        self.update_data()

    def update_data(self) -> None:
        st = self.getState()
        p, s = st["pos"], st["size"]
        self.bounds = (float(p.x()), float(p.y()), float(s[0]), float(s[1]))

    def region_slice(self):
        """(row_slice, col_slice) for indexing the image array, clamped >=0."""
        x, y, wd, ht = self.bounds
        x0, y0 = round(min(x, x + wd)), round(min(y, y + ht))
        x1, y1 = round(max(x, x + wd)), round(max(y, y + ht))
        return slice(max(0, y0), max(0, y1)), slice(max(0, x0), max(0, x1))
