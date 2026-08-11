"""Interactive image view: display, log scaling, histogram/levels,
over-max highlighting and interactive shape fitting.

Built on pyqtgraph.  A :class:`HistogramLUTItem` gives the histogram plus
draggable min/max level bars (requirement 3).  Pixels above the chosen max
are painted red by a separate RGBA overlay so the highlight is independent
of the chosen colormap (requirement 4).  A log-scale toggle applies
:func:`dfxm.transform.adaptive_log` (requirement 2).  Ellipse fitting reuses
the least-squares fitter in :mod:`dfxm.ellipse_fit` (requirement 6).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import fitting as ef

# White-on-black scientific look, sensible defaults.
pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

# Colormaps offered in the toolbar. "gray" first so plain frames look normal.
COLORMAPS = ["gray", "viridis", "inferno", "magma", "cividis", "turbo"]


def _nice_length(x: float) -> float:
    """Round a length to the nearest 1/2/5 x 10^n, ImageJ scale-bar style."""
    if x <= 0:
        return 1.0
    exp = np.floor(np.log10(x))
    base = x / 10**exp
    nice = 1.0 if base < 1.5 else 2.0 if base < 3.5 else 5.0 if base < 7.5 else 10.0
    return nice * 10**exp


def _fmt_length(x: float) -> str:
    return f"{x:g}"


class _FloatHist(QtWidgets.QWidget):
    """Top-level window hosting a detached histogram; signals when closed."""

    closed = QtCore.Signal()

    def closeEvent(self, e) -> None:
        self.closed.emit()
        super().closeEvent(e)


class DrawViewBox(pg.ViewBox):
    """ViewBox that, in ``draw_mode``, turns a left-drag into a rubber-band
    rectangle instead of a pan (used by the region tool)."""

    drawStarted = QtCore.Signal(object)  # start point (view coords)
    drawUpdated = QtCore.Signal(object, object)  # p0, current
    drawFinished = QtCore.Signal(object, object)  # p0, p1

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.draw_mode = False
        self._p0 = None

    def mouseDragEvent(self, ev, axis=None):
        if self.draw_mode and ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            cur = self.mapSceneToView(ev.scenePos())
            if ev.isStart():
                self._p0 = self.mapSceneToView(ev.buttonDownScenePos())
                self.drawStarted.emit(self._p0)
            if self._p0 is not None:
                self.drawUpdated.emit(self._p0, cur)
            if ev.isFinish() and self._p0 is not None:
                self.drawFinished.emit(self._p0, cur)
                self._p0 = None
        else:
            super().mouseDragEvent(ev, axis)


class _ScaleBar(pg.ScaleBar):
    """ScaleBar with the label centered BELOW the bar (ImageJ style), so a
    screenshot of the image is self-contained."""

    def updateBar(self) -> None:
        view = self.parentItem()
        if view is None:
            return
        p1 = view.mapFromViewToItem(self, QtCore.QPointF(0, 0))
        p2 = view.mapFromViewToItem(self, QtCore.QPointF(self.size, 0))
        w = (p2 - p1).x()
        self.bar.setRect(QtCore.QRectF(-w, 0, w, self._width))
        self.text.setPos(-w / 2.0, self._width + 4)


_POINT_PEN = pg.mkPen("#22d3ee", width=1.6)
_ELLIPSE_PEN = pg.mkPen("#ff375f", width=1.8)
_AXIS_MAJOR_PEN = pg.mkPen("#facc15", width=1.2, style=QtCore.Qt.DashLine)
_AXIS_MINOR_PEN = pg.mkPen("#4ade80", width=1.2, style=QtCore.Qt.DashLine)


class ImageView(QtWidgets.QWidget):
    """Image display with histogram, level control and ellipse fitting."""

    #: (x, y, value) under the cursor; value is NaN when outside the image.
    cursorMoved = QtCore.Signal(float, float, float)
    #: Emitted with a human-readable report whenever a fit succeeds/clears.
    fitReported = QtCore.Signal(str)
    #: (distance_axis, values, unit) for the current line profile.
    lineProfileChanged = QtCore.Signal(object, object, str)
    #: (segment_lengths, cumulative, unit) for the distance tool.
    distanceChanged = QtCore.Signal(object, object, str)
    #: Current zoom in percent (100 = one image pixel per screen pixel).
    zoomChanged = QtCore.Signal(float)
    #: Emitted when a detached histogram window is closed and re-embedded.
    histogramReattached = QtCore.Signal()
    #: (roi) added to / removed from / edited on the canvas.
    roiAdded = QtCore.Signal(object)
    roiRemoved = QtCore.Signal(str)
    roiChanged = QtCore.Signal(object)
    #: (roi | None) selection changed on the canvas.
    roiSelected = QtCore.Signal(object)
    #: (x0, y0, x1, y1) a line was drawn by two clicks -> create a line ROI.
    lineDrawn = QtCore.Signal(float, float, float, float)
    #: (x0, y0, x1, y1) a rectangle was drawn by drag -> create a region ROI.
    rectDrawn = QtCore.Signal(float, float, float, float)

    #: Valid tool names.  "select" = pointer (default; ESC returns here).
    TOOLS = ("select", "pan", "zoom", "distance", "line", "ellipse", "rect")

    _CURSORS: ClassVar[dict] = {
        "select": QtCore.Qt.ArrowCursor,
        "pan": QtCore.Qt.OpenHandCursor,
        "zoom": QtCore.Qt.CrossCursor,
        "distance": QtCore.Qt.CrossCursor,
        "line": QtCore.Qt.CrossCursor,
        "ellipse": QtCore.Qt.CrossCursor,
        "rect": QtCore.Qt.CrossCursor,
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Raw frame as loaded, and the array actually displayed (may be log).
        self._raw: np.ndarray | None = None
        self._display: np.ndarray | None = None

        self._overmax_enabled = True
        self._overmax_color = (255, 0, 0)

        self._tool = "select"

        # Ellipse-fit state.
        self._points: list[tuple[float, float]] = []
        self._last_geom: dict | None = None

        # Distance-measure state.
        self._dist_points: list[tuple[float, float]] = []
        self._dist_labels: list[pg.TextItem] = []

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground("#1a1a1a")

        self._vb = DrawViewBox()
        self._glw.ci.addItem(self._vb, row=0, col=0)
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)  # image convention: y downward
        self._vb.setBorder(pg.mkPen("#3a3a3a", width=1))  # thin, clean frame
        self._vb.setMouseMode(pg.ViewBox.PanMode)
        self._vb.drawStarted.connect(self._on_rect_draw_start)
        self._vb.drawUpdated.connect(self._on_rect_draw_update)
        self._vb.drawFinished.connect(self._on_rect_draw_finish)

        # Rubber-band rectangle preview (region tool) + two-click line preview.
        self._rect_preview = QtWidgets.QGraphicsRectItem()
        self._rect_preview.setPen(
            pg.mkPen("#4ade80", width=1.5, style=QtCore.Qt.DashLine)
        )
        self._rect_preview.setZValue(40)
        self._rect_preview.setVisible(False)
        self._vb.addItem(self._rect_preview)
        self._line_pending = None
        self._line_preview = pg.PlotCurveItem(
            pen=pg.mkPen("#facc15", width=1.5, style=QtCore.Qt.DashLine)
        )
        self._line_preview.setZValue(40)
        self._vb.addItem(self._line_preview)

        self._image_item = pg.ImageItem()
        self._vb.addItem(self._image_item)

        # Red overlay for over-max pixels, drawn on top of the image.
        self._overlay_item = pg.ImageItem()
        self._overlay_item.setZValue(10)
        self._vb.addItem(self._overlay_item)

        # Histogram + level bars, linked to the main image item.
        self._hist = pg.HistogramLUTItem(image=self._image_item)
        self._hist.gradient.loadPreset("grey")
        self._glw.addItem(self._hist, row=0, col=1)
        self._hist.sigLevelsChanged.connect(self._refresh_overlay)
        self._hist.sigLookupTableChanged.connect(self._refresh_overlay)
        # The histogram can be collapsed (hidden) or detached to a float window.
        # _active_hist is whichever LUT item currently drives the image.
        self._active_hist = self._hist
        self._hist_visible = True
        self._hist_win: _FloatHist | None = None
        self._float_hw = None

        # Fit overlays.
        self._scatter = pg.ScatterPlotItem(
            size=9, pen=_POINT_PEN, brush=pg.mkBrush(34, 211, 238, 40), symbol="o"
        )
        self._scatter.setZValue(20)
        self._vb.addItem(self._scatter)

        self._ellipse_curve = pg.PlotCurveItem(pen=_ELLIPSE_PEN)
        self._ellipse_curve.setZValue(20)
        self._vb.addItem(self._ellipse_curve)

        self._center_marker = pg.ScatterPlotItem(
            size=13,
            pen=pg.mkPen("#facc15", width=1.5),
            brush=pg.mkBrush(None),
            symbol="+",
        )
        self._center_marker.setZValue(21)
        self._vb.addItem(self._center_marker)

        self._major_axis = pg.PlotCurveItem(pen=_AXIS_MAJOR_PEN)
        self._minor_axis = pg.PlotCurveItem(pen=_AXIS_MINOR_PEN)
        for it in (self._major_axis, self._minor_axis):
            it.setZValue(20)
            self._vb.addItem(it)

        # Distance tool: connected polyline + point markers + per-segment labels.
        self._dist_scatter = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen("#ffd60a", width=2),
            brush=pg.mkBrush(None),
            symbol="o",
        )
        self._dist_scatter.setZValue(22)
        self._vb.addItem(self._dist_scatter)
        self._dist_curve = pg.PlotCurveItem(pen=pg.mkPen("#ffd60a", width=1.5))
        self._dist_curve.setZValue(22)
        self._vb.addItem(self._dist_curve)

        # Line-profile ROIs are now managed objects (roi.py); the bottom plot
        # follows whichever line ROI is currently "active".
        self._active_line = None
        self._selected_roi = None
        self._grid = None

        # Empty / unopenable placeholder: the classic "no image" look — a light
        # gray square with a thin X across it (requirement 1).
        self._placeholder_rect = QtWidgets.QGraphicsRectItem(0, 0, 1, 1)
        self._placeholder_rect.setBrush(pg.mkBrush("#d9d9d9"))
        self._placeholder_rect.setPen(pg.mkPen("#b0b0b0", width=1))
        self._placeholder_rect.setZValue(49)
        self._placeholder_rect.setVisible(False)
        self._vb.addItem(self._placeholder_rect)

        self._error_curve = pg.PlotCurveItem(pen=pg.mkPen("#9a9a9a", width=2))
        self._error_curve.setZValue(50)
        self._error_curve.setVisible(False)
        self._vb.addItem(self._error_curve)
        self._error_text = pg.TextItem(color="#555555", anchor=(0.5, 0.5))
        self._error_text.textItem.setFont(QtGui.QFont("Arial", 12))
        self._error_text.setZValue(51)
        self._error_text.setVisible(False)
        self._vb.addItem(self._error_text)

        # Scale bar pinned to the lower-right corner (requirement 3).
        self._px_size = 1.0  # real units per pixel
        self._px_unit = "px"
        self._scale_calibrated = False
        self._scalebar = _ScaleBar(
            size=100,
            width=6,
            brush=pg.mkBrush("#ffffff"),
            pen=pg.mkPen("#000000", width=1),
            suffix="px",
            offset=(-25, -32),
        )
        # Anchor label top-center so it hangs below the bar; translucent black
        # backing keeps it legible on any image.
        self._scalebar.text.anchor = pg.Point(0.5, 0)
        self._scalebar.text.fill = pg.mkBrush(0, 0, 0, 140)
        self._scalebar.text.border = pg.mkPen(None)
        self._scalebar.text.setColor("#ffffff")
        self._scalebar.text.textItem.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self._scalebar.setParentItem(self._vb)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._glw)

        # Translucent HUD overlay (zoom % + scale) in the top-left corner.
        self._hud = QtWidgets.QLabel(self._glw)
        self._hud.setStyleSheet(
            "background: rgba(0,0,0,0.55); color: #e6e6e6; padding: 4px 8px;"
            "border-radius: 6px;"
        )
        self._hud.setFont(QtGui.QFont("Consolas", 9))
        self._hud.move(12, 12)
        self._hud.hide()

        # Mouse handling for cursor readout and point picking.
        self._glw.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._image_item.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self._vb.sigRangeChanged.connect(self._on_range_changed)

    # -------------------------------------------------------------- loading
    def set_image(self, img: np.ndarray) -> None:
        """Load a new raw frame, resetting fit state and autoscaling levels."""
        self._hide_error()
        self._raw = np.asarray(img, dtype=np.float32)
        self.clear_fit()
        self.clear_distance()
        self._recompute_display(autolevel=True)
        self._update_scalebar()
        self._scalebar.setVisible(True)
        self._emit_active_profile()
        self._vb.autoRange()

    def has_image(self) -> bool:
        return self._raw is not None

    def image_shape(self) -> tuple[int, int]:
        return (0, 0) if self._raw is None else self._raw.shape

    # ----------------------------------------------------------- zoom/error
    def reset_zoom(self) -> None:
        """Fit the whole image (or error marker) back into the view."""
        self._vb.autoRange()

    def current_zoom(self) -> float:
        px = self._vb.viewPixelSize()[0]
        return 100.0 / px if px > 0 else 100.0

    def apply_theme(self, dark: bool) -> None:
        self._glw.setBackground("#1a1a1d" if dark else "#eef0f2")
        self._vb.setBorder(pg.mkPen("#3a3a3a" if dark else "#c2c2ca", width=1))
        hud_bg = "rgba(0,0,0,0.55)" if dark else "rgba(255,255,255,0.8)"
        hud_fg = "#e6e6e6" if dark else "#1c1c22"
        self._hud.setStyleSheet(
            f"background:{hud_bg}; color:{hud_fg}; padding:4px 8px; border-radius:6px;"
        )

    def set_grid(self, on: bool) -> None:
        if on and self._grid is None:
            self._grid = pg.GridItem()
            self._grid.setZValue(5)
            self._vb.addItem(self._grid)
        elif not on and self._grid is not None:
            self._vb.removeItem(self._grid)
            self._grid = None

    def zoom_in(self) -> None:
        self._vb.scaleBy((0.8, 0.8))

    def zoom_out(self) -> None:
        self._vb.scaleBy((1.25, 1.25))

    def set_zoom(self, percent: float) -> None:
        if self._raw is None or percent <= 0:
            return
        cur_px = self._vb.viewPixelSize()[0]
        target_px = 100.0 / percent
        if cur_px > 0:
            self._vb.scaleBy((target_px / cur_px, target_px / cur_px))

    def _on_range_changed(self, *_) -> None:
        self.zoomChanged.emit(self.current_zoom())
        self._update_hud()

    def _update_hud(self) -> None:
        if self._raw is None:
            self._hud.hide()
            return
        scale = self._scalebar.text.toPlainText() if self._scalebar.isVisible() else ""
        txt = f"{self.current_zoom():.0f}%"
        if scale:
            txt += f"   ·   {scale}"
        self._hud.setText(txt)
        self._hud.adjustSize()
        self._hud.show()
        self._hud.raise_()

    def show_error(self, message: str) -> None:
        """Blank the image and show the empty-image placeholder + message."""
        self._raw = None
        self._display = None
        self._image_item.clear()
        self._overlay_item.clear()
        self._scalebar.setVisible(False)
        self.clear_fit()
        # Light gray box with a thin full-size X, mimicking a missing image.
        self._placeholder_rect.setVisible(True)
        self._error_curve.setData([0, 1, 0, 1], [0, 1, 1, 0], connect="pairs")
        self._error_text.setText(message)
        self._error_text.setPos(0.5, 0.5)
        self._error_curve.setVisible(True)
        self._error_text.setVisible(True)
        self._vb.setRange(xRange=(0, 1), yRange=(0, 1), padding=0.1)

    def _hide_error(self) -> None:
        self._placeholder_rect.setVisible(False)
        self._error_curve.setVisible(False)
        self._error_text.setVisible(False)

    # ------------------------------------------------------------- scale bar
    def set_scale(self, px_size: float, unit: str) -> None:
        """Calibrate: each pixel spans ``px_size`` of the given real unit."""
        self._px_size = float(px_size) if px_size > 0 else 1.0
        self._px_unit = unit or "unit"
        self._scale_calibrated = True
        self._update_scalebar()
        self._redraw_distance()
        self._emit_active_profile()

    def set_scale_pixels(self) -> None:
        """Revert the scale bar to raw pixels."""
        self._scale_calibrated = False
        self._update_scalebar()
        self._redraw_distance()
        self._emit_active_profile()

    def _update_scalebar(self) -> None:
        if self._raw is None:
            return
        _, w = self._raw.shape
        if self._scale_calibrated:
            nice_real = _nice_length(w * self._px_size / 5.0)
            bar_px = nice_real / self._px_size
            label = f"{_fmt_length(nice_real)} {self._px_unit}"
        else:
            bar_px = _nice_length(w / 5.0)
            label = f"{_fmt_length(bar_px)} px"
        self._scalebar.size = bar_px
        self._scalebar.updateBar()
        self._scalebar.text.setText(label)
        self._update_hud()

    def _recompute_display(self, autolevel: bool) -> None:
        if self._raw is None:
            return
        # The array handed in is ALREADY the processed image (Core computes the
        # pipeline). This view just displays it — no analysis math here.
        self._display = self._raw

        self._image_item.setImage(self._display, autoLevels=False)

        if autolevel:
            finite = self._display[np.isfinite(self._display)]
            if finite.size:
                lo = float(np.percentile(finite, 0.5))
                hi = float(np.percentile(finite, 99.5))
                if hi <= lo:
                    hi = lo + 1.0
                self._active_hist.setLevels(lo, hi)
        # Refresh histogram region to current data and repaint overlay.
        self._active_hist.setImageItem(self._image_item)
        self._refresh_overlay()

    def update_processed(self, img: np.ndarray) -> None:
        """Swap the displayed (already-processed) image, keeping ROIs and zoom.

        Used when the Core pipeline changes (preproc / log toggled). Unlike
        :meth:`set_image` it does NOT clear fits or reset the view range.
        """
        self._raw = np.asarray(img, dtype=np.float32)
        self._recompute_display(autolevel=True)

    def set_colormap(self, name: str) -> None:
        try:
            cmap = pg.colormap.get(name, source="matplotlib")
        except Exception:
            cmap = pg.colormap.get("gray") if name != "gray" else None
        if name == "gray":
            self._active_hist.gradient.loadPreset("grey")
        elif cmap is not None:
            self._active_hist.gradient.setColorMap(cmap)
        self._refresh_overlay()

    def set_overmax(self, enabled: bool) -> None:
        self._overmax_enabled = enabled
        self._refresh_overlay()

    def set_overmax_color(self, color: tuple[int, int, int] | QtGui.QColor) -> None:
        if isinstance(color, QtGui.QColor):
            color = (color.red(), color.green(), color.blue())
        self._overmax_color = color
        self._refresh_overlay()

    def get_levels(self) -> tuple[float, float]:
        lo, hi = self._active_hist.getLevels()
        return float(lo), float(hi)

    def set_levels(self, lo: float, hi: float) -> None:
        self._active_hist.setLevels(lo, hi)

    def autoscale_levels(self) -> None:
        self._recompute_display(autolevel=True)

    # ------------------------------------------------------- histogram panel
    def histogram_visible(self) -> bool:
        return self._hist_visible

    def is_histogram_detached(self) -> bool:
        return self._hist_win is not None

    def set_histogram_visible(self, vis: bool) -> None:
        """Collapse (hide) or restore the embedded histogram column."""
        if self._hist_win is not None or vis == self._hist_visible:
            return  # detached window owns visibility
        if vis:
            self._glw.ci.addItem(self._hist, row=0, col=1)
            self._hist.setImageItem(self._image_item)
        else:
            self._glw.ci.removeItem(self._hist)
        self._hist_visible = vis

    def detach_histogram(self) -> None:
        """Pop the histogram out into its own floating window."""
        if self._hist_win is not None:
            self._hist_win.raise_()
            self._hist_win.activateWindow()
            return
        if self._hist_visible:
            self._glw.ci.removeItem(self._hist)
            self._hist_visible = False

        win = _FloatHist()
        win.setWindowTitle("Histogram")
        win.resize(210, 470)
        win.setStyleSheet("background:#1f1f22;")
        lay = QtWidgets.QVBoxLayout(win)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        header = QtWidgets.QLabel("  Histogram")
        header.setStyleSheet(
            "background:#26262a; color:#d4d4d8; font-weight:bold;"
            "padding:6px 8px; border-bottom:1px solid #303036;"
        )
        lay.addWidget(header)
        hw = pg.HistogramLUTWidget()
        hw.setImageItem(self._image_item)
        lo, hi = self._hist.getLevels()
        hw.item.setLevels(lo, hi)
        hw.item.gradient.restoreState(self._hist.gradient.saveState())
        hw.item.sigLevelsChanged.connect(self._refresh_overlay)
        hw.item.sigLookupTableChanged.connect(self._refresh_overlay)
        lay.addWidget(hw)

        self._float_hw = hw
        self._active_hist = hw.item
        self._hist_win = win
        win.closed.connect(self._reattach_histogram)
        win.show()
        self._refresh_overlay()

    def _reattach_histogram(self) -> None:
        if self._hist_win is None:
            return
        lo, hi = self._float_hw.item.getLevels()
        grad = self._float_hw.item.gradient.saveState()
        self._hist_win = None
        self._float_hw = None
        self._active_hist = self._hist
        self._glw.ci.addItem(self._hist, row=0, col=1)
        self._hist.setImageItem(self._image_item)
        self._hist.setLevels(lo, hi)
        self._hist.gradient.restoreState(grad)
        self._hist_visible = True
        self._refresh_overlay()
        self.histogramReattached.emit()

    def _refresh_overlay(self, *_) -> None:
        """Paint red wherever display value exceeds the current max level."""
        if self._display is None or not self._overmax_enabled:
            self._overlay_item.clear()
            return
        _, hi = self._active_hist.getLevels()
        mask = self._display > hi
        if not mask.any():
            self._overlay_item.clear()
            return
        rgba = np.zeros((*self._display.shape, 4), dtype=np.ubyte)
        rgba[mask] = (*self._overmax_color, 255)
        self._overlay_item.setImage(rgba, autoLevels=False)

    # --------------------------------------------------------------- cursor
    def _on_mouse_moved(self, scene_pos) -> None:
        if self._raw is None:
            return
        vb_pt = self._vb.mapSceneToView(scene_pos)
        x, y = vb_pt.x(), vb_pt.y()
        h, w = self._raw.shape
        col, row = int(np.floor(x)), int(np.floor(y))
        if 0 <= row < h and 0 <= col < w:
            val = float(self._raw[row, col])
        else:
            val = float("nan")
        self.cursorMoved.emit(x, y, val)
        # Live preview of the line being placed (two-click line tool).
        if self._tool == "line" and self._line_pending is not None:
            x0, y0 = self._line_pending
            self._line_preview.setData([x0, x], [y0, y])

    # ---- region rubber-band (rect tool) ----
    def _on_rect_draw_start(self, p0) -> None:
        self._rect_preview.setVisible(True)

    def _on_rect_draw_update(self, p0, cur) -> None:
        x0, y0, x1, y1 = p0.x(), p0.y(), cur.x(), cur.y()
        self._rect_preview.setRect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

    def _on_rect_draw_finish(self, p0, p1) -> None:
        self._rect_preview.setVisible(False)
        if abs(p1.x() - p0.x()) > 2 and abs(p1.y() - p0.y()) > 2:
            self.rectDrawn.emit(p0.x(), p0.y(), p1.x(), p1.y())

    # -------------------------------------------------------------- tools
    def set_tool(self, name: str) -> None:
        """Switch the active interaction tool (state machine)."""
        if name not in self.TOOLS:
            return
        self._tool = name
        if name == "zoom":
            self._vb.setMouseMode(pg.ViewBox.RectMode)
        else:
            self._vb.setMouseMode(pg.ViewBox.PanMode)
        self._vb.draw_mode = name == "rect"  # left-drag draws a region rect
        if name != "line":  # abandon a half-placed line
            self._line_pending = None
            self._line_preview.setData([], [])
        self._glw.setCursor(self._CURSORS.get(name, QtCore.Qt.ArrowCursor))

    def current_tool(self) -> str:
        return self._tool

    def _scaled(self, px: float) -> tuple[float, str]:
        if self._scale_calibrated:
            return px * self._px_size, self._px_unit
        return px, "px"

    def _on_mouse_clicked(self, event) -> None:
        # Double-click anywhere resets the zoom. In point tools, undo the stray
        # point placed by the first click of the double so nothing is left.
        if event.double():
            if self._tool == "ellipse" and self._points:
                self._points.pop()
                self._redraw_points()
            elif self._tool == "distance" and self._dist_points:
                self._dist_points.pop()
                self._redraw_distance()
            self._vb.autoRange()
            event.accept()
            return
        if self._raw is None:
            return
        vb_pt = self._vb.mapSceneToView(event.scenePos())

        if self._tool == "ellipse":
            self._click_pointlist(event, vb_pt, self._points, self._redraw_points)
        elif self._tool == "distance":
            self._click_pointlist(
                event, vb_pt, self._dist_points, self._redraw_distance
            )
        elif self._tool == "line" and event.button() == QtCore.Qt.LeftButton:
            # Two-click line: 1st click sets start, 2nd click sets end.
            x, y = vb_pt.x(), vb_pt.y()
            if self._line_pending is None:
                self._line_pending = (x, y)
            else:
                x0, y0 = self._line_pending
                self._line_pending = None
                self._line_preview.setData([], [])
                self.lineDrawn.emit(x0, y0, x, y)
            event.accept()
        elif self._tool == "select" and event.button() == QtCore.Qt.LeftButton:
            self.select_roi(None)  # click empty canvas -> deselect

    def _click_pointlist(self, event, vb_pt, points, redraw) -> None:
        if event.button() == QtCore.Qt.RightButton:
            # Delete the point NEAREST the cursor so any single point can be
            # picked off individually (requirement 4 / ellipse).
            if points:
                arr = np.array(points)
                d2 = (arr[:, 0] - vb_pt.x()) ** 2 + (arr[:, 1] - vb_pt.y()) ** 2
                points.pop(int(np.argmin(d2)))
                redraw()
            event.accept()
        elif event.button() == QtCore.Qt.LeftButton:
            points.append((vb_pt.x(), vb_pt.y()))
            redraw()
            event.accept()

    def _redraw_points(self) -> None:
        if self._points:
            arr = np.array(self._points)
            self._scatter.setData(arr[:, 0], arr[:, 1])
        else:
            self._scatter.setData([], [])

    def point_count(self) -> int:
        return len(self._points)

    def clear_fit(self) -> None:
        self._points = []
        self._last_geom = None
        # setData([], []) repaints immediately; .clear() can lag a frame.
        self._scatter.setData([], [])
        self._ellipse_curve.setData([], [])
        self._center_marker.setData([], [])
        self._major_axis.setData([], [])
        self._minor_axis.setData([], [])

    # -------------------------------------------------------- ROI objects (B)
    def add_roi_item(self, roi) -> None:
        """Draw a managed ROI on the canvas and emit signals on edits."""
        roi.setZValue(25)
        roi._base_pen = roi.pen  # remember for selection highlight
        self._vb.addItem(roi)
        roi.sigRegionChanged.connect(lambda *_a, r=roi: self.roiChanged.emit(r))
        # Grabbing an ROI selects it.
        roi.sigRegionChangeStarted.connect(lambda *_a, r=roi: self.select_roi(r))
        self.roiAdded.emit(roi)

    def remove_roi_item(self, roi) -> None:
        try:
            roi.sigRegionChanged.disconnect()
            roi.sigRegionChangeStarted.disconnect()
        except TypeError, RuntimeError:
            pass
        if roi is self._active_line:
            self._active_line = None
        if roi is self._selected_roi:
            self._selected_roi = None
        self._vb.removeItem(roi)
        self.roiRemoved.emit(roi.roi_id)

    def select_roi(self, roi) -> None:
        """Highlight a ROI (cyan, thicker); deselect the previous one."""
        if self._selected_roi is roi:
            return
        prev = self._selected_roi
        if prev is not None:
            try:
                prev.setPen(prev._base_pen)
            except RuntimeError, AttributeError:
                pass
        self._selected_roi = roi
        if roi is not None:
            roi.setPen(pg.mkPen("#00e5ff", width=3))
            if roi.roi_type == "line":
                self.set_active_line(roi)
        self.roiSelected.emit(roi)

    def selected_roi(self):
        return self._selected_roi

    # ---- active line binding: the bottom profile follows this line ROI ----
    def set_active_line(self, roi) -> None:
        if self._active_line is roi:
            self._emit_active_profile()
            return
        if self._active_line is not None:
            try:
                self._active_line.sigRegionChanged.disconnect(self._emit_active_profile)
            except TypeError, RuntimeError:
                pass
        self._active_line = roi
        if roi is not None:
            roi.sigRegionChanged.connect(self._emit_active_profile)
        self._emit_active_profile()

    def _emit_active_profile(self, *_) -> None:
        if self._raw is None or self._active_line is None:
            return
        try:
            vals = self._active_line.getArrayRegion(self._raw, self._image_item)
        except Exception:
            return
        if vals is None or np.size(vals) < 2:
            return
        vals = np.asarray(vals, dtype=float).ravel()
        idx = np.arange(vals.size, dtype=float)
        axis, unit = (
            (idx * self._px_size, self._px_unit)
            if self._scale_calibrated
            else (idx, "px")
        )
        self.lineProfileChanged.emit(axis, vals, unit)

    # ---------------------------------------------------------- distance tool
    def _redraw_distance(self) -> None:
        for t in self._dist_labels:
            self._vb.removeItem(t)
        self._dist_labels = []

        if not self._dist_points:
            self._dist_scatter.setData([], [])
            self._dist_curve.setData([], [])
            self.distanceChanged.emit([], [], "px")
            return

        arr = np.array(self._dist_points)
        self._dist_scatter.setData(arr[:, 0], arr[:, 1])
        if len(arr) >= 2:
            self._dist_curve.setData(arr[:, 0], arr[:, 1])
        else:
            self._dist_curve.setData([], [])

        seg_px, cum_px = [], []
        running = 0.0
        _, unit = self._scaled(1.0)
        for i in range(1, len(arr)):
            d_px = float(np.hypot(*(arr[i] - arr[i - 1])))
            running += d_px
            seg_px.append(d_px)
            cum_px.append(running)
            val, unit = self._scaled(d_px)
            mid = (arr[i] + arr[i - 1]) / 2.0
            lbl = pg.TextItem(f"{val:.1f} {unit}", color="#ffd60a", anchor=(0.5, 1.2))
            lbl.setPos(mid[0], mid[1])
            lbl.setZValue(24)
            self._vb.addItem(lbl)
            self._dist_labels.append(lbl)

        seg = [self._scaled(d)[0] for d in seg_px]
        cum = [self._scaled(d)[0] for d in cum_px]
        self.distanceChanged.emit(seg, cum, unit)

    def clear_distance(self) -> None:
        self._dist_points = []
        self._redraw_distance()

    def distance_count(self) -> int:
        return len(self._dist_points)

    def fit_ellipse(self) -> dict | None:
        """Fit an ellipse to the picked points and draw it. Returns geometry."""
        if len(self._points) < 5:
            self.fitReported.emit(f"Need at least 5 points (have {len(self._points)}).")
            return None
        pts = np.array(self._points, dtype=float)
        try:
            coeffs = ef.fit_ellipse(pts[:, 0], pts[:, 1])
            geom = ef.conic_to_geometry(coeffs)
            resid = ef.sampson_residuals(coeffs, pts[:, 0], pts[:, 1])
        except Exception as exc:  # collinear points, etc.
            self.fitReported.emit(f"Fit failed: {exc}")
            return None

        self._last_geom = geom
        self._draw_ellipse(geom)
        self.fitReported.emit(ef.format_report(coeffs, geom, pts, resid))
        return geom

    def _draw_ellipse(self, geom: dict) -> None:
        ex, ey = ef.ellipse_polyline(geom)
        self._ellipse_curve.setData(ex, ey)
        self._center_marker.setData([geom["center_x"]], [geom["center_y"]])

        th = np.radians(geom["angle_major_from_x_deg"])
        cx, cy = geom["center_x"], geom["center_y"]
        a, b = geom["semi_major_axis"], geom["semi_minor_axis"]
        maj = (np.cos(th), np.sin(th))
        minv = (-np.sin(th), np.cos(th))
        self._major_axis.setData(
            [cx - a * maj[0], cx + a * maj[0]], [cy - a * maj[1], cy + a * maj[1]]
        )
        self._minor_axis.setData(
            [cx - b * minv[0], cx + b * minv[0]], [cy - b * minv[1], cy + b * minv[1]]
        )

    def last_geometry(self) -> dict | None:
        return self._last_geom

    def picked_points(self) -> np.ndarray:
        return np.array(self._points, dtype=float)
