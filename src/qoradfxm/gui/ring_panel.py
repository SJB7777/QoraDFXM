"""링 프로파일 패널 — 타원 둘레 밝기 I(k) 실시간 플롯.

The panel owns no pixel math: it asks its *source* callback for the current
(linear) image plus the live ellipse geometry, hands both to
:func:`qoradfxm.core.profile.ring_profile`, and draws what comes back. Dragging the
ellipse ROI therefore re-measures instead of re-implementing anything.

Updates are coalesced through a short timer so a drag produces a handful of
recomputes, not one per mouse move.
"""

from __future__ import annotations

import math

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.profile import ring_profile

#: Delay between the last change and the recompute, in ms.
THROTTLE_MS = 120


class RingProfilePanel(QtWidgets.QWidget):
    """Plot of arc-length-weighted mean brightness vs. the ellipse scale k."""

    def __init__(self, source=None, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._source = source  # () -> (image, geom dict) | None
        self._profile = None

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(THROTTLE_MS)
        self._timer.timeout.connect(self.refresh)

        self._build_ui()

    # ------------------------------------------------------------------ ui
    def _build_ui(self) -> None:
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(6)

        self._k0 = self._spin(0.05, 10.0, 0.2, 2, "k 시작")
        self._k1 = self._spin(0.05, 20.0, 2.0, 2, "k 끝")
        self._dk = self._spin(0.001, 1.0, 0.01, 3, "k 간격")
        self._width = self._spin(0.0, 50.0, 0.0, 1, "링 두께 (px, 0 = 선)")
        for label, w in (
            ("k", self._k0),
            ("→", self._k1),
            ("Δk", self._dk),
            ("두께", self._width),
        ):
            bar.addWidget(QtWidgets.QLabel(label))
            bar.addWidget(w)

        self._angles = QtWidgets.QSpinBox()
        self._angles.setRange(60, 7200)
        self._angles.setSingleStep(60)
        self._angles.setValue(720)
        self._angles.setToolTip("둘레 샘플 개수 (호길이 가중 평균)")
        bar.addWidget(QtWidgets.QLabel("샘플"))
        bar.addWidget(self._angles)

        self._live = QtWidgets.QCheckBox("실시간")
        self._live.setChecked(True)
        self._live.setToolTip("ROI를 끌 때마다 자동 재계산")
        bar.addWidget(self._live)

        btn_calc = QtWidgets.QPushButton("계산")
        btn_calc.clicked.connect(self.refresh)
        bar.addWidget(btn_calc)

        self._btn_csv = QtWidgets.QPushButton("CSV 저장")
        self._btn_csv.setEnabled(False)
        self._btn_csv.clicked.connect(self._export_csv)
        bar.addWidget(self._btn_csv)

        bar.addStretch(1)
        lay.addLayout(bar)

        for w in (self._k0, self._k1, self._dk, self._width):
            w.valueChanged.connect(self.request_update)
        self._angles.valueChanged.connect(self.request_update)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("#202225")
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        self._plot.setLabel("bottom", "타원 배율 k  (1.0 = 피팅된 타원)")
        self._plot.setLabel("left", "둘레 평균 밝기 (선밀도)")
        for ax in ("bottom", "left"):
            axis = self._plot.getAxis(ax)
            axis.setPen(pg.mkPen("#666", width=1))
            axis.setTextPen(pg.mkPen("#bbb"))
        self._curve = self._plot.plot(pen=pg.mkPen("#22d3ee", width=1.6))
        self._unit_line = pg.InfiniteLine(
            pos=1.0,
            angle=90,
            pen=pg.mkPen("#6b7280", width=1, style=QtCore.Qt.DashLine),
        )
        self._peak_line = pg.InfiniteLine(
            pos=0.0, angle=90, pen=pg.mkPen("#ff375f", width=1.2)
        )
        self._peak_line.setVisible(False)
        self._plot.addItem(self._unit_line)
        self._plot.addItem(self._peak_line)
        lay.addWidget(self._plot, 1)

        self._info = QtWidgets.QLabel("타원 피팅 후 [계산]")
        self._info.setStyleSheet("color:#9ca3af;")
        lay.addWidget(self._info)

    @staticmethod
    def _spin(lo, hi, val, decimals, tip) -> QtWidgets.QDoubleSpinBox:
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setSingleStep(10**-decimals * 10)
        w.setValue(val)
        w.setToolTip(tip)
        w.setMaximumWidth(90)
        return w

    # --------------------------------------------------------------- logic
    def set_source(self, source) -> None:
        """``source() -> (image, geom) | None`` — linear image + live ellipse."""
        self._source = source

    def request_update(self) -> None:
        """Coalesced refresh; ignored while 실시간 is off (buttons still work)."""
        if self._live.isChecked() and self.isVisible():
            self._timer.start()

    def refresh(self) -> None:
        if self._source is None:
            return
        data = self._source()
        if data is None:
            self._clear("타원 ROI가 없습니다 — 점을 찍고 피팅하세요")
            return
        img, geom = data
        k0, k1, dk = self._k0.value(), self._k1.value(), self._dk.value()
        if k1 <= k0:
            self._clear("k 끝은 k 시작보다 커야 합니다")
            return
        try:
            prof = ring_profile(
                img,
                geom,
                k=(k0, k1, dk),
                n_theta=self._angles.value(),
                width=self._width.value(),
                width_unit="px",
                n_sub=3,
            )
        except (ValueError, KeyError) as exc:
            self._clear(f"계산 실패: {exc}")
            return

        self._profile = prof
        self._curve.setData(prof.k, prof.mean)
        kp, ip = prof.peak()
        self._peak_line.setPos(0.0 if math.isnan(kp) else kp)
        self._peak_line.setVisible(not math.isnan(kp))
        a = prof.params["semi_major_axis"]
        worst = float(prof.valid_frac.min())
        note = f"   ⚠ 링 일부가 이미지 밖 (유효 {worst:.0%})" if worst < 1.0 else ""
        self._info.setText(
            f"피크 k={kp:.4f}  (장반경 {kp * a:.1f} px)   I={ip:.6g}   "
            f"FWHM={prof.fwhm():.4f} k   샘플 {len(prof)}×{self._angles.value()}{note}"
        )
        self._btn_csv.setEnabled(True)

    def _clear(self, message: str) -> None:
        self._profile = None
        self._curve.setData([], [])
        self._peak_line.setVisible(False)
        self._info.setText(message)
        self._btn_csv.setEnabled(False)

    def _export_csv(self) -> None:
        if self._profile is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "링 프로파일 저장", "ring_profile.csv", "CSV (*.csv)"
        )
        if path:
            self._profile.to_csv(path)
            self._info.setText(f"{self._info.text()}   → 저장: {path}")

    @property
    def profile(self):
        """The last computed :class:`~qoradfxm.core.profile.RingProfile`, or None."""
        return self._profile
