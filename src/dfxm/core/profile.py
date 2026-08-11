"""Ellipse ring profile — brightness along a fitted ellipse as it is scaled.

Given a fitted ellipse (centre, semi-axes a/b, tilt θ) the contour is scaled by
a factor ``k`` about its centre and the image is sampled along it::

    x(t, k) = cx + k·a·cos t·cosθ − k·b·sin t·sinθ
    y(t, k) = cy + k·a·cos t·sinθ + k·b·sin t·cosθ

Sweeping ``k`` gives ``I(k)`` — the mean brightness per unit contour length,
i.e. the *line density* of intensity, which is what a Debye–Scherrer-like ring
lets you compare across shots.

Two things this module is careful about:

* **Equal-angle sampling is biased.** Steps in ``t`` are not steps in arc
  length, so the high-curvature ends of the major axis would be over-weighted.
  Every sample is therefore weighted by its arc-length element ``ds/dt``; the
  weighted mean is exactly the intensity per unit length.
* **Measure on linear data.** ``mean(log I) ≠ log(mean I)``. Strip the display
  transforms first — :meth:`dfxm.core.dataset.DFXMDataset.linear_view` does
  that while keeping geometry and background correction.

Pure NumPy + OpenCV (sampling only), no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Columns of :meth:`RingProfile.to_dataframe` — the CSV schema.
RING_COLUMNS = [
    "k",
    "semi_major_px",
    "mean",
    "std",
    "total",
    "perimeter_px",
    "valid_frac",
]


def _cv2():
    import cv2

    return cv2


def k_axis(start: float, stop: float, step: float) -> np.ndarray:
    """Inclusive scale axis, e.g. ``k_axis(0.2, 2.0, 0.01)``."""
    if step <= 0:
        raise ValueError("k step must be > 0")
    if stop < start:
        raise ValueError("k stop must be >= start")
    n = round((stop - start) / step) + 1
    return start + step * np.arange(n, dtype=float)


def _geom_of(fit) -> tuple[float, float, float, float, float]:
    """(cx, cy, a, b, theta_rad) from a FitResult, its geom dict, or a row."""
    g = getattr(fit, "geom", None) or (fit if isinstance(fit, dict) else None)
    if g and "semi_major_axis" in g:
        return (
            float(g["center_x"]),
            float(g["center_y"]),
            float(g["semi_major_axis"]),
            float(g["semi_minor_axis"]),
            np.radians(float(g.get("angle_major_from_x_deg", 0.0))),
        )
    # FitResult / Master row: diameters, not semi-axes.
    get = (lambda k: fit[k]) if isinstance(fit, dict) else (lambda k: getattr(fit, k))
    return (
        float(get("center_x")),
        float(get("center_y")),
        float(get("major_axis")) / 2.0,
        float(get("minor_axis")) / 2.0,
        np.radians(float(get("angle_deg"))),
    )


@dataclass
class RingProfile:
    """``I(k)`` plus everything needed to interpret it."""

    k: np.ndarray
    mean: np.ndarray  # arc-length-weighted mean = intensity per unit length
    std: np.ndarray
    total: np.ndarray  # mean × perimeter = integrated intensity on the ring
    perimeter: np.ndarray  # contour length in px
    valid_frac: np.ndarray  # fraction of samples inside the image and finite
    semi_major: np.ndarray  # k·a, the radius in px along the major axis
    theta: np.ndarray | None = None  # angular axis, if the map was kept
    map: np.ndarray | None = None  # I(k, θ), if keep_map=True
    params: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.k.size)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "k": self.k,
                "semi_major_px": self.semi_major,
                "mean": self.mean,
                "std": self.std,
                "total": self.total,
                "perimeter_px": self.perimeter,
                "valid_frac": self.valid_frac,
            }
        )[RING_COLUMNS]

    def to_csv(self, path) -> None:
        self.to_dataframe().to_csv(path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------- summary
    def peak(self) -> tuple[float, float]:
        """(k, intensity) at the maximum, parabola-refined between samples."""
        y = self.mean
        ok = np.isfinite(y)
        if not ok.any():
            return (float("nan"), float("nan"))
        i = int(np.flatnonzero(ok)[np.argmax(y[ok])])
        if 0 < i < len(y) - 1 and np.isfinite(y[i - 1]) and np.isfinite(y[i + 1]):
            y0, y1, y2 = y[i - 1], y[i], y[i + 1]
            denom = y0 - 2 * y1 + y2
            if denom != 0:
                d = 0.5 * (y0 - y2) / denom  # sub-sample offset in [-0.5, 0.5]
                dk = self.k[i + 1] - self.k[i - 1]
                return (
                    float(self.k[i] + d * dk / 2.0),
                    float(y1 - 0.25 * (y0 - y2) * d),
                )
        return (float(self.k[i]), float(y[i]))

    def fwhm(self) -> float:
        """Full width (in k) at half maximum above the curve's own baseline."""
        y = self.mean
        ok = np.isfinite(y)
        if ok.sum() < 3:
            return float("nan")
        base = float(np.nanmin(y))
        _, ypk = self.peak()
        half = base + 0.5 * (ypk - base)
        i = int(np.nanargmax(np.where(ok, y, -np.inf)))
        return float(self._cross(i, +1, half) - self._cross(i, -1, half))

    def _cross(self, i: int, direction: int, half: float) -> float:
        """Walk out from the peak to where the curve first drops to ``half``."""
        y, k = self.mean, self.k
        j = i
        while 0 <= j + direction < len(y):
            j += direction
            if np.isfinite(y[j]) and y[j] <= half:
                y_out, y_in = y[j], y[j - direction]
                if y_in == y_out:
                    return float(k[j])
                f = (half - y_out) / (y_in - y_out)
                return float(k[j] + f * (k[j - direction] - k[j]))
        return float("nan")


def ring_profile(
    img,
    fit,
    *,
    k=(0.2, 2.0, 0.01),
    n_theta: int = 720,
    width: float = 0.0,
    width_unit: str = "px",
    n_sub: int = 3,
    keep_map: bool = False,
) -> RingProfile:
    """Sample mean brightness along the fitted ellipse over a range of scales.

    Parameters
    ----------
    img : 2-D array (use LINEAR intensity — see the module docstring)
    fit : FitResult, its ``geom`` dict, or a Master row dict
    k : ``(start, stop, step)`` or an explicit array of scale factors
    n_theta : samples around the contour (arc-length weighted)
    width, width_unit : ring thickness, in ``px`` (converted via the mean
        radius) or in ``k`` units. 0 = a single one-pixel-wide contour.
    n_sub : sub-rings averaged across that thickness (ignored if width == 0)
    keep_map : also return the full ``I(k, θ)`` array (unrolled ring image)
    """
    img = np.ascontiguousarray(np.asarray(img, dtype=np.float32))
    if img.ndim != 2:
        raise ValueError(f"ring_profile needs a 2-D image, got shape {img.shape}")

    ks = np.asarray(k_axis(*k) if isinstance(k, (tuple, list)) else k, dtype=float)
    if ks.size == 0:
        raise ValueError("empty k axis")

    cx, cy, a, b, th = _geom_of(fit)
    if not (a > 0 and b > 0):
        raise ValueError("ellipse has a non-positive semi-axis")

    dk = 0.0
    if width > 0:
        dk = float(width) / ((a + b) / 2.0) if width_unit == "px" else float(width)
    subs = np.array([0.0]) if dk <= 0 else np.linspace(-dk / 2, dk / 2, max(2, n_sub))

    t = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(t), np.sin(t)
    ct, st = np.cos(th), np.sin(th)

    # Arc-length weight of each sample on the NOMINAL ring (k, not the sub-rings):
    # |d(x,y)/dt| for the scaled ellipse is k × that of the unit-k ellipse, and a
    # constant factor cancels in the weighted mean, so compute it once.
    dxdt = -a * sin_t * ct - b * cos_t * st
    dydt = -a * sin_t * st + b * cos_t * ct
    ds_unit = np.hypot(dxdt, dydt)  # per unit k, per unit t

    acc = np.zeros((ks.size, n_theta), dtype=np.float64)
    cnt = np.zeros((ks.size, n_theta), dtype=np.float64)
    for off in subs:
        kk = (ks + off)[:, None]
        xs = cx + kk * (a * cos_t * ct - b * sin_t * st)
        ys = cy + kk * (a * cos_t * st + b * sin_t * ct)
        vals = _sample(img, xs, ys)
        good = np.isfinite(vals)
        acc += np.where(good, vals, 0.0)
        cnt += good
    with np.errstate(invalid="ignore", divide="ignore"):
        ring = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

    valid = np.isfinite(ring)
    w = np.broadcast_to(ds_unit, ring.shape) * valid
    wsum = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(
            wsum > 0, (np.where(valid, ring, 0.0) * w).sum(axis=1) / wsum, np.nan
        )
        var = np.where(
            wsum > 0,
            (np.where(valid, (ring - mean[:, None]) ** 2, 0.0) * w).sum(axis=1) / wsum,
            np.nan,
        )

    dt = 2.0 * np.pi / n_theta
    perimeter = ks * ds_unit.sum() * dt
    return RingProfile(
        k=ks,
        mean=mean,
        std=np.sqrt(var),
        total=mean * perimeter,
        perimeter=perimeter,
        valid_frac=valid.mean(axis=1),
        semi_major=ks * a,
        theta=t if keep_map else None,
        map=ring if keep_map else None,
        params={
            "center_x": cx,
            "center_y": cy,
            "semi_major_axis": a,
            "semi_minor_axis": b,
            "angle_deg": float(np.degrees(th)),
            "n_theta": n_theta,
            "width": float(width),
            "width_unit": width_unit,
            "n_sub": int(n_sub) if dk > 0 else 1,
        },
    )


def _sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear sample at float coordinates; outside the image → NaN."""
    cv2 = _cv2()
    out = cv2.remap(
        img,
        np.ascontiguousarray(xs, dtype=np.float32),
        np.ascontiguousarray(ys, dtype=np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    return out.astype(np.float64, copy=False)
