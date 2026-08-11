"""Geometric transforms — ops that resample the pixel grid.

Unlike the intensity ops in :mod:`dfxm.core.transform` these change *where* a
pixel is (and usually the image shape), so they are not commutative with
anything that references another frame by shape (``sub_bg`` / ``divide``) and
they invalidate coordinates picked on an earlier version of the image. Order in
the history therefore matters — put geometry first if a fit is to follow.

Pure NumPy + OpenCV, no Qt. OpenCV is imported lazily so that merely importing
the engine stays cheap.
"""

from __future__ import annotations

import numpy as np

#: Public interpolation names → the OpenCV flag, resolved lazily.
INTERP_KINDS = ("auto", "nearest", "linear", "cubic", "area", "lanczos")

FLIP_AXES = ("h", "v", "both")


def _cv2():
    import cv2

    return cv2


def _interp_flag(name: str, shrinking: bool):
    cv2 = _cv2()
    if name == "auto":
        # Area averaging is the right answer when throwing pixels away;
        # bilinear when inventing them.
        name = "area" if shrinking else "linear"
    flags = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    if name not in flags:
        raise ValueError(f"unknown interpolation '{name}' (use: {INTERP_KINDS})")
    return flags[name]


def _as_f32(img) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(img, dtype=np.float32))


def scale(img, sx: float = 1.0, sy: float | None = None, interp="auto") -> np.ndarray:
    """Resample by independent x / y factors (``sy`` defaults to ``sx``).

    ``sx != sy`` is the aspect-ratio change: 1.0 keeps the axis as is, 0.5
    halves it, 2.0 doubles it. Output shape is the input shape times the
    factors, rounded, floored at 1 pixel.
    """
    img = _as_f32(img)
    sx = float(sx)
    sy = float(sx if sy is None else sy)
    if sx <= 0 or sy <= 0:
        raise ValueError(f"scale factors must be > 0 (got sx={sx}, sy={sy})")
    h, w = img.shape
    nw, nh = max(1, round(w * sx)), max(1, round(h * sy))
    if (nw, nh) == (w, h):
        return img
    flag = _interp_flag(interp, shrinking=(nw * nh < w * h))
    return _cv2().resize(img, (nw, nh), interpolation=flag)


def rotate(
    img,
    angle_deg: float,
    expand: bool = True,
    interp="linear",
    fill: float = 0.0,
) -> np.ndarray:
    """Rotate about the image centre. Positive angle = counter-clockwise.

    With ``expand`` the canvas grows so nothing is cut off (corners are filled
    with ``fill``); without it the original shape is kept and content leaves
    the frame.
    """
    cv2 = _cv2()
    img = _as_f32(img)
    angle_deg = float(angle_deg)
    if angle_deg % 360.0 == 0.0:
        return img

    h, w = img.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    nw, nh = w, h
    if expand:
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = round(h * sin + w * cos), round(h * cos + w * sin)
        M[0, 2] += (nw - 1) / 2.0 - cx
        M[1, 2] += (nh - 1) / 2.0 - cy

    return cv2.warpAffine(
        img,
        M,
        (nw, nh),
        flags=_interp_flag(interp, shrinking=False),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(fill),
    )


def flip(img, axis: str = "h") -> np.ndarray:
    """Mirror the image: ``h`` = left↔right, ``v`` = top↔bottom, ``both``."""
    img = _as_f32(img)
    if axis == "h":
        return np.ascontiguousarray(img[:, ::-1])
    if axis == "v":
        return np.ascontiguousarray(img[::-1, :])
    if axis == "both":
        return np.ascontiguousarray(img[::-1, ::-1])
    raise ValueError(f"unknown flip axis '{axis}' (use: {FLIP_AXES})")
