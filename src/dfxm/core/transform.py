"""Pixel-domain transforms. Pure NumPy, no Qt."""

from __future__ import annotations

import numpy as np


def adaptive_log(img) -> np.ndarray:
    """Normalized log1p — compresses dynamic range to [0, 1].

    NaN-tolerant: geometric ops can leave NaN in the corners, and a single NaN
    in the max would otherwise blank the whole image.
    """
    peak = np.nanmax(img) if np.isfinite(img).any() else 0.0
    denom = np.log1p(peak)
    c = 1.0 / denom if denom > 0 else 1.0
    return c * np.log1p(img)
