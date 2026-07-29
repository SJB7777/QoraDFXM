"""Pixel-domain transforms. Pure NumPy, no Qt."""

from __future__ import annotations

import numpy as np


def adaptive_log(img) -> np.ndarray:
    """Normalized log1p — compresses dynamic range to [0, 1]."""
    c = 1.0 / np.log1p(np.max(img))
    return c * np.log1p(img)
