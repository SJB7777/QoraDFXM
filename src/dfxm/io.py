"""Backward-compat shim. The real implementation now lives in dfxm.core.io.

Kept so existing imports (`from dfxm import io`) keep working after the
Engine-First refactor. New code should import from ``dfxm.core.io``.
"""

from __future__ import annotations

from .core.io import (  # noqa: F401
    H5_SUFFIXES,
    IMAGE_SUFFIXES,
    TEXT_SUFFIXES,
    FramePath,
    H5Node,
    list_frames,
    load_dataset,
    load_frame,
    load_image_file,
    read_structure,
)
