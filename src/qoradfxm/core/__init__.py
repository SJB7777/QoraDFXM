"""QoraDFXM Core — Pure analysis engine. Zero Qt / GUI dependencies.

Everything here is importable from a plain Python / CLI / notebook context.
"""

from __future__ import annotations

from . import fitting, io, ops, profile, transform, warp
from .dataset import QoraDFXMDataset
from .history import History
from .ops import OP_LABELS, Operation
from .profile import RingProfile, ring_profile
from .results import MASTER_COLUMNS, FitResult, ResultsFrame

__all__ = [
    "MASTER_COLUMNS",
    "OP_LABELS",
    "QoraDFXMDataset",
    "FitResult",
    "History",
    "Operation",
    "ResultsFrame",
    "RingProfile",
    "fitting",
    "io",
    "ops",
    "profile",
    "ring_profile",
    "transform",
    "warp",
]
