"""DFXM Core — Pure analysis engine. Zero Qt / GUI dependencies.

Everything here is importable from a plain Python / CLI / notebook context.
"""

from __future__ import annotations

from . import fitting, io, ops, transform
from .dataset import DFXMDataset
from .history import History
from .ops import OP_LABELS, Operation
from .results import MASTER_COLUMNS, FitResult, ResultsFrame

__all__ = [
    "MASTER_COLUMNS",
    "OP_LABELS",
    "DFXMDataset",
    "FitResult",
    "History",
    "Operation",
    "ResultsFrame",
    "fitting",
    "io",
    "ops",
    "transform",
]
