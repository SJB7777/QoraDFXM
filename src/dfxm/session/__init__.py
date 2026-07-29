"""Session layer.

The Master results data now lives in the Core engine
(:mod:`dfxm.core.results`). This package is reserved for Phase 3 SQLite
``.dfxm_proj`` session persistence. Re-exports kept for backward-compat.
"""

from ..core.results import MASTER_COLUMNS, ResultsFrame

__all__ = ["MASTER_COLUMNS", "ResultsFrame"]
