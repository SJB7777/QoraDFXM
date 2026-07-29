"""Backward-compat shim. Ellipse math moved to dfxm.core.fitting.

The old interactive matplotlib picker / ``main()`` was dropped — the GUI
(``dfxm.gui``) is the interactive front-end now. New code should import from
``dfxm.core.fitting``.
"""

from __future__ import annotations

from .core.fitting import (  # noqa: F401
    conic_to_geometry,
    ellipse_polyline,
    fit_ellipse,
    format_equation,
    format_report,
    rms_error,
    sampson_residuals,
)
