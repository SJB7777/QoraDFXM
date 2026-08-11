"""Recipe ⇄ command-line translation. Pure: no Qt, no argparse, no I/O.

One definition of what ``--op sub_bg:/dark`` means, shared by

* the CLI parser (string → :class:`~dfxm.core.ops.Operation`), and
* the GUI, which turns the recipe it built interactively back into an argv and
  hands it to the CLI as a batch job (:mod:`dfxm.gui.cli_bridge`).

Keeping both directions here is what makes "the GUI runs the same thing you
could have typed" true by construction instead of by convention.
"""

from __future__ import annotations

from pathlib import Path

from ..core.io import IMAGE_SUFFIXES
from ..core.ops import Operation
from ..core.warp import FLIP_AXES

#: Ops that take no argument at all.
PLAIN_OPS = ("log", "pure_log", "sqrt", "normalize")
#: Ops whose argument is a reference frame (dark / flat).
SOURCE_OPS = ("sub_bg", "divide")
#: Ops whose argument is one scalar → ``{kind: param_key}``.
SCALAR_OPS = {"gamma": "gamma", "rotate": "angle"}
#: Geometric ops with their own little grammar.
GEOMETRIC_OPS = ("scale", "rotate", "flip")

OP_ARG_HELP = (
    "pipeline op, repeatable & ordered: sub_bg:/dark, divide:flat.tif, "
    "log, pure_log, sqrt, gamma:0.5, normalize, "
    "scale:1.5, scale:2x0.5 (sx × sy), rotate:30 (deg, CCW), flip:h|v|both"
)


class SpecError(ValueError):
    """A malformed ``KIND[:SRC]`` op spec."""


def _is_dataset_ref(src: str) -> bool:
    """An h5 dataset path inside the shot's own file, vs. an external file.

    ``/run/scan00001/det/d/data`` → dataset. ``flat.tif``, ``C:\\ref\\flat.tif``
    and ``/mnt/data/flat.tif`` → file (leading slash alone is not enough — a
    POSIX absolute path to an image would be misread).
    """
    if not src.startswith("/"):
        return False
    return Path(src).suffix.lower() not in IMAGE_SUFFIXES


def _parse_scale(src: str) -> dict:
    """``"1.5"`` → uniform; ``"2x0.5"`` → sx=2, sy=0.5 (``:`` is taken, so 'x')."""
    if not src:
        raise SpecError("op 'scale' needs a factor, e.g. scale:1.5 or scale:2x0.5")
    parts = src.lower().split("x")
    if len(parts) > 2:
        raise SpecError(f"scale takes 'sx' or 'sxXsy' (got '{src}')")
    try:
        sx = float(parts[0])
        sy = float(parts[1]) if len(parts) == 2 else sx
    except ValueError:
        raise SpecError(f"scale factors must be numbers (got '{src}')") from None
    if sx <= 0 or sy <= 0:
        raise SpecError(f"scale factors must be > 0 (got '{src}')")
    return {"sx": sx, "sy": sy}


def parse_op(spec: str) -> Operation:
    """``"sub_bg:/dark"`` → ``Operation("sub_bg", {"dataset_path": "/dark"})``."""
    kind, _, src = spec.partition(":")  # partition keeps ':' inside "C:\path"
    kind = kind.strip()
    src = src.strip()

    if kind in PLAIN_OPS:
        if src:
            raise SpecError(f"op '{kind}' takes no argument (got '{src}')")
        return Operation(kind, {})

    if kind == "scale":
        return Operation("scale", _parse_scale(src))

    if kind == "flip":
        axis = src or "h"
        if axis not in FLIP_AXES:
            raise SpecError(f"flip axis must be one of {FLIP_AXES} (got '{axis}')")
        return Operation("flip", {"axis": axis})

    if kind in SCALAR_OPS:
        key = SCALAR_OPS[kind]
        default = 0.5 if kind == "gamma" else 0.0
        try:
            value = float(src) if src else default
        except ValueError:
            raise SpecError(f"op '{kind}' needs a number, e.g. {kind}:0.5") from None
        return Operation(kind, {key: value})

    if kind in SOURCE_OPS:
        if not src:
            raise SpecError(f"op '{kind}' needs a source, e.g. {kind}:/dark")
        key = "dataset_path" if _is_dataset_ref(src) else "file_path"
        return Operation(kind, {key: src})

    known = ", ".join(PLAIN_OPS + SOURCE_OPS + tuple(SCALAR_OPS) + GEOMETRIC_OPS)
    raise SpecError(f"unknown op '{kind}' (known: {known})")


def format_op(op: Operation) -> str:
    """Inverse of :func:`parse_op` — ``Operation`` → ``"sub_bg:/dark"``."""
    if op.kind in PLAIN_OPS:
        return op.kind
    if op.kind == "scale":
        sx = float(op.params.get("sx", 1.0))
        sy = float(op.params.get("sy", sx))
        return f"scale:{sx:g}" if sx == sy else f"scale:{sx:g}x{sy:g}"
    if op.kind == "flip":
        return f"flip:{op.params.get('axis', 'h')}"
    if op.kind in SCALAR_OPS:
        key = SCALAR_OPS[op.kind]
        return f"{op.kind}:{float(op.params.get(key, 0.0)):g}"
    if op.kind in SOURCE_OPS:
        src = op.source
        if not src:
            raise SpecError(f"op '{op.kind}' has no source to serialize")
        return f"{op.kind}:{src}"
    raise SpecError(f"op '{op.kind}' cannot be expressed on the command line")


def parse_ops(specs) -> list[Operation]:
    return [parse_op(s) for s in (specs or [])]


def format_ops(history) -> list[str]:
    return [format_op(op) for op in (history or [])]


def argv_for_fit(
    file,
    *,
    dataset_path: str | None = None,
    ops=(),
    points_file=None,
    out=None,
) -> list[str]:
    """Build the argv of a ``dfxm fit`` run (without the ``dfxm`` prog name)."""
    argv = ["fit", str(file)]
    if dataset_path:
        argv += ["--dataset", dataset_path]
    for spec in ops:
        argv += ["--op", spec if isinstance(spec, str) else format_op(spec)]
    if points_file:
        argv += ["--points", str(points_file)]
    if out:
        argv += ["--out", str(out)]
    return argv


def argv_for_dataset(ds, *, points_file=None, out=None) -> list[str]:
    """Turn a live :class:`~dfxm.core.dataset.DFXMDataset` into a ``fit`` argv.

    This is how the GUI replays its own recipe through the CLI. Requires the
    dataset to know where it came from (``source_path``).
    """
    if ds.source_path is None:
        raise SpecError("dataset has no source_path — nothing for the CLI to open")
    return argv_for_fit(
        ds.source_path,
        dataset_path=ds.meta.get("dataset_path"),
        ops=format_ops(ds.history),
        points_file=points_file,
        out=out,
    )
