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

#: Ops that take no argument at all.
PLAIN_OPS = ("log", "pure_log", "sqrt", "normalize")
#: Ops whose argument is a reference frame (dark / flat).
SOURCE_OPS = ("sub_bg", "divide")
#: Ops whose argument is a scalar.
SCALAR_OPS = ("gamma",)

OP_ARG_HELP = (
    "pipeline op, repeatable & ordered: sub_bg:/dark, divide:flat.tif, "
    "log, pure_log, sqrt, gamma:0.5, normalize"
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


def parse_op(spec: str) -> Operation:
    """``"sub_bg:/dark"`` → ``Operation("sub_bg", {"dataset_path": "/dark"})``."""
    kind, _, src = spec.partition(":")  # partition keeps ':' inside "C:\path"
    kind = kind.strip()
    src = src.strip()

    if kind in PLAIN_OPS:
        if src:
            raise SpecError(f"op '{kind}' takes no argument (got '{src}')")
        return Operation(kind, {})

    if kind in SCALAR_OPS:
        try:
            value = float(src) if src else 0.5
        except ValueError:
            raise SpecError(f"op '{kind}' needs a number, e.g. {kind}:0.5") from None
        return Operation(kind, {kind: value})

    if kind in SOURCE_OPS:
        if not src:
            raise SpecError(f"op '{kind}' needs a source, e.g. {kind}:/dark")
        key = "dataset_path" if _is_dataset_ref(src) else "file_path"
        return Operation(kind, {key: src})

    known = ", ".join(PLAIN_OPS + SOURCE_OPS + SCALAR_OPS)
    raise SpecError(f"unknown op '{kind}' (known: {known})")


def format_op(op: Operation) -> str:
    """Inverse of :func:`parse_op` — ``Operation`` → ``"sub_bg:/dark"``."""
    if op.kind in PLAIN_OPS:
        return op.kind
    if op.kind in SCALAR_OPS:
        return f"{op.kind}:{float(op.params.get(op.kind, 0.5)):g}"
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
