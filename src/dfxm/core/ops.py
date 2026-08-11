"""Preprocessing operations as pure, serializable records.

An :class:`Operation` stores a *kind* plus *params*. Crucially, reference data
(dark / flat frames) is stored as a **location** (``dataset_path`` inside the
dataset's own h5, or an absolute ``file_path``), NOT as a raw array. That keeps
the whole history JSON-serializable — the foundation for Replay, Undo/Redo and
SQLite session storage. Arrays are resolved lazily at apply time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import io as _io
from . import warp as _warp
from .transform import adaptive_log

# Human-readable labels, shared by GUI and CLI.
OP_LABELS = {
    "sub_bg": "Dark 빼기 (A − D)",
    "divide": "Flat 나누기 (A / F)",
    "log": "Log 변환 (adaptive)",
    "pure_log": "Log 변환 (pure)",
    "sqrt": "제곱근 (√)",
    "gamma": "감마 (γ)",
    "normalize": "정규화 (/max)",
    "scale": "크기·비율 변형",
    "rotate": "회전",
    "flip": "뒤집기",
}

# Ops that resample the pixel grid — they change the image shape and invalidate
# coordinates picked before them (see dfxm.core.warp).
GEOMETRIC_KINDS = ("scale", "rotate", "flip")


def _fmt_detail(kind: str, p: dict) -> str:
    """The parameter part of an op's label ('' for parameterless ops)."""
    if kind == "gamma":
        return f"={float(p.get('gamma', 0.5)):g}"
    if kind == "scale":
        sx = float(p.get("sx", 1.0))
        return f"  {sx:g} × {float(p.get('sy', sx)):g}"
    if kind == "rotate":
        return f"  {float(p.get('angle', 0.0)):+g}°"
    if kind == "flip":
        return f"  ({p.get('axis', 'h')})"
    return ""


@dataclass(frozen=True)
class Operation:
    kind: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: dict) -> Operation:
        return cls(d["kind"], dict(d.get("params", {})))

    @property
    def source(self) -> str | None:
        return self.params.get("dataset_path") or self.params.get("file_path")

    def label(self) -> str:
        base = OP_LABELS.get(self.kind, self.kind) + _fmt_detail(self.kind, self.params)
        return f"{base}  ⟵ {self.source}" if self.source else base


def _resolve_ref(params: dict, base_path) -> np.ndarray | None:
    """Load a reference array from its stored location."""
    if "dataset_path" in params:
        if base_path is None:
            return None
        return np.asarray(
            _io.load_dataset(base_path, params["dataset_path"]), dtype=np.float32
        )
    if "file_path" in params:
        return np.asarray(_io.load_image_file(params["file_path"]), dtype=np.float32)
    return None


def _op_sub_bg(img, params, base):
    ref = _resolve_ref(params, base)
    if ref is None or ref.shape != img.shape:
        return img
    return img - ref


def _op_divide(img, params, base):
    ref = _resolve_ref(params, base)
    if ref is None or ref.shape != img.shape:
        return img
    denom = np.where(np.abs(ref) < 1e-12, np.nan, ref)
    return img / denom


def _op_log(img, params, base):
    return adaptive_log(img)


def _op_pure_log(img, params, base):
    """Plain log1p (no adaptive [0,1] rescale). Negatives clipped to 0."""
    return np.log1p(np.clip(img, 0.0, None))


def _op_sqrt(img, params, base):
    return np.sqrt(np.clip(img, 0.0, None))


def _op_gamma(img, params, base):
    """Power-law (gamma) correction on the non-negative signal."""
    g = float(params.get("gamma", 0.5))
    return np.clip(img, 0.0, None) ** g


def _op_normalize(img, params, base):
    finite = img[np.isfinite(img)]
    mx = float(finite.max()) if finite.size else 0.0
    return img / mx if mx > 0 else img


def _op_scale(img, params, base):
    sx = float(params.get("sx", 1.0))
    return _warp.scale(
        img, sx, params.get("sy", sx), interp=params.get("interp", "auto")
    )


def _op_rotate(img, params, base):
    return _warp.rotate(
        img,
        float(params.get("angle", 0.0)),
        expand=bool(params.get("expand", True)),
        interp=params.get("interp", "linear"),
        fill=float(params.get("fill", 0.0)),
    )


def _op_flip(img, params, base):
    return _warp.flip(img, str(params.get("axis", "h")))


_OPS = {
    "sub_bg": _op_sub_bg,
    "divide": _op_divide,
    "log": _op_log,
    "pure_log": _op_pure_log,
    "sqrt": _op_sqrt,
    "gamma": _op_gamma,
    "normalize": _op_normalize,
    "scale": _op_scale,
    "rotate": _op_rotate,
    "flip": _op_flip,
}

# Op kinds that count as a background/intensity correction (for bg_applied flag).
BG_KINDS = ("sub_bg", "divide", "normalize")

# Ops that bend the intensity axis. Quantitative measurements (ring profiles,
# integrated intensity) must run with these stripped — mean(log I) != log(mean I).
# `normalize` is a plain scalar divide, so it stays.
NONLINEAR_KINDS = ("log", "pure_log", "sqrt", "gamma")


def apply_op(op: Operation, img: np.ndarray, base_path=None) -> np.ndarray:
    fn = _OPS.get(op.kind)
    if fn is None:
        return img
    return fn(img, op.params, base_path)
