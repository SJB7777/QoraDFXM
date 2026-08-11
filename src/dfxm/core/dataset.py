"""DFXMDataset — the immutable, fluent domain object.

Every preprocessing method returns a NEW dataset that shares the (never
mutated) ``raw`` array and carries a grown :class:`History`. Because the recipe
is data, not code, a dataset supports Replay (rebuild from history), Undo
(``.undo()`` → parent recipe) and full serialization for SQLite sessions.

    ds = (DFXMDataset.from_h5(path, "/run/s1/det/d/data")
            .sub_bg(dataset_path="/dark")
            .divide(file_path="flat.tif")
            .apply_log()
            .fit_ellipse(points))
    img  = ds.image        # raw with history replayed (cached)
    row  = ds.to_record()  # one Master-table row
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import io as _io
from .history import History
from .ops import BG_KINDS, NONLINEAR_KINDS, Operation, apply_op
from .results import FitResult


@dataclass
class DFXMDataset:
    raw: np.ndarray
    source_path: Path | None = None  # base for dataset_path refs
    meta: dict = field(default_factory=dict)  # shot_id, frame label, ...
    history: History = field(default_factory=History)
    fit: FitResult | None = None
    # memoized processed image; excluded from equality, reset by _with()
    _cache: np.ndarray | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------ constructors
    @classmethod
    def from_array(cls, arr, source_path=None, meta=None) -> DFXMDataset:
        return cls(
            raw=np.asarray(arr, dtype=np.float32),
            source_path=Path(source_path) if source_path else None,
            meta=dict(meta or {}),
        )

    @classmethod
    def from_h5(cls, path, dataset_path, meta=None) -> DFXMDataset:
        path = Path(path)
        arr = _io.load_dataset(path, dataset_path)
        m = {"shot_id": f"{path.stem}:{dataset_path}", "dataset_path": dataset_path}
        m.update(meta or {})
        return cls.from_array(arr, source_path=path, meta=m)

    @classmethod
    def from_frame(cls, path, frame: _io.FramePath | None = None) -> DFXMDataset:
        path = Path(path)
        arr = _io.load_frame(path, frame)
        label = str(frame) if frame is not None else "frame0"
        m = {"shot_id": f"{path.stem}:{label}", "frame": label}
        return cls.from_array(arr, source_path=path, meta=m)

    @classmethod
    def from_image_file(cls, path) -> DFXMDataset:
        path = Path(path)
        arr = _io.load_image_file(path)
        return cls.from_array(arr, source_path=path, meta={"shot_id": path.stem})

    # ------------------------------------------------------ fluent core
    def _with(self, **kw) -> DFXMDataset:
        """Return a copy with the processed-image cache invalidated."""
        return replace(self, _cache=None, **kw)

    def _add_op(self, kind: str, **params) -> DFXMDataset:
        return self._with(history=self.history.add(Operation(kind, params)))

    def sub_bg(self, **source) -> DFXMDataset:
        """Dark subtract. Source: dataset_path=... or file_path=..."""
        return self._add_op("sub_bg", **source)

    def divide(self, **source) -> DFXMDataset:
        """Flat-field divide. Source: dataset_path=... or file_path=..."""
        return self._add_op("divide", **source)

    def apply_log(self) -> DFXMDataset:
        return self._add_op("log")

    def pure_log(self) -> DFXMDataset:
        return self._add_op("pure_log")

    def sqrt(self) -> DFXMDataset:
        return self._add_op("sqrt")

    def gamma(self, g: float = 0.5) -> DFXMDataset:
        return self._add_op("gamma", gamma=float(g))

    def normalize(self) -> DFXMDataset:
        return self._add_op("normalize")

    # -------------------------------------------------- geometric (warp)
    # These change the image SHAPE, so anything picked on the old grid (fit
    # points, ROIs) no longer lines up — apply them before fitting.
    def scale(self, sx: float, sy: float | None = None, interp="auto") -> DFXMDataset:
        """Resize by factors; ``sx != sy`` changes the aspect ratio."""
        return self._add_op(
            "scale", sx=float(sx), sy=float(sx if sy is None else sy), interp=interp
        )

    def rotate(self, angle_deg: float, expand: bool = True) -> DFXMDataset:
        """Rotate about the centre, positive = counter-clockwise."""
        return self._add_op("rotate", angle=float(angle_deg), expand=bool(expand))

    def flip(self, axis: str = "h") -> DFXMDataset:
        """Mirror: ``h`` (left↔right), ``v`` (top↔bottom) or ``both``."""
        return self._add_op("flip", axis=axis)

    def add_op(self, kind: str, **params) -> DFXMDataset:
        """Generic append (used when replaying a GUI/serialized recipe)."""
        return self._add_op(kind, **params)

    def fit_ellipse(self, points) -> DFXMDataset:
        """Fit an ellipse to picked points (on the PROCESSED image)."""
        return self._with(fit=FitResult.from_points(points))

    # ------------------------------------------------------ measurements
    def linear_view(self) -> DFXMDataset:
        """Same recipe with the intensity-bending ops (log/sqrt/gamma) dropped.

        Geometry and background correction are kept, so coordinates picked on
        the displayed image still line up — but the values are linear again,
        which is what any quantitative measurement needs.
        """
        ops = [op for op in self.history if op.kind not in NONLINEAR_KINDS]
        return self.set_history(History(tuple(ops)))

    def ring_profile(self, **kw):
        """Brightness along the fitted ellipse vs. its scale (see core.profile).

        Always measured on :meth:`linear_view`; requires a fit.
        """
        from .profile import ring_profile as _ring_profile

        if self.fit is None:
            raise ValueError("no ellipse fit on this dataset — fit first")
        return _ring_profile(self.linear_view().image, self.fit, **kw)

    # ------------------------------------------------------ history ops
    def undo(self) -> DFXMDataset:
        return self._with(history=self.history.pop())

    def set_history(self, history: History) -> DFXMDataset:
        return self._with(history=history)

    def replay(self, history: History) -> DFXMDataset:
        """New dataset from the same raw with a given recipe (Replay/Macro)."""
        return DFXMDataset.from_array(
            self.raw, source_path=self.source_path, meta=dict(self.meta)
        )._with(history=history)

    # ------------------------------------------------------ derived data
    @property
    def image(self) -> np.ndarray:
        """raw with the full history applied, in order (memoized)."""
        if self._cache is None:
            img = np.asarray(self.raw, dtype=np.float32)
            for op in self.history:
                img = apply_op(op, img, self.source_path)
            self._cache = img
        return self._cache

    @property
    def bg_applied(self) -> bool:
        return any(op.kind in BG_KINDS for op in self.history)

    @property
    def log_scale(self) -> bool:
        return any(op.kind in ("log", "pure_log") for op in self.history)

    # ------------------------------------------------------ export
    def to_record(self, status: str = "OK") -> dict | None:
        """One Master-table row, or None if not yet fitted."""
        if self.fit is None:
            return None
        return self.fit.to_row(
            shot_id=self.meta.get("shot_id", ""),
            status=status,
            bg_applied=self.bg_applied,
            log_scale=self.log_scale,
        )

    def to_dict(self) -> dict:
        """Serialize recipe + result + identity (arrays excluded → SQLite)."""
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "meta": dict(self.meta),
            "history": self.history.to_list(),
            "fit": self.fit.to_dict() if self.fit else None,
        }

    @classmethod
    def from_dict(cls, d: dict, raw=None) -> DFXMDataset:
        """Rebuild from a serialized session. If ``raw`` is None it is reloaded
        from ``source_path`` (needs meta['dataset_path'] for h5)."""
        src = d.get("source_path")
        src = Path(src) if src else None
        meta = dict(d.get("meta", {}))
        if raw is None:
            if src is None:
                raise ValueError("Cannot rebuild dataset: no raw and no source_path")
            if "dataset_path" in meta:
                raw = _io.load_dataset(src, meta["dataset_path"])
            elif src.suffix.lower() in _io.H5_SUFFIXES:
                raw = _io.load_frame(src)
            else:
                raw = _io.load_image_file(src)
        ds = cls.from_array(raw, source_path=src, meta=meta)
        ds = ds._with(history=History.from_list(d.get("history", [])))
        if d.get("fit"):
            ds = ds._with(fit=FitResult.from_dict(d["fit"]))
        return ds
