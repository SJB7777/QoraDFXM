"""HDF5 loading helpers for DFXM detector images.

The files produced by the beamline store one or more scans under ``run/``,
each holding a ``det`` group with one or more detectors.  A single detector
frame lives at::

    run/<scan>/det/<detector>/data

and, after squeezing singleton dimensions, is a 2-D image.  These helpers
enumerate the available scans / detectors so the GUI can offer them for
selection, and load the chosen frame as a float32 array.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tifffile


@dataclass(frozen=True)
class FramePath:
    """Identifies a single 2-D frame inside an HDF5 file."""

    scan: str
    detector: str

    @property
    def dataset_path(self) -> str:
        return f"run/{self.scan}/det/{self.detector}/data"

    def __str__(self) -> str:
        return f"{self.scan} / {self.detector}"


def list_frames(h5_file: Path | str) -> list[FramePath]:
    """Return every ``(scan, detector)`` pair that holds a ``data`` dataset.

    Missing ``run`` group or malformed files yield an empty list rather than
    raising, so the caller can report "nothing to show" cleanly.
    """
    frames: list[FramePath] = []
    with h5py.File(h5_file, "r") as hf:
        run = hf.get("run")
        if run is None:
            return frames
        for scan in run.keys():
            det = run.get(f"{scan}/det")
            if det is None:
                continue
            for detector in det.keys():
                if "data" in det[detector]:
                    frames.append(FramePath(scan=scan, detector=detector))
    return frames


@dataclass
class H5Node:
    """One entry in an HDF5 tree (group or dataset)."""

    name: str
    path: str
    is_group: bool
    shape: tuple | None = None
    dtype: str | None = None
    attrs: dict | None = None
    children: list["H5Node"] | None = None


def read_structure(h5_file: Path | str) -> H5Node:
    """Walk the whole file into a nested :class:`H5Node` tree for display."""

    def visit(name: str, obj) -> H5Node:
        attrs = {k: _attr_repr(v) for k, v in obj.attrs.items()}
        if isinstance(obj, h5py.Group):
            children = [visit(f"{name}/{k}" if name else k, obj[k]) for k in obj.keys()]
            return H5Node(
                name=name.split("/")[-1] or "/",
                path="/" + name if name else "/",
                is_group=True,
                attrs=attrs,
                children=children,
            )
        return H5Node(
            name=name.split("/")[-1],
            path="/" + name,
            is_group=False,
            shape=tuple(obj.shape),
            dtype=str(obj.dtype),
            attrs=attrs,
        )

    with h5py.File(h5_file, "r") as hf:
        root = H5Node(name="/", path="/", is_group=True, attrs={}, children=[])
        root.children = [visit(k, hf[k]) for k in hf.keys()]
    return root


def _attr_repr(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray):
        return np.array2string(v, threshold=8, edgeitems=3)
    return str(v)


def load_dataset(h5_file: Path | str, dataset_path: str) -> np.ndarray:
    """Load an arbitrary dataset by its full path, squeezed to 2-D float32."""
    with h5py.File(h5_file, "r") as hf:
        img = hf[dataset_path][()].squeeze().astype(np.float32)
    if img.ndim != 2:
        raise ValueError(
            f"Dataset {dataset_path} is not 2-D after squeeze: shape={img.shape}"
        )
    return img


IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
H5_SUFFIXES = (".h5", ".hdf5")
TEXT_SUFFIXES = (".json", ".txt")


def load_image_file(path: Path | str) -> np.ndarray:
    """Load a plain image file (tif/png/jpg/...) as a 2-D float32 array.

    Colour images are converted to grayscale luminance so the analysis tools
    (levels, profile, fitting) apply uniformly.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        img = tifffile.imread(path)
    else:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image: {path}")

    img = np.asarray(img)
    if img.ndim == 3:
        # Drop alpha, average channels (cv2 is BGR but luminance-avg is fine).
        img = img[..., :3].mean(axis=2)
    if img.ndim != 2:
        raise ValueError(f"Unsupported image shape {img.shape} in {path}")
    return img.astype(np.float32)


def load_frame(h5_file: Path | str, frame: FramePath | None = None) -> np.ndarray:
    """Load a 2-D detector frame as float32.

    If ``frame`` is None the first available frame is used, mirroring the
    original ``h5totif`` behaviour (first scan, first detector).
    """
    with h5py.File(h5_file, "r") as hf:
        if frame is None:
            found = list_frames(h5_file)
            if not found:
                raise ValueError(f"No detector frames found in {h5_file}")
            frame = found[0]

        img = hf[frame.dataset_path][()].squeeze().astype(np.float32)

    if img.ndim != 2:
        raise ValueError(
            f"Frame {frame} in {h5_file} is not 2-D after squeeze: shape={img.shape}"
        )
    return img
