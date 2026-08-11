"""Shared fixtures. GUI tests run head-less; nothing touches real user state."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # before any Qt import

import h5py
import numpy as np
import pytest
import tifffile

RING_CENTER = (200.0, 150.0)  # cx, cy
RING_AXES = (120.0, 60.0)  # semi-major a, semi-minor b
RING_ANGLE_DEG = 30.0
RING_K = 1.3  # the bright ring sits at k = 1.3
RING_SIGMA = 0.05  # in k units
RING_AMP = 100.0
RING_FLOOR = 5.0
DETECTOR_PATH = "run/scan00001/det/eh1hama_img/data"


def make_ring_image(h: int = 300, w: int = 400) -> np.ndarray:
    """A Gaussian ring lying on a rotated ellipse — the ground truth for tests."""
    cx, cy = RING_CENTER
    a, b = RING_AXES
    th = np.radians(RING_ANGLE_DEG)
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
    y = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    r = np.sqrt((x / a) ** 2 + (y / b) ** 2)  # r == 1 on the fitted ellipse
    ring = RING_AMP * np.exp(-((r - RING_K) ** 2) / (2 * RING_SIGMA**2))
    return (ring + RING_FLOOR).astype(np.float32)


def ellipse_geom() -> dict:
    a, b = RING_AXES
    return {
        "center_x": RING_CENTER[0],
        "center_y": RING_CENTER[1],
        "semi_major_axis": a,
        "semi_minor_axis": b,
        "angle_major_from_x_deg": RING_ANGLE_DEG,
    }


def ellipse_points(n: int = 8) -> list[list[float]]:
    """Points exactly on the ellipse, so a fit reproduces it."""
    cx, cy = RING_CENTER
    a, b = RING_AXES
    th = np.radians(RING_ANGLE_DEG)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return [
        [
            cx + a * np.cos(u) * np.cos(th) - b * np.sin(u) * np.sin(th),
            cy + a * np.cos(u) * np.sin(th) + b * np.sin(u) * np.cos(th),
        ]
        for u in t
    ]


@pytest.fixture
def ring_image() -> np.ndarray:
    return make_ring_image()


@pytest.fixture
def ring_tif(tmp_path, ring_image):
    path = tmp_path / "ring.tif"
    tifffile.imwrite(path, ring_image)
    return path


@pytest.fixture
def ring_h5(tmp_path, ring_image):
    """Beamline-shaped HDF5: a singleton axis that must be squeezed away."""
    path = tmp_path / "shot.h5"
    with h5py.File(path, "w") as hf:
        hf.create_dataset(DETECTOR_PATH, data=ring_image[None, ...])
        hf.create_dataset("run/scan00001/det/eh1hama_img/dark", data=ring_image[None])
    return path


@pytest.fixture
def points_json(tmp_path):
    import json

    path = tmp_path / "pts.json"
    path.write_text(json.dumps(ellipse_points()), encoding="utf-8")
    return path


# ------------------------------------------------------------------ Qt
@pytest.fixture(scope="session", autouse=True)
def isolated_qsettings(tmp_path_factory):
    """Redirect QSettings to a temp dir — tests must not touch the real app state."""
    from PySide6 import QtCore

    d = tmp_path_factory.mktemp("qsettings")
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(d)
    )
    return d


@pytest.fixture(scope="session")
def qapp(isolated_qsettings):
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def pump(qapp):
    """Run the event loop for a while (offscreen tests have no real loop)."""
    from PySide6 import QtCore

    def _pump(ms: int = 300) -> None:
        t = QtCore.QElapsedTimer()
        t.start()
        while t.elapsed() < ms:
            qapp.processEvents()
            QtCore.QThread.msleep(5)

    return _pump


@pytest.fixture
def main_window(qapp):
    """A fresh window with fresh settings.

    MainWindow records last_file / last_dir and restores them on start, so
    without wiping the store each test would inherit the previous test's tabs.
    """
    from PySide6 import QtCore

    from dfxm.gui.main_window import MainWindow

    QtCore.QSettings("DFXM", "ImageAnalyzer").clear()
    win = MainWindow()
    win.show()
    yield win
    win.close()
