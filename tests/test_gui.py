"""Head-less GUI behaviour — the wiring that keeps burning us, pinned down.

Every test here drives the real MainWindow on the offscreen platform with
synthetic files, so none of it depends on beamline data being present.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
from conftest import ellipse_points, make_ring_image

pytestmark = pytest.mark.usefixtures("qapp")


@pytest.fixture
def three_tifs(tmp_path):
    """Three distinguishable images in one folder."""
    paths = []
    for i in range(3):
        p = tmp_path / f"shot_{i:02d}.tif"
        tifffile.imwrite(p, (make_ring_image() * (i + 1)).astype(np.float32))
        paths.append(p)
    return paths


def kinds(doc) -> list[str]:
    return [op.kind for op in doc.ds.history]


# ---------------------------------------------------------------- branding
def test_logo_assets_ship_with_the_package():
    from dfxm.gui.icons import LOGO_FULL_PATH, LOGO_MARK_PATH

    assert LOGO_MARK_PATH.exists() and LOGO_FULL_PATH.exists()
    # inside the package, so an installed wheel still finds them
    assert LOGO_MARK_PATH.parent.parent.name == "gui"


def test_window_icon_uses_the_artwork_not_the_fallback(main_window):
    assert not main_window.windowIcon().isNull()
    assert main_window.windowIcon().availableSizes(), "a raster logo was expected"


def test_mark_stays_square_at_icon_sizes(qapp):
    from dfxm.gui.icons import logo_pixmap

    for size in (16, 22, 64):
        pm = logo_pixmap(size)
        assert (pm.width(), pm.height()) == (size, size)


def test_wordmark_keeps_its_aspect(qapp):
    from dfxm.gui.icons import wordmark_pixmap

    pm = wordmark_pixmap(64)
    assert pm.height() == 64
    assert pm.width() < pm.height() * 4  # the lockup is portrait-ish, not a banner


def test_wordmark_has_a_readable_variant_per_theme(qapp):
    """The navy wordmark disappears on a dark panel, so a light twin exists."""
    from PySide6 import QtGui

    from dfxm.gui.icons import wordmark_pixmap

    def mean_lightness(pm):
        img = pm.toImage().convertToFormat(QtGui.QImage.Format.Format_ARGB32)
        rows = img.height() * 3 // 4  # the text sits under the badge
        vals = [
            img.pixelColor(x, y).lightness()
            for y in range(rows, img.height())
            for x in range(0, img.width(), 4)
            if img.pixelColor(x, y).alpha() > 128
        ]
        return sum(vals) / max(1, len(vals))

    assert mean_lightness(wordmark_pixmap(160, on_dark=True)) > mean_lightness(
        wordmark_pixmap(160, on_dark=False)
    )


def test_quick_access_bar_is_actually_attached(main_window):
    """It was built and never installed, so the brand mark had nowhere to show."""
    from PySide6 import QtCore, QtWidgets

    ribbon = main_window.menuWidget()
    assert isinstance(ribbon, QtWidgets.QTabWidget)
    corner = ribbon.cornerWidget(QtCore.Qt.TopLeftCorner)
    assert corner is not None and corner.isVisible()
    marks = [
        lbl
        for lbl in corner.findChildren(QtWidgets.QLabel)
        if not lbl.pixmap().isNull()
    ]
    assert marks, "the brand mark should live in the quick-access bar"


# ----------------------------------------------------------- open / tabs
def test_opening_an_image_creates_a_document(main_window, ring_tif):
    main_window.open_path(ring_tif)
    doc = main_window._cur()
    assert doc.file_path == ring_tif
    assert doc.view.has_image()


def test_preview_tab_is_replaced_not_stacked(main_window, three_tifs):
    a, b, _ = three_tifs
    main_window._open_document(a, preview=True)
    tabs_after_first = main_window._tabs.count()
    main_window._open_document(b, preview=True)
    assert main_window._tabs.count() == tabs_after_first
    assert main_window._cur().file_path == b


def test_pinned_tab_survives_the_next_preview(main_window, three_tifs):
    a, b, _ = three_tifs
    main_window._open_document(a, preview=False)
    main_window._open_document(b, preview=True)
    assert main_window._tabs.count() == 2


# ------------------------------------------------- log across documents
def test_log_applies_to_documents_opened_later(main_window, three_tifs):
    """Regression: the checkbox said Log while a new image stayed linear."""
    a, b = three_tifs[0], three_tifs[1]
    main_window._open_document(a, preview=False)
    main_window._log_chk.setChecked(True)
    assert "log" in kinds(main_window._cur())

    main_window._open_document(b, preview=False)
    doc_b = main_window._cur()
    assert "log" in kinds(doc_b)
    assert main_window._log_chk.isChecked()
    assert doc_b.ds.image.max() == pytest.approx(1.0)


def test_turning_log_off_stops_applying_it(main_window, three_tifs):
    a, b = three_tifs[0], three_tifs[1]
    main_window._open_document(a, preview=False)
    main_window._log_chk.setChecked(True)
    main_window._log_chk.setChecked(False)
    main_window._open_document(b, preview=False)
    assert kinds(main_window._cur()) == []


def test_checkbox_follows_the_document_it_belongs_to(main_window, three_tifs):
    a, b = three_tifs[0], three_tifs[1]
    main_window._open_document(a, preview=False)
    main_window._log_chk.setChecked(True)  # a: log on
    main_window._open_document(b, preview=False)
    main_window._log_chk.setChecked(False)  # b: log off
    main_window._tabs.setCurrentIndex(
        main_window._tabs.indexOf(
            main_window._docs
            and next(w for w, d in main_window._docs.items() if d.file_path == a)
        )
    )
    assert main_window._log_chk.isChecked(), "switching back must show a's own state"


# ---------------------------------------------------------- preprocessing
def test_geometric_op_resizes_the_displayed_image(main_window, ring_tif):
    main_window.open_path(ring_tif)
    doc = main_window._cur()
    before = doc.view.image_shape()
    main_window._preproc_add(doc, "scale", {"sx": 0.5, "sy": 1.0})
    assert doc.view.image_shape() == (before[0], before[1] // 2)


def test_preproc_ops_are_per_document(main_window, three_tifs):
    a, b = three_tifs[0], three_tifs[1]
    main_window._open_document(a, preview=False)
    main_window._preproc_add(main_window._cur(), "sqrt")
    main_window._open_document(b, preview=False)
    assert "sqrt" not in kinds(main_window._cur())


# ------------------------------------------------------- ring profile tab
def test_ring_panel_measures_the_active_ellipse(main_window, ring_tif):
    from dfxm.gui.roi import EllipseFitROI

    main_window.open_path(ring_tif)
    doc = main_window._cur()
    doc.ds = doc.ds.fit_ellipse(ellipse_points())
    main_window.add_roi(EllipseFitROI(doc.ds.fit.geom, points=ellipse_points()))
    main_window._ring_panel.refresh()

    prof = main_window._ring_panel.profile
    assert prof is not None
    assert prof.peak()[0] == pytest.approx(1.3, abs=0.01)


def test_ring_panel_ignores_a_document_without_an_ellipse(main_window, ring_tif):
    main_window.open_path(ring_tif)
    main_window._ring_panel.refresh()
    assert main_window._ring_panel.profile is None


def test_ring_panel_measures_linear_data_even_with_log_on(main_window, ring_tif):
    from dfxm.gui.roi import EllipseFitROI

    main_window.open_path(ring_tif)
    doc = main_window._cur()
    doc.ds = doc.ds.fit_ellipse(ellipse_points())
    main_window.add_roi(EllipseFitROI(doc.ds.fit.geom, points=ellipse_points()))
    main_window._ring_panel.refresh()
    linear_peak = main_window._ring_panel.profile.peak()

    main_window._log_chk.setChecked(True)
    main_window._ring_panel.refresh()
    assert main_window._ring_panel.profile.peak() == pytest.approx(linear_peak)


# ------------------------------------------------------ explorer ".." row
def test_up_row_moves_the_tree_to_the_parent(main_window, tmp_path):
    child = tmp_path / "run01"
    child.mkdir()
    main_window._set_tree_root(child)
    assert main_window._tree_root() == child

    main_window._up_btn.click()
    assert main_window._tree_root() == tmp_path


def test_up_row_selects_the_folder_it_came_from(main_window, tmp_path, pump):
    child = tmp_path / "run01"
    child.mkdir()
    main_window._set_tree_root(child)
    main_window._go_up_dir()
    pump(200)
    from pathlib import Path

    current = main_window._fs_model.filePath(main_window._tree.currentIndex())
    assert Path(current) == child  # Qt hands back forward slashes on Windows


def test_backspace_goes_up(main_window, tmp_path, pump):
    from PySide6 import QtCore, QtGui, QtWidgets

    child = tmp_path / "run01"
    child.mkdir()
    main_window._set_tree_root(child)
    main_window._tree.setFocus()
    QtWidgets.QApplication.sendEvent(
        main_window._tree,
        QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress, QtCore.Qt.Key_Backspace, QtCore.Qt.NoModifier
        ),
    )
    pump(200)
    assert main_window._tree_root() == tmp_path


def test_up_row_is_disabled_when_every_drive_is_listed(main_window):
    main_window._fs_model.setRootPath("")
    main_window._tree.setRootIndex(main_window._fs_model.index(""))
    main_window._update_up_row()
    assert main_window._tree_root() is None
    assert not main_window._up_btn.isEnabled()

    main_window._go_up_dir()  # must be a no-op, not a crash
    assert main_window._tree_root() is None


def test_up_row_from_a_drive_root_lists_the_drives(main_window, tmp_path):
    from pathlib import Path

    root = Path(tmp_path.anchor or "/")
    main_window._set_tree_root(root)
    main_window._go_up_dir()
    assert main_window._tree_root() is None


# ------------------------------------------------ explorer arrow-key preview
def test_arrow_keys_move_the_preview(main_window, three_tifs, pump):
    from PySide6 import QtCore, QtGui, QtWidgets

    folder = three_tifs[0].parent
    tree, model = main_window._tree, main_window._fs_model
    model.setRootPath(str(folder))
    tree.setRootIndex(model.index(str(folder)))
    for _ in range(20):  # QFileSystemModel populates asynchronously
        pump(100)
        if model.rowCount(tree.rootIndex()) >= len(three_tifs):
            break
    rows = [
        model.index(r, 0, tree.rootIndex())
        for r in range(model.rowCount(tree.rootIndex()))
    ]
    tifs = [i for i in rows if model.filePath(i).endswith(".tif")]
    assert len(tifs) >= 2

    tree.setCurrentIndex(tifs[0])
    main_window._on_tree_preview(tifs[0])
    pump(200)
    first = main_window._cur().file_path

    tree.setFocus()
    QtWidgets.QApplication.sendEvent(
        tree,
        QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress, QtCore.Qt.Key_Down, QtCore.Qt.NoModifier
        ),
    )
    pump(400)
    assert main_window._cur().file_path != first
    assert main_window._cur().preview is True
    assert main_window._tabs.count() == 1, "navigation must not stack tabs"


def test_arrow_keys_do_nothing_without_a_preview_tab(main_window, three_tifs, pump):
    from PySide6 import QtCore, QtGui, QtWidgets

    folder = three_tifs[0].parent
    tree, model = main_window._tree, main_window._fs_model
    model.setRootPath(str(folder))
    tree.setRootIndex(model.index(str(folder)))
    for _ in range(20):
        pump(100)
        if model.rowCount(tree.rootIndex()) >= len(three_tifs):
            break

    main_window._open_document(three_tifs[0], preview=False)  # pinned
    pinned = main_window._cur().file_path
    tree.setCurrentIndex(model.index(str(three_tifs[0])))
    tree.setFocus()
    QtWidgets.QApplication.sendEvent(
        tree,
        QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress, QtCore.Qt.Key_Down, QtCore.Qt.NoModifier
        ),
    )
    pump(400)
    assert main_window._cur().file_path == pinned
    assert main_window._tabs.count() == 1
