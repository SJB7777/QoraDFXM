"""View preferences: one owner per piece of state, and the scoping rules."""

from __future__ import annotations

import pytest
import tifffile
from conftest import make_ring_image

from qoradfxm.gui.prefs import ViewPrefs


class FakeView:
    """Records what apply_to() pushes, so the contract is checked, not mocked."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, args))

        return record


def test_prefs_never_holds_intensity_state():
    """log/gamma live in doc.ds.history; a copy here could only disagree."""
    names = set(ViewPrefs().to_dict())
    assert not names & {"log", "gamma", "sqrt", "normalize", "history"}


def test_copy_does_not_mutate_the_original():
    base = ViewPrefs()
    changed = base.copy(colormap="viridis")
    assert base.colormap == "gray" and changed.colormap == "viridis"


def test_apply_to_pushes_every_field(qapp):
    view = FakeView()
    ViewPrefs(
        colormap="magma",
        overmax=False,
        scale_on=True,
        px_size=0.5,
        unit="nm",
        tool="line",
    ).apply_to(view)
    pushed = dict(view.calls)
    assert pushed["set_colormap"] == ("magma",)
    assert pushed["set_overmax"] == (False,)
    assert pushed["set_scale"] == (0.5, "nm")
    assert pushed["set_tool"] == ("line",)
    assert "set_scale_pixels" not in pushed


def test_apply_to_falls_back_to_pixel_scale(qapp):
    view = FakeView()
    ViewPrefs(scale_on=False).apply_to(view)
    assert any(name == "set_scale_pixels" for name, _ in view.calls)


def test_round_trips_through_qsettings(qapp, tmp_path):
    from PySide6 import QtCore

    settings = QtCore.QSettings(str(tmp_path / "prefs.ini"), QtCore.QSettings.IniFormat)
    saved = ViewPrefs(
        colormap="plasma",
        overmax=False,
        overmax_color="#00ff00",
        scale_on=True,
        px_size=0.25,
        unit="nm",
    )
    saved.save(settings)
    loaded = ViewPrefs.load(settings)
    assert loaded.to_dict() | {"tool": saved.tool} == saved.to_dict()


def test_tool_is_not_persisted(qapp, tmp_path):
    from PySide6 import QtCore

    settings = QtCore.QSettings(str(tmp_path / "p.ini"), QtCore.QSettings.IniFormat)
    ViewPrefs(tool="ellipse").save(settings)
    assert ViewPrefs.load(settings).tool == "select"


# ------------------------------------------------------- window semantics
@pytest.fixture
def two_tifs(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"img_{i}.tif"
        tifffile.imwrite(p, make_ring_image())
        paths.append(p)
    return paths


def test_panel_change_applies_to_the_current_document(main_window, two_tifs):
    main_window.open_path(two_tifs[0])
    main_window._cmap_combo.setCurrentText("viridis")
    assert main_window._cur().prefs.colormap == "viridis"


def test_new_documents_inherit_the_current_choice(main_window, two_tifs):
    main_window.open_path(two_tifs[0])
    main_window._cmap_combo.setCurrentText("magma")
    main_window._scale_chk.setChecked(True)
    main_window._px_size_spin.setValue(0.25)

    main_window.open_path(two_tifs[1])
    prefs = main_window._cur().prefs
    assert prefs.colormap == "magma"
    assert prefs.scale_on and prefs.px_size == pytest.approx(0.25)


def test_each_document_keeps_its_own_prefs(main_window, two_tifs):
    main_window.open_path(two_tifs[0])
    main_window._cmap_combo.setCurrentText("viridis")
    main_window.open_path(two_tifs[1])
    main_window._cmap_combo.setCurrentText("inferno")

    first = next(w for w, d in main_window._docs.items() if d.file_path == two_tifs[0])
    main_window._tabs.setCurrentWidget(first)
    assert main_window._cur().prefs.colormap == "viridis"
    assert main_window._cmap_combo.currentText() == "viridis"


def test_documents_do_not_share_one_prefs_object(main_window, two_tifs):
    main_window.open_path(two_tifs[0])
    main_window.open_path(two_tifs[1])
    docs = list(main_window._docs.values())
    assert docs[0].prefs is not docs[1].prefs


def test_choice_survives_a_new_window(main_window, two_tifs, qapp):
    from qoradfxm.gui.main_window import MainWindow

    main_window.open_path(two_tifs[0])
    main_window._cmap_combo.setCurrentText("cividis")

    second = MainWindow()  # same QSettings store, fresh window
    try:
        assert second._app_prefs.colormap == "cividis"
        assert second._cmap_combo.currentText() == "cividis"
    finally:
        second.close()
