"""Main application window: folder browser, image view, controls."""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import io
from .image_view import COLORMAPS, ImageView
from .roi import EllipseFitROI, LineProfileROI, RectRegionROI
from .icons import AppIcons, logo_icon, logo_pixmap


APP_NAME = "DFXM OptiCalc"
APP_VERSION = "0.1.0"


@dataclass
class DocumentSession:
    """Independent per-file session: image, ROI objects, logs, view settings."""

    file_path: Path
    kind: str  # 'image' | 'text'
    view: ImageView | None = None
    frames: list = field(default_factory=list)
    frame: object = None
    structure: object = None
    preview: bool = False
    info: str = ""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    logs: list = field(default_factory=list)
    # roi_id -> AnalysisROI (Phase B multi-object management).
    roi_objects: dict = field(default_factory=dict)
    view_settings: dict = field(default_factory=dict)

    def add_log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{ts}] {message}")

    # --- ROI management ----------------------------------------------------
    def add_roi(self, roi) -> None:
        self.roi_objects[roi.roi_id] = roi

    def remove_roi(self, roi_id: str):
        return self.roi_objects.pop(roi_id, None)

    def get_roi(self, roi_id: str):
        return self.roi_objects.get(roi_id)

    def rois_of_type(self, roi_type: str) -> list:
        return [r for r in self.roi_objects.values() if r.roi_type == roi_type]


_STYLE_TMPL = """
QMainWindow { background: %BASE%; }
QWidget { background: %WIDGET%; color: %TEXT%; font-size: 9pt; }
QToolTip { background: %SURF%; color: %TEXT%; border: 1px solid %BORDER2%; padding: 3px; }
QMenu { background: %SURF%; border: 1px solid %BORDER2%; }
QMenu::item:selected { background: %ACCBG%; }

QDockWidget { titlebar-close-icon: none; }
QDockWidget::title {
    background: %PANEL%; padding: 8px 12px; font-weight: bold; font-size: 9pt;
    color: %TEXT2%; border-bottom: 1px solid %BORDER%;
}

QGroupBox { border: 1px solid %BORDER%; border-radius: 6px; margin-top: 8px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: %DIM%; }

QPushButton { background: %BTN%; border: 1px solid %BORDER2%; border-radius: 5px; padding: 6px 11px; }
QPushButton:hover { background: %HOVER2%; border-color: %BORDER2%; }
QPushButton:pressed { background: %PANEL%; }

QTreeView, QTreeWidget, QTableWidget, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background: %INPUT%; border: 1px solid %BORDER%; border-radius: 5px; padding: 3px;
    selection-background-color: %ACCBG%;
}
QComboBox::drop-down { border: none; width: 18px; }
QHeaderView::section { background: %SURF%; padding: 4px; border: none; color: %DIM%; }

QTabWidget::pane { border: 1px solid %BORDER%; border-radius: 6px; top: -1px; }
QTabBar { background: transparent; }
QTabBar::tab {
    background: %PANEL%; color: %DIM%; padding: 6px 16px; margin-right: 2px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    border: 1px solid transparent;
}
QTabBar::tab:hover { background: %HOVER%; color: %TEXT2%; }
QTabBar::tab:selected {
    background: %TABSEL%; color: %SELTEXT%; border-bottom: 2px solid %ACCENT%;
}

QStatusBar { background: %STATUSBG%; min-height: 24px; font-size: 8pt; color: %DIM%; }
QStatusBar { border-top: 1px solid %BORDER%; }
QStatusBar::item { border: none; }

QMainWindow::separator { background: %BORDER%; width: 4px; height: 4px; }
QMainWindow::separator:hover { background: %ACCENT%; }

QToolButton { border: 1px solid transparent; border-radius: 5px; padding: 3px 6px; color: %TEXT%; }
QToolButton:hover { background: %HOVER%; border-color: %BORDER2%; }
QToolButton:checked { background: %ACCBG%; color: %SELTEXT%; }
QScrollArea { border: 1px solid %BORDER%; }

QCheckBox { 
    padding: 2px 0; 
    spacing: 6px;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid %BORDER2%;
    border-radius: 3px;
    background: %INPUT%;
}

QCheckBox::indicator:hover {
    border-color: %ACCENT%;
}

QCheckBox::indicator:checked {
    width: 14px;
    height: 14px;
    background: %ACCENT%;
    border: 1px solid %ACCENT%;
    image: url("%CHECKMARK_ICON%");
    padding: 0px;
}
QLabel { background: transparent; }

QSlider::groove:horizontal { height: 4px; background: %HOVER2%; border-radius: 2px; }
QSlider::handle:horizontal { background: %ACCENT%; width: 12px; margin: -5px 0; border-radius: 6px; }

#Ribbon { background: %STATUSBG%; }
#Ribbon::pane { background: %RIBBON%; border: none; border-bottom: 1px solid %SHADOW%; }
#Ribbon QTabBar { background: %STATUSBG%; }
#Ribbon QTabBar::tab {
    background: transparent; color: %DIM%; padding: 7px 20px;
    margin: 0; border: 2px solid transparent;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}
#Ribbon QTabBar::tab:hover { color: %TEXT2%; background: %HOVER%; }
#Ribbon QTabBar::tab:selected {
    color: %SELTEXT%; background: %RIBBON%; border-top: 2px solid %ACCENT%;
    border-bottom: none;
}

.SectionHeader {
    background: %SURF%; border: none; border-left: 3px solid %ACCENT%;
    border-radius: 4px; padding: 6px 10px;
    text-align: left; font-weight: bold; color: %TEXT2%;
}
.SectionHeader:hover { background: %HOVER%; }

.RibbonBtn { background: transparent; border: 1px solid %TABSEL%; border-radius: 5px; }
.RibbonBtn:hover { background: %HOVER%; border-color: %BORDER2%; }
.RibbonBtn:checked { background: %ACCBG%; border-color: %ACCENT%; }
.RibbonBtn:pressed { background: %PANEL%; }

/* Theme-aware surfaces that used to be hard-coded (fixes Light mode) */
#RibbonPage { background: %RIBBON%; }
#RibbonGroupBox, #RibbonHolder { background: transparent; }
#SectionBody { border: 1px solid %BORDER%; border-top: none; border-radius: 0 0 5px 5px; }
QFrame#RibbonDiv { background: %BORDER%; }
QFrame#QSep { background: %BORDER%; }
QLabel#RibbonCap { color: %DIM%; background: transparent; }
QDockWidget { border: 1px solid %BORDER%; }
"""

_PALETTES = {
    "Dark": {
        "BASE": "#1b1b1d", "WIDGET": "#232326", "PANEL": "#202024", "SURF": "#26262a",
        "RIBBON": "#252526", "INPUT": "#191919", "BORDER": "#303036",
        "BORDER2": "#3a3a42", "BTN": "#2e2e33", "HOVER": "#2c2c32", "HOVER2": "#38383e",
        "TEXT": "#e4e4e7", "TEXT2": "#d4d4d8", "DIM": "#9a9aa2", "ACCBG": "#234844",
        "SELTEXT": "#ffffff", "STATUSBG": "#1a1a1d", "TABSEL": "#2b2b31",
        "SHADOW": "#141416", "ACCENT": "#00e5ff",
    },
    "Light": {
        "BASE": "#e5e5ea", "WIDGET": "#f1f1f4", "PANEL": "#e2e2e8", "SURF": "#ececef",
        "RIBBON": "#f7f7f9", "INPUT": "#ffffff", "BORDER": "#c6c6cf",
        "BORDER2": "#b2b2bd", "BTN": "#e9e9ef", "HOVER": "#dcdce3", "HOVER2": "#d2d2da",
        "TEXT": "#1c1c22", "TEXT2": "#2a2a32", "DIM": "#6a6a74", "ACCBG": "#c9edf3",
        "SELTEXT": "#0b1113", "STATUSBG": "#dcdce2", "TABSEL": "#f7f7f9",
        "SHADOW": "#b6b6be", "ACCENT": "#0891b2",
    },
}


def build_style(theme: str) -> str:
    toks = _PALETTES.get(theme, _PALETTES["Dark"])
    s = _STYLE_TMPL
    for k, v in toks.items():
        s = s.replace("%" + k + "%", v)
    
    # qtawesome 체크마크 아이콘을 임시 PNG로 저장하여 QSS에 주입
    check_pix = AppIcons.get_pixmap(AppIcons.CHECK, size=12, color="#ffffff")
    tmp_check_path = Path(tempfile.gettempdir()) / "dfxm_qta_check.png"
    check_pix.save(str(tmp_check_path))
    
    s = s.replace("%CHECKMARK_ICON%", str(tmp_check_path).replace("\\", "/"))
    return s


class CollapsibleSection(QtWidgets.QWidget):
    """A titled section whose body folds away when the header is clicked."""

    def __init__(self, title: str, body: QtWidgets.QWidget, expanded: bool = True):
        super().__init__()
        self._body = body
        self._header = QtWidgets.QToolButton()
        self._header.setText(f"  {title}")
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.setProperty("class", "SectionHeader")
        self._header.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._header.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self._header.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self._header.toggled.connect(self._on_toggle)

        self._body.setObjectName("SectionBody")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 6)
        lay.setSpacing(0)
        lay.addWidget(self._header)
        lay.addWidget(self._body)
        self._body.setVisible(expanded)

    def _on_toggle(self, on: bool) -> None:
        self._body.setVisible(on)
        self._header.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)


class DropOverlay(QtWidgets.QWidget):
    """Full-window translucent overlay shown while dragging files in."""

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet("background: rgba(0,0,0,0.6);")
        self._box = QtWidgets.QLabel()
        self._box.setAlignment(QtCore.Qt.AlignCenter)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addStretch(1)
        lay.addWidget(self._box, 0, QtCore.Qt.AlignCenter)
        lay.addStretch(1)
        self.hide()

    def set_ok(self, ok: bool) -> None:
        color = "#4da3ff" if ok else "#ff5252"
        text = (
            "📥  여기에 파일 또는 폴더를 떨어뜨리세요\n\n"
            "지원 형식: .h5  .tif  .png  .jpg  .json ..."
            if ok
            else "⚠  지원하지 않는 파일 형식입니다"
        )
        self._box.setText(text)
        self._box.setStyleSheet(
            f"border: 3px dashed {color}; border-radius: 18px; color: #fff;"
            f"padding: 60px 80px; font-size: 13pt; font-weight: bold;"
            f"background: rgba(255,255,255,0.03);"
        )

    def cover(self) -> None:
        self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()


class SettingsDialog(QtWidgets.QDialog):
    """Sidebar-tabbed preferences: General / Display / Shortcuts."""

    def __init__(self, parent, settings: QtCore.QSettings):
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle("환경설정")
        self.setMinimumSize(560, 420)

        nav = QtWidgets.QListWidget()
        nav.setFixedWidth(140)
        nav.addItems(["일반", "디스플레이", "단축키"])
        stack = QtWidgets.QStackedWidget()
        stack.addWidget(self._general_page())
        stack.addWidget(self._display_page())
        stack.addWidget(self._shortcuts_page())
        nav.currentRowChanged.connect(stack.setCurrentIndex)
        nav.setCurrentRow(0)

        body = QtWidgets.QHBoxLayout()
        body.addWidget(nav)
        body.addWidget(stack, 1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(body, 1)
        root.addWidget(buttons)

    def _general_page(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(w)

        # Brand header: logo + name.
        brand = QtWidgets.QHBoxLayout()
        logo = QtWidgets.QLabel()
        logo.setPixmap(logo_pixmap(72))
        brand.addWidget(logo)
        title = QtWidgets.QLabel(f"<b style='font-size:15pt'>{APP_NAME}</b>"
                                 f"<br><span style='color:#888'>v{APP_VERSION}</span>")
        brand.addWidget(title, 1)
        outer.addLayout(brand)
        outer.addSpacing(8)

        form_w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(form_w)
        outer.addWidget(form_w)
        self.lang_combo = QtWidgets.QComboBox()
        self.lang_combo.addItems(["한국어 (KO)", "English (EN)"])
        self.lang_combo.setCurrentText(
            self._settings.value("lang", "한국어 (KO)", type=str)
        )
        self.lang_combo.currentTextChanged.connect(
            lambda s: self._settings.setValue("lang", s)
        )
        f.addRow("언어", self.lang_combo)

        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setCurrentText(self._settings.value("theme", "Dark", type=str))
        _parent = self.parent()
        if hasattr(_parent, "apply_theme"):
            self.theme_combo.currentTextChanged.connect(_parent.apply_theme)
        f.addRow("테마", self.theme_combo)

        note = QtWidgets.QLabel("언어 전환은 설정 저장 후 반영 예정.")
        note.setStyleSheet("color:#888;")
        f.addRow(note)
        outer.addStretch(1)
        return w

    def _display_page(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.def_cmap = QtWidgets.QComboBox()
        self.def_cmap.addItems(COLORMAPS)
        self.def_cmap.setCurrentText(self._settings.value("def_cmap", "gray", type=str))
        f.addRow("기본 Colormap", self.def_cmap)
        self.def_unit = QtWidgets.QComboBox()
        self.def_unit.setEditable(True)
        self.def_unit.addItems(["µm", "nm", "mm", "Å"])
        self.def_unit.setCurrentText(self._settings.value("def_unit", "µm", type=str))
        f.addRow("기본 Scalebar 단위", self.def_unit)
        return w

    def _shortcuts_page(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        table = QtWidgets.QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["기능", "단축키"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().hide()
        rows = [
            ("이동(Select) 모드", "Esc"),
            ("기본 줌 복귀", "더블클릭"),
            ("점 삭제(가까운)", "우클릭"),
            ("스크린샷", "—"),
        ]
        table.setRowCount(len(rows))
        for i, (a, b) in enumerate(rows):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(a))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(b))
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        lay.addWidget(table)
        return w

    def result_values(self) -> dict:
        return {
            "lang": self.lang_combo.currentText(),
            "theme": self.theme_combo.currentText(),
            "def_cmap": self.def_cmap.currentText(),
            "def_unit": self.def_unit.currentText(),
        }


class MainWindow(QtWidgets.QMainWindow):
    _SUPPORTED = io.H5_SUFFIXES + io.IMAGE_SUFFIXES + io.TEXT_SUFFIXES

    def __init__(self) -> None:
        super().__init__()

        self._default_overmax_color: QtGui.QColor = QtGui.QColor("#ff0000")

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(logo_icon())
        self.resize(1500, 950)

        self._current_file: Path | None = None
        self._frames: list[io.FramePath] = []
        self._settings = QtCore.QSettings("DFXM", "ImageAnalyzer")
        self._theme = self._settings.value("theme", "Dark", type=str)
        self._icon_actions: dict[QtGui.QAction, str] = {}
        self._icon_buttons: dict[QtWidgets.QToolButton, str] = {}
        self._icon_color = "#d4d4d8" if self._theme == "Dark" else "#33333a"
        self.setStyleSheet(build_style(self._theme))

        # Central area: VSCode-style document tabs, one per open file/frame.
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.setDocumentMode(True)
        self._tabs.tabCloseRequested.connect(self._on_tab_close)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._tabs.tabBarDoubleClicked.connect(
            lambda i: self._pin_tab(self._tabs.widget(i))
        )
        self.setCentralWidget(self._tabs)
        self._docs: dict[QtWidgets.QWidget, dict] = {}
        self._current_tool = "select"
        self._cmap_syncing = False
        self.apply_theme(self._theme)

        self._build_file_dock()
        self._build_control_dock()
        self._build_analysis_dock()
        self._build_ribbon()
        self._build_statusbar()

        # Drag & drop of files/folders anywhere on the window.
        self.setAcceptDrops(True)
        self._overlay = DropOverlay(self)

        self._restore_session()

    # ------------------------------------------------------ drag & drop
    def _default_settings(self) -> dict:
        """Return initial view settings, inheriting current active control values."""
        return {
            "log": getattr(self, "_log_chk", None) and self._log_chk.isChecked() or False,
            "overmax": getattr(self, "_overmax_chk", None) and self._overmax_chk.isChecked() or True,
            "colormap": getattr(self, "_cmap_combo", None) and self._cmap_combo.currentText() or self._settings.value("def_cmap", "gray", type=str),
            "scale_on": getattr(self, "_scale_chk", None) and self._scale_chk.isChecked() or False,
            "px_size": getattr(self, "_px_size_spin", None) and self._px_size_spin.value() or 1.0,
            "unit": getattr(self, "_unit_combo", None) and self._unit_combo.currentText() or self._settings.value("def_unit", "µm", type=str),
            "tool": getattr(self, "_current_tool", "select"),
        }

    def _urls_have_supported(self, urls) -> bool:
        for u in urls:
            p = Path(u.toLocalFile())
            if p.is_dir() or p.suffix.lower() in self._SUPPORTED:
                return True
        return False

    def _gather_supported(self, paths: list[Path]) -> list[Path]:
        files: list[Path] = []
        for p in paths:
            if p.is_dir():
                files += sorted(
                    f for f in p.rglob("*")
                    if f.is_file() and f.suffix.lower() in self._SUPPORTED
                )
            elif p.is_file() and p.suffix.lower() in self._SUPPORTED:
                files.append(p)
        return files

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            self._overlay.set_ok(self._urls_have_supported(e.mimeData().urls()))
            self._overlay.cover()
            e.acceptProposedAction()

    def dragMoveEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragLeaveEvent(self, e) -> None:
        self._overlay.hide()

    def dropEvent(self, e) -> None:
        self._overlay.hide()
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls()]
        files = self._gather_supported(paths)
        if not files:
            self._status.showMessage("지원하지 않는 파일 형식입니다.", 4000)
            return
        # A dropped folder becomes the file-tree root for convenient browsing.
        for p in paths:
            if p.is_dir():
                self._fs_model.setRootPath(str(p))
                self._tree.setRootIndex(self._fs_model.index(str(p)))
                break
        for f in files:
            self._open_document(f, preview=False)
        self._status.showMessage(f"{len(files)}개 파일 열림.", 3000)
        e.acceptProposedAction()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self._overlay.isVisible():
            self._overlay.setGeometry(self.rect())

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self, self._settings)
        dlg.exec()
        vals = dlg.result_values()
        for k, v in vals.items():
            self._settings.setValue(k, v)
        # Apply display defaults immediately to the panel.
        self._cmap_combo.setCurrentText(vals["def_cmap"])
        self._status.showMessage("설정 저장됨.", 3000)

    # ------------------------------------------------ active-document access
    def _cur(self) -> DocumentSession | None:
        return self._docs.get(self._tabs.currentWidget())

    def _cur_view(self) -> ImageView | None:
        doc = self._cur()
        if doc and doc.kind == "image":
            return doc.view
        return None

    # ---------------------------------------------------------- theme + icons

    def _set_action_icon(self, action: QtGui.QAction, icon_name: str) -> None:
        self._icon_actions[action] = icon_name
        action.setIcon(AppIcons.get(icon_name, self._icon_color))

    def apply_theme(self, theme: str) -> None:
        self._theme = theme
        self._icon_color = "#d4d4d8" if theme == "Dark" else "#33333a"
        self.setStyleSheet(build_style(theme))
        
        # 등록된 Action 및 Button들 테마 색상 재적용
        for act, name in self._icon_actions.items():
            act.setIcon(AppIcons.get(name, self._icon_color))
        for btn, name in self._icon_buttons.items():
            btn.setIcon(AppIcons.get(name, self._icon_color))
            
        dark = theme == "Dark"
        for doc in self._docs.values():
            if doc.kind == "image" and doc.view is not None:
                doc.view.apply_theme(dark)
        if hasattr(self, "_profile_plot"):
            self._profile_plot.setBackground("#202225" if dark else "#f4f4f6")
            for ax in ("bottom", "left"):
                a = self._profile_plot.getAxis(ax)
                a.setPen(pg.mkPen("#666" if dark else "#9a9aa2"))
                a.setTextPen(pg.mkPen("#bbb" if dark else "#333"))
        self._settings.setValue("theme", theme)

    # Dispatch wrappers act on the ACTIVE tab's view AND update its session.
    def _do_reset(self) -> None:
        if v := self._cur_view():
            v.reset_zoom()

    def _do_log(self, b) -> None:
        if (v := self._cur_view()) and (s := self._cur_settings()) is not None:
            v.set_log(b)
            s["log"] = b
            self._settings.setValue("persistent_log", b)
            self._log(f"Log scale {'ON' if b else 'OFF'}")

    def _do_overmax(self, b) -> None:
        if (v := self._cur_view()) and (s := self._cur_settings()) is not None:
            v.set_overmax(b)
            s["overmax"] = b
            self._settings.setValue("persistent_overmax", b)

    def _do_colormap(self, name) -> None:
        if (v := self._cur_view()) and (s := self._cur_settings()) is not None:
            v.set_colormap(name)
            s["colormap"] = name
            self._settings.setValue("def_cmap", name)
        if not self._cmap_syncing and hasattr(self, "_ribbon_cmap"):
            self._cmap_syncing = True
            self._ribbon_cmap.setCurrentText(name)
            self._cmap_syncing = False

    def _on_ribbon_cmap(self, name) -> None:
        if self._cmap_syncing:
            return
        self._cmap_syncing = True
        self._cmap_combo.setCurrentText(name)  # drives _do_colormap
        self._cmap_syncing = False

    def _do_zoom100(self) -> None:
        if v := self._cur_view():
            v.set_zoom(100.0)

    def _do_grid(self, on) -> None:
        if v := self._cur_view():
            v.set_grid(on)

    def _do_autoscale(self) -> None:
        if v := self._cur_view():
            v.autoscale_levels()
            self._log("Auto 레벨")

    def _do_clear_distance(self) -> None:
        if v := self._cur_view():
            v.clear_distance()

    def _do_fit(self) -> None:
        v = self._cur_view()
        doc = self._cur()
        if v is None or doc is None:
            return
        geom = v.fit_ellipse()  # draws the static preview + returns geometry
        if geom is None:
            return
        pts = v.picked_points()
        v.clear_fit()  # replace the static preview with a managed ROI object
        name = self._next_roi_name(doc, "ellipse", "Ellipse")
        self.add_roi(EllipseFitROI(geom, points=pts, name=name))

    # ------------------------------------------------------------- ROI (B-1)
    @staticmethod
    def _next_roi_name(doc: DocumentSession, roi_type: str, prefix: str) -> str:
        return f"{prefix}_{len(doc.rois_of_type(roi_type)) + 1}"

    def add_roi(self, roi) -> None:
        """Add an ROI to the active canvas and register it in the session."""
        doc = self._cur()
        v = self._cur_view()
        if doc is None or v is None:
            return
        doc.add_roi(roi)  # register BEFORE canvas add (rebuild reads the session)
        v.add_roi_item(roi)
        doc.add_log(f"ROI 추가: {roi.name} [{roi.roi_id}]")
        self._refresh_log()

    def remove_roi(self, roi_id: str) -> None:
        # Works for any open document (multi-root pipeline), not just the active.
        owned = self._roi_owner_doc(roi_id) if hasattr(self, "_roi_owner") else None
        if owned is not None:
            _, doc, view, _ = owned
        else:
            doc, view = self._cur(), self._cur_view()
        if doc is None:
            return
        roi = doc.remove_roi(roi_id)
        if roi is not None and view is not None:
            view.remove_roi_item(roi)
            doc.add_log(f"ROI 삭제: {roi.name} [{roi_id}]")
            self._refresh_log()

    def add_line_roi(self, positions=None) -> LineProfileROI:
        """Create + register a line-profile ROI (used by later phases/tools)."""
        doc = self._cur()
        v = self._cur_view()
        if positions is None and v is not None and v.has_image():
            h, w = v.image_shape()
            positions = [[w * 0.2, h * 0.5], [w * 0.8, h * 0.5]]
        roi = LineProfileROI(positions, name=self._next_roi_name(doc, "line", "Line"))
        self.add_roi(roi)
        return roi

    def _on_line_drawn(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Two-click line finished -> spawn a line-profile ROI."""
        doc = self._cur()
        v = self._cur_view()
        if doc is None or v is None:
            return
        roi = LineProfileROI(
            [[x0, y0], [x1, y1]], name=self._next_roi_name(doc, "line", "Line")
        )
        self.add_roi(roi)
        v.set_active_line(roi)
        v.select_roi(roi)
        self._analysis_dock.show()

    def _on_rect_drawn(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Drag-drawn rectangle finished -> spawn a region ROI."""
        doc = self._cur()
        v = self._cur_view()
        if doc is None or v is None:
            return
        px, py = min(x0, x1), min(y0, y1)
        sw, sh = abs(x1 - x0), abs(y1 - y0)
        roi = RectRegionROI(
            [px, py], [sw, sh], name=self._next_roi_name(doc, "rect", "Region")
        )
        self.add_roi(roi)
        v.select_roi(roi)

    # =============================================== Pipeline object tree (B-3)
    def _build_object_tree(self) -> QtWidgets.QTreeWidget:
        self._obj_updating = False
        self._roi_items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self._roi_owner: dict[str, QtWidgets.QWidget] = {}
        tree = QtWidgets.QTreeWidget()
        self._obj_tree = tree
        tree.setColumnCount(3)
        tree.setHeaderLabels(["오브젝트", "👁", "🔒"])
        hdr = tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        tree.setColumnWidth(1, 30)
        tree.setColumnWidth(2, 30)
        tree.setRootIsDecorated(True)

        tree.itemChanged.connect(self._on_obj_item_changed)
        tree.itemClicked.connect(self._on_obj_item_clicked)
        dele = QtGui.QShortcut(QtGui.QKeySequence.Delete, tree)
        dele.setContext(QtCore.Qt.WidgetShortcut)
        dele.activated.connect(self._on_obj_delete)
        return tree

    @staticmethod
    def _set_eye_lock(item, visible: bool, locked: bool) -> None:
        item.setText(1, "👁" if visible else "—")
        item.setForeground(
            1, QtGui.QBrush(QtGui.QColor("#4ade80" if visible else "#777"))
        )
        item.setTextAlignment(1, QtCore.Qt.AlignCenter)
        item.setText(2, "🔒" if locked else "")
        item.setTextAlignment(2, QtCore.Qt.AlignCenter)

    @staticmethod
    def _roi_icon(roi_type: str) -> QtGui.QIcon:
        pm = QtGui.QPixmap(14, 14)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        if roi_type == "ellipse":
            p.setPen(QtGui.QPen(QtGui.QColor("#ff375f"), 1.6))
            p.drawEllipse(1, 3, 12, 8)
        elif roi_type == "rect":
            p.setPen(QtGui.QPen(QtGui.QColor("#4ade80"), 1.6))
            p.drawRect(2, 3, 10, 8)
        else:  # line
            p.setPen(QtGui.QPen(QtGui.QColor("#facc15"), 2))
            p.drawLine(2, 11, 12, 3)
        p.end()
        return QtGui.QIcon(pm)

    _ROLE_ROI = QtCore.Qt.UserRole
    _ROLE_DOC = QtCore.Qt.UserRole + 1

    def _add_obj_item(self, roi, root, widget) -> None:
        item = QtWidgets.QTreeWidgetItem([roi.name, "", ""])
        item.setIcon(0, self._roi_icon(roi.roi_type))
        item.setData(0, self._ROLE_ROI, roi.roi_id)
        item.setFlags(
            QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
            | QtCore.Qt.ItemIsEditable
        )
        self._set_eye_lock(item, roi.visible, roi.locked)
        root.addChild(item)
        self._roi_items[roi.roi_id] = item
        self._roi_owner[roi.roi_id] = widget

    def _rebuild_object_tree(self) -> None:
        # ParaView-style: EVERY open image is a source (root); ROIs are children.
        self._obj_updating = True
        self._obj_tree.clear()
        self._roi_items = {}
        self._roi_owner = {}
        active = self._tabs.currentWidget()
        bold = QtGui.QFont()
        bold.setBold(True)
        for i in range(self._tabs.count()):
            widget = self._tabs.widget(i)
            doc = self._docs.get(widget)
            if doc is None or doc.kind != "image":
                continue
            name = doc.file_path.name
            if doc.frame is not None:
                name += f"  ·  {doc.frame}"
            root = QtWidgets.QTreeWidgetItem([f"🖼  {name}", ""])
            root.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            root.setData(0, self._ROLE_DOC, widget)
            if widget is active:
                root.setFont(0, bold)
            self._obj_tree.addTopLevelItem(root)
            for roi in doc.roi_objects.values():
                self._add_obj_item(roi, root, widget)
            root.setExpanded(True)
        self._obj_updating = False

    def _on_roi_added(self, roi) -> None:
        self._rebuild_object_tree()

    def _on_roi_removed(self, roi_id: str) -> None:
        self._rebuild_object_tree()

    def _roi_owner_doc(self, roi_id: str):
        """(widget, doc, view, roi) for a roi_id across all open documents."""
        widget = self._roi_owner.get(roi_id)
        doc = self._docs.get(widget) if widget else None
        if doc is None:
            return None
        roi = doc.get_roi(roi_id)
        if roi is None:
            return None
        return widget, doc, doc.view, roi

    def _on_obj_item_changed(self, item, col) -> None:
        if self._obj_updating or col != 0:
            return
        rid = item.data(0, self._ROLE_ROI)
        if rid is None:
            return
        owned = self._roi_owner_doc(rid)
        if owned is None:
            return
        _, _, _, roi = owned
        name = item.text(0).strip()
        if name and name != roi.name:
            roi.rename(name)

    def _on_obj_item_clicked(self, item, col) -> None:
        rid = item.data(0, self._ROLE_ROI)
        if rid is None:
            # Root (image) item -> switch to that document tab.
            widget = item.data(0, self._ROLE_DOC)
            if widget is not None:
                self._tabs.setCurrentWidget(widget)
            return
        owned = self._roi_owner_doc(rid)
        if owned is None:
            return
        widget, _, view, roi = owned
        if col == 1:  # 👁 toggle visibility
            roi.set_visible(not roi.visible)
            self._obj_updating = True
            self._set_eye_lock(item, roi.visible, roi.locked)
            self._obj_updating = False
            return
        if col == 2:  # 🔒 toggle lock
            roi.set_locked(not roi.locked)
            self._obj_updating = True
            self._set_eye_lock(item, roi.visible, roi.locked)
            self._obj_updating = False
            return
        # col 0 = focus/select: switch to the owning tab first, then highlight.
        if widget is not self._tabs.currentWidget():
            self._tabs.setCurrentWidget(widget)
        view.select_roi(roi)

    def _on_obj_delete(self) -> None:
        item = self._obj_tree.currentItem()
        if item is None:
            return
        rid = item.data(0, self._ROLE_ROI)
        if rid:
            self.remove_roi(rid)

    def _delete_selected_roi(self) -> None:
        """Delete key: remove the ROI selected on the active canvas."""
        v = self._cur_view()
        if v is None:
            return
        roi = v.selected_roi()
        if roi is not None:
            self.remove_roi(roi.roi_id)

    def _on_roi_selected(self, roi) -> None:
        self._obj_updating = True
        if roi is None:
            self._obj_tree.setCurrentItem(None)
        else:
            it = self._roi_items.get(roi.roi_id)
            if it is not None:
                self._obj_tree.setCurrentItem(it)
        self._obj_updating = False
        if roi is not None and roi.roi_type == "line":
            self._analysis_dock.show()

    def _do_hist_visible(self, b) -> None:
        if v := self._cur_view():
            v.set_histogram_visible(b)

    def _do_hist_detach(self) -> None:
        if v := self._cur_view():
            v.detach_histogram()
            with QtCore.QSignalBlocker(self._hist_chk):
                self._hist_chk.setChecked(False)
                self._hist_chk.setEnabled(False)

    # ============================================================ ribbon UI
    def _act(self, text: str, slot, tip: str = "", checkable: bool = False):
        a = QtGui.QAction(text, self, checkable=checkable)
        if tip:
            a.setToolTip(tip)
        if checkable:
            a.triggered.connect(lambda _=False: slot())
        else:
            a.triggered.connect(slot)
        return a

    @staticmethod
    def _rbtn(action, big: bool = True) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setDefaultAction(action)
        b.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        b.setAutoRaise(True)
        b.setProperty("class", "RibbonBtn")
        b.setIconSize(QtCore.QSize(26, 26))
        b.setMinimumWidth(62)
        b.setMinimumHeight(62)  # uniform height for 1- or 2-line labels
        return b

    def _ribbon_group(self, title: str, inner: QtWidgets.QLayout) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        box.setObjectName("RibbonGroupBox")
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(4, 4, 4, 2)
        v.setSpacing(3)
        holder = QtWidgets.QWidget()
        holder.setObjectName("RibbonHolder")
        holder.setLayout(inner)
        v.addWidget(holder, 1)
        cap = QtWidgets.QLabel(title)
        cap.setObjectName("RibbonCap")
        cap.setAlignment(QtCore.Qt.AlignCenter)
        cap.setStyleSheet("font-size:8pt; border:none;")
        v.addWidget(cap)
        return box

    @staticmethod
    def _vdiv() -> QtWidgets.QWidget:
        f = QtWidgets.QFrame()
        f.setObjectName("RibbonDiv")
        f.setFixedWidth(1)
        f.setContentsMargins(0, 8, 0, 8)
        return f

    def _ribbon_page(self, groups: list[QtWidgets.QWidget]) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page.setObjectName("RibbonPage")
        h = QtWidgets.QHBoxLayout(page)
        h.setContentsMargins(6, 3, 6, 3)
        h.setSpacing(2)
        for i, g in enumerate(groups):
            if i:
                h.addWidget(self._vdiv())
            h.addWidget(g)
        h.addStretch(1)
        return page

    def _build_ribbon(self) -> None:
        # Actions (reused across ribbon tabs).
        self._act_folder = self._act("📁\n폴더", self._open_folder, "폴더 열기")
        self._act_file = self._act("📂\n파일", self._open_file, "파일 열기")
        self._act_shot = self._act("📷\n스크린샷", self._screenshot, "PNG 저장 +클립보드")
        self._act_reset = self._act("⤢\n기본 줌", self._do_reset, "전체 보기 복귀")
        self._act_clear = self._act("🧹\n초기화", self._on_clear_fit, "측정/피팅 초기화")
        self._act_autolevel = self._act("📊\nAuto", self._do_autoscale, "Auto 레벨 0.5–99.5%")
        self._act_fit = self._act("화면 맞춤", self._do_reset, "Fit to Window")

        # Tool actions (exclusive).
        self._tool_group = QtGui.QActionGroup(self)
        self._tool_group.setExclusive(True)
        tools = [
            ("select", "⬉\n선택", "선택/이동: ROI 클릭 선택·드래그, ESC로 복귀"),
            ("pan", "✋\n이동", "손바닥: 좌드래그 이동"),
            ("zoom", "🔍\n확대", "박스 드래그 / 스크롤 줌"),
            ("distance", "📏\n거리", "점 찍어 구간 거리 (링 간격)"),
            ("line", "〰\n라인", "빈 곳 클릭 → 라인 생성, 드래그로 프로파일"),
            ("ellipse", "⬭\n타원 피팅", "점 찍어 타원 피팅 (우측 패널에서 Fit)"),
            ("rect", "▭\n구역", "빈 곳 클릭 → 사각 ROI 구역 생성"),
        ]
        self._tool_actions: dict[str, QtGui.QAction] = {}
        for name, label, tip in tools:
            act = QtGui.QAction(label, self, checkable=True)
            act.setToolTip(tip)
            act.triggered.connect(lambda _=False, n=name: self._set_tool(n))
            self._tool_group.addAction(act)
            self._tool_actions[name] = act
        self._tool_actions["select"].setChecked(True)

        # PNG icons (theme-tinted)
        self._act_folder.setText("폴더")
        self._set_action_icon(self._act_folder, AppIcons.FOLDER)
        self._act_shot.setText("스크린샷")
        self._set_action_icon(self._act_shot, AppIcons.CAMERA)
        self._act_reset.setText("기본 줌")
        self._set_action_icon(self._act_reset, AppIcons.DEFAULT_ZOOM)
        self._set_action_icon(self._act_fit, AppIcons.DEFAULT_ZOOM)

        for tname, (lbl, icn) in {
            "pan": ("이동", AppIcons.HAND), "zoom": ("확대", AppIcons.ZOOM_IN),
            "distance": ("거리", AppIcons.RULER), "line": ("라인", AppIcons.LINE),
            "ellipse": ("타원 피팅", AppIcons.OVAL),
        }.items():
            self._tool_actions[tname].setText(lbl)
            self._set_action_icon(self._tool_actions[tname], icn)

        # ESC always returns to Select mode
        esc = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        esc.activated.connect(lambda: self._tool_actions["select"].trigger())
        # Delete removes selected ROI
        dsc = QtGui.QShortcut(QtGui.QKeySequence.Delete, self)
        dsc.activated.connect(self._delete_selected_roi)

        # Scan / Det selectors.
        self._scan_combo = QtWidgets.QComboBox()
        self._scan_combo.setMinimumWidth(120)
        self._scan_combo.currentIndexChanged.connect(self._on_frame_selected)
        self._det_combo = QtWidgets.QComboBox()
        self._det_combo.setMinimumWidth(120)
        self._det_combo.currentIndexChanged.connect(self._on_frame_selected)

        # Panel-visibility toggles
        tog_file = self._file_dock.toggleViewAction()
        tog_file.setText("📂\n좌측 패널")
        tog_ctrl = self._control_dock.toggleViewAction()
        tog_ctrl.setText("🎛\n우측 속성창")
        tog_ana = self._analysis_dock.toggleViewAction()
        tog_ana.setText("📉\n하단 프로파일")

        # View-control actions.
        self._act_zoom100 = self._act("💯\n100%", self._do_zoom100, "100% 확대")
        self._grid_chk_act = QtGui.QAction("▦\nGrid", self, checkable=True)
        self._grid_chk_act.setToolTip("눈금(Grid) 표시 On/Off")
        self._grid_chk_act.toggled.connect(self._do_grid)

        # Ribbon quick colormap
        self._ribbon_cmap = QtWidgets.QComboBox()
        self._ribbon_cmap.setIconSize(QtCore.QSize(72, 12))
        for name in COLORMAPS:
            self._ribbon_cmap.addItem(self._cmap_icon(name, 72, 12), name)
        self._cmap_syncing = False
        self._ribbon_cmap.currentTextChanged.connect(self._on_ribbon_cmap)

        # --- Ribbon tabs ---
        ribbon = QtWidgets.QTabWidget()
        ribbon.setObjectName("Ribbon")
        ribbon.setFixedHeight(140)
        ribbon.setDocumentMode(True)

        # Home
        g_file = self._ribbon_group("파일 관리", self._row(
            self._rbtn(self._act_folder), self._rbtn(self._act_file),
            self._rbtn(self._act_shot)))
        g_frame = self._ribbon_group("프레임", self._frame_row())
        g_edit = self._ribbon_group("편집", self._row(
            self._rbtn(self._act_reset), self._rbtn(self._act_clear)))
        ribbon.addTab(self._ribbon_page([g_file, g_frame, g_edit]), "홈")

        # Analyze
        g_meas = self._ribbon_group("측정 / 피팅 도구", self._row(
            self._rbtn(self._tool_actions["distance"]),
            self._rbtn(self._tool_actions["line"]),
            self._rbtn(self._tool_actions["ellipse"]),
            self._rbtn(self._tool_actions["rect"])))
        ribbon.addTab(self._ribbon_page([g_meas]), "분석")

        # View
        g_panels = self._ribbon_group("패널 표시", self._row(
            self._toggle_btn(tog_file), self._toggle_btn(tog_ctrl),
            self._toggle_btn(tog_ana)))
        cmap_holder = QtWidgets.QVBoxLayout()
        cmap_holder.setContentsMargins(0, 0, 0, 0)
        cmap_holder.addWidget(self._ribbon_cmap)
        cmap_holder.addWidget(QtWidgets.QLabel("Colormap", alignment=QtCore.Qt.AlignCenter))
        disp_row = self._row(self._rbtn(self._act_autolevel),
                             self._toggle_btn(self._grid_chk_act))
        disp_row.addLayout(cmap_holder)
        g_disp = self._ribbon_group("디스플레이", disp_row)
        g_viewport = self._ribbon_group("뷰 포트", self._row(
            self._rbtn(self._act_fit), self._rbtn(self._act_zoom100)))
        ribbon.addTab(self._ribbon_page([g_panels, g_disp, g_viewport]), "보기 (View)")

        # Tools
        g_tools = self._ribbon_group("탐색 / 선택", self._row(
            self._rbtn(self._tool_actions["select"]),
            self._rbtn(self._tool_actions["pan"]),
            self._rbtn(self._tool_actions["zoom"])))
        ribbon.addTab(self._ribbon_page([g_tools]), "도구")

        # Quick-access (top-left)
        qa = QtWidgets.QWidget()
        qh = QtWidgets.QHBoxLayout(qa)
        qh.setContentsMargins(6, 0, 6, 0)
        qh.setSpacing(1)
        brand = QtWidgets.QLabel()
        brand.setPixmap(logo_pixmap(22))
        brand.setToolTip(APP_NAME)
        qh.addWidget(brand)
        qh.addWidget(self._qsep())
        
        qh.addWidget(self._quick_btn(AppIcons.FOLDER, self._open_file, "파일 열기"))
        qh.addWidget(self._quick_btn(AppIcons.CAMERA, self._screenshot, "스크린샷"))
        qh.addWidget(self._qsep())
        qh.addWidget(self._quick_btn(AppIcons.PAN, lambda: self._set_tool("pan"), "이동 도구"))
        qh.addWidget(self._quick_btn(
            AppIcons.ZOOM_OUT, lambda: self._cur_view() and self._cur_view().zoom_out(), "축소"))
        qh.addWidget(self._quick_btn(
            AppIcons.ZOOM_IN, lambda: self._cur_view() and self._cur_view().zoom_in(), "확대"))
        qh.addWidget(self._quick_btn(AppIcons.DEFAULT_ZOOM, self._do_reset, "화면에 맞춤"))

        glob = QtWidgets.QWidget()
        gh = QtWidgets.QHBoxLayout(glob)
        gh.setContentsMargins(6, 0, 6, 0)
        gh.setSpacing(4)
        gh.addWidget(QtWidgets.QLabel("🌐"))
        self._lang_combo = QtWidgets.QComboBox()
        self._lang_combo.addItems(["KO", "EN"])
        self._lang_combo.setCurrentText(self._settings.value("lang_short", "KO", type=str))
        self._lang_combo.currentTextChanged.connect(
            lambda s: self._settings.setValue("lang_short", s))
        gh.addWidget(self._lang_combo)
        act_set = self._mini(self._act("⚙", self._open_settings, "환경설정"))
        gh.addWidget(act_set)
        ribbon.setCornerWidget(glob, QtCore.Qt.TopRightCorner)

        self.setMenuWidget(ribbon)

    @staticmethod
    def _row(*widgets) -> QtWidgets.QHBoxLayout:
        h = QtWidgets.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(2)
        for wd in widgets:
            h.addWidget(wd)
        return h

    def _frame_row(self) -> QtWidgets.QVBoxLayout:
        v = QtWidgets.QVBoxLayout()
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        r1 = QtWidgets.QHBoxLayout()
        r1.addWidget(QtWidgets.QLabel("Scan"))
        r1.addWidget(self._scan_combo)
        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(QtWidgets.QLabel(" Det"))
        r2.addWidget(self._det_combo)
        v.addLayout(r1)
        v.addLayout(r2)
        return v

    def _toggle_btn(self, action) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setDefaultAction(action)
        b.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        b.setCheckable(True)
        b.setAutoRaise(True)
        b.setProperty("class", "RibbonBtn")
        b.setMinimumWidth(62)
        b.setMinimumHeight(62)
        return b

    def _mini(self, action) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setDefaultAction(action)
        b.setAutoRaise(True)
        return b

    def _quick_btn(self, icon_name: str, slot, tip: str) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.clicked.connect(slot)
        self._icon_buttons[b] = icon_name
        b.setIcon(AppIcons.get(icon_name, self._icon_color))
        b.setIconSize(QtCore.QSize(18, 18))
        return b

    @staticmethod
    def _qsep() -> QtWidgets.QFrame:
        f = QtWidgets.QFrame()
        f.setObjectName("QSep")
        f.setFixedWidth(1)
        return f

    _TOOL_PAGE = {
        "select": ("view", "선택 / 이동"),
        "pan": ("view", "이동 / 확대"),
        "zoom": ("view", "이동 / 확대"),
        "distance": ("distance", "거리 측정"),
        "line": ("line", "라인 프로파일"),
        "ellipse": ("ellipse", "타원 피팅 (Ellipse Fit)"),
        "rect": ("view", "사각 ROI 구역"),
    }

    def _set_tool(self, name: str) -> None:
        self._current_tool = name
        if v := self._cur_view():
            v.set_tool(name)
        if (s := self._cur_settings()) is not None and s.get("tool") != name:
            s["tool"] = name
            self._log(f"도구: {name}")
        page_key, title = self._TOOL_PAGE.get(name, ("view", "도구 옵션"))
        self._tool_stack.setCurrentWidget(self._stack_pages[page_key])
        self._tool_title.setText(title)
        # The bottom profile plot only makes sense for the line tool.
        if name == "line":
            self._analysis_dock.show()
        else:
            self._analysis_dock.hide()

    # --------------------------------------------------------- file dock
    def _build_file_dock(self) -> None:
        dock = self._file_dock = QtWidgets.QDockWidget("탐색기 (Explorer)", self)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        tabs = QtWidgets.QTabWidget()

        self._fs_model = QtWidgets.QFileSystemModel()
        self._fs_model.setNameFilters(
            ["*.h5", "*.hdf5", "*.tif", "*.tiff", "*.png", "*.jpg",
             "*.jpeg", "*.bmp", "*.json", "*.txt"]
        )
        self._fs_model.setNameFilterDisables(False)
        self._fs_model.setRootPath("")

        self._tree = QtWidgets.QTreeView()
        self._tree.setModel(self._fs_model)
        self._tree.setRootIndex(self._fs_model.index(""))
        for col in (1, 2, 3):  # hide size / type / date
            self._tree.hideColumn(col)
        self._tree.setHeaderHidden(True)
        self._tree.clicked.connect(self._on_tree_preview)
        self._tree.doubleClicked.connect(self._on_tree_open)

        self._tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_context_menu)

        tabs.addTab(self._tree, "파일")

        # HDF5 structure viewer
        self._struct_tree = QtWidgets.QTreeWidget()
        self._struct_tree.setColumnCount(4)
        self._struct_tree.setHeaderLabels(["이름", "종류", "shape", "dtype"])
        self._struct_tree.setAlternatingRowColors(True)
        self._struct_tree.itemDoubleClicked.connect(self._on_struct_activated)
        tabs.addTab(self._struct_tree, "구조")

        tabs.addTab(self._build_object_tree(), "오브젝트")

        dock.setWidget(tabs)
        dock.setMinimumWidth(280)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

    # -------------------------------------------------- 파일 트리 컨텍스트 메뉴
    def _on_tree_context_menu(self, pos: QtCore.QPoint) -> None:
        index = self._tree.indexAt(pos)
        if not index.isValid():
            return

        file_path_str = self._fs_model.filePath(index)
        path = Path(file_path_str)

        menu = QtWidgets.QMenu(self)

        # 1. 파일 열기 (파일일 경우만)
        if path.is_file():
            icn_file = AppIcons.get(AppIcons.FILE, self._icon_color)
            act_open = menu.addAction(icn_file, "파일 열기")
            act_open.triggered.connect(lambda: self._open_document(path, preview=False))
            menu.addSeparator()

        # 2. 파일 탐색기에서 열기
        icn_folder = AppIcons.get(AppIcons.FOLDER, self._icon_color)
        act_explorer = menu.addAction(icn_folder, "파일 탐색기에서 열기")
        act_explorer.triggered.connect(lambda: self._show_in_file_manager(path))

        # 3. 경로 복사
        icn_copy = AppIcons.get(AppIcons.COPY, self._icon_color)
        act_copy_path = menu.addAction(icn_copy, "경로 복사")
        act_copy_path.triggered.connect(lambda: self._copy_path_to_clipboard(path))

        # 메뉴 출력
        menu.exec_(self._tree.viewport().mapToGlobal(pos))

    def _show_in_file_manager(self, path: Path) -> None:
        """시스템 파일 탐색기(Windows Explorer 등)에서 해당 파일/폴더를 엽니다."""
        abs_path = path.resolve()
        if abs_path.is_file():
            # Windows의 경우 파일 선택 상태로 탐색기 열기 지원
            import platform
            if platform.system() == "Windows":
                import subprocess
                subprocess.run(["explorer", "/select,", str(abs_path)])
                return
            target_url = QtCore.QUrl.fromLocalFile(str(abs_path.parent))
        else:
            target_url = QtCore.QUrl.fromLocalFile(str(abs_path))

        QtGui.QDesktopServices.openUrl(target_url)

    def _copy_path_to_clipboard(self, path: Path) -> None:
        """클립보드에 파일/폴더 절대 경로를 복사합니다."""
        abs_path = str(path.resolve())
        QtWidgets.QApplication.clipboard().setText(abs_path)
        self._status.showMessage(f"경로 복사됨: {abs_path}", 3000)

    # ------------------------------------------------------ control dock
    def _build_control_dock(self) -> None:
        dock = self._control_dock = QtWidgets.QDockWidget("컨트롤", self)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setSpacing(4)

        # --- Display section (collapsible)
        disp = QtWidgets.QWidget()
        dl = QtWidgets.QFormLayout(disp)
        dl.setContentsMargins(8, 4, 8, 4)

        self._log_chk = QtWidgets.QCheckBox("Log scale")
        self._log_chk.toggled.connect(self._do_log)
        dl.addRow(self._log_chk)

        self._overmax_chk = QtWidgets.QCheckBox("Over-max Highlight")
        self._overmax_chk.setChecked(True)
        self._overmax_chk.toggled.connect(self._do_overmax)

        self._overmax_color_btn = QtWidgets.QPushButton()
        self._overmax_color_btn.setFixedWidth(28)
        self._overmax_color_btn.setFixedHeight(20)
        self._overmax_color = QtGui.QColor(self._default_overmax_color)
        self._update_overmax_btn_style()
        self._overmax_color_btn.clicked.connect(self._choose_overmax_color)

        overmax_row = QtWidgets.QHBoxLayout()
        overmax_row.setContentsMargins(0, 0, 0, 0)
        overmax_row.addWidget(self._overmax_chk)
        overmax_row.addWidget(self._overmax_color_btn)
        dl.addRow(overmax_row)

        self._cmap_combo = QtWidgets.QComboBox()
        self._cmap_combo.setIconSize(QtCore.QSize(96, 14))
        for name in COLORMAPS:
            self._cmap_combo.addItem(self._cmap_icon(name), name)
        self._cmap_combo.currentTextChanged.connect(self._do_colormap)
        dl.addRow("Colormap", self._cmap_combo)

        btn_auto = QtWidgets.QPushButton("Auto 레벨 (0.5–99.5%)")
        btn_auto.clicked.connect(self._do_autoscale)
        dl.addRow(btn_auto)

        # Histogram
        self._hist_chk = QtWidgets.QCheckBox("히스토그램 표시")
        self._hist_chk.setChecked(True)
        self._hist_chk.toggled.connect(self._do_hist_visible)
        hist_row = QtWidgets.QHBoxLayout()
        hist_row.setContentsMargins(0, 0, 0, 0)
        hist_row.addWidget(self._hist_chk)
        btn_detach = QtWidgets.QPushButton("창 분리")
        btn_detach.setToolTip("히스토그램을 별도 창으로 분리")
        btn_detach.clicked.connect(self._do_hist_detach)
        hist_row.addWidget(btn_detach)
        dl.addRow(hist_row)
        v.addWidget(CollapsibleSection("디스플레이", disp))

        # --- Scale section (collapsible)
        scale = QtWidgets.QWidget()
        sl = QtWidgets.QFormLayout(scale)
        sl.setContentsMargins(8, 4, 8, 4)

        self._scale_chk = QtWidgets.QCheckBox("실거리 사용 (기본: pixel)")
        self._scale_chk.toggled.connect(self._on_scale_changed)
        sl.addRow(self._scale_chk)

        self._px_size_spin = QtWidgets.QDoubleSpinBox()
        self._px_size_spin.setDecimals(6)
        self._px_size_spin.setRange(1e-6, 1e6)
        self._px_size_spin.setValue(1.0)
        self._px_size_spin.valueChanged.connect(self._on_scale_changed)
        sl.addRow("픽셀당 거리", self._px_size_spin)

        self._unit_combo = QtWidgets.QComboBox()
        self._unit_combo.setEditable(True)
        self._unit_combo.addItems(["µm", "nm", "mm", "Å"])
        self._unit_combo.currentTextChanged.connect(self._on_scale_changed)
        sl.addRow("단위", self._unit_combo)
        v.addWidget(CollapsibleSection("척도 (Scale bar)", scale))

        # --- Tool-specific area
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet("color:#3a3a3a;")
        v.addWidget(line)
        self._tool_title = QtWidgets.QLabel("도구 옵션")
        self._tool_title.setStyleSheet(
            "color:#2dd4bf; font-weight:bold; font-size:11pt; padding:2px 4px;"
        )
        v.addWidget(self._tool_title)

        self._tool_stack = QtWidgets.QStackedWidget()
        self._stack_pages = {
            "view": self._build_view_page(),
            "distance": self._build_distance_page(),
            "line": self._build_line_page(),
            "ellipse": self._build_ellipse_page(),
        }
        for page in self._stack_pages.values():
            self._tool_stack.addWidget(page)
        v.addWidget(self._tool_stack, 1)

        dock.setWidget(panel)
        dock.setMinimumWidth(330)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    # ----- overmax color -----
    def _update_overmax_btn_style(self) -> None:
        self._overmax_color_btn.setStyleSheet(
            f"background-color: {self._overmax_color.name()}; border: 1px solid #555; border-radius: 3px;"
        )

    def _choose_overmax_color(self) -> None:
        color = QtWidgets.QColorDialog.getColor(
            self._overmax_color, self, "Over-max Highlight Color"
        )
        if color.isValid():
            self._overmax_color = color
            self._update_overmax_btn_style()
            self._settings.setValue("persistent_overmax_color", color.name())

            if v := self._cur_view():
                v.set_overmax_color(color)
            if s := self._cur_settings():
                s["overmax_color"] = color.name()
                self._log(f"Over-max highlight color: {color.name()}")

    # ----- tool pages -----
    def _build_view_page(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        msg = QtWidgets.QLabel(
            "이동 / 확대 도구.\n\n"
            "• ✋ 이동: 좌드래그로 사진 위치 이동\n"
            "• 🔍 확대: 박스 드래그 / 스크롤 줌\n"
            "• 더블클릭: 기본 줌 복귀\n\n"
            "밝기·색·척도는 위 패널에서 조절합니다."
        )
        msg.setWordWrap(True)
        msg.setStyleSheet("color:#999;")
        lay.addWidget(msg)
        lay.addStretch(1)
        return w

    def _build_distance_page(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("점을 클릭해 구간 거리 측정 (링 간격).\n우클릭=가까운 점 삭제."))

        self._dist_table = QtWidgets.QTableWidget(0, 3)
        self._dist_table.setHorizontalHeaderLabels(["구간", "거리", "누적"])
        self._dist_table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self._dist_table, 1)

        self._dist_total = QtWidgets.QLabel("합계: –")
        self._dist_total.setStyleSheet("font-weight:bold;")
        lay.addWidget(self._dist_total)

        btn = QtWidgets.QPushButton("거리 초기화")
        btn.clicked.connect(self._do_clear_distance)
        lay.addWidget(btn)
        return w

    def _build_line_page(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("라인을 드래그하면 강도 프로파일이\n하단 플롯에 표시됩니다."))

        self._line_stats = QtWidgets.QFormLayout()
        self._line_len = QtWidgets.QLabel("–")
        self._line_min = QtWidgets.QLabel("–")
        self._line_max = QtWidgets.QLabel("–")
        self._line_mean = QtWidgets.QLabel("–")
        self._line_stats.addRow("길이", self._line_len)
        self._line_stats.addRow("min", self._line_min)
        self._line_stats.addRow("max", self._line_max)
        self._line_stats.addRow("mean", self._line_mean)
        box = QtWidgets.QGroupBox("프로파일 통계")
        box.setLayout(self._line_stats)
        lay.addWidget(box)
        lay.addStretch(1)
        return w

    def _build_ellipse_page(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(w)

        self._fit_shape = QtWidgets.QComboBox()
        self._fit_shape.addItems(["타원 (Ellipse)"])
        fl.addWidget(self._fit_shape)

        hint = QtWidgets.QLabel("좌클릭=점 추가, 우클릭=가까운 점 삭제\n5점 이상에서 Fit.")
        hint.setStyleSheet("color:#888;")
        fl.addWidget(hint)

        row = QtWidgets.QHBoxLayout()
        btn_fit = QtWidgets.QPushButton("Fit")
        btn_fit.clicked.connect(self._do_fit)
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.clicked.connect(self._on_clear_fit)
        btn_save = QtWidgets.QPushButton("결과 저장")
        btn_save.clicked.connect(self._save_fit)
        row.addWidget(btn_fit)
        row.addWidget(btn_clear)
        row.addWidget(btn_save)
        fl.addLayout(row)

        self._results = QtWidgets.QPlainTextEdit()
        self._results.setReadOnly(True)
        self._results.setFont(QtGui.QFont("Consolas", 9))
        fl.addWidget(self._results, 1)
        return w

    # ----------------------------------------------------- analysis dock
    def _build_analysis_dock(self) -> None:
        self._analysis_dock = QtWidgets.QDockWidget("분석 / 로그", self)
        self._analysis_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        tabs = QtWidgets.QTabWidget()

        self._profile_plot = pg.PlotWidget()
        self._profile_plot.setBackground("#202225")
        self._profile_plot.showGrid(x=True, y=True, alpha=0.18)
        self._profile_plot.setLabel("bottom", "거리 (px)")
        self._profile_plot.setLabel("left", "강도")
        for ax in ("bottom", "left"):
            axis = self._profile_plot.getAxis(ax)
            axis.setPen(pg.mkPen("#666", width=1))
            axis.setTextPen(pg.mkPen("#bbb"))
        self._profile_curve = self._profile_plot.plot(
            pen=pg.mkPen("#22d3ee", width=1.6)
        )
        tabs.addTab(self._profile_plot, "라인 프로파일")

        # Per-file operation log
        self._log_view = QtWidgets.QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QtGui.QFont("Consolas", 9))
        tabs.addTab(self._log_view, "로그")

        self._analysis_dock.setWidget(tabs)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._analysis_dock)
        self._analysis_dock.hide()

    def _log(self, message: str) -> None:
        """Append a log line to the active document session and refresh view."""
        doc = self._cur()
        if doc is None:
            return
        doc.add_log(message)
        self._refresh_log()

    def _refresh_log(self) -> None:
        doc = self._cur()
        self._log_view.setPlainText("\n".join(doc.logs) if doc else "")
        self._log_view.moveCursor(QtGui.QTextCursor.End)

    def _on_line_profile(self, axis, values, unit: str) -> None:
        self._profile_curve.setData(axis, values)
        self._profile_plot.setLabel("bottom", f"거리 ({unit})")
        self._profile_plot.setLabel("left", "강도")
        vals = np.asarray(values, dtype=float)
        if vals.size:
            self._line_len.setText(f"{axis[-1]:.2f} {unit}")
            self._line_min.setText(f"{vals.min():.4g}")
            self._line_max.setText(f"{vals.max():.4g}")
            self._line_mean.setText(f"{vals.mean():.4g}")

    def _on_distance(self, seg, cum, unit: str) -> None:
        self._dist_table.setRowCount(len(seg))
        for i, (s, c) in enumerate(zip(seg, cum)):
            self._dist_table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{i+1}→{i+2}"))
            self._dist_table.setItem(
                i, 1, QtWidgets.QTableWidgetItem(f"{s:.2f} {unit}")
            )
            self._dist_table.setItem(
                i, 2, QtWidgets.QTableWidgetItem(f"{c:.2f} {unit}")
            )
        self._dist_total.setText(
            f"합계: {cum[-1]:.2f} {unit}" if cum else "합계: –"
        )

    def _build_statusbar(self) -> None:
        self._status = self.statusBar()
        self._pos_label = QtWidgets.QLabel("커서: –")
        self._info_label = QtWidgets.QLabel("이미지 없음")
        self._status.addWidget(self._pos_label, 1)
        self._status.addPermanentWidget(self._info_label)

        # Right side: zoom controls.
        btn_out = QtWidgets.QToolButton()
        btn_out.setText("−")
        btn_out.setAutoRaise(True)
        btn_out.clicked.connect(lambda: self._cur_view() and self._cur_view().zoom_out())
        btn_in = QtWidgets.QToolButton()
        btn_in.setText("+")
        btn_in.setAutoRaise(True)
        btn_in.clicked.connect(lambda: self._cur_view() and self._cur_view().zoom_in())
        btn_fit = QtWidgets.QToolButton()
        btn_fit.setText("⤢")
        btn_fit.setToolTip("화면에 맞춤 (fit)")
        btn_fit.setAutoRaise(True)
        btn_fit.clicked.connect(self._do_reset)

        self._zoom_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._zoom_slider.setRange(10, 800)
        self._zoom_slider.setValue(100)
        self._zoom_slider.setFixedWidth(120)
        self._zoom_slider.sliderMoved.connect(self._on_zoom_slider)
        self._zoom_label = QtWidgets.QLabel("100%")
        self._zoom_label.setFixedWidth(44)
        self._zoom_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        for wd in (btn_out, self._zoom_slider, btn_in, self._zoom_label, btn_fit):
            self._status.addPermanentWidget(wd)

    def _on_zoom_slider(self, val: int) -> None:
        if v := self._cur_view():
            v.set_zoom(float(val))

    def _on_zoom_changed(self, percent: float) -> None:
        self._zoom_label.setText(f"{percent:.0f}%")
        if not self._zoom_slider.isSliderDown():
            with QtCore.QSignalBlocker(self._zoom_slider):
                self._zoom_slider.setValue(int(max(10, min(800, percent))))

    def _set_tree_root(self, folder: Path) -> None:
        """Root the file tree AT this folder, not the whole disk."""
        self._fs_model.setRootPath(str(folder))
        self._tree.setRootIndex(self._fs_model.index(str(folder)))

    @staticmethod
    def _cmap_icon(name: str, w: int = 96, h: int = 14) -> QtGui.QIcon:
        """A horizontal gradient swatch previewing a colormap in the dropdown."""
        try:
            cm = pg.colormap.get(name, source="matplotlib")
        except Exception:
            cm = pg.colormap.get("gray", source="matplotlib")
        lut = cm.getLookupTable(0.0, 1.0, w, alpha=False)
        arr = np.ascontiguousarray(np.repeat(lut[np.newaxis, :, :], h, axis=0))
        qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        return QtGui.QIcon(QtGui.QPixmap.fromImage(qimg.copy()))

    @staticmethod
    def _under_temp(p: str | Path) -> bool:
        """True if a path lives inside the system temp/cache directory."""
        try:
            Path(p).resolve().relative_to(Path(tempfile.gettempdir()).resolve())
            return True
        except (ValueError, OSError):
            return False

    # ------------------------------------------------------ session restore
    def _restore_session(self) -> None:
        """Reopen at the previously used folder / file."""
        last_dir = self._settings.value("last_dir", "", type=str)
        if last_dir and Path(last_dir).exists() and not self._under_temp(last_dir):
            self._set_tree_root(Path(last_dir))
        elif self._under_temp(last_dir):
            self._settings.remove("last_dir")

        last_file = self._settings.value("last_file", "", type=str)
        if (
            last_file
            and Path(last_file).is_file()
            and not self._under_temp(last_file)
        ):
            self._open_document(Path(last_file))
        else:
            if self._under_temp(last_file):
                self._settings.remove("last_file")
            self._on_tab_changed()

    def _remember(self) -> None:
        if self._current_file is not None and not self._under_temp(self._current_file):
            self._settings.setValue("last_file", str(self._current_file))
            self._settings.setValue("last_dir", str(self._current_file.parent))

    # =================================================== document / tab system
    def _make_image_view(self) -> ImageView:
        v = ImageView()
        v.cursorMoved.connect(self._on_cursor)
        v.fitReported.connect(self._on_fit_reported)
        v.lineProfileChanged.connect(self._on_line_profile)
        v.distanceChanged.connect(self._on_distance)
        v.zoomChanged.connect(self._on_zoom_changed)
        v.histogramReattached.connect(self._sync_hist_chk)
        v.roiAdded.connect(self._on_roi_added)
        v.roiRemoved.connect(self._on_roi_removed)
        v.roiSelected.connect(self._on_roi_selected)
        v.lineDrawn.connect(self._on_line_drawn)
        v.rectDrawn.connect(self._on_rect_drawn)
        return v

    def _sync_hist_chk(self) -> None:
        v = self._cur_view()
        with QtCore.QSignalBlocker(self._hist_chk):
            if v is None:
                self._hist_chk.setEnabled(False)
            elif v.is_histogram_detached():
                self._hist_chk.setChecked(False)
                self._hist_chk.setEnabled(False)
            else:
                self._hist_chk.setEnabled(True)
                self._hist_chk.setChecked(v.histogram_visible())

    def _apply_settings_to(self, view: ImageView, s: dict) -> None:
        """Apply a session's stored view_settings onto its view."""
        view.set_log(s.get("log", False))
        view.set_overmax(s.get("overmax", True))
        if "overmax_color" in s:
            view.set_overmax_color(QtGui.QColor(s["overmax_color"]))
        view.set_colormap(s.get("colormap", "gray"))
        if s.get("scale_on"):
            view.set_scale(s.get("px_size", 1.0), s.get("unit", "µm"))
        else:
            view.set_scale_pixels()
        view.set_tool(s.get("tool", "pan"))

    def _sync_panel_from(self, s: dict) -> None:
        """Reflect a session's settings in the shared control panel (no signals)."""
        widgets = [
            self._log_chk, self._overmax_chk, self._cmap_combo,
            self._scale_chk, self._px_size_spin, self._unit_combo,
        ]
        if hasattr(self, "_ribbon_cmap"):
            widgets.append(self._ribbon_cmap)
        blockers = [QtCore.QSignalBlocker(x) for x in widgets]
        self._log_chk.setChecked(s.get("log", False))
        self._overmax_chk.setChecked(s.get("overmax", True))
        if "overmax_color" in s:
            self._overmax_color = QtGui.QColor(s["overmax_color"])
            self._update_overmax_btn_style()

        self._cmap_combo.setCurrentText(s.get("colormap", "gray"))
        if hasattr(self, "_ribbon_cmap"):
            self._ribbon_cmap.setCurrentText(s.get("colormap", "gray"))
        self._scale_chk.setChecked(s.get("scale_on", False))
        self._px_size_spin.setValue(s.get("px_size", 1.0))
        self._unit_combo.setCurrentText(s.get("unit", "µm"))
        del blockers
        tool = s.get("tool", "select")
        if tool in self._tool_actions:
            self._tool_actions[tool].setChecked(True)

    def _cur_settings(self) -> dict | None:
        doc = self._cur()
        return doc.view_settings if doc and doc.kind == "image" else None

    def _open_document(self, path: Path, preview: bool = False) -> None:
        path = Path(path)
        for w, d in self._docs.items():
            if d.file_path == path:
                if not preview and d.preview:
                    self._pin_tab(w)
                self._tabs.setCurrentWidget(w)
                return
        if preview:
            self._close_preview()
        suf = path.suffix.lower()
        try:
            if suf in io.H5_SUFFIXES:
                self._open_h5(path)
            elif suf in io.TEXT_SUFFIXES:
                self._open_text(path)
            elif suf in io.IMAGE_SUFFIXES:
                self._open_image_file(path)
            else:
                try:
                    self._open_image_file(path)
                except Exception:
                    self._open_text(path)
        except Exception as exc:
            self._status.showMessage(f"열기 실패: {path.name} — {exc}", 6000)
            return
        w = self._tabs.currentWidget()
        if w in self._docs:
            self._docs[w].preview = preview
            self._style_tab(w)
        if not self._under_temp(path):
            self._settings.setValue("last_file", str(path))
            self._settings.setValue("last_dir", str(path.parent))

    def _close_preview(self) -> None:
        for w, d in list(self._docs.items()):
            if d.preview:
                self._docs.pop(w, None)
                self._tabs.removeTab(self._tabs.indexOf(w))
                w.deleteLater()
                break

    def _pin_tab(self, w: QtWidgets.QWidget) -> None:
        if w in self._docs:
            self._docs[w].preview = False
            self._style_tab(w)

    def _reveal_in_tree(self, path: Path) -> None:
        """Scroll the file sidebar to the given file (matches the active tab)."""
        root_path = self._fs_model.filePath(self._tree.rootIndex())
        if root_path:
            try:
                path.relative_to(Path(root_path))
                inside = True
            except ValueError:
                inside = False
            if not inside:
                self._set_tree_root(path.parent)

        def do(retry: int = 0) -> None:
            idx = self._fs_model.index(str(path))
            if not idx.isValid():
                return
            self._tree.scrollTo(idx, QtWidgets.QAbstractItemView.PositionAtCenter)
            self._tree.setCurrentIndex(idx)
            if retry == 0:
                QtCore.QTimer.singleShot(150, lambda: do(1))

        do()

    def _style_tab(self, w: QtWidgets.QWidget) -> None:
        """Dim cue for preview tabs; normal for pinned."""
        i = self._tabs.indexOf(w)
        if i < 0 or w not in self._docs:
            return
        doc = self._docs[w]
        self._tabs.tabBar().setTabTextColor(
            i, QtGui.QColor("#8a8a8a") if doc.preview else QtGui.QColor("#e6e6e6")
        )
        self._tabs.setTabToolTip(
            i,
            f"{doc.file_path}\n(미리보기 — 더블클릭으로 고정)"
            if doc.preview
            else str(doc.file_path),
        )

    def _add_tab(self, widget: QtWidgets.QWidget, doc: DocumentSession, title: str) -> None:
        self._docs[widget] = doc
        idx = self._tabs.addTab(widget, title)
        self._tabs.setTabToolTip(idx, str(doc.file_path))
        self._tabs.setCurrentWidget(widget)

    def _open_h5(self, path: Path) -> None:
        root = io.read_structure(path)
        frames = io.list_frames(path)
        view = self._make_image_view()
        doc = DocumentSession(
            file_path=path, kind="image", view=view, frames=frames,
            frame=frames[0] if frames else None, structure=root, info=path.name,
            view_settings=self._default_settings(),
        )
        if frames:
            img = io.load_frame(path, frames[0])
            view.set_image(img)
            doc.info = self._img_info(path, str(frames[0]), img)
            doc.add_log(f"열기: {path.name} ({frames[0]})")
        else:
            view.show_error("DFXM 프레임 없음\n구조 탭에서 2D 데이터셋을 더블클릭하세요")
            doc.add_log(f"열기: {path.name} (프레임 없음)")
        self._add_tab(view, doc, path.name)

    def _open_image_file(self, path: Path) -> None:
        img = io.load_image_file(path)
        view = self._make_image_view()
        view.set_image(img)
        doc = DocumentSession(
            file_path=path, kind="image", view=view,
            info=self._img_info(path, path.suffix.lstrip("."), img),
            view_settings=self._default_settings(),
        )
        doc.add_log(f"열기: {path.name}  {img.shape[1]}×{img.shape[0]}")
        self._add_tab(view, doc, path.name)

    def _open_text(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() == ".json":
            try:
                text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
            except Exception:
                pass
        editor = QtWidgets.QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setFont(QtGui.QFont("Consolas", 10))
        editor.setPlainText(text)
        doc = DocumentSession(file_path=path, kind="text", info=path.name)
        doc.add_log(f"열기: {path.name} (텍스트)")
        self._add_tab(editor, doc, path.name)

    @staticmethod
    def _img_info(path: Path, sub: str, img) -> str:
        return (
            f"{path.name}  |  {sub}  |  {img.shape[1]}×{img.shape[0]}  |  "
            f"min {img.min():.3g}  max {img.max():.3g}"
        )

    def _on_tab_close(self, index: int) -> None:
        w = self._tabs.widget(index)
        self._docs.pop(w, None)
        self._tabs.removeTab(index)
        w.deleteLater()

    def _on_tab_changed(self, *_) -> None:
        doc = self._cur()
        self._refresh_log()
        if doc is None:
            self._info_label.setText("파일을 열어보세요")
            self._struct_tree.clear()
            with QtCore.QSignalBlocker(self._scan_combo), QtCore.QSignalBlocker(
                self._det_combo
            ):
                self._scan_combo.clear()
                self._det_combo.clear()
            return

        self._current_file = doc.file_path
        self._info_label.setText(doc.info or doc.file_path.name)
        self._reveal_in_tree(doc.file_path)

        if doc.kind != "image":
            self._struct_tree.clear()
            with QtCore.QSignalBlocker(self._scan_combo), QtCore.QSignalBlocker(
                self._det_combo
            ):
                self._scan_combo.clear()
                self._det_combo.clear()
            return

        if doc.structure is not None:
            self._populate_structure(doc.structure)
        else:
            self._struct_tree.clear()

        self._frames = doc.frames
        with QtCore.QSignalBlocker(self._scan_combo), QtCore.QSignalBlocker(
            self._det_combo
        ):
            self._scan_combo.clear()
            self._det_combo.clear()
            if doc.frames:
                self._scan_combo.addItems(sorted({f.scan for f in doc.frames}))
                cur = doc.frame
                if cur is not None:
                    self._scan_combo.setCurrentText(cur.scan)
                self._populate_dets(self._scan_combo.currentText())
                if cur is not None:
                    self._det_combo.setCurrentText(cur.detector)

        doc.view_settings["log"] = self._log_chk.isChecked()
        doc.view_settings["overmax"] = self._overmax_chk.isChecked()
        doc.view_settings["colormap"] = self._cmap_combo.currentText()

        self._sync_panel_from(doc.view_settings)
        self._apply_settings_to(doc.view, doc.view_settings)
        self._sync_hist_chk()
        self._rebuild_object_tree()

    def _screenshot(self) -> None:
        """Save image + scale bar + an info strip as PNG, and copy to clipboard."""
        view = self._cur_view()
        if view is None or not view.has_image():
            self._status.showMessage("먼저 이미지를 여세요.", 3000)
            return
        pix = view.grab()

        lo, hi = view.get_levels()
        scale_txt = (
            f"{self._px_size_spin.value():g} {self._unit_combo.currentText()}/px"
            if self._scale_chk.isChecked()
            else "pixel"
        )
        info = (
            f"{self._info_label.text()}   |   levels [{lo:.4g}, {hi:.4g}]"
            f"   |   scale: {scale_txt}"
        )

        strip_h = 30
        out = QtGui.QImage(
            pix.width(), pix.height() + strip_h, QtGui.QImage.Format_ARGB32
        )
        out.fill(QtGui.QColor("#232323"))
        p = QtGui.QPainter(out)
        p.drawPixmap(0, 0, pix)
        p.setPen(QtGui.QColor("#e6e6e6"))
        p.setFont(QtGui.QFont("Arial", 9))
        p.drawText(
            QtCore.QRect(10, pix.height(), pix.width() - 20, strip_h),
            QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
            info,
        )
        p.end()

        QtWidgets.QApplication.clipboard().setImage(out)

        default = "screenshot.png"
        if self._current_file is not None:
            default = f"{self._current_file.stem}_shot.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "스크린샷 저장", default, "PNG (*.png)"
        )
        if path:
            out.save(path)
            self._status.showMessage(f"저장됨(+클립보드 복사): {path}", 4000)
        else:
            self._status.showMessage("클립보드에 복사됨.", 3000)

    def _on_scale_changed(self, *_) -> None:
        view = self._cur_view()
        s = self._cur_settings()
        if view is None or s is None:
            return
        on = self._scale_chk.isChecked()
        s["scale_on"] = on
        s["px_size"] = self._px_size_spin.value()
        s["unit"] = self._unit_combo.currentText()
        if on:
            view.set_scale(s["px_size"], s["unit"])
        else:
            view.set_scale_pixels()

    def _on_fit_reported(self, text: str) -> None:
        self._results.setPlainText(text)
        if text.startswith("ELLIPSE FIT RESULT"):
            self._log("타원 피팅 완료")

    # -------------------------------------------------------- file logic
    def _open_folder(self) -> None:
        start = self._settings.value("last_dir", "", type=str)
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "폴더 선택", start)
        if path:
            self._fs_model.setRootPath(path)
            self._tree.setRootIndex(self._fs_model.index(path))
            self._settings.setValue("last_dir", path)

    def _open_file(self) -> None:
        start = self._settings.value("last_dir", "", type=str)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "파일 선택",
            start,
            "지원 파일 (*.h5 *.hdf5 *.tif *.tiff *.png *.jpg *.jpeg *.bmp *.json *.txt);;"
            "모든 파일 (*.*)",
        )
        if path:
            self._open_document(Path(path))

    def _tree_path(self, index: QtCore.QModelIndex) -> Path | None:
        path = Path(self._fs_model.filePath(index))
        if path.is_file() and path.suffix.lower() in self._SUPPORTED:
            return path
        return None

    def _on_tree_preview(self, index: QtCore.QModelIndex) -> None:
        if p := self._tree_path(index):
            self._open_document(p, preview=True)

    def _on_tree_open(self, index: QtCore.QModelIndex) -> None:
        if p := self._tree_path(index):
            self._open_document(p, preview=False)

    def _populate_structure(self, root: io.H5Node) -> None:
        self._struct_tree.clear()

        def add(parent, node: io.H5Node) -> None:
            if node.is_group:
                kind, shape, dtype = "group", "", ""
            else:
                kind, shape, dtype = "dataset", str(node.shape), node.dtype or ""
            item = QtWidgets.QTreeWidgetItem([node.name, kind, shape, dtype])
            item.setData(0, QtCore.Qt.UserRole, node)
            if node.attrs:
                tip = "\n".join(f"{k} = {v}" for k, v in node.attrs.items())
                item.setToolTip(0, tip)
            if parent is None:
                self._struct_tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            for child in node.children or []:
                add(item, child)

        for child in root.children or []:
            add(None, child)
        self._struct_tree.expandToDepth(1)
        for c in range(4):
            self._struct_tree.resizeColumnToContents(c)

    def _on_struct_activated(self, item: QtWidgets.QTreeWidgetItem, _col: int) -> None:
        node: io.H5Node | None = item.data(0, QtCore.Qt.UserRole)
        doc = self._cur()
        if node is None or node.is_group or doc is None or doc.kind != "image":
            return
        try:
            img = io.load_dataset(doc.file_path, node.path)
        except Exception as exc:
            self._status.showMessage(f"이 데이터셋은 2D로 볼 수 없음: {exc}", 5000)
            return
        doc.view.set_image(img)
        doc.frame = None
        doc.info = self._img_info(doc.file_path, node.path, img)
        self._apply_settings_to(doc.view, doc.view_settings)
        self._pin_tab(self._tabs.currentWidget())
        doc.add_log(f"데이터셋 열기: {node.path}")
        self._refresh_log()
        self._info_label.setText(doc.info)

    def _populate_dets(self, scan: str) -> None:
        dets = [f.detector for f in self._frames if f.scan == scan]
        self._det_combo.clear()
        self._det_combo.addItems(dets)

    def _on_frame_selected(self, *_) -> None:
        doc = self._cur()
        if doc is None or doc.kind != "image" or not doc.frames:
            return
        scan = self._scan_combo.currentText()
        expected = [f.detector for f in doc.frames if f.scan == scan]
        if [self._det_combo.itemText(i) for i in range(self._det_combo.count())] != expected:
            with QtCore.QSignalBlocker(self._det_combo):
                self._populate_dets(scan)
        det = self._det_combo.currentText()
        if not scan or not det:
            return
        frame = io.FramePath(scan=scan, detector=det)
        try:
            img = io.load_frame(doc.file_path, frame)
        except Exception as exc:
            self._status.showMessage(f"로드 실패: {exc}", 5000)
            return
        doc.view.set_image(img)
        doc.frame = frame
        doc.info = self._img_info(doc.file_path, str(frame), img)
        self._apply_settings_to(doc.view, doc.view_settings)
        self._pin_tab(self._tabs.currentWidget())
        doc.add_log(f"프레임 전환: {frame}")
        self._refresh_log()
        self._remember()
        self._info_label.setText(doc.info)
        self._results.clear()

    # --------------------------------------------------------- fit logic
    def _on_clear_fit(self) -> None:
        if v := self._cur_view():
            v.clear_fit()
            v.clear_distance()
        self._results.clear()

    def _save_fit(self) -> None:
        view = self._cur_view()
        if view is None:
            self._status.showMessage("이미지 탭이 아닙니다.", 3000)
            return
        geom = view.last_geometry()
        if geom is None:
            self._status.showMessage("저장할 피팅 결과 없음. 먼저 Fit 실행.", 4000)
            return
        default = "ellipse.json"
        if self._current_file is not None:
            default = f"{self._current_file.stem}_ellipse.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "피팅 결과 저장", default, "JSON (*.json);;텍스트 (*.txt)"
        )
        if not path:
            return
        payload = {
            "source_file": str(self._current_file) if self._current_file else None,
            "scan": self._scan_combo.currentText(),
            "detector": self._det_combo.currentText(),
            "points": view.picked_points().tolist(),
            "geometry": geom,
        }
        p = Path(path)
        if p.suffix.lower() == ".txt":
            p.write_text(self._results.toPlainText(), encoding="utf-8")
        else:
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._status.showMessage(f"저장됨: {p}", 4000)

    # ------------------------------------------------------------ cursor
    def _on_cursor(self, x: float, y: float, val: float) -> None:
        if np.isnan(val):
            self._pos_label.setText(f"커서:  x={x:.1f}  y={y:.1f}  (범위 밖)")
        else:
            self._pos_label.setText(f"커서:  x={x:.1f}  y={y:.1f}  값={val:.4g}")