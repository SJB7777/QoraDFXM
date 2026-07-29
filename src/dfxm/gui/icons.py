from pathlib import Path
from PySide6 import QtGui
import qtawesome as qta

# repo root anchored to this file, not CWD
_REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = _REPO_ROOT / "assets"
ICON_DIR = ASSETS_DIR / "icons"
LOGO_PATH = ASSETS_DIR / "logo.png"

# qtawesome fallback when logo.png missing
APP_ICON_QTA = "fa5s.x-ray"
APP_ICON_COLOR = "#4da3ff"

def logo_icon() -> QtGui.QIcon:
    if LOGO_PATH.exists():
        icon = QtGui.QIcon(str(LOGO_PATH))
        if not icon.isNull():
            return icon
    try:
        return qta.icon(APP_ICON_QTA, color=APP_ICON_COLOR)
    except Exception:
        return QtGui.QIcon()

def logo_pixmap(size: int = 120) -> QtGui.QPixmap:
    if LOGO_PATH.exists():
        pm = QtGui.QPixmap(str(LOGO_PATH))
        if not pm.isNull():
            return pm.scaled(size, size, QtGui.Qt.KeepAspectRatio, QtGui.Qt.SmoothTransformation)
    try:
        return qta.icon(APP_ICON_QTA, color=APP_ICON_COLOR).pixmap(size, size)
    except Exception:
        return QtGui.QPixmap()


class AppIcons:
    # 검증된 QtAwesome 아이콘 키값들
    HAND = "fa5s.hand-paper"
    FOLDER = "ph.folder-open-bold"
    CAMERA = "fa5s.camera"               # <--- 'msc.camera' 수정!
    DEFAULT_ZOOM = "fa5s.expand-arrows-alt"
    PAN = "fa5s.hand-paper"
    ZOOM_IN = "fa5s.search-plus"
    ZOOM_OUT = "fa5s.search-minus"
    RULER = "ph.ruler-bold"
    LINE = "ph.line-segment-bold"
    OVAL = "ph.ellipse-bold"
    CHECK = "fa5s.check"
    FILE = "ph.file-bold"
    COPY = "ph.copy-bold"
    EXTERNAL = "ph.arrow-square-out-bold"
    DOWNLOAD = "ph.download-simple-bold"
    WARNING = "ph.warning-circle-bold"

    @staticmethod
    def get(name_or_qta: str, color: str = "#d4d4d8") -> QtGui.QIcon:
        try:
            return qta.icon(name_or_qta, color=color)
        except Exception:
            return QtGui.QIcon()
            
    @staticmethod
    def get_pixmap(name_or_qta: str, size: int = 14, color: str = "#ffffff") -> QtGui.QPixmap:
        """QSS 또는 QPainter용 Pixmap 반환"""
        icon = AppIcons.get(name_or_qta, color=color)
        return icon.pixmap(size, size)