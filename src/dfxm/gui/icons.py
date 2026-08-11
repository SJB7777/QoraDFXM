from pathlib import Path

import qtawesome as qta
from PySide6 import QtGui

# Ships inside the package, so an installed wheel / frozen build finds it too.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_MARK_PATH = ASSETS_DIR / "logo_mark.png"  # square badge — icons, tabs
LOGO_FULL_PATH = ASSETS_DIR / "logo_full.png"  # badge + DFXM wordmark
# The wordmark is navy: unreadable on a dark panel, so it has a light twin.
LOGO_FULL_DARK_PATH = ASSETS_DIR / "logo_full_on_dark.png"

# qtawesome fallback when the artwork is missing
APP_ICON_QTA = "fa5s.x-ray"
APP_ICON_COLOR = "#4da3ff"


def logo_icon() -> QtGui.QIcon:
    """Window / taskbar icon: the square mark, never the wordmark."""
    if LOGO_MARK_PATH.exists():
        icon = QtGui.QIcon(str(LOGO_MARK_PATH))
        if not icon.isNull():
            return icon
    try:
        return qta.icon(APP_ICON_QTA, color=APP_ICON_COLOR)
    except Exception:
        return QtGui.QIcon()


def _scaled(path: Path, w: int, h: int) -> QtGui.QPixmap | None:
    if not path.exists():
        return None
    pm = QtGui.QPixmap(str(path))
    if pm.isNull():
        return None
    return pm.scaled(w, h, QtGui.Qt.KeepAspectRatio, QtGui.Qt.SmoothTransformation)


def logo_pixmap(size: int = 120) -> QtGui.QPixmap:
    """The square mark at `size` px — legible down to toolbar height."""
    pm = _scaled(LOGO_MARK_PATH, size, size)
    if pm is not None:
        return pm
    try:
        return qta.icon(APP_ICON_QTA, color=APP_ICON_COLOR).pixmap(size, size)
    except Exception:
        return QtGui.QPixmap()


def wordmark_pixmap(height: int = 72, on_dark: bool = True) -> QtGui.QPixmap:
    """The full lockup — only where there is room to read it (About/settings)."""
    path = LOGO_FULL_DARK_PATH if on_dark else LOGO_FULL_PATH
    pm = _scaled(path, height * 4, height) or _scaled(
        LOGO_FULL_PATH, height * 4, height
    )
    return pm if pm is not None else logo_pixmap(height)


class AppIcons:
    # 검증된 QtAwesome 아이콘 키값들
    HAND = "fa5s.hand-paper"
    FOLDER = "ph.folder-open-bold"
    CAMERA = "fa5s.camera"  # <--- 'msc.camera' 수정!
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
    def get_pixmap(
        name_or_qta: str, size: int = 14, color: str = "#ffffff"
    ) -> QtGui.QPixmap:
        """QSS 또는 QPainter용 Pixmap 반환"""
        icon = AppIcons.get(name_or_qta, color=color)
        return icon.pixmap(size, size)
