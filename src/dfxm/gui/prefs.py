"""View preferences — one owner for display state.

Before this existed the same setting lived in up to four places (the widget's
own checked state, a QSettings key, a free-form ``view_settings`` dict on the
document, and sometimes the Core recipe), and they disagreed. The rules now:

* **Document-scoped, Core-owned** — anything that changes pixel VALUES is an op
  in ``doc.ds.history`` (log, gamma, sub_bg, geometry). Never duplicated here.
  The Log checkbox is a *view* of ``ds.log_scale``, not a place state lives.
* **Document-scoped, view-owned** — how one image is drawn right now:
  :class:`ViewPrefs` on the document.
* **App-scoped** — the same object held by the window as "what a newly opened
  document starts with", persisted to QSettings.

So a panel widget never stores state: it edits the app prefs and the current
document's prefs, and both are re-read when a tab is activated.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

#: QSettings keys, per field. Only these survive a restart.
_SETTINGS_KEYS = {
    "colormap": "def_cmap",
    "overmax": "persistent_overmax",
    "overmax_color": "persistent_overmax_color",
    "scale_on": "persistent_scale_on",
    "px_size": "persistent_px_size",
    "unit": "def_unit",
}

DEFAULT_OVERMAX_COLOR = "#ff3b30"


@dataclass
class ViewPrefs:
    """How one image is displayed. No intensity ops — those live in the recipe."""

    colormap: str = "gray"
    overmax: bool = True
    overmax_color: str = DEFAULT_OVERMAX_COLOR
    scale_on: bool = False  # scale bar in real units vs. pixels
    px_size: float = 1.0
    unit: str = "µm"
    tool: str = "select"

    # --------------------------------------------------------- persistence
    @classmethod
    def load(cls, settings) -> ViewPrefs:
        """Read the app-scoped defaults (``tool`` is deliberately not stored)."""
        d = cls()
        return cls(
            colormap=settings.value(_SETTINGS_KEYS["colormap"], d.colormap, type=str),
            overmax=settings.value(_SETTINGS_KEYS["overmax"], d.overmax, type=bool),
            overmax_color=settings.value(
                _SETTINGS_KEYS["overmax_color"], d.overmax_color, type=str
            ),
            scale_on=settings.value(_SETTINGS_KEYS["scale_on"], d.scale_on, type=bool),
            px_size=settings.value(_SETTINGS_KEYS["px_size"], d.px_size, type=float),
            unit=settings.value(_SETTINGS_KEYS["unit"], d.unit, type=str),
        )

    def save(self, settings) -> None:
        for field_name, key in _SETTINGS_KEYS.items():
            settings.setValue(key, getattr(self, field_name))

    # ---------------------------------------------------------------- use
    def copy(self, **changes) -> ViewPrefs:
        """A new instance — documents must never share one prefs object."""
        return replace(self, **changes)

    def apply_to(self, view) -> None:
        """The single path from prefs to an ImageView."""
        from PySide6 import QtGui

        view.set_overmax(self.overmax)
        view.set_overmax_color(QtGui.QColor(self.overmax_color))
        view.set_colormap(self.colormap)
        if self.scale_on:
            view.set_scale(self.px_size, self.unit)
        else:
            view.set_scale_pixels()
        view.set_tool(self.tool)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
