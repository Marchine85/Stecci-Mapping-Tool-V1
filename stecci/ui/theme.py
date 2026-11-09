"""App themes: colours, shapes and fonts of the whole UI, switchable at run time (Settings → Appearance).

`apply(key)` sets the application palette, font and one application-wide style sheet, then emits
`manager().changed` so custom-painted widgets (photo canvas, 3D viewer, busy overlay, icons) can follow.
Widgets pick their look through object names (#Section, #Badge, #Link …) and the dynamic property
`role` = "primary" | "secondary" on push buttons (see widgets.set_role); no widget sets its own colours.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


@dataclass(frozen=True)
class Theme:
    key: str
    dark: bool
    styled: bool                 # False = keep the system palette + Fusion defaults (Classic)
    font: str | None             # body font family (None = system font)
    head_font: str | None        # section titles / mode tag
    head_px: int
    upper_titles: bool
    window: str
    menu: str
    card: str
    line: str
    text: str
    title: str
    title_dis: str
    muted: str
    hint: str
    accent: str
    accent_hover: str
    on_accent: str
    link: str
    link_hover: str
    link_dis: str
    sec: str
    sec_line: str
    sec_text: str
    sec_hover: str
    dis_bg: str
    dis_fg: str
    dis_line: str
    list_bg: str
    sel: str
    sel_text: str
    tag: str
    tag_text: str
    badge_idle: str
    badge_text: str
    more: str
    more_text: str
    more_hover: str
    track: str
    knob: str
    canvas: str                  # photo canvas background
    viewer: str                  # 3D viewer background
    icon: str
    icon_active: str
    icon_dis: str
    ov_card: str                 # busy overlay
    ov_title: str
    ov_msg: str
    ov_time: str
    ov_track: str                # #AARRGGBB
    r_card: int
    r_btn: int
    r_start: int
    r_badge: int
    r_list: int
    r_row: int
    r_tag: int
    r_knob: int
    # optional colour per workflow step (1 Detect … 4 Draw Cracks): (fill, hover, light text) — badge, main
    # button and links of that step's card use it; None = everything uses `accent`
    steps: tuple[tuple[str, str, str], ...] | None = None
    slider_fill: tuple[str, str] | None = None   # gradient (left, right) of the filled slider part


THEMES: dict[str, Theme] = {
    "classic": Theme(
        key="classic", dark=True, styled=False, font=None, head_font=None, head_px=14, upper_titles=False,
        window="#202226", menu="#2b2d31", card="#24272c", line="#30343a", text="#e3e5e8", title="#e8eaed",
        title_dis="#5d6168", muted="gray", hint="#8a9099", accent="#2f7ae5", accent_hover="#4189f0",
        on_accent="#ffffff", link="#6ea8ff", link_hover="#9cc3ff", link_dis="#50545b",
        sec="#363a40", sec_line="#4a4f56", sec_text="#e3e5e8", sec_hover="#41464d",
        dis_bg="#2a2c30", dis_fg="#63676e", dis_line="#34373c", list_bg="#1d1f23", sel="#2f4f80",
        sel_text="#ffffff", tag="#26344a", tag_text="#9cc3ff", badge_idle="#3a3e45", badge_text="#ffffff",
        more="#30343a", more_text="#b4bac3", more_hover="#3d424a", track="#3a3e45", knob="#d6d9de",
        canvas="#282828", viewer="#1c1f22", icon="#e1e4e8", icon_active="#ffffff", icon_dis="#5f636a",
        ov_card="#1e2127", ov_title="#f0f2f5", ov_msg="#a0a6b0", ov_time="#6e747e",
        ov_track="#10ffffff",
        r_card=0, r_btn=0, r_start=0, r_badge=0, r_list=0, r_row=0, r_tag=0, r_knob=0),
    "nocturne": Theme(
        key="nocturne", dark=True, styled=True, font="IBM Plex Sans", head_font="IBM Plex Mono", head_px=12,
        upper_titles=True,
        window="#090d18", menu="#05070e", card="#161e36", line="#323e66", text="#f3f6ff", title="#ffffff",
        title_dis="#5c6690", muted="#c0c8e4", hint="#aab3d4", accent="#9ba6ff", accent_hover="#b6beff",
        on_accent="#070a14", link="#a8ecff", link_hover="#d2f6ff", link_dis="#4a5580",
        sec="#202a48", sec_line="#3e4b78", sec_text="#f3f6ff", sec_hover="#2b375e",
        dis_bg="#111729", dis_fg="#5c6690", dis_line="#232c4a", list_bg="#060914", sel="#3646a8",
        sel_text="#ffffff", tag="#0f3a38", tag_text="#5fe0c6", badge_idle="#2c3760", badge_text="#e2e7ff",
        more="#202a48", more_text="#c0c8e4", more_hover="#2e3a64", track="#2c3760", knob="#ffffff",
        canvas="#04060c", viewer="#04060c", icon="#f3f6ff", icon_active="#a8ecff", icon_dis="#5c6690",
        ov_card="#161e36", ov_title="#ffffff", ov_msg="#c0c8e4", ov_time="#aab3d4",
        ov_track="#1ef3f6ff",
        r_card=0, r_btn=0, r_start=0, r_badge=0, r_list=0, r_row=0, r_tag=0, r_knob=0,
        steps=(("#ffb547", "#ffc870", "#ffd08a"),      # 1 Detect: amber
               ("#38d2b4", "#5fe0c6", "#7ae6d0"),      # 2 Segmentation: teal
               ("#9ba6ff", "#b6beff", "#c4caff"),      # 3 Assign Classes: violet
               ("#ff5c5c", "#ff7f7f", "#ff9a9a")),     # 4 Draw Cracks: red
        slider_fill=("#9ba6ff", "#38d2b4")),
    "nordic": Theme(
        key="nordic", dark=False, styled=True, font="IBM Plex Sans", head_font="IBM Plex Mono", head_px=12,
        upper_titles=True,
        window="#f4f6f8", menu="#e9edf1", card="#ffffff", line="#dde3ea", text="#17212b", title="#17212b",
        title_dis="#a3acb6", muted="#4f5c69", hint="#5f6b78", accent="#0f6e6e", accent_hover="#138080",
        on_accent="#ffffff", link="#0d6363", link_hover="#0a4f4f", link_dis="#a3acb6",
        sec="#eef2f5", sec_line="#cfd7df", sec_text="#17212b", sec_hover="#e3e9ee",
        dis_bg="#f0f2f4", dis_fg="#a3acb6", dis_line="#e1e6eb", list_bg="#f7f9fb", sel="#d4ecea",
        sel_text="#0b3d3d", tag="#dcefee", tag_text="#0d5c5c", badge_idle="#e3e8ee", badge_text="#4f5c69",
        more="#eef2f5", more_text="#4f5c69", more_hover="#e1e7ec", track="#dde3ea", knob="#ffffff",
        canvas="#cfd6dd", viewer="#cfd6dd", icon="#17212b", icon_active="#0f6e6e", icon_dis="#a3acb6",
        ov_card="#ffffff", ov_title="#17212b", ov_msg="#4f5c69", ov_time="#5f6b78",
        ov_track="#1617212b",
        r_card=0, r_btn=0, r_start=0, r_badge=0, r_list=0, r_row=0, r_tag=0, r_knob=0),
    "graphite": Theme(
        key="graphite", dark=True, styled=True, font="IBM Plex Sans", head_font="IBM Plex Mono", head_px=12,
        upper_titles=True,
        window="#161616", menu="#111111", card="#1c1c1c", line="#2e2e2e", text="#f0f0f0", title="#f0f0f0",
        title_dis="#5a5a5a", muted="#a8a8a8", hint="#9a9a9a", accent="#ffb000", accent_hover="#ffc133",
        on_accent="#111111", link="#ffc947", link_hover="#ffdb85", link_dis="#555555",
        sec="#232323", sec_line="#3a3a3a", sec_text="#f0f0f0", sec_hover="#2c2c2c",
        dis_bg="#1a1a1a", dis_fg="#5a5a5a", dis_line="#2a2a2a", list_bg="#121212", sel="#3a2f10",
        sel_text="#ffe2a0", tag="#2a2410", tag_text="#ffc947", badge_idle="#2e2e2e", badge_text="#d0d0d0",
        more="#232323", more_text="#a8a8a8", more_hover="#2e2e2e", track="#2e2e2e", knob="#f0f0f0",
        canvas="#0d0d0d", viewer="#0d0d0d", icon="#f0f0f0", icon_active="#ffc947", icon_dis="#5a5a5a",
        ov_card="#1c1c1c", ov_title="#f0f0f0", ov_msg="#a8a8a8", ov_time="#9a9a9a",
        ov_track="#12ffffff",
        r_card=0, r_btn=0, r_start=0, r_badge=0, r_list=0, r_row=0, r_tag=0, r_knob=0),
}

_current: Theme = THEMES["nocturne"]
_orig: tuple[QPalette, QFont] | None = None
_fonts_loaded = False


class _Manager(QObject):
    changed = Signal()


_manager: _Manager | None = None


def manager() -> _Manager:
    global _manager
    if _manager is None:
        _manager = _Manager()
    return _manager


def current() -> Theme:
    return _current


def _load_fonts():
    global _fonts_loaded
    if _fonts_loaded:
        return
    _fonts_loaded = True
    if FONT_DIR.is_dir():
        for f in sorted(FONT_DIR.glob("*.ttf")):
            QFontDatabase.addApplicationFont(str(f))


def _classic_palette(pal: QPalette) -> QPalette:
    # Windows' dark scheme sets Base (checkbox / input fill) to the accent blue, which makes
    # unticked checkboxes look ticked: use a neutral dark fill instead
    pal = QPalette(pal)
    if pal.color(QPalette.Window).lightness() < 128:
        for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
            pal.setColor(group, QPalette.Base, QColor("#1d1f23"))
            pal.setColor(group, QPalette.AlternateBase, QColor("#24272c"))
    return pal


def _palette(t: Theme) -> QPalette:
    c = QColor
    pal = QPalette()
    sec = c(t.sec)
    roles = {
        QPalette.Window: t.window, QPalette.WindowText: t.text, QPalette.Base: t.list_bg,
        QPalette.AlternateBase: t.card, QPalette.ToolTipBase: t.card, QPalette.ToolTipText: t.text,
        QPalette.PlaceholderText: t.hint, QPalette.Text: t.text, QPalette.Button: t.sec,
        QPalette.ButtonText: t.sec_text, QPalette.BrightText: "#ffffff", QPalette.Highlight: t.accent,
        QPalette.HighlightedText: t.on_accent, QPalette.Link: t.link, QPalette.LinkVisited: t.link,
        QPalette.Mid: t.sec_line,
    }
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        for role, col in roles.items():
            pal.setColor(group, role, c(col))
        pal.setColor(group, QPalette.Light, sec.lighter(130))
        pal.setColor(group, QPalette.Midlight, sec.lighter(115))
        pal.setColor(group, QPalette.Dark, sec.darker(150))
        pal.setColor(group, QPalette.Shadow, c("#000000") if t.dark else c("#9aa4ae"))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, c(t.dis_fg))
    pal.setColor(QPalette.Disabled, QPalette.Highlight, c(t.dis_line))
    return pal


def style_sheet(t: Theme) -> str:
    head = f'font-family: "{t.head_font}";' if t.head_font else ""
    css = f"""
QPushButton[role="primary"], QPushButton[role="secondary"] {{ font-size: 14px; font-weight: 600;
    border-radius: {t.r_btn}px; padding: 8px 12px; }}
QPushButton[role="primary"] {{ background: {t.accent}; color: {t.on_accent}; border: 1px solid {t.accent}; }}
QPushButton[role="primary"]:hover {{ background: {t.accent_hover}; }}
QPushButton[role="secondary"] {{ background: {t.sec}; color: {t.sec_text}; border: 1px solid {t.sec_line}; }}
QPushButton[role="secondary"]:hover {{ background: {t.sec_hover}; }}
QPushButton[role="primary"]:checked, QPushButton[role="secondary"]:checked {{ background: {t.accent};
    color: {t.on_accent}; border-color: {t.accent}; }}
QPushButton[role="primary"]:disabled, QPushButton[role="secondary"]:disabled {{ background: {t.dis_bg};
    color: {t.dis_fg}; border-color: {t.dis_line}; }}
QPushButton#StartButton {{ font-size: 18px; border-radius: {t.r_start}px; }}
QLabel#StartTitle {{ font-size: 26px; font-weight: 600; }}
QLabel#StartSub {{ color: {t.muted}; font-size: 13px; }}
QLabel#StartCaption {{ color: {t.muted}; font-size: 11px; }}
QLabel#Muted {{ color: {t.muted}; }}
QLabel#SettingsHeading {{ font-weight: 600; font-size: 13px; }}
QPushButton#StartLink {{ color: {t.link}; background: transparent; border: none; }}
QPushButton#StartLink:hover {{ color: {t.link_hover}; }}

#Panel QFrame#Section {{ background: {t.card}; border: 1px solid {t.line}; border-radius: {t.r_card}px; }}
#Panel QLabel#SectionTitle {{ font-size: {t.head_px}px; font-weight: 600; color: {t.title}; {head} }}
#Panel QLabel#SectionTitle:disabled {{ color: {t.title_dis}; }}
#Panel QLabel#Hint {{ color: {t.hint}; font-size: 11px; }}
#Panel QLabel#ModeTag {{ color: {t.tag_text}; background: {t.tag}; border-radius: {t.r_tag}px; padding: 3px 8px;
    font-size: 11px; font-weight: 600; {head} }}
#Panel QPushButton#Link {{ color: {t.link}; background: transparent; border: none; text-align: left; padding: 2px; }}
#Panel QPushButton#Link:hover {{ color: {t.link_hover}; }}
#Panel QPushButton#Link:checked {{ color: {t.on_accent}; background: {t.accent}; border-radius: 0px;
    padding: 2px 6px; }}
#Panel QPushButton#Link:disabled {{ color: {t.link_dis}; }}
#Panel QPushButton#View {{ color: {t.more_text}; background: {t.more}; border: none; border-radius: 0px;
    padding: 3px 6px; }}
#Panel QPushButton#View:hover {{ background: {t.more_hover}; color: {t.text}; }}
#Panel QListWidget {{ background: {t.list_bg}; border: 1px solid {t.line}; border-radius: {t.r_list}px;
    padding: 4px; }}
#Panel QListWidget::item {{ padding: 4px 6px; border-radius: {t.r_row}px; }}
#Panel QListWidget::item:selected {{ background: {t.sel}; color: {t.sel_text}; }}

QLabel#Badge {{ background: {t.badge_idle}; color: {t.badge_text}; border-radius: {t.r_badge}px;
    font-weight: 700; }}
QLabel#Badge[active="true"] {{ background: {t.accent}; color: {t.on_accent}; }}
QPushButton#More {{ color: {t.more_text}; background: {t.more}; border: none; border-radius: 0px;
    font-size: 10px; font-weight: 700; letter-spacing: 1px; }}
QPushButton#More:hover {{ background: {t.more_hover}; color: {t.text}; }}
QPushButton#More:disabled {{ color: {t.link_dis}; background: transparent; }}
QPushButton#More::menu-indicator {{ width: 0; image: none; }}
"""
    if t.styled:
        fill = (f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t.slider_fill[0]}, stop:1 {t.slider_fill[1]})"
                if t.slider_fill else t.accent)
        knob_line = t.slider_fill[1] if t.slider_fill else t.accent
        css += f"""
QMenuBar {{ background: {t.menu}; }}
QStatusBar {{ background: {t.menu}; }}
QToolTip {{ background: {t.card}; color: {t.text}; border: 1px solid {t.line}; padding: 4px; }}
QSlider {{ min-height: 18px; }}
QSlider::groove:horizontal {{ height: 4px; background: {t.track}; border-radius: 0px; }}
QSlider::sub-page:horizontal {{ background: {fill}; border-radius: 0px; }}
QSlider::add-page:horizontal {{ background: {t.track}; border-radius: 0px; }}
QSlider::handle:horizontal {{ width: 12px; height: 12px; margin: -6px 0; background: {t.knob};
    border: 2px solid {knob_line}; border-radius: {t.r_knob}px; }}
QSlider::sub-page:horizontal:disabled {{ background: {t.dis_line}; }}
QSlider::handle:horizontal:disabled {{ background: {t.dis_bg}; border-color: {t.dis_line}; }}
"""
    for n, (c, hover, light) in enumerate(t.steps or (), start=1):
        sec = f'#Panel QFrame#Section[step="{n}"]'
        # :enabled keeps the general :disabled look (these selectors are more specific)
        css += f"""
{sec} QLabel#Badge[active="true"] {{ background: {c}; color: {t.on_accent}; }}
{sec} QPushButton[role="primary"]:enabled {{ background: {c}; color: {t.on_accent}; border-color: {c}; }}
{sec} QPushButton[role="primary"]:enabled:hover {{ background: {hover}; border-color: {hover}; }}
{sec} QPushButton[role="secondary"]:enabled:hover {{ border-color: {c}; }}
{sec} QPushButton:checked:enabled {{ background: {c}; color: {t.on_accent}; border-color: {c}; }}
{sec} QPushButton#Link:enabled {{ color: {light}; }}
{sec} QPushButton#Link:enabled:hover {{ color: {hover}; }}
{sec} QPushButton#Link:checked:enabled {{ color: {t.on_accent}; background: {c}; }}
"""
    return css


def apply(key: str):
    """Switch the whole app to theme `key` (unknown keys fall back to Nocturne, the default)."""
    global _current, _orig
    app = QApplication.instance()
    if app is None:
        return
    if _orig is None:
        _orig = (QPalette(app.palette()), QFont(app.font()))
    _load_fonts()
    t = THEMES.get(key, THEMES["nocturne"])
    if t.styled:
        app.setPalette(_palette(t))
        f = QFont(_orig[1])
        if t.font:
            f.setFamily(t.font)
        app.setFont(f)
    else:
        app.setPalette(_classic_palette(_orig[0]))
        app.setFont(_orig[1])
    app.setStyleSheet(style_sheet(t))
    _current = t
    manager().changed.emit()
