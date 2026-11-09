"""Styles and small widgets shared by the main window and both workspaces."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QVBoxLayout

from . import theme


def set_role(button: QPushButton, role: str):
    """Give a push button its themed look: "primary" (accent fill) or "secondary" (see theme.style_sheet)."""
    if button.property("role") != role:
        button.setProperty("role", role)
        _repolish(button)


def _repolish(w):
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()


def capitalize_words(text: str) -> str:
    """Capitalise the first letter of every word, keep the rest as typed (e.g. 'pH' stays)."""
    return " ".join(w[:1].upper() + w[1:] for w in text.split(" "))


def arrow_icon(mirrored: bool = False) -> QIcon:
    """Curved undo arrow (mirrored = redo), drawn so it is crisp at any DPI."""
    icon = QIcon()
    t = theme.current()
    for mode, color in ((QIcon.Normal, QColor(t.icon)), (QIcon.Active, QColor(t.icon_active)),
                        (QIcon.Disabled, QColor(t.icon_dis))):
        pm = QPixmap(40, 40)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        if mirrored:
            p.translate(40, 0)
            p.scale(-1, 1)
        p.setPen(QPen(color, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        path = QPainterPath()
        path.moveTo(11, 17)
        path.cubicTo(17, 9, 31, 10, 32, 22)       # curved shaft
        path.cubicTo(32, 28, 27, 31, 22, 31)
        p.drawPath(path)
        p.setBrush(color)
        p.setPen(Qt.NoPen)
        head = QPainterPath()
        head.moveTo(5, 18)
        head.lineTo(15, 9)
        head.lineTo(17, 22)
        head.closeSubpath()
        p.drawPath(head)
        p.end()
        icon.addPixmap(pm, mode)
    return icon


def color_icon(rgb, size=14) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(*rgb))
    icon = QIcon()
    for mode in (QIcon.Normal, QIcon.Selected, QIcon.Active):   # no tinting on selected rows
        icon.addPixmap(pm, mode)
    return icon


class Section(QFrame):
    """Rounded card with a numbered header and an optional ••• menu."""

    def __init__(self, number: int | None, title: str, menu: QMenu | None = None):
        super().__init__()
        self.setObjectName("Section")
        if number is not None:
            self.setProperty("step", str(number))     # per-step colours (theme.Theme.steps)
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 14)
        v.setSpacing(8)
        head = QHBoxLayout()
        self.badge = QLabel(str(number))
        self.badge.setObjectName("Badge")
        self.badge.setFixedSize(22, 22)
        self.badge.setAlignment(Qt.AlignCenter)
        head.addWidget(self.badge)
        self.badge.setVisible(number is not None)
        self._title = title
        self.title_label = QLabel(title)
        self.title_label.setObjectName("SectionTitle")
        head.addWidget(self.title_label)
        head.addStretch(1)
        if menu is not None:
            mb = QPushButton("•••")
            mb.setFixedSize(34, 24)
            mb.setCursor(Qt.PointingHandCursor)
            mb.setToolTip(f"More {title.lower()} options")
            mb.setMenu(menu)
            mb.setObjectName("More")
            head.addWidget(mb)
        v.addLayout(head)
        self.body = v
        self.set_active(False)
        self._apply_theme()
        theme.manager().changed.connect(self._apply_theme)

    def _apply_theme(self):
        self.title_label.setText(self._title.upper() if theme.current().upper_titles else self._title)

    def set_active(self, on: bool):
        value = "true" if on else "false"
        if self.badge.property("active") != value:
            self.badge.setProperty("active", value)
            _repolish(self.badge)
