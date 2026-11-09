"""Modern animated busy overlay shown over the canvas while a model job runs."""
from __future__ import annotations

import random
import time
from math import cos, pi, radians, sin

from PySide6.QtCore import QEvent, QPropertyAnimation, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QConicalGradient, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget

from . import theme

ACCENT = QColor(64, 140, 255)
ACCENT_2 = QColor(56, 210, 180)
ACCENT_3 = QColor(255, 92, 92)

GRID, TILE, GAP = 3, 18, 5      # tile grid spinner: 3×3 tiles of 18 px, 5 px apart
FLIP_S = 0.35                   # duration of one tile flip


class BusyOverlay(QWidget):
    """Covers its parent, dims it and shows a spinner card. Blocks mouse input underneath."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.title = ""
        self.message = ""
        self._angle = 0.0
        self._t0 = 0.0
        self.style = "grid"          # "grid" (tile grid) or "rings" (rotating rings); set from Settings
        self._tiles: list[dict] = []
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)
        self._fade = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade.setDuration(220)
        self._fade.finished.connect(self._fade_done)

        parent.installEventFilter(self)
        self.hide()

    # ── public ───────────────────────────────────────────────────────────────
    def start(self, title: str, message: str | None = None):
        self.title = title
        if message is not None:
            self.message = message
        self._t0 = time.monotonic()
        self._tiles = []                 # fresh random grid each time
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        self._timer.start()
        self._animate_to(1.0)

    def set_message(self, message: str):
        self.message = message
        self.update()

    def stop(self):
        self._animate_to(0.0)

    # ── internals ────────────────────────────────────────────────────────────
    def _animate_to(self, value: float):
        self._fade.stop()
        self._fade.setStartValue(self._effect.opacity())
        self._fade.setEndValue(value)
        self._fade.start()

    def _fade_done(self):
        if self._effect.opacity() <= 0.01:
            self._timer.stop()
            self.hide()

    def _tick(self):
        # time-based and never wrapped here: wrapping the base angle at 360° made the inner arc
        # (speed -1.6) jump once per revolution, and per-tick increments stutter when ticks are late
        self._angle = (time.monotonic() - self._t0) * 300.0   # deg/s
        self.update()

    def eventFilter(self, obj, e):
        if obj is self.parentWidget() and e.type() == QEvent.Resize:
            self.setGeometry(obj.rect())
        return False

    # swallow input while visible
    def mousePressEvent(self, e):
        e.accept()

    def mouseReleaseEvent(self, e):
        e.accept()

    def wheelEvent(self, e):
        e.accept()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        t = theme.current()
        p.fillRect(self.rect(), QColor(8, 10, 14, 160))

        # card
        cw, ch = 340, 210
        card = QRectF((self.width() - cw) / 2, (self.height() - ch) / 2, cw, ch)
        path = QPainterPath()
        path.addRect(card)                  # sharp corners, like the rest of the UI
        card_c = QColor(t.ov_card)
        card_c.setAlpha(240)
        p.fillPath(path, card_c)
        p.setPen(QPen(QColor(t.line) if t.styled else QColor(255, 255, 255, 28), 1))
        p.drawPath(path)

        # spinner (Settings → Appearance → Loading animation)
        cx, cy = card.center().x(), card.top() + 72
        if self.style == "rings":
            self._paint_rings(p, t, cx, cy)
        else:
            self._paint_grid(p, t, cx, cy)

        # texts
        p.setPen(QColor(t.ov_title))
        f = QFont(self.font())
        f.setPointSizeF(12.5)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.drawText(QRectF(card.left(), cy + 46, cw, 24), Qt.AlignCenter, self.title)

        f.setPointSizeF(9.5)
        f.setWeight(QFont.Normal)
        p.setFont(f)
        p.setPen(QColor(t.ov_msg))
        msg = p.fontMetrics().elidedText(self.message, Qt.ElideRight, cw - 40)
        p.drawText(QRectF(card.left(), cy + 72, cw, 20), Qt.AlignCenter, msg)

        p.setPen(QColor(t.ov_time))
        p.drawText(QRectF(card.left(), cy + 94, cw, 20), Qt.AlignCenter,
                   f"{time.monotonic() - self._t0:.0f} s")
        p.end()

    # ── spinners ─────────────────────────────────────────────────────────────
    def _paint_rings(self, p: QPainter, t, cx: float, cy: float):
        """Three gradient arcs, each turning opposite to its neighbour."""
        for radius, width, color, speed, span in ((32, 6, QColor(t.accent) if t.styled else ACCENT, 1.0, 280),
                                                  (19, 4, ACCENT_2, -1.6, 200),
                                                  (9, 3, ACCENT_3, 2.4, 160)):
            ang = (self._angle * speed) % 360
            rect = QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius)
            # faint track
            p.setPen(QPen(QColor(t.ov_track), width))
            p.drawEllipse(rect)
            # arc covers angles [start - span, start]; the bright head leads the rotation
            start = -ang
            grad = QConicalGradient(cx, cy, start)
            clear = QColor(color)
            clear.setAlpha(0)
            p_far = 1 - span / 360          # gradient position of (start - span)
            if speed > 0:                   # clockwise: head at start - span
                grad.setColorAt(0.0, clear)
                grad.setColorAt(p_far, color)
                grad.setColorAt(1.0, clear)
                head = start - span
            else:                           # counter-clockwise: head at start
                grad.setColorAt(0.0, color)
                grad.setColorAt(p_far, clear)
                grad.setColorAt(1.0, color)
                head = start
            p.setPen(QPen(grad, width, Qt.SolidLine, Qt.FlatCap))
            p.drawArc(rect, int(start * 16), int(-span * 16))
            # glowing round head
            hx = cx + radius * cos(radians(head))
            hy = cy - radius * sin(radians(head))
            glow = QColor(color)
            glow.setAlpha(70)
            p.setPen(Qt.NoPen)
            p.setBrush(glow)
            p.drawEllipse(QRectF(hx - width, hy - width, 2 * width, 2 * width))
            p.setBrush(color)
            p.drawEllipse(QRectF(hx - width / 2, hy - width / 2, width, width))
            p.setBrush(Qt.NoBrush)

    def _grid_colours(self, t) -> list[QColor]:
        if t.steps:                                   # the theme's step colours (Nocturne)
            return [QColor(c) for c, _, _ in t.steps]
        return [QColor(t.accent) if t.styled else ACCENT, ACCENT_2, ACCENT_3, QColor(255, 181, 71)]

    def _paint_grid(self, p: QPainter, t, cx: float, cy: float):
        """3×3 tiles; each one flips over (a card turn about its horizontal axis) to a random colour."""
        now = time.monotonic()
        colours = self._grid_colours(t)
        if len(self._tiles) != GRID * GRID:
            self._tiles = [{"c": random.choice(colours + [None]), "prev": None, "t0": -1.0,
                            "next": now + random.uniform(0.05, 0.9)} for _ in range(GRID * GRID)]
        dim = QColor(t.line)
        span = GRID * TILE + (GRID - 1) * GAP
        x0, y0 = cx - span / 2, cy - span / 2
        p.setPen(Qt.NoPen)
        for k, tile in enumerate(self._tiles):
            if now >= tile["next"]:                   # start a flip to a different colour (or to "off")
                choices = [c for c in colours + [None, None] if c != tile["c"]]
                tile["prev"], tile["c"], tile["t0"] = tile["c"], random.choice(choices), now
                tile["next"] = now + FLIP_S + random.uniform(0.25, 1.1)
            f = (now - tile["t0"]) / FLIP_S
            if 0 <= f < 1:
                scale = abs(cos(pi * f))              # 1 → 0 (edge-on) → 1
                col = tile["prev"] if f < 0.5 else tile["c"]
            else:
                scale, col = 1.0, tile["c"]
            h = max(TILE * scale, 1.0)
            x = x0 + (k % GRID) * (TILE + GAP)
            y = y0 + (k // GRID) * (TILE + GAP) + (TILE - h) / 2
            p.setBrush(col if col is not None else dim)
            p.drawRect(QRectF(x, y, TILE, h))
        p.setBrush(Qt.NoBrush)

