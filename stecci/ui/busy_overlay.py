"""Modern animated busy overlay shown over the canvas while a model job runs."""
from __future__ import annotations

import time
from math import cos, radians, sin

from PySide6.QtCore import QEvent, QPropertyAnimation, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QConicalGradient, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget

from . import theme

ACCENT = QColor(64, 140, 255)
ACCENT_2 = QColor(56, 210, 180)
ACCENT_3 = QColor(255, 92, 92)


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

        # spinner: three gradient arcs, each turning opposite to its neighbour
        cx, cy = card.center().x(), card.top() + 72
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
