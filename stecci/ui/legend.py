"""Class legend, rendered as an image in photo pixels.

It is placed inside the crop box (top-right corner), so it zooms with the photo on screen and appears
at the same spot in the exported overlay. Sizes below are for scale 1; they are multiplied by `scale`.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPen

PAD = 10
SWATCH = 12
ROW_GAP = 6
FONT_PX = 12
TITLE = "Legend"
VALUE_SAMPLE = "100.0 %"         # fixed value column, so the size does not change while assigning

Row = tuple  # (rgb, name, value_text, is_line)


def _fonts(scale: float):
    f = QFont()
    f.setPixelSize(max(1, round(FONT_PX * scale)))
    b = QFont(f)
    b.setBold(True)
    return f, b


def legend_size(rows: list[Row], scale: float) -> tuple[int, int]:
    f, b = _fonts(scale)
    fm, fb = QFontMetricsF(f), QFontMetricsF(b)
    s = scale
    name_w = max([fm.horizontalAdvance(r[1]) for r in rows] + [fb.horizontalAdvance(TITLE)])
    val_w = fm.horizontalAdvance(VALUE_SAMPLE)
    row_h = max(fm.height(), SWATCH * s)
    w = PAD * s + SWATCH * s + 8 * s + name_w + 14 * s + val_w + PAD * s
    h = PAD * s + fb.height() + 6 * s + len(rows) * (row_h + ROW_GAP * s) - ROW_GAP * s + PAD * s
    return int(np.ceil(w)), int(np.ceil(h))


def render_legend(rows: list[Row], scale: float, dark: bool = False) -> QImage:
    """Legend as a transparent ARGB image of legend_size(rows, scale).

    dark=True: identical layout with dark text and a light shadow, for white backgrounds (white exports).
    """
    w, h = legend_size(rows, scale)
    title_c, name_c, value_c = ((QColor(20, 22, 26), QColor(30, 32, 36), QColor(85, 90, 98)) if dark else
                                (QColor(245, 246, 248), QColor(240, 242, 245), QColor(200, 205, 212)))
    shadow_c = QColor(255, 255, 255, 170) if dark else QColor(0, 0, 0, 170)
    img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    s = scale
    f, b = _fonts(s)
    fm, fb = QFontMetricsF(f), QFontMetricsF(b)
    row_h = max(fm.height(), SWATCH * s)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    # no card background: text gets a soft dark shadow so it stays readable on bright parts of the photo
    def text(rect, flags, string, color):
        off = max(1.0, s)
        p.setPen(shadow_c)
        p.drawText(rect.translated(off, off), flags, string)
        p.setPen(color)
        p.drawText(rect, flags, string)

    p.setFont(b)
    text(QRectF(PAD * s, PAD * s, w, fb.height()), Qt.AlignLeft | Qt.AlignVCenter, TITLE, title_c)

    p.setFont(f)
    y = PAD * s + fb.height() + 6 * s
    for rgb, name, value, is_line in rows:
        cy = y + row_h / 2
        col = QColor(*rgb)
        sx = PAD * s
        if is_line:                                        # cracks: short thick line
            p.setPen(QPen(col, 3 * s, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(int(sx + s), int(cy), int(sx + SWATCH * s - s), int(cy))
        else:
            p.setPen(QPen(QColor(0, 0, 0, 140), max(1.0, s)))      # thin dark rim: visible on any background
            p.setBrush(col)
            p.drawRoundedRect(QRectF(sx, cy - SWATCH * s / 2, SWATCH * s, SWATCH * s), 3 * s, 3 * s)
        text(QRectF(sx + SWATCH * s + 8 * s, y, w, row_h), Qt.AlignLeft | Qt.AlignVCenter, name, name_c)
        if value:
            text(QRectF(0, y, w - PAD * s, row_h), Qt.AlignRight | Qt.AlignVCenter, value, value_c)
        y += row_h + ROW_GAP * s
    p.end()
    return img


def qimage_to_rgba(img: QImage) -> np.ndarray:
    """QImage -> straight (non-premultiplied) RGBA uint8 array."""
    img = img.convertToFormat(QImage.Format_RGBA8888)
    a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())
    return a[:, : img.width() * 4].reshape(img.height(), img.width(), 4).copy()
