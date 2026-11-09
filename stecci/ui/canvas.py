"""Zoomable image canvas with overlay layers and tool-specific mouse handling."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem,
                               QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsView)

from . import theme

LAYERS = ("base", "fill", "boundary", "selection", "crack")
BRUSH_TOOLS = ("brush",)             # crack brush (the class brush is Shift+drag in the assign tool)
CRACK_ERASER_FACTOR = 5              # right-drag with the crack brush erases with 5× the brush size


def rgba_to_pixmap(a: np.ndarray) -> QPixmap:
    a = np.ascontiguousarray(a)
    h, w = a.shape[:2]
    if a.shape[2] == 3:
        img = QImage(a.data, w, h, 3 * w, QImage.Format_RGB888)
    else:
        img = QImage(a.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return QPixmap.fromImage(img.copy())


class Canvas(QGraphicsView):
    hovered = Signal(int, int)
    assign_press = Signal(int, int, object, bool)    # x, y, button, ctrl
    assign_drag = Signal(int, int, object)
    assign_release = Signal()
    rock_box = Signal(float, float, float, float)
    rock_point = Signal(float, float, bool)
    split_stroke = Signal(list)
    brush_stroke = Signal(list, int, bool)           # points, size, erase
    paint_stroke = Signal(list, int, bool)           # class brush (Shift+drag): points, size, right button
    paint_resized = Signal(int)                      # Shift+wheel changed the class brush size
    brush_resized = Signal(int)                      # Ctrl+wheel changed the brush size
    legend_moved = Signal(float, float)              # legend dropped at (x, y) top-left, photo px
    legend_dragging = Signal(float, float)           # legend is being dragged (live position)
    legend_wheel = Signal(int)                       # wheel over the legend: +1 bigger / -1 smaller
    legend_reset = Signal()                          # double-click on the legend

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        # zoom anchoring is done by hand in wheelEvent (Qt's AnchorUnderMouse does nothing while the
        # image is smaller than the view and is clamped at the scroll limits)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._apply_theme()
        theme.manager().changed.connect(self._apply_theme)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setDragMode(QGraphicsView.NoDrag)

        self.items: dict[str, QGraphicsPixmapItem] = {}
        for z, name in enumerate(LAYERS):
            it = QGraphicsPixmapItem()
            it.setZValue(z)
            if name == "base":
                it.setTransformationMode(Qt.SmoothTransformation)
            self.scene().addItem(it)
            self.items[name] = it

        self.tool = "assign"
        self.fit_box = None             # (x0, y0, x1, y1): Fit shows only this part (crop preview)
        self.brush_size = 6
        self.paint_size = 50                            # class brush (Shift+drag)
        self._painting = False                          # a Shift+drag class stroke is in progress
        self._erasing = False           # right button held with the crack brush
        self.crack_color = QColor(230, 30, 30, 200)
        self.paint_color = QColor(60, 175, 70, 170)     # active class (class brush)
        self.img_w = self.img_h = 0
        self._pan_last = None
        self._space = False
        self._press_pos: QPointF | None = None
        self._press_button = None
        self._stroke: list[tuple[float, float]] = []

        # temporary items
        self._rect = QGraphicsRectItem()
        pen = QPen(QColor(255, 200, 0), 2, Qt.DashLine)
        pen.setCosmetic(True)
        self._rect.setPen(pen)
        self._rect.setZValue(20)
        self._rect.hide()
        self.scene().addItem(self._rect)

        self._path = QGraphicsPathItem()
        self._path.setZValue(21)
        self._path.hide()
        self.scene().addItem(self._path)

        self._cursor = QGraphicsEllipseItem()
        cpen = QPen(QColor(255, 255, 255), 1)
        cpen.setCosmetic(True)
        self._cursor.setPen(cpen)
        self._cursor.setZValue(30)
        self._cursor.hide()
        self.scene().addItem(self._cursor)

        self._prompt_items: list[QGraphicsItem] = []

        self._legend = QGraphicsPixmapItem()          # legend card, placed in photo coordinates
        self._legend.setZValue(18)
        self._legend.setTransformationMode(Qt.SmoothTransformation)
        self._legend.hide()
        self.scene().addItem(self._legend)
        self._legend_drag = None                      # offset between cursor and legend corner while dragging

    # ── layers ───────────────────────────────────────────────────────────────
    def _apply_theme(self):
        self.setBackgroundBrush(QColor(theme.current().canvas))

    def set_image(self, rgb: np.ndarray):
        self.img_h, self.img_w = rgb.shape[:2]
        self.items["base"].setPixmap(rgba_to_pixmap(rgb))
        for n in LAYERS[1:]:
            self.items[n].setPixmap(QPixmap())
        # generous margin around the photo so any point can be brought under the cursor
        mx, my = max(self.img_w, 2000), max(self.img_h, 2000)
        self.scene().setSceneRect(QRectF(-mx, -my, self.img_w + 2 * mx, self.img_h + 2 * my))
        self.fit()

    def set_layer(self, name: str, rgba: np.ndarray | None):
        self.items[name].setPixmap(QPixmap() if rgba is None else rgba_to_pixmap(rgba))

    def set_layer_visible(self, name: str, visible: bool):
        self.items[name].setVisible(visible)

    def fit(self):
        if self.img_w:
            r = QRectF(0, 0, self.img_w, self.img_h) if self.fit_box is None else \
                QRectF(QPointF(self.fit_box[0], self.fit_box[1]), QPointF(self.fit_box[2], self.fit_box[3]))
            self.fitInView(r, Qt.KeepAspectRatio)

    def set_tool(self, tool: str):
        self.tool = tool
        self._cursor.setVisible(tool in BRUSH_TOOLS)
        self._update_cursor_item(None)
        self.viewport().setCursor(Qt.CrossCursor if tool in ("rock", "split", *BRUSH_TOOLS) else Qt.PointingHandCursor)

    def set_legend(self, image, pos=None, scale: float = 1.0):
        """image: QImage rendered at `1/scale` × its size in photo px, or None to hide."""
        if image is None:
            self._legend.hide()
            return
        self._legend.setPixmap(QPixmap.fromImage(image))
        self._legend.setScale(scale)
        self._legend.setPos(pos[0], pos[1])
        self._legend.show()

    def set_rock_prompts(self, box, points, color=(255, 20, 200), style: str = "solid"):
        for it in self._prompt_items:
            self.scene().removeItem(it)
        self._prompt_items.clear()
        if box is not None:
            r = QGraphicsRectItem(QRectF(QPointF(box[0], box[1]), QPointF(box[2], box[3])))
            pen_style = {"dashed": Qt.DashLine, "dotted": Qt.DotLine}.get(style, Qt.SolidLine)
            pen = QPen(QColor(*color), 2, pen_style)
            pen.setCosmetic(True)               # same on-screen width at any zoom
            r.setPen(pen)
            r.setZValue(19)
            self.scene().addItem(r)
            self._prompt_items.append(r)
        for x, y, pos in points:
            e = QGraphicsEllipseItem(-6, -6, 12, 12)
            e.setPos(x, y)
            e.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            e.setBrush(QBrush(QColor(40, 220, 40) if pos else QColor(230, 40, 40)))
            e.setPen(QPen(Qt.white, 1.5))
            e.setZValue(25)
            self.scene().addItem(e)
            self._prompt_items.append(e)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _inside(self, p: QPointF) -> bool:
        return 0 <= p.x() < self.img_w and 0 <= p.y() < self.img_h

    def _clamp(self, p: QPointF) -> QPointF:
        return QPointF(min(max(p.x(), 0), self.img_w - 1), min(max(p.y(), 0), self.img_h - 1))

    def _shift_brush(self) -> bool:
        """Assign tool with Shift held (or a Shift stroke running): the class brush is active."""
        return self.tool == "assign" and (self._painting or
                                          bool(QApplication.keyboardModifiers() & Qt.ShiftModifier))

    def _update_cursor_item(self, p: QPointF | None):
        shift = self._shift_brush()
        if (self.tool not in BRUSH_TOOLS and not shift) or p is None:
            self._cursor.setVisible(False)
            return
        r = (self.paint_size if shift else self._stroke_size()) / 2
        self._cursor.setRect(p.x() - r, p.y() - r, 2 * r, 2 * r)
        self._cursor.setVisible(True)

    def _stroke_size(self) -> int:
        """Current stroke width: the crack eraser is CRACK_ERASER_FACTOR × the crack brush."""
        return self.brush_size * CRACK_ERASER_FACTOR if self._erasing else self.brush_size

    def _start_path(self, p: QPointF, color: QColor, width: float):
        pen = QPen(color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        if width <= 0:
            pen.setWidth(2)
            pen.setCosmetic(True)
        self._path.setPen(pen)
        path = QPainterPath(p)
        path.lineTo(p + QPointF(0.01, 0.01))
        self._path.setPath(path)
        self._path.show()

    # ── events ───────────────────────────────────────────────────────────────
    def _over_legend(self, scene_pt) -> bool:
        return self._legend.isVisible() and self._legend.sceneBoundingRect().contains(scene_pt)

    def wheelEvent(self, e):
        if not self.img_w:
            return
        if not e.modifiers() and self._over_legend(self.mapToScene(e.position().toPoint())):
            self.legend_wheel.emit(1 if e.angleDelta().y() > 0 else -1)
            return
        if e.modifiers() & Qt.ShiftModifier and self.tool == "assign":
            d = e.angleDelta().y() or e.angleDelta().x()          # Shift+wheel may arrive as horizontal
            self.paint_size = int(np.clip(self.paint_size + (2 if d > 0 else -2), 1, 400))
            self.paint_resized.emit(self.paint_size)
            self._update_cursor_item(self.mapToScene(e.position().toPoint()))
            return
        if e.modifiers() & Qt.ControlModifier and self.tool in BRUSH_TOOLS:
            self.brush_size = int(np.clip(self.brush_size + (1 if e.angleDelta().y() > 0 else -1), 1, 200))
            self.brush_resized.emit(self.brush_size)
            self._update_cursor_item(self.mapToScene(e.position().toPoint()))
            return
        f = 1.25 ** (e.angleDelta().y() / 120)
        cur = self.transform().m11()
        if (cur * f > 40 and f > 1) or (cur * f < 0.02 and f < 1):
            return
        pos = e.position().toPoint()
        before = self.mapToScene(pos)
        self.scale(f, f)
        after = self.mapToScene(pos)
        d = after - before                       # keep the point under the cursor fixed
        self.translate(d.x(), d.y())
        self._update_cursor_item(self.mapToScene(pos))

    def _cursor_scene_pos(self):
        vp = self.viewport().mapFromGlobal(QCursor.pos())
        return self.mapToScene(vp) if self.viewport().rect().contains(vp) else None

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Shift:
            self._update_cursor_item(self._cursor_scene_pos())
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Shift:
            QTimer.singleShot(0, lambda: self._update_cursor_item(self._cursor_scene_pos()))
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self.set_tool(self.tool)
            return
        super().keyReleaseEvent(e)

    def mousePressEvent(self, e):
        if not self.img_w:
            return
        if e.button() == Qt.MiddleButton or (self._space and e.button() == Qt.LeftButton):
            self._pan_last = e.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        p = self.mapToScene(e.position().toPoint())
        if e.button() == Qt.LeftButton and self._over_legend(p):
            self._legend_drag = p - self._legend.pos()
            self._legend_start = self._legend.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        self._press_pos = p
        self._press_button = e.button()
        if self.tool == "assign":
            if e.modifiers() & Qt.ShiftModifier and e.button() in (Qt.LeftButton, Qt.RightButton):
                self._painting = True                   # class brush: left = active class, right = stone
                self._stroke = [(p.x(), p.y())]
                c = self.paint_color if e.button() == Qt.LeftButton else QColor(160, 160, 160, 170)
                self._start_path(p, c, self.paint_size)
            elif self._inside(p):
                ctrl = bool(e.modifiers() & Qt.ControlModifier)
                self.assign_press.emit(int(p.x()), int(p.y()), e.button(), ctrl)
        elif self.tool == "rock":
            self._rect.setRect(QRectF(p, p))
            self._rect.show()
        elif self.tool == "split" and e.button() == Qt.LeftButton:
            self._stroke = [(p.x(), p.y())]
            self._start_path(p, QColor(0, 230, 255), 0)
        elif self.tool == "brush" and e.button() in (Qt.LeftButton, Qt.RightButton):
            self._stroke = [(p.x(), p.y())]
            self._erasing = e.button() == Qt.RightButton
            c = self.crack_color if not self._erasing else QColor(255, 255, 255, 160)
            self._start_path(p, c, self._stroke_size())
            self._update_cursor_item(p)

    def mouseMoveEvent(self, e):
        if self._pan_last is not None:
            d = e.position() - self._pan_last
            self._pan_last = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if not self.img_w:
            return
        p = self.mapToScene(e.position().toPoint())
        if self._legend_drag is not None:
            self._legend.setPos(p - self._legend_drag)
            pos = self._legend.pos()
            self.legend_dragging.emit(pos.x(), pos.y())
            return
        if self._press_pos is None:                   # hover feedback: the legend can be grabbed
            over = self._over_legend(p)
            if over != getattr(self, "_hover_legend", False):
                self._hover_legend = over
                if over:
                    self.viewport().setCursor(Qt.OpenHandCursor)
                else:
                    self.set_tool(self.tool)
            if over:
                self._cursor.setVisible(False)
                return
        self._update_cursor_item(p)
        if self._inside(p):
            self.hovered.emit(int(p.x()), int(p.y()))
        if self._press_pos is None:
            return
        if self.tool == "assign" and self._painting:
            self._stroke.append((p.x(), p.y()))
            path = self._path.path()
            path.lineTo(p)
            self._path.setPath(path)
        elif self.tool == "assign":
            if self._inside(p) and not (e.modifiers() & Qt.ShiftModifier):
                self.assign_drag.emit(int(p.x()), int(p.y()), self._press_button)
        elif self.tool == "rock":
            self._rect.setRect(QRectF(self._press_pos, p).normalized())
        elif self.tool in ("split", *BRUSH_TOOLS) and self._stroke:
            self._stroke.append((p.x(), p.y()))
            path = self._path.path()
            path.lineTo(p)
            self._path.setPath(path)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton and self._over_legend(self.mapToScene(e.position().toPoint())):
            self._legend_drag = None
            self.legend_reset.emit()
            return
        super().mouseDoubleClickEvent(e)

    def mouseReleaseEvent(self, e):
        if self._legend_drag is not None and e.button() == Qt.LeftButton:
            self._legend_drag = None
            self.viewport().setCursor(Qt.OpenHandCursor)
            pos = self._legend.pos()
            if (pos - self._legend_start).manhattanLength() > 1:      # a plain click is not a move
                self.legend_moved.emit(pos.x(), pos.y())
            return
        if self._pan_last is not None and e.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._pan_last = None
            self.set_tool(self.tool)
            if self._space:
                self.viewport().setCursor(Qt.OpenHandCursor)
            return
        if self._press_pos is None:
            return
        p = self.mapToScene(e.position().toPoint())
        start, button = self._press_pos, self._press_button
        self._press_pos = None
        if self.tool == "assign" and self._painting:
            pts, self._stroke = self._stroke, []
            self._painting = False
            self._path.hide()
            self.paint_stroke.emit(pts, self.paint_size, button == Qt.RightButton)
            self._update_cursor_item(p)
        elif self.tool == "assign":
            self.assign_release.emit()
        elif self.tool == "rock":
            self._rect.hide()
            r = QRectF(self._clamp(start), self._clamp(p)).normalized()
            if r.width() > 8 and r.height() > 8:
                self.rock_box.emit(r.left(), r.top(), r.right(), r.bottom())
            elif self._inside(p):
                self.rock_point.emit(p.x(), p.y(), button == Qt.LeftButton)
        elif self.tool in ("split", *BRUSH_TOOLS) and self._stroke:
            pts, self._stroke = self._stroke, []
            self._path.hide()
            if self.tool == "split":
                if len(pts) > 1:
                    self.split_stroke.emit(pts)
            else:
                size = self._stroke_size()
                self._erasing = False
                self._update_cursor_item(p)
                self.brush_stroke.emit(pts, size, button == Qt.RightButton)

    def leaveEvent(self, e):
        self._cursor.setVisible(False)
        super().leaveEvent(e)
