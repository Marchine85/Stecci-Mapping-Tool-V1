"""3D-scan workspace: same workflow as the photo workspace, on a textured 3D scan, areas measured in 3D."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QActionGroup, QColor, QPainter, QPixmap
from PySide6.QtWidgets import (QColorDialog, QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QMenu,
                               QMessageBox, QPushButton, QSlider)

from ..config import GRANULARITY, PROJECT_EXT_3D, SCAN_EXTS, UNWRAP_METHODS, granularity
from ..scan3d import surface as SF
from ..scan3d.isolate import file_up, ground_plane, isolate
from ..scan3d.mesh import load_scan
from ..scan3d.project import Project
from .export_dialog import ExportDialog
from .legend import qimage_to_rgba, render_legend
from .viewer import Viewer
from .widgets import Section, capitalize_words, color_icon, set_role
from .workspace import Workspace

SCAN_FILTER = "3D scans (" + " ".join(f"*{e}" for e in SCAN_EXTS) + ")"
PROJECT_FILTER = f"Stecci 3D project (*{PROJECT_EXT_3D})"
NAV_HINT = ("Middle-Drag / Alt+Drag / Drag Beside The Stone: Rotate · Shift+Middle-Drag: Pan · Wheel: Zoom · "
            "Double-Click: Rotate Around Point")
BRUSH_LIMITS = {"crack": (5, 1, 30), "paint": (50, 6, 200)}
DEFAULT_DISPLAY = {"boundary_color": [57, 255, 20], "boundary_thickness": 2, "outline_color": [255, 20, 200],
                   "crack_color": [230, 30, 30], "dim_strength": 80, "shading": 35}

HELP_HTML = """
<h3>3D-scan workflow</h3>
<ol>
<li><b>Open</b> a textured 3D scan (Ctrl+O; .glb, .gltf, .obj, .ply), then press <b>Detect and Isolate Stećak</b> (Ctrl+D):
the ground plane is found from the scan's outer edge and everything above it is the stone.</li>
<li>If needed, <b>Refine Isolation</b> (R): left-drag adds surface to the stone, right-drag removes it (e.g. plants at the base).
The ground cut height and the <b>scale</b> (true stone length → areas in m²) are in the card's ••• menu.</li>
<li><b>Segmentation</b>: choose the <b>unwrap</b> (ARAP, box net or belt) and the <b>fidelity</b>, press <b>Segment</b> (Ctrl+G).
The stone is flattened and segmented in the background with the photo method; only the outlines appear on the model.
<b>Split Segments Manually</b> (X): draw across a segment.</li>
<li><b>Assign classes</b>: pick a class (keys 0–9), click or drag over segments. Right-click = unassign.
Ctrl+click / Ctrl+drag = select, <b>M</b> = merge. <b>Shift+drag</b> brushes the active class, Shift+right-drag brushes Unassigned/Stone,
Shift+wheel = brush size.</li>
<li><b>Draw cracks</b> (B): left-drag paints, right-drag erases. Ctrl+wheel = brush size.</li>
<li><b>Export</b> (Ctrl+E): <i>name</i>.png (the current view), <i>name</i>_views.png (six views), <i>name</i>.glb (the stone with
the mapping in its texture) and <i>name</i>.json (areas per class measured on the 3D surface).</li>
</ol>
<h3>Navigation</h3>
Middle-drag, Alt+drag or drag beside the stone = rotate · Shift+middle-drag = pan · wheel = zoom ·
double-click = rotate around that point · F = fit · Top / Front / Back / Left / Right buttons in Annotation Settings
"""


class ScanWorkspace(Workspace):
    KIND = "3d"
    MODE_LABEL = "3D SCAN"
    HELP_HTML = HELP_HTML
    PANEL_MIN_WIDTH = 330

    def __init__(self, app):
        super().__init__(app)
        self.display = self._load_display("display3d", DEFAULT_DISPLAY)
        self.project = Project()
        self.hover_seg = 0
        self.tree = None                       # KD-tree of face centroids (brushes)
        self._unwrap_cache = None              # (method, stone array, max side, Unwrap)

        self.viewer = Viewer(self)
        self.viewer.installEventFilter(self)
        self.legend_label = QLabel(self.viewer)
        self.legend_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.legend_label.hide()

        self._build_actions()
        self._build_menus()
        self._build_panel()
        self._connect_viewer()
        self._apply_display()
        self._refresh_classes()
        self._update_enabled()

    def eventFilter(self, obj, ev):
        if obj is self.viewer and ev.type() == QEvent.Resize:
            self._place_legend()
        return super().eventFilter(obj, ev)

    # ════════════════════════════════════════════════════════════════════════
    # UI construction
    # ════════════════════════════════════════════════════════════════════════
    def _build_actions(self):
        A = self._act
        self.a_detect = A("Detect and Isolate Stećak", self.detect_stone, "Ctrl+D")
        self.a_cut = A("Ground Cut Height…", self.set_cut_height)
        self.a_scale = A("Set Scale (True Stone Length)…", self.set_scale)
        self.a_outline_color = A("Outline Colour…", self.pick_outline_color)

        self.a_segment = A("Segment", self.auto_segment, "Ctrl+G")
        self.a_clear_seg = A("Clear Segmentation…", self.clear_segmentation)
        self.keep_cb = A("Keep Class Assignments When Re-Segmenting", checkable=True, checked=True)
        self.a_bound_color = A("Boundary Colour…", self.pick_boundary_color)
        self.a_gran_info = A("")
        self.a_gran_info.setEnabled(False)

        self.a_add_class = A("Add Class…", self.add_class)
        self.a_rename_class = A("Rename Active Class…", self.rename_class)
        self.a_color_class = A("Change Colour Of Active Class…", self.recolor_class)
        self.a_delete_class = A("Delete Active Class", self.delete_class)
        self.a_assign_sel = A("Assign Active Class To Selection", self.assign_selection)
        self.a_merge = A("Merge Selected Segments", self.merge_selected, "M")
        self.a_clear_sel = A("Clear Selection", self.clear_selection, "Esc")
        self.a_clear_assign = A("Clear All Assignments…", self.clear_assignments)

        self.a_brush_smaller = A("Smaller Brush", lambda: self._brush_step(-1), "[")
        self.a_brush_larger = A("Larger Brush", lambda: self._brush_step(1), "]")
        self.a_clear_cracks = A("Clear All Cracks", self.clear_cracks)
        self.a_crack_color = A("Crack Colour…", self.pick_crack_color)

        self.a_fit = A("Fit to window", self.fit_view, "F")
        self.show_fill = A("Class colours", checkable=True, checked=True)
        self.show_bound = A("Segment boundaries", checkable=True, checked=True)
        self.show_crack = A("Cracks", checkable=True, checked=True)
        self.dim_bg = A("Dim background", checkable=True, checked=True)
        self.show_legend = A("Legend", checkable=True, checked=True)
        self.show_shade = A("Shading", checkable=True, checked=True)
        for a in (self.show_fill, self.show_bound, self.show_crack, self.dim_bg, self.show_shade):
            a.toggled.connect(lambda _: self._apply_display())
        self.show_legend.toggled.connect(lambda _: self._refresh_legend())

        self.tool_group = QActionGroup(self)
        self.tool_actions = {}
        for key, text, sc in [("assign", "Assign Tool", "A"), ("refine", "Refine Isolation Tool", "R"),
                              ("split", "Split Segments Manually", "X"), ("crack", "Crack Brush", "B")]:
            a = A(text, lambda k=key: self.set_tool(k), sc, checkable=True)
            self.tool_group.addAction(a)
            self.tool_actions[key] = a
        self.tool_actions["assign"].setChecked(True)
        self._class_shortcuts()

    def _build_menus(self):
        self.menu_detect = QMenu("&Detect", self)
        for a in (self.a_detect, self.tool_actions["refine"], None, self.a_cut, self.a_scale, None,
                  self.a_outline_color):
            self.menu_detect.addSeparator() if a is None else self.menu_detect.addAction(a)

        self.menu_seg = QMenu("&Segmentation", self)
        for a in (self.a_segment, self.tool_actions["split"], None, self.a_clear_seg, None, self.keep_cb, None,
                  self.a_bound_color):
            self.menu_seg.addSeparator() if a is None else self.menu_seg.addAction(a)
        self.bound_thick = QSlider(Qt.Horizontal)
        self.bound_thick.setRange(1, 6)
        self.bound_thick.setValue(int(self.display["boundary_thickness"]))
        self.bound_thick_label = QLabel()
        self.bound_thick.valueChanged.connect(self._on_boundary_thickness)
        self.menu_seg.addAction(self._slider_action("Boundary Thickness", self.bound_thick, self.bound_thick_label))
        self.bound_thick_label.setText(f"{self.bound_thick.value()} px")
        self.menu_seg.addSeparator()
        self.menu_seg.addAction(self.a_gran_info)

        self.menu_classes = QMenu("&Classes", self)
        for a in (self.tool_actions["assign"], None, self.a_add_class, self.a_rename_class, self.a_color_class,
                  self.a_delete_class, None, self.a_assign_sel, self.a_merge, self.a_clear_sel, None,
                  self.a_clear_assign):
            self.menu_classes.addSeparator() if a is None else self.menu_classes.addAction(a)

        self.menu_cracks = QMenu("C&racks", self)
        self.menu_cracks.addAction(self.a_crack_color)
        self.crack_slider = QSlider(Qt.Horizontal)
        self.crack_slider.setRange(*BRUSH_LIMITS["crack"][1:])
        self.crack_slider.valueChanged.connect(lambda v: self.set_brush(v, "crack"))
        self.crack_size_label = QLabel()
        self.menu_cracks.addAction(self._slider_action("Brush Size", self.crack_slider, self.crack_size_label))
        self.menu_cracks.addSeparator()
        self.menu_cracks.addAction(self.a_clear_cracks)

    def _build_panel(self):
        # 1 · Detect
        self.sec_detect = Section(1, "Detect", self.menu_detect)
        self.btn_detect = self._big_button("Detect and Isolate Stećak", self.detect_stone,
                                           "Find the ground plane and cut the stone from it  ·  Ctrl+D")
        self.sec_detect.body.addWidget(self.btn_detect)
        self.btn_refine = self._link_toggle("✎  Refine Isolation", "refine",
                                            "Left-drag adds surface to the stone, right-drag removes it  ·  R")
        self.sec_detect.body.addWidget(self.btn_refine)
        self.stone_info = self._hint("")
        self.sec_detect.body.addWidget(self.stone_info)

        # 2 · Segmentation
        self.sec_seg = Section(2, "Segmentation", self.menu_seg)
        row = QHBoxLayout()
        row.addWidget(QLabel("Unwrap"))
        self.unwrap_combo = QComboBox()
        for value, label in UNWRAP_METHODS:
            self.unwrap_combo.addItem(label.split("  ·  ")[0], value)
            self.unwrap_combo.setItemData(self.unwrap_combo.count() - 1, label, Qt.ToolTipRole)
        self.unwrap_combo.setCurrentIndex(max(0, self.unwrap_combo.findData(self.settings.unwrap_method)))
        self.unwrap_combo.setToolTip("How the stone is flattened for the segmentation (in the background):\n"
                                     + "\n".join(lab for _, lab in UNWRAP_METHODS))
        self.unwrap_combo.currentIndexChanged.connect(self._on_unwrap_changed)
        row.addWidget(self.unwrap_combo, 1)
        self.sec_seg.body.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Fidelity"))
        self.gran = QSlider(Qt.Horizontal)
        self.gran.setRange(0, 10)
        self.gran.setValue(self.project.granularity)
        self.gran.setPageStep(1)
        row.addWidget(self.gran, 1)
        self.gran_value = QLabel()
        self.gran_value.setFixedWidth(26)
        self.gran_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.gran_value)
        self.sec_seg.body.addLayout(row)
        self.gran.valueChanged.connect(self._update_gran_label)
        self.btn_seg = self._big_button("Segment", self.auto_segment,
                                        "Unwrap the stone, segment it with SAM and map the segments onto the model"
                                        "  ·  Ctrl+G")
        self.sec_seg.body.addWidget(self.btn_seg)
        self.btn_split = self._link_toggle("✂  Split Segments Manually", "split",
                                           "Draw a line across a segment to cut it  ·  X")
        self.sec_seg.body.addWidget(self.btn_split)
        self._update_gran_label()

        # 3 · Assign classes
        self.sec_classes = self._build_classes_section(
            self.menu_classes, BRUSH_LIMITS["paint"][1:],
            "Class brush: hold Shift and drag over the stone  ·  Shift+wheel = size (screen pixels)")

        # 4 · Draw cracks
        self.sec_cracks = Section(4, "Draw Cracks", self.menu_cracks)
        self.btn_crack = QPushButton("Draw Cracks")
        self.btn_crack.setCheckable(True)
        self.btn_crack.setMinimumHeight(40)
        self.btn_crack.setCursor(Qt.PointingHandCursor)
        set_role(self.btn_crack, "secondary")
        self.btn_crack.setToolTip("Left-drag paints, right-drag erases  ·  B")
        self.btn_crack.toggled.connect(lambda on: self.set_tool("crack" if on else "assign"))
        self.sec_cracks.body.addWidget(self.btn_crack)
        self.sec_cracks.body.addWidget(self._hint("Left-Drag: Paint · Right-Drag: Erase · Ctrl+Wheel: Brush Size"))
        self._update_crack_swatch()
        self.set_brush(BRUSH_LIMITS["crack"][0], "crack")
        self.set_brush(BRUSH_LIMITS["paint"][0], "paint")

        # Annotation settings
        self.sec_view = Section(None, "Annotation Settings")
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(50)
        self.opacity.setToolTip("Opacity of the class colours")
        self.opacity.valueChanged.connect(lambda _: self._apply_display())
        self.dim_slider = QSlider(Qt.Horizontal)
        self.dim_slider.setRange(0, 100)
        self.dim_slider.setValue(int(self.display["dim_strength"]))
        self.dim_slider.setToolTip("How strongly the ground around the stećak is darkened")
        self.dim_slider.valueChanged.connect(self._on_dim_strength)
        self.shade_slider = QSlider(Qt.Horizontal)
        self.shade_slider.setRange(0, 100)
        self.shade_slider.setValue(int(self.display["shading"]))
        self.shade_slider.setToolTip("Light shading from the viewing direction (shows the relief)")
        self.shade_slider.valueChanged.connect(self._on_shading)
        self.legend_size = QSlider(Qt.Horizontal)
        self.legend_size.setRange(50, 200)
        self.legend_size.setSingleStep(10)
        self.legend_size.setValue(100)
        self.legend_size.valueChanged.connect(self._on_legend_size)
        grid, _ = self._annotation_grid((
            (self.show_fill, "Class Colours", self.opacity, None),
            (self.show_bound, "Boundaries", None, "Segment Boundaries"),
            (self.show_crack, "Cracks", None, None),
            (self.dim_bg, "Dim Background", self.dim_slider, None),
            (self.show_shade, "Shading", self.shade_slider, None),
            (self.show_legend, "Legend", self.legend_size, None)))
        self.sec_view.body.addLayout(grid)
        views = QHBoxLayout()
        views.setSpacing(4)
        for name in ("Top", "Front", "Back", "Left", "Right"):
            b = QPushButton(name)
            b.setObjectName("View")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, n=name.lower(): self.viewer.view_preset(n))
            views.addWidget(b)
        self.sec_view.body.addLayout(views)
        self.sec_view.body.addWidget(self._link_button("⤢  Fit To Window", self.fit_view, "F"))
        self.sec_view.body.addWidget(self._hint(NAV_HINT))

        self._install(self.viewer, [self.sec_detect, self.sec_seg, self.sec_classes, self.sec_cracks],
                      self.sec_view)

    def _connect_viewer(self):
        v = self.viewer
        v.hovered.connect(self._on_hover)
        v.face_press.connect(lambda face, button, ctrl: self._assign_press(
            self._segment_of(face), button == Qt.LeftButton, ctrl))
        v.face_drag.connect(lambda face, buttons: self._assign_drag(self._segment_of(face),
                                                                    bool(buttons & Qt.LeftButton)))
        v.face_release.connect(self._assign_release)
        v.background_click.connect(lambda: self.selected and self.clear_selection())
        v.stroke.connect(self._on_stroke)
        v.brush_wheel.connect(lambda kind, d: self._brush_step(d, kind))

    # ════════════════════════════════════════════════════════════════════════
    # main-window interface
    # ════════════════════════════════════════════════════════════════════════
    def has_document(self) -> bool:
        return self.project.has_scan

    def can_export(self) -> bool:
        return self.project.has_scan and self.project.iso is not None

    def on_busy(self, busy: bool):
        self.viewer.setEnabled(not busy)
        super().on_busy(busy)

    def close_document(self):
        classes = self.project.classes
        self.project = Project()
        self.project.classes = classes
        self.project_path = None
        self.tree = None
        self._unwrap_cache = None
        self.selected.clear()
        self.hover_seg = 0
        self.viewer.clear()
        self.legend_label.hide()
        self.refresh_all()

    def settings_changed(self, old, new):
        if new.unwrap_max_side != old.unwrap_max_side:
            self._unwrap_cache = None
        self.unwrap_combo.blockSignals(True)
        self.unwrap_combo.setCurrentIndex(max(0, self.unwrap_combo.findData(new.unwrap_method)))
        self.unwrap_combo.blockSignals(False)
        self._update_gran_label()

    # ════════════════════════════════════════════════════════════════════════
    # helpers
    # ════════════════════════════════════════════════════════════════════════
    def _granularity(self, level: int):
        return granularity(level, self.settings.seg_mode, self.settings.colour_points)

    def _update_gran_label(self):
        lvl = self.gran.value()
        if lvl == 0:
            self.gran_value.setText("Off")
            self.a_gran_info.setText("Fidelity 0: Segmentation Off, The Whole Stone Is One Segment")
            self.gran.setToolTip(self.a_gran_info.text() + " (Shift+drag to brush classes)")
            self.btn_seg.setText("Use Whole Stone")
            return
        self.btn_seg.setText("Segment")
        g = self._granularity(lvl)
        self.gran_value.setText(str(lvl))
        extra = " + Zoomed Pass" if g.crop_layers else ""
        self.a_gran_info.setText(f"Fidelity {lvl}: {g.points_per_side}×{g.points_per_side} SAM Points{extra}, "
                                 f"Min Segment {100 * g.min_frac:.2f} % Of Stone")
        if self.settings.seg_mode == "colour":
            self.a_gran_info.setText(self.a_gran_info.text() + ", Detail From Colour")
        self.gran.setToolTip(self.a_gran_info.text())

    def _on_unwrap_changed(self):
        self.settings.unwrap_method = self.unwrap_combo.currentData()
        self._save_settings()

    def _update_enabled(self):
        p = self.project
        idle = not self.runner.busy
        has, stone, seg = p.has_scan, p.iso is not None, p.seg is not None
        self.a_detect.setEnabled(has and idle)
        self.btn_detect.setEnabled(has and idle)
        self.btn_refine.setEnabled(has and idle and stone)
        self.tool_actions["refine"].setEnabled(has and stone)
        for a in (self.a_cut, self.a_scale):
            a.setEnabled(has and stone and idle)
        self.sec_seg.setEnabled(stone)
        self.a_segment.setEnabled(stone and idle)
        self.btn_seg.setEnabled(stone and idle)
        self.btn_split.setEnabled(seg)
        self.tool_actions["split"].setEnabled(seg)
        self.a_clear_seg.setEnabled(seg and idle)
        self.sec_classes.setEnabled(seg)
        self.sec_cracks.setEnabled(stone)
        self.tool_actions["crack"].setEnabled(stone)
        next_step = self.btn_detect if not stone else (self.btn_seg if not seg else None)
        for b in (self.btn_detect, self.btn_seg):
            set_role(b, "primary" if b is next_step else "secondary")
        self.sec_detect.set_active(has)
        self.sec_seg.set_active(stone)
        self.sec_classes.set_active(seg)
        self.sec_cracks.set_active(stone)
        self.sec_view.set_active(has)
        self.app.update_file_actions()

    def _segments(self):
        return self.project.seg

    def current_tool(self) -> str:
        return self.viewer.tool

    def set_tool(self, tool: str):
        if tool in ("split",) and self.project.seg is None:
            tool = "assign"
        if tool in ("refine", "crack") and self.project.iso is None:
            tool = "assign"
        self.viewer.tool = tool
        self.tool_actions[tool].setChecked(True)
        for b, t in ((self.btn_refine, "refine"), (self.btn_split, "split"), (self.btn_crack, "crack")):
            b.blockSignals(True)
            b.setChecked(tool == t)
            b.blockSignals(False)
        self.btn_crack.setText("Stop Drawing Cracks" if tool == "crack" else "Draw Cracks")
        self.viewer.setCursor(Qt.CrossCursor if tool != "assign" else Qt.ArrowCursor)
        self.viewer.update()
        hints = {"refine": "Refine isolation: left-drag adds surface to the stone, right-drag removes it.",
                 "split": "Split: draw a line across a segment.",
                 "crack": "Cracks: left-drag paints, right-drag erases.",
                 "assign": "Click segments to assign the active class."}
        if self.project.has_scan:
            self.status_msg.setText(hints[tool])
        self.viewer.setFocus()

    def set_brush(self, size: int, which: str | None = None):
        which = which or "crack"
        lo, hi = BRUSH_LIMITS[which][1:]
        size = int(np.clip(size, lo, hi))
        self.viewer.brush_px[which] = size
        slider, label = ((self.paint_slider, self.paint_label) if which == "paint"
                         else (self.crack_slider, self.crack_size_label))
        slider.blockSignals(True)
        slider.setValue(size)
        slider.blockSignals(False)
        label.setText(f"{size} px")
        self.viewer.update()

    def _brush_step(self, d: int, which: str | None = None):
        which = which or ("crack" if self.viewer.tool == "crack" else "paint")
        size = self.viewer.brush_px[which]
        step = max(1, round(size * 0.12)) if which == "paint" else 1
        self.set_brush(size + d * step, which)

    def _apply_display(self):
        o = self.viewer.opts
        o.update(opacity=self.opacity.value() / 100 if hasattr(self, "opacity") else 0.5,
                 show_fill=self.show_fill.isChecked(), show_bound=self.show_bound.isChecked(),
                 show_crack=self.show_crack.isChecked(),
                 dim=self.display["dim_strength"] / 100 if self.dim_bg.isChecked() else 0.0,
                 shade=self.display["shading"] / 100 if self.show_shade.isChecked() else 0.0,
                 crack_color=tuple(self.display["crack_color"]), bound_color=tuple(self.display["boundary_color"]),
                 outline_color=tuple(self.display["outline_color"]),
                 bound_width=float(self.display["boundary_thickness"]))
        self.viewer.update()

    def _pick_colour(self, key: str, title: str):
        col = QColorDialog.getColor(QColor(*self.display[key]), self, title)
        if col.isValid():
            self.display[key] = [col.red(), col.green(), col.blue()]
            self._save_display()
            self._apply_display()
            self._update_crack_swatch()
            self._refresh_legend()

    def pick_outline_color(self):
        self._pick_colour("outline_color", "Outline colour")

    def pick_boundary_color(self):
        self._pick_colour("boundary_color", "Boundary colour")

    def pick_crack_color(self):
        self._pick_colour("crack_color", "Crack colour")

    def _update_crack_swatch(self):
        self.a_crack_color.setIcon(color_icon(tuple(self.display["crack_color"])))

    def _on_boundary_thickness(self, value: int):
        self.display["boundary_thickness"] = value
        self.bound_thick_label.setText(f"{value} px")
        self._save_display()
        self._apply_display()

    def _on_dim_strength(self, value: int):
        self.display["dim_strength"] = value
        self._save_display()
        self._apply_display()

    def _on_shading(self, value: int):
        self.display["shading"] = value
        self._save_display()
        self._apply_display()

    def fit_view(self):
        self.viewer.fit(self.project.stone if self.project.iso is not None else None)

    # ── refresh ──────────────────────────────────────────────────────────────
    def _refresh_surface(self):
        """Per-face state and outlines (after isolation / segmentation / brush / crack changes)."""
        p = self.project
        if not p.has_scan:
            return
        self.viewer.set_face_state(p.stone, p.seg, p.crack)
        if p.iso is not None:
            seg = p.seg if p.seg is not None else np.zeros(p.scan.n_faces, np.int32)
            seg_e, out_e = SF.boundary_edges(p.scan, seg, p.stone)
            self.viewer.set_lines("seg", p.scan.P, seg_e if p.seg is not None else seg_e[:0])
            self.viewer.set_lines("outline", p.scan.P, out_e)
        else:
            self.viewer.set_lines("seg", p.scan.P, np.zeros((0, 2), np.int32))
            self.viewer.set_lines("outline", p.scan.P, np.zeros((0, 2), np.int32))
        self._refresh_colors()

    def _refresh_colors(self):
        p = self.project
        if not p.has_scan:
            return
        n = int(p.seg.max()) + 1 if p.seg is not None else 1
        rgba = np.zeros((n, 4), np.uint8)
        flags = np.zeros(n, np.uint8)
        if p.seg is not None:
            p.ensure_class_capacity()
            pal = np.zeros((max(p.classes) + 1, 4), np.uint8)
            for cid, c in p.classes.items():
                pal[cid, :3] = c["color"]
                pal[cid, 3] = 255 if cid else 0
            cls = np.clip(p.seg_class[:n], 0, len(pal) - 1)
            rgba = pal[cls]
            sel = [s for s in self.selected if s < n]
            flags[sel] = 1
            if 0 < self.hover_seg < n and self.viewer.tool == "assign":
                flags[self.hover_seg] |= 2
        self.viewer.set_segment_colors(rgba, flags)

    _refresh_class_colours = _refresh_colors
    _refresh_selection_view = _refresh_colors

    def _class_area_stats(self):
        p = self.project
        if not (p.has_scan and p.iso is not None):
            return {}, 0
        return p.class_areas(), p.stone_area()

    def _legend_rows(self):
        p = self.project
        if p.seg is None:
            return []
        areas = p.class_areas()
        total = p.stone_area()
        rows = [(tuple(c["color"]), capitalize_words(c["name"]),
                 f"{100 * areas.get(cid, 0) / total:.1f} %" if total else "", False)
                for cid, c in sorted(p.classes.items())]
        if p.crack is not None and (p.crack & p.stone).any():
            ca = float(p.area()[p.crack & p.stone].sum())
            rows.append((tuple(self.display["crack_color"]), "Cracks", f"{100 * ca / total:.2f} %", True))
        return rows

    def _refresh_legend(self):
        rows = self._legend_rows()
        if not rows or not self.show_legend.isChecked():
            self.legend_label.hide()
            return
        img = render_legend(rows, 1.15 * self.legend_size.value() / 100)
        self.legend_label.setPixmap(QPixmap.fromImage(img))
        self.legend_label.resize(img.width(), img.height())
        self._place_legend()
        self.legend_label.show()

    def _place_legend(self):
        m = 16
        self.legend_label.move(self.viewer.width() - self.legend_label.width() - m, m)

    def _on_legend_size(self, value: int):
        if self.project.legend_scale_pct != value:
            self.project.legend_scale_pct = value
            self.project.dirty = True
        self._refresh_legend()

    def _refresh_stone_info(self):
        p = self.project
        if p.iso is None:
            self.stone_info.setText("")
            return
        e = p.extents()
        u = p.unit
        note = "" if p.scale_set else "  ·  no scale set (••• → Set Scale)"
        self.stone_info.setText(f"Stone {e[0]:.2f} × {e[1]:.2f} × {e[2]:.2f} {u}, surface {p.stone_area():.2f} "
                                f"{u}²{note}")

    def refresh_all(self):
        self._refresh_surface()
        self._refresh_classes()
        self._refresh_stone_info()
        self._update_enabled()

    # ════════════════════════════════════════════════════════════════════════
    # file operations
    # ════════════════════════════════════════════════════════════════════════
    def _load_job(self, path, max_tri: int, bake_px: int):
        def job(progress):
            from scipy.spatial import cKDTree
            scan = load_scan(path, progress, max_tri, bake_px)
            progress("Indexing the surface …")
            tree = cKDTree(scan.V[scan.F].mean(1))
            gp = ground_plane(scan)
            up = gp[1] if gp is not None else file_up(scan)
            return scan, tree, up
        return job

    def _show_scan(self, scan, tree, up):
        self.tree = tree
        self._unwrap_cache = None
        self.selected.clear()
        self.hover_seg = 0
        self.app.show_workspace(self)
        self.viewer.set_scan(scan, self.settings.max_texture_side)
        self.viewer.set_up(up)
        QTimer.singleShot(0, self.fit_view)

    def open_scan(self, path: str):
        if self.runner.busy or not self._confirm_discard():
            return
        self.qsettings.setValue("last_dir", str(Path(path).parent))

        def done(res):
            scan, tree, up = res
            self.project.set_scan(scan)
            self.project.max_triangles, self.project.bake_texture_px = max_tri, bake_px
            self.project.cut_cm = self.settings.ground_cut_cm
            self.project_path = None
            self._show_scan(scan, tree, up)
            self.app.set_document_title(scan.path.name)
            size = (f"reduced from {scan.n_original:,} to {scan.n_faces:,} triangles"
                    if scan.n_original > scan.n_faces else f"{scan.n_faces:,} triangles")
            self.status_msg.setText(f"Loaded {scan.path.name} ({size}).  Press Detect and Isolate Stećak.")
            self.refresh_all()

        max_tri, bake_px = self.settings.max_triangles, self.settings.bake_texture_px
        self._run("Opening scan", self._load_job(path, max_tri, bake_px), done, overlay_delay_ms=0)

    def save_project(self, ask: bool = False) -> bool:
        p = self.project
        if not p.has_scan:
            return False
        path = None if ask else self.project_path
        if path is None:
            q, _ = QFileDialog.getSaveFileName(self, "Save project",
                                               str(Path(self._last_dir()) / f"{p.scan.path.stem}{PROJECT_EXT_3D}"),
                                               PROJECT_FILTER)
            if not q:
                return False
            path = Path(q)
        try:
            p.save(path, camera=self.viewer.cam.to_dict())
        except Exception as e:  # noqa: BLE001
            self._error(f"Saving failed: {e}")
            return False
        self.project_path = path
        self.app.set_document_title(path.name)
        self.status_msg.setText(f"Saved {path}")
        return True

    def load_project(self, path: str):
        if self.runner.busy or not self._confirm_discard():
            return
        self.qsettings.setValue("last_dir", str(Path(path).parent))
        try:
            meta = Project.read_meta(path)
        except Exception as e:  # noqa: BLE001
            self._error(f"Cannot open project: {e}")
            return
        scan_path = Path(meta["scan_path"])
        if not scan_path.is_file():
            alt = Path(path).parent / scan_path.name
            if alt.is_file():
                scan_path = alt
            else:
                q, _ = QFileDialog.getOpenFileName(self, f"Where is {scan_path.name}?", str(Path(path).parent),
                                                   SCAN_FILTER)
                if not q:
                    return
                scan_path = Path(q)

        def done(res):
            scan, tree, up = res
            proj = Project()
            try:
                meta = proj.load(path, scan)
            except Exception as e:  # noqa: BLE001
                self._error(f"Cannot open project: {e}")
                return
            self.project = proj
            self.project_path = Path(path)
            self._show_scan(scan, tree, proj.iso.R[2] if proj.iso is not None else up)
            if proj.iso is not None:
                self.viewer.set_up(proj.iso.R[2], proj.iso.R[0])
            self.gran.setValue(proj.granularity)
            self.legend_size.setValue(proj.legend_scale_pct)
            proj.dirty = False
            if meta.get("camera"):
                QTimer.singleShot(10, lambda: (self.viewer.cam.from_dict(meta["camera"]),
                                               self.viewer._camera_moved()))
            self.app.set_document_title(Path(path).name)
            self.status_msg.setText(f"Opened project {path}")
            self.refresh_all()

        # a project always reopens on the mesh it was made with (its triangles are numbered)
        job = self._load_job(scan_path, int(meta.get("max_triangles", 0)), int(meta.get("bake_texture_px", 4096)))
        self._run("Opening project", job, done, overlay_delay_ms=0)

    def export(self):
        p = self.project
        if p.iso is None:
            self.status_msg.setText("Detect the stećak first.")
            return
        default_name = p.export_name or p.scan.path.stem
        default_dir = self.qsettings.value("export_dir", "") or str(p.scan.path.parent / "exports")
        site = dict(p.site_info)
        site.setdefault("surveyor", self.qsettings.value("surveyor", ""))
        dlg = ExportDialog(default_name, site, default_dir, self, scan=True)
        if not dlg.exec():
            return
        name, site, d = dlg.values()
        out = Path(d)
        targets = [f"{name}.png", f"{name}_views.png", f"{name}.glb", f"{name}.json"]
        existing = [f for f in targets if (out / f).exists()]
        if existing and QMessageBox.question(
                self, "Overwrite?", "These files already exist:\n" + "\n".join(existing) + "\n\nOverwrite them?") \
                != QMessageBox.Yes:
            return
        out.mkdir(parents=True, exist_ok=True)
        p.export_name = name
        if site != p.site_info:
            p.site_info = site
            p.dirty = True
        self.qsettings.setValue("export_dir", d)
        self.qsettings.setValue("surveyor", site.get("surveyor", ""))

        # on the GUI thread: screenshot with the legend, and the legend image for the views sheet
        shot = self.viewer.grabFramebuffer()
        rows = self._legend_rows()
        legend = None
        if rows and self.show_legend.isChecked():
            dpr = shot.devicePixelRatio()
            lr = render_legend(rows, 1.15 * self.legend_size.value() / 100 * dpr)
            lr.setDevicePixelRatio(dpr)
            painter = QPainter(shot)
            painter.drawImage(int(shot.width() / dpr - lr.width() / dpr - 16), 16, lr)
            painter.end()
            legend = qimage_to_rgba(render_legend(rows, 1.6 * self.legend_size.value() / 100, dark=True))
        shot.save(str(out / f"{name}.png"))
        opts = dict(self.viewer.opts)
        project_file = str(self.project_path) if self.project_path else None
        from ..scan3d import export3d

        def job(progress):
            progress("Measuring on the 3D surface …")
            rep = {"name": name, "site": site, "exported_at": datetime.now().isoformat(timespec="seconds"),
                   "source": {"scan": str(p.scan.path), "triangles": p.scan.n_faces,
                              "triangles_in_file": p.scan.n_original, "project_file": project_file},
                   "isolation": {"ground_plane": p.iso.ground is not None, "ground_cut_cm": p.cut_cm,
                                 "refine_strokes": p.refined}}
            rep.update(p.report())
            rep["files"] = targets
            (out / f"{name}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
            sheet = export3d.views_sheet(p, opts, legend=legend, progress=progress)
            sheet.save(out / f"{name}_views.png")
            export3d.export_glb(p, opts, out / f"{name}.glb", progress=progress)
            return [out / t for t in targets]

        def done(files):
            self.status_msg.setText(f"Exported {len(files)} files to {d}")
            if QMessageBox.question(self, "Export done", "Exported:\n" + "\n".join(f.name for f in files)
                                    + f"\n\nto {d}\n\nOpen folder?") == QMessageBox.Yes:
                os.startfile(d)  # noqa: S606

        self._run("Exporting", job, done)

    # ════════════════════════════════════════════════════════════════════════
    # isolation
    # ════════════════════════════════════════════════════════════════════════
    def detect_stone(self):
        p = self.project
        if not p.has_scan:
            return
        scan, cut, scale = p.scan, p.cut_cm, p.scale

        def job(progress):
            return isolate(scan, cut, scale if p.scale_set else 1.0, progress)

        def done(iso):
            old_seg = p.seg
            if p.iso is not None:
                p.push_undo()
            p.iso = iso
            p.dirty = True
            if old_seg is not None:
                p.seg = SF.apply_stone_change(scan, old_seg, iso.stone)
            self._unwrap_cache = None
            self.viewer.set_up(iso.R[2], iso.R[0])
            self.viewer.view_preset("iso")
            self.fit_view()
            self.refresh_all()
            e = p.extents()
            msg = f"Stone isolated: {iso.stone.sum():,} of {scan.n_faces:,} triangles."
            if iso.ground is None:
                msg = "No open ground edge in this scan: the whole model is used as the stone."
            self.status_msg.setText(msg + "  Press Segment, or Refine Isolation (R) first.")
            if not p.scale_set and max(e) > 3.5:
                self.status_msg.setText(self.status_msg.text() + f"  Note: the stone is {e[0]:.2f} units long – "
                                        "if that is not metres, set the scale (Detect ••• menu).")

        self._run("Isolating the stone", job, done)

    def set_cut_height(self):
        p = self.project
        u = "cm" if p.scale_set else "cm (model units ÷ 100 – no scale set)"
        val, ok = QInputDialog.getDouble(self, "Ground cut height", f"Faces lower than this above the ground plane "
                                         f"count as ground ({u}):", p.cut_cm, 0.0, 200.0, 1)
        if ok:
            p.cut_cm = val
            self.settings.ground_cut_cm = val
            self._save_settings()
            self.detect_stone()

    def set_scale(self):
        p = self.project
        L = p.scan.V[np.unique(p.scan.F[p.stone])].astype(np.float64) @ p.iso.R.T
        ext = L.max(0) - L.min(0)
        cur = ext[0] * p.scale if p.scale_set else ext[0]
        val, ok = QInputDialog.getDouble(
            self, "Set scale", f"True length of the stone in metres\n(longest side; the model measures "
                               f"{ext[0]:.3f} units):", cur, 0.01, 100.0, 3)
        if not ok:
            return
        p.scale = val / ext[0]
        p.scale_set = True
        p.dirty = True
        self.refresh_all()
        self.status_msg.setText(f"Scale set: 1 model unit = {p.scale:.4f} m. Areas are now in m².")

    # ════════════════════════════════════════════════════════════════════════
    # segmentation
    # ════════════════════════════════════════════════════════════════════════
    def auto_segment(self):
        p = self.project
        if p.iso is None:
            return
        level = self.gran.value()
        method = self.unwrap_combo.currentData()
        method_label = self.unwrap_combo.currentText()
        scan, iso, stone = p.scan, p.iso, p.stone
        max_side = self.settings.unwrap_max_side
        cache = self._unwrap_cache
        cached = cache[3] if cache and cache[0] == method and cache[1] is stone and cache[2] == max_side else None
        g = self._granularity(max(level, 1))
        models = self.models
        from ..scan3d import unwrap as UW

        def job(progress):
            if level == 0:
                return None, stone.astype(np.int32), 0
            uw = cached or UW.compute(scan, iso, method, max_side, progress)
            labels = SF.segment_unwrap(uw, models, g, level, progress)
            progress(f"Mapping {labels.max()} segments onto the 3D surface …")
            seg = SF.faces_from_unwrap(scan, uw, labels, stone, level)
            return uw, seg, int(labels.max())

        def done(res):
            uw, seg, n2d = res
            if uw is not None:
                self._unwrap_cache = (method, stone, max_side, uw)
            old_seg, old_cls = p.seg, p.seg_class
            p.push_undo()
            p.seg = seg
            if self.keep_cb.isChecked() and old_seg is not None:
                p.seg_class = SF.transfer_classes(old_seg, old_cls, seg, p.scan.area)
            else:
                p.seg_class = np.zeros(int(seg.max()) + 1, np.int16)
            p.granularity = level
            p.unwrap_method = method if level else ""
            self.selected.clear()
            self.refresh_all()
            if level == 0:
                self.status_msg.setText("Segmentation off: the whole stone is one segment. "
                                        "Brush classes with Shift+drag.")
            else:
                self.status_msg.setText(f"{int(seg.max())} segments on the stone ({method_label} "
                                        f"unwrap, fidelity {level}). Pick a class and click segments.")

        self._run("Segmenting", job, done)

    def clear_segmentation(self):
        p = self.project
        if self.runner.busy or p.seg is None:
            return
        msg = "Remove all segments"
        if p.seg_class.any():
            msg += " and their class assignments"
        if QMessageBox.question(self, "Clear segmentation",
                                msg + "? Cracks and the isolation are kept. (Undo with Ctrl+Z)") != QMessageBox.Yes:
            return
        p.push_undo()
        p.seg = None
        p.seg_class = np.zeros(1, np.int16)
        self.selected.clear()
        if self.viewer.tool == "split":
            self.set_tool("assign")
        self.refresh_all()
        self.status_msg.setText("Segmentation cleared.")

    # ════════════════════════════════════════════════════════════════════════
    # assigning
    # ════════════════════════════════════════════════════════════════════════
    def _segment_of(self, face: int) -> int:
        p = self.project
        if p.seg is None or face < 0 or not p.stone[face]:
            return 0
        return int(p.seg[face])

    def _on_hover(self, face: int):
        """Highlight the segment under the mouse."""
        sid = self._segment_of(face) if self.project.has_scan and face >= 0 else 0
        if sid != self.hover_seg:
            self.hover_seg = sid
            self._refresh_colors()

    def merge_selected(self):
        if not self._require_segments():
            return
        if len(self.selected) < 2:
            self.status_msg.setText("Select at least two segments (Ctrl+click) to merge.")
            return
        p = self.project
        p.ensure_class_capacity()
        ids = sorted(self.selected)
        areas = np.bincount(p.seg[p.stone], p.scan.area[p.stone], minlength=len(p.seg_class))[ids]
        cls = int(p.seg_class[ids[int(np.argmax(areas))]])
        p.push_undo()
        p.seg, target = SF.merge(p.seg, ids)
        p.seg_class = p.seg_class.copy()
        p.seg_class[target] = cls
        self.selected = {target}
        self.refresh_all()
        self.status_msg.setText(f"Merged {len(ids)} segments into segment {target}.")

    # ── brushes ──────────────────────────────────────────────────────────────
    def _faces_in_stroke(self, pts, radii) -> np.ndarray:
        hits = self.tree.query_ball_point(pts, radii, workers=-1)
        idx = (np.unique(np.concatenate([np.asarray(h, np.int64) for h in hits])) if len(hits)
               else np.zeros(0, np.int64))
        m = np.zeros(self.project.scan.n_faces, bool)
        m[idx] = True
        return m

    def _on_stroke(self, pts, radii, button, kind):
        p = self.project
        if not p.has_scan or self.tree is None:
            return
        right = button == Qt.RightButton
        if kind == "crack":
            m = self._faces_in_stroke(pts, radii * (5 if right else 1)) & p.stone
            if not m.any():
                return
            p.push_undo()
            p.crack = (p.crack & ~m) if right else (p.crack | m)
            self._refresh_surface()
            self._refresh_legend()
        elif kind == "split":
            if not self._require_segments():
                return
            m = self._faces_in_stroke(pts, radii)
            min_area = GRANULARITY[max(1, p.granularity)].min_frac * float(p.scan.area[p.stone].sum()) * 0.1
            seg, created = SF.split(p.scan, p.seg, m, p.stone, min_area)
            if not created:
                self.status_msg.setText("Nothing split – draw the line fully across a segment.")
                return
            p.push_undo()
            p.seg = seg
            p.ensure_class_capacity()
            p.seg_class = p.seg_class.copy()
            for new, parent in created:
                p.seg_class[new] = p.seg_class[parent]
            self.selected.clear()
            self.refresh_all()
            self.status_msg.setText(f"Split created {len(created)} new segment(s).")
        elif self.viewer.tool == "refine":
            m = self._faces_in_stroke(pts, radii)
            new = (p.stone & ~m) if right else (p.stone | m)
            if np.array_equal(new, p.stone):
                return
            p.push_undo()
            p.iso.stone = new
            p.refined += 1
            if p.seg is not None:
                p.seg = SF.apply_stone_change(p.scan, p.seg, new)
            self._unwrap_cache = None
            self.refresh_all()
            self.status_msg.setText("Removed from the stone." if right else "Added to the stone.")
        else:                                   # class brush (Shift+drag)
            if not self._require_segments():
                return
            m = self._faces_in_stroke(pts, radii)
            cls = 0 if right else self.active_class()
            res = SF.paint(p.scan, p.seg, p.seg_class, m, cls, p.stone,
                           crumb_area=float(p.scan.area[p.stone].sum()) * 2e-5)
            if res is None:
                return
            p.push_undo()
            p.seg, p.seg_class = res
            self.selected.clear()
            self.refresh_all()
            self.status_msg.setText(f"Painted {capitalize_words(p.classes.get(cls, {'name': '?'})['name'])}.")
        self._update_enabled()

    def clear_cracks(self):
        p = self.project
        if p.has_scan and p.crack.any():
            p.push_undo()
            p.crack = np.zeros_like(p.crack)
            self._refresh_surface()
            self._refresh_legend()
            self._update_enabled()

    # ════════════════════════════════════════════════════════════════════════
    # undo / redo
    # ════════════════════════════════════════════════════════════════════════
    def undo(self):
        if not self.runner.busy and self.project.undo():
            self.selected.clear()
            self.refresh_all()
            self.status_msg.setText("Undo.")

    def redo(self):
        if not self.runner.busy and self.project.redo():
            self.selected.clear()
            self.refresh_all()
            self.status_msg.setText("Redo.")
