"""Photo workspace: isolate the stećak in a photo, segment it, assign classes, draw cracks, export."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QActionGroup, QColor
from PySide6.QtWidgets import QColorDialog, QFileDialog, QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton, QSlider

from .. import segmentation as S
from ..config import GRANULARITY, PROJECT_EXT_2D, granularity
from ..image2d import rendering as R
from ..image2d.project import Project
from .canvas import Canvas
from .export_dialog import ExportDialog
from .legend import legend_size, qimage_to_rgba, render_legend
from .widgets import Section, capitalize_words, color_icon, set_role
from .workspace import ASSIGN_HINT, Workspace

PROJECT_FILTER = f"Stecci photo project (*{PROJECT_EXT_2D})"
LEGEND_PCT_MIN, LEGEND_PCT_MAX = 50, 200     # legend size slider (100 % = automatic size)
# brush sizes in px: (default, min, max)
BRUSH_LIMITS = {"brush": (5, 1, 20),       # cracks
                "paint": (50, 5, 100)}     # class brush (Shift+drag)
DISPLAY_VERSION = 4       # bump when defaults change; RESET_KEYS says which stored values to drop
RESET_KEYS = {2: ("boundary_color", "outline_color"),
              3: ("dim_strength",),
              4: ("dim_strength",)}
DEFAULT_DISPLAY = {"version": DISPLAY_VERSION,
                   "boundary_color": [57, 255, 20], "boundary_thickness": 2, "boundary_style": "solid",  # neon green
                   "outline_color": [255, 20, 200], "outline_style": "solid",   # neon pink: stećak outline + box
                   "crack_color": [230, 30, 30],
                   "dim_strength": 100}     # % darkening of the background around the stećak

HELP_HTML = """
<h3>Photo workflow</h3>
<ol>
<li><b>Open</b> a photo (Ctrl+O), then press <b>Detect and Isolate Stećak</b> (Ctrl+D).</li>
<li>If needed, <b>Refine outline</b> (R): drag a box, left-click = include, right-click = exclude.</li>
<li><b>Segmentation</b>: set the <b>fidelity</b> and press <b>Segment</b> (Ctrl+G). Fidelity 0 = off: the whole stone is one segment.</li>
<li><b>Assign classes</b>: pick a class (keys 0–9), click or drag over segments. Right-click = unassign.
Ctrl+left click / drag = select segments, Ctrl+right = unselect, <b>M</b> = merge them. <b>Split Segments Manually</b> (X): draw a line across a segment.
<br><b>Shift+drag</b> brushes the active class freely, Shift+right-drag brushes Unassigned/Stone, Shift+wheel = brush size.</li>
<li><b>Draw cracks</b> (B): left-drag paints, right-drag erases (eraser = 5× the brush size). <b>[ ]</b> or Ctrl+wheel = brush size.</li>
<li><b>Export</b> (Ctrl+E): enter the stećak name and site information; writes <i>name</i>.png (as on screen), <i>name</i>_surroundings.png, <i>name</i>_stone_outlines.png, <i>name</i>_white.png (class map), <i>name</i>_cracks.png, <i>name</i>_stone.png and <i>name</i>.json (measurements).</li>
</ol>
<h3>Navigation</h3>
Mouse wheel = zoom · middle-drag or Space+drag = pan · F = fit to window<br>Legend: drag = move · wheel over it = size · double-click = back to the corner
"""


class PhotoWorkspace(Workspace):
    KIND = "2d"
    MODE_LABEL = "PHOTO  ·  2D"
    HELP_HTML = HELP_HTML
    PANEL_MIN_WIDTH = 320

    LEGEND_REF_SIDE = 1100      # legend drawn at scale 1 (12 px text) for a 1100 px crop box: text ≈ 1 % of the box
    LEGEND_MIN_SCALE = 0.375    # automatic size never below 4.5 px text (the size slider can enlarge it)
    LEGEND_MARGIN = 0.03        # free space around the legend, fraction of the crop box size

    def __init__(self, app):
        super().__init__(app)
        self.display = self._load_display("display2d", DEFAULT_DISPLAY, RESET_KEYS)
        self.project = Project()
        self.canvas = Canvas(self)
        self._build_actions()
        self._build_menus()
        self._build_panel()
        self._connect_canvas()
        self._refresh_classes()
        self._update_enabled()

    # ════════════════════════════════════════════════════════════════════════
    # UI construction
    # ════════════════════════════════════════════════════════════════════════
    def _build_actions(self):
        A = self._act
        self.a_detect = A("Detect and Isolate Stećak", self.detect_rock, "Ctrl+D")
        self.a_clear_pts = A("Clear Refine Points", self.clear_rock_points)
        self.fill_holes_cb = A("Fill Holes In Rock Mask", checkable=True, checked=True)

        self.a_segment = A("Segment", self.auto_segment, "Ctrl+G")
        self.a_clear_seg = A("Clear Segmentation…", self.clear_segmentation)
        self.keep_cb = A("Keep Class Assignments When Re-Segmenting", checkable=True, checked=True)
        self.a_bound_color = A("Boundary Colour…", self.pick_boundary_color)
        self.a_outline_color = A("Outline Colour…", self.pick_outline_color)
        self.a_crop = A("Crop To Box On Export", checkable=True, checked=True)
        self.a_crop.setToolTip("Hide everything outside the box around the stećak. The project keeps the full "
                               "photo, so this can be switched off again; files are only cropped on Export.")
        self.a_crop.toggled.connect(self._on_crop_toggled)
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

        self.a_brush_smaller = A("Smaller Brush", lambda: self.set_brush(self.canvas.brush_size - 2), "[")
        self.a_brush_larger = A("Larger Brush", lambda: self.set_brush(self.canvas.brush_size + 2), "]")
        self.a_clear_cracks = A("Clear All Cracks", self.clear_cracks)
        self.a_crack_color = A("Crack Colour…", self.pick_crack_color)

        self.a_fit = A("Fit to window", self.canvas.fit, "F")
        self.show_fill = A("Class colours", checkable=True, checked=True)
        self.show_bound = A("Segment boundaries", checkable=True, checked=True)
        self.show_crack = A("Cracks", checkable=True, checked=True)
        self.dim_bg = A("Dim background", checkable=True, checked=True)
        self.show_legend = A("Legend", checkable=True, checked=True)
        self.show_legend.toggled.connect(lambda _: self._refresh_legend())
        self.show_fill.toggled.connect(lambda _: self._refresh_fill())
        self.show_crack.toggled.connect(lambda on: self.canvas.set_layer_visible("crack", on))
        self.show_bound.toggled.connect(lambda _: self._refresh_boundary())
        self.dim_bg.toggled.connect(lambda _: self._refresh_fill())

        # tools (exclusive)
        self.tool_group = QActionGroup(self)
        self.tool_actions = {}
        for key, text, sc in [("assign", "Assign Tool", "A"), ("rock", "Refine Outline Tool", "R"),
                              ("split", "Split Segments Manually", "X"), ("brush", "Crack Brush", "B")]:
            a = A(text, lambda k=key: self.set_tool(k), sc, checkable=True)
            self.tool_group.addAction(a)
            self.tool_actions[key] = a
        self.tool_actions["assign"].setChecked(True)
        self._class_shortcuts()

    def _build_menus(self):
        """The ••• menus on the step cards."""
        self.menu_detect = QMenu("&Detect", self)
        for a in (self.a_detect, self.tool_actions["rock"], self.a_clear_pts, None,
                  self.fill_holes_cb, None, self.a_outline_color):
            self.menu_detect.addSeparator() if a is None else self.menu_detect.addAction(a)
        self.menu_detect.addMenu(self._style_menu("Outline Style", "outline_style"))
        self.menu_detect.addSeparator()
        self.menu_detect.addAction(self.a_crop)

        self.menu_seg = QMenu("&Segmentation", self)
        for a in (self.a_segment, self.tool_actions["split"], None, self.a_clear_seg, None, self.keep_cb, None,
                  self.a_bound_color):
            self.menu_seg.addSeparator() if a is None else self.menu_seg.addAction(a)
        self.bound_thick = QSlider(Qt.Horizontal)
        self.bound_thick.setRange(1, 8)
        self.bound_thick.setValue(int(self.display["boundary_thickness"]))
        self.bound_thick_label = QLabel()
        self.bound_thick.valueChanged.connect(self._on_boundary_thickness)
        self.menu_seg.addMenu(self._style_menu("Boundary Style", "boundary_style"))
        self.menu_seg.addAction(self._slider_action("Boundary Thickness", self.bound_thick, self.bound_thick_label))
        self._on_boundary_thickness(self.bound_thick.value(), save=False)
        self.menu_seg.addSeparator()
        self.menu_seg.addAction(self.a_gran_info)

        self.menu_classes = QMenu("&Classes", self)
        for a in (self.tool_actions["assign"], None, self.a_add_class, self.a_rename_class,
                  self.a_color_class, self.a_delete_class, None, self.a_assign_sel, self.a_merge,
                  self.a_clear_sel, None, self.a_clear_assign):
            self.menu_classes.addSeparator() if a is None else self.menu_classes.addAction(a)

        self.menu_cracks = QMenu("C&racks", self)
        self.menu_cracks.addAction(self.a_crack_color)
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(*BRUSH_LIMITS["brush"][1:])
        self.brush_slider.valueChanged.connect(lambda v: self.set_brush(v, "brush"))
        self.brush_label = QLabel()
        self.menu_cracks.addAction(self._slider_action("Brush Size", self.brush_slider, self.brush_label))
        self.menu_cracks.addSeparator()
        self.menu_cracks.addAction(self.a_clear_cracks)

    def _style_menu(self, title: str, key: str) -> QMenu:
        """Solid / Dashed / Dotted choice stored in the display preferences under `key`."""
        m = QMenu(title, self)
        group = QActionGroup(m)
        for style in R.LINE_STYLES:
            a = m.addAction(style.capitalize())
            a.setCheckable(True)
            a.setChecked(self.display.get(key, "solid") == style)
            group.addAction(a)
            a.triggered.connect(lambda _=False, s=style: self._set_line_style(key, s))
        return m

    def _set_line_style(self, key: str, style: str):
        self.display[key] = style
        self._save_display()
        self._refresh_boundary()

    def _build_panel(self):
        self.brush_sizes = {k: v[0] for k, v in BRUSH_LIMITS.items()}   # crack / class brush keep their own size

        # 1 · Detect
        self.sec_detect = Section(1, "Detect", self.menu_detect)
        self.btn_detect = self._big_button("Detect and Isolate Stećak", self.detect_rock,
                                           "Find the stećak and isolate it from the background (SAM)  ·  Ctrl+D")
        self.sec_detect.body.addWidget(self.btn_detect)
        self.btn_refine = self._link_toggle("✎  Refine Outline", "rock",
                                            "Drag a box, or click: left = include, right = exclude  ·  R")
        self.sec_detect.body.addWidget(self.btn_refine)

        # 2 · Segmentation
        self.sec_seg = Section(2, "Segmentation", self.menu_seg)
        row = QHBoxLayout()
        row.addWidget(QLabel("Fidelity"))
        self.gran = QSlider(Qt.Horizontal)
        self.gran.setRange(0, 10)                 # 0 = off: whole stone is one segment
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
                                        "Pre-segment the rock with SAM at the chosen fidelity  ·  Ctrl+G")
        self.sec_seg.body.addWidget(self.btn_seg)
        self.btn_split = self._link_toggle("✂  Split Segments Manually", "split",
                                           "Draw a line across a segment to cut it  ·  X")
        self.sec_seg.body.addWidget(self.btn_split)
        self._update_gran_label()

        # 3 · Assign classes
        self.sec_classes = self._build_classes_section(
            self.menu_classes, BRUSH_LIMITS["paint"][1:],
            "Class brush: hold Shift and drag over the stone  ·  Shift+wheel = size")

        # 4 · Draw cracks
        self.sec_cracks = Section(4, "Draw Cracks", self.menu_cracks)
        self.btn_crack = QPushButton("Draw Cracks")
        self.btn_crack.setCheckable(True)
        self.btn_crack.setMinimumHeight(40)
        self.btn_crack.setCursor(Qt.PointingHandCursor)
        set_role(self.btn_crack, "secondary")
        self.btn_crack.setToolTip("Left-drag paints, right-drag erases  ·  B")
        self.btn_crack.toggled.connect(lambda on: self.set_tool("brush" if on else "assign"))
        self.sec_cracks.body.addWidget(self.btn_crack)
        self._update_crack_swatch()
        self.sec_cracks.body.addWidget(self._hint("Left-Drag: Paint · Right-Drag: Erase"))
        self.set_brush(self.brush_sizes["brush"], "brush")
        self.set_brush(self.brush_sizes["paint"], "paint")

        # Annotation settings (bottom of the panel): one row per layer, slider on the right where it applies
        self.sec_view = Section(None, "Annotation Settings")
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(50)
        self.opacity.setToolTip("Opacity of the class colours")
        self.opacity.valueChanged.connect(lambda _: self._refresh_fill())
        self.dim_slider = QSlider(Qt.Horizontal)
        self.dim_slider.setRange(0, 100)
        self.dim_slider.setValue(int(self.display["dim_strength"]))
        self.dim_slider.setToolTip("How strongly the background around the stećak is darkened")
        self.dim_slider.valueChanged.connect(self._on_dim_strength)
        self.legend_size = QSlider(Qt.Horizontal)
        self.legend_size.setRange(LEGEND_PCT_MIN, LEGEND_PCT_MAX)
        self.legend_size.setSingleStep(10)
        self.legend_size.setValue(100)
        self.legend_size.setToolTip("Also: mouse wheel over the legend. Drag the legend to move it, "
                                    "double-click it to put it back in the corner.")
        self.legend_size.valueChanged.connect(self._on_legend_size)
        grid, labels = self._annotation_grid((
            (self.show_fill, "Class Colours", self.opacity, None),
            (self.show_bound, "Boundaries", None, "Segment Boundaries"),
            (self.show_crack, "Cracks", None, None),
            (self.dim_bg, "Dim Background", self.dim_slider, None),
            (self.show_legend, "Legend", self.legend_size, None)))
        self.legend_size_label = labels["Legend"]
        self.sec_view.body.addLayout(grid)
        self.sec_view.body.addWidget(self._link_button("⤢  Fit To Window", self.canvas.fit, "F"))

        self._install(self.canvas, [self.sec_detect, self.sec_seg, self.sec_classes, self.sec_cracks],
                      self.sec_view)

    def _connect_canvas(self):
        c = self.canvas
        c.assign_press.connect(lambda x, y, button, ctrl: self._assign_press(
            self._segment_at(x, y), button == Qt.LeftButton, ctrl))
        c.assign_drag.connect(lambda x, y, button: self._assign_drag(self._segment_at(x, y),
                                                                     button == Qt.LeftButton))
        c.assign_release.connect(self._assign_release)
        c.rock_box.connect(self._on_rock_box)
        c.rock_point.connect(self._on_rock_point)
        c.split_stroke.connect(self._on_split)
        c.brush_stroke.connect(self._on_brush)
        c.paint_stroke.connect(self._on_paint)
        c.brush_resized.connect(lambda v: self.set_brush(v, "brush"))
        c.paint_resized.connect(lambda v: self.set_brush(v, "paint"))
        c.legend_moved.connect(self._on_legend_moved)
        c.legend_dragging.connect(self._on_legend_dragging)
        c.legend_wheel.connect(lambda d: self.legend_size.setValue(self.legend_size.value() + 10 * d))
        c.legend_reset.connect(self._on_legend_reset)

    # ════════════════════════════════════════════════════════════════════════
    # main-window interface
    # ════════════════════════════════════════════════════════════════════════
    def has_document(self) -> bool:
        return self.project.has_image

    def close_document(self):
        classes = self.project.classes            # custom classes carry over to the next photo
        self.project = Project()
        self.project.classes = classes
        self.project_path = None
        self.selected.clear()
        self.canvas.set_legend(None)
        self.refresh_all()

    def settings_changed(self, old, new):
        self._update_gran_label()                # segment detail settings show in the fidelity tooltip

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
                                 f"Min Segment {100 * g.min_frac:.2f} % Of Rock")
        if self.settings.seg_mode == "colour":
            self.a_gran_info.setText(self.a_gran_info.text() + ", Detail From Colour")
        self.gran.setToolTip(self.a_gran_info.text())

    def _update_enabled(self):
        p = self.project
        idle = not self.runner.busy
        has, rock, seg = p.has_image, p.rock is not None, p.labels is not None
        for a in (self.a_detect, self.tool_actions["rock"]):
            a.setEnabled(has and idle)
        self.btn_detect.setEnabled(has and idle)
        self.btn_refine.setEnabled(has and idle and rock)
        self.a_clear_pts.setEnabled(has and idle and bool(p.rock_points))
        self.sec_seg.setEnabled(rock)
        self.a_segment.setEnabled(rock and idle)
        self.btn_seg.setEnabled(rock and idle)
        self.btn_split.setEnabled(seg)
        self.tool_actions["split"].setEnabled(seg)
        self.a_clear_seg.setEnabled(seg and idle)
        self.sec_view.set_active(has)
        self.sec_classes.setEnabled(seg)
        self.sec_cracks.setEnabled(has)
        # highlight the next step
        next_step = self.btn_detect if not rock else (self.btn_seg if not seg else None)
        for b in (self.btn_detect, self.btn_seg):
            set_role(b, "primary" if b is next_step else "secondary")
        self.sec_detect.set_active(has)
        self.sec_seg.set_active(rock)
        self.sec_classes.set_active(seg)
        self.sec_cracks.set_active(has)
        self.a_crop.setEnabled(rock and idle)
        self.a_crop.blockSignals(True)
        self.a_crop.setChecked(p.crop_to_rock)
        self.a_crop.blockSignals(False)
        self.app.update_file_actions()

    def _require_image(self) -> bool:
        if not self.project.has_image:
            self.status_msg.setText("Open a photo first.")
            return False
        return True

    def _segments(self):
        return self.project.labels

    def _min_area(self) -> int:
        frac = GRANULARITY[max(1, self.gran.value())].min_frac
        p = self.project
        rock_px = int(p.rock.sum()) if p.rock is not None else p.shape[0] * p.shape[1]
        return max(30, int(frac * rock_px))

    def current_tool(self) -> str:
        return self.canvas.tool

    def set_tool(self, tool: str):
        self.tool_actions[tool].setChecked(True)
        self.canvas.set_tool(tool)
        for b, t in ((self.btn_refine, "rock"), (self.btn_split, "split"), (self.btn_crack, "brush")):
            b.blockSignals(True)
            b.setChecked(tool == t)
            b.blockSignals(False)
        self.btn_crack.setText("Stop Drawing Cracks" if tool == "brush" else "Draw Cracks")
        self.canvas.brush_size = self.brush_sizes["brush"]
        self._on_active_class_changed()
        self._refresh_selection()            # hint text depends on the tool
        self.canvas.setFocus()

    def _on_active_class_changed(self):
        """Class brush (Shift+drag): the stroke colour follows the active class."""
        c = self.project.classes.get(self.active_class())
        if c is not None:
            self.canvas.paint_color = QColor(*c["color"], 170)

    def set_brush(self, size: int, which: str | None = None):
        """Set the size of the crack brush ("brush", default) or the Shift class brush ("paint")."""
        which = which or "brush"
        size = int(np.clip(size, *BRUSH_LIMITS[which][1:]))
        self.brush_sizes[which] = size
        if which == "paint":
            self.canvas.paint_size = size
        else:
            self.canvas.brush_size = size
        s, lab = (self.paint_slider, self.paint_label) if which == "paint" else (self.brush_slider, self.brush_label)
        s.blockSignals(True)
        s.setValue(size)
        s.blockSignals(False)
        lab.setText(f"{size} Px")

    # ── display preferences ──────────────────────────────────────────────────
    def pick_outline_color(self):
        col = QColorDialog.getColor(QColor(*self.display["outline_color"]), self, "Outline Colour")
        if col.isValid():
            self.display["outline_color"] = [col.red(), col.green(), col.blue()]
            self._save_display()
            self._refresh_boundary()

    def pick_boundary_color(self):
        col = QColorDialog.getColor(QColor(*self.display["boundary_color"]), self, "Boundary Colour")
        if col.isValid():
            self.display["boundary_color"] = [col.red(), col.green(), col.blue()]
            self._save_display()
            self._refresh_boundary()

    def _on_dim_strength(self, value: int):
        self.display["dim_strength"] = int(value)
        self._save_display()
        if not self.dim_bg.isChecked() and value > 0:
            self.dim_bg.setChecked(True)         # moving the slider means: show the dimming
        self._refresh_fill()

    def _on_crop_toggled(self, on: bool):
        p = self.project
        if p.crop_to_rock == on:
            return
        p.crop_to_rock = on
        p.dirty = True
        self._refresh_fill()
        self.canvas.fit()
        self.status_msg.setText("Export will be cropped to the box around the stećak (the project keeps the "
                                "full photo; untick to undo)." if on else "Export will use the full photo.")

    def _on_boundary_thickness(self, value: int, save: bool = True):
        self.display["boundary_thickness"] = int(value)
        self.bound_thick_label.setText(f"{value} Px")
        if save:
            self._save_display()
            self._refresh_boundary()

    def pick_crack_color(self):
        col = QColorDialog.getColor(QColor(*self.display["crack_color"]), self, "Crack Colour")
        if col.isValid():
            self.display["crack_color"] = [col.red(), col.green(), col.blue()]
            self._save_display()
            self._update_crack_swatch()
            self._refresh_crack()

    def _update_crack_swatch(self):
        r, g, b = self.display["crack_color"]
        self.a_crack_color.setIcon(color_icon((r, g, b)))     # current colour shown in the ••• menu
        self.canvas.crack_color = QColor(r, g, b, 200)

    # ── refresh ──────────────────────────────────────────────────────────────
    def _refresh_fill(self):
        if self.project.has_image:
            opacity = self.opacity.value() / 100 if self.show_fill.isChecked() else 0.0
            p = self.project
            self.canvas.set_layer("fill", R.fill_layer(
                p, opacity, self.dim_bg.isChecked(), round(2.55 * self.display["dim_strength"]),
                crop_preview=p.crop_to_rock))
            self.canvas.fit_box = p.crop_box() if p.crop_to_rock else None

    _refresh_class_colours = _refresh_fill

    def _refresh_boundary(self):
        if self.project.has_image:
            self.canvas.set_layer("boundary", self._boundary_layer(self.show_bound.isChecked()))
            self._refresh_prompts()          # the box follows the outline colour / style

    def _boundary_layer(self, show_segments: bool = True):
        d = self.display
        return R.boundary_layer(self.project, show_segments, tuple(d["boundary_color"]),
                                int(d["boundary_thickness"]), d["boundary_style"],
                                tuple(d["outline_color"]), d["outline_style"])

    def _refresh_selection(self):
        if not self.project.has_image:
            return
        self.selected = {s for s in self.selected
                         if self.project.labels is not None and (self.project.labels == s).any()} \
            if self.selected else set()
        self.canvas.set_layer("selection", R.selection_layer(self.project, self.selected))
        n = len(self.selected)
        self.sel_label.setText(f"{n} Segments Selected · M: Merge · Esc: Clear" if n else ASSIGN_HINT)

    _refresh_selection_view = _refresh_selection

    def _refresh_crack(self):
        if not self.project.has_image:
            return
        self.canvas.set_layer("crack", R.crack_layer(self.project, tuple(self.display["crack_color"])))
        self._refresh_legend()

    def _class_area_stats(self):
        p = self.project
        if not p.has_image:
            return {}, 0
        return p.class_areas(), int(p.rock.sum()) if p.rock is not None else 0

    def _legend_rows(self):
        p = self.project
        areas = p.class_areas()
        rock_px = int(p.rock.sum()) if p.rock is not None else 0
        rows = []
        for cid, c in sorted(p.classes.items()):
            pct = f"{100 * areas.get(cid, 0) / rock_px:.1f} %" if rock_px else ""
            rows.append((tuple(c["color"]), capitalize_words(c["name"]), pct, False))
        crack_px = int((p.crack & p.rock).sum()) if p.rock is not None else int(p.crack.sum())
        if crack_px:
            pct = f"{100 * crack_px / rock_px:.2f} %" if rock_px else ""
            rows.append((tuple(self.display["crack_color"]), "Cracks", pct, True))
        return rows

    def _legend_scale(self) -> float:
        p = self.project
        x0, y0, x1, y1 = p.base_crop_box()
        auto = max(self.LEGEND_MIN_SCALE, max(x1 - x0, y1 - y0) / self.LEGEND_REF_SIDE)
        return auto * p.legend_scale_pct / 100

    def _on_legend_size(self, value: int):
        p = self.project
        if p.legend_scale_pct != value:
            p.legend_scale_pct = value
            p.dirty = True
            self._refresh_legend()

    def _on_legend_dragging(self, x: float, y: float):
        """While dragging: the box follows the legend live (the crop preview catches up on release)."""
        self.project.legend_user_pos = (round(x), round(y))
        self._refresh_prompts()

    def _on_legend_moved(self, x: float, y: float):
        p = self.project
        p.legend_user_pos = (round(x), round(y))
        p.dirty = True
        self._refresh_legend()
        self._refresh_prompts()
        if p.crop_to_rock:
            self._refresh_fill()
        self.status_msg.setText("Legend moved (double-click it to put it back in the corner).")

    def _on_legend_reset(self):
        p = self.project
        p.legend_user_pos = None
        p.legend_scale_pct = 100
        p.dirty = True
        self.legend_size.blockSignals(True)
        self.legend_size.setValue(100)
        self.legend_size_label.setText("100 %")
        self.legend_size.blockSignals(False)
        self._refresh_legend()
        self.status_msg.setText("Legend back in the corner at its default size.")

    def _refresh_legend(self):
        """Legend in the crop box's top-right corner; the box grows so the legend never covers the stone."""
        p = self.project
        before = (p.legend_wh, p.legend_margin, p.legend_user_pos)
        p.legend_scale_pct = int(np.clip(p.legend_scale_pct, LEGEND_PCT_MIN, LEGEND_PCT_MAX))
        if self.legend_size.value() != p.legend_scale_pct:      # project opened / switched
            self.legend_size.blockSignals(True)
            self.legend_size.setValue(p.legend_scale_pct)
            self.legend_size_label.setText(f"{p.legend_scale_pct} %")
            self.legend_size.blockSignals(False)
        if not (self.show_legend.isChecked() and p.has_image and p.labels is not None
                and p.base_crop_box() is not None):
            p.legend_wh = None
            self.canvas.set_legend(None)
        else:
            rows = self._legend_rows()
            s = self._legend_scale()
            x0, y0, x1, y1 = p.base_crop_box()
            p.legend_wh = legend_size(rows, s)
            p.legend_margin = round(self.LEGEND_MARGIN * max(x1 - x0, y1 - y0))
            # rendered at 2× and shown at half size: stays sharp when zooming in
            self.canvas.set_legend(render_legend(rows, 2 * s), p.legend_pos(), 0.5)
        if (p.legend_wh, p.legend_margin, p.legend_user_pos) != before:
            # crop box changed with the legend
            self._refresh_prompts()
            if p.crop_to_rock:
                self._refresh_fill()

    def _refresh_prompts(self):
        box = self.project.crop_box() if self.project.has_image else None
        self.canvas.set_rock_prompts(box, self.project.rock_points,
                                     tuple(self.display["outline_color"]), self.display["outline_style"])

    def refresh_all(self):
        self._refresh_fill()
        self._refresh_boundary()
        self._refresh_selection()
        self._refresh_crack()
        self._refresh_prompts()
        self._refresh_classes()
        self._update_enabled()

    # ════════════════════════════════════════════════════════════════════════
    # file operations
    # ════════════════════════════════════════════════════════════════════════
    def _show_image_in_canvas(self):
        self.app.show_workspace(self)
        self.canvas.set_image(self.project.image)
        QTimer.singleShot(0, self.canvas.fit)   # fit again once the layout has settled

    def open_image(self, path: str):
        if self.runner.busy or not self._confirm_discard():
            return
        self.qsettings.setValue("last_dir", str(Path(path).parent))
        try:
            self.project.open_image(path, self.settings.working_max_side)
        except Exception as e:  # noqa: BLE001
            self._error(f"Cannot open image: {e}")
            return
        self.project_path = None
        self.selected.clear()
        self._show_image_in_canvas()
        self.app.set_document_title(Path(path).name)
        h, w = self.project.shape
        ow, oh = self.project.original_size
        self.status_msg.setText(f"Loaded {Path(path).name} ({ow}×{oh}, working at {w}×{h}).  "
                                "Press Detect and Isolate Stećak.")
        self.refresh_all()

    def save_project(self, ask: bool = False) -> bool:
        if not self._require_image():
            return False
        path = None if ask else self.project_path
        if path is None:
            stem = self.project.image_path.stem if self.project.image_path else "stecci"
            p, _ = QFileDialog.getSaveFileName(self, "Save project",
                                               str(Path(self._last_dir()) / f"{stem}{PROJECT_EXT_2D}"),
                                               PROJECT_FILTER)
            if not p:
                return False
            path = Path(p)
        try:
            self.project.save(path)
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
        proj = Project()
        try:
            proj.load(path, self.settings.working_max_side)
        except Exception as e:  # noqa: BLE001
            self._error(f"Cannot open project: {e}")
            return
        self.project = proj
        self.project_path = Path(path)
        self.selected.clear()
        self.gran.setValue(self.project.granularity)
        self._show_image_in_canvas()
        self.app.set_document_title(Path(path).name)
        self.status_msg.setText(f"Opened project {path}")
        self.refresh_all()

    def _export_legend(self, rows, dark: bool):
        """Editor legend at export resolution for the given rows: (rgba, (x, y) working px) or None."""
        p = self.project
        if not p.legend_wh or not rows:
            return None
        f = p.original_size[0] / p.shape[1] if p.original_size else 1.0
        return qimage_to_rgba(render_legend(rows, self._legend_scale() * f, dark=dark)), p.legend_pos()

    def export(self):
        """Export <name>.png (as on screen), _white, _surroundings, _cracks, _stone, _stone_outlines and <name>.json."""
        if not self._require_image():
            return
        p = self.project
        if p.rock is None:
            self.status_msg.setText("Detect the stećak first.")
            return
        default_name = p.export_name or (p.image_path.stem if p.image_path else "stecak")
        default_dir = self.qsettings.value("export_dir", "") or \
            (str(p.image_path.parent / "exports") if p.image_path else self._last_dir())
        site = dict(p.site_info)
        site.setdefault("surveyor", self.qsettings.value("surveyor", ""))     # remembered across projects
        dlg = ExportDialog(default_name, site, default_dir, self)
        if not dlg.exec():
            return
        name, site, d = dlg.values()
        suffixes = ("", "_white", "_surroundings", "_cracks", "_stone", "_stone_outlines")
        targets = [f"{name}{s}.png" for s in suffixes] + [f"{name}.json"]
        existing = [f for f in targets if (Path(d) / f).exists()]
        if existing and QMessageBox.question(
                self, "Overwrite?", "These files already exist:\n" + "\n".join(existing) + "\n\nOverwrite them?") \
                != QMessageBox.Yes:
            return
        p.export_name = name
        if site != p.site_info:
            p.site_info = site
            p.dirty = True
        self.qsettings.setValue("export_dir", d)
        self.qsettings.setValue("surveyor", site.get("surveyor", ""))

        # 1 · as on the screen
        opacity = self.opacity.value() / 100 if self.show_fill.isChecked() else 0.0
        layers = [R.fill_layer(p, opacity, self.dim_bg.isChecked(), round(2.55 * self.display["dim_strength"])),
                  self._boundary_layer(self.show_bound.isChecked())]
        if self.show_crack.isChecked():
            layers.append(R.crack_layer(p, tuple(self.display["crack_color"])))
        screen_legend = self._export_legend(self._legend_rows(), dark=False)
        images = [dict(suffix="", rgb=R.composite(p.image, *layers), pad=(0, 0, 0), legend=screen_legend)]

        # 2 · stone on white with classes (stone outline only, no segment outlines),
        # 3 · stone on white with cracks only (all segment outlines); both with the stone outline in black
        lines = dict(boundary_px=int(self.display["boundary_thickness"]),
                     boundary_style=self.display["boundary_style"])
        rows = self._legend_rows()                     # exactly the editor's rows (with percentages)
        class_rows = [r for r in rows if not r[3]]
        crack_rows = [r for r in rows if r[3]] or [(tuple(self.display["crack_color"]), "Cracks", "0.00 %", True)]
        crack_rgb = tuple(self.display["crack_color"])
        white = (255, 255, 255)
        fill = self.opacity.value() / 100
        images.append(dict(suffix="_white", rgb=R.white_map(p, classes=True, cracks=False, fill_alpha=fill,
                                                             segment_lines=False, **lines),
                           pad=white, legend=self._export_legend(class_rows, dark=True)))
        # 4 · whole photo with the surroundings and the black outlines only (no class colours, no legend)
        images.append(dict(suffix="_surroundings",
                           rgb=R.white_map(p, classes=False, cracks=False, surroundings=True, **lines),
                           pad=(0, 0, 0), legend=None))
        images.append(dict(suffix="_cracks", rgb=R.white_map(p, classes=False, cracks=True, crack_color=crack_rgb,
                                                             **lines),
                           pad=white, legend=self._export_legend(crack_rows, dark=True)))
        # 5 · stone only on white with the black segment outlines (no classes, no legend)
        images.append(dict(suffix="_stone_outlines", rgb=R.white_map(p, classes=False, cracks=False, **lines),
                           pad=white, legend=None))
        try:
            extra = {
                "site": site,
                "source": {"project_file": str(self.project_path) if self.project_path else None},
                "detection": {"method": p.detection_method or None,
                              "outline_refined": bool(p.rock_points) or p.rock_box_user,
                              "refine_box_drawn": p.rock_box_user,
                              "refine_points": len(p.rock_points)},
            }
            files = p.export(d, name, images, stone_on_white=True,      # + <name>_stone.png
                             report_extra=extra)                        # + <name>.json
        except Exception as e:  # noqa: BLE001
            self._error(f"Export failed: {e}")
            return
        self.status_msg.setText(f"Exported {len(files)} images to {d}")
        if QMessageBox.question(self, "Export done",
                                "Exported:\n" + "\n".join(f.name for f in files) + f"\n\nto {d}\n\nOpen folder?") \
                == QMessageBox.Yes:
            os.startfile(d)  # noqa: S606 (Windows)

    # ════════════════════════════════════════════════════════════════════════
    # rock isolation
    # ════════════════════════════════════════════════════════════════════════
    def detect_rock(self):
        if not self._require_image():
            return
        img, key = self.project.image, self.project.image_key
        use_dino = self.settings.use_dino

        def job(progress):
            if not self.models.ready:
                progress("Loading models …")
            if not use_dino:
                box, score, mask = self.models.detect_rock_sam(img, key, progress)
                return box, score, False, mask
            box, score = self.models.detect_rock_box(img, progress)
            fallback = box is None
            if fallback:
                h, w = img.shape[:2]
                box = np.array([0.03 * w, 0.03 * h, 0.97 * w, 0.97 * h], np.float32)
            mask = self.models.predict_mask(img, key, box=box, progress=progress)
            return box, score, fallback, mask

        def done(res):
            box, score, fallback, mask = res
            self._apply_rock(mask, [float(v) for v in box], [])
            self.project.detection_method = (
                "whole photo (nothing detected)" if fallback else
                f"Grounding DINO ({self.settings.gdino_model_id.split('/')[-1]}, score {score:.2f}) + SAM "
                f"{self.models.sam_type}" if use_dino else f"SAM {self.models.sam_type} from the photo centre")
            self.device_label.setText(self.models.describe())
            if fallback:
                msg = "No stećak detected – used the whole photo as box. Draw a box with Refine Outline (R)."
            elif use_dino:
                msg = f"Stećak isolated (DINO score {score:.2f}). Use Refine Outline (R) if needed, then press Segment."
            else:
                msg = "Stećak isolated. Use Refine Outline (R) if needed, then press Segment."
                rock = self.project.rock
                b = max(2, int(0.01 * max(rock.shape)))       # 1 % band along the photo edges
                edge = max(rock[:b].any(0).mean(), rock[-b:].any(0).mean(),
                           rock[:, :b].any(1).mean(), rock[:, -b:].any(1).mean())
                if edge > 0.05:     # runs into the photo border: maybe a neighbouring stone was included
                    msg = ("Stećak isolated, but the outline reaches the photo border – if another stone is "
                           "included, fix it with Refine Outline (R).")
            self.status_msg.setText(msg)
            if self.project.crop_to_rock:
                self.canvas.fit()                # zoom to the box that will be exported

        self._run("Detecting and isolating stećak", job, done)

    def _run_rock_sam(self, box, points, user_box: bool | None = None):
        img, key = self.project.image, self.project.image_key
        pts = [(x, y) for x, y, _ in points]
        lbl = [p for _, _, p in points]

        def job(progress):
            return self.models.predict_mask(img, key, box=box, points=pts or None,
                                            point_labels=lbl or None, progress=progress)

        def done(mask):
            self._apply_rock(mask, box, points,
                             self.project.rock_box_user if user_box is None else user_box)
            self.status_msg.setText(f"Rock updated ({len(points)} point(s)).")

        self._run("Refining rock outline", job, done)

    def _on_rock_box(self, x0, y0, x1, y1):
        if self.runner.busy:
            return
        self._run_rock_sam([x0, y0, x1, y1], [], user_box=True)   # drawn box = crop box, no margin

    def _on_rock_point(self, x, y, positive):
        if self.runner.busy:
            return
        pts = list(self.project.rock_points) + [(x, y, 1 if positive else 0)]
        self._run_rock_sam(self.project.rock_box, pts)

    def clear_rock_points(self):
        if self.project.rock_box is not None:
            self._run_rock_sam(self.project.rock_box, [])

    def _apply_rock(self, mask, box, points, user_box: bool = False):
        p = self.project
        p.push_undo()
        rock = S.clean_rock_mask(mask, self.fill_holes_cb.isChecked())
        if p.labels is not None:
            old_labels, old_cls = p.labels, p.seg_class
            p.rock = rock
            p.labels = S.apply_rock_change(old_labels, rock, self._min_area())
            p.seg_class = S.transfer_classes(old_labels, old_cls, p.labels)
        else:
            p.rock = rock
        p.rock_box, p.rock_points, p.rock_box_user = box, list(points), user_box
        self.refresh_all()

    # ════════════════════════════════════════════════════════════════════════
    # pre-segmentation
    # ════════════════════════════════════════════════════════════════════════
    def auto_segment(self):
        p = self.project
        if not self._require_image() or p.rock is None:
            return
        level = self.gran.value()
        if level == 0:
            self._apply_segmentation(p.rock.astype(np.int32), 0, keep=False,
                                     msg="Segmentation off: the whole stone is one segment. "
                                         "Brush classes with Shift+drag.")
            return
        min_area = self._min_area()
        g = self._granularity(level)
        x0, y0, x1, y1 = S.bbox_of(p.rock, pad=8)
        rock_crop = p.rock[y0:y1, x0:x1].copy()
        crop = p.image[y0:y1, x0:x1].copy()
        crop[~rock_crop] = 127          # neutral background so SAM focuses on the rock
        shape = p.shape

        def job(progress):
            masks = self.models.auto_masks(crop, g, min_area, progress=progress)
            progress(f"Building segments from {len(masks)} masks …")
            lab_crop = S.masks_to_labels(masks, rock_crop, min_area)
            max_area = max(min_area * 4, int(g.max_frac * rock_crop.sum()))
            if (np.bincount(lab_crop.ravel())[1:] > max_area).any():
                progress("Subdividing large segments by colour …")
                lab_crop = S.subdivide_large(lab_crop, crop, max_area, min_area)
            labels = np.zeros(shape, np.int32)
            labels[y0:y1, x0:x1] = lab_crop
            return labels, len(masks)

        def done(res):
            labels, n_masks = res
            self._apply_segmentation(labels, level, keep=self.keep_cb.isChecked(),
                                     msg=f"{int(labels.max())} segments (from {n_masks} SAM masks). "
                                         f"Pick a class and click segments.")

        self._run("Segmenting", job, done)

    def _apply_segmentation(self, labels, level: int, keep: bool, msg: str):
        p = self.project
        old_labels, old_cls = p.labels, p.seg_class
        p.push_undo()
        p.labels = labels
        if keep and old_labels is not None:
            p.seg_class = S.transfer_classes(old_labels, old_cls, labels)
        else:
            p.seg_class = np.zeros(int(labels.max()) + 1, np.int16)
        p.granularity = level
        self.selected.clear()
        self.refresh_all()
        self.status_msg.setText(msg)
        if self.canvas.tool == "rock":
            self.set_tool("assign")

    def clear_segmentation(self):
        p = self.project
        if self.runner.busy or p.labels is None:
            return
        msg = "Remove all segments"
        if p.seg_class.any():
            msg += " and their class assignments"
        if QMessageBox.question(self, "Clear segmentation",
                                msg + "? Cracks and the stone outline are kept. (Undo with Ctrl+Z)") \
                != QMessageBox.Yes:
            return
        p.push_undo()
        p.labels = None
        p.seg_class = np.zeros(1, np.int16)
        self.selected.clear()
        if self.canvas.tool == "split":
            self.set_tool("assign")
        self.refresh_all()
        self.status_msg.setText("Segmentation cleared.")

    # ════════════════════════════════════════════════════════════════════════
    # assign / select / merge / split
    # ════════════════════════════════════════════════════════════════════════
    def _segment_at(self, x, y) -> int:
        if self.project.labels is None:
            return 0
        return int(self.project.labels[y, x])

    def merge_selected(self):
        if not self._require_segments():
            return
        if len(self.selected) < 2:
            self.status_msg.setText("Select at least two segments (Ctrl+click) to merge.")
            return
        p = self.project
        p.ensure_class_capacity()
        ids = sorted(self.selected)
        areas = np.bincount(p.labels.ravel(), minlength=len(p.seg_class))[ids]
        cls = int(p.seg_class[ids[int(np.argmax(areas))]])
        p.push_undo()
        p.labels, target = S.merge_segments(p.labels, ids)
        p.seg_class = p.seg_class.copy()
        p.seg_class[target] = cls
        self.selected = {target}
        self.refresh_all()
        self.status_msg.setText(f"Merged {len(ids)} segments into segment {target}.")

    def _on_split(self, pts):
        if not self._require_segments():
            return
        p = self.project
        stroke = S.stroke_to_mask(p.shape, pts, 2)
        labels, created = S.split_by_stroke(p.labels, stroke)
        if not created:
            self.status_msg.setText("Nothing split – draw the line fully across a segment.")
            return
        p.push_undo()
        p.labels = labels
        p.ensure_class_capacity()
        p.seg_class = p.seg_class.copy()
        for new, parent in created:
            p.seg_class[new] = p.seg_class[parent]
        self.selected.clear()
        self.refresh_all()
        self.status_msg.setText(f"Split created {len(created)} new segment(s).")

    # ════════════════════════════════════════════════════════════════════════
    # cracks and the class brush
    # ════════════════════════════════════════════════════════════════════════
    def _on_brush(self, pts, size, erase):
        if not self._require_image():
            return
        p = self.project
        m = S.stroke_to_mask(p.shape, pts, size)
        p.push_undo()
        p.crack = (p.crack & ~m) if erase else (p.crack | m)
        self._refresh_crack()
        self.app.update_file_actions()

    def _on_paint(self, pts, size, to_stone):
        if not self._require_segments():
            return
        p = self.project
        cls = 0 if to_stone else self.active_class()
        res = S.paint_region(p.labels, p.seg_class, S.stroke_to_mask(p.shape, pts, size), cls)
        if res is None:
            return
        p.push_undo()
        p.labels, p.seg_class, _ = res
        self.selected.clear()
        self.refresh_all()
        name = capitalize_words(p.classes.get(cls, {"name": "?"})["name"])
        self.status_msg.setText(f"Painted {name}.")

    def clear_cracks(self):
        p = self.project
        if p.has_image and p.crack.any():
            p.push_undo()
            p.crack = np.zeros_like(p.crack)
            self._refresh_crack()
            self.app.update_file_actions()

    # ════════════════════════════════════════════════════════════════════════
    # undo / redo
    # ════════════════════════════════════════════════════════════════════════
    def undo(self):
        if not self.runner.busy and self.project.undo():
            self.refresh_all()
            self.status_msg.setText("Undo.")

    def redo(self):
        if not self.runner.busy and self.project.redo():
            self.refresh_all()
            self.status_msg.setText("Redo.")
