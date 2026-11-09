"""Settings dialog: models (both workspaces), photo processing and 3D-scan processing."""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QWidget)

from ..config import SAM_MODELS, SEG_MODES, UI_THEMES, UNWRAP_METHODS, Settings
from ..models import device_description
from . import theme


def _heading(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setStyleSheet("font-weight: 600; font-size: 13px; padding-top: 10px;")
    return lab


class SettingsDialog(QDialog):
    def __init__(self, s: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        f = QFormLayout(self)

        # ── appearance ──────────────────────────────────────────────────────
        f.addRow(_heading("Appearance"))
        self.theme = QComboBox()
        for value, label in UI_THEMES:
            self.theme.addItem(label, value)
        self.theme.setCurrentIndex(max(0, self.theme.findData(s.theme)))
        self.theme.setToolTip("Colours, shapes and fonts of the app. Previewed right away; Cancel restores "
                              "the previous one.")
        self.theme.currentIndexChanged.connect(lambda: theme.apply(self.theme.currentData()))
        f.addRow("Theme", self.theme)

        # ── models ──────────────────────────────────────────────────────────
        f.addRow(_heading("Models  ·  photos and 3D scans"))
        device = QLabel(device_description())       # fixed by the build (GPU / CPU) and the PC
        device.setToolTip("The GPU build uses an NVIDIA GPU when one is present, otherwise the CPU.\n"
                          "The CPU build always runs on the CPU.")
        f.addRow("Device", device)

        self.sam_type = QComboBox()
        for value, label in SAM_MODELS:
            self.sam_type.addItem(label, value)
        self.sam_type.setCurrentIndex(max(0, self.sam_type.findData(s.sam_model_type)))
        self.sam_type.setToolTip("Segmentation model. ViT-B is the default: fastest and smallest download.\n"
                                 "ViT-H gives slightly cleaner edges but is ~3× slower.")
        f.addRow("SAM model", self.sam_type)

        self.ckpt = QLineEdit(s.sam_checkpoint)
        self.ckpt.setPlaceholderText("empty = use the app's models folder")
        b = QPushButton("…")
        b.setFixedWidth(30)
        b.clicked.connect(self._browse)
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.ckpt)
        hl.addWidget(b)
        f.addRow("SAM checkpoint", row)

        self.seg_mode = QComboBox()
        for value, label in SEG_MODES:
            self.seg_mode.addItem(label, value)
        self.seg_mode.setCurrentIndex(max(0, self.seg_mode.findData(s.seg_mode)))
        self.seg_mode.setToolTip("Colour (default): SAM finds the main regions with a few points; large regions\n"
                                 "are then split by colour. Weathering patches follow colour, so this suits\n"
                                 "stećci, and it takes seconds even on a CPU.\n"
                                 "SAM shapes: dense SAM point grids (up to 64×64 + zoomed pass). Borders follow\n"
                                 "physical edges more; ~10 s on a GPU, minutes on a CPU at high fidelity.")
        f.addRow("Segment detail", self.seg_mode)

        self.colour_pts = QSpinBox()
        self.colour_pts.setRange(6, 32)
        self.colour_pts.setValue(s.colour_points)
        self.colour_pts.setToolTip("Colour mode: SAM points per side at fidelity 10 (6 at fidelity 1, linear in "
                                   "between).\nMore points = more SAM regions and longer runs (time grows with "
                                   "points²).")
        f.addRow("SAM points at fidelity 10", self.colour_pts)
        self.seg_mode.currentIndexChanged.connect(
            lambda: self.colour_pts.setEnabled(self.seg_mode.currentData() == "colour"))
        self.colour_pts.setEnabled(s.seg_mode == "colour")

        # ── photos ──────────────────────────────────────────────────────────
        f.addRow(_heading("Photos"))
        self.max_side = QSpinBox()
        self.max_side.setRange(512, 8192)
        self.max_side.setSingleStep(256)
        self.max_side.setValue(s.working_max_side)
        self.max_side.setToolTip("Photos are downscaled to this size for processing.\n"
                                 "Exports are upscaled back to the original resolution.\n"
                                 "Takes effect when a photo is opened.")
        f.addRow("Working max side (px)", self.max_side)

        # ── 3D scans ────────────────────────────────────────────────────────
        f.addRow(_heading("3D scans"))
        self.unwrap = QComboBox()
        for value, label in UNWRAP_METHODS:
            self.unwrap.addItem(label, value)
        self.unwrap.setCurrentIndex(max(0, self.unwrap.findData(s.unwrap_method)))
        f.addRow("Unwrap (default)", self.unwrap)

        self.unwrap_px = QSpinBox()
        self.unwrap_px.setRange(1024, 8192)
        self.unwrap_px.setSingleStep(512)
        self.unwrap_px.setValue(s.unwrap_max_side)
        self.unwrap_px.setToolTip("Size of the hidden unwrap image that is segmented. Larger = finer segments,\n"
                                  "slower colour subdivision.")
        f.addRow("Unwrap resolution (px)", self.unwrap_px)

        self.max_tri = QSpinBox()
        self.max_tri.setRange(0, 20_000_000)
        self.max_tri.setSingleStep(50_000)
        self.max_tri.setGroupSeparatorShown(True)
        self.max_tri.setSpecialValueText("Full resolution")
        self.max_tri.setValue(s.max_triangles)
        self.max_tri.setToolTip("Scans with more triangles are reduced to this number when they are opened:\n"
                                "the mesh is simplified and its texture re-baked from the full scan (once, then "
                                "cached).\nFewer triangles = faster everything, but segment borders and cracks "
                                "follow\nthe coarser triangles. 0 = keep the full resolution.\n"
                                "Projects always reopen with the resolution they were made with.")
        f.addRow("Max triangles on import", self.max_tri)

        self.bake_px = QComboBox()
        for px in (2048, 4096, 8192):
            self.bake_px.addItem(f"{px} × {px} px", px)
        self.bake_px.setCurrentIndex(max(0, self.bake_px.findData(s.bake_texture_px)))
        self.bake_px.setToolTip("Texture baked for a reduced scan. 4096: ~2 mm per texel on a 5 m stone; "
                                "8192: ~1 mm\n(about 4× longer to bake, 4× more GPU memory).")
        f.addRow("Texture of reduced scans", self.bake_px)

        self.tex_px = QSpinBox()
        self.tex_px.setRange(1024, 16384)
        self.tex_px.setSingleStep(1024)
        self.tex_px.setValue(s.max_texture_side)
        self.tex_px.setToolTip("Scan textures are downscaled to this size for display (GPU memory). "
                               "Takes effect when a scan is opened.")
        f.addRow("Display texture size (px)", self.tex_px)

        self.cut = QDoubleSpinBox()
        self.cut.setRange(0, 200)
        self.cut.setDecimals(1)
        self.cut.setValue(s.ground_cut_cm)
        self.cut.setSuffix(" cm")
        self.cut.setToolTip("Default for new scans: faces lower than this above the ground plane count as ground.")
        f.addRow("Ground cut height", self.cut)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self._base = s

    def reject(self):
        theme.apply(self._base.theme)            # undo the live preview
        super().reject()

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "SAM checkpoint", "", "PyTorch checkpoint (*.pth *.pt)")
        if p:
            self.ckpt.setText(p)

    def result_settings(self) -> Settings:
        d = self._base.to_dict()
        d.update(theme=self.theme.currentData(), sam_model_type=self.sam_type.currentData(), sam_checkpoint=self.ckpt.text().strip(),
                 seg_mode=self.seg_mode.currentData(), colour_points=self.colour_pts.value(),
                 working_max_side=self.max_side.value(),
                 unwrap_method=self.unwrap.currentData(), unwrap_max_side=self.unwrap_px.value(),
                 max_texture_side=self.tex_px.value(), ground_cut_cm=self.cut.value(),
                 max_triangles=self.max_tri.value(), bake_texture_px=self.bake_px.currentData())
        s = Settings.from_dict(d)
        s.use_dino = self._base.use_dino            # from_dict never restores it (no UI switch)
        return s
