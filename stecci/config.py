"""Paths, defaults and persisted settings (shared by the photo and the 3D-scan workspace)."""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

APP_NAME = "Stecci Mapping Tool"
APP_VERSION = "1.0.0"


def app_dir() -> Path:
    """Directory of the app (works both from source and from a frozen exe)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def cache_dir() -> Path:
    """Per-user cache for converted 3D scans (mesh arrays, textures, topology).

    Same folder as the stand-alone 3D tool, so scans it already converted open quickly here too.
    """
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    d = base / "Stecci3D" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


SAM_FILES = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/"


def candidate_model_dirs() -> list[Path]:
    """Only the app's own models folder: the project is self-contained (no sibling project folders)."""
    return [app_dir() / "models"]


def find_sam_checkpoint(model_type: str) -> Path | None:
    name = SAM_FILES[model_type]
    for d in candidate_model_dirs():
        p = d / name
        if p.is_file():
            return p
    return None


# ── file types ───────────────────────────────────────────────────────────────
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
SCAN_EXTS = (".glb", ".gltf", ".obj", ".ply")
PROJECT_EXT_2D = ".stecci"
PROJECT_EXT_3D = ".stecci3d"


def file_kind(path: str | Path) -> str | None:
    """'photo', 'scan', 'project2d', 'project3d' or None (unsupported)."""
    ext = Path(path).suffix.lower()
    if ext == PROJECT_EXT_2D:
        return "project2d"
    if ext == PROJECT_EXT_3D:
        return "project3d"
    if ext in IMAGE_EXTS:
        return "photo"
    if ext in SCAN_EXTS:
        return "scan"
    return None


# ── segmentation fidelity ────────────────────────────────────────────────────
@dataclass(frozen=True)
class Granularity:
    points_per_side: int      # SAM point grid density
    min_frac: float           # min segment area as fraction of the stone area
    iou_thresh: float         # SAM predicted-IoU filter (lower = more masks)
    stability: float          # SAM stability filter (lower = more masks)
    crop_layers: int          # 1 = additional zoomed-in pass on 2x2 crops (finds small patches)
    max_frac: float           # segments larger than this fraction of the stone are subdivided by colour


# level 1 = coarse … 10 = fine
GRANULARITY = {
    1: Granularity(8, 0.030, 0.88, 0.92, 0, 0.60),
    2: Granularity(12, 0.020, 0.86, 0.92, 0, 0.40),
    3: Granularity(16, 0.012, 0.86, 0.90, 0, 0.25),
    4: Granularity(20, 0.008, 0.84, 0.90, 0, 0.16),
    5: Granularity(24, 0.005, 0.82, 0.88, 0, 0.10),
    6: Granularity(32, 0.003, 0.80, 0.86, 0, 0.07),
    7: Granularity(32, 0.002, 0.78, 0.85, 1, 0.05),
    8: Granularity(40, 0.0012, 0.76, 0.84, 1, 0.035),
    9: Granularity(48, 0.0007, 0.74, 0.82, 1, 0.025),
    10: Granularity(64, 0.0004, 0.70, 0.80, 1, 0.015),
}

# Segment detail source
#   "colour" (default): few SAM points, no zoomed pass; the colour subdivision (max_frac) supplies the fine detail.
#            Suits stećci (weathering patches follow colour, not object shapes) and is fast on CPU too.
#   "sam":   the dense SAM grids above + zoomed pass at 7–10 (detailed shapes, minutes per run on CPU).
SEG_MODES = [
    ("colour", "Colour  ·  few SAM points, detail from colour (default, fast)"),
    ("sam", "SAM shapes  ·  dense SAM points (slow on CPU)"),
]
COLOUR_POINTS_MIN = 6           # SAM points per side at fidelity 1 in colour mode


def colour_points(level: int, max_points: int) -> int:
    """SAM points per side in colour mode: 6 at fidelity 1 rising linearly to max_points at 10."""
    lo = min(COLOUR_POINTS_MIN, max_points)
    return round(lo + (max_points - lo) * (level - 1) / 9)


def granularity(level: int, mode: str = "colour", max_points: int = 16) -> Granularity:
    g = GRANULARITY[level]
    if mode == "sam":
        return g
    return replace(g, points_per_side=colour_points(level, max_points), crop_layers=0)


# How a 3D stone surface is flattened for the (hidden) 2D segmentation
UNWRAP_METHODS = [
    ("arap", "ARAP  ·  one piece, no seams (default)"),
    ("boxnet", "Box net  ·  5 straight-on views, true-scale walls"),
    ("belt", "Belt  ·  top + walls rolled out in one strip"),
]

# Grounding DINO prompt used to locate the stecci in a photo (fixed: it is always a rock / stone)
ROCK_PROMPT = "rock. stone."

DEFAULT_CLASSES = [
    # id, name, RGB
    (0, "Unassigned/Stone", (128, 128, 128)),
    (1, "Biological Colonization", (60, 175, 70)),     # green
    (2, "Microkarst", (120, 195, 240)),                # light blue
    (3, "Fragmentation", (255, 170, 200)),             # light pink
    (4, "Pitting", (185, 145, 235)),                   # lilac
]

# (value, label) – first entry is the default: fastest and smallest
SAM_MODELS = [
    ("vit_b", "SAM ViT-B  ·  fastest, smallest (default)"),
    ("vit_l", "SAM ViT-L  ·  medium"),
    ("vit_h", "SAM ViT-H  ·  best quality, slowest"),
]
GDINO_MODELS = [
    ("IDEA-Research/grounding-dino-tiny", "Grounding DINO Tiny  ·  fastest, smallest (default)"),
    ("IDEA-Research/grounding-dino-base", "Grounding DINO Base  ·  more accurate, slower"),
]

# (value, label) – first entry is the default; colours / shapes / fonts live in ui/theme.py
UI_THEMES = [
    ("nocturne", "Nocturne  ·  deep navy, violet (default)"),
    ("classic", "Classic  ·  dark grey, blue"),
    ("nordic", "Nordic Light  ·  light grey, teal"),
    ("graphite", "Graphite Pro  ·  black, amber"),
]

# loading animation of the busy overlay (ui/busy_overlay.py); first entry is the default
SPINNERS = [
    ("grid", "Tile grid  ·  3×3 tiles flipping colours (default)"),
    ("rings", "Rotating rings"),
]

SETTINGS_VERSION = 2              # 2: Nocturne became the default theme


@dataclass
class Settings:
    version: int = SETTINGS_VERSION
    theme: str = UI_THEMES[0][0]    # appearance (ui/theme.py)
    spinner: str = SPINNERS[0][0]    # loading animation
    # ── models (both workspaces) ──
    sam_model_type: str = SAM_MODELS[0][0]
    sam_checkpoint: str = ""        # empty = search known folders
    seg_mode: str = SEG_MODES[0][0]  # colour | sam  (see SEG_MODES)
    colour_points: int = 16         # colour mode: SAM points per side at fidelity 10
    # ── photos ──
    # Grounding DINO detection is kept in the backend but not offered in the UI (and not shipped in the builds).
    # Off: SAM isolates the stone from the photo centre. Saved settings cannot switch it on (see from_dict);
    # set the default to True here to use it again (needs transformers + the DINO model files).
    use_dino: bool = False
    gdino_model_id: str = GDINO_MODELS[0][0]
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    working_max_side: int = 2048    # photos are processed at this max resolution
    # ── 3D scans ──
    unwrap_method: str = UNWRAP_METHODS[0][0]
    unwrap_max_side: int = 3072     # px of the hidden unwrap image (SAM runs on <= 2048 px of it)
    max_triangles: int = 500_000    # scans are reduced to this on import (0 = keep full resolution)
    bake_texture_px: int = 4096     # texture baked for a reduced scan (px per side)
    max_texture_side: int = 8192    # scan textures are downscaled to this for display
    ground_cut_cm: float = 4.0      # faces lower than this above the ground plane count as ground

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        s = cls()
        try:
            old = int(d.get("version", 1))
        except (TypeError, ValueError):
            old = 1
        for k, v in d.items():
            if k == "version":
                continue
            if k == "theme" and old < 2:        # saved before Nocturne became the default: use the new default
                continue
            if k == "sam_model_type" and v not in dict(SAM_MODELS):
                continue
            if k == "unwrap_method" and v not in dict(UNWRAP_METHODS):
                continue
            if k == "theme" and v not in dict(UI_THEMES):
                continue
            if k == "spinner" and v not in dict(SPINNERS):
                continue
            if k == "use_dino":                 # no UI switch: an old saved "on" must not stick
                continue
            if hasattr(s, k):
                default = getattr(s, k)
                try:
                    if isinstance(default, bool):
                        v = v in (True, "true", "True", 1, "1")
                    else:
                        v = type(default)(v)
                except (TypeError, ValueError):
                    continue
                setattr(s, k, v)
        return s
