"""Document state: image, rock mask, segments, classes, cracks; undo; save / load / export."""
from __future__ import annotations

import copy
import json
import zlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..config import DEFAULT_CLASSES
from .measure import build_report
from ..segmentation import padded_box, place_legend

CROP_MARGIN = 0.10      # crop box = stećak bounding box + 10 % of its size on every side


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load as RGB uint8, honouring EXIF orientation (phone photos)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.array(im.convert("RGB"))


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _pack(a: np.ndarray | None):
    if a is None:
        return None
    return (a.dtype.str, a.shape, zlib.compress(np.ascontiguousarray(a).tobytes(), 1))


def _unpack(p):
    if p is None:
        return None
    dt, shape, data = p
    return np.frombuffer(zlib.decompress(data), dtype=np.dtype(dt)).reshape(shape).copy()


class Project:
    STATE_ARRAYS = ("rock", "labels", "seg_class", "crack")
    MAX_UNDO = 40

    def __init__(self):
        self.image_path: Path | None = None
        self.original_size: tuple[int, int] | None = None   # (w, h)
        self.image: np.ndarray | None = None                # working-resolution RGB
        self.rock: np.ndarray | None = None                 # bool
        self.labels: np.ndarray | None = None               # int32, 0 = none
        self.seg_class: np.ndarray = np.zeros(1, np.int16)  # segment id -> class id
        self.crack: np.ndarray | None = None                # bool
        self.classes: dict[int, dict] = {}
        self.rock_box: list[float] | None = None
        self.rock_points: list[tuple[float, float, int]] = []   # x, y, 1=include / 0=exclude
        self.rock_box_user = False      # rock_box was drawn by the user (Refine Outline): use it as the crop box
        self.granularity = 5
        self.crop_to_rock = True        # crop to crop_box() on export (the project keeps the full photo)
        self.legend_wh: tuple[int, int] | None = None   # legend size in working px (set by the UI), None = no legend
        self.legend_margin = 0
        self.legend_user_pos: tuple[int, int] | None = None   # dragged by the user (working px), None = automatic
        self.legend_scale_pct = 100                            # legend size relative to the automatic size
        self.export_name = ""                                  # stećak name tag = base name of the exported files
        self.detection_method = ""                             # how the stone outline was found (for the report)
        self.site_info: dict = {}                              # site, side, survey date, surveyor, notes
        self._undo: list = []
        self._redo: list = []
        self.dirty = False
        self.reset_classes()

    # ── basics ───────────────────────────────────────────────────────────────
    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def shape(self):
        return self.image.shape[:2]

    @property
    def image_key(self):
        return (str(self.image_path), self.image.shape, id(self.image))

    def crop_box(self) -> tuple[int, int, int, int] | None:
        """Box around the stećak (working px, exclusive), grown so the legend fits without touching the stone.

        May extend beyond the photo; the export pads those parts.
        """
        lay = self._legend_layout()
        return lay[0] if lay else self.base_crop_box()

    def legend_pos(self) -> tuple[int, int] | None:
        lay = self._legend_layout()
        return lay[1] if lay else None

    def _legend_layout(self):
        base = self.base_crop_box()
        if base is None or not self.legend_wh:
            return None
        if self.legend_user_pos is not None:          # dragged: the box grows to include it (+ margin)
            lx, ly = self.legend_user_pos
            (lw, lh), m = self.legend_wh, self.legend_margin
            box = (min(base[0], lx - m), min(base[1], ly - m), max(base[2], lx + lw + m), max(base[3], ly + lh + m))
            return box, (lx, ly)
        return place_legend(self.rock, base, *self.legend_wh, self.legend_margin)

    def base_crop_box(self) -> tuple[int, int, int, int] | None:
        """Box without the legend: drawn refine box, or stećak + 10 % margin."""
        if self.rock is None or not self.rock.any():
            return None
        if self.rock_box_user and self.rock_box is not None:      # drawn box: exactly as drawn
            h, w = self.shape
            x0, y0, x1, y1 = self.rock_box
            return (int(max(0, np.floor(x0))), int(max(0, np.floor(y0))),
                    int(min(w, np.ceil(x1) + 1)), int(min(h, np.ceil(y1) + 1)))
        return padded_box(self.rock, CROP_MARGIN)                  # default: stećak + 10 % margin

    def reset_classes(self):
        self.classes = {cid: {"name": n, "color": list(c)} for cid, n, c in DEFAULT_CLASSES}

    def open_image(self, path: str | Path, max_side: int):
        full = load_image_rgb(path)
        self.image_path = Path(path)
        self.original_size = (full.shape[1], full.shape[0])
        self.image = resize_max_side(full, max_side)
        h, w = self.shape
        self.rock = None
        self.labels = None
        self.seg_class = np.zeros(1, np.int16)
        self.crack = np.zeros((h, w), bool)
        self.rock_box = None
        self.rock_points = []
        self.rock_box_user = False
        self.crop_to_rock = True
        self.legend_user_pos = None
        self.legend_scale_pct = 100
        self.export_name = ""
        self.detection_method = ""
        self.site_info = {}
        self._undo.clear()
        self._redo.clear()
        self.dirty = False

    def ensure_class_capacity(self):
        n = int(self.labels.max()) + 1 if self.labels is not None else 1
        if len(self.seg_class) < n:
            self.seg_class = np.concatenate([self.seg_class, np.zeros(n - len(self.seg_class), np.int16)])

    def class_map(self) -> np.ndarray:
        """Per-pixel class id (0 where unassigned or no segment)."""
        if self.labels is None:
            return np.zeros(self.shape, np.int16)
        self.ensure_class_capacity()
        return self.seg_class[self.labels]

    def next_class_id(self) -> int:
        return max(self.classes) + 1

    # ── undo / redo ──────────────────────────────────────────────────────────
    def _snapshot(self):
        return ({k: _pack(getattr(self, k)) for k in self.STATE_ARRAYS},
                copy.deepcopy(self.classes), copy.deepcopy(self.rock_box),
                list(self.rock_points), self.rock_box_user)

    def _restore(self, snap):
        arrays, classes, box, pts, user_box = snap
        for k, v in arrays.items():
            setattr(self, k, _unpack(v))
        self.classes, self.rock_box, self.rock_points, self.rock_box_user = classes, box, pts, user_box

    def push_undo(self):
        self._undo.append(self._snapshot())
        if len(self._undo) > self.MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()
        self.dirty = True

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    # ── statistics ───────────────────────────────────────────────────────────
    def class_areas(self) -> dict[int, int]:
        if self.labels is None or self.rock is None:
            return {}
        cm = self.class_map()[self.rock]
        n = max(max(self.classes) + 1, int(cm.max()) + 1 if cm.size else 1)
        counts = np.bincount(cm.astype(np.int64), minlength=n)
        return {cid: int(counts[cid]) for cid in self.classes}

    # ── save / load project ──────────────────────────────────────────────────
    def save(self, path: str | Path):
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
        meta = {
            "version": 1,
            "image_path": str(self.image_path) if self.image_path else "",
            "original_size": list(self.original_size) if self.original_size else None,
            "classes": {str(k): v for k, v in self.classes.items()},
            "rock_box": self.rock_box,
            "rock_points": self.rock_points,
            "rock_box_user": self.rock_box_user,
            "granularity": self.granularity,
            "crop_to_rock": self.crop_to_rock,
            "legend_user_pos": list(self.legend_user_pos) if self.legend_user_pos else None,
            "legend_scale_pct": self.legend_scale_pct,
            "export_name": self.export_name,
            "detection_method": self.detection_method,
            "site_info": self.site_info,
        }
        arrays = {"image_jpg": jpg, "meta": np.frombuffer(json.dumps(meta).encode(), np.uint8),
                  "seg_class": self.seg_class, "crack": self.crack}
        if self.rock is not None:
            arrays["rock"] = self.rock
        if self.labels is not None:
            arrays["labels"] = self.labels
        with open(path, "wb") as f:
            np.savez_compressed(f, **arrays)
        self.dirty = False

    def load(self, path: str | Path, max_side: int):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode())
            img_path = Path(meta["image_path"]) if meta.get("image_path") else None
            image = cv2.cvtColor(cv2.imdecode(z["image_jpg"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            self.image_path = img_path
            self.original_size = tuple(meta["original_size"]) if meta.get("original_size") else None
            # prefer the full-quality original when it is still available at the same size
            if img_path and img_path.is_file():
                try:
                    full = resize_max_side(load_image_rgb(img_path), max(image.shape[:2]))
                    if full.shape == image.shape:
                        image = full
                except Exception:
                    pass
            self.image = image
            self.rock = z["rock"].astype(bool) if "rock" in z else None
            self.labels = z["labels"].astype(np.int32) if "labels" in z else None
            self.seg_class = z["seg_class"].astype(np.int16)
            self.crack = z["crack"].astype(bool)
        self.classes = {int(k): v for k, v in meta["classes"].items()}
        self.rock_box = meta.get("rock_box")
        self.rock_points = [tuple(p) for p in meta.get("rock_points", [])]
        self.rock_box_user = bool(meta.get("rock_box_user", False))
        self.granularity = meta.get("granularity", 5)
        self.crop_to_rock = bool(meta.get("crop_to_rock", True))
        lp = meta.get("legend_user_pos")
        self.legend_user_pos = (int(lp[0]), int(lp[1])) if lp else None
        self.legend_scale_pct = int(meta.get("legend_scale_pct", 100))
        self.export_name = meta.get("export_name", "")
        self.detection_method = meta.get("detection_method", "")
        self.site_info = dict(meta.get("site_info", {}))
        self._undo.clear()
        self._redo.clear()
        self.dirty = False

    # ── export ───────────────────────────────────────────────────────────────
    def export(self, out_dir: str | Path, name: str, images: list[dict], stone_on_white: bool = True,
               full_resolution: bool = True, report_extra: dict | None = None) -> list[Path]:
        """Write the export images as <name><suffix>.png into out_dir.

        images: dicts with
          suffix   file name suffix ("" = as on screen)
          rgb      working-resolution RGB image
          pad      RGB used where the crop box reaches beyond the photo
          legend   optional (rgba image at export resolution, (x, y) top-left in working px)
        stone_on_white: also write <name>_stone.png (the stone photo on white), from the original photo.
        report_extra: if given, also write <name>.json (measurements, see measure.py) with these extra fields.
        Everything is upscaled to the original resolution and cropped to crop_box() when cropping is on.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        h, w = self.shape
        target = (w, h)
        if full_resolution and self.original_size and tuple(self.original_size) != (w, h):
            target = tuple(self.original_size)
        sx, sy = target[0] / w, target[1] / h
        crop = None
        box = self.crop_box() if self.crop_to_rock else None
        if box is not None:
            crop = (int(np.floor(box[0] * sx)), int(np.floor(box[1] * sy)),
                    int(np.ceil(box[2] * sx)), int(np.ceil(box[3] * sy)))

        def crop_pad(a, fill):
            """Cut out the crop box; parts beyond the photo get `fill`."""
            if crop is None:
                return a
            x0, y0, x1, y1 = crop
            H, W = a.shape[:2]
            out_a = np.empty((y1 - y0, x1 - x0) + a.shape[2:], a.dtype)
            out_a[...] = fill
            sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if sx0 < sx1 and sy0 < sy1:
                out_a[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
            return out_a

        def up(a, interp):
            return a if (a.shape[1], a.shape[0]) == target else cv2.resize(a, target, interpolation=interp)

        def paste(img, rgba, pos):
            lx, ly = round(pos[0] * sx), round(pos[1] * sy)
            if crop is not None:
                lx, ly = lx - crop[0], ly - crop[1]
            H, W = img.shape[:2]
            ax0, ay0 = max(0, lx), max(0, ly)
            ax1, ay1 = min(W, lx + rgba.shape[1]), min(H, ly + rgba.shape[0])
            if ax0 < ax1 and ay0 < ay1:
                part = rgba[ay0 - ly:ay1 - ly, ax0 - lx:ax1 - lx].astype(np.float32)
                a = part[..., 3:4] / 255
                dst = img[ay0:ay1, ax0:ax1].astype(np.float32)
                img[ay0:ay1, ax0:ax1] = (part[..., :3] * a + dst * (1 - a)).astype(np.uint8)

        def write(suffix, rgb):
            path = out_dir / f"{name}{suffix}.png"
            ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                raise IOError(f"could not encode {path.name}")
            path.write_bytes(buf.tobytes())        # works with non-ASCII names (e.g. ć) on Windows
            return path

        written = []
        for spec in images:
            img = crop_pad(up(spec["rgb"], cv2.INTER_LINEAR), spec.get("pad", (0, 0, 0)))
            if spec.get("legend") is not None:
                paste(img, *spec["legend"])
            written.append(write(spec["suffix"], img))

        if stone_on_white and self.rock is not None:
            base = None
            if self.image_path and self.image_path.is_file():
                try:
                    full = load_image_rgb(self.image_path)
                    if (full.shape[1], full.shape[0]) == target:
                        base = full
                except Exception:
                    base = None
            if base is None:
                base = up(self.image, cv2.INTER_LINEAR)
            stone = base.copy()
            stone[~(up(self.rock.astype(np.uint8), cv2.INTER_NEAREST) > 0)] = 255
            written.append(write("_stone", crop_pad(stone, (255, 255, 255))))

        if report_extra is not None:
            rep = build_report(self, name, (sx, sy), target, crop, report_extra)
            path = out_dir / f"{name}.json"
            rep["export"]["files"] = [f.name for f in written] + [path.name]
            path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
            written.append(path)
        return written
