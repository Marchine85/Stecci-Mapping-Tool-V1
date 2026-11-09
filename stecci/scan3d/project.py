"""Document state: scan, isolated stone, segments, classes, cracks; undo; save / load; measurements.

Per-face arrays (length = number of scan faces) are never modified in place once they are part of the state
(edits build new arrays), so undo snapshots can share them and compress each array only once.
"""
from __future__ import annotations

import copy
import json
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import DEFAULT_CLASSES
from .isolate import Isolation
from .mesh import Scan, face_components

PROJECT_EXT = ".stecci3d"


def _pack(a):
    if a is None:
        return None
    return (a.dtype.str, a.shape, zlib.compress(np.ascontiguousarray(a).tobytes(), 1))


def _unpack(p):
    if p is None:
        return None
    dt, shape, data = p
    return np.frombuffer(zlib.decompress(data), dtype=np.dtype(dt)).reshape(shape).copy()


class Project:
    STATE = ("stone", "seg", "seg_class", "crack")
    MAX_UNDO = 30

    def __init__(self):
        self.scan: Scan | None = None
        self.iso: Isolation | None = None
        self.seg: np.ndarray | None = None              # int32 per face, 0 = none
        self.seg_class = np.zeros(1, np.int16)          # segment id -> class id
        self.crack: np.ndarray | None = None            # bool per face
        self.classes: dict[int, dict] = {}
        self.granularity = 7
        self.unwrap_method = ""                         # method used for the current segments
        self.cut_cm = 4.0
        self.scale = 1.0                                # metres per model unit
        self.scale_set = False
        self.export_name = ""
        self.site_info: dict = {}
        self.legend_scale_pct = 100
        self.refined = 0                                # number of refine strokes on the isolation
        self.max_triangles = 0                          # reduction the scan was opened with (0 = full)
        self.bake_texture_px = 4096
        self._undo: list = []
        self._redo: list = []
        self._packed: dict = {}
        self.dirty = False
        self.reset_classes()

    # ── basics ───────────────────────────────────────────────────────────────
    @property
    def has_scan(self) -> bool:
        return self.scan is not None

    @property
    def stone(self) -> np.ndarray | None:
        return self.iso.stone if self.iso is not None else None

    @stone.setter
    def stone(self, value):
        if self.iso is not None and value is not None:
            self.iso.stone = value

    def reset_classes(self):
        self.classes = {cid: {"name": n, "color": list(c)} for cid, n, c in DEFAULT_CLASSES}

    def set_scan(self, scan: Scan):
        self.scan = scan
        self.iso = None
        self.seg = None
        self.seg_class = np.zeros(1, np.int16)
        self.crack = np.zeros(scan.n_faces, bool)
        self.unwrap_method = ""
        self.export_name = ""
        self.site_info = {}
        self.scale, self.scale_set = 1.0, False
        self.refined = 0
        self._undo.clear()
        self._redo.clear()
        self._packed.clear()
        self.dirty = False

    def ensure_class_capacity(self):
        n = int(self.seg.max()) + 1 if self.seg is not None else 1
        if len(self.seg_class) < n:
            self.seg_class = np.concatenate([self.seg_class, np.zeros(n - len(self.seg_class), np.int16)])

    def class_faces(self) -> np.ndarray:
        """Class id per face (0 where unassigned or no segment)."""
        if self.seg is None:
            return np.zeros(self.scan.n_faces, np.int16)
        self.ensure_class_capacity()
        return self.seg_class[self.seg]

    def next_class_id(self) -> int:
        return max(self.classes) + 1

    # ── undo / redo ──────────────────────────────────────────────────────────
    def _pack_cached(self, name, a):
        if a is None:
            return None
        if name == "seg_class":
            return _pack(a)
        hit = self._packed.get(name)
        if hit is not None and hit[0] is a:
            return hit[1]
        p = _pack(a)
        self._packed[name] = (a, p)                     # keeps `a` alive, so its id cannot be reused
        return p

    def _snapshot(self):
        return ({k: self._pack_cached(k, getattr(self, k)) for k in self.STATE}, copy.deepcopy(self.classes))

    def _restore(self, snap):
        arrays, classes = snap
        for k, v in arrays.items():
            setattr(self, k, _unpack(v))
        self.classes = classes
        self._packed.clear()

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

    # ── measurements ─────────────────────────────────────────────────────────
    @property
    def unit(self) -> str:
        return "m" if self.scale_set else "units"

    def area(self) -> np.ndarray:
        """Face areas in m² (or model units² while no scale is set)."""
        return self.scan.area * self.scale ** 2

    def stone_area(self) -> float:
        return float(self.area()[self.stone].sum()) if self.stone is not None else 0.0

    def class_areas(self) -> dict[int, float]:
        if self.seg is None or self.stone is None:
            return {}
        cf = self.class_faces()[self.stone].astype(np.int64)
        n = max(max(self.classes) + 1, int(cf.max()) + 1 if cf.size else 1)
        a = np.bincount(cf, self.area()[self.stone], minlength=n)
        return {cid: float(a[cid]) for cid in self.classes}

    def extents(self) -> np.ndarray:
        L = self.scan.V[np.unique(self.scan.F[self.stone])].astype(np.float64) @ self.iso.R.T
        return (L.max(0) - L.min(0)) * self.scale

    def report(self) -> dict:
        s = self.scan
        area = self.area()
        stone = self.stone
        total = self.stone_area()
        u = self.unit
        ext = self.extents()
        rep = {
            "units": "metres" if self.scale_set else "model units (no scale set: areas in units², lengths in units)",
            "scale_m_per_unit": self.scale if self.scale_set else None,
            "stone": {f"area_{u}2": total, f"length_{u}": float(ext[0]), f"width_{u}": float(ext[1]),
                      f"height_{u}": float(ext[2]), "faces": int(stone.sum())},
            "segmentation": {"unwrap": self.unwrap_method or None, "fidelity": self.granularity,
                             "segments": int(self.seg.max()) if self.seg is not None else 0},
            "classes": [],
        }
        if self.seg is not None:
            cf = self.class_faces()
            areas = self.class_areas()
            comp = face_components(s, stone, same=cf)
            m = comp >= 0
            ccls = np.zeros(int(comp.max()) + 1, np.int64)
            ccls[comp[m]] = cf[m]
            for cid, c in sorted(self.classes.items()):
                a = areas.get(cid, 0.0)
                rep["classes"].append({"id": cid, "name": c["name"], "color": c["color"], f"area_{u}2": a,
                                       "percent_of_stone": 100 * a / total if total else 0.0,
                                       "patches": int((ccls == cid).sum())})
        cr = self.crack & stone
        n_cracks = int(face_components(s, cr).max()) + 1 if cr.any() else 0
        ca = float(area[cr].sum())
        rep["cracks"] = {"count": n_cracks, f"area_{u}2": ca, "percent_of_stone": 100 * ca / total if total else 0.0}
        return rep

    # ── save / load ──────────────────────────────────────────────────────────
    def save(self, path: str | Path, camera: dict | None = None):
        iso = self.iso
        meta = {
            "version": 1,
            "scan_path": str(self.scan.path),
            "n_faces": self.scan.n_faces,
            "max_triangles": self.max_triangles, "bake_texture_px": self.bake_texture_px,
            "classes": {str(k): v for k, v in self.classes.items()},
            "granularity": self.granularity, "unwrap_method": self.unwrap_method, "cut_cm": self.cut_cm,
            "scale": self.scale, "scale_set": self.scale_set,
            "export_name": self.export_name, "site_info": self.site_info,
            "legend_scale_pct": self.legend_scale_pct, "refined": self.refined,
            "camera": camera,
            "saved": datetime.now().isoformat(timespec="seconds"),
        }
        arrays = {"meta": np.frombuffer(json.dumps(meta).encode(), np.uint8),
                  "seg_class": self.seg_class, "crack": np.packbits(self.crack)}
        if iso is not None:
            arrays.update(stone=np.packbits(iso.stone), R=iso.R, origin=iso.origin, up=iso.up)
            if iso.ground is not None:
                arrays.update(ground_c=iso.ground[0], ground_n=iso.ground[1])
        if self.seg is not None:
            arrays["seg"] = self.seg
        with open(path, "wb") as f:
            np.savez_compressed(f, **arrays)
        self.dirty = False

    @staticmethod
    def read_meta(path: str | Path) -> dict:
        with np.load(path, allow_pickle=False) as z:
            return json.loads(bytes(z["meta"]).decode())

    def load(self, path: str | Path, scan: Scan):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode())
            m = scan.n_faces
            if meta.get("n_faces") != m:
                raise ValueError("The scan does not match the project (different number of triangles).")
            self.set_scan(scan)
            if "stone" in z:
                ground = (z["ground_c"], z["ground_n"]) if "ground_c" in z else None
                self.iso = Isolation(np.unpackbits(z["stone"])[:m].astype(bool), z["R"], z["origin"], ground, z["up"])
            self.seg = z["seg"].astype(np.int32) if "seg" in z else None
            self.seg_class = z["seg_class"].astype(np.int16)
            self.crack = np.unpackbits(z["crack"])[:m].astype(bool)
        self.classes = {int(k): v for k, v in meta["classes"].items()}
        self.granularity = int(meta.get("granularity", 7))
        self.unwrap_method = meta.get("unwrap_method", "")
        self.cut_cm = float(meta.get("cut_cm", 4.0))
        self.scale, self.scale_set = float(meta.get("scale", 1.0)), bool(meta.get("scale_set", False))
        self.export_name = meta.get("export_name", "")
        self.site_info = dict(meta.get("site_info", {}))
        self.legend_scale_pct = int(meta.get("legend_scale_pct", 100))
        self.refined = int(meta.get("refined", 0))
        self.max_triangles = int(meta.get("max_triangles", 0))
        self.bake_texture_px = int(meta.get("bake_texture_px", 4096))
        self.dirty = False
        return meta
