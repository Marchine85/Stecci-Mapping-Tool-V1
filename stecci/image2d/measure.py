"""Measurements for the export report (<name>.json): stone, classes, cracks.

All values are computed on the working-resolution masks and converted to pixels of the
original photo (areas × sx·sy, lengths × √(sx·sy)). There is no physical scale (cm) yet.
"""
from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np


# ── crack geometry ───────────────────────────────────────────────────────────
def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: 1-px wide centre line of a binary mask (vectorised)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros_like(mask, bool)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img = np.pad(mask[y0:y1, x0:x1].astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p = img
            n = [np.roll(np.roll(p, dy, 0), dx, 1) for dy, dx in
                 ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))]
            # neighbours P2..P9 clockwise starting north: (y-1,x), (y-1,x+1), (y,x+1), ...
            P2, P3, P4, P5, P6, P7, P8, P9 = n
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                c = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            rm = (p == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                img = img.copy()
                img[rm] = 0
                changed = True
        if not changed:
            break
    out = np.zeros_like(mask, bool)
    out[y0:y1, x0:x1] = img[1:-1, 1:-1] > 0
    return out


def skeleton_length(sk: np.ndarray) -> float:
    """Length of a 1-px skeleton: straight steps count 1, diagonal steps √2."""
    if not sk.any():
        return 0.0
    straight = int((sk[:, 1:] & sk[:, :-1]).sum() + (sk[1:, :] & sk[:-1, :]).sum())
    # diagonal neighbours that are not already joined through a straight neighbour
    d1 = sk[1:, 1:] & sk[:-1, :-1] & ~sk[:-1, 1:] & ~sk[1:, :-1]
    d2 = sk[1:, :-1] & sk[:-1, 1:] & ~sk[:-1, :-1] & ~sk[1:, 1:]
    return straight + np.sqrt(2) * int(d1.sum() + d2.sum())


def crack_stats(crack: np.ndarray, rock: np.ndarray | None, lin: float, area_f: float) -> dict:
    """Cracks inside the stone: count (connected lines), lengths, areas, widths."""
    m = crack & rock if rock is not None else crack.copy()
    stone_px = float(rock.sum()) * area_f if rock is not None else 0.0
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    items = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        comp = cc[y:y + h, x:x + w] == i
        length = max(skeleton_length(thin(comp)), 1.0) * lin
        area = a * area_f
        items.append({"id": len(items) + 1,
                      "length_px": round(length, 1),
                      "area_px": round(area),
                      "mean_width_px": round(area / length, 2),
                      "bbox_xywh_px": [round(x * lin), round(y * lin), round(w * lin), round(h * lin)]})
    items.sort(key=lambda d: -d["length_px"])
    for k, d in enumerate(items, start=1):
        d["id"] = k
    total_len = sum(d["length_px"] for d in items)
    total_area = sum(d["area_px"] for d in items)
    return {
        "count": len(items),
        "total_length_px": round(total_len, 1),
        "area_px": round(total_area),
        "percent_of_stone": round(100 * total_area / stone_px, 3) if stone_px else 0.0,
        "mean_width_px": round(total_area / total_len, 2) if total_len else 0.0,
        "longest_px": items[0]["length_px"] if items else 0.0,
        "length_per_megapixel_of_stone": round(total_len / (stone_px / 1e6), 1) if stone_px else 0.0,
        "items": items,
    }


# ── photo metadata (EXIF) ────────────────────────────────────────────────────
def photo_exif(path) -> dict:
    """Date taken, camera and GPS position from the photo's EXIF data (empty if not available)."""
    out = {}
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769)                       # Exif IFD
            gps = ex.get_ifd(0x8825)                       # GPS IFD
        taken = sub.get(36867) or ex.get(306)              # DateTimeOriginal / DateTime
        if taken:
            out["date_taken"] = str(taken).strip()
        cam = " ".join(str(v).strip() for v in (ex.get(271), ex.get(272)) if v)
        if cam:
            out["camera"] = cam

        def deg(v, ref):
            d, m, s = (float(x) for x in v)
            val = d + m / 60 + s / 3600
            return round(-val if ref in ("S", "W") else val, 7)
        if gps.get(2) and gps.get(4):
            out["gps_lat"] = deg(gps[2], gps.get(1, "N"))
            out["gps_lon"] = deg(gps[4], gps.get(3, "E"))
            if gps.get(6) is not None:
                out["gps_altitude_m"] = round(float(gps[6]), 1)
    except Exception:
        pass
    return out


# ── stone and classes ────────────────────────────────────────────────────────
def stone_stats(rock: np.ndarray, lin: float, area_f: float) -> dict:
    area = float(rock.sum()) * area_f
    contours, _ = cv2.findContours(rock.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours) * lin
    pts = np.concatenate(contours) if contours else np.zeros((0, 1, 2), np.int32)
    x, y, w, h = cv2.boundingRect(pts) if len(pts) else (0, 0, 0, 0)
    (_, _), (rw, rh), angle = cv2.minAreaRect(pts) if len(pts) else ((0, 0), (0, 0), 0)
    return {
        "area_px": round(area),
        "percent_of_photo": round(100 * rock.mean(), 2),
        "bbox_width_px": round(w * lin), "bbox_height_px": round(h * lin),
        "length_px": round(max(rw, rh) * lin), "width_px": round(min(rw, rh) * lin),
        "perimeter_px": round(perimeter),
        "parts": len(contours),
    }


def class_stats(project, lin: float, area_f: float) -> list[dict]:
    rock = project.rock
    stone_px = float(rock.sum()) if rock is not None else 0.0
    cm = project.class_map()
    if rock is not None:
        cm = np.where(rock, cm, -1)
    out = []
    for cid, c in sorted(project.classes.items()):
        m = cm == cid
        area = float(m.sum())
        n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        sizes = st[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
        out.append({"id": cid, "name": c["name"], "color_rgb": list(c["color"]),
                    "area_px": round(area * area_f),
                    "percent_of_stone": round(100 * area / stone_px, 2) if stone_px else 0.0,
                    "patches": int(n - 1),
                    "largest_patch_px": round(float(sizes.max()) * area_f)})
    return out


def build_report(project, name: str, scale: tuple[float, float], target: tuple[int, int], crop, extra: dict) -> dict:
    sx, sy = scale
    lin, area_f = float(np.sqrt(sx * sy)), sx * sy
    p = project
    labels = p.labels
    n_seg = int(len(np.unique(labels[labels > 0]))) if labels is not None else 0
    rep = {
        "name": name,
        "site": extra.get("site", {}),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "units": "pixels of the original photo (no physical scale set)",
        "source": {
            "photo": str(p.image_path) if p.image_path else None,
            "photo_size_px": list(target),
            "working_size_px": [p.shape[1], p.shape[0]],
            **(photo_exif(p.image_path) if p.image_path and p.image_path.is_file() else {}),
            **extra.get("source", {}),
        },
        "export": {
            "cropped": crop is not None,
            "crop_box_xyxy_px": list(crop) if crop is not None else None,
            "image_size_px": [crop[2] - crop[0], crop[3] - crop[1]] if crop is not None else list(target),
        },
        "detection": extra.get("detection", {}),
        "segmentation": {"fidelity": p.granularity if p.granularity else "off", "segments": n_seg},
        "stone": stone_stats(p.rock, lin, area_f) if p.rock is not None else None,
        "classes": class_stats(p, lin, area_f),
        "cracks": crack_stats(p.crack, p.rock, lin, area_f),
    }
    return rep
