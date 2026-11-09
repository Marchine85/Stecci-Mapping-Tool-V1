"""Pure numpy/OpenCV operations on label maps.

Conventions
-----------
labels : int32 (H, W). 0 = no segment (outside rock), 1..N = segment ids.
rock   : bool  (H, W).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


# ── rock mask cleanup ────────────────────────────────────────────────────────
def clean_rock_mask(mask: np.ndarray, fill_holes: bool = True) -> np.ndarray:
    """Keep the largest connected component and optionally fill holes."""
    m = mask.astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = cc == largest
    if fill_holes:
        out = ndi.binary_fill_holes(out)
    return out


def bbox_of(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    return (max(0, xs.min() - pad), max(0, ys.min() - pad),
            min(w, xs.max() + 1 + pad), min(h, ys.max() + 1 + pad))


# ── building the label map ───────────────────────────────────────────────────
def padded_box(mask: np.ndarray, frac: float = 0.10) -> tuple[int, int, int, int] | None:
    """Bounding box of the mask (exclusive x1/y1) grown by `frac` of its width / height on every side."""
    b = bbox_of(mask)
    if b is None:
        return None
    x0, y0, x1, y1 = b
    dx, dy = round(frac * (x1 - x0)), round(frac * (y1 - y0))
    h, w = mask.shape
    return int(max(0, x0 - dx)), int(max(0, y0 - dy)), int(min(w, x1 + dx)), int(min(h, y1 + dy))


def place_legend(rock: np.ndarray, box, lw: int, lh: int, margin: int):
    """Put a lw×lh legend in the top-right corner of `box` without touching the stone.

    The legend keeps `margin` px to the box edges and to the stone. If it would overlap the stone,
    the box's top-right corner is pushed outwards: to the right or upwards, whichever adds less area.
    Returns (box, (legend_x, legend_y)); the box may extend beyond the photo.
    """
    x0, y0, x1, y1 = box
    h, w = rock.shape
    lx, ly = x1 - margin - lw, y0 + margin

    def hits(ax, ay):          # stone pixels inside the legend rectangle grown by the margin
        ra, rb = max(0, ay - margin), min(h, ay + lh + margin)
        ca, cb = max(0, ax - margin), min(w, ax + lw + margin)
        return ra < rb and ca < cb and bool(rock[ra:rb, ca:cb].any())

    if not hits(lx, ly):
        return (x0, y0, x1, y1), (lx, ly)
    # option A: shift right until the legend is right of the stone in its rows
    ra, rb = max(0, ly - margin), min(h, ly + lh + margin)
    cols = np.flatnonzero(rock[ra:rb].any(0)) if ra < rb else np.array([], int)
    ax = int(cols.max()) + 1 + margin if cols.size else lx
    box_a = (x0, y0, max(x1, ax + lw + margin), y1)
    # option B: shift up until the legend is above the stone in its columns
    ca, cb = max(0, lx - margin), min(w, lx + lw + margin)
    rows = np.flatnonzero(rock[:, ca:cb].any(1)) if ca < cb else np.array([], int)
    by = int(rows.min()) - margin - lh if rows.size else ly
    box_b = (x0, min(y0, by - margin), x1, y1)
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    if area(box_a) <= area(box_b):
        return box_a, (ax, ly)
    return box_b, (lx, by)


def masks_to_labels(masks: list[np.ndarray], rock: np.ndarray, min_area: int) -> np.ndarray:
    """Flatten overlapping SAM masks into a partition of the rock.

    Larger masks are painted first, so smaller (more specific) masks win.
    Disconnected pieces become separate segments, tiny fragments are absorbed
    by their neighbours and uncovered rock areas become their own segments.
    """
    labels = np.zeros(rock.shape, np.int32)
    order = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))
    for k, i in enumerate(order, start=1):
        labels[masks[i] & rock] = k
    labels = split_disconnected(labels, min_area)
    labels = fill_gaps(labels, rock, min_area)
    return relabel_sequential(labels)


def split_disconnected(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Give every connected piece its own id; pieces smaller than min_area -> 0."""
    out = np.zeros_like(labels)
    next_id = 1
    for sid, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        sub = labels[sl] == sid
        n, cc, stats, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=4)
        osub = out[sl]
        for c in range(1, n):
            if stats[c, cv2.CC_STAT_AREA] < min_area:
                continue
            osub[cc == c] = next_id
            next_id += 1
    return out


def absorb_into_nearest(labels: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Assign pixels in `holes` the label of the nearest labelled pixel (labels>0 & ~holes)."""
    if not holes.any():
        return labels
    src = (labels > 0) & ~holes
    if not src.any():
        return labels
    bb = bbox_of(holes, pad=64)
    x0, y0, x1, y1 = bb
    sub_src = src[y0:y1, x0:x1]
    if not sub_src.any():            # fall back to whole image
        x0, y0, x1, y1 = 0, 0, labels.shape[1], labels.shape[0]
        sub_src = src
    _, (iy, ix) = ndi.distance_transform_edt(~sub_src, return_indices=True)
    sub_lab = labels[y0:y1, x0:x1]
    sub_holes = holes[y0:y1, x0:x1]
    sub_lab[sub_holes] = sub_lab[iy[sub_holes], ix[sub_holes]]
    return labels


def fill_gaps(labels: np.ndarray, rock: np.ndarray, min_area: int) -> np.ndarray:
    """Uncovered rock pixels: big areas -> new segments, small ones -> nearest segment."""
    labels = labels.copy()
    labels[~rock] = 0
    gap = rock & (labels == 0)
    if not gap.any():
        return labels
    n, cc, stats, _ = cv2.connectedComponentsWithStats(gap.astype(np.uint8), connectivity=4)
    next_id = int(labels.max()) + 1
    big = stats[:, cv2.CC_STAT_AREA] >= min_area
    big[0] = False
    if next_id == 1 and not big[1:].any():
        big[1] = True                            # nothing labelled at all: keep one piece
    lut = np.zeros(n, np.int32)
    lut[big] = np.arange(next_id, next_id + int(big.sum()), dtype=np.int32)
    new = lut[cc]
    labels[new > 0] = new[new > 0]
    small_lut = ~big
    small_lut[0] = False
    return absorb_into_nearest(labels, small_lut[cc])


def subdivide_large(labels: np.ndarray, image: np.ndarray, max_area: int, min_area: int,
                    compactness: float = 20.0) -> np.ndarray:
    """Split segments larger than max_area into colour-coherent pieces.

    SAM often returns one mask for a whole, evenly textured stone. Such segments are
    cut with a SLIC-like k-means on (Lab colour, position) restricted to the segment,
    so moss / lichen patches with a different colour end up in their own pieces.
    """
    big = [sid for sid, sl in enumerate(ndi.find_objects(labels), start=1)
           if sl is not None and int((labels[sl] == sid).sum()) > max_area]
    if not big:
        return labels
    lab = cv2.cvtColor(cv2.GaussianBlur(image, (0, 0), 2.0), cv2.COLOR_RGB2LAB).astype(np.float32)
    out = labels.copy()
    next_id = int(labels.max()) + 1
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
    for sid in big:
        ys, xs = np.nonzero(labels == sid)
        area = len(ys)
        k = int(min(64, np.ceil(area / (0.5 * max_area))))
        step = np.sqrt(area / k)                     # expected piece diameter
        w = compactness / step
        feats = np.column_stack([lab[ys, xs], xs * w, ys * w]).astype(np.float32)
        cv2.setRNGSeed(0)                            # same input -> same segments
        sample = feats if area <= 40000 else feats[np.random.default_rng(0).choice(area, 40000, replace=False)]
        _, _, centers = cv2.kmeans(sample, k, None, crit, 1, cv2.KMEANS_PP_CENTERS)
        km = np.empty(area, np.int64)
        for i in range(0, area, 200000):            # nearest centre for every pixel, in chunks
            f = feats[i:i + 200000]
            km[i:i + 200000] = ((f[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        out[ys[km > 0], xs[km > 0]] = next_id + km[km > 0] - 1
        next_id += k - 1
    rock = out > 0
    out = split_disconnected(out, min_area)          # k-means clusters can be scattered
    out = fill_gaps(out, rock, min_area)
    return relabel_sequential(out)


def paint_region(labels: np.ndarray, seg_class: np.ndarray, mask: np.ndarray, cls: int,
                 min_piece: int = 20, merge_same: bool = True) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Carve the brushed pixels out of the existing segments into one new segment of class `cls`.

    Only segmented (rock) pixels are painted. Pieces of the old segments that get cut off by the
    stroke become their own segments (keeping their class); crumbs < min_piece join the stroke.
    Returns (labels, seg_class, new_id), or None if nothing changes.
    """
    mask = mask & (labels > 0)
    if seg_class.size < int(labels.max()) + 1:
        seg_class = np.concatenate([seg_class, np.zeros(int(labels.max()) + 1 - seg_class.size, seg_class.dtype)])
    mask &= seg_class[labels] != cls
    if not mask.any():
        return None
    affected = np.unique(labels[mask])
    labels = labels.copy()
    new_id = int(labels.max()) + 1
    labels[mask] = new_id
    extra_cls = [cls]                                   # classes of ids new_id, new_id+1, …
    next_id = new_id + 1
    objs = ndi.find_objects(labels, max_label=int(affected.max()))
    for sid in affected:
        sl = objs[sid - 1]
        if sl is None:                                  # fully painted over
            continue
        sub = labels[sl] == sid
        n, cc, stats, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=4)
        if n <= 2:
            continue
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        lsub = labels[sl]
        for c in range(1, n):
            if c == keep:
                continue
            if stats[c, cv2.CC_STAT_AREA] < min_piece:
                lsub[cc == c] = new_id
            else:
                lsub[cc == c] = next_id
                extra_cls.append(int(seg_class[sid]))
                next_id += 1
    seg_class = np.concatenate([seg_class[:new_id], np.array(extra_cls, seg_class.dtype)])
    if merge_same and cls != 0:
        # fuse the stroke with touching segments of the same class: no boundary inside a painted area
        ys, xs = np.nonzero(labels == new_id)
        y0, y1 = max(ys.min() - 2, 0), ys.max() + 3
        x0, x1 = max(xs.min() - 2, 0), xs.max() + 3
        sub = labels[y0:y1, x0:x1]
        stroke = (sub == new_id).astype(np.uint8)
        ring = (cv2.dilate(stroke, np.ones((3, 3), np.uint8)) > 0) & (stroke == 0)
        nb = np.unique(sub[ring])
        nb = nb[(nb > 0) & (seg_class[nb] == cls)]
        if nb.size:
            ids = [new_id, *nb.tolist()]
            labels, new_id = merge_segments(labels, ids)
            seg_class[new_id] = cls
    return labels, seg_class, new_id


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lut = np.zeros(int(labels.max()) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return lut[labels]


def transfer_classes(old_labels: np.ndarray, old_seg_class: np.ndarray,
                     new_labels: np.ndarray, min_fraction: float = 0.5) -> np.ndarray:
    """For each new segment take the majority class of the old assignment (if > min_fraction)."""
    n_new = int(new_labels.max()) + 1
    out = np.zeros(n_new, np.int16)
    old_cls = old_seg_class[old_labels] if old_labels is not None else None
    if old_cls is None or old_cls.max() == 0:
        return out
    valid = new_labels > 0
    nl = new_labels[valid].astype(np.int64)
    oc = old_cls[valid].astype(np.int64)
    n_cls = int(oc.max()) + 1
    counts = np.bincount(nl * n_cls + oc, minlength=n_new * n_cls).reshape(n_new, n_cls)
    totals = counts.sum(1)
    best = counts.argmax(1)
    frac = counts[np.arange(n_new), best] / np.maximum(totals, 1)
    ok = (frac >= min_fraction) & (best > 0)
    out[ok] = best[ok]
    return out


# ── editing ──────────────────────────────────────────────────────────────────
def merge_segments(labels: np.ndarray, ids: list[int]) -> tuple[np.ndarray, int]:
    target = int(min(ids))
    labels = labels.copy()
    labels[np.isin(labels, list(ids))] = target
    return labels, target


def split_by_stroke(labels: np.ndarray, stroke: np.ndarray, min_area: int = 30
                    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Cut every segment touched by the stroke along it.

    Returns the new label map and a list of (new_id, parent_id).
    """
    labels = labels.copy()
    touched = [int(t) for t in np.unique(labels[stroke]) if t > 0]
    next_id = int(labels.max()) + 1
    created: list[tuple[int, int]] = []
    for sid in touched:
        seg = labels == sid
        bb = bbox_of(seg, pad=1)
        x0, y0, x1, y1 = bb
        sub_seg = seg[y0:y1, x0:x1]
        sub_cut = sub_seg & ~stroke[y0:y1, x0:x1]
        n, cc, stats, _ = cv2.connectedComponentsWithStats(sub_cut.astype(np.uint8), connectivity=4)
        big = [c for c in range(1, n) if stats[c, cv2.CC_STAT_AREA] >= min_area]
        if len(big) < 2:
            continue
        big.sort(key=lambda c: -stats[c, cv2.CC_STAT_AREA])
        sub_lab = labels[y0:y1, x0:x1]
        sub_lab[sub_seg] = 0                       # temporarily clear the segment
        sub_lab[cc == big[0]] = sid                # largest piece keeps the id
        for c in big[1:]:
            sub_lab[cc == c] = next_id
            created.append((next_id, sid))
            next_id += 1
        # stroke pixels and slivers go to the nearest piece (only within this segment)
        holes = sub_seg & (sub_lab == 0)
        if holes.any():
            mask_pieces = np.where(sub_seg, sub_lab, 0)
            _, (iy, ix) = ndi.distance_transform_edt(mask_pieces == 0, return_indices=True)
            sub_lab[holes] = mask_pieces[iy[holes], ix[holes]]
    return labels, created


def apply_rock_change(labels: np.ndarray, rock: np.ndarray, min_area: int) -> np.ndarray:
    """Clip segments to a new rock mask and segment newly added rock areas."""
    if labels is None or labels.max() == 0:
        return labels
    labels = split_disconnected(np.where(rock, labels, 0), 1)
    return relabel_sequential(fill_gaps(labels, rock, min_area))


# ── rendering helpers ────────────────────────────────────────────────────────
def boundaries(labels: np.ndarray) -> np.ndarray:
    b = np.zeros(labels.shape, bool)
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    return b


def mask_outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    return (cv2.dilate(m, k, iterations=thickness) - cv2.erode(m, k, iterations=1)) > 0


def stroke_to_mask(shape, points, thickness: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(points, np.float32)).astype(np.int32)
    if len(pts) == 1:
        cv2.circle(m, tuple(int(v) for v in pts[0]), max(1, thickness // 2), 1, -1, cv2.LINE_8)
    else:
        cv2.polylines(m, [pts.reshape(-1, 1, 2)], False, 1, max(1, thickness), cv2.LINE_8)
    return m.astype(bool)
