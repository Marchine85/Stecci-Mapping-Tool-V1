"""Segments on the 3D surface: one segment id per face (0 = no segment / not stone).

Same conventions as the 2D tool's label maps, with face adjacency (shared edges) instead of pixel neighbours.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from scipy import ndimage as ndi

from .. import segmentation as S
from ..config import GRANULARITY
from .mesh import Scan, face_components
from .unwrap import Unwrap

SAM_SIDE = 2048


# ── segmentation of the (hidden) unwrap image ────────────────────────────────
def segment_unwrap(uw: Unwrap, models, g, level: int, progress=None) -> np.ndarray:
    """The 2D tool's Segment step on the unwrap image: SAM masks -> partition -> colour subdivision."""
    rock = uw.mask
    min_area = max(30, int(GRANULARITY[level].min_frac * int(rock.sum())))
    x0, y0, x1, y1 = S.bbox_of(rock, pad=8)
    rock_crop = rock[y0:y1, x0:x1].copy()
    crop = uw.image[y0:y1, x0:x1].copy()
    crop[~rock_crop] = 127
    # SAM works at 1024 px internally: run it on at most SAM_SIDE px (larger inputs only cost time and GPU
    # memory for the mask upsampling), then scale its partition up; the colour subdivision below uses full resolution
    h, w = rock_crop.shape
    f = min(1.0, SAM_SIDE / max(h, w))
    if f < 1:
        sz = (max(1, round(w * f)), max(1, round(h * f)))
        small = cv2.resize(crop, sz, interpolation=cv2.INTER_AREA)
        rock_s = cv2.resize(rock_crop.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
        min_s = max(30, int(min_area * f * f))
        masks = models.auto_masks(small, g, min_s, progress=progress)
        if progress:
            progress(f"Building segments from {len(masks)} masks …")
        lab_s = S.masks_to_labels(masks, rock_s, min_s)
        lab = cv2.resize(lab_s, (w, h), interpolation=cv2.INTER_NEAREST)
        lab[~rock_crop] = 0
        lab = S.relabel_sequential(S.fill_gaps(lab, rock_crop, min_area))
    else:
        masks = models.auto_masks(crop, g, min_area, progress=progress)
        if progress:
            progress(f"Building segments from {len(masks)} masks …")
        lab = S.masks_to_labels(masks, rock_crop, min_area)
    max_area = max(min_area * 4, int(g.max_frac * rock_crop.sum()))
    if (np.bincount(lab.ravel())[1:] > max_area).any():
        if progress:
            progress("Subdividing large segments by colour …")
        lab = S.subdivide_large(lab, crop, max_area, min_area)
    labels = np.zeros(rock.shape, np.int32)
    labels[y0:y1, x0:x1] = lab
    return labels


def labels_to_faces(scan: Scan, uw: Unwrap, labels: np.ndarray) -> np.ndarray:
    """Segment id per face from the label image (faces just off a segment take the nearest one)."""
    filled = labels
    if (labels == 0).any() and labels.max() > 0:
        _, (iy, ix) = ndi.distance_transform_edt(labels == 0, return_indices=True)
        filled = labels[iy, ix]
    H, W = labels.shape
    x = np.clip(uw.px[:, 0].astype(np.int64), 0, W - 1)
    y = np.clip(uw.px[:, 1].astype(np.int64), 0, H - 1)
    lab = filled[y, x]
    hidden = ~uw.visible
    if hidden.any() and uw.visible.any():
        # faces hidden in the projection (undersides, ledges): label of the nearest visible face in 3D
        from scipy.spatial import cKDTree
        _, near = cKDTree(uw.centroids[uw.visible]).query(uw.centroids[hidden], workers=-1)
        lab[hidden] = lab[np.nonzero(uw.visible)[0][near]]
    seg = np.zeros(scan.n_faces, np.int32)
    seg[uw.faces] = lab
    return seg


# ── clean-up ─────────────────────────────────────────────────────────────────
def relabel_sequential(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ids 1..N in order of first use; returns (new seg, old id per new id)."""
    ids = np.unique(seg)
    ids = ids[ids > 0]
    lut = np.zeros(int(seg.max()) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return lut[seg], np.concatenate([[0], ids])


def absorb_small(scan: Scan, seg: np.ndarray, stone: np.ndarray, min_area: float, rounds: int = 6) -> np.ndarray:
    """Connected pieces smaller than min_area join the neighbouring segment they share most edges with."""
    seg = seg.copy()
    a, b = scan.adj[:, 0], scan.adj[:, 1]
    inside = stone[a] & stone[b]
    a, b = a[inside], b[inside]
    for _ in range(rounds):
        comp = face_components(scan, stone & (seg > 0), same=seg)
        n = int(comp.max()) + 1
        if n <= 1:
            break
        carea = np.bincount(comp[stone & (seg > 0)], scan.area[stone & (seg > 0)], minlength=n)
        small = carea < min_area
        ca, cb = comp[a], comp[b]
        cross = (ca != cb) & (ca >= 0) & (cb >= 0)
        # both directions: small piece on either side
        pa = np.concatenate([ca[cross], cb[cross]])
        sb = np.concatenate([seg[b][cross], seg[a][cross]])
        sel = small[pa]
        if not sel.any():
            break
        pa, sb = pa[sel], sb[sel]
        key = pa.astype(np.int64) * (int(seg.max()) + 1) + sb
        u, cnt = np.unique(key, return_counts=True)
        pc, ps = u // (int(seg.max()) + 1), u % (int(seg.max()) + 1)
        o = np.lexsort((-cnt, pc))                         # per piece: most shared edges first
        pc, ps = pc[o], ps[o]
        first = np.ones(len(pc), bool)
        first[1:] = pc[1:] != pc[:-1]
        target = np.full(n, -1, np.int64)
        target[pc[first]] = ps[first]
        # don't let two small neighbours swap into each other forever: smallest pieces move first
        move = target[comp] >= 0
        move &= comp >= 0
        seg[move] = target[comp[move]]
    return seg


def split_disconnected(scan: Scan, seg: np.ndarray, stone: np.ndarray) -> np.ndarray:
    comp = face_components(scan, stone & (seg > 0), same=seg)
    out = np.zeros_like(seg)
    m = comp >= 0
    out[m] = comp[m] + 1
    return out


def faces_from_unwrap(scan: Scan, uw: Unwrap, labels: np.ndarray, stone: np.ndarray, level: int) -> np.ndarray:
    seg = labels_to_faces(scan, uw, labels)
    seg[~stone] = 0
    min_area = GRANULARITY[max(level, 1)].min_frac * float(scan.area[stone].sum()) * 0.5
    seg = absorb_small(scan, seg, stone, min_area)
    seg = split_disconnected(scan, seg, stone)
    return relabel_sequential(seg)[0]


def transfer_classes(old_seg, old_cls, new_seg, area, min_fraction: float = 0.5) -> np.ndarray:
    """Area-weighted majority class of the old assignment for each new segment."""
    n_new = int(new_seg.max()) + 1
    out = np.zeros(n_new, np.int16)
    if old_seg is None or not np.any(old_cls):
        return out
    oc = old_cls[np.minimum(old_seg, len(old_cls) - 1)].astype(np.int64)
    valid = new_seg > 0
    nl = new_seg[valid].astype(np.int64)
    oc = oc[valid]
    n_cls = int(oc.max()) + 1
    counts = np.bincount(nl * n_cls + oc, weights=area[valid], minlength=n_new * n_cls).reshape(n_new, n_cls)
    best = counts.argmax(1)
    frac = counts[np.arange(n_new), best] / np.maximum(counts.sum(1), 1e-30)
    ok = (frac >= min_fraction) & (best > 0)
    out[ok] = best[ok]
    return out


# ── editing ──────────────────────────────────────────────────────────────────
def grow_into(scan: Scan, seg: np.ndarray, holes: np.ndarray, allowed: np.ndarray, iters: int = 500) -> np.ndarray:
    """Hole faces take the segment of a labelled neighbour (repeated until filled), within `allowed`."""
    seg = seg.copy()
    a, b = scan.adj[:, 0], scan.adj[:, 1]
    k = allowed[a] & allowed[b]
    a, b = a[k], b[k]
    holes = holes.copy()
    for _ in range(iters):
        if not holes.any():
            break
        ha, hb = holes[a], holes[b]
        x = ha & ~hb
        y = hb & ~ha
        dst = np.concatenate([a[x], b[y]])
        src = np.concatenate([b[x], a[y]])
        if len(dst) == 0:
            break
        seg[dst] = seg[src]
        holes[dst] = False
    return seg


def paint(scan: Scan, seg: np.ndarray, seg_class: np.ndarray, mask: np.ndarray, cls: int, stone: np.ndarray,
          crumb_area: float, merge_same: bool = True):
    """Carve the brushed faces into one new segment of class `cls` (cf. segmentation.paint_region).

    Pieces of the old segments that get cut off become their own segments (keeping their class); crumbs smaller
    than crumb_area join the stroke. The stroke merges with touching segments of the same class.
    Returns (seg, seg_class) or None if nothing changes.
    """
    seg_class = np.asarray(seg_class)
    if len(seg_class) < int(seg.max()) + 1:
        seg_class = np.concatenate([seg_class, np.zeros(int(seg.max()) + 1 - len(seg_class), seg_class.dtype)])
    mask = mask & (seg > 0) & stone
    mask &= seg_class[seg] != cls
    if not mask.any():
        return None
    affected = np.unique(seg[mask])
    seg = seg.copy()
    new_id = int(seg.max()) + 1
    seg[mask] = new_id
    extra = [cls]
    aff = np.isin(seg, affected)
    comp = face_components(scan, aff, same=seg)
    m = comp >= 0
    n = int(comp.max()) + 1
    if n > 0:
        carea = np.bincount(comp[m], scan.area[m], minlength=n)
        csid = np.zeros(n, np.int64)
        csid[comp[m]] = seg[m]
        # largest piece of every affected segment keeps the id
        o = np.lexsort((-carea, csid))
        first = np.ones(n, bool)
        first[1:] = csid[o][1:] != csid[o][:-1]
        keep = np.zeros(n, bool)
        keep[o[first]] = True
        newid = np.zeros(n, np.int64)
        crumbs = ~keep & (carea < crumb_area)
        newid[crumbs] = new_id
        pieces = np.nonzero(~keep & ~crumbs)[0]
        newid[pieces] = new_id + 1 + np.arange(len(pieces))
        extra += [int(seg_class[csid[c]]) for c in pieces]
        change = m & (newid[np.maximum(comp, 0)] > 0)
        seg[change] = newid[comp[change]]
    seg_class = np.concatenate([seg_class[:new_id], np.array(extra, seg_class.dtype)])
    if merge_same and cls != 0:
        a, b = scan.adj[:, 0], scan.adj[:, 1]
        na = seg[a] == new_id
        nb = seg[b] == new_id
        nbrs = np.unique(np.concatenate([seg[b][na & ~nb], seg[a][nb & ~na]]))
        nbrs = nbrs[(nbrs > 0) & (seg_class[nbrs] == cls)]
        if len(nbrs):
            ids = [new_id, *nbrs.tolist()]
            target = min(ids)
            seg[np.isin(seg, ids)] = target
            seg_class[target] = cls
    return seg, seg_class


def merge(seg: np.ndarray, ids) -> tuple[np.ndarray, int]:
    target = int(min(ids))
    seg = seg.copy()
    seg[np.isin(seg, list(ids))] = target
    return seg, target


def split(scan: Scan, seg: np.ndarray, stroke: np.ndarray, stone: np.ndarray, min_area: float):
    """Cut every segment touched by the stroke along it. Returns (seg, [(new_id, parent_id)])."""
    touched = [int(t) for t in np.unique(seg[stroke & stone]) if t > 0]
    seg = seg.copy()
    next_id = int(seg.max()) + 1
    created = []
    for sid in touched:
        inseg = seg == sid
        rest = inseg & ~stroke
        comp = face_components(scan, rest)
        n = int(comp.max()) + 1
        if n < 2:
            continue
        carea = np.bincount(comp[rest], scan.area[rest], minlength=n)
        big = [c for c in np.argsort(-carea) if carea[c] >= min_area]
        if len(big) < 2:
            continue
        lab = np.zeros_like(seg)
        lab[comp == big[0]] = sid
        for c in big[1:]:
            lab[comp == c] = next_id
            created.append((next_id, sid))
            next_id += 1
        holes = inseg & (lab == 0)
        lab = grow_into(scan, lab, holes, inseg)
        seg[inseg] = np.where(lab[inseg] > 0, lab[inseg], sid)
    return seg, created


def apply_stone_change(scan: Scan, seg: np.ndarray, stone: np.ndarray) -> np.ndarray:
    """Clip segments to a new stone mask; newly added stone faces join a neighbouring segment."""
    seg = np.where(stone, seg, 0).astype(np.int32)
    holes = stone & (seg == 0)
    if holes.any() and seg.max() > 0:
        seg = grow_into(scan, seg, holes, stone)
        left = stone & (seg == 0)
        if left.any():                                       # islands without a segment: their own segment
            seg[left] = int(seg.max()) + 1
    return seg


# ── drawing helpers ──────────────────────────────────────────────────────────
def boundary_edges(scan: Scan, seg: np.ndarray, stone: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Welded vertex pairs on segment boundaries, and on the stone's outline (stone / ground border)."""
    a, b = scan.adj[:, 0], scan.adj[:, 1]
    sa, sb = stone[a], stone[b]
    seg_e = sa & sb & (seg[a] != seg[b])
    out_e = sa != sb
    return scan.adj[seg_e, 2:4], scan.adj[out_e, 2:4]


def selection_edges(scan: Scan, seg: np.ndarray, stone: np.ndarray, selected) -> np.ndarray:
    """Welded vertex pairs on the outline of the selected segments (selected face next to an unselected one)."""
    if not len(selected):
        return np.zeros((0, 2), np.int32)
    insel = np.isin(seg, np.fromiter(selected, np.int64)) & stone
    a, b = scan.adj[:, 0], scan.adj[:, 1]
    return scan.adj[insel[a] != insel[b], 2:4]


def class_patches(scan: Scan, cls_face: np.ndarray, stone: np.ndarray) -> dict[int, int]:
    comp = face_components(scan, stone, same=cls_face)
    m = comp >= 0
    out = {}
    first = np.zeros(int(comp.max()) + 1, np.int64)
    first[comp[m]] = cls_face[m]
    for c in np.unique(first):
        out[int(c)] = int((first == c).sum())
    return out
