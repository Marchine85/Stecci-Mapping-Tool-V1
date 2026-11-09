"""Reduce a scan to a maximum number of triangles before it is processed.

1. simplify the welded mesh (quadric decimation, fast_simplification)
2. new UV atlas for the simplified mesh (box_charts): the scanner's atlas does not survive the simplification.
   Faces are grouped by their smoothed normal into the 6 axis directions; every connected group is a chart,
   projected along its axis (each face is at most ~55° from it), turned to its main axes and shelf-packed.
   (xatlas gives tighter charts but needs ~2.5 min on a rough 100k-triangle scan; this takes seconds.)
3. bake one texture: every texel finds its surface point on the full-resolution scan (closest point among the
   nearest original triangles) and takes that point's colour from the original textures
Untextured scans keep vertex colours (nearest original vertex).
"""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .raster import rasterize


def reduce_scan(scan, max_faces: int, tex_px: int, progress=None):
    from .mesh import Scan, TextureSampler, weld
    from .unwrap import closest_on_triangles
    import fast_simplification

    t0 = time.perf_counter()
    m0 = scan.n_faces
    if progress:
        progress(f"Reducing {m0:,} → {max_faces:,} triangles …")
    Fw, P = weld(scan.V, scan.F)
    v, f = fast_simplification.simplify(P.astype(np.float32), Fw.astype(np.int32), 1 - max_faces / m0)
    v, f = v.astype(np.float64), f.astype(np.int64)
    T = v[f]
    ok = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) > 1e-14
    f = f[ok]
    used, f = np.unique(f, return_inverse=True)
    v, f = v[used], f.reshape(-1, 3)

    if not scan.textures:
        _, near = cKDTree(scan.V).query(v, workers=-1)
        out = Scan(scan.path, v.astype(np.float32), f.astype(np.int32), np.zeros((len(v), 2), np.float32),
                   scan.C[near], np.full(len(f), -1, np.int16), [])
        out.n_original = m0
        return out

    if progress:
        progress(f"Reduced to {len(f):,} triangles · new texture layout …")
    V2, F2, uv = box_charts(v, f, tex_px)
    W = H = tex_px

    if progress:
        progress(f"Baking a {tex_px}×{tex_px} texture from the full-resolution scan …")
    Ppx = np.c_[uv[:, 0] * W, (1 - uv[:, 1]) * H]       # image rows top-down, uv v up
    face, bary, _ = rasterize(Ppx, np.zeros(len(Ppx)), F2, W, H)
    cov = face >= 0
    fm = face[cov]
    l0, l1 = bary[cov, 0].astype(np.float64), bary[cov, 1].astype(np.float64)
    X = V2[F2[fm, 0]] * l0[:, None] + V2[F2[fm, 1]] * l1[:, None] + V2[F2[fm, 2]] * (1 - l0 - l1)[:, None]

    Vo = scan.V.astype(np.float64)
    tree = cKDTree(Vo[scan.F].mean(1))
    src_face = np.empty(len(X), np.int64)
    src_bary = np.empty((len(X), 2), np.float32)
    chunk = 500_000
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        _, cand = tree.query(x, k=4, workers=-1)
        Tc = Vo[scan.F[cand]]
        b, d = closest_on_triangles(x[:, None], Tc[..., 0, :], Tc[..., 1, :], Tc[..., 2, :])
        j = d.argmin(1)
        r = np.arange(len(x))
        src_face[s:s + chunk] = cand[r, j]
        src_bary[s:s + chunk] = b[r, j, :2]
        if progress and (s // chunk) % 6 == 0:
            progress(f"Baking the texture … {100 * min(s + chunk, len(X)) // len(X)} %")
    cols = TextureSampler(scan).sample(src_face, src_bary)
    tex = np.zeros((H, W, 3), np.uint8)
    tex[cov] = cols
    # fill the gutters around the charts (texture filtering and mip-maps read across chart borders)
    _, (iy, ix) = ndi.distance_transform_edt(~cov, return_indices=True)
    tex = tex[iy, ix]

    _, near = cKDTree(Vo).query(V2, workers=-1)
    out = Scan(scan.path, V2.astype(np.float32), F2.astype(np.int32), uv.astype(np.float32), scan.C[near],
               np.zeros(len(F2), np.int16), [tex])
    out.n_original = m0
    if progress:
        progress(f"Reduced {m0:,} → {len(F2):,} triangles in {time.perf_counter() - t0:.0f} s")
    return out


def box_charts(v: np.ndarray, f: np.ndarray, tex_px: int, pad: int = 2):
    """Charts + packing. Returns (vertices split at chart borders, faces, uv in [0,1] with v up)."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    m, nv = len(f), len(v)
    T = v[f]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    A = sparse.csr_matrix((np.ones(3 * m), (np.repeat(np.arange(m), 3), f.ravel())), shape=(m, nv))
    for _ in range(2):                                    # smooth normals: steadier direction groups
        n = A @ (A.T @ n)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    dirs = np.concatenate([np.eye(3), -np.eye(3)])
    d = np.argmax(n @ dirs.T, 1)
    # charts = connected components of faces with the same direction (shared edges)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(1)
    face = np.tile(np.arange(m), 3)
    key = e[:, 0] * nv + e[:, 1]
    o = np.argsort(key, kind="stable")
    key, face = key[o], face[o]
    i = np.nonzero(key[1:] == key[:-1])[0]
    a, b = face[i], face[i + 1]
    keep = d[a] == d[b]
    G = sparse.csr_matrix((np.ones(int(keep.sum())), (a[keep], b[keep])), shape=(m, m))
    nc, chart = connected_components(G, directed=False)
    # large charts (e.g. the ground ring around the stone) are cut into grid cells: their bounding boxes
    # would otherwise be mostly empty. Cells are then split into connected pieces again.
    cen = T.mean(1)
    cell = float(np.linalg.norm(v.max(0) - v.min(0))) / 10
    proj = np.empty((m, 2))
    for k in range(3):
        sel = d % 3 == k
        proj[sel] = cen[sel][:, [j for j in range(3) if j != k]]
    g = np.floor(proj / cell).astype(np.int64)
    g -= g.min(0)
    sub = (chart.astype(np.int64) * (g[:, 0].max() + 1) + g[:, 0]) * (g[:, 1].max() + 1) + g[:, 1]
    keep2 = keep & (sub[a] == sub[b])
    G = sparse.csr_matrix((np.ones(int(keep2.sum())), (a[keep2], b[keep2])), shape=(m, m))
    nc, chart = connected_components(G, directed=False)

    # one vertex per (chart, vertex): charts get their own copies along their borders
    ck = chart[:, None].astype(np.int64) * nv + f
    uk, inv = np.unique(ck.ravel(), return_inverse=True)
    F2 = inv.reshape(-1, 3)
    vid = (uk % nv).astype(np.int64)
    vchart = (uk // nv).astype(np.int64)
    V2 = v[vid]
    # project along the chart's axis, then rotate to the chart's main axes (smaller boxes pack better)
    cdir = np.zeros(nc, np.int64)
    cdir[chart] = d
    axis = cdir[vchart] % 3
    P = np.empty((len(V2), 2))
    for k in range(3):
        sel = axis == k
        others = [j for j in range(3) if j != k]
        P[sel] = V2[sel][:, others]
    flip = cdir[vchart] >= 3                              # seen from the negative side: mirror
    P[flip, 0] *= -1
    order = np.argsort(vchart, kind="stable")
    bounds = np.searchsorted(vchart[order], np.arange(nc + 1))
    for c in range(nc):
        idx = order[bounds[c]:bounds[c + 1]]
        q = P[idx] - P[idx].mean(0)
        if len(idx) >= 3:
            _, _, vt = np.linalg.svd(q, full_matrices=False)
            q = q @ vt.T
        P[idx] = q
    lo = np.full((nc, 2), np.inf); hi = np.full((nc, 2), -np.inf)
    np.minimum.at(lo, vchart, P)
    np.maximum.at(hi, vchart, P)
    size = hi - lo

    def pack(scale):
        """Shelf packing at `scale` px per unit; returns (offsets px, used height) or None if too wide."""
        w = np.ceil(size[:, 0] * scale) + 2 * pad
        h = np.ceil(size[:, 1] * scale) + 2 * pad
        if w.max() > tex_px:
            return None, np.inf
        order_c = np.argsort(-h, kind="stable")
        off = np.zeros((nc, 2))
        x = y = shelf = 0.0
        for c in order_c:
            if x + w[c] > tex_px:
                x, y, shelf = 0.0, y + shelf, 0.0
            off[c] = (x + pad, y + pad)
            x += w[c]
            shelf = max(shelf, h[c])
        return off, y + shelf

    total = float((size[:, 0] * size[:, 1]).sum())
    s_lo, s_hi = 0.0, tex_px / np.sqrt(max(total, 1e-30))
    best = None
    for _ in range(18):                                   # largest scale that still fits the square
        s = (s_lo + s_hi) / 2
        off, used = pack(s)
        if used <= tex_px:
            s_lo, best = s, (s, off)
        else:
            s_hi = s
    s, off = best
    px = (P - lo[vchart]) * s + off[vchart]
    uv = np.c_[px[:, 0] / tex_px, 1 - px[:, 1] / tex_px]
    return V2, F2, uv
