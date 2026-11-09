"""Vectorised triangle rasteriser (numpy only) with z-buffer, plus an orthographic projection helper.

rasterize(P, depth, F, W, H) -> face id per pixel (-1 = empty) + barycentric coordinates + depth.
P: (n,2) pixel coordinates per vertex (x right, y down; pixel centres at i + 0.5), depth: (n,) larger = nearer.
Works for millions of small triangles: triangles are grouped by their pixel bounding-box size and every
group is tested against its k*k candidate pixels at once.
"""
from __future__ import annotations

import numpy as np


def rasterize(P: np.ndarray, depth: np.ndarray, F: np.ndarray, W: int, H: int, faces: np.ndarray | None = None,
              chunk: int = 4_000_000):
    if faces is None:
        faces = np.arange(len(F))
    tri = P[F[faces]].astype(np.float64)                     # (m,3,2)
    dz = depth[F[faces]].astype(np.float64)                 # (m,3)
    x0 = np.ceil(tri[..., 0].min(1) - 0.5).astype(np.int64)
    x1 = np.floor(tri[..., 0].max(1) - 0.5).astype(np.int64)
    y0 = np.ceil(tri[..., 1].min(1) - 0.5).astype(np.int64)
    y1 = np.floor(tri[..., 1].max(1) - 0.5).astype(np.int64)
    x0 = np.maximum(x0, 0); y0 = np.maximum(y0, 0)
    x1 = np.minimum(x1, W - 1); y1 = np.minimum(y1, H - 1)
    k = np.maximum(x1 - x0 + 1, y1 - y0 + 1)
    ok = (x1 >= x0) & (y1 >= y0)

    best_d = np.full(W * H, -np.inf)
    best_f = np.full(W * H, -1, np.int64)
    best_b = np.zeros((W * H, 2), np.float32)

    # signed-area denominators for barycentrics
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok &= np.abs(den) > 1e-12

    for ks in np.unique(k[ok]):
        idx = np.nonzero(ok & (k == ks))[0]
        per = max(1, chunk // int(ks * ks))
        gy, gx = np.mgrid[0:ks, 0:ks]
        gx = gx.ravel(); gy = gy.ravel()
        for s in range(0, len(idx), per):
            t = idx[s:s + per]
            px = (x0[t][:, None] + gx[None]).ravel()
            py = (y0[t][:, None] + gy[None]).ravel()
            tt = np.repeat(t, len(gx))
            keep = (px <= x1[tt]) & (py <= y1[tt])
            px, py, tt = px[keep], py[keep], tt[keep]
            cx, cy = px + 0.5, py + 0.5
            l0 = ((b[tt, 1] - c[tt, 1]) * (cx - c[tt, 0]) + (c[tt, 0] - b[tt, 0]) * (cy - c[tt, 1])) / den[tt]
            l1 = ((c[tt, 1] - a[tt, 1]) * (cx - c[tt, 0]) + (a[tt, 0] - c[tt, 0]) * (cy - c[tt, 1])) / den[tt]
            l2 = 1 - l0 - l1
            inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
            px, py, tt, l0, l1, l2 = px[inside], py[inside], tt[inside], l0[inside], l1[inside], l2[inside]
            d = l0 * dz[tt, 0] + l1 * dz[tt, 1] + l2 * dz[tt, 2]
            pix = py * W + px
            o = np.lexsort((-d, pix))                        # per pixel: nearest first
            pix, d, tt, l0, l1 = pix[o], d[o], tt[o], l0[o], l1[o]
            first = np.ones(len(pix), bool)
            first[1:] = pix[1:] != pix[:-1]
            pix, d, tt, l0, l1 = pix[first], d[first], tt[first], l0[first], l1[first]
            win = d > best_d[pix]
            pix = pix[win]
            best_d[pix] = d[win]
            best_f[pix] = faces[tt[win]]
            best_b[pix, 0] = l0[win]
            best_b[pix, 1] = l1[win]
    return best_f.reshape(H, W), best_b.reshape(H, W, 2), best_d.reshape(H, W)


def ortho(V: np.ndarray, right: np.ndarray, up: np.ndarray, mm_per_px: float, center=None, size=None):
    """Orthographic projection looking along -(right x up): returns pixel coords, depth, W, H."""
    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)
    view = np.cross(right, up)                               # towards the viewer
    x = V @ right
    y = V @ up
    z = V @ view
    s = 1000.0 / mm_per_px                                   # model units assumed to be metres
    if center is None:
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    else:
        cx, cy = center
        hw, hh = size[0] / 2, size[1] / 2
        x0, x1, y0, y1 = cx - hw, cx + hw, cy - hh, cy + hh
    W = int(np.ceil((x1 - x0) * s)); H = int(np.ceil((y1 - y0) * s))
    P = np.stack([(x - x0) * s, (y1 - y) * s], 1)
    return P, z, W, H
