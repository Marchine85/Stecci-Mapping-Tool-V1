"""Flatten the isolated stone into one image for the (hidden) 2D segmentation.

Methods
  arap    one piece without seams: LSCM start + as-rigid-as-possible iterations (Liu et al. 2008) on a
          simplified copy of the stone; every full-resolution face is located on it by its closest point
  boxnet  5 straight-on projections (top + 4 walls) folded out as a cross
  belt    the same projections, walls in one strip in walking order (back, right end, front, left end)
All coordinates are in the stone's aligned frame (x length, y width, z up), in model units.

compute(...) -> Unwrap: the rendered image, its stone mask and the pixel position of every stone face.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .isolate import Isolation
from .mesh import Scan, TextureSampler
from .raster import rasterize


@dataclass
class Unwrap:
    method: str
    image: np.ndarray        # (H,W,3) uint8, white outside the stone
    mask: np.ndarray         # (H,W) bool: stone pixels
    faces: np.ndarray        # (k,) stone face ids
    px: np.ndarray           # (k,2) float pixel position (x, y) of each face's centroid
    unit_per_px: float       # model units per pixel
    visible: np.ndarray      # (k,) bool: the face shows in the image (box net / belt hide undersides)
    centroids: np.ndarray    # (k,3) face centroids (aligned frame)


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class StoneFrame:
    """The stone's faces in the aligned frame."""

    def __init__(self, scan: Scan, iso: Isolation):
        self.scan = scan
        self.faces = np.nonzero(iso.stone)[0]
        self.F = scan.F[self.faces]
        self.Fw = scan.Fw[self.faces]
        self.L = scan.V.astype(np.float64) @ iso.R.T - iso.origin
        T = self.L[self.F]
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        self.area = np.linalg.norm(n, axis=1) / 2
        self.fn = n / np.maximum(2 * self.area, 1e-30)[:, None]
        P = self.L[np.unique(self.F)]
        self.lo, self.hi = P.min(0), P.max(0)
        self.size = float((self.hi - self.lo).max())

    def smooth_normals(self, iters: int = 4) -> np.ndarray:
        ids, Fl = np.unique(self.Fw, return_inverse=True)
        Fl = Fl.reshape(-1, 3)
        m, nv = len(Fl), len(ids)
        A = sparse.csr_matrix((np.ones(3 * m), (np.repeat(np.arange(m), 3), Fl.ravel())), shape=(m, nv))
        fn = self.fn * self.area[:, None]
        for _ in range(iters):
            fn = A @ (A.T @ fn)
        return fn / np.maximum(np.linalg.norm(fn, axis=1), 1e-30)[:, None]

    def classify(self) -> np.ndarray:
        """0 top, 1 front (+y), 2 back (-y), 3 right end (+x), 4 left end (-x)."""
        n = self.smooth_normals()
        cls = np.where(np.abs(n[:, 0]) > np.abs(n[:, 1]), np.where(n[:, 0] > 0, 3, 4), np.where(n[:, 1] > 0, 1, 2))
        cls[n[:, 2] > 0.55] = 0
        return cls


# ── box net / belt: corner coordinates of every stone face ────────────────────
def boxnet(st: StoneFrame):
    cls = st.classify()
    gap = 0.01 * st.size
    T = st.L[st.F] - st.lo
    Lx, Ly = st.hi[0] - st.lo[0], st.hi[1] - st.lo[1]
    x, y, z = T[..., 0], T[..., 1], T[..., 2]
    zc = np.zeros(5)
    for c in range(1, 5):
        if (cls == c).any():
            zc[c] = T[cls == c][..., 2].max()
    Hm = max(zc[1:]) + gap
    u = np.zeros_like(x); v = np.zeros_like(x); dep = np.zeros_like(x)
    d = zc[cls][:, None] - z
    s = cls == 0; u[s] = Hm + x[s]; v[s] = Hm + Ly - y[s]; dep[s] = z[s]
    s = cls == 1; u[s] = Hm + x[s]; v[s] = Hm - gap - d[s]; dep[s] = y[s]
    s = cls == 2; u[s] = Hm + x[s]; v[s] = Hm + Ly + gap + d[s]; dep[s] = -y[s]
    s = cls == 3; u[s] = Hm + Lx + gap + d[s]; v[s] = Hm + Ly - y[s]; dep[s] = x[s]
    s = cls == 4; u[s] = Hm - gap - d[s]; v[s] = Hm + Ly - y[s]; dep[s] = -x[s]
    return np.stack([u, v], -1), dep


def belt(st: StoneFrame):
    cls = st.classify()
    gap = 0.01 * st.size
    T = st.L[st.F] - st.lo
    Lx, Ly = st.hi[0] - st.lo[0], st.hi[1] - st.lo[1]
    x, y, z = T[..., 0], T[..., 1], T[..., 2]
    zc = T[cls > 0][..., 2].max() if (cls > 0).any() else 0.0
    d = zc - z
    u = np.zeros_like(x); v = np.zeros_like(x); dep = np.zeros_like(x)
    s = cls == 0; u[s] = x[s]; v[s] = Ly - y[s]; dep[s] = z[s]
    v0 = Ly + gap
    s = cls == 2; u[s] = x[s]; v[s] = v0 + d[s]; dep[s] = -y[s]
    s = cls == 3; u[s] = Lx + gap + y[s]; v[s] = v0 + d[s]; dep[s] = x[s]
    s = cls == 1; u[s] = Lx + Ly + 2 * gap + (Lx - x[s]); v[s] = v0 + d[s]; dep[s] = y[s]
    s = cls == 4; u[s] = 2 * Lx + Ly + 3 * gap + (Ly - y[s]); v[s] = v0 + d[s]; dep[s] = -x[s]
    return np.stack([u, v], -1), dep


# ── ARAP on a simplified copy ─────────────────────────────────────────────────
def simplified(st: StoneFrame, target: int = 150_000):
    import fast_simplification
    ids, Fl = np.unique(st.Fw, return_inverse=True)
    Fl = Fl.reshape(-1, 3)
    pos = np.zeros((len(ids), 3))
    pos[Fl.ravel()] = st.L[st.F.ravel()]
    if len(Fl) > target:
        v, f = fast_simplification.simplify(pos.astype(np.float32), Fl.astype(np.int32), 1 - target / len(Fl))
        v, f = v.astype(np.float64), f.astype(np.int64)
    else:
        v, f = pos, Fl.astype(np.int64)
    return clean_disc(v, f)


def clean_disc(v, f):
    """Largest connected, edge-manifold piece; unreferenced vertices removed."""
    for _ in range(5):
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
        key = e[:, 0] * len(v) + e[:, 1]
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        bad = (cnt[inv] > 2).reshape(3, -1).T.any(1)
        f = f[~bad]
        A = sparse.csr_matrix((np.ones(3 * len(f)), (np.repeat(np.arange(len(f)), 3), f.ravel())),
                              shape=(len(f), len(v)))
        G = sparse.bmat([[None, A], [A.T, None]]).tocsr()
        _, lab = sparse.csgraph.connected_components(G, directed=False)
        flab = lab[:len(f)]
        f = f[flab == np.bincount(flab).argmax()]
        used, f = np.unique(f, return_inverse=True)
        v, f = v[used], f.reshape(-1, 3)
        if not bad.any():
            break
    return v, f


def local_frames(v, f):
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    e1 = p1 - p0
    l1 = np.linalg.norm(e1, axis=1)
    ex = e1 / np.maximum(l1, 1e-30)[:, None]
    n = np.cross(e1, p2 - p0)
    ey = np.cross(n, ex)
    ey /= np.maximum(np.linalg.norm(ey, axis=1), 1e-30)[:, None]
    d2 = p2 - p0
    X = np.zeros((len(f), 3, 2))
    X[:, 1, 0] = l1
    X[:, 2, 0] = np.einsum("ij,ij->i", d2, ex)
    X[:, 2, 1] = np.einsum("ij,ij->i", d2, ey)
    return X


def lscm_solve(v, f):
    """Least-squares conformal map (Lévy et al. 2002), free boundary, two pinned vertices on the base."""
    from scipy.sparse.linalg import spsolve
    X = local_frames(v, f)
    m, n = len(f), len(v)
    area2 = np.abs(cross2(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]))
    s = 1 / np.sqrt(np.maximum(area2, 1e-30))
    Wr = np.zeros((m, 3)); Wi = np.zeros((m, 3))
    for j in range(3):
        d = X[:, (j + 2) % 3] - X[:, (j + 1) % 3]
        Wr[:, j], Wi[:, j] = d[:, 0] * s, d[:, 1] * s
    rows = np.repeat(np.arange(m), 3)
    cols = f.ravel()
    M = sparse.vstack([
        sparse.hstack([sparse.csr_matrix((Wr.ravel(), (rows, cols)), (m, n)),
                       sparse.csr_matrix((-Wi.ravel(), (rows, cols)), (m, n))]),
        sparse.hstack([sparse.csr_matrix((Wi.ravel(), (rows, cols)), (m, n)),
                       sparse.csr_matrix((Wr.ravel(), (rows, cols)), (m, n))]),
    ]).tocsc()
    low = np.nonzero(v[:, 2] < np.quantile(v[:, 2], 0.1))[0]
    pins = np.array([low[np.argmin(v[low, 0])], low[np.argmax(v[low, 0])]])
    pin_val = np.array([[0.0, 0.0], [np.linalg.norm(v[pins[1]] - v[pins[0]]), 0.0]])
    fixed = np.concatenate([pins, pins + n])
    fixed_val = np.concatenate([pin_val[:, 0], pin_val[:, 1]])
    free = np.setdiff1d(np.arange(2 * n), fixed)
    A = M[:, free]
    x = np.zeros(2 * n)
    x[fixed] = fixed_val
    N = (A.T @ A).tocsc()
    N = N + sparse.identity(N.shape[0], format="csc") * (1e-9 * N.diagonal().mean())   # stays solvable
    x[free] = spsolve(N, A.T @ (-M[:, fixed] @ fixed_val))
    return np.stack([x[:n], x[n:]], 1)


def arap_solve(v, f, uv0, iters: int = 15):
    """As-rigid-as-possible parameterisation, local/global iterations from uv0."""
    from scipy.sparse.linalg import splu
    X = local_frames(v, f)
    m, n = len(f), len(v)
    cot = np.zeros((m, 3))
    for k in range(3):
        a = X[:, k] - X[:, (k + 2) % 3]
        b = X[:, (k + 1) % 3] - X[:, (k + 2) % 3]
        cot[:, k] = np.einsum("ij,ij->i", a, b) / np.maximum(np.abs(cross2(a, b)), 1e-30)
    cot = np.clip(cot, 1e-3, 1e3)
    I = np.concatenate([f[:, k] for k in range(3)])
    J = np.concatenate([f[:, (k + 1) % 3] for k in range(3)])
    w = np.concatenate([cot[:, k] for k in range(3)])
    L = sparse.csr_matrix((np.concatenate([w, w, -w, -w]),
                           (np.concatenate([I, J, I, J]), np.concatenate([I, J, J, I]))), (n, n)).tolil()
    L[0, 0] += 1.0
    L = (L.tocsc() + sparse.identity(n, format="csc") * 1e-9)
    lu = splu(L)
    uv = uv0.copy()
    for _ in range(iters):
        U = uv[f]
        S = np.zeros((m, 2, 2))
        for k in range(3):
            du = U[:, k] - U[:, (k + 1) % 3]
            dx = X[:, k] - X[:, (k + 1) % 3]
            S += cot[:, k, None, None] * du[:, :, None] * dx[:, None, :]
        Us, _, Vt = np.linalg.svd(S)
        flip = np.linalg.det(Us @ Vt) < 0
        Us[flip, :, 1] *= -1
        R = Us @ Vt
        b = np.zeros((n, 2))
        for k in range(3):
            dx = X[:, k] - X[:, (k + 1) % 3]
            r = cot[:, k, None] * np.einsum("mij,mj->mi", R, dx)
            np.add.at(b, f[:, k], r)
            np.add.at(b, f[:, (k + 1) % 3], -r)
        b[0] += uv[0]
        uv = np.stack([lu.solve(b[:, 0]), lu.solve(b[:, 1])], 1)
    return uv


def scale_uv(uv, v, f):
    """Scale uv so that the unwrap has the same total area as the surface."""
    a3 = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum()
    U = uv[f]
    a2 = np.abs(cross2(U[:, 1] - U[:, 0], U[:, 2] - U[:, 0])).sum()
    uv = uv - uv.min(0)
    return uv * np.sqrt(a3 / a2)


def closest_on_triangles(p, A, B, C):
    e0, e1, d = B - A, C - A, p - A
    d00, d01, d11 = (e0 * e0).sum(-1), (e0 * e1).sum(-1), (e1 * e1).sum(-1)
    d20, d21 = (d * e0).sum(-1), (d * e1).sum(-1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    b1 = (d11 * d20 - d01 * d21) / den
    b2 = (d00 * d21 - d01 * d20) / den
    b = np.clip(np.stack([1 - b1 - b2, b1, b2], -1), 0, None)
    b /= np.maximum(b.sum(-1, keepdims=True), 1e-30)
    q = A * b[..., :1] + B * b[..., 1:2] + C * b[..., 2:]
    return b, np.linalg.norm(p - q, axis=-1)


def locate(points, v, f, uv, k: int = 6, chunk: int = 400_000):
    """uv of points on (or near) the simplified surface: closest point among the k nearest triangles."""
    tree = cKDTree(v[f].mean(1))
    out = np.empty((len(points), 2))
    for s in range(0, len(points), chunk):
        c = points[s:s + chunk]
        _, cand = tree.query(c, k=k, workers=-1)
        T = v[f[cand]]
        b, d = closest_on_triangles(c[:, None], T[..., 0, :], T[..., 1, :], T[..., 2, :])
        j = d.argmin(1)
        r = np.arange(len(c))
        out[s:s + chunk] = (uv[f[cand[r, j]]] * b[r, j][..., None]).sum(1)
    return out


# ── main entry ────────────────────────────────────────────────────────────────
def compute(scan: Scan, iso: Isolation, method: str, max_side: int, progress=None) -> Unwrap:
    t0 = time.perf_counter()
    st = StoneFrame(scan, iso)
    sampler = TextureSampler(scan)
    names = {"arap": "ARAP", "boxnet": "box net", "belt": "belt"}
    if progress:
        progress(f"Unwrapping the stone ({names[method]}) …")
    if method in ("boxnet", "belt"):
        C, dep = (boxnet if method == "boxnet" else belt)(st)
        upp = float(max(C[..., 0].max(), C[..., 1].max())) / (max_side - 4)
        P = C.reshape(-1, 2) / upp
        W, H = int(np.ceil(P[:, 0].max())) + 2, int(np.ceil(P[:, 1].max())) + 2
        m = len(C)
        face, bary, zbuf = rasterize(P, dep.reshape(-1), np.arange(3 * m).reshape(m, 3), W, H)
        full = np.where(face >= 0, st.faces[np.maximum(face, 0)], -1)
        image = sampler.sample(full, bary)
        px = C.mean(1) / upp
        # visible = not behind the front surface at its pixel (faces are often smaller than a pixel)
        ix = np.clip(px[:, 0].astype(np.int64), 0, W - 1)
        iy = np.clip(px[:, 1].astype(np.int64), 0, H - 1)
        visible = dep.mean(1) >= zbuf[iy, ix] - 0.001 * st.size
    else:
        if progress:
            progress("Unwrapping the stone (ARAP): simplifying the mesh …")
        v, f = simplified(st)
        if progress:
            progress(f"Unwrapping the stone (ARAP): flattening {len(f):,} triangles …")
        uv = scale_uv(arap_solve(v, f, lscm_solve(v, f)), v, f)
        upp = float(uv.max()) / (max_side - 4)
        P = uv / upp
        W, H = int(np.ceil(P[:, 0].max())) + 2, int(np.ceil(P[:, 1].max())) + 2
        if progress:
            progress("Unwrapping the stone (ARAP): rendering the unwrap …")
        face, bary, _ = rasterize(P, np.zeros(len(P)), f, W, H)
        m = face >= 0
        fm = face[m]
        l0, l1 = bary[m, 0], bary[m, 1]
        X = v[f[fm, 0]] * l0[:, None] + v[f[fm, 1]] * l1[:, None] + v[f[fm, 2]] * (1 - l0 - l1)[:, None]
        cen = st.L[st.F].mean(1)
        _, near = cKDTree(cen).query(X, workers=-1)
        full = np.full(face.shape, -1, np.int64)
        full[m] = st.faces[near]
        image = sampler.sample(full, np.full(face.shape + (2,), 1 / 3, np.float32))
        if progress:
            progress("Unwrapping the stone (ARAP): locating every face on the unwrap …")
        px = locate(cen, v, f, uv) / upp
        visible = np.ones(len(cen), bool)
    if progress:
        progress(f"Unwrap {image.shape[1]}×{image.shape[0]} px in {time.perf_counter() - t0:.0f} s")
    return Unwrap(method, image, face >= 0, st.faces, px, upp, visible, st.L[st.F].mean(1))
