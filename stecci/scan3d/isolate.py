"""Isolate the stone from the scanned ground and align it to its own box axes.

1. ground plane = plane fitted (RANSAC) to the open outer edge of the scan (the edge of the scanned ground)
2. stone = largest connected piece of the faces more than `cut` above that plane
3. alignment: up from the ground normal, then the box axes refined from the area-weighted face normals
Scans without an open edge (a closed stone model) keep all faces; up is then the file's up axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Scan, face_components


@dataclass
class Isolation:
    stone: np.ndarray        # bool per face
    R: np.ndarray            # 3x3, rows = stone axes (x length, y width, z up) in model coordinates
    origin: np.ndarray       # min corner of the stone in the rotated frame
    ground: tuple | None     # (point, normal) of the ground plane, or None
    up: np.ndarray           # up direction in model coordinates


def boundary_vertices(Fw: np.ndarray) -> np.ndarray:
    e = np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]).astype(np.int64), 1)
    key = e[:, 0] * (int(Fw.max()) + 1) + e[:, 1]
    u, c = np.unique(key, return_counts=True)
    b = u[c == 1]
    return np.unique(np.concatenate([b // (int(Fw.max()) + 1), b % (int(Fw.max()) + 1)]))


def fit_plane(P: np.ndarray, tol: float, iters: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    best_n, best = -1, None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < tol
        if inl.sum() > best_n:
            best_n, best = int(inl.sum()), inl
    Q = P[best]
    c = Q.mean(0)
    return c, np.linalg.svd(Q - c)[2][2]


def ground_plane(scan: Scan):
    """(point, upward normal) of the scanned ground, or None for a closed model."""
    bv = boundary_vertices(scan.Fw)
    if len(bv) < 30:
        return None
    P = scan.P[bv].astype(np.float64)
    diag = float(np.linalg.norm(scan.P.max(0) - scan.P.min(0)))
    c, n = fit_plane(P, tol=0.004 * diag)          # ~3 cm on a 7 m scan
    h = (scan.P.astype(np.float64) - c) @ n
    if abs(np.quantile(h, 0.001)) > abs(np.quantile(h, 0.999)):   # the stone sticks out on the up side
        n = -n
    return c, n


def align_box(N: np.ndarray, A: np.ndarray, R0: np.ndarray, iters: int = 20) -> np.ndarray:
    """Rotation (rows = box axes) that best maps area-weighted face normals onto the 6 box directions."""
    R = R0.copy()
    dirs = np.concatenate([np.eye(3), -np.eye(3)])
    for _ in range(iters):
        local = N @ R.T
        k = np.argmax(local @ dirs.T, 1)
        target = dirs[k]
        w = A * (np.einsum("ij,ij->i", local, target) > np.cos(np.radians(30)))
        H = (N * w[:, None]).T @ target
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
    return R


def file_up(scan: Scan) -> np.ndarray:
    return np.array([0, 1.0, 0]) if scan.path.suffix.lower() in (".glb", ".gltf") else np.array([0, 0, 1.0])


def isolate(scan: Scan, cut_cm: float, scale: float = 1.0, progress=None) -> Isolation:
    """cut_cm: ground cut height in cm; scale: metres per model unit (to convert the cut)."""
    if progress:
        progress("Fitting the ground plane …")
    gp = ground_plane(scan)
    V = scan.V.astype(np.float64)
    if gp is None:
        stone = np.ones(scan.n_faces, bool)
        up = file_up(scan)
    else:
        c, up = gp
        if progress:
            progress("Cutting the stone from the ground …")
        h = (V - c) @ up
        keep = h[scan.F].min(1) > cut_cm / 100 / scale
        comp = face_components(scan, keep)
        big = np.bincount(comp[keep]).argmax()
        stone = keep & (comp == big)
    R, origin = align(scan, stone, up)
    return Isolation(stone, R, origin, gp, up)


def align(scan: Scan, stone: np.ndarray, up: np.ndarray):
    V = scan.V.astype(np.float64)
    Fs = scan.F[stone]
    T = V[Fs]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    A = np.linalg.norm(n, axis=1)
    N = n / np.maximum(A, 1e-20)[:, None]
    P = V[np.unique(Fs)]
    tmp = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    ex = tmp - up * (tmp @ up); ex /= np.linalg.norm(ex)
    ey = np.cross(up, ex)
    xy = np.stack([P @ ex, P @ ey], 1)
    ev = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))[1][:, ::-1]
    ax = ev[0, 0] * ex + ev[1, 0] * ey
    R0 = np.stack([ax, np.cross(up, ax), up])
    R = align_box(N, A / 2, R0)
    if R[2] @ up < 0:
        R[2] *= -1; R[1] *= -1
    L = P @ R.T
    return R, L.min(0)
