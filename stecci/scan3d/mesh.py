"""Scan loading (GLB / glTF / OBJ / PLY via trimesh), cache and mesh topology.

Scan arrays (merged over all parts of the file):
  V  (n,3) float32 vertex positions (node transforms applied)
  F  (m,3) int32   faces
  UV (n,2) float32 texture coordinates (v up), C (n,3) uint8 vertex colours
  FT (m,)  int16   texture page per face (-1 = no texture: vertex colours)
  textures: list of RGB uint8 arrays
Topology (computed once per scan, cached):
  Fw   (m,3) welded vertex ids (the scanner splits vertices along texture seams)
  P    (k,3) position per welded vertex
  adj  (e,4) faces sharing an edge: face a, face b, welded vertex 0, welded vertex 1
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse

from ..config import cache_dir

Image.MAX_IMAGE_PIXELS = None
SCAN_EXTS = (".glb", ".gltf", ".obj", ".ply")
CACHE_VERSION = 2


@dataclass
class Scan:
    path: Path
    V: np.ndarray
    F: np.ndarray
    UV: np.ndarray
    C: np.ndarray
    FT: np.ndarray
    textures: list = field(default_factory=list)
    Fw: np.ndarray | None = None
    P: np.ndarray | None = None
    adj: np.ndarray | None = None
    area: np.ndarray | None = None          # per face, model units²
    n_original: int = 0                     # triangles in the file (before reduction)

    @property
    def n_faces(self) -> int:
        return len(self.F)

    def face_graph(self, mask: np.ndarray | None = None, same: np.ndarray | None = None) -> sparse.csr_matrix:
        """Face adjacency (shared edges), optionally only between faces in `mask` and with equal `same` values."""
        a, b = self.adj[:, 0], self.adj[:, 1]
        keep = np.ones(len(a), bool)
        if mask is not None:
            keep &= mask[a] & mask[b]
        if same is not None:
            keep &= same[a] == same[b]
        a, b = a[keep], b[keep]
        m = self.n_faces
        return sparse.csr_matrix((np.ones(len(a), np.int8), (a, b)), shape=(m, m))


def _key(path: Path, max_faces: int, tex_px: int) -> str:
    st = path.stat()
    h = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{CACHE_VERSION}|{max_faces}|{tex_px}"
                     .encode()).hexdigest()
    return h[:16]


def load_scan(path: str | Path, progress=None, max_faces: int = 0, tex_px: int = 4096) -> Scan:
    """Load a scan; with max_faces > 0 larger scans are reduced to that many triangles (texture re-baked)."""
    path = Path(path)
    cdir = cache_dir() / _key(path, max_faces, tex_px)
    if (cdir / "mesh.npz").is_file() and (cdir / "topo.npz").is_file():
        if progress:
            progress(f"Loading {path.name} (cached) …")
        d = np.load(cdir / "mesh.npz")
        t = np.load(cdir / "topo.npz")
        tex = [np.asarray(Image.open(p).convert("RGB")) for p in sorted(cdir.glob("tex_*.jpg"))]
        s = Scan(path, d["V"], d["F"], d["UV"], d["C"], d["FT"], tex, t["Fw"], t["P"], t["adj"])
        s.n_original = int(d["n_original"]) if "n_original" in d else s.n_faces
    else:
        s = _import(path, progress)
        s.n_original = s.n_faces
        if 0 < max_faces < s.n_faces:
            from .reduce import reduce_scan
            s = reduce_scan(s, max_faces, tex_px, progress)
        if progress:
            progress("Analysing the mesh topology …")
        s.Fw, s.P = weld(s.V, s.F)
        s.adj = edge_adjacency(s.Fw)
        cdir.mkdir(parents=True, exist_ok=True)
        np.savez(cdir / "mesh.npz", V=s.V, F=s.F, UV=s.UV, C=s.C, FT=s.FT, n_original=s.n_original)
        np.savez(cdir / "topo.npz", Fw=s.Fw, P=s.P, adj=s.adj)
        for i, t in enumerate(s.textures):
            Image.fromarray(t).save(cdir / f"tex_{i}.jpg", quality=95)
    T = s.V[s.F].astype(np.float64)
    s.area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
    return s


def _import(path: Path, progress=None) -> Scan:
    import trimesh
    if progress:
        progress(f"Reading {path.name} …")
    t0 = time.perf_counter()
    obj = trimesh.load(path, process=False)
    scene = obj if isinstance(obj, trimesh.Scene) else trimesh.Scene(obj)
    Vs, Fs, UVs, Cs, FTs, tex = [], [], [], [], [], []
    off = 0
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        v = np.asarray(g.vertices, np.float64)
        v = (v @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        f = np.asarray(g.faces, np.int32)
        vis = g.visual
        uv = np.zeros((len(v), 2), np.float32)
        col = np.full((len(v), 3), 180, np.uint8)
        page = -1
        if isinstance(vis, trimesh.visual.TextureVisuals) and vis.uv is not None:
            mat = vis.material
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if img is not None:
                uv = np.asarray(vis.uv, np.float32)
                tex.append(np.asarray(img.convert("RGB")))
                page = len(tex) - 1
            try:
                c = np.asarray(vis.vertex_attributes["color"])
                col = c[:, :3].astype(np.uint8)
            except Exception:
                pass
        else:
            try:
                col = np.asarray(vis.vertex_colors)[:, :3].astype(np.uint8)
            except Exception:
                pass
        Vs.append(v); Fs.append(f + off); UVs.append(uv); Cs.append(col)
        FTs.append(np.full(len(f), page, np.int16))
        off += len(v)
    if not Fs:
        raise ValueError(f"{path.name} contains no triangle mesh")
    if progress:
        progress(f"Read {sum(len(f) for f in Fs):,} triangles in {time.perf_counter() - t0:.0f} s")
    return Scan(path, np.concatenate(Vs), np.concatenate(Fs), np.concatenate(UVs), np.concatenate(Cs),
                np.concatenate(FTs), tex)


def weld(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices (texture seams): welded face ids + one position per welded vertex."""
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    q = np.round(V.astype(np.float64) / max(diag * 1e-7, 1e-12)).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    P = np.zeros((inv.max() + 1, 3), np.float32)
    P[inv] = V
    return inv[F].astype(np.int32), P


def edge_adjacency(Fw: np.ndarray) -> np.ndarray:
    """Pairs of faces sharing an edge (manifold edges; on non-manifold edges consecutive faces are paired)."""
    m = len(Fw)
    e = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]).astype(np.int64)
    e.sort(1)
    face = np.tile(np.arange(m), 3)
    key = e[:, 0] * (int(Fw.max()) + 1) + e[:, 1]
    o = np.argsort(key, kind="stable")
    key, face, e = key[o], face[o], e[o]
    same = key[1:] == key[:-1]
    i = np.nonzero(same)[0]
    return np.stack([face[i], face[i + 1], e[i, 0], e[i, 1]], 1).astype(np.int32)


def face_components(scan: Scan, mask: np.ndarray, same: np.ndarray | None = None) -> np.ndarray:
    """Connected components (shared edges) of the faces in `mask`; -1 outside the mask."""
    from scipy.sparse.csgraph import connected_components
    idx = np.nonzero(mask)[0]
    out = np.full(scan.n_faces, -1, np.int64)
    if len(idx) == 0:
        return out
    a, b = scan.adj[:, 0], scan.adj[:, 1]
    keep = mask[a] & mask[b]
    if same is not None:
        keep &= same[a] == same[b]
    loc = np.full(scan.n_faces, -1, np.int64)
    loc[idx] = np.arange(len(idx))
    G = sparse.csr_matrix((np.ones(int(keep.sum()), np.int8), (loc[a[keep]], loc[b[keep]])),
                          shape=(len(idx), len(idx)))
    _, lab = connected_components(G, directed=False)
    out[idx] = lab
    return out


class TextureSampler:
    """Colour of surface points given face ids and barycentric coordinates (via the scan's own UV atlas)."""

    def __init__(self, scan: Scan):
        self.s = scan

    def sample(self, face: np.ndarray, bary: np.ndarray, background=(255, 255, 255)) -> np.ndarray:
        s = self.s
        out = np.empty(face.shape + (3,), np.uint8)
        out[:] = background
        m = face >= 0
        f = face[m]
        l0, l1 = bary[m, 0], bary[m, 1]
        l2 = 1 - l0 - l1
        w = np.stack([l0, l1, l2], 1)
        cols = np.empty((len(f), 3), np.uint8)
        ft = s.FT[f]
        notex = ft < 0
        if notex.any():
            c = (s.C[s.F[f[notex]]].astype(np.float32) * w[notex, :, None]).sum(1)
            cols[notex] = np.clip(c, 0, 255).astype(np.uint8)
        uv = (s.UV[s.F[f]] * w[..., None]).sum(1)
        for i, t in enumerate(s.textures):
            sel = ft == i
            if not sel.any():
                continue
            th, tw = t.shape[:2]
            u = np.clip((uv[sel, 0] * tw).astype(np.int64), 0, tw - 1)
            v = np.clip(((1 - uv[sel, 1]) * th).astype(np.int64), 0, th - 1)
            cols[sel] = t[v, u]
        out[m] = cols
        return out
