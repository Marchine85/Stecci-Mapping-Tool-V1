"""Export renderings: six views of the stone and a GLB with the mapping burned into its texture.

Both work on the CPU (no GL context needed) and use the same colours as the viewer:
  per face  rgb + alpha (class colour at the chosen opacity, crack colour opaque), boundaries where `key` changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .mesh import TextureSampler
from .project import Project
from .raster import ortho, rasterize


def face_colours(p: Project, opts: dict):
    """(m,3) rgb, (m,) alpha, (m,) boundary key for every face of the scan."""
    m = p.scan.n_faces
    rgb = np.zeros((m, 3), np.float32)
    alpha = np.zeros(m, np.float32)
    key = np.zeros(m, np.int64)
    if p.seg is not None:
        cf = p.class_faces()
        pal = np.zeros((max(p.classes) + 1, 3), np.float32)
        for cid, c in p.classes.items():
            pal[cid] = c["color"]
        if opts.get("show_fill", True):
            rgb = pal[cf]
            alpha = np.where(cf > 0, opts.get("opacity", 0.5), 0.0).astype(np.float32)
        if opts.get("show_bound", True):
            key = p.seg.astype(np.int64)
    if opts.get("show_crack", True) and p.crack is not None:
        rgb[p.crack] = opts.get("crack_color", (230, 30, 30))
        alpha[p.crack] = 1.0
    return rgb, alpha, key


def views_sheet(p: Project, opts: dict, mm_per_px: float | None = None, legend: np.ndarray | None = None,
                progress=None) -> Image.Image:
    """Six views (top, four obliques, one end) of the stone with the mapping, as one image."""
    scan = p.scan
    stone_idx = np.nonzero(p.stone)[0]
    used, Fl = np.unique(scan.F[stone_idx], return_inverse=True)
    Fl = Fl.reshape(-1, 3)
    L = scan.V[used].astype(np.float64) @ p.iso.R.T
    size = float((L.max(0) - L.min(0)).max())
    upp = size / 1400 if mm_per_px is None else mm_per_px / 1000 / p.scale
    rgb, alpha, key = face_colours(p, opts)
    sampler = TextureSampler(scan)
    bcol = np.array(opts.get("bound_color", (57, 255, 20)), np.float32)

    def cam(w):
        w = np.asarray(w, float); w /= np.linalg.norm(w)
        r = np.cross([0, 0, 1], w); r /= np.linalg.norm(r)
        return r, np.cross(w, r)
    views = {"Top": (np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
             "Front right": cam([0.55, 1, 0.75]), "Front left": cam([-0.55, 1, 0.75]),
             "Back right": cam([0.55, -1, 0.75]), "Back left": cam([-0.55, -1, 0.75]),
             "Left end": cam([-1, 0.25, 0.5])}
    tiles = []
    for name, (r, u) in views.items():
        if progress:
            progress(f"Rendering view: {name.lower()} …")
        P, z, W, H = ortho(L, r, u, upp * 1000)              # ortho takes mm/px with metres = units
        face, bary, _ = rasterize(P, z, Fl, W, H)
        m = face >= 0
        full = np.where(m, stone_idx[np.maximum(face, 0)], -1)
        img = sampler.sample(full, bary).astype(np.float32)
        f = full[m]
        a = alpha[f][:, None]
        img[m] = img[m] * (1 - a) + rgb[f] * a
        k = np.where(m, key[np.maximum(full, 0)], -1)
        edge = np.zeros(k.shape, bool)
        edge[:, 1:] |= (k[:, 1:] != k[:, :-1]) & m[:, 1:] & m[:, :-1]
        edge[1:, :] |= (k[1:, :] != k[:-1, :]) & m[1:, :] & m[:-1, :]
        if opts.get("show_bound", True):
            edge = ndi.binary_dilation(edge, iterations=max(0, int(opts.get("bound_width", 2)) // 2))
            img[edge & m] = bcol
        tiles.append((name, Image.fromarray(img.astype(np.uint8))))
    w = max(t.width for _, t in tiles); h = max(t.height for _, t in tiles)
    pad = 24
    sheet = Image.new("RGB", (3 * w + 4 * pad, 2 * h + 3 * pad + 30), "white")
    d = ImageDraw.Draw(sheet)
    for i, (name, t) in enumerate(tiles):
        x0, y0 = pad + (i % 3) * (w + pad), pad + 30 + (i // 3) * (h + pad)
        sheet.paste(t, (x0 + (w - t.width) // 2, y0 + (h - t.height) // 2))
        d.text((x0, y0 - 18), name, fill=(40, 40, 40))
    if legend is not None:
        lg = Image.fromarray(legend, "RGBA")
        sheet.paste(lg, (sheet.width - lg.width - pad, pad), lg)
    return sheet


def export_glb(p: Project, opts: dict, path: Path, tex_px: int = 4096, progress=None):
    """The isolated stone with the mapping burned into a copy of its texture atlas (rasterised in uv space)."""
    import trimesh
    scan = p.scan
    rgb, alpha, key = face_colours(p, opts)
    bcol = np.array(opts.get("bound_color", (57, 255, 20)), np.float32)
    stone_idx = np.nonzero(p.stone)[0]
    meshes = []
    for page in np.unique(scan.FT[stone_idx]):
        sel = stone_idx[scan.FT[stone_idx] == page]
        Fp = scan.F[sel]
        used, Fl = np.unique(Fp, return_inverse=True)
        verts = scan.V[used].astype(np.float64)
        if page < 0:                                         # no texture: vertex colours, no burn-in
            meshes.append(trimesh.Trimesh(verts, Fl.reshape(-1, 3), vertex_colors=scan.C[used], process=False))
            continue
        if progress:
            progress(f"3D model: texture page {page + 1} …")
        tex = Image.fromarray(scan.textures[page])
        side = min(tex_px, max(tex.size))
        if max(tex.size) != side:
            tex = tex.resize((side, side), Image.LANCZOS)
        img = np.asarray(tex).astype(np.float32)
        P = np.c_[scan.UV[:, 0] * side, (1 - scan.UV[:, 1]) * side]
        face, _, _ = rasterize(P, np.zeros(len(P)), Fp, side, side)
        cov = face >= 0
        # extend the face assignment a few texels into the chart gutters (texture filtering / mip-maps)
        dist, (iy, ix) = ndi.distance_transform_edt(~cov, return_indices=True)
        fid = np.where(cov, face, face[iy, ix])
        fid[dist > 3] = -1
        g = fid >= 0
        f = sel[fid[g]]
        a = alpha[f][:, None]
        img[g] = img[g] * (1 - a) + rgb[f] * a
        if opts.get("show_bound", True):
            k = np.where(g, key[sel[np.maximum(fid, 0)]], -1)
            edge = np.zeros(k.shape, bool)
            edge[:, 1:] |= (k[:, 1:] != k[:, :-1]) & cov[:, 1:] & cov[:, :-1]
            edge[1:, :] |= (k[1:, :] != k[:-1, :]) & cov[1:, :] & cov[:-1, :]
            img[edge] = bcol
        visual = trimesh.visual.TextureVisuals(uv=scan.UV[used], image=Image.fromarray(img.astype(np.uint8)))
        meshes.append(trimesh.Trimesh(verts, Fl.reshape(-1, 3), visual=visual, process=False))
    if progress:
        progress("Writing the 3D model …")
    trimesh.Scene(meshes).export(path)
