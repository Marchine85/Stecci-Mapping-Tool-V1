"""Build RGBA overlay layers (numpy) from the project state."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..segmentation import boundaries, mask_outline

CRACK_RGB = (230, 30, 30)
ROCK_OUTLINE_RGB = (255, 20, 200)     # neon pink (default stećak outline)
SELECT_RGB = (0, 0, 0)          # selection hatch / outline
SELECT_SPACING = 9              # px between hatch lines


def _empty(shape):
    return np.zeros((*shape, 4), np.uint8)


def _bbox(mask):
    ys = np.flatnonzero(mask.any(1))
    xs = np.flatnonzero(mask.any(0))
    return xs[0], ys[0], xs[-1] + 1, ys[-1] + 1


def fill_layer(project, opacity: float, dim_background: bool, dim_alpha: int = 150,
               crop_preview: bool = False) -> np.ndarray:
    h, w = project.shape
    out = _empty((h, w))
    if project.labels is not None:
        n = max(project.classes) + 1
        lut = np.zeros((max(n, int(project.seg_class.max()) + 1), 4), np.uint8)
        a = int(round(255 * opacity))
        for cid, c in project.classes.items():
            if cid == 0:
                continue
            lut[cid] = (*c["color"], a)
        out = lut[project.class_map()]
    if dim_background and project.rock is not None:
        out[~project.rock] = (0, 0, 0, dim_alpha)
    if crop_preview:
        box = project.crop_box()
        if box is not None:                      # everything outside the export crop is hidden
            x0, y0, x1, y1 = box
            keep = np.zeros(out.shape[:2], bool)
            keep[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True     # box may extend beyond the photo
            out[~keep] = (24, 25, 28, 255)
    return out


LINE_STYLES = ("solid", "dashed", "dotted")


def _pattern_mask(labels: np.ndarray, style: str, thickness: int) -> np.ndarray:
    """Pixels to keep for a dashed / dotted line.

    The pattern runs along each region's traced contour, so dashes look the same in every
    direction (a fixed screen pattern would turn lines of some angles solid).
    """
    t = max(1, int(thickness))
    on, period = (4 * t + 3, 7 * t + 5) if style == "dashed" else (1, 3 * t + 2)
    keep = np.zeros(labels.shape, bool)
    for sid, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        sub = (labels[sl] == sid).astype(np.uint8)
        contours, _ = cv2.findContours(sub, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        kv = keep[sl]
        for c in contours:
            pts = c[:, 0, :]
            sel = (np.arange(len(pts)) % period) < on
            kv[pts[sel, 1], pts[sel, 0]] = True
    return keep


def _styled(edge: np.ndarray, labels: np.ndarray, style: str, thickness: int) -> np.ndarray:
    """1-px edge pixels -> line mask with the given style and thickness."""
    if style in ("dashed", "dotted"):
        edge = edge & _pattern_mask(labels, style, thickness)
    if thickness > 1:
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((thickness, thickness), np.uint8)) > 0
    return edge


def boundary_layer(project, show_segments: bool = True, color=(255, 255, 255),
                   thickness: int = 2, style: str = "solid",
                   outline_color=ROCK_OUTLINE_RGB, outline_style: str = "solid") -> np.ndarray:
    h, w = project.shape
    out = _empty((h, w))
    if show_segments and project.labels is not None:
        b = boundaries(project.labels)
        if project.rock is not None:
            b &= project.rock                    # the rock outline is drawn separately
        out[_styled(b, project.labels, style, thickness)] = (*color, 255)
    if project.rock is not None:
        if outline_style in ("dashed", "dotted"):
            r8 = project.rock.astype(np.uint8)          # 1-px inner edge of the stone
            edge = project.rock & ~(cv2.erode(r8, np.ones((3, 3), np.uint8)) > 0)
            line = _styled(edge, project.rock.astype(np.int32), outline_style, 3)
        else:
            line = mask_outline(project.rock, 2)
        out[line] = (*outline_color, 255)
    return out


def white_map(project, classes: bool = True, cracks: bool = False, crack_color=CRACK_RGB,
              fill_alpha: float = 0.5, line_rgb=(0, 0, 0), boundary_px: int = 2, boundary_style: str = "solid",
              outline_px: int = 3, surroundings: bool = False, segment_lines: bool = True) -> np.ndarray:
    """Printable map: the stone photo on a white background, class colours, black segment outlines
    (unless segment_lines=False) and black stone outline."""
    h, w = project.shape
    rock = project.rock
    if surroundings:                                          # whole photo, background not dimmed
        img = project.image.copy()
    else:
        img = np.full((h, w, 3), 255, np.uint8)
        if rock is not None:
            img[rock] = project.image[rock]                   # the stone itself, background stays white
    if classes and project.labels is not None:
        cm = project.class_map()
        for cid, c in project.classes.items():
            if cid == 0:
                continue
            m = cm == cid
            if m.any():
                col = np.array(c["color"], np.float32)
                img[m] = np.round(fill_alpha * col + (1 - fill_alpha) * img[m]).astype(np.uint8)
    if segment_lines and project.labels is not None:          # every segment outline, in black
        b = boundaries(project.labels)
        if rock is not None:
            b &= rock
        img[_styled(b, project.labels, boundary_style, boundary_px)] = line_rgb
    if cracks and project.crack is not None:
        img[project.crack] = crack_color
    if rock is not None:
        img[mask_outline(rock, outline_px - 1)] = line_rgb
    return img


def selection_layer(project, selected: set[int]) -> np.ndarray:
    h, w = project.shape
    out = _empty((h, w))
    m = np.isin(project.labels, list(selected)) if selected and project.labels is not None else None
    if m is not None and m.any():
        # black diagonal crosshatch: clearly different from any class colour fill
        x0, y0, x1, y1 = _bbox(m)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        hatch = (((xx + yy) % SELECT_SPACING) < 2) | (((xx - yy) % SELECT_SPACING) < 2)
        sub = out[y0:y1, x0:x1]
        sub[m[y0:y1, x0:x1] & hatch] = (*SELECT_RGB, 210)
        out[mask_outline(m, 3)] = (*SELECT_RGB, 255)
    return out


def crack_layer(project, color=CRACK_RGB) -> np.ndarray:
    out = _empty(project.shape)
    out[project.crack] = (*color, 235)
    return out


def composite(base_rgb: np.ndarray, *layers: np.ndarray) -> np.ndarray:
    img = base_rgb.astype(np.float32)
    for L in layers:
        a = L[..., 3:4].astype(np.float32) / 255
        img = img * (1 - a) + L[..., :3].astype(np.float32) * a
    return np.clip(img, 0, 255).astype(np.uint8)
