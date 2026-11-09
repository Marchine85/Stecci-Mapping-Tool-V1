"""Grounding DINO + SAM wrappers.

Heavy imports (torch, transformers, segment_anything) are done lazily so the
window opens quickly. Model calls run in worker threads; a lock serialises the
background preload and the user's jobs.

Everything works offline: Grounding DINO is read from the app's models folder
(or the local Hugging Face cache), never downloaded.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from .config import (ROCK_PROMPT, Settings, candidate_model_dirs, find_sam_checkpoint, SAM_FILES,
                     SAM_URL)


_AMG_PATCHED = False


def _patch_segment_anything():
    """Speed up SamAutomaticMaskGenerator's bookkeeping.

    The stock MaskData deep-copies the RLE lists on every batch and indexes a
    tensor element by element when filtering, which dominates run time at fine
    granularity (~2/3 of total). The RLEs are never mutated, so a shallow copy
    is safe.
    """
    global _AMG_PATCHED
    if _AMG_PATCHED:
        return
    import torch
    from copy import deepcopy
    from segment_anything.utils import amg

    def filter(self, keep):
        keep_list = None
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list):
                if keep_list is None:
                    kk = keep.detach().cpu()
                    keep_list = (kk.nonzero().flatten() if kk.dtype == torch.bool else kk).tolist()
                self._stats[k] = [v[i] for i in keep_list]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def cat(self, new_stats):
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = list(v) if isinstance(v, list) else deepcopy(v)
            elif isinstance(v, torch.Tensor):
                self._stats[k] = torch.cat([self._stats[k], v], dim=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + v
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    amg.MaskData.filter = filter
    amg.MaskData.cat = cat
    _AMG_PATCHED = True


def device_description() -> str:
    """Which device the models run on, for the Settings dialog."""
    import torch
    if torch.cuda.is_available():
        return f"GPU · {torch.cuda.get_device_name(0)}"
    if torch.version.cuda is None:
        return "CPU · this build has no GPU support"
    return "CPU · no NVIDIA GPU found"


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._sam = None
        self._predictor = None
        self._gdino = None
        self._gdino_proc = None
        self._embedded_key = None
        self.device = None
        self.sam_type = None
        self.ready = False              # preload + warm-up finished
        self._lock = threading.RLock()

    # ── configuration ────────────────────────────────────────────────────────
    def reset(self, settings: Settings):
        """Drop loaded models (e.g. after settings changed)."""
        with self._lock:
            self.settings = settings
            self._sam = self._predictor = self._gdino = self._gdino_proc = None
            self._embedded_key = None
            self.ready = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _resolve_device(self) -> str:
        """The GPU whenever this build and the PC support it (GPU build + NVIDIA card), else the CPU."""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def describe(self) -> str:
        if self.device is None:
            return "models not loaded"
        dino = " + DINO" if self._gdino is not None else ""
        state = "" if self.ready else " · loading …"
        return f"device: {self.device} · SAM {self.sam_type}{dino}{state}"

    def preload(self, progress=None):
        """Load the models needed for detection and warm up the GPU (run in the background at start)."""
        with self._lock:
            self.load_sam(progress)
            dummy = np.zeros((256, 256, 3), np.uint8)
            self._predictor.set_image(dummy)          # first-run CUDA / cuDNN setup
            self._predictor.predict(point_coords=np.array([[128, 128]], np.float32),
                                    point_labels=np.array([1], np.int32), multimask_output=True)
            self._embedded_key = None
            if self.settings.use_dino:
                self.load_gdino(progress)
                self.detect_rock_box(dummy)
            self.ready = True

    # ── loading ──────────────────────────────────────────────────────────────
    def load_sam(self, progress=None):
        with self._lock:
            self._load_sam(progress)

    def _load_sam(self, progress=None):
        if self._sam is not None:
            return
        import torch
        from segment_anything import sam_model_registry, SamPredictor

        self.device = self._resolve_device()
        mtype = self.settings.sam_model_type
        if mtype == "auto":
            mtype = "vit_h" if self.device == "cuda" else "vit_b"
        ckpt = Path(self.settings.sam_checkpoint) if self.settings.sam_checkpoint else find_sam_checkpoint(mtype)
        if ckpt is None or not ckpt.is_file():
            raise FileNotFoundError(
                f"SAM checkpoint for '{mtype}' not found.\n"
                f"Download {SAM_FILES[mtype]} from {SAM_URL}\n"
                f"and put it into the 'models' folder, or set the path in Settings.")
        if progress:
            progress(f"Loading SAM {mtype} on {self.device} …")
        sam = sam_model_registry[mtype](checkpoint=str(ckpt))
        sam.to(self.device).eval()
        self._sam = sam
        self._predictor = SamPredictor(sam)
        self.sam_type = mtype
        self._embedded_key = None
        torch.set_grad_enabled(False)

    def gdino_source(self) -> str:
        """models/<name> folder if present (offline bundle), else the Hugging Face id (local cache)."""
        mid = self.settings.gdino_model_id
        name = mid.split("/")[-1]
        for d in candidate_model_dirs():
            if (d / name / "config.json").is_file():
                return str(d / name)
        return mid

    def load_gdino(self, progress=None):
        with self._lock:
            self._load_gdino(progress)

    def _load_gdino(self, progress=None):
        if self._gdino is not None:
            return
        os.environ["HF_HUB_OFFLINE"] = "1"          # never touch the internet
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError as e:                    # the packaged app ships without transformers
            raise RuntimeError("Grounding DINO is not available (transformers is not installed).\n"
                               "Set use_dino = False in config.py.") from e

        if self.device is None:
            self.device = self._resolve_device()
        src = self.gdino_source()
        if progress:
            progress(f"Loading Grounding DINO ({Path(src).name}) …")
        try:
            proc = AutoProcessor.from_pretrained(src, local_files_only=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(src, local_files_only=True)
        except Exception as e:
            name = self.settings.gdino_model_id.split("/")[-1]
            raise FileNotFoundError(
                f"Grounding DINO model files not found ({self.settings.gdino_model_id}).\n"
                f"Put the model folder into 'models/{name}' next to the app, "
                f"or set use_dino = False in config.py.\n\n{e}") from e
        self._gdino_proc = proc
        self._gdino = model.to(self.device).eval()

    # ── Grounding DINO: rock box ─────────────────────────────────────────────
    def detect_rock_box(self, image: np.ndarray, progress=None) -> tuple[np.ndarray | None, float]:
        """Return (box xyxy, score) of the detection nearest the image centre, or (None, 0)."""
        with self._lock:
            return self._detect_rock_box(image, progress)

    def _detect_rock_box(self, image: np.ndarray, progress=None) -> tuple[np.ndarray | None, float]:
        import torch
        from PIL import Image

        self.load_gdino(progress)
        if progress:
            progress("Grounding DINO: locating stećak …")
        s = self.settings
        prompt = ROCK_PROMPT
        pil = Image.fromarray(image)
        inputs = self._gdino_proc(images=pil, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._gdino(**inputs)
        h, w = image.shape[:2]
        res = self._gdino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=s.box_threshold,
            text_threshold=s.text_threshold, target_sizes=[(h, w)])[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        if len(boxes) == 0:
            return None, 0.0
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = areas >= 0.03 * h * w          # ignore small parts (pebbles, details)
        if keep.any():
            boxes, scores = boxes[keep], scores[keep]
        centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
        d = np.linalg.norm(centres - np.array([w / 2, h / 2]), axis=1)
        i = int(np.argmin(d))
        return boxes[i].astype(np.float32), float(scores[i])

    # ── SAM: rock mask from box / points ─────────────────────────────────────
    def _ensure_embedding(self, image: np.ndarray, key, progress=None):
        with self._lock:
            self.load_sam(progress)
            if self._embedded_key != key:
                if progress:
                    progress("SAM: analysing photo …")
                self._predictor.set_image(image)
                self._embedded_key = key

    def detect_rock_sam(self, image: np.ndarray, key, progress=None):
        """Isolate the stone without a detector: SAM from the photo centre, then refined with that mask's box.

        Returns (box xyxy, score, mask).
        """
        from .segmentation import bbox_of

        with self._lock:
            self._ensure_embedding(image, key, progress)
            if progress:
                progress("SAM: isolating the stone in the centre of the photo …")
            h, w = image.shape[:2]
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array([[w / 2, h / 2]], np.float32), point_labels=np.array([1], np.int32),
                multimask_output=True)
            # SAM gives part / object / whole-scene candidates: take the whole stone, not a part of it
            best, best_s = int(np.argmax(scores)), -1.0
            for i, (m, s) in enumerate(zip(masks, scores)):
                frac = float(m.mean())
                if 0.03 <= frac <= 0.85 and s + 0.3 * frac > best_s:
                    best, best_s = i, s + 0.3 * frac
            box = np.array(bbox_of(masks[best]), np.float32)
            m2, s2, _ = self._predictor.predict(box=box, multimask_output=False)
            return box, float(s2[0]), m2[0].astype(bool)

    def predict_mask(self, image: np.ndarray, key, box=None, points=None, point_labels=None,
                     progress=None) -> np.ndarray:
        with self._lock:
            self._ensure_embedding(image, key, progress)
            pc = np.asarray(points, np.float32) if points is not None and len(points) else None
            pl = np.asarray(point_labels, np.int32) if pc is not None else None
            bx = np.asarray(box, np.float32) if box is not None else None
            multimask = bx is None and (pc is None or len(pc) == 1)
            masks, scores, _ = self._predictor.predict(
                point_coords=pc, point_labels=pl, box=bx, multimask_output=multimask)
            return masks[int(np.argmax(scores))].astype(bool)

    # ── SAM: automatic mask generation ───────────────────────────────────────
    def auto_masks(self, image_crop: np.ndarray, g, min_area: int, progress=None) -> list[np.ndarray]:
        """g: config.Granularity"""
        from segment_anything import SamAutomaticMaskGenerator

        _patch_segment_anything()
        self.load_sam(progress)
        if progress:
            extra = " + zoomed crops" if g.crop_layers else ""
            progress(f"SAM auto-segmentation ({g.points_per_side}×{g.points_per_side} points{extra}) …")
        gen = SamAutomaticMaskGenerator(
            self._sam,
            points_per_side=g.points_per_side,
            points_per_batch=128 if self.device == "cuda" else 32,
            pred_iou_thresh=g.iou_thresh,
            stability_score_thresh=g.stability,
            crop_n_layers=g.crop_layers,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=0,     # cleanup is done in segmentation.py
        )
        with self._lock:
            out = gen.generate(image_crop)
        return [m["segmentation"] for m in out if m["area"] >= min_area]
