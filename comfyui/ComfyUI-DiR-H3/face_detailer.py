# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""DiR H3 Face Detailer: redraw a small face (or head and shoulders) at a higher resolution with DiR's last
iterations, then paste it back.

In a 768x512 isometric shot a face is about 2x2 DiT tokens (32 px per token), too few to draw eyes or a mouth;
two-track's late full-resolution steps cannot redraw them either (docs/TWO_TRACK.md). This node:

1. finds the face in every frame (insightface ``buffalo_l`` detector, CPU) and takes ONE box for the clip (the
   union of the expanded per-frame boxes, so the crop video is temporally stable), or uses ``box`` if given;
2. crops that box from the decoded video, resizes it to ``crop_size`` (long side) and VAE-encodes it;
3. re-noises the crop to DiR's sigma at ``start_iteration`` (fresh noise from ``seed``) and runs DiR's
   iterations from there, with the crop's own estimate as the previous estimate (``video_extra``) and the given
   conditioning: a Ref2VA reference keeps the identity, a close-up prompt describes the crop;
4. VAE-decodes the crop, matches its colour to the original crop (per channel mean / std over the clip), and
   blends it back through a feathered mask.

Everything DiR does here is the ordinary DiR sampler on a smaller latent; only the starting state differs.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("ComfyUI-DiR-H3")
_APP = None


def _detector():
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis

        _APP = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"])
        _APP.prepare(ctx_id=-1, det_size=(640, 640))
    return _APP


def detect_faces(frames: torch.Tensor, every: int = 2, min_score: float = 0.4) -> list:
    """Largest face box ``(x0, y0, x1, y1)`` per sampled frame (``None`` when no face). ``frames``: (T, H, W, 3) 0-1."""
    app = _detector()
    boxes = []
    for i in range(0, frames.shape[0], max(1, every)):
        bgr = (frames[i].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()
        # small faces: detect on a 2x upscale
        scale = 2 if max(bgr.shape[:2]) <= 1024 else 1
        if scale > 1:
            import cv2

            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        faces = [f for f in app.get(bgr) if f.det_score >= min_score]
        if faces:
            f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            boxes.append(tuple(float(v) / scale for v in f.bbox))
        else:
            boxes.append(None)
    return boxes


def clip_box(boxes: list, width: int, height: int, expand: float, aspect: float, min_px: int = 64):
    """One box for the clip: the union of the per-frame face boxes, each grown ``expand`` times about its centre
    (and shifted down a little to take in hair, neck and shoulders), widened to ``aspect`` (w / h) and kept inside
    the frame. ``None`` when no face was found."""
    found = [b for b in boxes if b is not None]
    if not found:
        return None
    grown = []
    for x0, y0, x1, y1 in found:
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        side = max(w, h) * expand
        cy += 0.15 * side
        grown.append((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
    x0 = min(b[0] for b in grown)
    y0 = min(b[1] for b in grown)
    x1 = max(b[2] for b in grown)
    y1 = max(b[3] for b in grown)
    w, h = max(x1 - x0, min_px), max(y1 - y0, min_px)
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    w, h = min(w, width), min(h, height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = min(max(cx - w / 2, 0), width - w)
    y0 = min(max(cy - h / 2, 0), height - h)
    return int(round(x0)), int(round(y0)), int(round(x0 + w)), int(round(y0 + h))


def parse_box(spec: str):
    parts = [p for p in spec.replace(",", " ").split() if p]
    if len(parts) != 4:
        raise ValueError(f"box {spec!r}: give x0 y0 x1 y1 in pixels")
    return tuple(int(float(p)) for p in parts)


def crop_shape(box, crop_size: int) -> tuple[int, int]:
    """Crop pixel size: long side ``crop_size``, both sides multiples of 32, aspect kept."""
    w, h = box[2] - box[0], box[3] - box[1]
    s = crop_size / max(w, h)
    return max(64, int(round(w * s / 32)) * 32), max(64, int(round(h * s / 32)) * 32)


def resize_frames(frames: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """(T, H, W, 3) -> (T, height, width, 3), bicubic (antialiased when shrinking)."""
    x = frames.permute(0, 3, 1, 2)
    down = width < x.shape[3] or height < x.shape[2]
    x = F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False, antialias=down)
    return x.clamp(0, 1).permute(0, 2, 3, 1)


def feather_mask(height: int, width: int, feather: float) -> torch.Tensor:
    """(H, W) mask, 1 inside, cosine fall-off over ``feather`` of the smaller side at every edge."""
    band = max(1.0, feather * min(height, width))

    def ramp(n):
        d = torch.minimum(torch.arange(n, dtype=torch.float32) + 0.5, n - torch.arange(n, dtype=torch.float32) - 0.5)
        t = (d / band).clamp(0, 1)
        return 0.5 - 0.5 * torch.cos(math.pi * t)

    return ramp(height)[:, None] * ramp(width)[None, :]


def match_color(new: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-channel mean / std of ``new`` matched to ``ref`` over the whole clip (both (T, H, W, 3))."""
    dims = (0, 1, 2)
    nm, ns = new.mean(dims), new.std(dims).clamp_min(1e-4)
    rm, rs = ref.mean(dims), ref.std(dims).clamp_min(1e-4)
    return ((new - nm) / ns * rs + rm).clamp(0, 1)


def detail_loop(x0: torch.Tensor, start: int):
    """A ``dir_core.dir_loop``-shaped recursion that starts at iteration ``start`` from ``x0`` re-noised to that
    iteration's sigma (with the sampler's own ``z``), feeding ``x0`` as the previous estimate."""
    from .vendor.dir_core import next_state

    def loop(step, z, iters, sigmas, mix_state):
        clean = x0.to(z.device, torch.float32)
        state = sigmas[start] * z.float() + (1.0 - sigmas[start]) * clean
        estimate = clean
        estimates = []
        for r in range(start, iters):
            out = step(r, state, estimate, sigmas[r])
            estimate = state.float() + sigmas[r] * out.float()
            estimates.append(estimate)
            if r + 1 < iters:
                state = next_state(state, estimate, sigmas[r], sigmas[r + 1], z if mix_state else None)
        return estimate, estimates, state

    return loop


def detail(model, bundle, positive, vae, images: torch.Tensor, *, seed: int, start_iteration: int, crop_size: int,
           expand: float, feather: float, color_match: bool, box: str = "", sample_fn=None) -> tuple[torch.Tensor, str]:
    """Returns the detailed frames and a one-line report."""
    import comfy.nested_tensor

    from . import dir_runtime
    from .vendor.dir_core import audio_frames

    frames = images.float()
    t, height, width, _c = frames.shape
    if box.strip():
        region = parse_box(box)
    else:
        found = detect_faces(frames)
        region = clip_box(found, width, height, expand, aspect=1.0)
        if region is None:
            log.warning("DiR face detailer: no face found; frames returned unchanged")
            return images, "no face found"
    x0, y0, x1, y1 = region
    cw, ch = crop_shape(region, crop_size)
    crop = frames[:, y0:y1, x0:x1]
    up = resize_frames(crop, cw, ch)
    latent = vae.encode(up[..., :3])
    if latent.dim() != 5:
        raise RuntimeError(f"expected a video latent (1, 24, T, h, w) from the VAE, got {tuple(latent.shape)}")
    latent_t = latent.shape[2]
    empty = comfy.nested_tensor.NestedTensor((torch.zeros_like(latent),
                                              torch.zeros(1, 32, 2, audio_frames(latent_t), dtype=latent.dtype)))
    iters = len(bundle.sigmas)
    if not 0 <= start_iteration < iters:
        raise ValueError(f"start_iteration {start_iteration} is outside 0..{iters - 1}")
    run = sample_fn or dir_runtime.sample
    result = run(model, bundle.head, positive, {"samples": empty}, seed=seed, sigmas=bundle.sigmas,
                 mix_state=bundle.mix_state, loop=detail_loop(latent.float(), start_iteration))
    estimate = result["estimate"].to(latent.dtype)
    decoded = vae.decode(estimate)
    decoded = decoded.reshape(-1, *decoded.shape[-3:])[:t].float()
    if decoded.shape[0] < t:                       # the VAE's frame count: repeat the last frame if short
        decoded = torch.cat([decoded, decoded[-1:].expand(t - decoded.shape[0], -1, -1, -1)])
    back = resize_frames(decoded[..., :3], x1 - x0, y1 - y0)
    if color_match:
        back = match_color(back, crop[..., :3])
    mask = feather_mask(y1 - y0, x1 - x0, feather)[None, :, :, None]
    out = frames.clone()
    out[:, y0:y1, x0:x1, :3] = mask * back + (1 - mask) * crop[..., :3]
    report = (f"box {region} ({x1 - x0}x{y1 - y0} px) -> crop {cw}x{ch} ({cw / (x1 - x0):.1f}x), "
              f"DiR iterations {start_iteration}..{iters - 1} from sigma {bundle.sigmas[start_iteration]:.3f}")
    log.info("DiR face detailer: %s", report)
    return out.to(images.dtype), report
