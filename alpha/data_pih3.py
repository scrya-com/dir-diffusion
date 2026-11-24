# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""RGBA stills -> composited clips at pi-H3's canvas, for an alpha head on pi-H3's DWD refiner.

At inference the alpha head reads what pi-H3 drew. A probe of pi-H3 4-step on ten captions of the RGBA set
(/media/2TB/alpha/probe) put the subject on flat black (5 of 10), white (2) or studio grey (1), or in a real scene
(1); none looked like LayerDiffuse padding. So training clips composite the moving subject over (weights in
``BACKGROUNDS``): flat dark, flat light, a random flat colour, a smooth gradient, a real-scene frame, or the
LayerDiffuse padding of P1. The target is the true alpha.
"""

from __future__ import annotations

import math
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .alpha_math import layerdiffuse_pad_rgb
from .data import _ease, load_rgba

BACKGROUNDS = {"dark": 0.3, "light": 0.2, "flat": 0.1, "gradient": 0.15, "scene": 0.15, "padded": 0.1}
SCENE_DIRS = ("dojo", "dojo_nat", "nautica", "camfix", "mpeg_video", "dir_h3")


def build_scene_pool(comfy_output: Path, out: Path, count: int = 240, size=(512, 768), seed: int = 0) -> None:
    """One frame from each of ``count`` rendered clips, resized to cover ``size``: (N, 3, H, W) uint8."""
    rng = random.Random(seed)
    clips = sorted(p for d in SCENE_DIRS for p in (comfy_output / d).rglob("*.mp4"))
    rng.shuffle(clips)
    h, w = size
    frames = []
    for clip in clips:
        if len(frames) >= count:
            break
        cmd = ["ffmpeg", "-v", "error", "-ss", "0.5", "-i", str(clip), "-frames:v", "1", "-vf",
               f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(cmd, capture_output=True).stdout
        if len(raw) == h * w * 3:
            frames.append(torch.from_numpy(np.frombuffer(raw, np.uint8).copy()).view(h, w, 3).permute(2, 0, 1))
    torch.save(torch.stack(frames), out)
    print(f"scene pool: {len(frames)} frames from {len(clips)} clips -> {out}")


def place(still: torch.Tensor, canvas, rng: random.Random, train: bool = True) -> torch.Tensor:
    """(4, S, S) straight RGBA -> (4, H, W) straight RGBA on a transparent canvas, the subject scaled to 55-95% of
    the canvas height (fixed 75% when not ``train``) and placed so most of it stays in frame."""
    h, w = canvas
    frac = rng.uniform(0.55, 0.95) if train else 0.75
    side = max(32, int(h * frac / 0.9))  # stills leave ~10% margin
    pm = torch.cat([still[:3] * still[3:4], still[3:4]], 0)[None]
    pm = F.interpolate(pm, size=(side, side), mode="bicubic", antialias=True, align_corners=False)[0].clamp(0, 1)
    out = torch.zeros(4, h, w)
    cy = rng.randint(h // 2 - h // 8, h // 2 + h // 8) if train else h // 2
    cx = rng.randint(w // 2 - w // 5, w // 2 + w // 5) if train else w // 2
    y0, x0 = cy - side // 2, cx - side // 2
    ys, xs = max(0, -y0), max(0, -x0)
    ye, xe = min(side, h - y0), min(side, w - x0)
    out[:, y0 + ys:y0 + ye, x0 + xs:x0 + xe] = pm[:, ys:ye, xs:xe]
    a = out[3:4]
    rgb = torch.where(a > 1e-4, out[:3] / a.clamp_min(1e-4), torch.zeros_like(out[:3])).clamp(0, 1)
    return torch.cat([rgb, a], 0)


def background(kind: str, frames: int, h: int, w: int, gen: torch.Generator, scenes: torch.Tensor | None,
               device) -> torch.Tensor:
    """(3, T, H, W) in [0, 1]. Static, except gradients drift slowly."""
    r = lambda lo, hi: lo + (hi - lo) * float(torch.rand((), generator=gen))
    if kind == "dark":
        c = torch.full((3,), r(0.0, 0.08))
        return c.view(3, 1, 1, 1).expand(3, frames, h, w).to(device)
    if kind == "light":
        c = torch.full((3,), r(0.85, 1.0)) - torch.rand(3, generator=gen) * 0.04
        return c.view(3, 1, 1, 1).expand(3, frames, h, w).to(device)
    if kind == "flat":
        return torch.rand(3, generator=gen).view(3, 1, 1, 1).expand(3, frames, h, w).to(device)
    if kind == "gradient":
        c0, c1 = torch.rand(3, generator=gen), torch.rand(3, generator=gen)
        ang, drift = r(0, 2 * math.pi), r(-0.3, 0.3)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        t = torch.linspace(0, 1, frames).view(-1, 1, 1)
        u = ((xx * math.cos(ang) + yy * math.sin(ang))[None] + drift * t) * 0.5 + 0.5
        u = u.clamp(0, 1)[None]
        return (c0.view(3, 1, 1, 1) * (1 - u) + c1.view(3, 1, 1, 1) * u).to(device)
    if kind == "scene" and scenes is not None:
        k = int(torch.randint(len(scenes), (1,), generator=gen))
        img = scenes[k].float().div(255)
        if img.shape[-2:] != (h, w):
            img = F.interpolate(img[None], size=(h, w), mode="bilinear", align_corners=False)[0]
        return img.view(3, 1, h, w).expand(3, frames, h, w).to(device)
    raise ValueError(kind)


def make_clip(still: torch.Tensor, frames: int, gen: torch.Generator, *, scenes=None, kind: str | None = None,
              mode: str | None = None):
    """``still`` (4, H, W) straight RGBA on the GPU, already placed on the canvas -> (RGB (3, T, H, W) composited
    in [0, 1], alpha (1, T, H, W), background kind, motion mode). Motion as ``data.make_clips``."""
    from .data import make_clips

    _c, h, w = still.shape
    if mode is None:
        modes, probs = ("static", "affine", "drift", "fade"), (0.15, 0.45, 0.3, 0.1)
        mode = modes[int(torch.multinomial(torch.tensor(probs), 1, generator=gen))]
    if kind is None:
        names = list(BACKGROUNDS)
        kind = names[int(torch.multinomial(torch.tensor([BACKGROUNDS[n] for n in names]), 1, generator=gen))]
        if kind == "scene" and scenes is None:
            kind = "gradient"
    padded, alpha, premul = make_clips(still[None], frames, gen, modes=(mode,), probs=(1.0,), return_premul=True)
    padded, alpha, premul = padded[0], alpha[0], premul[0]
    if kind == "padded":
        return padded, alpha, kind, mode
    bg = background(kind, frames, h, w, gen, scenes, still.device)
    return (premul + bg * (1 - alpha)).clamp(0, 1), alpha, kind, mode  # C = alpha F + (1 - alpha) B
