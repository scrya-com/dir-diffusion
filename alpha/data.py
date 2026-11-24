# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""RGBA stills -> RGBA pseudo-videos for training an alpha channel on a video VAE.

A still is cropped, then moved through ``frames`` frames along a random smooth path:
* static;
* affine: pan, zoom, roll, eased;
* drift: affine plus a slowly morphing low-frequency warp (smoke, flames, cloth);
* fade: the alpha ramps up or down (premultiplied, so colour fades with it).

Warps act on premultiplied RGBA with transparent padding, so edges stay physically consistent. The colour under
transparent pixels is LayerDiffuse-padded (``alpha_math.layerdiffuse_pad_rgb``): that padded colour is what the
VAE encodes and what a DiT is later trained to generate.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .alpha_math import layerdiffuse_pad_rgb

SOFT_LO, SOFT_HI = 0.02, 0.98


def load_rgba(path, size: int) -> torch.Tensor:
    """(4, size, size) straight RGBA in [0, 1], resized with premultiplied filtering."""
    im = Image.open(path).convert("RGBA")
    x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float().div(255.0)
    a = x[3:4]
    pm = torch.cat([x[:3] * a, a], 0)[None]
    pm = F.interpolate(pm, size=(size, size), mode="bicubic", antialias=True, align_corners=False)[0].clamp(0, 1)
    a = pm[3:4]
    rgb = torch.where(a > 1e-4, pm[:3] / a.clamp_min(1e-4), torch.zeros_like(pm[:3])).clamp(0, 1)
    return torch.cat([rgb, a], 0)


def soft_fraction(alpha: torch.Tensor) -> float:
    return float(((alpha > SOFT_LO) & (alpha < SOFT_HI)).float().mean())


class RGBAStills(torch.utils.data.Dataset):
    """Rows of the AlphaAnything t2rgba jsonl. Returns a (4, crop, crop) straight-RGBA crop that contains
    foreground, after resizing the 1024 still to a random size in ``scale_px``."""

    def __init__(self, root, rows: list[dict], crop: int = 256, scale_px=(272, 448), train: bool = True):
        self.root, self.rows, self.crop, self.scale_px, self.train = Path(root), rows, crop, scale_px, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        rng = random.Random(None if self.train else i)
        size = rng.randint(*self.scale_px) if self.train else (self.scale_px[0] + self.scale_px[1]) // 2
        x = load_rgba(self.root / row["image"], size)
        a = x[3]
        c = self.crop
        ys, xs = torch.nonzero(a > 0.05, as_tuple=True)
        if len(ys):
            k = rng.randrange(len(ys))
            cy, cx = int(ys[k]), int(xs[k])
        else:
            cy, cx = size // 2, size // 2
        y0 = min(max(cy - c // 2 + rng.randint(-c // 4, c // 4), 0), size - c)
        x0 = min(max(cx - c // 2 + rng.randint(-c // 4, c // 4), 0), size - c)
        return x[:, y0:y0 + c, x0:x0 + c], i


def _ease(t):
    return t * t * (3 - 2 * t)


def make_clips(stills: torch.Tensor, frames: int, gen: torch.Generator, modes=("static", "affine", "drift", "fade"),
               probs=(0.15, 0.4, 0.3, 0.15), return_modes: bool = False, return_premul: bool = False):
    """``stills`` (B, 4, H, W) straight RGBA on the GPU -> (padded RGB (B, 3, T, H, W) in [0, 1], alpha (B, 1, T, H, W))."""
    b, _, h, w = stills.shape
    dev = stills.device
    rgb, a = stills[:, :3], stills[:, 3:4]
    pm = torch.cat([rgb * a, a], 1)
    t = torch.linspace(0, 1, frames, device=dev)
    out, picked = [], []
    for i in range(b):
        mode = modes[int(torch.multinomial(torch.tensor(probs), 1, generator=gen))]
        picked.append(mode)
        r = lambda lo, hi: lo + (hi - lo) * float(torch.rand((), generator=gen))
        e = _ease(t) if mode != "static" else torch.zeros_like(t)
        tx0, ty0, tx1, ty1 = r(-.1, .1), r(-.1, .1), r(-.15, .15), r(-.15, .15)
        s0, s1 = r(.9, 1.1), r(.9, 1.15)
        th0, th1 = math.radians(r(-6, 6)), math.radians(r(-10, 10))
        tx, ty = tx0 + (tx1 - tx0) * e, ty0 + (ty1 - ty0) * e
        s, th = s0 + (s1 - s0) * e, th0 + (th1 - th0) * e
        if mode == "static":
            tx, ty, s, th = torch.zeros_like(t), torch.zeros_like(t), torch.ones_like(t), torch.zeros_like(t)
        cos, sin = torch.cos(th) / s, torch.sin(th) / s
        theta = torch.stack([torch.stack([cos, -sin, tx], -1), torch.stack([sin, cos, ty], -1)], 1)  # (T, 2, 3)
        grid = F.affine_grid(theta, (frames, 4, h, w), align_corners=False)
        if mode == "drift":
            amp = r(0.01, 0.05)
            f0 = torch.randn(1, 2, 4, 4, generator=gen).to(dev)
            f1 = torch.randn(1, 2, 4, 4, generator=gen).to(dev)
            field = f0 + (f1 - f0) * t.view(-1, 1, 1, 1)
            field = F.interpolate(field, size=(h, w), mode="bicubic", align_corners=False) * amp
            grid = grid + field.permute(0, 2, 3, 1)
        clip = F.grid_sample(pm[i:i + 1].expand(frames, -1, -1, -1), grid, mode="bilinear", padding_mode="zeros",
                             align_corners=False)
        if mode == "fade":
            lo = r(0.2, 0.7)
            ramp = (1 - (1 - lo) * t) if float(torch.rand((), generator=gen)) < 0.5 else (lo + (1 - lo) * t)
            clip = clip * ramp.view(-1, 1, 1, 1)
        out.append(clip)
    clip = torch.stack(out, 0).clamp(0, 1)  # (B, T, 4, H, W) premultiplied
    alpha = clip[:, :, 3:4]
    straight = torch.where(alpha > 1e-4, clip[:, :, :3] / alpha.clamp_min(1e-4), torch.zeros_like(clip[:, :, :3]))
    padded = layerdiffuse_pad_rgb(straight.flatten(0, 1), alpha.flatten(0, 1)).view(b, frames, 3, h, w)
    res = (padded.transpose(1, 2).contiguous(), alpha.transpose(1, 2).contiguous())
    if return_premul:  # premultiplied colour, for compositing over a background
        res = res + (clip[:, :, :3].transpose(1, 2).contiguous(),)
    return res + (picked,) if return_modes else res


def split_rows(jsonl, root, out_json, n_soft=(("effect", 100), ("translucent", 60), ("person", 60)), n_other=80,
               seed=0):
    """Hold out a soft-alpha-heavy eval split: the softest images of the soft categories plus random others.
    Writes {"train": [...], "eval": [...], "soft_frac": {...}} to ``out_json`` and returns it."""
    rows = [json.loads(line) for line in open(jsonl)]
    fracs = {}
    for r in rows:
        a = load_rgba(Path(root) / r["image"], 256)[3]
        fracs[r["image"]] = soft_fraction(a)
    rng = random.Random(seed)
    held = set()
    for cat, n in n_soft:
        cands = sorted([r for r in rows if r["category"] == cat], key=lambda r: -fracs[r["image"]])
        top = cands[: 3 * n]
        held.update(r["image"] for r in rng.sample(top, min(n, len(top))))
    rest = [r for r in rows if r["image"] not in held]
    held.update(r["image"] for r in rng.sample(rest, n_other))
    split = {"train": [r for r in rows if r["image"] not in held], "eval": [r for r in rows if r["image"] in held],
             "soft_frac": fracs}
    Path(out_json).write_text(json.dumps(split))
    return split
