# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""DWD (decoding without the VAE) inference pieces, vendored from dir-diffusion ``lot/dwd_h3.py``,
``lot/refine_dwd_h3.py`` and ``lot/train_dwd_h3.py`` (commit db1f6f3). The arithmetic is unchanged; training code
is dropped. Two additions for ComfyUI: ``DWDHead.decode`` takes ``device`` / ``dtype`` and moves one class's
weights at a time (the extended ridge head is 3.15 GB in fp32), and ``load_head`` / ``load_refiner`` map with
``mmap`` so a load does not copy the file into RAM twice.

The head reads ``[h, s]`` per token: ``h`` is the modulated last-layer state ``final_layer.video_out`` reads (the
final DiR iteration, with the DWD LoRA on), ``s`` the final iteration's input state packed like the video rows.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 5376
TOKEN_DIM = 96
PATCH = 32
SLOTS = 4
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
OUT = SLOTS * 3 * PATCH * PATCH
CLASSES = 5
ESLOTS = SLOTS + 2


def frames_of(latent_t: int) -> list[int]:
    return [FRAME_PER_TOKEN[k % 5] for k in range(latent_t)]


def pixel_frame_count(latent_t: int) -> int:
    return sum(frames_of(latent_t))


def pack_rows(latent: torch.Tensor) -> torch.Tensor:
    """``(1, 24, T, H, W)`` -> ``(T*(H/2)*(W/2), 96)``, the order ``pack_video_rows`` uses."""
    _b, c, t, h, w = latent.shape
    rows = latent[0].reshape(c, t, h // 2, 2, w // 2, 2).permute(1, 2, 4, 0, 3, 5)
    return rows.reshape(t * (h // 2) * (w // 2), c * 4)


def tokens_to_pixels(tokens: torch.Tensor, latent_t: int, token_h: int, token_w: int) -> torch.Tensor:
    """``(N, 12288)`` -> ``(1, 3, F, 32*token_h, 32*token_w)``."""
    per = token_h * token_w
    if tokens.shape[0] != latent_t * per:
        raise ValueError(f"{tokens.shape[0]} tokens, expected {latent_t * per}")
    frames = []
    for k, n in enumerate(frames_of(latent_t)):
        tiles = tokens[k * per:(k + 1) * per].reshape(token_h, token_w, SLOTS, 3, PATCH, PATCH)
        clip = tiles.permute(3, 2, 0, 4, 1, 5).reshape(3, SLOTS, token_h * PATCH, token_w * PATCH)
        frames.append(clip[:, :n])
    return torch.cat(frames, dim=1).unsqueeze(0)


def token_classes(latent_t: int, token_h: int, token_w: int, device=None) -> torch.Tensor:
    return (torch.arange(latent_t, device=device) % CLASSES).repeat_interleave(token_h * token_w)


def ext_out(halo: int) -> int:
    return ESLOTS * 3 * (PATCH + 2 * halo) ** 2


def ext_halo(out: int) -> int | None:
    side = round((out / (ESLOTS * 3)) ** 0.5)
    if out == OUT or side < PATCH or (side - PATCH) % 2 or ESLOTS * 3 * side * side != out:
        return None
    return (side - PATCH) // 2


def blend_tokens(tokens: torch.Tensor, latent_t: int, token_h: int, token_w: int, halo: int,
                 blend: float = 0.25) -> torch.Tensor:
    """Overlap-add decode of extended tokens -> ``(1, 3, F, 32*token_h, 32*token_w)``."""
    side, per, dev = PATCH + 2 * halo, token_h * token_w, tokens.device
    height, width = token_h * PATCH + 2 * halo, token_w * PATCH + 2 * halo
    ramp = (torch.arange(2 * halo, device=dev) + 0.5) / max(2 * halo, 1)
    window = torch.cat([ramp, torch.ones(side - 4 * halo, device=dev), ramp.flip(0)])
    window = torch.outer(window, window)

    def fold(values: torch.Tensor) -> torch.Tensor:
        return nn.functional.fold(values, (height, width), kernel_size=side, stride=PATCH)

    cover = fold(window.reshape(1, -1, 1).expand(1, -1, per))
    counts = frames_of(latent_t)
    f = sum(counts)
    out = torch.zeros(f, 3, height, width, device=dev)
    weight = torch.zeros(f, device=dev)
    start = 0
    for k, n in enumerate(counts):
        tiles = tokens[k * per:(k + 1) * per].reshape(per, ESLOTS, 3, side, side).float() * window
        frames = fold(tiles.permute(1, 2, 3, 4, 0).reshape(ESLOTS, 3 * side * side, per))
        for i in range(n):
            on_boundary = (i == 0 and start > 0) or (i == n - 1 and start + n < f)
            w = 1.0 - blend if on_boundary else 1.0
            out[start + i] += w * frames[1 + i]
            weight[start + i] += w
        if blend > 0 and start > 0:
            out[start - 1] += blend * frames[0]
            weight[start - 1] += blend
        if blend > 0 and start + n < f:
            out[start + n] += blend * frames[ESLOTS - 1]
            weight[start + n] += blend
        start += n
    out = out / weight[:, None, None, None] / cover
    return out[:, :, halo:height - halo, halo:width - halo].permute(1, 0, 2, 3).unsqueeze(0)


class DWDHead(nn.Module):
    """``pixels = W_c ((x - mean) / scale) + b_c`` for ``x = [h, s]`` and ``c`` the token's class."""

    def __init__(self, hidden: int = HIDDEN, state_dim: int = TOKEN_DIM, out: int = OUT, classes: int = CLASSES):
        super().__init__()
        self.inputs, self.classes = hidden + state_dim, classes
        self.halo, self.blend = ext_halo(out), 0.25
        self.weight = nn.Parameter(torch.zeros(classes, out, self.inputs))
        self.bias = nn.Parameter(torch.zeros(classes, out))
        self.register_buffer("mean", torch.zeros(self.inputs))
        self.register_buffer("scale", torch.ones(self.inputs))

    def features(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return (torch.cat([h.float(), s.float()], dim=-1) - self.mean.to(h.device)) / self.scale.to(h.device)

    @torch.no_grad()
    def forward(self, h: torch.Tensor, s: torch.Tensor, classes: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Same as ``lot/dwd_h3.DWDHead.forward``. The weights may live on another device (CPU, mmap); each
        class's ``(out, inputs)`` block is copied to ``h.device`` (in ``dtype``, default fp32) only while it is used."""
        x = self.features(h, s)
        dtype = dtype or torch.float32
        out = x.new_empty(x.shape[0], self.weight.shape[1])
        for c in range(self.classes):
            index = (classes % self.classes) == c if self.classes > 1 else torch.ones_like(classes, dtype=torch.bool)
            if not index.any():
                continue
            weight = self.weight[c].to(x.device, dtype, non_blocking=True)
            bias = self.bias[c].to(x.device, dtype)
            out[index] = (x[index].to(dtype) @ weight.T + bias).float()
            del weight
        return out

    @torch.no_grad()
    def decode(self, h: torch.Tensor, s: torch.Tensor, latent_t: int, token_h: int, token_w: int,
               dtype: torch.dtype | None = None) -> torch.Tensor:
        """Pixels in [-1, 1], ``(1, 3, F, 32*token_h, 32*token_w)``, on ``h.device``."""
        out = self(h, s, token_classes(latent_t, token_h, token_w, h.device), dtype=dtype)
        if self.halo is not None:
            return blend_tokens(out, latent_t, token_h, token_w, self.halo, self.blend).clamp(-1, 1)
        return tokens_to_pixels(out, latent_t, token_h, token_w).clamp(-1, 1)


def load_head(path) -> DWDHead:
    """``lot/train_dwd_h3.load_head`` on the CPU, memory-mapped (``assign`` keeps the mapped tensors)."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    blend = state.pop("blend", None)
    with torch.device("meta"):
        head = DWDHead(classes=state["weight"].shape[0], out=state["weight"].shape[1],
                       hidden=state["weight"].shape[2] - TOKEN_DIM)
    head.load_state_dict(state, assign=True)
    if blend is not None:
        head.blend = float(blend)
    return head.eval().requires_grad_(False)


# ----------------------------------------------------------------------------------------------- refiner
# lot/refine_dwd_h3.py: a small U-Net that reads a DWD frame, its two neighbours and (optionally) the latent.

class Block(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.silu(self.c1(F.silu(x))))


class LatentPath(nn.Module):
    def __init__(self, in_ch: int, width: int, blocks: int, out_ch: int):
        super().__init__()
        self.inp = nn.Conv2d(in_ch, width, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.out = nn.Conv2d(width, out_ch * 16, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, lat):
        return F.pixel_shuffle(self.out(F.silu(self.blocks(self.inp(lat)))), 4)


class Refiner(nn.Module):
    def __init__(self, base: int = 48, latent: bool | int = False, grid: int = 0, grid_blocks: int = 4):
        super().__init__()
        b = base
        self.latent_ch = 0 if not latent else (24 if latent is True else int(latent))
        self.latent = self.latent_ch > 0
        self.inp = nn.Conv2d(9 + self.latent_ch, b, 3, padding=1)
        self.e1 = Block(b)
        self.d1 = nn.Conv2d(b, 2 * b, 4, stride=2, padding=1)
        self.e2 = Block(2 * b)
        self.d2 = nn.Conv2d(2 * b, 4 * b, 4, stride=2, padding=1)
        self.mid = nn.Sequential(Block(4 * b), Block(4 * b))
        self.grid_width, self.grid_blocks = (grid, grid_blocks) if grid else (0, 0)
        if grid and not self.latent:
            raise ValueError("the latent-grid path needs a latent input")
        self.grid = LatentPath(self.latent_ch, grid, grid_blocks, 4 * b) if grid else None
        self.u2 = nn.ConvTranspose2d(4 * b, 2 * b, 4, stride=2, padding=1)
        self.m2 = Block(2 * b)
        self.u1 = nn.ConvTranspose2d(2 * b, b, 4, stride=2, padding=1)
        self.m1 = Block(b)
        self.out = nn.Conv2d(b, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, lat=None):
        inp = x if not self.latent else torch.cat([x, F.interpolate(lat.to(x.dtype), scale_factor=16, mode="nearest")], 1)
        h1 = self.e1(self.inp(inp))
        h2 = self.e2(self.d1(h1))
        h = self.d2(h2)
        if self.grid is not None:
            h = h + self.grid(lat.to(x.dtype))
        h = self.mid(h)
        h = self.m2(self.u2(h) + h2)
        h = self.m1(self.u1(h) + h1)
        return x[:, 3:6] + self.out(h)


CONTEXT_CH = 3 * 24 + 4


def spec_of(saved: dict) -> dict:
    return {"base": saved.get("base", 48), "latent": saved.get("latent_spec", saved.get("latent", False)),
            "grid": saved.get("grid", 0), "grid_blocks": saved.get("grid_blocks", 4)}


def frame_groups(latent_t: int) -> list[int]:
    return [k for k, n in enumerate(frames_of(latent_t)) for _ in range(n)]


def frame_context(lat: torch.Tensor, ts: list[int]) -> torch.Tensor:
    """``lat`` ``(24, T, h, w)`` -> ``(len(ts), 76, h, w)``: latent frames before, at and after each pixel frame's
    own, and a one-hot of its place in that latent frame."""
    t_lat = lat.shape[1]
    group = frame_groups(t_lat)
    first = {}
    for i, g in enumerate(group):
        first.setdefault(g, i)
    out = []
    for t in ts:
        g = group[t]
        cur = lat[:, [max(g - 1, 0), g, min(g + 1, t_lat - 1)]].flatten(0, 1)
        phase = torch.zeros(4, *lat.shape[2:], dtype=lat.dtype)
        phase[t - first[g]] = 1
        out.append(torch.cat([cur, phase]))
    return torch.stack(out)


def neighbours(frames: torch.Tensor) -> torch.Tensor:
    prev = torch.cat([frames[:1], frames[:-1]])
    nxt = torch.cat([frames[1:], frames[-1:]])
    return torch.cat([prev, frames, nxt], dim=1)


@torch.no_grad()
def refine_clip(model: Refiner, clip: torch.Tensor, latent: torch.Tensor | None = None, chunk: int = 8) -> torch.Tensor:
    """``(1, 3, F, H, W)`` in [-1, 1] -> refined clip, same shape (``latent`` ``(1, 24, T, H/16, W/16)``)."""
    frames = clip[0].permute(1, 0, 2, 3)
    x = neighbours(frames)
    lat = None
    if model.latent_ch == CONTEXT_CH:
        lat = frame_context(latent[0].float().cpu(), list(range(frames.shape[0]))).to(frames.device)
    elif model.latent:
        lat = latent[0][:, frame_groups(latent.shape[2])].permute(1, 0, 2, 3).float()
    out = torch.cat([model(x[i:i + chunk], None if lat is None else lat[i:i + chunk])
                     for i in range(0, x.shape[0], chunk)])
    return out.clamp(-1, 1).permute(1, 0, 2, 3).unsqueeze(0)


def load_refiner(path) -> Refiner:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = Refiner(**spec_of(saved))
    model.load_state_dict(saved["model"])
    return model.eval().requires_grad_(False)
