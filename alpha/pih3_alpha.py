# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""An alpha channel on pi-H3's DWD head, after AlphaAnything (arXiv:2610.32413).

pi-H3 (and DiR + DWD) never runs the H3 VAE decoder: at the last sampling step the DiT's block output goes
through ``dwd_norm_out`` and the Sol-H3 pixel refiner (8 ViT decoder blocks over 16-pixel tokens, one per latent
frame) straight to RGB. So the alpha branch hangs off that refiner, one way, the same as ``h3_alpha`` does on the
VAE decoder:

* **taps:** the refiner's final token states, the input of its ``norm_out``, caught with a pre-hook while the
  refiner runs untouched. RGB is bit-identical with and without the branch (``test_pih3_alpha``).
* **token head:** LayerNorm + projection + a few transformer blocks, then per-token alpha patches laid out
  exactly like the refiner's RGB: the first latent of every 5-latent block gives 1 frame, the others 4.
* **pixel refiner:** a small dilated conv over [RGB, coarse score] that reaches across token borders.

``ea_score_to_alpha`` turns the score into alpha.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .h3_alpha import _Block

PATCH, PATCH_T, FIRST_T = 16, 4, 1  # Sol tokens: 16 px; 4 frames, 1 for a block's first latent (4 - 3 trimmed)


@contextlib.contextmanager
def capture_refiner_tokens(refiner):
    """Collect the Sol refiner's final token states: one ``(n * H * W, D)`` tensor per 5-latent block, in order."""
    chunks: list[torch.Tensor] = []

    def hook(_module, args):
        chunks.append(args[0])

    handle = refiner.norm_out.register_forward_pre_hook(hook)
    try:
        yield chunks
    finally:
        handle.remove()


class AlphaDWDHead(nn.Module):
    """Sol refiner tokens -> per-token alpha patches -> pixel refiner. ~6M parameters at the defaults."""

    def __init__(self, sol_dim: int = 2048, dim: int = 384, heads: int = 6, blocks: int = 3, refine_ch: int = 48):
        super().__init__()
        self.norm_in = nn.LayerNorm(sol_dim, elementwise_affine=False)
        self.proj_in = nn.Linear(sol_dim, dim)
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(blocks)])
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, PATCH_T * PATCH * PATCH)
        self.first_proj_out = nn.Linear(dim, FIRST_T * PATCH * PATCH)
        for p in (self.proj_out, self.first_proj_out):
            nn.init.normal_(p.weight, std=1e-4)
            nn.init.zeros_(p.bias)
        c = refine_ch
        self.refine = nn.Sequential(
            nn.Conv2d(4, c, 3, padding=1), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=2, dilation=2), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=4, dilation=4), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=8, dilation=8), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.SiLU(),
            nn.Conv2d(c, 1, 3, padding=1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def coarse(self, chunks: list[torch.Tensor]) -> torch.Tensor:
        """Per-block ``(n, H, W, D)`` token states -> ``(F, 1, 16H, 16W)`` coarse alpha scores."""
        out = []
        for y in chunks:
            n, h, w, _d = y.shape
            x = self.proj_in(self.norm_in(y.float())).reshape(1, n * h * w, -1)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm_out(x).view(n, h, w, -1)
            first = self.first_proj_out(x[:1]).view(h, w, FIRST_T, PATCH, PATCH).permute(2, 0, 3, 1, 4)
            out.append(first.reshape(FIRST_T, 1, h * PATCH, w * PATCH))
            if n > 1:
                rest = self.proj_out(x[1:]).view(n - 1, h, w, PATCH_T, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5)
                out.append(rest.reshape((n - 1) * PATCH_T, 1, h * PATCH, w * PATCH))
        return torch.cat(out)

    def forward(self, chunks: list[torch.Tensor], rgb: torch.Tensor) -> torch.Tensor:
        """``rgb`` ``(F, 3, H, W)`` in [-1, 1] (the refiner's own output) -> ``(F, 1, H, W)`` alpha score."""
        c = self.coarse(chunks)
        return c + self.refine(torch.cat([rgb.float(), c], 1))


def rgba_decode(head_pih3, alpha_head: AlphaDWDHead, h: torch.Tensor, t_emb: torch.Tensor, grid) -> tuple:
    """pi-H3's DWD decode plus alpha: ``(1, 3, F, H, W)`` RGB in [-1, 1] (exactly ``PiH3Head.decode``) and the
    ``(1, 1, F, H, W)`` alpha score. Only the alpha head takes gradients."""
    with torch.no_grad(), capture_refiner_tokens(head_pih3.refiner) as chunks:
        rgb = head_pih3.decode(h, t_emb, grid)                       # (1, 3, F, H, W)
    sol_h, sol_w = grid[1] * 2, grid[2] * 2                         # a DiT token is 2x2 Sol tokens
    chunks = [c.detach().view(-1, sol_h, sol_w, c.shape[-1]) for c in chunks]
    score = alpha_head(chunks, rgb[0].transpose(0, 1))                # (F, 1, H, W)
    return rgb, score.transpose(0, 1)[None]
