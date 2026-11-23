# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""An alpha (opacity) channel for the MiniMax H3 video VAE, after AlphaAnything (arXiv:2610.32413).

AlphaAnything hangs an alpha branch off a frozen conv U-Net VAE decoder. H3's video decoder is a ViT (36 blocks of
2048-d tokens, one token per 4 frames x 16 x 16 pixels, then a linear ``proj_out`` to pixels), so the branch here
is shaped for that:

* **taps:** hidden states of a few decoder blocks, read under ``no_grad``. The RGB path is untouched, so RGB is
  bit-identical with and without the branch (``tests``: ``test_rgb_unchanged``).
* **token head:** per-tap LayerNorm + low-rank projection, summed, a few small transformer blocks, then a linear
  to 4 x 16 x 16 alpha scores per token, unpatchified exactly like RGB.
* **pixel refiner:** a small 2-D conv over [decoded RGB, coarse score] that sharpens edges (hair, smoke wisps).

The score rides along as a 4th channel through the VAE's own spatial tiling, temporal chunking and blending
(``install``), so any clip length or size that decodes in RGB decodes in RGBA. ``ea_score_to_alpha`` turns the
score into alpha.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TAPS = (11, 23, 35)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.ff[2].weight)
        nn.init.zeros_(self.ff[2].bias)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))
        return x + self.ff(self.norm2(x))


class AlphaHead(nn.Module):
    """Decoder taps -> per-token alpha patches -> pixel refiner. ~10M parameters at the defaults."""

    def __init__(self, dec_dim: int = 2048, taps=DEFAULT_TAPS, dim: int = 384, heads: int = 6, blocks: int = 4,
                 patch: int = 16, patch_t: int = 4, refine_ch: int = 32):
        super().__init__()
        self.taps = tuple(int(t) for t in taps)
        self.patch, self.patch_t = patch, patch_t
        self.tap_norm = nn.ModuleList([nn.LayerNorm(dec_dim, elementwise_affine=False) for _ in self.taps])
        self.tap_proj = nn.ModuleList([nn.Linear(dec_dim, dim) for _ in self.taps])
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(blocks)])
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, patch_t * patch * patch)
        nn.init.normal_(self.proj_out.weight, std=1e-4)
        nn.init.zeros_(self.proj_out.bias)
        self.refine = nn.Sequential(
            nn.Conv2d(4, refine_ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(refine_ch, refine_ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(refine_ch, refine_ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(refine_ch, 1, 3, padding=1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def tokens(self, taps: list[torch.Tensor]) -> torch.Tensor:
        """``taps``: (B, N, dec_dim) each -> (B, N, patch_t * patch * patch) coarse scores."""
        x = sum(p(n(t.float())) for t, n, p in zip(taps, self.tap_norm, self.tap_proj))
        for blk in self.blocks:
            x = blk(x)
        return self.proj_out(self.norm_out(x))

    def pixels(self, coarse: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """``coarse`` (B, 1, T, H, W) score, ``rgb`` (B, 3, T, H, W) raw decoder pixels -> refined score."""
        b, _, t, h, w = coarse.shape
        flat = torch.cat([rgb.float(), coarse], 1).transpose(1, 2).reshape(b * t, 4, h, w)
        res = self.refine(flat).reshape(b, t, 1, h, w).transpose(1, 2)
        return coarse + res


def _unpatchify(output, b, t, h, w, c, pt, p):
    output = output.view(b, t, h, w, c, pt, p, p)
    return output.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(b, c, t * pt, h * p, w * p)


def rgba_forward(decoder, head: AlphaHead, x: torch.Tensor) -> torch.Tensor:
    """``ViT3DDecoder.forward`` plus the alpha head: (B, 24, T, H, W) -> (B, 4, 4T, 16H, 16W), channel 3 = score.
    The decoder trunk runs without gradients; only the head is trained."""
    from comfy.ldm.minimax import vae as h3vae
    import comfy.ops

    b, _c, lt, lh, lw = x.shape
    taps = []
    want = set(head.taps)
    with torch.no_grad():
        h = decoder.x_embedder(x.flatten(2).transpose(1, 2))
        n = h.shape[1]
        h = torch.cat([h, comfy.ops.cast_to_input(decoder.register_tokens, h).expand(b, -1, -1),
                       torch.zeros_like(h[:, 0:1, :])], dim=1)
        ids = decoder._token_ids(lt, lh, lw, x.device, x.dtype).expand(b, -1, -1)
        ids = torch.cat([ids, torch.zeros((b, 1 + decoder.num_register_tokens, decoder.pos_embed.n_dim),
                                          device=x.device, dtype=ids.dtype)], dim=1)
        rope = decoder.pos_embed(ids)
        for i, block in enumerate(decoder.transformer_blocks):
            h = block(h, rope)
            if i in want:
                taps.append(h[:, :n].detach())
        rgb = decoder.proj_out(decoder.norm_out(h))
        rgb = h3vae.unpatchify(rgb[:, :n], b, lt, lh, lw, decoder.out_channels, decoder.patch_size_t, decoder.patch_size)
    coarse = _unpatchify(head.tokens(taps), b, lt, lh, lw, 1, head.patch_t, head.patch)
    score = head.pixels(coarse, rgb)
    return torch.cat([rgb.to(score.dtype), score], 1)


def install(vae_model, head: AlphaHead):
    """Make ``MiniMaxH3VideoVAE.decode`` return (B, 4, T, H, W): RGB in [0, 1] plus the raw alpha score.
    Returns an ``uninstall`` callable. Pass a 4-channel ``output_buffer`` on the GPU to keep gradients cheap."""
    dec = vae_model.decoder
    orig_forward, orig_finalize, orig_shape = dec.forward, vae_model._finalize_pixels, vae_model.decode_output_shape

    def forward(self, x):
        return rgba_forward(self, head, x)

    def finalize(self, part):
        return torch.cat([orig_finalize(part[:, :3]), part[:, 3:].float()], 1)

    def out_shape(self, input_shape):
        shape = list(orig_shape(input_shape))
        shape[1] = 4
        return tuple(shape)

    dec.forward = types.MethodType(forward, dec)
    vae_model._finalize_pixels = types.MethodType(finalize, vae_model)
    vae_model.decode_output_shape = types.MethodType(out_shape, vae_model)

    def uninstall():
        dec.forward, vae_model._finalize_pixels, vae_model.decode_output_shape = orig_forward, orig_finalize, orig_shape
        for name in ("forward",):
            dec.__dict__.pop(name, None)
        for name in ("_finalize_pixels", "decode_output_shape"):
            vae_model.__dict__.pop(name, None)

    return uninstall
