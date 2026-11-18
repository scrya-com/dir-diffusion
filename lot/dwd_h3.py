"""Decoding without a decoder (DWD) for DiR on H3: a linear pixel head on the DiT's last-layer tokens.

After "Decoding Without a Decoder" (Chen et al.): a linear head on a diffusion transformer's last-layer
tokens predicts pixels directly, so inference skips the VAE decoder. Here the head reads the final DiR
iteration of an H3 generator.

Geometry. H3 packs a ``(1, 24, T, H, W)`` latent into ``T * (H/2) * (W/2)`` rows in ``(T, H/2, W/2)``
raster order (``VIDEO_PATCH_SIZE = (1, 2, 2)``). One latent pixel is 16×16 video pixels, so one token
covers 32×32 pixels. In time, latent frame ``k`` covers ``FRAME_PER_TOKEN[k % 5]`` pixel frames: 1 for
``k % 5 == 0`` (each 17-frame VAE clip opens on one frame), 4 otherwise. A token's target is therefore
``4 frame slots × 3 × 32 × 32 = 12288`` values. A one-frame token uses slot 0 only. For fitting, its
slots 1-3 repeat slot 0, so a plain least-squares fit needs no mask, and decoding drops them.

Inputs. The head reads ``[h, s]``:
- ``h`` (``HIDDEN`` wide) is the modulated last-layer state that ``final_layer.video_out`` reads, caught
  with a forward hook. The dense forward only: LoT never calls ``video_out``.
- ``s`` (``TOKEN_DIM`` = 96 wide) is the iteration's input state, packed the same way.
DiR's clean estimate is ``state + sigma * video_out(h)``, so the latent ``x0_hat`` is exactly linear in
``[h, s]``. With a last iteration at sigma 0.667, a third of ``x0_hat`` is the input state; a head on
``h`` alone would have to recover it from the residual stream.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

HIDDEN = 5376
TOKEN_DIM = 96
PATCH = 32          # pixels per token side (2 latent pixels × 16)
SLOTS = 4           # pixel frames per token, at most
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
OUT = SLOTS * 3 * PATCH * PATCH


def frames_of(latent_t: int) -> list[int]:
    """Pixel frames covered by each latent frame: ``[1, 4, 4, 4, 4, 1, 4, ...]``."""
    return [FRAME_PER_TOKEN[k % 5] for k in range(latent_t)]


def pixel_frame_count(latent_t: int) -> int:
    return sum(frames_of(latent_t))


def pack_rows(latent: torch.Tensor) -> torch.Tensor:
    """``(1, 24, T, H, W)`` -> ``(T*(H/2)*(W/2), 96)``, the order ``pack_video_rows`` uses."""
    _b, c, t, h, w = latent.shape
    rows = latent[0].reshape(c, t, h // 2, 2, w // 2, 2).permute(1, 2, 4, 0, 3, 5)
    return rows.reshape(t * (h // 2) * (w // 2), c * 4)


def pixels_to_tokens(pixels: torch.Tensor, latent_t: int) -> torch.Tensor:
    """``(1, 3, F, 16H, 16W)`` pixels -> ``(N, 12288)`` per-token targets, ``N = T * (H/2) * (W/2)``.

    Slot order inside a target is ``(slot, channel, y, x)``. One-frame tokens repeat their frame in
    slots 1-3.
    """
    _b, c, f, ph, pw = pixels.shape
    if f != pixel_frame_count(latent_t):
        raise ValueError(f"{f} pixel frames, expected {pixel_frame_count(latent_t)} for {latent_t} latent frames")
    if ph % PATCH or pw % PATCH:
        raise ValueError(f"pixel size {ph}x{pw} is not a multiple of {PATCH}")
    th, tw = ph // PATCH, pw // PATCH
    chunks, start = [], 0
    for n in frames_of(latent_t):
        frames = pixels[0, :, start:start + n]                          # (3, n, ph, pw)
        if n < SLOTS:
            frames = torch.cat([frames, frames[:, :1].expand(-1, SLOTS - n, -1, -1)], dim=1)
        start += n
        tiles = frames.reshape(c, SLOTS, th, PATCH, tw, PATCH).permute(2, 4, 1, 0, 3, 5)
        chunks.append(tiles.reshape(th * tw, OUT))
    return torch.cat(chunks, dim=0)


def tokens_to_pixels(tokens: torch.Tensor, latent_t: int, token_h: int, token_w: int) -> torch.Tensor:
    """Inverse of ``pixels_to_tokens``: ``(N, 12288)`` -> ``(1, 3, F, 32*token_h, 32*token_w)``."""
    per = token_h * token_w
    if tokens.shape[0] != latent_t * per:
        raise ValueError(f"{tokens.shape[0]} tokens, expected {latent_t * per}")
    frames = []
    for k, n in enumerate(frames_of(latent_t)):
        tiles = tokens[k * per:(k + 1) * per].reshape(token_h, token_w, SLOTS, 3, PATCH, PATCH)
        clip = tiles.permute(3, 2, 0, 4, 1, 5).reshape(3, SLOTS, token_h * PATCH, token_w * PATCH)
        frames.append(clip[:, :n])
    return torch.cat(frames, dim=1).unsqueeze(0)


class DWDHead(nn.Module):
    """``pixels = W [h, s] + b``, centred on the training features' mean (kept as a buffer)."""

    def __init__(self, hidden: int = HIDDEN, state_dim: int = TOKEN_DIM, out: int = OUT):
        super().__init__()
        self.inputs = hidden + state_dim
        self.linear = nn.Linear(self.inputs, out)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.register_buffer("mean", torch.zeros(self.inputs))

    def features(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return torch.cat([h.float(), s.float()], dim=-1) - self.mean

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.linear(self.features(h, s))

    @torch.no_grad()
    def decode(self, h: torch.Tensor, s: torch.Tensor, latent_t: int, token_h: int, token_w: int,
               chunk: int = 4096) -> torch.Tensor:
        """Pixels in [-1, 1], ``(1, 3, F, 32*token_h, 32*token_w)``."""
        out = torch.cat([self(h[i:i + chunk], s[i:i + chunk]) for i in range(0, h.shape[0], chunk)])
        return tokens_to_pixels(out, latent_t, token_h, token_w).clamp(-1, 1)


@contextlib.contextmanager
def capture_last_layer(dit, store: dict, key: str = "h"):
    """Catch the input of ``dit.final_layer.video_out``, the modulated last-layer tokens ``(1, N, HIDDEN)``.

    Dense forward only. ``store[key]`` holds the most recent call's tokens (the batch row dropped).
    """
    def hook(_module, args):
        store[key] = args[0][0].detach()

    handle = dit.final_layer.video_out.register_forward_pre_hook(hook)
    try:
        yield store
    finally:
        handle.remove()


def pseudo_huber(pred: torch.Tensor, target: torch.Tensor, c: float | None = None) -> torch.Tensor:
    """Per-token pseudo-Huber, ``sqrt(|d|^2 + c^2) - c``, averaged over tokens (iCT / TDM).

    ``c`` defaults to ``0.00054 * sqrt(d)`` for ``d`` values per token, the iCT setting. Its gradient is
    close to ``d / |d|`` for large errors, so no token dominates the update.
    """
    d = pred.shape[-1]
    c = 0.00054 * d ** 0.5 if c is None else c
    return (((pred - target) ** 2).sum(-1) + c * c).sqrt().sub(c).mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR in dB of two [-1, 1] tensors (peak-to-peak 2)."""
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(4.0 / mse)).item()


def flicker(pixels: torch.Tensor) -> float:
    """Mean |frame t+1 - frame t| of a ``(1, 3, F, H, W)`` clip in [-1, 1] (0 for a still)."""
    if pixels.shape[2] < 2:
        return 0.0
    return (pixels[:, :, 1:].float() - pixels[:, :, :-1].float()).abs().mean().item()
