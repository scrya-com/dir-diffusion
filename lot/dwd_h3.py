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


def unpack_rows(rows: torch.Tensor, latent_t: int, token_h: int, token_w: int) -> torch.Tensor:
    """Inverse of ``pack_rows``: ``(T*th*tw, 96)`` -> ``(1, 24, T, 2*th, 2*tw)``."""
    c = rows.shape[-1] // 4
    x = rows.reshape(latent_t, token_h, token_w, c, 2, 2).permute(3, 0, 1, 4, 2, 5)
    return x.reshape(1, c, latent_t, 2 * token_h, 2 * token_w)


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


CLASSES = 5        # one head per latent frame position k % 5 (k % 5 == 0 covers one pixel frame)


def token_classes(latent_t: int, token_h: int, token_w: int, device=None) -> torch.Tensor:
    """``(N,)`` class of each token: its latent frame's ``k % 5``, in packed row order."""
    return (torch.arange(latent_t, device=device) % CLASSES).repeat_interleave(token_h * token_w)


def slot_mask(classes: torch.Tensor) -> torch.Tensor:
    """``(N, OUT)`` 1 where a target value is real; one-frame tokens (class 0) only have slot 0."""
    mask = torch.ones(classes.shape[0], OUT, device=classes.device)
    mask[classes == 0, OUT // SLOTS:] = 0
    return mask


# ------------------------------------------------------------------------ overlapping patches
# A per-token head has no say over its neighbours, so its errors change abruptly at every 32 px token edge
# and at every latent-frame boundary (measured: the error step is mostly detail finer than 8 px, not a bias).
# An "extended" head also predicts a halo of ``halo`` pixels around its token and the pixel frame before and
# after its own; decoding cross-fades the overlaps, which turns each jump into a ramp.

ESLOTS = SLOTS + 2      # [the frame before, own slot 0..3, the frame after]


def ext_out(halo: int) -> int:
    return ESLOTS * 3 * (PATCH + 2 * halo) ** 2


def ext_halo(out: int) -> int | None:
    """The halo of an extended head with ``out`` outputs; ``None`` for a plain head."""
    side = round((out / (ESLOTS * 3)) ** 0.5)
    if out == OUT or side < PATCH or (side - PATCH) % 2 or ESLOTS * 3 * side * side != out:
        return None
    return (side - PATCH) // 2


def pixels_to_tokens_ext(pixels: torch.Tensor, latent_t: int, halo: int) -> torch.Tensor:
    """``(1, 3, F, H, W)`` pixels -> ``(N, ext_out(halo))`` targets, slot order ``(slot, channel, y, x)``.

    The picture is edge-replicated in space and time. One-frame tokens repeat their frame in own slots 1-3.
    """
    x = pixels[0]
    f, side = x.shape[1], PATCH + 2 * halo
    if f != pixel_frame_count(latent_t):
        raise ValueError(f"{f} pixel frames, expected {pixel_frame_count(latent_t)} for {latent_t} latent frames")
    if halo:
        x = nn.functional.pad(x, (halo, halo, halo, halo), mode="replicate")
    chunks, start = [], 0
    for n in frames_of(latent_t):
        index = [max(start - 1, 0)] + [start + min(i, n - 1) for i in range(SLOTS)] + [min(start + n, f - 1)]
        tiles = x[:, index].unfold(2, side, PATCH).unfold(3, side, PATCH)      # (3, ESLOTS, th, tw, side, side)
        chunks.append(tiles.permute(2, 3, 1, 0, 4, 5).reshape(tiles.shape[2] * tiles.shape[3], -1))
        start += n
    return torch.cat(chunks, dim=0)


def ext_slot_mask(classes: torch.Tensor, per_frame: int, halo: int) -> torch.Tensor:
    """``(N, ext_out(halo))`` 1 where an extended target is real: not the repeats of a one-frame token, and
    not the frame before the clip's first latent frame or after its last."""
    mask = torch.ones(classes.shape[0], ESLOTS, device=classes.device)
    mask[classes == 0, 2:1 + SLOTS] = 0
    mask[:per_frame, 0] = 0
    mask[-per_frame:, ESLOTS - 1] = 0
    return mask.repeat_interleave(3 * (PATCH + 2 * halo) ** 2, dim=1)


def blend_tokens(tokens: torch.Tensor, latent_t: int, token_h: int, token_w: int, halo: int,
                 blend: float = 0.25) -> torch.Tensor:
    """Overlap-add decode of extended tokens -> ``(1, 3, F, 32*token_h, 32*token_w)``.

    In space each token's ``32 + 2*halo`` patch is weighted by a window that ramps linearly over the
    ``2*halo`` pixels it shares with a neighbour. In time a boundary frame is ``1 - blend`` its own token and
    ``blend`` the neighbouring latent frame's prediction of it (``blend=0``: no temporal blending).
    """
    side, per, dev = PATCH + 2 * halo, token_h * token_w, tokens.device
    height, width = token_h * PATCH + 2 * halo, token_w * PATCH + 2 * halo
    ramp = (torch.arange(2 * halo, device=dev) + 0.5) / max(2 * halo, 1)
    window = torch.cat([ramp, torch.ones(side - 4 * halo, device=dev), ramp.flip(0)])
    window = torch.outer(window, window)

    def fold(values: torch.Tensor) -> torch.Tensor:                           # (B, C*side*side, per)
        return nn.functional.fold(values, (height, width), kernel_size=side, stride=PATCH)

    cover = fold(window.reshape(1, -1, 1).expand(1, -1, per))
    counts = frames_of(latent_t)
    f = sum(counts)
    out = torch.zeros(f, 3, height, width, device=dev)
    weight = torch.zeros(f, device=dev)
    start = 0
    for k, n in enumerate(counts):
        tiles = tokens[k * per:(k + 1) * per].reshape(per, ESLOTS, 3, side, side).float() * window
        frames = fold(tiles.permute(1, 2, 3, 4, 0).reshape(ESLOTS, 3 * side * side, per))   # (ESLOTS, 3, H, W)
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
    """``pixels = W_c ((x - mean) / scale) + b_c`` for ``x = [h, s]`` and ``c`` the token's class.

    Features are standardised per dimension (the buffers ``mean`` and ``scale``): the last-layer state's
    dimensions differ in spread by about three orders of magnitude, so an unscaled ridge penalty shrinks most
    of them. ``classes=1`` is a single shared map.

    With ``out = ext_out(halo)`` the head is extended (``halo`` is not ``None``): it predicts overlapping
    patches and ``decode`` cross-fades them (``blend`` is the temporal weight, see ``blend_tokens``).
    """

    def __init__(self, hidden: int = HIDDEN, state_dim: int = TOKEN_DIM, out: int = OUT, classes: int = CLASSES):
        super().__init__()
        self.inputs, self.classes = hidden + state_dim, classes
        self.halo, self.blend = ext_halo(out), 0.25
        self.weight = nn.Parameter(torch.zeros(classes, out, self.inputs))
        self.bias = nn.Parameter(torch.zeros(classes, out))
        self.register_buffer("mean", torch.zeros(self.inputs))
        self.register_buffer("scale", torch.ones(self.inputs))

    def features(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return (torch.cat([h.float(), s.float()], dim=-1) - self.mean) / self.scale

    def half_weights(self) -> "DWDHead":
        """Keep the map in fp16 (half the memory, about 3x faster); the feature statistics stay fp32."""
        self.weight.data = self.weight.data.half()
        self.bias.data = self.bias.data.half()
        return self

    def forward(self, h: torch.Tensor, s: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        x = self.features(h, s).to(self.weight.dtype)
        if self.classes == 1:
            return x @ self.weight[0].T + self.bias[0]
        out = x.new_empty(x.shape[0], self.weight.shape[1])
        for c in range(self.classes):
            index = (classes % self.classes) == c
            if index.any():
                out[index] = x[index] @ self.weight[c].T + self.bias[c]
        return out

    @torch.no_grad()
    def decode(self, h: torch.Tensor, s: torch.Tensor, latent_t: int, token_h: int, token_w: int) -> torch.Tensor:
        """Pixels in [-1, 1], ``(1, 3, F, 32*token_h, 32*token_w)``."""
        out = self(h, s, token_classes(latent_t, token_h, token_w, h.device))
        if self.halo is not None:
            return blend_tokens(out, latent_t, token_h, token_w, self.halo, self.blend).clamp(-1, 1)
        return tokens_to_pixels(out, latent_t, token_h, token_w).clamp(-1, 1)

    def targets(self, y: torch.Tensor, latent_t: int, token_h: int, token_w: int) -> torch.Tensor:
        """Per-token targets for this head from the plain ``(N, OUT)`` ones."""
        if self.halo is None:
            return y
        return pixels_to_tokens_ext(tokens_to_pixels(y, latent_t, token_h, token_w), latent_t, self.halo)

    def target_mask(self, classes: torch.Tensor, per_frame: int) -> torch.Tensor:
        return slot_mask(classes) if self.halo is None else ext_slot_mask(classes, per_frame, self.halo)

    @torch.no_grad()
    def masked_mse_grad(self, h: torch.Tensor, s: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                        classes: torch.Tensor, chunk: int = 1024) -> tuple[float, torch.Tensor]:
        """Masked MSE of a frozen head and its gradient with respect to ``h``, in chunks and without autograd,
        so the ``(N, out)`` prediction and its graph are never held (an extended head would not fit)."""
        z = self.features(h, s)
        grad = torch.zeros_like(z)
        count = mask.sum()
        total = torch.zeros((), device=z.device)
        for c in range(self.classes):
            rows = ((classes % self.classes) == c).nonzero()[:, 0]
            for i in range(0, rows.shape[0], chunk):
                r = rows[i:i + chunk]
                err = (z[r] @ self.weight[c].T + self.bias[c] - y[r]) * mask[r]
                total += (err * err).sum()
                grad[r] = (2.0 / count) * (err @ self.weight[c])
        width = h.shape[-1]
        return (total / count).item(), grad[:, :width] / self.scale[:width]


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


def flicker_split(pixels: torch.Tensor, latent_t: int) -> tuple[float, float]:
    """Mean |frame t+1 - frame t| for pairs inside one latent frame, and for pairs that cross into the next.

    A per-token head can be smooth inside a latent frame and jump between them; the overall mean hides that.
    """
    diff = (pixels[:, :, 1:].float() - pixels[:, :, :-1].float()).abs().mean(dim=(0, 1, 3, 4))
    group = torch.tensor([k for k, n in enumerate(frames_of(latent_t)) for _ in range(n)])
    across = group[1:] != group[:-1]
    inside = diff[~across.to(diff.device)]
    cross = diff[across.to(diff.device)]
    return (inside.mean().item() if inside.numel() else 0.0, cross.mean().item() if cross.numel() else 0.0)


def seam_ratio(pixels: torch.Tensor) -> float:
    """Mean |jump| across 32-pixel token edges over the mean |jump| between other neighbouring pixels.

    About 1 for an image with no grid; a per-token decoder with visible seams scores well above 1.
    """
    x = pixels.float()
    dx = (x[..., 1:] - x[..., :-1]).abs()
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs()
    ex = torch.zeros(dx.shape[-1], dtype=torch.bool, device=x.device)
    ex[PATCH - 1::PATCH] = True
    ey = torch.zeros(dy.shape[-2], dtype=torch.bool, device=x.device)
    ey[PATCH - 1::PATCH] = True
    edge = torch.cat([dx[..., ex].flatten(), dy[..., ey, :].flatten()]).mean()
    inner = torch.cat([dx[..., ~ex].flatten(), dy[..., ~ey, :].flatten()]).mean()
    return (edge / inner.clamp_min(1e-12)).item()


def flicker(pixels: torch.Tensor) -> float:
    """Mean |frame t+1 - frame t| of a ``(1, 3, F, H, W)`` clip in [-1, 1] (0 for a still)."""
    if pixels.shape[2] < 2:
        return 0.0
    return (pixels[:, :, 1:].float() - pixels[:, :, :-1].float()).abs().mean().item()
