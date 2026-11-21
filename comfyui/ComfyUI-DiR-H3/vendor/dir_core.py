# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""DiR inference pieces, vendored from dir-diffusion ``lot/dir_h3.py``, ``lot/h3.py`` and ``lot/h3_musubi.py``
(commit db1f6f3). Unchanged in behaviour; the musubi and LoT imports are dropped.

* ``DirHead``: the generator's new parameters, ``x0_proj`` (previous estimate -> hidden, added to the
  target-video rows) and ``r_embed`` (one timestep-embedding delta per iteration).
* ``parse_sigmas`` / ``next_state``: the per-iteration sigma schedule and the state update of ``dir_sample``.
"""

from __future__ import annotations

import torch
from torch import nn

TOKEN_DIM = 96
DEFAULT_R = 4
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def shift_sigma(sigma, shift: float):
    """``shift * sigma / (1 + (shift - 1) * sigma)`` (``lot/h3_musubi.py``)."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def unshift_sigma(sigma, shift: float):
    """Inverse of ``shift_sigma``."""
    return sigma / (shift + sigma * (1.0 - shift))


def audio_sigma(video_sigma):
    """The audio sigma at the same base-schedule position as ``video_sigma`` (shifts 12 / 3, as in training)."""
    return shift_sigma(unshift_sigma(video_sigma, VIDEO_SHIFT), AUDIO_SHIFT)


def audio_frames(latent_t: int) -> int:
    """40 Hz audio latents covering ``latent_t`` latent frames: H3's ``(10 f + 3) // 6`` for ``f`` pixel frames."""
    pixel = sum((1, 4, 4, 4, 4)[k % 5] for k in range(int(latent_t)))
    return (10 * pixel + 3) // 6


def patchify(latent: torch.Tensor) -> torch.Tensor:
    """``(B, C, T, H, W)`` -> ``(B, T, H/2, W/2, C*2*2)``, channel order ``(c, ph, pw)`` (``lot/h3.py``)."""
    batch, channels, time, height, width = latent.shape
    if height % 2 or width % 2:
        raise ValueError(f"latent {(time, height, width)} is not divisible by the (1, 2, 2) patch")
    tokens = latent.reshape(batch, channels, time, 1, height // 2, 2, width // 2, 2)
    tokens = tokens.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return tokens.reshape(batch, time, height // 2, width // 2, channels * 4)


class DirHead(nn.Module):
    """The generator's new parameters: previous-estimate projection and the iteration embedding."""

    def __init__(self, hidden: int, temb_dim: int, iterations: int = DEFAULT_R):
        super().__init__()
        self.iterations = int(iterations)
        self.x0_proj = nn.Linear(TOKEN_DIM, hidden)
        nn.init.zeros_(self.x0_proj.weight)
        nn.init.zeros_(self.x0_proj.bias)
        self.r_embed = nn.Parameter(torch.zeros(self.iterations, temb_dim))


def estimate_extra(head: DirHead, estimate: torch.Tensor) -> torch.Tensor:
    """``(N, hidden)`` embedding of the previous estimate ``(1, 24, T, H, W)`` (dense ``estimate_extra``)."""
    return head.x0_proj(patchify(estimate.float()).flatten(1, 3))[0]


def parse_sigmas(spec, iters: int) -> list[float]:
    """Per-iteration flow sigma: ``1`` (pinned), ``grid`` / ``grid:SHIFT``, ``turbo`` / ``turbo:SHIFT``, or ``R``
    comma-separated values in (0, 1] starting at 1 (or a list)."""
    if spec is None or str(spec) in ("1", "1.0", "pinned"):
        return [1.0] * iters
    if isinstance(spec, (list, tuple)):
        values = [float(v) for v in spec]
    elif str(spec).split(":", 1)[0] in ("grid", "turbo"):
        name, _, value = str(spec).partition(":")
        shift = float(value) if value else (6.0 if name == "turbo" else float(VIDEO_SHIFT))
        grid = torch.linspace(1.0, 0.0, iters + 1, dtype=torch.float64)[:iters]
        values = [float(shift_sigma(q, shift)) for q in grid]
    else:
        values = [float(v) for v in str(spec).split(",")]
    if len(values) != iters or values[0] != 1.0 or any(not 0.0 < v <= 1.0 for v in values) \
            or any(b > a for a, b in zip(values, values[1:])):
        raise ValueError(f"sigmas {spec!r}: need {iters} non-increasing values in (0, 1] starting at 1")
    return values


def next_state(state: torch.Tensor, clean: torch.Tensor, sigma: float, sigma_next: float,
               z: torch.Tensor | None = None) -> torch.Tensor:
    """The next iteration's state: a deterministic Euler step, or with ``z`` (``mix_state``) the newest
    estimate re-noised with the same ``z``."""
    if z is not None:
        return sigma_next * z + (1.0 - sigma_next) * clean.to(z.dtype)
    if sigma_next == sigma:
        return state
    ratio = sigma_next / sigma
    return ratio * state + (1.0 - ratio) * clean.to(state.dtype)


def dir_loop(step, z: torch.Tensor, iters: int, sigmas: list[float], mix_state: bool = False):
    """``dir_sample``'s recursion with the model call factored out.

    ``step(r, state, estimate, sigma)`` returns the raw head output ``out`` (``x0 - eps``, fp32, shape of
    ``z``); the clean estimate is ``state + sigma * out``. Returns ``(final estimate, every estimate,
    the state the final iteration read)``.
    """
    estimate = torch.zeros_like(z)
    state = z
    estimates = []
    for r in range(iters):
        out = step(r, state, estimate, sigmas[r])
        estimate = state.float() + sigmas[r] * out.float()
        estimates.append(estimate)
        if r + 1 < iters:
            state = next_state(state, estimate, sigmas[r], sigmas[r + 1], z if mix_state else None)
    return estimate, estimates, state
