#!/usr/bin/env python3
"""Posterior Matching Distillation (PMD) for MiniMax H3, on one 32 GB card.

Paper: Hidir Yesiltepe, Julius Berner, Arash Vahdat and Pinar Yanardag, "Posterior Matching Distillation",
arXiv 2611.26629 (official code not released). docs/PMD.md lists what is from the paper and what we guessed.

    # CPU: the maths and the H3 wiring on a toy model
    python3 lot/test_pmd_h3.py

    # GPU: one update with gradients, grad sums, peak memory, nothing written
    LOT_H3_TRAIN=1 python3 lot/pmd_h3.py train --check --init-run johndpope/dir-diffusion-h3 --out lot/runs/pmd_check

    # GPU: the first trial (512x288, 12 latent frames, 300 updates or 120 minutes, resumes from OUT/pmd_last.pt)
    LOT_H3_TRAIN=1 python3 lot/pmd_h3.py train --init-run johndpope/dir-diffusion-h3 --out lot/runs/pmd_trial

    # any-step samples from the same student (4 steps and 2 steps), and the pass check (docs/PMD.md)
    python3 lot/pmd_h3.py probe --out lot/runs/probe_pmd --name pmd4 --pmd lot/runs/pmd_trial --levels 0,1,2,3
    python3 lot/pmd_h3.py probe --out lot/runs/probe_pmd --name pmd2 --pmd lot/runs/pmd_trial --levels 0,2
    python3 lot/pmd_h3.py score --out lot/runs/probe_pmd --names pmd2 pmd4 dir4 dir2 --pass pmd2 dir4 dir2

What one update does (the paper's section 4.4, Algorithm 1 and 2, appendix C):

1. Pick a level ``k`` of the 4-level grid other than the pure-noise one. Roll the student out from Gaussian noise
   to ``z_k`` without gradients, re-noising each clean prediction to the next level (CPS, ``eta``). Detach.
2. Draw ``K`` coarse tilts ``lambda`` (low-resolution Gaussian, upsampled, unit L2 norm, radius ~ U[0.25, 0.5]).
3. Teacher targets through Tilt=Shift: ``T_lambda = D(z_k + rho * lambda)``, one frozen-teacher call each, with
   ``rho = sigma^2 / (1 - sigma)`` (rectified flow: alpha = 1 - sigma). The teacher is the same DiT with every
   LoRA off and no noise input: base H3, guidance 1, one call.
4. ``M`` student draws ``x_m = G(z_k, u_m, k)`` without gradients, the same draws for every tilt.
5. Loss: Algorithm 2 (jackknife + cross-fit over balanced splits; ``--estimator plain`` is Algorithm 1's softmax
   mean). Its gradient w.r.t. each draw is taken on the cached draws; then each draw is replayed with gradients
   (same ``u``, same audio seed) and back-propagated with that gradient: one student graph alive at a time.

Stochastic student (appendix C.1): the student mixes ``z_k`` with fresh noise ``u`` at the level's noise dial,
feeds the mixed latent through the PRETRAINED patch embedding at its effective level, and feeds the raw ``z_k``
through a NEW zero-initialised projection (``NoiseInput``) added to the embedded rows, which equals widening the
input projection from 96 to 192 features with the new half zero. Step 0 is therefore the starting model applied
to the mixed latent; it equals the starting model exactly only where the dial is 0.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOKEN_DIM = 96
DEFAULT_DIALS = (0.0, 0.30, 0.45, 0.15)          # the paper's FLUX.2 table (RF, pure-noise level 0)
DEFAULT_RUN = "johndpope/dir-diffusion-h3"


# ============================================================================== the maths (model-agnostic)

def rho_of(sigma: float) -> float:
    """Tilt=Shift's ``rho_t = sigma_t^2 / alpha_t`` for H3's rectified flow ``z = (1 - sigma) x + sigma eps``:
    ``alpha = 1 - sigma`` so ``rho = sigma^2 / (1 - sigma)`` (infinite at pure noise: no PMD loss there)."""
    sigma = float(sigma)
    if not 0.0 < sigma < 1.0:
        raise ValueError(f"Tilt=Shift needs 0 < sigma < 1, got {sigma}")
    return sigma * sigma / (1.0 - sigma)


def mix_noise(z: torch.Tensor, u: torch.Tensor, sigma: float, dial: float) -> tuple[torch.Tensor, float]:
    """The stochastic student's mixed input ``(y, sigma_tilde)`` in RF coordinates (our reading of C.1).

    ``(1 - d) z + d u`` with ``z = (1 - sigma) x + sigma eps`` has signal ``a = (1 - d)(1 - sigma)`` and noise std
    ``s = sqrt((1 - d)^2 sigma^2 + d^2)``. Dividing by ``a + s`` puts it back on the RF path at
    ``sigma_tilde = s / (a + s)``. ``d = 0`` returns ``(z, sigma)`` unchanged."""
    d, sigma = float(dial), float(sigma)
    if d == 0.0:
        return z, sigma
    if not 0.0 < d < 1.0:
        raise ValueError(f"noise dial must be in [0, 1), got {d}")
    a = (1.0 - d) * (1.0 - sigma)
    s = math.sqrt((1.0 - d) ** 2 * sigma ** 2 + d ** 2)
    return ((1.0 - d) * z + d * u.to(z.dtype)) / (a + s), s / (a + s)


def coarse_shape(shape, factor) -> tuple[int, ...]:
    """``(C, T, H, W)`` latent -> ``(C, ceil(T/ft), ceil(H/fh), ceil(W/fw))``."""
    c, t, h, w = shape
    ft, fh, fw = factor
    return c, max(1, math.ceil(t / ft)), max(1, math.ceil(h / fh)), max(1, math.ceil(w / fw))


def sample_tilts(like: torch.Tensor, count: int, *, factor=(4, 8, 8), radius=(0.25, 0.5),
                 generator: torch.Generator | None = None) -> list[torch.Tensor]:
    """``count`` coarse tilt directions for a ``(1, C, T, H, W)`` latent (appendix D.1, Fig. 9).

    Independent Gaussians on a grid ``factor`` times coarser in (T, H, W) (channels independent), trilinearly
    upsampled to the latent, normalised to unit L2 norm over the whole latent, times a radius ~ U(radius).
    ``factor=(1, 1, 1)`` is the paper's isotropic variant."""
    if like.ndim != 5 or like.shape[0] != 1:
        raise ValueError(f"latent must be (1, C, T, H, W), got {tuple(like.shape)}")
    _b, c, t, h, w = like.shape
    small = coarse_shape((c, t, h, w), factor)
    device = like.device
    raw = torch.randn((count,) + small, device=device, generator=generator, dtype=torch.float32)
    if small[1:] != (t, h, w):
        raw = nn.functional.interpolate(raw, size=(t, h, w), mode="trilinear", align_corners=False)
    lo, hi = radius
    radii = lo + (hi - lo) * torch.rand(count, device=device, generator=generator)
    flat = raw.flatten(1)
    tilts = flat / flat.norm(dim=1, keepdim=True) * radii[:, None]
    return [tilt.view(1, c, t, h, w) for tilt in tilts]


def _logits(draws: torch.Tensor, tilt: torch.Tensor) -> torch.Tensor:
    return (draws.flatten(1).float() * tilt.flatten().float()).sum(1)


def softmax_mean(draws: torch.Tensor, tilt: torch.Tensor) -> torch.Tensor:
    """Algorithm 1's student tilted mean: ``sum_m softmax(lambda^T x_m) x_m`` over the ``M`` draws ``(M, ...)``."""
    w = torch.softmax(_logits(draws, tilt), dim=0)
    return (w.view(-1, *[1] * (draws.ndim - 1)) * draws.float()).sum(0)


def loss_plain(draws: torch.Tensor, tilt: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Algorithm 1: ``|| softmax mean - T_lambda ||^2`` (squared norm over the latent; batch of one)."""
    return ((softmax_mean(draws, tilt) - target.float().reshape(draws.shape[1:])) ** 2).sum()


def balanced_splits(m: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Every unordered partition of ``range(m)`` into groups of ``floor(m/2)`` and ``ceil(m/2)`` (3 for m = 4)."""
    if m < 2:
        raise ValueError("the cross-fit estimator needs at least 2 draws")
    size = m // 2
    seen, out = set(), []
    for a in itertools.combinations(range(m), size):
        b = tuple(i for i in range(m) if i not in a)
        key = frozenset((a, b))
        if key not in seen:
            seen.add(key)
            out.append((a, b))
    return out


def loss_jackknife(draws: torch.Tensor, tilt: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Algorithm 2: delete-one jackknife residual per group, cross-fitted over all balanced splits. Signed, not
    clamped. At zero tilt it is the unbiased U-statistic of ``||E[x] - T||^2``."""
    m = draws.shape[0]
    h = draws.float() - target.float().reshape(draws.shape[1:])
    logits = _logits(draws, tilt)

    def residual(idx):
        idx = list(idx)
        w = torch.softmax(logits[idx], dim=0)
        return (w.view(-1, *[1] * (h.ndim - 1)) * h[idx]).sum(0)

    def corrected(group):
        n = len(group)
        if n == 1:
            return residual(group)
        loo = torch.stack([residual([j for j in group if j != i]) for i in group]).mean(0)
        return n * residual(group) - (n - 1) * loo

    terms = [(corrected(a) * corrected(b)).sum() for a, b in balanced_splits(m)]
    return torch.stack(terms).mean()


ESTIMATORS = {"jk": loss_jackknife, "plain": loss_plain}


def pmd_loss(draws: torch.Tensor, tilts, targets, estimator: str = "jk") -> torch.Tensor:
    """Mean over the ``K`` tilts of the posterior-matching loss; the same ``M`` draws serve every tilt (C.1)."""
    fn = ESTIMATORS[estimator]
    return torch.stack([fn(draws, tilt, target) for tilt, target in zip(tilts, targets)]).mean()


def weight_stats(draws: torch.Tensor, tilts) -> dict:
    """Softmax health over the tilts: effective sample size (``1 / sum w^2``, M = flat, 1 = one-hot) and max weight."""
    ess, top = [], []
    for tilt in tilts:
        w = torch.softmax(_logits(draws, tilt), dim=0)
        ess.append(float(1.0 / (w * w).sum()))
        top.append(float(w.max()))
    return {"ess": sum(ess) / len(ess), "w_max": max(top)}


def renoise(clean: torch.Tensor, state: torch.Tensor, sigma: float, sigma_next: float, eta: float,
            noise: torch.Tensor) -> torch.Tensor:
    """Re-noise a clean prediction from level ``sigma`` (state ``state``) to ``sigma_next``.

    ``eta = 1``: fresh Gaussian noise (section 4.5, the SDXL setting). ``eta < 1``: a coefficient-preserving mix of
    the state's implied noise and fresh noise, ``cos(eta pi/2) eps_hat + sin(eta pi/2) eps`` (our reading of the
    CPS sampler the FLUX.2 setting uses at eta = 0.9)."""
    clean = clean.float()
    if eta >= 1.0:
        mixed = noise.float()
    else:
        eps_hat = (state.float() - (1.0 - sigma) * clean) / sigma
        angle = 0.5 * math.pi * float(eta)
        mixed = math.cos(angle) * eps_hat + math.sin(angle) * noise.float()
    return (1.0 - sigma_next) * clean + sigma_next * mixed


def rollout(student, z: torch.Tensor, sigmas, upto: int, *, eta: float, generator: torch.Generator) -> torch.Tensor:
    """On-policy state at level ``upto``: levels ``0 .. upto-1`` run without gradients from pure noise ``z``."""
    state = z
    with torch.no_grad():
        for k in range(upto):
            u = torch.randn(z.shape, device=z.device, generator=generator)
            clean = student(state, u, k)
            state = renoise(clean, state, sigmas[k], sigmas[k + 1], eta,
                            torch.randn(z.shape, device=z.device, generator=generator))
    return state.detach()


def any_step(student, z: torch.Tensor, sigmas, levels, *, eta: float, generator: torch.Generator) -> list[torch.Tensor]:
    """Section 4.5: sample with any subset of the trained levels (``levels``, increasing, starting at 0). Returns the
    clean prediction after each step; the last one is the sample (not re-noised)."""
    levels = list(levels)
    if not levels or levels[0] != 0 or any(b <= a for a, b in zip(levels, levels[1:])) or levels[-1] >= len(sigmas):
        raise ValueError(f"levels {levels}: increasing indices into the {len(sigmas)}-level grid, starting at 0")
    state, outs = z, []
    with torch.no_grad():
        for j, k in enumerate(levels):
            u = torch.randn(z.shape, device=z.device, generator=generator)
            clean = student(state, u, k).float()
            outs.append(clean)
            if j + 1 < len(levels):
                state = renoise(clean, state, sigmas[k], sigmas[levels[j + 1]], eta,
                                torch.randn(z.shape, device=z.device, generator=generator))
    return outs


def pmd_step(student, teacher, z: torch.Tensor, sigma: float, k: int, tilts, noises, *, estimator: str = "jk",
             backward: bool = True, extra_loss=None, scale: float = 1.0) -> tuple[torch.Tensor, dict]:
    """One PMD loss at a detached state ``z`` (level ``k``, noise ``sigma``), back-propagated by replay.

    ``student(z, u, k)`` -> clean prediction; ``teacher(z_shifted)`` -> the frozen teacher's clean prediction at
    the same level. ``noises`` are the ``M`` inputs ``u_m``. With ``backward``, the gradient of the loss w.r.t.
    each cached draw is computed first, then each draw is replayed with gradients and back-propagated with it.
    ``extra_loss(clean, m)`` (optional, e.g. the GAN term) is added to draw ``m``'s replay. ``scale`` multiplies
    what is back-propagated (``1 / accum`` for gradient accumulation); the returned loss is unscaled."""
    z = z.detach()
    rho = rho_of(sigma)
    with torch.no_grad():
        targets = [teacher(z + rho * tilt).float() for tilt in tilts]
        cached = torch.stack([student(z, u, k).float()[0] for u in noises])
    leaf = cached.clone().requires_grad_(True)
    with torch.enable_grad():
        loss = pmd_loss(leaf, [t[0] for t in tilts], [t[0] for t in targets], estimator)
        grads = torch.autograd.grad(loss, leaf)[0] if backward else None
    stats = weight_stats(cached, [t[0] for t in tilts])
    stats["loss_plain"] = float(pmd_loss(cached, [t[0] for t in tilts], [t[0] for t in targets], "plain"))
    stats["shift_ratio"] = float(rho * torch.stack([t.norm() for t in tilts]).mean() / z.norm().clamp_min(1e-12))
    stats["target_spread"] = float(torch.stack([(t - targets[0]).norm() for t in targets[1:]]).mean()
                                   / targets[0].norm().clamp_min(1e-12)) if len(targets) > 1 else 0.0
    stats["draw_spread"] = float((cached - cached.mean(0)).pow(2).sum(dim=tuple(range(1, cached.ndim))).mean().sqrt()
                                 / cached.mean(0).norm().clamp_min(1e-12))
    stats["extra"] = 0.0
    if backward:
        with torch.enable_grad():
            for m, u in enumerate(noises):
                out = student(z, u, k).float()
                total = (out[0] * grads[m]).sum()
                if extra_loss is not None:
                    extra = extra_loss(out, m)
                    stats["extra"] += float(extra.detach())
                    total = total + extra
                (scale * total).backward()
    return loss.detach(), stats


# ======================================================================== discriminator (GAN term, off by default)

class DiscHead(nn.Module):
    """A trainable head on frozen-teacher video-token features (appendix C: caption-conditioned midpoint features,
    real and generated clean latents noised independently). Mean-pooled logit; the head's shape is our guess."""

    def __init__(self, hidden: int, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).mean(dim=(-2, -1))


def d_loss(real_logit: torch.Tensor, fake_logit: torch.Tensor) -> torch.Tensor:
    """Non-saturating logistic discriminator loss."""
    return nn.functional.softplus(-real_logit).mean() + nn.functional.softplus(fake_logit).mean()


def g_loss(fake_logit: torch.Tensor) -> torch.Tensor:
    """Non-saturating logistic generator loss."""
    return nn.functional.softplus(-fake_logit).mean()


def gan_level(generator: torch.Generator, device) -> float:
    """An RF noise level ``k / 1000`` with ``k`` uniform in {1, ..., 999} (C.1)."""
    return int(torch.randint(1, 1000, (1,), generator=generator, device=device)) / 1000.0


# ============================================================================== H3 wiring

class NoiseInput(nn.Module):
    """The widened input projection's new half: a ``TOKEN_DIM -> hidden`` linear, no bias, zero-initialised, reading
    the RAW state's 96-wide patch tokens. Its output is added to the pretrained ``video_patch_proj`` of the mixed
    latent, which is exactly ``Linear(192 -> hidden)`` on ``[mixed, raw]`` with the raw half zero at init."""

    def __init__(self, hidden: int, in_dim: int = TOKEN_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        from h3 import patchify

        return self.proj(patchify(state.float()).flatten(1, 3))


@contextlib.contextmanager
def lora_modes(modules_on: list):
    """Set ``.enabled`` on LoRA modules ``[(module, on), ...]``, then restore. Disabled musubi modules run their
    original forward: no graph, no compute."""
    saved = [(m, m.enabled) for m, _on in modules_on]
    try:
        for m, on in modules_on:
            m.enabled = bool(on)
        yield
    finally:
        for m, prev in saved:
            m.enabled = prev


class H3PMD:
    """The student and the teacher on ONE DiT.

    * student: the DiT with ``init_net`` (the DiR / Turbo LoRA, frozen) and ``pmd_net`` (trainable; only the
      ``active`` modules on) enabled, the DiR head's iteration embedding ``r_embed[k]`` if a DiR run is the init,
      the mixed latent at its effective level, and ``NoiseInput`` on the raw state;
    * teacher: the same DiT with EVERY LoRA module disabled, no ``video_extra``, no ``video_temb_delta``, at the
      state's own level: base H3, one call, guidance 1 (optionally CFG against ``neg_text``).

    ``forward`` is ``h3_musubi.forward_video`` (a toy forward in the tests)."""

    def __init__(self, dit, *, sigmas, dials, noise_input: NoiseInput, init_net=None, pmd_net=None, active=None,
                 head=None, forward=None, teacher_cfg: float = 1.0, neg_text: torch.Tensor | None = None):
        if len(dials) != len(sigmas):
            raise ValueError(f"{len(dials)} dials for {len(sigmas)} levels")
        self.dit, self.sigmas, self.dials = dit, [float(s) for s in sigmas], [float(d) for d in dials]
        self.noise_input, self.head = noise_input, head
        self.init_modules = list(init_net.unet_loras) if init_net is not None else []
        self.pmd_modules = list(pmd_net.unet_loras) if pmd_net is not None else []
        self.active = set(id(m) for m in (self.pmd_modules if active is None else active))
        if forward is None:
            from h3_musubi import forward_video as forward
        self.forward = forward
        self.teacher_cfg, self.neg_text = float(teacher_cfg), neg_text
        if self.teacher_cfg != 1.0 and neg_text is None:
            raise ValueError("--teacher-cfg needs a negative / empty prompt embedding")

    @property
    def device(self):
        return self.dit.video_patch_proj.weight.device

    def student_modes(self):
        return lora_modes([(m, True) for m in self.init_modules] + [(m, id(m) in self.active) for m in self.pmd_modules])

    def teacher_modes(self):
        return lora_modes([(m, False) for m in self.init_modules + self.pmd_modules])

    def _audio(self, seed: int) -> torch.Generator:
        return torch.Generator(self.device).manual_seed(int(seed))

    def student(self, z: torch.Tensor, u: torch.Tensor, k: int, text: torch.Tensor, audio_seed: int) -> torch.Tensor:
        """Clean prediction ``x_hat = y + sigma_tilde * out`` (H3's head predicts ``x0 - eps``)."""
        y, sigma_t = mix_noise(z.float(), u, self.sigmas[k], self.dials[k])
        extra = self.noise_input(z)
        delta = None
        if self.head is not None:
            from dir_h3 import estimate_extra

            extra = extra + estimate_extra(self.head, torch.zeros_like(z, dtype=torch.float32), None)
            delta = self.head.r_embed[k]
        with self.student_modes():
            out = self.forward(self.dit, y, 1.0 - sigma_t, text, video_extra=extra, video_temb_delta=delta,
                               audio_generator=self._audio(audio_seed))
        return y.float() + sigma_t * out.float()

    def teacher(self, z: torch.Tensor, sigma: float, text: torch.Tensor, audio_seed: int) -> torch.Tensor:
        """The frozen teacher's posterior-mean estimate ``D(z) = z + sigma * out`` at level ``sigma``."""
        with self.teacher_modes():
            cond = self.forward(self.dit, z, 1.0 - float(sigma), text, audio_generator=self._audio(audio_seed)).float()
            if self.teacher_cfg != 1.0:
                uncond = self.forward(self.dit, z, 1.0 - float(sigma), self.neg_text,
                                      audio_generator=self._audio(audio_seed)).float()
                cond = uncond + self.teacher_cfg * (cond - uncond)
        return z.float() + float(sigma) * cond

    def bind(self, text: torch.Tensor, audio_seed: int):
        """``(student(z, u, k), teacher_at(sigma)(z))`` closures for one prompt, as ``pmd_step`` takes them."""
        def student(z, u, k):
            return self.student(z, u, k, text, audio_seed)

        def teacher_at(sigma):
            return lambda z: self.teacher(z, sigma, text, audio_seed)

        return student, teacher_at


@contextlib.contextmanager
def capture_block(dit, index: int, store: dict, count: int):
    """Keep the last ``count`` rows (the target video, the last packed segment) of block ``index``'s output."""
    def hook(_module, _args, output):
        store["f"] = output[:, -count:]

    handle = dit.blocks[index].register_forward_hook(hook)
    try:
        yield store
    finally:
        handle.remove()


def teacher_features(model: H3PMD, clean: torch.Tensor, level: float, text: torch.Tensor, *, block: int,
                     generator: torch.Generator, audio_seed: int) -> torch.Tensor:
    """Frozen-teacher features of ``clean`` noised to ``level`` (keeps the graph to ``clean`` if it has one)."""
    eps = torch.randn(clean.shape, device=clean.device, generator=generator)
    noisy = (1.0 - level) * clean.float() + level * eps
    count = (clean.shape[2]) * (clean.shape[3] // 2) * (clean.shape[4] // 2)
    store: dict = {}
    with capture_block(model.dit, block, store, count), model.teacher_modes():
        model.forward(model.dit, noisy, 1.0 - level, text, audio_generator=model._audio(audio_seed))
    return store["f"]


# ============================================================================== run files

def lora_state(network, modules) -> dict:
    names = tuple(m.lora_name + "." for m in modules)
    return {k: v.detach().cpu() for k, v in network.state_dict().items() if k.startswith(names)}


def parse_floats(spec, count: int | None = None) -> list[float]:
    values = [float(v) for v in (spec.split(",") if isinstance(spec, str) else spec)]
    if count is not None and len(values) != count:
        raise ValueError(f"{spec!r}: expected {count} values")
    return values


def parse_size(spec: str) -> tuple[int, int]:
    width, height = (int(v) for v in spec.lower().split("x"))
    if width % 32 or height % 32:
        raise ValueError(f"size {spec}: both sides must be multiples of 32 (one H3 token)")
    return width, height


def resolve_run(spec):
    if spec is None:
        return None
    from dir_two_track_h3 import resolve_run as resolve

    return resolve(str(spec))


def build_model(args, dev, *, train: bool):
    """Load H3 once, the init run's LoRA (frozen) and head, a fresh PMD LoRA and the noise input."""
    import re

    from dir_h3 import load_dir_run, parse_sigmas
    from h3_musubi import load_dit, make_lora

    dit = load_dit(device=dev, swap=args.swap, train=train)
    head = init_net = None
    config = {}
    run = resolve_run(args.init_run)
    if run is not None:
        head, init_net, config = load_dir_run(run, dit, tag=args.init_tag, device=dev, ema=args.init_ema)
        init_net.requires_grad_(False)
        if args.no_head:
            head = None
    sigmas = parse_sigmas(args.sigmas or config.get("sigmas") or "grid", 4)
    dials = parse_floats(args.dials, len(sigmas))
    pmd_net = make_lora(dit, args.rank)
    pmd_net.to(dev, torch.float32)
    pmd_net.requires_grad_(False)
    blocks = len(dit.blocks)
    active = []
    for module in pmd_net.unet_loras:
        block = int(re.search(r"blocks_(\d+)_", module.lora_name).group(1))
        if args.lora_last <= 0 or block >= blocks - args.lora_last:
            module.requires_grad_(True)
            active.append(module)
        else:
            module.enabled = False
    noise_input = NoiseInput(dit.video_patch_proj.out_features, dit.video_patch_proj.in_features).to(dev)
    neg = None
    if getattr(args, "teacher_cfg", 1.0) != 1.0:
        from train_dwd_h3 import load_text

        neg = load_text(str(args.neg_te)).to(dev)
    model = H3PMD(dit, sigmas=sigmas, dials=dials, noise_input=noise_input, init_net=init_net, pmd_net=pmd_net,
                  active=active, head=head, teacher_cfg=getattr(args, "teacher_cfg", 1.0), neg_text=neg)
    return model, pmd_net, active, {"init_config": config, "sigmas": sigmas, "dials": dials}


def load_pmd(path: Path, model: H3PMD, pmd_net) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if pmd_net.load_state_dict(saved["lora"], strict=False).unexpected_keys:
        raise SystemExit(f"{path}: LoRA keys that are not in this network (rank / blocks differ?)")
    model.noise_input.load_state_dict(saved["noise_input"])
    return saved


# ============================================================================== train

def _guard() -> None:
    from gpu_guard import refuse_if_busy

    refuse_if_busy("pmd_h3.py")


def _prompts(args):
    """``(caches, train prompt files)``, split as every other run splits them (``accel_h3.prompt_lists``)."""
    from accel_h3 import default_caches, prompt_lists

    caches = [Path(p) for p in args.cache] if args.cache else default_caches()
    return caches, prompt_lists(caches, args.holdout)[0]


def _load_text(path):
    from train_dwd_h3 import load_text

    return load_text(path)


def peak_mb(dev) -> float:
    return torch.cuda.max_memory_allocated(dev) / 2**20 if torch.device(dev).type == "cuda" else 0.0


# swappable for the CPU smoke test (test_pmd_h3.py): the GPU guard, the prompt list, text loading, the model
HOOKS = {"guard": _guard, "prompts": _prompts, "load_text": _load_text, "build_model": build_model}


def train(args) -> None:
    if os.environ.get("LOT_H3_TRAIN") != "1" and not args.check:
        raise SystemExit("training is gated: set LOT_H3_TRAIN=1")
    HOOKS["guard"]()
    load_text = HOOKS["load_text"]
    dev = torch.device(args.device)
    width, height = parse_size(args.size)
    shape = (1, 24, args.frames, height // 16, width // 16)       # --frames is LATENT frames (12 -> 39 pixels)
    caches, train_te = HOOKS["prompts"](args)
    if not train_te:
        raise SystemExit(f"no prompts in {caches}")
    model, pmd_net, active, info = HOOKS["build_model"](args, dev, train=True)
    sigmas, dials = info["sigmas"], info["dials"]
    train_levels = [int(k) for k in args.train_levels.split(",")]
    if any(k <= 0 or k >= len(sigmas) for k in train_levels):
        raise SystemExit("--train-levels: indices 1..3 (no PMD loss at pure noise)")
    lora_params = [p for m in active for p in m.parameters() if p.requires_grad]
    groups = [{"params": lora_params, "lr": args.lr},
              {"params": list(model.noise_input.parameters()), "lr": args.noise_lr or args.lr}]
    opt = torch.optim.AdamW(groups, betas=tuple(args.betas), eps=1e-8, weight_decay=args.weight_decay)
    disc = d_opt = None
    if args.gan_weight > 0:
        from train_dir_h3 import load_item
        import train_h3

        train_h3.MAX_LATENT_HW = (shape[3], shape[4])
        train_h3.CROP_MULTIPLE = 2
        real_items = [it for c in caches for it in train_h3.list_items(c, args.holdout)[0]]
        disc = DiscHead(model.dit.video_patch_proj.out_features).to(dev)
        d_opt = torch.optim.AdamW(disc.parameters(), lr=args.d_lr, betas=tuple(args.betas), eps=1e-8, weight_decay=0)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"}
    meta.update(sigmas=sigmas, dials=dials, shape=list(shape), caches=[str(c) for c in caches],
                init_config=info["init_config"], paper="arXiv 2611.26629")
    if not args.check:
        (args.out / "config.json").write_text(json.dumps(meta, indent=1, default=str))
    last_path = args.out / "pmd_last.pt"
    step, elapsed, rng = 0, 0.0, random.Random(args.seed)
    if last_path.exists() and not args.check and not args.fresh:
        last = load_pmd(last_path, model, pmd_net)
        opt.load_state_dict(last["opt"])
        if disc is not None and last.get("disc") is not None:
            disc.load_state_dict(last["disc"])
            d_opt.load_state_dict(last["d_opt"])
        step, elapsed = last["step"], last["elapsed"]
        rng.setstate(last["rng"])
        print(f"LOT_H3 kind=pmd_resume step={step} elapsed_min={elapsed:.1f} out={last_path}", flush=True)
    print(f"LOT_H3 kind=pmd_setup init={args.init_run} sigmas={','.join(f'{s:.4f}' for s in sigmas)} "
          f"dials={','.join(str(d) for d in dials)} lora_modules={len(active)} lora_params={sum(p.numel() for p in lora_params)} "
          f"shape={shape} prompts={len(train_te)} gan={args.gan_weight}", flush=True)

    def save(path: Path, with_opt: bool) -> None:
        state = {"lora": lora_state(pmd_net, active), "noise_input": model.noise_input.state_dict(), "step": step,
                 "elapsed": (time.time() - t_start) / 60, "rng": rng.getstate(), "config": meta}
        if with_opt:
            state.update(opt=opt.state_dict(), disc=disc.state_dict() if disc else None,
                         d_opt=d_opt.state_dict() if d_opt else None)
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)                                   # atomic: a crash mid-save keeps the old file

    def d_update(gen: torch.Generator) -> float:
        """One discriminator update on a fresh generated clip and a fresh real latent (features detached)."""
        te = rng.choice(train_te)
        text = load_text(str(te)).to(dev)
        student, _t = model.bind(text, rng.randrange(1 << 30))
        z = torch.randn(shape, device=dev, generator=gen)
        fake = any_step(student, z, sigmas, range(len(sigmas)), eta=args.eta, generator=gen)[-1]
        real, real_text = load_item(rng.choice(real_items), rng, dev, args.frames)
        with torch.no_grad():
            f_fake = teacher_features(model, fake, gan_level(gen, dev), text, block=args.d_block, generator=gen,
                                      audio_seed=rng.randrange(1 << 30))
            f_real = teacher_features(model, real, gan_level(gen, dev), real_text, block=args.d_block, generator=gen,
                                      audio_seed=rng.randrange(1 << 30))
        loss = d_loss(disc(f_real), disc(f_fake))
        d_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), args.grad_clip)
        d_opt.step()
        return float(loss)

    t_start = time.time() - elapsed * 60
    started, done_here = time.perf_counter(), 0
    log = open(args.out / "log.jsonl", "a") if not args.check else None
    while step < args.steps and (args.minutes <= 0 or time.time() - t_start < args.minutes * 60):
        gen = torch.Generator(dev).manual_seed(args.seed * 1_000_003 + step)
        d_losses = [d_update(gen) for _ in range(args.d_updates)] if disc is not None else []
        opt.zero_grad(set_to_none=True)
        losses, parts = [], []
        for _a in range(args.accum):                       # samples per update (the paper's batch is 16)
            k = (train_levels[(step * args.accum + _a) % len(train_levels)] if args.level_schedule == "mod"
                 else rng.choice(train_levels))
            text = load_text(str(rng.choice(train_te))).to(dev)
            audio_seed = rng.randrange(1 << 30)
            student, teacher_at = model.bind(text, audio_seed)
            z = torch.randn(shape, device=dev, generator=gen)
            z_k = rollout(student, z, sigmas, k, eta=args.eta, generator=gen)
            tilts = sample_tilts(z_k, args.tilts, factor=tuple(args.tilt_factor), radius=tuple(args.radius),
                                 generator=gen)
            noises = [torch.randn(shape, device=dev, generator=gen) for _ in range(args.draws)]
            extra = None
            if disc is not None:
                disc.requires_grad_(False)

                def extra(clean, _m, text=text, audio_seed=audio_seed):
                    f = teacher_features(model, clean, gan_level(gen, dev), text, block=args.d_block, generator=gen,
                                         audio_seed=audio_seed)
                    return args.gan_weight / args.draws * g_loss(disc(f))
            loss_a, stats_a = pmd_step(student, teacher_at(sigmas[k]), z_k, sigmas[k], k, tilts, noises,
                                       estimator=args.estimator, extra_loss=extra, scale=1.0 / args.accum)
            losses.append(float(loss_a))
            parts.append(stats_a)
            if disc is not None:
                disc.requires_grad_(True)
        loss = torch.tensor(sum(losses) / len(losses))
        stats = {key: sum(p[key] for p in parts) / len(parts) for key in parts[0]}
        params = lora_params + list(model.noise_input.parameters())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
        if args.check:
            sums = {"lora": sum(float(p.grad.abs().sum()) for p in lora_params if p.grad is not None),
                    "noise_input": float(model.noise_input.proj.weight.grad.abs().sum())}
            print(f"LOT_H3 kind=pmd_check level={k} sigma={sigmas[k]:.4f} rho={rho_of(sigmas[k]):.3f} loss={float(loss):.4g} "
                  f"grad_lora={sums['lora']:.4g} grad_noise_input={sums['noise_input']:.4g} grad_norm={grad_norm:.4g} "
                  f"ess={stats['ess']:.2f} shift_ratio={stats['shift_ratio']:.3g} "
                  f"peak_mb={peak_mb(dev):.0f} ok={int(math.isfinite(float(loss)) and sums['lora'] > 0)}",
                  flush=True)
            return
        if not math.isfinite(float(loss)) or not math.isfinite(grad_norm):
            print(f"LOT_H3 kind=pmd_nonfinite step={step} loss={float(loss)} grad_norm={grad_norm}", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        step += 1
        done_here += 1
        record = {"step": step, "level": k, "sigma": sigmas[k], "loss": float(loss), "grad_norm": grad_norm,
                  "d_loss": sum(d_losses) / len(d_losses) if d_losses else None, **stats,
                  "s_per_step": (time.perf_counter() - started) / done_here}
        log.write(json.dumps(record) + "\n")
        if step % args.log_every == 0 or step == args.steps:
            log.flush()
            print(f"LOT_H3 kind=pmd step={step} level={k} loss={float(loss):.4g} loss_plain={stats['loss_plain']:.4g} "
                  f"ess={stats['ess']:.2f} w_max={stats['w_max']:.2f} shift_ratio={stats['shift_ratio']:.3g} "
                  f"draw_spread={stats['draw_spread']:.3g} gan={stats['extra']:.3g} grad_norm={grad_norm:.3g} "
                  f"s_per_step={record['s_per_step']:.1f} peak_mb={peak_mb(dev):.0f}",
                  flush=True)
        if args.save_every and step % args.save_every == 0:
            save(args.out / f"pmd_{step}.pt", with_opt=False)
            save(last_path, with_opt=True)
            print(f"LOT_H3 kind=pmd_save step={step} out={args.out / f'pmd_{step}.pt'}", flush=True)
    save(args.out / "pmd_final.pt", with_opt=False)
    save(last_path, with_opt=True)
    log.close()
    print(f"LOT_H3 kind=pmd_done steps={step} minutes={(time.time() - t_start) / 60:.1f} out={args.out} ok=1", flush=True)


# ============================================================================== probe / score (the pass check)

def probe(args) -> None:
    """Final latents per held-out item and seed, in ``probe_dir_h3.py``'s item set and crop, so its ``--reference``
    dense20 latents (``ref.pt``) are the comparison. Seed 0 uses probe_dir_h3's noise ``z`` (seed ``i``)."""
    import train_h3
    from train_dir_h3 import load_item

    train_h3.MAX_LATENT_HW = tuple(args.max_latent_hw)
    train_h3.CROP_MULTIPLE = 2
    dev = torch.device("cuda")
    _train, held = train_h3.list_items(args.cache or train_h3.CACHE, args.holdout)
    count = min(args.items or len(held), len(held))
    items = [load_item(held[i], random.Random(1000 + i), dev, args.frames) for i in range(count)]
    keep, rows = {}, []
    if args.pmd is not None:
        saved_path = args.pmd / (f"pmd_{args.tag}.pt" if args.tag else "pmd_final.pt")
        if not saved_path.exists():
            saved_path = args.pmd / "pmd_last.pt"
        cfg = json.loads((args.pmd / "config.json").read_text())
        for key in ("init_run", "init_tag", "init_ema", "rank", "lora_last", "sigmas", "dials", "no_head"):
            if key in cfg and key not in ("sigmas", "dials"):
                setattr(args, key, cfg[key])
        args.sigmas = ",".join(str(s) for s in cfg["sigmas"])
        args.dials = ",".join(str(d) for d in cfg["dials"])
        args.teacher_cfg = 1.0
        model, pmd_net, _active, info = build_model(args, dev, train=False)
        load_pmd(saved_path, model, pmd_net)
        levels = [int(k) for k in args.levels.split(",")]
        eta = cfg.get("eta", 0.9) if args.eta is None else args.eta
    else:
        from dir_h3 import load_dir_run, next_state, parse_sigmas, generator_step, use_networks
        from h3_musubi import load_dit

        dit = load_dit(device=dev, swap=args.swap)
        head, net, config = load_dir_run(resolve_run(args.dir_run), dit, tag=args.tag, ema=args.init_ema)
        sigmas_dir = parse_sigmas(config.get("sigmas") or "1", int(config.get("iters", 4)))
        mix = bool(config.get("mix_state", False))
    with torch.no_grad():
        for i, (x0, text) in enumerate(items):
            for s in range(args.seeds):
                z = torch.randn(x0.shape, device=dev, generator=torch.Generator(dev).manual_seed(i if s == 0 else 1000 * s + i))
                if args.pmd is not None:
                    student, _t = model.bind(text, 7 * i)
                    gen = torch.Generator(dev).manual_seed(77_777 + 1000 * s + i)
                    out = any_step(student, z, model.sigmas, levels, eta=eta, generator=gen)[-1]
                else:
                    with use_networks(net, None, gen=True, fake=False):
                        estimate, state = torch.zeros_like(z), z
                        for r in range(args.iters):
                            estimate = generator_step(dit, head, state, estimate, r, text, sigma=sigmas_dir[r],
                                                      audio_generator=torch.Generator(dev).manual_seed(7 * i))
                            if r + 1 < args.iters:
                                state = next_state(state, estimate, sigmas_dir[r], sigmas_dir[r + 1], z if mix else None)
                    out = estimate
                keep[(i, f"{args.name}_s{s}")] = out.float().cpu()
            rows.append({"item": i})
            torch.save(keep, args.out / f"{args.name}.pt")
            print(f"LOT_H3 kind=pmd_probe name={args.name} item={i} seeds={args.seeds}", flush=True)
    (args.out / f"{args.name}.json").write_text(json.dumps({"meta": {k: str(v) for k, v in vars(args).items() if k != "func"},
                                                            "items": rows}, indent=1))
    print(f"LOT_H3 kind=pmd_probe name={args.name} items={count} ok=1", flush=True)


def lpips_frames(net, a: torch.Tensor, b: torch.Tensor, stride: int) -> float:
    """Mean LPIPS over every ``stride``-th pixel frame of two ``(1, 3, F, H, W)`` clips in [-1, 1], 256x256 bicubic
    (the paper's protocol, AlexNet LPIPS)."""
    import torch.nn.functional as F

    values = []
    for f in range(0, a.shape[2], stride):
        x = F.interpolate(a[:, :, f].float(), size=(256, 256), mode="bicubic", align_corners=False).clamp(-1, 1)
        y = F.interpolate(b[:, :, f].float(), size=(256, 256), mode="bicubic", align_corners=False).clamp(-1, 1)
        values.append(float(net(x, y)))
    return sum(values) / len(values)


def score(args) -> None:
    """LPIPS of seed 0 to dense20 (same z) and seed diversity (mean pairwise LPIPS between seeds), per name."""
    import lpips

    from h3_musubi import load_vae
    from lot_paths import LOT_H3_VAE

    ref = torch.load(args.out / f"{args.ref_name}.pt")
    vae = load_vae(LOT_H3_VAE)
    vdt = next(vae.parameters()).dtype
    net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    decoded: dict = {}

    def pix(key, table):
        if key not in decoded:
            with torch.no_grad():
                decoded[key] = vae.decode(table[key].to("cuda", vdt)).float()
        return decoded[key]

    table, results = {}, {}
    for name in args.names:
        data = torch.load(args.out / f"{name}.pt")
        items = sorted({i for i, _c in data})
        seeds = sorted({int(c.rsplit("_s", 1)[1]) for _i, c in data})
        to_dense, diversity = [], []
        with torch.no_grad():
            for i in items:
                if (i, "dense20") in ref:
                    to_dense.append(lpips_frames(net, pix((i, f"{name}_s0"), data), pix((i, "dense20"), ref), args.frame_stride))
                for a, b in itertools.combinations(seeds, 2):
                    diversity.append(lpips_frames(net, pix((i, f"{name}_s{a}"), data), pix((i, f"{name}_s{b}"), data),
                                                  args.frame_stride))
            decoded.clear()
        mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
        sem = lambda v: (statistics_stdev(v) / math.sqrt(len(v))) if len(v) > 1 else float("nan")  # noqa: E731
        results[name] = {"lpips_to_dense20": mean(to_dense), "lpips_to_dense20_sem": sem(to_dense),
                         "seed_diversity": mean(diversity), "seed_diversity_sem": sem(diversity),
                         "items": len(items), "seeds": len(seeds)}
        table[name] = results[name]
        print(f"LOT_H3 kind=pmd_score name={name} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                               for k, v in results[name].items()), flush=True)
    lines = ["| name | LPIPS to dense20 (lower) | seed diversity LPIPS (higher) | items x seeds |", "|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['lpips_to_dense20']:.4f} ± {r['lpips_to_dense20_sem']:.4f} | "
                     f"{r['seed_diversity']:.4f} ± {r['seed_diversity_sem']:.4f} | {r['items']} x {r['seeds']} |")
    if args.passes:
        pmd, dir4, dir2 = args.passes
        p, d4, d2 = results[pmd], results[dir4], results[dir2]
        ok = (p["lpips_to_dense20"] <= d2["lpips_to_dense20"] and p["seed_diversity"] >= d4["seed_diversity"])
        verdict = (f"PASS={int(ok)}: {pmd} LPIPS-to-dense20 {p['lpips_to_dense20']:.4f} vs {dir2} {d2['lpips_to_dense20']:.4f} "
                   f"(need <=) and vs {dir4} {d4['lpips_to_dense20']:.4f} (reported); seed diversity {p['seed_diversity']:.4f} "
                   f"vs {dir4} {d4['seed_diversity']:.4f} (need >=)")
        lines += ["", verdict]
        print(f"LOT_H3 kind=pmd_pass {verdict}", flush=True)
    (args.out / "SCORE.md").write_text("\n".join(lines) + "\n")
    (args.out / "score.json").write_text(json.dumps(results, indent=1))


def statistics_stdev(values) -> float:
    import statistics

    return statistics.stdev(values)


# ============================================================================== CLI

def add_model_args(p) -> None:
    p.add_argument("--init-run", default=DEFAULT_RUN,
                   help="student init: a DiR run dir or HF id (default: the published DiR checkpoint); 'none' = base H3")
    p.add_argument("--init-tag", default="", help="tag of the init run's save (e.g. _100)")
    p.add_argument("--init-ema", action="store_true", help="load the init run's averaged weights")
    p.add_argument("--no-head", action="store_true", help="drop the DiR head (no r_embed, no x0_proj)")
    p.add_argument("--sigmas", default=None, help="the 4 levels (default: the init run's config sigmas, else 'grid')")
    p.add_argument("--dials", default=",".join(str(d) for d in DEFAULT_DIALS), help="noise dial per level")
    p.add_argument("--rank", type=int, default=32, help="PMD LoRA rank (alpha = rank)")
    p.add_argument("--lora-last", type=int, default=0, help="PMD LoRA on the last K blocks only (0 = all 50)")
    p.add_argument("--swap", type=int, default=4, help="H3 blocks streamed from CPU")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="phase", required=True)

    t = sub.add_parser("train", help="train the PMD student")
    add_model_args(t)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--check", action="store_true", help="one update with gradients, print sums and peak, write nothing")
    t.add_argument("--fresh", action="store_true", help="ignore OUT/pmd_last.pt")
    t.add_argument("--cache", nargs="*", default=None, help="prompt caches (default: $ACCEL_CACHES, else stills + clips)")
    t.add_argument("--holdout", type=int, default=8)
    t.add_argument("--size", default="512x288", help="WxH pixels (multiples of 32)")
    t.add_argument("--frames", type=int, default=12, help="LATENT frames (repo convention; 12 = 39 pixel frames)")
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--minutes", type=float, default=120, help="wall-clock budget across resumes (0 = none)")
    t.add_argument("--lr", type=float, default=2e-5, help="LoRA lr (paper: 5e-7, full fine-tune)")
    t.add_argument("--noise-lr", type=float, default=None, help="lr of the new input half (default --lr)")
    t.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--grad-clip", type=float, default=10.0)
    t.add_argument("--accum", type=int, default=1, help="samples per update (paper: global batch 16)")
    t.add_argument("--draws", type=int, default=4, help="M student draws per state")
    t.add_argument("--tilts", type=int, default=4, help="K tilted teacher queries per state (same draws)")
    t.add_argument("--radius", type=float, nargs=2, default=[0.25, 0.5])
    t.add_argument("--tilt-factor", type=int, nargs=3, default=[4, 8, 8], help="coarse grid factor in T H W")
    t.add_argument("--estimator", choices=sorted(ESTIMATORS), default="jk")
    t.add_argument("--train-levels", default="1,2,3")
    t.add_argument("--level-schedule", choices=("mod", "random"), default="random")
    t.add_argument("--eta", type=float, default=0.9, help="re-noising: 1 = fresh noise, <1 = CPS mix")
    t.add_argument("--teacher-cfg", type=float, default=1.0, help="paper: 1 (no CFG). !=1 needs --neg-te, 2x teacher cost")
    t.add_argument("--neg-te", type=Path, default=None, help="an empty / negative prompt _te.safetensors for --teacher-cfg")
    t.add_argument("--gan-weight", type=float, default=0.0, help="paper: 0.75; off by default (needs real clips)")
    t.add_argument("--d-updates", type=int, default=5)
    t.add_argument("--d-lr", type=float, default=5e-5)
    t.add_argument("--d-block", type=int, default=25, help="teacher block whose output the discriminator reads")
    t.add_argument("--save-every", type=int, default=50)
    t.add_argument("--log-every", type=int, default=1)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    t.set_defaults(func=train)

    p = sub.add_parser("probe", help="final latents per held-out item and seed (PMD any-step, or a DiR run)")
    add_model_args(p)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--pmd", type=Path, default=None, help="a pmd_h3.py train --out dir")
    p.add_argument("--tag", default="", help="PMD: save step N (default pmd_final.pt); DiR: the run's tag")
    p.add_argument("--levels", default="0,1,2,3", help="PMD any-step levels, e.g. 0,2 for 2 steps")
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--dir-run", default=DEFAULT_RUN, help="without --pmd: the DiR run to sample")
    p.add_argument("--iters", type=int, default=4, help="DiR: stop after this many iterations")
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--holdout", type=int, default=8)
    p.add_argument("--items", type=int, default=None)
    p.add_argument("--max-latent-hw", type=int, nargs=2, default=[18, 32], help="18 32 = 512x288")
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--seeds", type=int, default=4)
    p.set_defaults(func=probe)

    s = sub.add_parser("score", help="LPIPS to dense20 and seed diversity; the pass check")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--names", nargs="+", required=True)
    s.add_argument("--ref-name", default="ref")
    s.add_argument("--frame-stride", type=int, default=4)
    s.add_argument("--pass", dest="passes", nargs=3, default=None, metavar=("PMD", "DIR4", "DIR2"))
    s.set_defaults(func=score)

    args = parser.parse_args()
    if getattr(args, "init_run", None) in ("none", "None", ""):
        args.init_run = None
    if hasattr(args, "out"):
        args.out.mkdir(parents=True, exist_ok=True)
    try:
        from setproctitle import setproctitle

        setproctitle(f"pmd:{args.phase} {args.out.name}"[:60])
    except ImportError:
        pass
    args.func(args)


if __name__ == "__main__":
    main()
