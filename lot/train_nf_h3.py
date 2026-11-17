#!/usr/bin/env python3
"""Neighbor Forcing (Stage 1) and ConvKV Memory + step distillation (Stage 2) for H3. Experimental.

    python3 lot/train_nf_h3.py --stage 1 --check
    LOT_H3_TRAIN=1 python3 lot/train_nf_h3.py --stage 1 --steps 2000 --out lot/runs/nf_stage1
    LOT_H3_TRAIN=1 python3 lot/train_nf_h3.py --stage 2 --init lot/runs/nf_stage1 --steps 400 --out lot/runs/nf_stage2

Paper: SoulX-LiveAct, arXiv 2603.11746 (fig. 2 right; ``nf_h3.py``, ``docs/SOULX_LIVEACT.md``).

**Stage 1, Neighbor Forcing training** (sec. 2.2, eq. 2-5). A real clip of ``6 + 8k`` latent frames is
noised to ONE step for every block, ``x_t = (1 - t) x0 + t eps`` (eq. 2), and the block-causal model
(eq. 5) regresses the flow target in one pass (eq. 4):

    loss = mse(G(x_t, t, Mask), x0 - eps)          (H3's head sign; the paper writes eps - x)

``t ~ U(0, 1)`` as in eq. 4 (``--sigma-shift 12`` gives H3's own density instead). The clip's first
latent frame is the reference image ``x_ref``, fed clean as H3's FL2VA ``first`` condition
(``--ref none`` drops it). Trained: a LoRA on every block (the paper trains its audio cross-attention,
which H3 does not have).

**Stage 2, ConvKV Memory & step distillation** (sec. 2.3, sec. 3). The generator is the Stage-1 model
with the ConvKV context (``NFConfig(memory=True)``), sampled in 3 steps. One generator update:

1. the 3-step rollout of every block from noise, without grad, up to a random step ``s``
   (``nf_parallel_sample``: under Neighbor Forcing the autoregressive rollout of a clip is one masked
   pass per step), then step ``s`` with grad gives ``x0_hat = x_s + sigma_s G(x_s)``;
2. DMD [29]: ``x_tau = (1 - tau) x0_hat + tau eps``; the real score is the base H3 with every LoRA off
   and all frames visible, the fake score the same plus the fake LoRA; with DMD's normaliser
   ``w = 1 / mean|x0_hat - x0_real|``:

       grad = (x0_fake - x0_real) * w        loss = 0.5 * mse(x0_hat, (x0_hat - grad).detach())

3. the fake LoRA regresses the flow target on re-noised, detached ``x0_hat`` (``--fake-updates`` times).

Trained: the generator LoRA and the ConvKV convolutions (generator side), the fake LoRA (fake side).
What the paper does not say and is chosen here: the random exit step and detached earlier steps
(Self-Forcing's rollout), ``tau`` uniform on [0.02, 0.98], one fake update per generator update, AdamW
at 1e-4, LoRA rank 16. Batch size is 1 (H3's forward takes one sample; ``--accum`` accumulates).

Data: an H3 clip cache (``build_clip_cache_h3.py``), by default the Nikki clips of
``cache_atlas_clips`` (``--source set_nk``; 22 latent frames = blocks 6 + 8 + 8). Read only.

Writes only under ``--out`` (default ``lot/runs/train_nf_h3``, gitignored): ``log.jsonl`` and, at
``--save-every`` and the end, ``lora_gen{tag}.safetensors``, ``lora_fake{tag}.safetensors`` and
``convkv{tag}.pt`` (Stage 2), ``config.json``; load with ``nf_h3.load_nf_run``. Refuses to start unless
``LOT_H3_TRAIN=1`` (``--check`` is allowed).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA init

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_CLIPS_CACHE, LOT_H3_CLIPS_SOURCE  # noqa: E402

from dir_h3 import use_networks  # noqa: E402
from h3_musubi import VIDEO_SHIFT, sample_sigma  # noqa: E402
from h3_musubi import audio_frames  # noqa: E402
from nf_h3 import (AUDIO_CLEAN, BASE_SIGMAS, DENSE, FIRST_BLOCK, NEXT_BLOCK, ConvKV, NFConfig, NoiseBank,  # noqa: E402
                   dir_step, few_step_sigmas, nf_dir_parallel_sample, nf_forward, nf_parallel_sample, save_nf_run)

TAU_MIN, TAU_MAX = 0.02, 0.98
SIGMA_MIN, SIGMA_MAX = 1e-3, 1.0 - 1e-3
MATCH_BOUND = 1e-2      # --check: the grad re-forward against its no-grad value, relative RMS (measured: 0)


# --------------------------------------------------------------------------------- data

def list_clips(cache: Path, sources: list[str], frames: int, holdout: int) -> tuple[list, list]:
    """``(latent_path, te_path)`` pairs with at least ``frames`` latent frames whose ``source`` metadata
    starts with one of ``sources`` (empty: every clip), split deterministically into train / held out."""
    from safetensors import safe_open

    if not cache.is_dir():
        raise SystemExit(f"cache directory {cache} does not exist")
    items = []
    for latent in sorted(cache.glob("*_minimaxh3.safetensors")):
        stem = re.sub(r"_\d{4}x\d{4}_minimaxh3\.safetensors$", "", latent.name)
        te = cache / f"{stem}_minimaxh3_te.safetensors"
        if not te.is_file():
            continue
        with safe_open(str(latent), "pt") as handle:
            key = next((k for k in handle.keys() if k.startswith("latent")), None)
            source = (handle.metadata() or {}).get("source", "")
            shape = handle.get_slice(key).get_shape() if key else []
        if len(shape) != 4 or shape[1] < frames:
            continue
        if sources and not any(source.startswith(prefix) for prefix in sources):
            continue
        items.append((latent, te))
    if len(items) <= holdout:
        raise SystemExit(f"{cache}: {len(items)} clips of {frames}+ latent frames match {sources}, need more than {holdout}")
    random.Random(0).shuffle(items)
    return items[holdout:], items[:holdout]


def load_clip(item, rng, device, frames: int):
    """Clean latent ``(1, 24, frames, H, W)`` from the clip's first frame on, and its text."""
    import train_h3

    x0, text = train_h3.load_item(item, rng, device)
    return x0[:, :, :frames].contiguous(), text


def load_audio(item, device, frames: int, root: Path = LOT_H3_CLIPS_SOURCE) -> torch.Tensor:
    """The clip's clean H3 audio latent ``(1, 32, 2, A)`` for its first ``frames`` latent frames.

    The clip cache holds video and text only; the audio latent is ``audio_latent.pt`` in the clip's
    source folder (``<root>/<source metadata>``), written when the clip was made."""
    from safetensors import safe_open

    with safe_open(str(item[0]), "pt") as handle:
        source = (handle.metadata() or {}).get("source", "")
    path = Path(root) / source / "audio_latent.pt"
    if not source or not path.is_file():
        raise SystemExit(f"--audio driving: no audio latent for {item[0].name} at {path}")
    audio = torch.load(path, map_location="cpu", weights_only=True).float()
    count = audio_frames(frames)
    if audio.ndim != 4 or audio.shape[1:3] != (32, 2) or audio.shape[-1] < count:
        raise SystemExit(f"{path}: audio latent {tuple(audio.shape)} does not cover {count} frames")
    return audio[..., :count].contiguous().to(device)


def dmd_target(x0_hat: torch.Tensor, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """The detached regression target of the DMD generator loss ``0.5 * mse(x0_hat, target)``.

    ``target = x0_hat - (fake - real) * w`` with DMD's normaliser ``w = 1 / mean|x0_hat - real|``: descent
    moves ``x0_hat`` along ``(real - fake) * w``, toward what the real model calls clean and away from what
    the fake model (the generator's own distribution) calls clean."""
    weight = 1.0 / (x0_hat - real).abs().mean().clamp(min=1e-6)
    return (x0_hat - (fake - real) * weight).detach()


# ---------------------------------------------------------------------------------- run

class NFRun:
    """Everything one update needs, shared by ``--check``, training and the CPU test."""

    def __init__(self, dit, gen_net, fake_net, convkv, args, device, head=None, fake_head=None, dir_meta=None) -> None:
        self.dit, self.gen_net, self.fake_net, self.convkv = dit, gen_net, fake_net, convkv
        self.args, self.device = args, device
        self.causal = NFConfig(args.blocks[0], args.blocks[1], causal=True, memory=False)
        self.memory = NFConfig(args.blocks[0], args.blocks[1], causal=True, memory=True) if args.stage == 2 else None
        self.sigmas = few_step_sigmas([float(s) for s in args.sigmas.split(",")], args.sampler_shift)
        self.loss_scale = 1.0
        # DiR under Neighbor Forcing: the generator is a DiR run's LoRA + DirHead, sampled in R iterations
        self.head, self.fake_head = head, fake_head
        if head is not None:
            self.dir_sigmas = [float(s) for s in dir_meta["sigmas"]]
            self.mix_state = bool(dir_meta.get("mix_state", False))
            self.fake_final = dir_meta.get("fake_source", "iteration") == "final"
            self.tau_max = float(dir_meta.get("tau_max", TAU_MAX))

    # ---- shared pieces
    def reference(self, x0: torch.Tensor):
        """``x_ref``: the clip's first latent frame (the VAE latent of its first pixel frame), or None."""
        return x0[:, :, :1].contiguous() if self.args.ref == "first" else None

    def draw_sigma(self) -> float:
        """Eq. 4: ``t ~ U(0, 1)`` (``--sigma-shift 1``), or under a schedule shift."""
        sigma = float(sample_sigma(float(self.args.sigma_shift), "cpu"))
        return min(max(sigma, SIGMA_MIN), SIGMA_MAX)

    def draw_tau(self, cap: float | None = None) -> float:
        """A DMD noise level. ``cap`` (DiR's ``--tau-max``, generator reads only): redrawn, not clamped,
        until at or under the cap, as ``train_dir_h3.Run.draw_tau`` does."""
        dist = self.args.tau_shift

        def draw() -> float:
            if isinstance(dist, str) and dist.startswith("beta:"):      # DiR's importance sampling: tau ~ Beta(a, 1)
                tau = float(torch.distributions.Beta(float(dist.split(":", 1)[1]), 1.0).sample((1,)))
            else:
                tau = float(sample_sigma(float(dist), "cpu"))
            return min(max(tau, TAU_MIN), TAU_MAX)

        tau = draw()
        if cap is not None:
            for _ in range(64):
                if tau <= cap:
                    break
                tau = draw()
            tau = min(tau, cap)
        return tau

    def audio_kwargs(self, audio, seed: int | None) -> dict:
        """``nf_forward`` arguments for the audio rows: clean driving audio, or seeded silent noise."""
        if audio is not None:
            return {"audio": audio, "audio_t": AUDIO_CLEAN}
        return {"audio_generator": None if seed is None else torch.Generator(self.device).manual_seed(seed)}

    def dense_x0(self, x_tau, tau: float, text, first, seed: int, audio=None) -> torch.Tensor:
        """A bidirectional model's clean estimate at ``tau`` (the caller sets which LoRAs are on)."""
        out = nf_forward(self.dit, x_tau, 1.0 - tau, text, config=DENSE, first=first, **self.audio_kwargs(audio, seed))
        return x_tau.float() + tau * out.float()

    # ---- Stage 1
    def stage1_loss(self, x0, text, *, sigma: float | None = None, eps=None, config: NFConfig | None = None,
                    audio_seed: int | None = None, backward: bool = False, audio=None):
        """Eq. 4 on one clip. Returns ``(loss, stats)``; the generator LoRA is on.

        ``backward=True`` runs the backward here, inside the LoRA switch: activation checkpointing
        re-runs every block during backward and must find the same LoRAs on as in the forward."""
        sigma = self.draw_sigma() if sigma is None else float(sigma)
        eps = torch.randn_like(x0) if eps is None else eps
        x_t = ((1.0 - sigma) * x0.float() + sigma * eps.float()).to(x0.dtype)          # eq. 2, every block at one step
        with use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
            velocity = nf_forward(self.dit, x_t, 1.0 - sigma, text, config=config or self.causal,
                                  first=self.reference(x0), **self.audio_kwargs(audio, audio_seed))
            loss = F.mse_loss(velocity.float(), x0.float() - eps.float())
            if backward:
                (loss * self.loss_scale).backward()
        return loss, {"loss": float(loss.detach()), "sigma": sigma}

    # ---- Stage 2
    def generator_step(self, x_s, audio_s, step: int, text, first, driving: bool = False) -> torch.Tensor:
        """Step ``step`` of the few-step generator on a whole clip: the clean estimate ``x0_hat``."""
        sigma = float(self.sigmas[step])
        velocity = nf_forward(self.dit, x_s, 1.0 - sigma, text, config=self.memory, convkv=self.convkv, first=first,
                              audio=audio_s, audio_t=AUDIO_CLEAN if driving else None)
        return x_s.float() + sigma * velocity.float()

    def generator_update(self, x0, text, *, exit_step: int | None = None, backward: bool = True, audio=None):
        """One DMD generator update on a clip shaped like ``x0`` (its values are used for ``x_ref`` only).

        Order, as in ``train_dir_h3.py``: every no-grad forward first (rollout, ``x0_hat`` value, real,
        fake), then the grad forward followed straight by its backward, because the block-swap offloader
        changes mode between no-grad and grad forwards.
        """
        first = self.reference(x0)
        latent_t, lat_h, lat_w = x0.shape[2:]
        steps = len(self.sigmas) - 1
        seed = int(torch.randint(0, 2**31 - 1, (1,)))
        step = int(torch.randint(0, steps, (1,))) if exit_step is None else int(exit_step)
        noise = NoiseBank(seed, self.device)
        with torch.no_grad(), use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
            x_s, audio_s = nf_parallel_sample(self.dit, text, latent_t, lat_h=lat_h, lat_w=lat_w, config=self.memory,
                                              sigmas=self.sigmas, noise=noise, convkv=self.convkv, first=first,
                                              sampler=self.args.sampler, stop_step=step, audio=audio)
            x_s = x_s.to(x0.dtype)
            x0_hat = self.generator_step(x_s, audio_s, step, text, first, audio is not None)
        tau = self.draw_tau()
        eps = torch.randn_like(x0_hat)
        x_tau = ((1.0 - tau) * x0_hat + tau * eps).to(x0.dtype)
        with torch.no_grad():
            with use_networks(self.gen_net, self.fake_net, gen=False, fake=False):
                real = self.dense_x0(x_tau, tau, text, first, seed + 1, audio)
            with use_networks(self.gen_net, self.fake_net, gen=False, fake=True):
                fake = self.dense_x0(x_tau, tau, text, first, seed + 1, audio)
            target = dmd_target(x0_hat, real, fake)
        with use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
            live = self.generator_step(x_s, audio_s, step, text, first, audio is not None)
            loss = 0.5 * F.mse_loss(live, target)
            if backward:        # inside the switch: checkpointing re-runs the blocks with these LoRAs
                (loss * self.loss_scale).backward()
        stats = {"loss": float(loss.detach()), "tau": tau, "exit_step": step,
                 "match": float((live.detach() - x0_hat).abs().max()),
                 "match_rel": float((live.detach() - x0_hat).pow(2).mean().sqrt() / x0_hat.pow(2).mean().sqrt().clamp(min=1e-8)),
                 "gap": float((fake - real).abs().mean()), "x0_abs": float(x0_hat.abs().mean()),
                 "real_gap": float((x0_hat - real).abs().mean())}
        return loss, (x0_hat.detach(), text, first, seed, audio), stats

    # ---- Stage 2 with a DiR generator (not in the paper)
    def dir_rollout(self, shape, text, first, seed: int, r: int):
        """``(state, estimate, audio)`` entering iteration ``r`` of a clip of ``shape`` (call without grad, generator on)."""
        latent_t, lat_h, lat_w = shape
        return nf_dir_parallel_sample(self.dit, self.head, text, latent_t, lat_h=lat_h, lat_w=lat_w, config=self.memory,
                                      sigmas=self.dir_sigmas, noise=NoiseBank(seed, self.device), mix_state=self.mix_state,
                                      convkv=self.convkv, first=first, stop_iter=r)

    def dir_iteration(self, state, estimate, audio, r: int, text, first) -> torch.Tensor:
        return dir_step(self.dit, self.head, state, estimate, r, text, self.dir_sigmas[r], config=self.memory,
                        convkv=self.convkv, first=first, audio=audio)

    def dir_scores(self, x_tau, tau: float, r: int, text, first, seed: int):
        """``(real, fake)`` clean estimates at ``tau``: the base H3 with every LoRA off, and the DiR run's
        fake LoRA + fake head. Both bidirectional. Without a reference frame these are ``dir_h3.real_x0`` and
        ``dir_h3.fake_x0``; with one, the same forward with H3's FL2VA ``first`` rows."""
        from dir_h3 import fake_x0, real_x0

        index = len(self.dir_sigmas) - 1 if self.fake_final else r
        delta = self.fake_head.fake_r_embed[index]

        def audio():
            return torch.Generator(self.device).manual_seed(seed)

        with use_networks(self.gen_net, self.fake_net, gen=False, fake=False):
            real = (real_x0(self.dit, x_tau, tau, text, audio_generator=audio()) if first is None
                    else self.dense_x0(x_tau, tau, text, first, seed))
        with use_networks(self.gen_net, self.fake_net, gen=False, fake=True):
            if first is None:
                fake = fake_x0(self.dit, self.fake_head, x_tau, tau, index, text, audio_generator=audio())
            else:
                out = nf_forward(self.dit, x_tau, 1.0 - tau, text, config=DENSE, first=first, audio_generator=audio(),
                                 video_temb_delta=delta)
                fake = x_tau.float() + tau * out.float()
        return real, fake

    def dir_generator_update(self, x0, text, r: int, *, backward: bool = True):
        """One DMD generator update of the causal DiR generator at iteration ``r``: iterations ``< r`` without
        grad, iteration ``r`` with grad (``train_dir_h3.Run.generator_update`` with the block-causal reads and
        ConvKV). ``x0`` gives the clip shape and, with ``--ref first``, the reference frame."""
        first = self.reference(x0)
        seed = int(torch.randint(0, 2**31 - 1, (1,)))
        with torch.no_grad(), use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
            state, estimate, audio = self.dir_rollout(tuple(x0.shape[2:]), text, first, seed, r)
            x0_hat = self.dir_iteration(state, estimate, audio, r, text, first)
        # --dmd-draws K: the target is averaged over K independent (tau, eps) reads of this one sample
        targets = []
        with torch.no_grad():
            for draw in range(max(1, int(getattr(self.args, "dmd_draws", 1)))):
                tau = self.draw_tau(self.tau_max)
                eps = torch.randn_like(x0_hat)
                x_tau = (1.0 - tau) * x0_hat + tau * eps
                real, fake = self.dir_scores(x_tau, tau, r, text, first, seed + 1 + draw)
                targets.append(dmd_target(x0_hat, real, fake))
            target = sum(targets) / len(targets)
        with use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
            live = self.dir_iteration(state, estimate, audio, r, text, first)
            loss = 0.5 * F.mse_loss(live, target)
            if backward:
                (loss * self.loss_scale).backward()
        stats = {"loss": float(loss.detach()), "tau": tau, "r": r,
                 "match": float((live.detach() - x0_hat).abs().max()),
                 "gap": float((fake - real).abs().mean()), "x0_abs": float(x0_hat.abs().mean()),
                 "real_gap": float((x0_hat - real).abs().mean())}
        return loss, (x0_hat.detach(), text, first, seed, r), stats

    def dir_fake_update(self, sample, *, backward: bool = True, fresh: int = 0):
        """Flow regression of the fake model on a detached generator sample, conditioned on its iteration.
        ``fresh=k > 0`` draws a new rollout instead of reusing the generator update's sample. With an endpoint
        critic (``fake_source: final``) the sample is always a finished clip."""
        from dir_h3 import fake_velocity

        x0_hat, text, first, seed, r = sample
        last = len(self.dir_sigmas) - 1
        if fresh or (self.fake_final and r != last):
            r = last if self.fake_final else r
            with torch.no_grad(), use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
                state, estimate, audio = self.dir_rollout(tuple(x0_hat.shape[2:]), text, first, seed + 5 + 11 * fresh, r)
                x0_hat = self.dir_iteration(state, estimate, audio, r, text, first)
        index = last if self.fake_final else r
        tau = self.draw_tau()
        eps = torch.randn_like(x0_hat)
        x_tau = (1.0 - tau) * x0_hat + tau * eps
        audio = torch.Generator(self.device).manual_seed(seed + 2)
        with use_networks(self.gen_net, self.fake_net, gen=False, fake=True):
            if first is None:
                velocity = fake_velocity(self.dit, self.fake_head, x_tau, tau, index, text, audio_generator=audio)
            else:
                velocity = nf_forward(self.dit, x_tau, 1.0 - tau, text, config=DENSE, first=first, audio_generator=audio,
                                      video_temb_delta=self.fake_head.fake_r_embed[index]).float()
            loss = F.mse_loss(velocity, x0_hat - eps)
            if backward:
                (loss * self.loss_scale).backward()
        return loss, tau

    @torch.no_grad()
    def dir_eval(self, items, frames: int, tau: float = 0.6) -> dict:
        """Held-out read of the generator: for each item's prompt, a finished clip from fixed noise, re-noised
        to ``tau`` with fixed noise, and the frozen base model's clean answer. ``real_gap`` is
        ``mean|x0_hat - x0_real|`` (how far the teacher would move the sample), also per block; ``motion`` is the
        mean |difference| between latent frames 5 apart. Comparable across steps: every draw is fixed."""
        rng = random.Random(0)
        spans = self.memory.spans(frames)
        gaps, blocks, motion = [], [[] for _ in spans], []
        for index, item in enumerate(items):
            x0, text = load_clip(item, rng, self.device, frames)
            first = self.reference(x0)
            with use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
                state, estimate, audio = self.dir_rollout(tuple(x0.shape[2:]), text, first, 1000 + index, len(self.dir_sigmas) - 1)
                x0_hat = self.dir_iteration(state, estimate, audio, len(self.dir_sigmas) - 1, text, first)
            eps = torch.randn(x0_hat.shape, generator=torch.Generator().manual_seed(77 + index)).to(x0_hat)
            real, _fake = self.dir_scores((1.0 - tau) * x0_hat + tau * eps, tau, 0, text, first, 5)
            gaps.append(float((x0_hat - real).abs().mean()))
            for k, (a, b) in enumerate(spans):
                blocks[k].append(float((x0_hat[:, :, a:b] - real[:, :, a:b]).abs().mean()))
            motion.append(float((x0_hat[:, :, 5:] - x0_hat[:, :, :-5]).abs().mean()))
        mean = lambda values: sum(values) / max(1, len(values))      # noqa: E731
        return {"real_gap": mean(gaps), "real_gap_blocks": [mean(b) for b in blocks], "motion": mean(motion)}

    def fake_update(self, sample, *, backward: bool = True):
        """Flow regression of the fake model on the generator's (detached) sample."""
        x0_hat, text, first, seed, audio = sample
        tau = self.draw_tau()
        eps = torch.randn_like(x0_hat)
        x_tau = (1.0 - tau) * x0_hat + tau * eps
        with use_networks(self.gen_net, self.fake_net, gen=False, fake=True):
            velocity = nf_forward(self.dit, x_tau, 1.0 - tau, text, config=DENSE, first=first,
                                  **self.audio_kwargs(audio, seed + 2))
            loss = F.mse_loss(velocity.float(), x0_hat - eps)
            if backward:
                (loss * self.loss_scale).backward()
        return loss, tau


# -------------------------------------------------------------------------------- helpers

def grad_sum(params) -> float:
    return sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)


def zero_grads(params) -> None:
    for p in params:
        p.grad = None


@torch.no_grad()
def heldout_loss(run: NFRun, items, device, frames: int, sigmas=(0.3, 0.6, 0.9)) -> float:
    """Eq. 4 on the held-out clips at fixed noise levels and fixed noise (comparable across steps)."""
    total, count = 0.0, 0
    rng = random.Random(0)
    for index, item in enumerate(items):
        x0, text = load_clip(item, rng, device, frames)
        audio = load_audio(item, device, frames) if run.args.audio == "driving" else None
        for k, sigma in enumerate(sigmas):
            eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1000 * index + k)).to(x0)
            loss, _stats = run.stage1_loss(x0, text, sigma=sigma, eps=eps, audio_seed=7 + k, audio=audio)
            total, count = total + float(loss), count + 1
    return total / max(1, count)


def check(run: NFRun, gen_params, fake_params, items, rng, device, frames: int) -> None:
    """A dry pass: one update of each kind, nothing written."""
    x0, text = load_clip(items[0], rng, device, frames)
    audio = load_audio(items[0], device, frames) if run.args.audio == "driving" else None
    if run.head is not None:
        last = len(run.dir_sigmas) - 1
        head_params = [p for p in run.head.parameters() if p.requires_grad]
        with torch.no_grad(), use_networks(run.gen_net, run.fake_net, gen=True, fake=False):
            state, estimate, rows = run.dir_rollout(tuple(x0.shape[2:]), text, run.reference(x0), 0, last)
            sample = run.dir_iteration(state, estimate, rows, last, text, run.reference(x0))
        floss, _tau = run.dir_fake_update((sample, text, run.reference(x0), 0, last))
        fake_grad = grad_sum(fake_params)
        zero_grads(fake_params)
        started = time.perf_counter()
        loss, _sample, stats = run.dir_generator_update(x0, text, last)
        seconds = time.perf_counter() - started
        lora_grad = grad_sum([p for p in gen_params if not any(p is q for q in head_params + list(run.convkv.parameters()))])
        head_grad, conv_grad = grad_sum(head_params), grad_sum(list(run.convkv.parameters()))
        ok = int(math.isfinite(float(loss)) and math.isfinite(float(floss)) and fake_grad > 0 and lora_grad > 0
                 and head_grad > 0 and conv_grad > 0 and stats["match"] < 5e-2)
        print(f"LOT_H3 kind=nf_check stage=2 dir=1 frames={frames} latent_hw={tuple(x0.shape[-2:])} "
              f"sigmas={','.join(f'{s:.4f}' for s in run.dir_sigmas)} mix_state={int(run.mix_state)} loss={float(loss):.5g} "
              f"fake_loss={float(floss):.5f} gap={stats['gap']:.4g} real_gap={stats['real_gap']:.4g} match={stats['match']:.3g} "
              f"lora_grad={lora_grad:.4g} head_grad={head_grad:.4g} conv_grad={conv_grad:.4g} fake_grad={fake_grad:.4g} "
              f"gen_update_s={seconds:.1f} peak_mb={peak_mb(device):.0f} ok={ok}", flush=True)
    elif run.args.stage == 1:
        eps = torch.randn_like(x0)
        with torch.no_grad():
            dense, _ = run.stage1_loss(x0, text, sigma=0.6, eps=eps, config=DENSE, audio_seed=3, audio=audio)
            causal, _ = run.stage1_loss(x0, text, sigma=0.6, eps=eps, audio_seed=3, audio=audio)
        loss, _stats = run.stage1_loss(x0, text, sigma=0.6, eps=eps, audio_seed=3, backward=True, audio=audio)
        gen_grad = grad_sum(gen_params)
        # the grad forward (checkpointing, swap in training mode) against the no-grad one (measured: equal to 5 digits)
        ok = int(math.isfinite(float(loss)) and gen_grad > 0 and abs(float(loss) - float(causal)) < 1e-2 * float(causal))
        print(f"LOT_H3 kind=nf_check stage=1 audio={run.args.audio} frames={frames} latent_hw={tuple(x0.shape[-2:])} "
              f"blocks={run.causal.spans(frames)} loss_dense={float(dense):.5f} loss_causal={float(causal):.5f} "
              f"loss_grad={float(loss):.5f} gen_grad={gen_grad:.4g} peak_mb={peak_mb(device):.0f} ok={ok}", flush=True)
    else:
        # The fake LoRA starts equal to the base model, so the first DMD direction is exactly zero.
        # One fake step first, as in training after its first update, so the generator gradient is real.
        optimizer = torch.optim.AdamW(fake_params, lr=run.args.fake_lr)
        with torch.no_grad(), use_networks(run.gen_net, run.fake_net, gen=True, fake=False):
            noise = NoiseBank(0, device)
            sample = nf_parallel_sample(run.dit, text, frames, lat_h=x0.shape[-2], lat_w=x0.shape[-1], config=run.memory,
                                        sigmas=run.sigmas, noise=noise, convkv=run.convkv, first=run.reference(x0),
                                        sampler=run.args.sampler, audio=audio).float()
        floss, _tau = run.fake_update((sample, text, run.reference(x0), 0, audio))
        fake_grad = grad_sum(fake_params)
        optimizer.step()
        zero_grads(fake_params)
        loss, _sample, stats = run.generator_update(x0, text, exit_step=len(run.sigmas) - 2, audio=audio)
        gen_grad = grad_sum(gen_params)
        conv_grad = grad_sum(list(run.convkv.parameters()))
        ok = int(math.isfinite(float(loss)) and math.isfinite(float(floss)) and fake_grad > 0 and gen_grad > 0
                 and conv_grad > 0 and stats["match_rel"] < MATCH_BOUND)
        print(f"LOT_H3 kind=nf_check stage=2 audio={run.args.audio} frames={frames} latent_hw={tuple(x0.shape[-2:])} "
              f"sigmas={','.join(f'{s:.4f}' for s in run.sigmas)} loss={float(loss):.5g} fake_loss={float(floss):.5f} "
              f"gap={stats['gap']:.4g} match_rel={stats['match_rel']:.3g} exit_step={stats['exit_step']} gen_grad={gen_grad:.4g} conv_grad={conv_grad:.4g} "
              f"fake_grad={fake_grad:.4g} peak_mb={peak_mb(device):.0f} ok={ok}", flush=True)
    if not ok:
        raise SystemExit(1)


def peak_mb(device) -> float:
    return torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.device(device).type == "cuda" else 0.0


# ----------------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True,
                        help="1 = Neighbor Forcing training (eq. 4); 2 = ConvKV Memory + DMD step distillation")
    parser.add_argument("--cache", type=Path, default=LOT_H3_CLIPS_CACHE, help="H3 clip cache (latents + text)")
    parser.add_argument("--source", action="append", default=None,
                        help="keep clips whose source metadata starts with this (repeatable; default set_nk = Nikki; "
                             "'' keeps all)")
    parser.add_argument("--blocks", type=int, nargs=2, default=[FIRST_BLOCK, NEXT_BLOCK], metavar=("FIRST", "NEXT"),
                        help="latent frames of the first and of every later block (paper Table 5: 6 8)")
    parser.add_argument("--clip-blocks", type=int, default=3, help="blocks per training clip (3 = 22 latent frames)")
    parser.add_argument("--dir-init", type=Path, default=None,
                        help="Stage 2 with a DiR generator (not in the paper): a train_dir_h3.py run whose generator LoRA + "
                             "DirHead become the causal generator and whose fake LoRA + fake head become the fake score")
    parser.add_argument("--dir-tag", default="", help="checkpoint suffix of --dir-init, e.g. _50")
    parser.add_argument("--dir-ema", action="store_true", help="start from the *_ema generator files of --dir-init")
    parser.add_argument("--r-schedule", choices=["mod", "random"], default="mod", help="DiR: iteration trained at each update")
    parser.add_argument("--head-lr", type=float, default=None, help="DiR head learning rate (default: the DiR run's, else --lr)")
    parser.add_argument("--grad-clip", type=float, default=None, help="gradient-norm clip (default: the DiR run's, else off)")
    parser.add_argument("--ref", choices=["first", "none"], default=None,
                        help="reference image x_ref: the clip's first latent frame as H3's FL2VA first condition")
    parser.add_argument("--audio", choices=["silent", "driving"], default="silent",
                        help="audio rows: silent noise (this repo's convention), or the clip's clean H3 audio latent "
                             "as conditioning (the paper's c_audio; needs audio_latent.pt under LOT_H3_CLIPS_SOURCE)")
    parser.add_argument("--max-latent-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="cap the latent crop (a random window)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=None,
                        help="generator LoRA (default 1e-4; not given in the paper; with --dir-init: the DiR run's)")
    parser.add_argument("--dmd-draws", type=int, default=1,
                        help="DiR: (tau, eps) reads averaged per generator update (2 no-grad forwards each)")
    parser.add_argument("--conv-lr", type=float, default=None, help="ConvKV learning rate (default: --lr)")
    parser.add_argument("--fake-lr", type=float, default=None, help="default 1e-4, or the DiR run's")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank, generator and fake")
    parser.add_argument("--accum", type=int, default=1, help="gradient accumulation (batch size is 1)")
    parser.add_argument("--sigma-shift", type=float, default=1.0,
                        help="Stage 1 noise-level density: 1 = uniform (eq. 4), 12 = H3's own")
    parser.add_argument("--sigmas", default=",".join(str(s) for s in BASE_SIGMAS),
                        help="Stage 2 generator steps before the schedule shift (the release: 1, 0.75, 0.5)")
    parser.add_argument("--sampler-shift", type=float, default=VIDEO_SHIFT, help="schedule shift applied to --sigmas")
    parser.add_argument("--sampler", choices=["renoise", "euler"], default="renoise",
                        help="few-step update: renoise = the release's (clean estimate + fresh noise)")
    parser.add_argument("--tau-shift", default=None,
                        help="DMD noise-level density: a schedule shift (1 = uniform, the default) or beta:A (DiR; "
                             "default with --dir-init: the DiR run's)")
    parser.add_argument("--fake-updates", type=int, default=1, help="fake updates per generator update")
    parser.add_argument("--separate-v", action="store_true",
                        help="ConvKV: a second convolution for values (default: one for keys and values, eq. 6)")
    parser.add_argument("--init", type=Path, default=None, help="run dir to start from (Stage 2: the Stage 1 run)")
    parser.add_argument("--init-tag", default="")
    parser.add_argument("--swap", type=int, default=4, help="blocks streamed from CPU")
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=0, help="held-out eq. 4 loss every N steps (Stage 1; 0 = start/end)")
    parser.add_argument("--eval-items", type=int, default=2, help="held-out clips read by the DiR eval")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "train_nf_h3")
    parser.add_argument("--check", action="store_true", help="one dry update, nothing written")
    args = parser.parse_args()
    if not args.check and os.environ.get("LOT_H3_TRAIN") != "1":
        raise SystemExit("training is gated: set LOT_H3_TRAIN=1 to train (or --check for a dry pass)")
    from gpu_guard import refuse_if_busy

    refuse_if_busy("train_nf_h3.py")
    import train_h3
    from h3_musubi import load_dit, make_lora

    train_h3.MAX_LATENT_HW = tuple(args.max_latent_hw) if args.max_latent_hw else None
    train_h3.CROP_MULTIPLE = 2
    sources = ["set_nk"] if args.source is None else [s for s in args.source if s]
    dir_meta = None
    if args.dir_init is not None:
        if args.stage != 2 or args.init is not None or args.audio != "silent":
            raise SystemExit("--dir-init is a Stage 2 mode of its own: no --init, --audio silent")
        dir_meta = json.loads((args.dir_init / "config.json").read_text())
        if dir_meta.get("lot") or dir_meta.get("schedule", "dense") != "dense":
            raise SystemExit("--dir-init needs a dense DiR run")
    # A DiR generator was trained text-to-video, so it starts without a reference frame unless asked.
    args.ref = args.ref or ("none" if dir_meta else "first")
    args.tau_shift = args.tau_shift if args.tau_shift is not None else (dir_meta.get("tau_shift", 1.0) if dir_meta else 1.0)
    args.lr = args.lr if args.lr is not None else float((dir_meta or {}).get("lr", 1e-4))
    args.fake_lr = args.fake_lr if args.fake_lr is not None else float((dir_meta or {}).get("fake_lr", 1e-4))
    if args.stage == 2 and args.accum != 1:
        # The fake update regresses on the generator update's sample; with accumulation only the last of the
        # accumulated samples would reach it. Refused rather than silently trained that way.
        raise SystemExit("Stage 2 supports --accum 1 only")
    frames = NFConfig(args.blocks[0], args.blocks[1], memory=args.stage == 2).frames(args.clip_blocks)
    if args.stage == 2 and args.clip_blocks < 3:
        raise SystemExit("Stage 2 needs at least 3 blocks per clip: ConvKV is first used by the third block")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train_items, held_items = list_clips(args.cache, sources, frames, args.holdout)
    print(f"LOT_H3 kind=nf_data cache={args.cache} sources={sources} frames={frames} train={len(train_items)} "
          f"held={len(held_items)}", flush=True)

    dit = load_dit(device=device, swap=args.swap, train=True)
    head = fake_head = None
    fake_rank = args.rank
    if dir_meta is not None:
        from safetensors import safe_open

        from dir_h3 import DirFakeHead, DirHead

        suffix = args.dir_tag + ("_ema" if args.dir_ema else "")

        def lora_rank(path):
            with safe_open(str(path), "pt") as handle:
                key = next(k for k in handle.keys() if k.endswith("lora_down.weight"))
                return int(handle.get_slice(key).get_shape()[0])

        gen_file = args.dir_init / f"lora_gen{suffix}.safetensors"
        fake_file = args.dir_init / f"lora_fake{args.dir_tag}.safetensors"
        args.rank, fake_rank = lora_rank(gen_file), lora_rank(fake_file)
        gen_net, fake_net = make_lora(dit, args.rank), make_lora(dit, fake_rank)
        for net, path in ((gen_net, gen_file), (fake_net, fake_file)):
            if net.load_weights(str(path)).missing_keys:
                raise RuntimeError(f"{path}: missing LoRA keys")
        head = DirHead(int(dir_meta["hidden"]), int(dir_meta["temb_dim"]), int(dir_meta["iters"]))
        head.load_state_dict(torch.load(args.dir_init / f"dir_head{suffix}.pt", map_location="cpu", weights_only=True))
        fake_head = DirFakeHead(int(dir_meta["temb_dim"]), int(dir_meta["iters"]))
        fake_head.load_state_dict(torch.load(args.dir_init / f"dir_fake_head{args.dir_tag}.pt", map_location="cpu",
                                             weights_only=True))
        head, fake_head = head.to(device), fake_head.to(device)
        print(f"LOT_H3 kind=nf_dir_init from={args.dir_init} tag={suffix} rank={args.rank} fake_rank={fake_rank} "
              f"sigmas={dir_meta['sigmas']} mix_state={dir_meta.get('mix_state', False)}", flush=True)
    else:
        gen_net = make_lora(dit, args.rank)
        fake_net = make_lora(dit, args.rank) if args.stage == 2 else None
    convkv = ConvKV.for_dit(dit, shared=not args.separate_v).to(device) if args.stage == 2 else None
    if args.init is not None:
        info = gen_net.load_weights(str(args.init / f"lora_gen{args.init_tag}.safetensors"))
        if info.missing_keys:
            raise RuntimeError(f"--init LoRA is missing {len(info.missing_keys)} keys")
        conv_file = args.init / f"convkv{args.init_tag}.pt"
        if convkv is not None and conv_file.is_file():
            convkv.load_state_dict(torch.load(conv_file, map_location=device, weights_only=True))
        print(f"LOT_H3 kind=nf_init from={args.init}", flush=True)
    for net in (gen_net, fake_net):
        if net is not None:
            net.requires_grad_(True)
            net.to(device=device, dtype=torch.bfloat16)
    gen_groups, _ = gen_net.prepare_optimizer_params(args.lr)
    if convkv is not None:
        gen_groups = gen_groups + [{"params": list(convkv.parameters()), "lr": args.conv_lr or args.lr}]
    if head is not None:
        head_lr = args.head_lr or dir_meta.get("head_lr") or args.lr
        gen_groups = gen_groups + [{"params": [p for p in head.parameters() if p.requires_grad], "lr": head_lr}]
    gen_params = [p for group in gen_groups for p in group["params"]]
    fake_groups, fake_params = [], []
    if fake_net is not None:
        fake_groups, _ = fake_net.prepare_optimizer_params(args.fake_lr)
        if fake_head is not None:
            fake_groups = fake_groups + [{"params": list(fake_head.parameters()), "lr": args.fake_lr}]
        fake_params = [p for group in fake_groups for p in group["params"]]
    run = NFRun(dit, gen_net, fake_net, convkv, args, device, head=head, fake_head=fake_head, dir_meta=dir_meta)
    grad_clip = args.grad_clip if args.grad_clip is not None else float((dir_meta or {}).get("grad_clip", 0.0))
    betas = tuple((dir_meta or {}).get("betas", (0.9, 0.999)))

    if args.check:
        check(run, gen_params, fake_params, train_items, rng, device, frames)
        return

    gen_opt = torch.optim.AdamW(gen_groups, lr=args.lr, weight_decay=0.0, betas=betas)
    fake_opt = torch.optim.AdamW(fake_groups, lr=args.fake_lr, weight_decay=0.0, betas=betas) if fake_groups else None
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"paper": "arXiv 2603.11746", "stage": args.stage, "rank": args.rank, "blocks": list(args.blocks),
            "clip_blocks": args.clip_blocks, "frames": frames, "ref": args.ref, "audio": args.audio, "cache": str(args.cache),
            "sources": sources, "holdout": args.holdout, "ratio": 5, "shared_kv": not args.separate_v,
            "sigmas": run.sigmas, "sampler": args.sampler, "sigma_shift": args.sigma_shift, "lr": args.lr,
            "eval_items": args.eval_items,
            "steps": args.steps, "seed": args.seed, "max_latent_hw": args.max_latent_hw,
            "init": None if args.init is None else str(args.init), "fake_rank": fake_rank}
    if dir_meta is not None:
        meta.update({"dir": True, "dir_init": str(args.dir_init), "dir_tag": args.dir_tag, "dir_ema": args.dir_ema,
                     "iters": int(dir_meta["iters"]), "sigmas": run.dir_sigmas, "mix_state": run.mix_state,
                     "hidden": int(dir_meta["hidden"]), "temb_dim": int(dir_meta["temb_dim"]),
                     "fake_source": dir_meta.get("fake_source", "iteration"), "tau_shift": args.tau_shift,
                     "r_schedule": args.r_schedule, "grad_clip": grad_clip, "betas": list(betas), "sampler": "dir",
                     "tau_max": run.tau_max, "fake_lr": args.fake_lr, "fake_updates": args.fake_updates,
                     "dmd_draws": args.dmd_draws})
    log = open(args.out / "log.jsonl", "a")

    def save(tag: str) -> None:
        save_nf_run(args.out, tag, gen_network=gen_net, fake_network=fake_net, convkv=convkv, meta=meta,
                    head=head, fake_head=fake_head)

    def evaluate(step: int) -> None:
        if head is not None:
            result = run.dir_eval(held_items[:args.eval_items], frames)
            log.write(json.dumps({"step": step, "eval": result}) + "\n")
            log.flush()
            print(f"LOT_H3 kind=nf_dir_eval step={step} real_gap={result['real_gap']:.5f} "
                  f"real_gap_blocks={','.join(f'{g:.5f}' for g in result['real_gap_blocks'])} "
                  f"motion={result['motion']:.5f}", flush=True)
            return
        if args.stage != 1:
            return
        value = heldout_loss(run, held_items, device, frames)
        log.write(json.dumps({"step": step, "heldout_loss": value}) + "\n")
        log.flush()
        print(f"LOT_H3 kind=nf_eval step={step} heldout_loss={value:.5f}", flush=True)

    evaluate(0)
    run.loss_scale = 1.0 / args.accum
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        record = {"step": step}
        zero_grads(gen_params)
        for _ in range(args.accum):
            item = rng.choice(train_items)
            x0, text = load_clip(item, rng, device, frames)
            audio = load_audio(item, device, frames) if args.audio == "driving" else None
            if args.stage == 1:
                loss, stats = run.stage1_loss(x0, text, backward=True, audio=audio)
            elif head is not None:
                r = (step - 1) % len(run.dir_sigmas) if args.r_schedule == "mod" else rng.randrange(len(run.dir_sigmas))
                loss, sample, stats = run.dir_generator_update(x0, text, r)
            else:
                loss, sample, stats = run.generator_update(x0, text, audio=audio)
        record.update(stats)
        record["gen_grad"] = grad_sum(gen_params)
        if convkv is not None:
            record["conv_grad"] = grad_sum(list(convkv.parameters()))
        if grad_clip > 0:
            record["gen_norm"] = float(torch.nn.utils.clip_grad_norm_(gen_params, grad_clip))
        gen_opt.step()
        if args.stage == 2:
            for k in range(args.fake_updates):
                zero_grads(fake_params)
                if head is not None:
                    floss, ftau = run.dir_fake_update(sample, fresh=k)     # k > 0: a fresh rollout
                else:
                    if k:
                        with torch.no_grad():
                            _loss, sample, _stats = run.generator_update(x0, text, backward=False, audio=audio)
                    floss, ftau = run.fake_update(sample)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(fake_params, grad_clip)
                fake_opt.step()
            zero_grads(gen_params)
            record.update({"fake_loss": float(floss.detach()), "fake_tau": ftau})
        record["seconds"] = time.perf_counter() - started
        log.write(json.dumps(record) + "\n")
        log.flush()
        print("LOT_H3 kind=nf_train " + " ".join(
            f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
        if args.eval_every and step % args.eval_every == 0:
            evaluate(step)
        if args.save_every and step % args.save_every == 0:
            save(f"_{step}")
    if not (args.eval_every and args.steps % args.eval_every == 0):
        evaluate(args.steps)
    save("")
    print(f"LOT_H3 kind=nf_train_done stage={args.stage} steps={args.steps} out={args.out} "
          f"peak_mb={peak_mb(device):.0f} ok=1", flush=True)


if __name__ == "__main__":
    main()
