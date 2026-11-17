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
from lot_paths import LOT_H3_CLIPS_CACHE  # noqa: E402

from dir_h3 import use_networks  # noqa: E402
from h3_musubi import VIDEO_SHIFT, sample_sigma  # noqa: E402
from nf_h3 import (BASE_SIGMAS, DENSE, FIRST_BLOCK, NEXT_BLOCK, ConvKV, NFConfig, NoiseBank, few_step_sigmas,  # noqa: E402
                   nf_forward, nf_parallel_sample, save_nf_run)

TAU_MIN, TAU_MAX = 0.02, 0.98
SIGMA_MIN, SIGMA_MAX = 1e-3, 1.0 - 1e-3


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


# ---------------------------------------------------------------------------------- run

class NFRun:
    """Everything one update needs, shared by ``--check``, training and the CPU test."""

    def __init__(self, dit, gen_net, fake_net, convkv, args, device) -> None:
        self.dit, self.gen_net, self.fake_net, self.convkv = dit, gen_net, fake_net, convkv
        self.args, self.device = args, device
        self.causal = NFConfig(args.blocks[0], args.blocks[1], causal=True, memory=False)
        self.memory = NFConfig(args.blocks[0], args.blocks[1], causal=True, memory=True) if args.stage == 2 else None
        self.sigmas = few_step_sigmas([float(s) for s in args.sigmas.split(",")], args.sampler_shift)
        self.loss_scale = 1.0

    # ---- shared pieces
    def reference(self, x0: torch.Tensor):
        """``x_ref``: the clip's first latent frame (the VAE latent of its first pixel frame), or None."""
        return x0[:, :, :1].contiguous() if self.args.ref == "first" else None

    def draw_sigma(self) -> float:
        """Eq. 4: ``t ~ U(0, 1)`` (``--sigma-shift 1``), or under a schedule shift."""
        sigma = float(sample_sigma(float(self.args.sigma_shift), "cpu"))
        return min(max(sigma, SIGMA_MIN), SIGMA_MAX)

    def draw_tau(self) -> float:
        tau = float(sample_sigma(float(self.args.tau_shift), "cpu"))
        return min(max(tau, TAU_MIN), TAU_MAX)

    def dense_x0(self, x_tau, tau: float, text, first, seed: int) -> torch.Tensor:
        """A bidirectional model's clean estimate at ``tau`` (the caller sets which LoRAs are on)."""
        audio = torch.Generator(self.device).manual_seed(seed)
        out = nf_forward(self.dit, x_tau, 1.0 - tau, text, config=DENSE, first=first, audio_generator=audio)
        return x_tau.float() + tau * out.float()

    # ---- Stage 1
    def stage1_loss(self, x0, text, *, sigma: float | None = None, eps=None, config: NFConfig | None = None,
                    audio_seed: int | None = None):
        """Eq. 4 on one clip. Returns ``(loss, stats)``; the generator LoRA is on."""
        sigma = self.draw_sigma() if sigma is None else float(sigma)
        eps = torch.randn_like(x0) if eps is None else eps
        x_t = ((1.0 - sigma) * x0.float() + sigma * eps.float()).to(x0.dtype)          # eq. 2, every block at one step
        audio = None if audio_seed is None else torch.Generator(self.device).manual_seed(audio_seed)
        with use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
            velocity = nf_forward(self.dit, x_t, 1.0 - sigma, text, config=config or self.causal,
                                  first=self.reference(x0), audio_generator=audio)
        loss = F.mse_loss(velocity.float(), x0.float() - eps.float())
        return loss, {"loss": float(loss.detach()), "sigma": sigma}

    # ---- Stage 2
    def generator_step(self, x_s, audio_s, step: int, text, first) -> torch.Tensor:
        """Step ``step`` of the few-step generator on a whole clip: the clean estimate ``x0_hat``."""
        sigma = float(self.sigmas[step])
        velocity = nf_forward(self.dit, x_s, 1.0 - sigma, text, config=self.memory, convkv=self.convkv, first=first,
                              audio=audio_s)
        return x_s.float() + sigma * velocity.float()

    def generator_update(self, x0, text, *, exit_step: int | None = None, backward: bool = True):
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
                                              sampler=self.args.sampler, stop_step=step)
            x_s = x_s.to(x0.dtype)
            x0_hat = self.generator_step(x_s, audio_s, step, text, first)
        tau = self.draw_tau()
        eps = torch.randn_like(x0_hat)
        x_tau = ((1.0 - tau) * x0_hat + tau * eps).to(x0.dtype)
        with torch.no_grad():
            with use_networks(self.gen_net, self.fake_net, gen=False, fake=False):
                real = self.dense_x0(x_tau, tau, text, first, seed + 1)
            with use_networks(self.gen_net, self.fake_net, gen=False, fake=True):
                fake = self.dense_x0(x_tau, tau, text, first, seed + 1)
            weight = 1.0 / (x0_hat - real).abs().mean().clamp(min=1e-6)
            grad = (fake - real) * weight
            target = (x0_hat - grad).detach()
        with use_networks(self.gen_net, self.fake_net, gen=True, fake=False):
            live = self.generator_step(x_s, audio_s, step, text, first)
        loss = 0.5 * F.mse_loss(live, target)
        if backward:
            (loss * self.loss_scale).backward()
        stats = {"loss": float(loss.detach()), "tau": tau, "exit_step": step,
                 "match": float((live.detach() - x0_hat).abs().max()),
                 "gap": float((fake - real).abs().mean()), "x0_abs": float(x0_hat.abs().mean()),
                 "real_gap": float((x0_hat - real).abs().mean())}
        return loss, (x0_hat.detach(), text, first, seed), stats

    def fake_update(self, sample, *, backward: bool = True):
        """Flow regression of the fake model on the generator's (detached) sample."""
        x0_hat, text, first, seed = sample
        tau = self.draw_tau()
        eps = torch.randn_like(x0_hat)
        x_tau = (1.0 - tau) * x0_hat + tau * eps
        audio = torch.Generator(self.device).manual_seed(seed + 2)
        with use_networks(self.gen_net, self.fake_net, gen=False, fake=True):
            velocity = nf_forward(self.dit, x_tau, 1.0 - tau, text, config=DENSE, first=first, audio_generator=audio)
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
        for k, sigma in enumerate(sigmas):
            eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1000 * index + k)).to(x0)
            loss, _stats = run.stage1_loss(x0, text, sigma=sigma, eps=eps, audio_seed=7 + k)
            total, count = total + float(loss), count + 1
    return total / max(1, count)


def check(run: NFRun, gen_params, fake_params, items, rng, device, frames: int) -> None:
    """A dry pass: one update of each kind, nothing written."""
    x0, text = load_clip(items[0], rng, device, frames)
    if run.args.stage == 1:
        eps = torch.randn_like(x0)
        with torch.no_grad():
            dense, _ = run.stage1_loss(x0, text, sigma=0.6, eps=eps, config=DENSE, audio_seed=3)
            causal, _ = run.stage1_loss(x0, text, sigma=0.6, eps=eps, audio_seed=3)
        loss, _stats = run.stage1_loss(x0, text, sigma=0.6, eps=eps, audio_seed=3)
        loss.backward()
        gen_grad = grad_sum(gen_params)
        ok = int(math.isfinite(float(loss)) and gen_grad > 0 and abs(float(loss) - float(causal)) < 1e-2 * max(1.0, float(causal)))
        print(f"LOT_H3 kind=nf_check stage=1 frames={frames} latent_hw={tuple(x0.shape[-2:])} "
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
                                        sampler=run.args.sampler).float()
        floss, _tau = run.fake_update((sample, text, run.reference(x0), 0))
        fake_grad = grad_sum(fake_params)
        optimizer.step()
        zero_grads(fake_params)
        loss, _sample, stats = run.generator_update(x0, text, exit_step=len(run.sigmas) - 2)
        gen_grad = grad_sum(gen_params)
        conv_grad = grad_sum(list(run.convkv.parameters()))
        ok = int(math.isfinite(float(loss)) and math.isfinite(float(floss)) and fake_grad > 0 and gen_grad > 0
                 and conv_grad > 0 and stats["match"] < 5e-2)
        print(f"LOT_H3 kind=nf_check stage=2 frames={frames} latent_hw={tuple(x0.shape[-2:])} "
              f"sigmas={','.join(f'{s:.4f}' for s in run.sigmas)} loss={float(loss):.5g} fake_loss={float(floss):.5f} "
              f"gap={stats['gap']:.4g} match={stats['match']:.3g} gen_grad={gen_grad:.4g} conv_grad={conv_grad:.4g} "
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
    parser.add_argument("--ref", choices=["first", "none"], default="first",
                        help="reference image x_ref: the clip's first latent frame as H3's FL2VA first condition")
    parser.add_argument("--max-latent-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="cap the latent crop (a random window)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4, help="generator LoRA (not given in the paper)")
    parser.add_argument("--conv-lr", type=float, default=None, help="ConvKV learning rate (default: --lr)")
    parser.add_argument("--fake-lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank, generator and fake")
    parser.add_argument("--accum", type=int, default=1, help="gradient accumulation (batch size is 1)")
    parser.add_argument("--sigma-shift", type=float, default=1.0,
                        help="Stage 1 noise-level density: 1 = uniform (eq. 4), 12 = H3's own")
    parser.add_argument("--sigmas", default=",".join(str(s) for s in BASE_SIGMAS),
                        help="Stage 2 generator steps before the schedule shift (the release: 1, 0.75, 0.5)")
    parser.add_argument("--sampler-shift", type=float, default=VIDEO_SHIFT, help="schedule shift applied to --sigmas")
    parser.add_argument("--sampler", choices=["renoise", "euler"], default="renoise",
                        help="few-step update: renoise = the release's (clean estimate + fresh noise)")
    parser.add_argument("--tau-shift", type=float, default=1.0, help="DMD noise-level density (1 = uniform)")
    parser.add_argument("--fake-updates", type=int, default=1, help="fake updates per generator update")
    parser.add_argument("--separate-v", action="store_true",
                        help="ConvKV: a second convolution for values (default: one for keys and values, eq. 6)")
    parser.add_argument("--init", type=Path, default=None, help="run dir to start from (Stage 2: the Stage 1 run)")
    parser.add_argument("--init-tag", default="")
    parser.add_argument("--swap", type=int, default=4, help="blocks streamed from CPU")
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=0, help="held-out eq. 4 loss every N steps (Stage 1; 0 = start/end)")
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
    gen_params = [p for group in gen_groups for p in group["params"]]
    fake_groups, fake_params = [], []
    if fake_net is not None:
        fake_groups, _ = fake_net.prepare_optimizer_params(args.fake_lr)
        fake_params = [p for group in fake_groups for p in group["params"]]
    run = NFRun(dit, gen_net, fake_net, convkv, args, device)

    if args.check:
        check(run, gen_params, fake_params, train_items, rng, device, frames)
        return

    gen_opt = torch.optim.AdamW(gen_groups, lr=args.lr, weight_decay=0.0)
    fake_opt = torch.optim.AdamW(fake_groups, lr=args.fake_lr, weight_decay=0.0) if fake_groups else None
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"paper": "arXiv 2603.11746", "stage": args.stage, "rank": args.rank, "blocks": list(args.blocks),
            "clip_blocks": args.clip_blocks, "frames": frames, "ref": args.ref, "cache": str(args.cache),
            "sources": sources, "holdout": args.holdout, "ratio": 5, "shared_kv": not args.separate_v,
            "sigmas": run.sigmas, "sampler": args.sampler, "sigma_shift": args.sigma_shift, "lr": args.lr,
            "steps": args.steps, "seed": args.seed, "max_latent_hw": args.max_latent_hw,
            "init": None if args.init is None else str(args.init)}
    log = open(args.out / "log.jsonl", "a")

    def save(tag: str) -> None:
        save_nf_run(args.out, tag, gen_network=gen_net, fake_network=fake_net, convkv=convkv, meta=meta)

    def evaluate(step: int) -> None:
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
            x0, text = load_clip(rng.choice(train_items), rng, device, frames)
            if args.stage == 1:
                loss, stats = run.stage1_loss(x0, text)
                (loss * run.loss_scale).backward()
            else:
                loss, sample, stats = run.generator_update(x0, text)
        record.update(stats)
        record["gen_grad"] = grad_sum(gen_params)
        if convkv is not None:
            record["conv_grad"] = grad_sum(list(convkv.parameters()))
        gen_opt.step()
        if args.stage == 2:
            for k in range(args.fake_updates):
                zero_grads(fake_params)
                if k:
                    with torch.no_grad():
                        _loss, sample, _stats = run.generator_update(x0, text, backward=False)
                floss, ftau = run.fake_update(sample)
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
