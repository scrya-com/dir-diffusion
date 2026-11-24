#!/usr/bin/env python3
"""Posterior Matching Distillation (PMD) as the training objective of the pi-H3 2-step fast path. docs/PMD_PIH3.md.

PMD is by Hidir Yesiltepe, Julius Berner, Arash Vahdat and Pinar Yanardag (arXiv 2611.26629, code not released);
``pmd_h3.py`` is our port of its maths and is reused here unchanged (tilts, Tilt=Shift, Algorithm 1 / 2, the noise
mixing, the zero-init ``NoiseInput``, the cached-draw replay). pi-H3 is by Hansheng Chen, Shengqu Cai, Jiaming Song,
Gordon Wetzstein, Xinchen Yan and Leonidas Guibas. The fast path (``pih3_distill.fast_path``) is ours: 1 pi-H3 step
at half resolution, the learned latent x2, re-noise to ``sigma_1``, 1 pi-H3 step at full resolution, the DWD head.

What is trained: only the FULL-RESOLUTION step. The half-resolution step and the upsampler are frozen pi-H3 (no
LoRA there, no gradients), and they produce the on-policy state ``z_t`` at ``sigma_1`` (detached). No PMD loss on
the pure-noise half-resolution step (as in the paper).

* student (full-resolution step): pi-H3 (the adapter unmerged and switchable, ``pih3_distill.BypassAdapter``) + a
  trainable bypass LoRA + the zero-init ``NoiseInput`` on the raw ``z_t``; it reads ``z_t`` mixed with fresh noise
  ``u`` at ``--dial``, at the mixed level, so it is stochastic. H3-Tiny's blocks dropped with ``--drop h3tiny``.
* teacher: base H3 (the pi-H3 adapter switched OFF, no LoRA, no dropped blocks) at ``z_t + rho lambda``, one call per
  tilt, ``rho = sigma_1^2 / (1 - sigma_1)``. ``--teacher pih3`` keeps the adapter on instead. ``--teacher-device``
  loads base H3 a second time on another card and runs the teacher (and the discriminator's features) there.
* GAN term (``--gan-weight``, the paper's 0.75; off by default): ``pmd_h3``'s discriminator head on the frozen
  teacher's block ``--d-block`` features, ``--d-updates`` D updates per update on a fresh fast-path clip against a
  real latent, and the non-saturating G loss added to each draw's replay. docs/PMD_PIH3.md section 8.

    CUDA_VISIBLE_DEVICES="" python3 lot/test_pmd_pih3.py
    python3 lot/pmd_pih3.py train --data D --out D/pmd_a --drop h3tiny --check        # 3 updates: s/step, peak GB
    python3 lot/pmd_pih3.py probe --data D --out D/pmd_probe --drop h3tiny --scales 1 4 16   # are the tilts informative?
    python3 lot/pmd_pih3.py train --data D --out D/pmd_a --drop h3tiny --steps 1000 --stop-at 500 --stop-below-hf 0.9
    python3 lot/pmd_pih3.py eval --data D --out D/pmd_a/eval --drop h3tiny --ckpt D/pmd_a/pmd_500.pt
    python3 lot/pmd_pih3.py train ... --gan-weight 0.75 --real teacher [--teacher-device cuda:1]   # PMD + GAN, docs section 8

Every backward runs inside the student's switches (dropped blocks, LoRA on, adapter on): activation checkpointing
recomputes the blocks during backward, and a block that was skipped in the forward must be skipped again.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pmd_h3 as P  # noqa: E402

DEFAULT_DIAL = 0.30          # the paper's FLUX.2 dial of the first level after pure noise (a guess for sigma_1)
UP_DEFAULT = "/media/2TB/lora-data/h3_lot/pih3/cascade/latent_up_v2/latent_up.pt"


def _skip(_self, hidden, *_args, **_kwargs):
    return hidden


class FastPMD:
    """The fast path with a stochastic full-resolution step, and the PMD teacher, on ONE DiT.

    Three switches, always set together through ``switched`` (which restores the previous state on exit):
    ``drop`` (the ``drop`` blocks return their input), ``lora`` (the trainable bypass LoRA computed or not) and
    ``adapter`` (the pi-H3 adapter on or off; needs a ``BypassAdapter``).

    * ``student_mode()``: drop on, LoRA on, adapter on. The whole update runs inside it, the backward included.
    * the half-resolution step: the same with the LoRA off (``lora_lr_step`` keeps it on).
    * the teacher: drop off, LoRA off, adapter off (``teacher="pih3"``: adapter on). No gradients ever.

    ``forward`` is ``h3_musubi.forward_video`` (a toy forward in the tests)."""

    def __init__(self, dit, up, *, sigmas, dial: float, noise_input: P.NoiseInput, lora=None, adapter=None,
                 drop=(), forward=None, teacher: str = "base", lora_lr_step: bool = False, teacher_dit=None):
        if teacher not in ("base", "pih3"):
            raise ValueError(f"teacher {teacher!r}: base or pih3")
        if teacher_dit is not None and teacher != "base":
            raise ValueError("a separate teacher DiT is base H3 (no adapter): --teacher pih3 needs the shared DiT")
        self.teacher_dit = teacher_dit
        if not 0.0 < float(sigmas[1]) < 1.0:
            raise ValueError("the full-resolution step's sigma must be inside (0, 1)")
        self.dit, self.up = dit, up
        self.s0, self.s1 = float(sigmas[0]), float(sigmas[1])
        self.dial, self.noise_input = float(dial), noise_input
        self.lora_modules = list(lora.unet_loras) if lora is not None else []
        self.adapter, self.teacher_kind, self.lora_lr_step = adapter, teacher, bool(lora_lr_step)
        self.drop_blocks = [dit.blocks[i] for i in drop]
        if forward is None:
            from h3_musubi import forward_video as forward
        self.forward = forward
        self.state = {"drop": False, "lora": True, "adapter": True}
        self._apply(self.state)

    @property
    def device(self):
        return self.dit.video_patch_proj.weight.device

    @property
    def teacher_device(self):
        """Where the teacher and the discriminator run: the student's card, or ``--teacher-device``'s."""
        dit = self.dit if self.teacher_dit is None else self.teacher_dit
        return dit.video_patch_proj.weight.device

    def feature_view(self):
        """The frozen teacher as ``pmd_h3.teacher_features`` reads a model (``dit``, ``forward``,
        ``teacher_modes()``, ``_audio(seed)`` -> a generator): the shared DiT under the teacher's switches, or the
        separate teacher DiT as it is. Every feature call gets H3's silent audio from a seeded generator, real and
        generated clips alike (the carried audio ``ya`` would tell D which is which)."""
        separate = self.teacher_dit is not None
        dit = self.teacher_dit if separate else self.dit
        dev = self.teacher_device
        return types.SimpleNamespace(dit=dit, forward=self.forward,
                                     teacher_modes=contextlib.nullcontext if separate else self.teacher_mode,
                                     _audio=lambda seed: torch.Generator(dev).manual_seed(int(seed)))

    # ------------------------------------------------------------------ switches

    def _apply(self, state: dict) -> None:
        if not state["adapter"] and self.adapter is None:
            raise RuntimeError("the pi-H3 adapter is merged: it cannot be switched off (load it with BypassAdapter)")
        for block in self.drop_blocks:
            if state["drop"]:
                block.forward = types.MethodType(_skip, block)
            else:
                block.__dict__.pop("forward", None)
        for module in self.lora_modules:
            module.enabled = bool(state["lora"])
        if self.adapter is not None:
            self.adapter.set(bool(state["adapter"]))
        self.state = dict(state)

    @contextlib.contextmanager
    def switched(self, **changes):
        previous = dict(self.state)
        self._apply({**previous, **changes})
        try:
            yield self
        finally:
            self._apply(previous)

    def student_mode(self):
        return self.switched(drop=True, lora=True, adapter=True)

    def teacher_mode(self):
        return self.switched(drop=False, lora=False, adapter=self.teacher_kind == "pih3")

    # ------------------------------------------------------------------ the three calls

    def _audio(self, ya: torch.Tensor) -> dict:
        """The carried audio ``ya`` as ``pih3_distill.fast_path`` hands it to the full-resolution step."""
        from h3_musubi import audio_sigma

        sa = float(audio_sigma(self.s1))
        return {"audio": ya * (sa / self.s1), "t_audio": 1.0 - sa}

    @torch.no_grad()
    def lr_state(self, text, z, z_lr, za) -> tuple[torch.Tensor, torch.Tensor]:
        """The on-policy state: ``pih3_distill.fast_path`` up to the full-resolution step's input. ``(z_t, ya)``:
        the upsampled half-resolution estimate re-noised with ``z`` to ``sigma_1`` and the carried audio. Frozen."""
        from h3_musubi import audio_sigma
        from pih3_h3 import carried_audio_step

        s0, s1 = self.s0, self.s1
        x, y = s0 * z_lr, s0 * za
        sa = float(audio_sigma(s0))
        store: dict = {}
        with self.switched(lora=self.state["lora"] and self.lora_lr_step):
            v = self.forward(self.dit, x, 1.0 - s0, text, audio=y * (sa / s0), t_audio=1.0 - sa, out=store).float()
        est = x + s0 * v
        ya = carried_audio_step(y, store["audio"].float(), s0, s1)
        dev = self.device
        with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
            est_hr = self.up(est).float()
        return (s1 * z + (1.0 - s1) * est_hr).detach(), ya.detach()

    def student(self, z_t, u, text, ya, out: dict | None = None, plain: bool = False) -> torch.Tensor:
        """One stochastic draw of the clean latent: ``x_hat = y + sigma_tilde * v`` on the mixed latent ``y``
        (``pmd_h3.mix_noise``), the raw ``z_t`` through ``NoiseInput``. Runs in whatever mode is set (the caller
        holds ``student_mode()`` across the forward AND the backward). ``out`` gets the DWD head's inputs.
        ``plain``: no mixing, no ``NoiseInput`` (``fast_path``'s own deterministic full-resolution step)."""
        store = {} if out is None else out
        if plain:
            y, sigma_t, extra = z_t.float(), self.s1, None
        else:
            y, sigma_t = P.mix_noise(z_t.float(), u, self.s1, self.dial)
            extra = self.noise_input(z_t)
        v = self.forward(self.dit, y, 1.0 - sigma_t, text, video_extra=extra, out=store, **self._audio(ya)).float()
        return y.float() + sigma_t * v

    @torch.no_grad()
    def teacher(self, z_shifted, text, ya) -> torch.Tensor:
        """The frozen teacher's posterior-mean estimate ``D(z) = z + sigma_1 * v`` (base H3 by default). With a
        separate teacher DiT the inputs go to its card and the estimate comes back."""
        if self.teacher_dit is not None:
            dev = self.teacher_device
            audio = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in self._audio(ya).items()}
            v = self.forward(self.teacher_dit, z_shifted.to(dev), 1.0 - self.s1, text.to(dev), **audio).float()
            return z_shifted.float() + self.s1 * v.to(z_shifted.device)
        with self.teacher_mode():
            v = self.forward(self.dit, z_shifted, 1.0 - self.s1, text, **self._audio(ya)).float()
        return z_shifted.float() + self.s1 * v

    @torch.no_grad()
    def sample(self, text, z, z_lr, za, u, baseline: bool = False) -> dict:
        """The whole fast path with the stochastic full-resolution step (inference). ``latent``, ``h``, ``temb``
        as ``pih3_distill.fast_path`` returns them. ``baseline``: LoRA off, no mixing, no ``NoiseInput``: it IS
        ``fast_path`` without a LoRA (the untrained row of the gate table)."""
        with self.switched(drop=True, lora=not baseline, adapter=True):
            z_t, ya = self.lr_state(text, z, z_lr, za)
            store: dict = {}
            latent = self.student(z_t, u, text, ya, out=store, plain=baseline)
        return {"latent": latent, "h": store.get("hidden"), "temb": store.get("temb_video"), "z_t": z_t, "ya": ya}


def pmd_update(model: FastPMD, text, z, z_lr, za, *, tilts_n: int, draws_n: int, generator: torch.Generator,
               factor=(4, 8, 8), radius=(0.25, 0.5), estimator: str = "jk", scale: float = 1.0, extra_loss=None):
    """One PMD loss on the full-resolution step, gradients accumulated into the LoRA and ``NoiseInput``.

    The on-policy ``z_t`` (frozen half-resolution step, detached), ``tilts_n`` coarse tilts, one teacher call per
    tilt at ``z_t + rho lambda``, ``draws_n`` student draws, ``pmd_h3.pmd_step`` (Algorithm 2 by default, replayed
    draw by draw). Everything, the backward included, runs inside ``student_mode()``. ``extra_loss(clean, m)``
    (the GAN term, ``gan_extra``) is added to draw ``m``'s replay; ``None`` is plain PMD."""
    with model.student_mode():
        z_t, ya = model.lr_state(text, z, z_lr, za)
        tilts = P.sample_tilts(z_t, tilts_n, factor=tuple(factor), radius=tuple(radius), generator=generator)
        noises = [torch.randn(z_t.shape, device=z_t.device, generator=generator) for _ in range(draws_n)]
        loss, stats = P.pmd_step(lambda s, u, _k: model.student(s, u, text, ya), lambda s: model.teacher(s, text, ya),
                                 z_t, model.s1, 1, tilts, noises, estimator=estimator, scale=scale,
                                 extra_loss=extra_loss)
    return loss, stats


# ------------------------------------------------------------------------------------------- the GAN term

def disc_features(model: FastPMD, clean, text, *, block: int, generator: torch.Generator, audio_seed: int):
    """``pmd_h3.teacher_features`` on this trainer's frozen teacher: ``clean`` (moved to the teacher's card) noised
    to a level ``k / 1000`` (``pmd_h3.gan_level``), block ``block``'s video rows. ``generator`` lives on the
    teacher's card. Keeps the graph to ``clean``; the caller decides under which switches the backward runs."""
    dev = model.teacher_device
    view = model.feature_view()
    return P.teacher_features(view, clean.to(dev), P.gan_level(generator, dev), text.to(dev), block=block,
                              generator=generator, audio_seed=audio_seed)


def g_term(model: FastPMD, disc, clean, text, *, weight: float, block: int, generator: torch.Generator,
           audio_seed: int) -> tuple[torch.Tensor, float]:
    """``weight * g_loss(D(features(clean)))`` as a surrogate whose gradient w.r.t. ``clean`` is the G loss's.

    The gradient is taken HERE, under the teacher's switches: H3 checkpoints activations, so a backward after
    ``teacher_mode()`` has restored the student's switches would recompute the teacher's blocks with the LoRA and
    the adapter on and the blocks dropped (``test_backward_under_other_switches_is_wrong``). The returned surrogate
    ``(clean * dG/dclean).sum()`` (valued at the G loss) then goes back through the student's replay under the
    student's switches. D's parameters get no gradient (only ``clean``'s is asked for). Returns
    ``(surrogate, |dG/dclean|)``."""
    view = model.feature_view()
    leaf = clean.detach().to(model.teacher_device).float().requires_grad_(True)
    with torch.enable_grad(), view.teacher_modes():
        f = disc_features(model, leaf, text, block=block, generator=generator, audio_seed=audio_seed)
        g = weight * P.g_loss(disc(f))
        (grad,) = torch.autograd.grad(g, leaf)
    grad = grad.to(clean.device)
    surrogate = (clean.float() * grad).sum()
    return surrogate + (g.detach().to(clean.device) - surrogate.detach()), float(grad.norm())


def gan_extra(model: FastPMD, disc, text, *, weight: float, draws_n: int, block: int, generator: torch.Generator,
              rng: random.Random, log: list | None = None):
    """``pmd_update``'s ``extra_loss``: the G term on each replayed draw, ``weight / draws_n`` each (``pmd_h3``'s
    split). ``log`` collects ``|dG/dx|`` per draw."""
    def extra(clean, _m):
        value, norm = g_term(model, disc, clean, text, weight=weight / draws_n, block=block, generator=generator,
                             audio_seed=rng.randrange(1 << 30))
        if log is not None:
            log.append(norm)
        return value

    return extra


def d_step(model: FastPMD, disc, d_opt, fake, text, real, real_text, *, block: int, generator: torch.Generator,
           rng: random.Random, clip: float) -> float:
    """One discriminator update (``pmd_h3.d_loss``) on a generated clean latent and a real one, features detached."""
    with torch.no_grad():
        f_fake = disc_features(model, fake, text, block=block, generator=generator, audio_seed=rng.randrange(1 << 30))
        f_real = disc_features(model, real, real_text, block=block, generator=generator,
                               audio_seed=rng.randrange(1 << 30))
    loss = P.d_loss(disc(f_real), disc(f_fake))
    d_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(disc.parameters(), clip)
    d_opt.step()
    return float(loss)


@torch.no_grad()
def probe_state(model: FastPMD, text, z, z_lr, za, *, scales, tilts_n: int, draws_n: int,
                generator: torch.Generator, factor=(4, 8, 8), radius=(0.25, 0.5), estimator: str = "jk") -> list[dict]:
    """No training: how informative the tilts are at one on-policy state, per radius multiplier in ``scales``.

    The same tilt directions and the same ``draws_n`` student draws serve every multiplier. Per multiplier:
    ``ess`` / ``w_max`` (softmax weights over the draws: ``draws_n`` = flat = mean matching, 1 = one-hot),
    ``teacher_shift`` (``|D(z + rho lambda) - D(z)| / |D(z)|``), ``shift_ratio`` (``|rho lambda| / |z_t|``),
    ``gain`` (``|D(z + rho lambda) - D(z)| / |rho lambda|``: the teacher's response per unit of input shift; it
    should stay about constant while the teacher is in its linear, sane range), ``std_ratio`` (std of the shifted
    target over std of ``D(z)``: far from 1 = the teacher left its range), ``tilt_over_mean`` (the tilt part of
    the target against the student's mean residual, ``|D(z + rho lambda) - D(z)| / |mean(x) - D(z)|``) and the
    two losses. ``draw_spread`` is the draws' spread over their mean (the same for every multiplier)."""
    rows = []
    with model.student_mode():
        z_t, ya = model.lr_state(text, z, z_lr, za)
        base = P.sample_tilts(z_t, tilts_n, factor=tuple(factor), radius=tuple(radius), generator=generator)
        noises = [torch.randn(z_t.shape, device=z_t.device, generator=generator) for _ in range(draws_n)]
        draws = torch.stack([model.student(z_t, u, text, ya).float()[0] for u in noises])
        d0 = model.teacher(z_t, text, ya).float()
        rho = P.rho_of(model.s1)
        mean = draws.mean(0)
        spread = float((draws - mean).pow(2).sum(dim=tuple(range(1, draws.ndim))).mean().sqrt()
                       / mean.norm().clamp_min(1e-12))
        mean_gap = float((mean - d0[0]).norm())
        for scale in scales:
            tilts = [float(scale) * t for t in base]
            targets = [model.teacher(z_t + rho * t, text, ya).float() for t in tilts]
            stats = P.weight_stats(draws, [t[0] for t in tilts])
            moved = [float((target - d0).norm()) for target in targets]
            count = len(tilts)
            rows.append({
                "scale": float(scale), "ess": stats["ess"], "w_max": stats["w_max"], "draw_spread": spread,
                "teacher_shift": sum(moved) / count / float(d0.norm().clamp_min(1e-12)),
                "shift_ratio": sum(float(rho * t.norm()) for t in tilts) / count / float(z_t.norm().clamp_min(1e-12)),
                "gain": sum(m / float(rho * t.norm()) for m, t in zip(moved, tilts)) / count,
                "std_ratio": sum(float(target.std() / d0.std().clamp_min(1e-12)) for target in targets) / count,
                "tilt_over_mean": sum(moved) / count / max(mean_gap, 1e-12),
                "loss": float(P.pmd_loss(draws, [t[0] for t in tilts], [t[0] for t in targets], estimator)),
                "loss_plain": float(P.pmd_loss(draws, [t[0] for t in tilts], [t[0] for t in targets], "plain")),
            })
    return rows


# ------------------------------------------------------------------------------------------- GPU: build, eval

def build(args, *, train: bool):
    """``(model, lora network, DWD head)``. ``adapter=bypass``: base H3 + the switchable pi-H3 adapter (what PMD
    trains on: the teacher needs it off). ``adapter=merged``: the int8 pi-H3 adapter merged (deployment; no teacher)."""
    from h3_musubi import load_dit, make_lora
    from pih3_cascade import parse_drop
    from pih3_distill import BypassAdapter, load_student, load_up_net
    from pih3_h3 import PiH3Head, adapter_path, pih3_sigmas

    if args.adapter == "bypass":
        dit = load_dit(device="cuda", swap=0, train=train)
        adapter = BypassAdapter(dit, adapter_path(2))
        network = make_lora(dit, args.rank)
        network.to("cuda", torch.float32)
        head = PiH3Head(adapter.dwd, dit.final_layer.norm.eps)
    else:
        if train:
            raise SystemExit("training needs --adapter bypass (the teacher is base H3: the adapter switched off)")
        dit, network, head = load_student(rank=args.rank, lora=None, train=False)
        adapter = None
    network.requires_grad_(train)
    proj = dit.video_patch_proj
    hidden = getattr(proj, "out_features", None) or proj.weight.shape[0]
    noise_input = P.NoiseInput(hidden, P.TOKEN_DIM).to("cuda")
    noise_input.requires_grad_(train)
    teacher_dit = None
    if getattr(args, "teacher_device", None):
        if args.adapter != "bypass" or getattr(args, "teacher", "base") != "base":
            raise SystemExit("--teacher-device loads base H3 as the teacher: needs --adapter bypass, --teacher base")
        # a second base H3, no LoRA, no adapter, all blocks; checkpointed when the GAN term back-propagates through it
        teacher_dit = load_dit(device=args.teacher_device, swap=0, train=train and getattr(args, "gan_weight", 0) > 0)
        teacher_dit.requires_grad_(False)
    model = FastPMD(dit, load_up_net(args.up), sigmas=pih3_sigmas(2), dial=args.dial, noise_input=noise_input,
                    lora=network, adapter=adapter, drop=parse_drop(args.drop),
                    teacher=getattr(args, "teacher", "base"), lora_lr_step=args.lora_lr_step, teacher_dit=teacher_dit)
    return model, network, head


def save_ckpt(path: Path, model: FastPMD, network, args, step: int, opt=None, disc=None, d_opt=None) -> None:
    """``disc`` (the GAN term's head) is saved whenever given; ``d_opt`` only next to ``opt`` (``pmd_last.pt``)."""
    state = {"lora": {k: v.detach().cpu() for k, v in network.state_dict().items()},
             "noise_input": {k: v.detach().cpu() for k, v in model.noise_input.state_dict().items()},
             "rank": args.rank, "step": step, "drop": args.drop, "up": str(args.up), "dial": model.dial,
             "adapter": "bypass", "lora_lr_step": model.lora_lr_step, "method": "pmd (arXiv 2611.26629)"}
    if opt is not None:
        state["opt"] = opt.state_dict()
    if disc is not None:
        state["disc"] = {k: v.detach().cpu() for k, v in disc.state_dict().items()}
        state["gan_weight"] = float(getattr(args, "gan_weight", 0.0))
        if opt is not None and d_opt is not None:
            state["d_opt"] = d_opt.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_ckpt(path: Path, model: FastPMD, network) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if network.load_state_dict(saved["lora"], strict=False).unexpected_keys:
        raise SystemExit(f"{path}: LoRA keys that are not in this network (rank differs?)")
    model.noise_input.load_state_dict(saved["noise_input"])
    return saved


def eval_noise(shape, seed: int, device) -> torch.Tensor:
    """The student's noise input ``u`` for a held-out clip: fixed by the clip's seed (``offset`` 2, 3, ...)."""
    return torch.randn(shape, device=device, generator=torch.Generator(device).manual_seed(int(seed)))


@torch.no_grad()
def held_eval(model: FastPMD, head, lp, items, sheet_path: Path | None = None, sheets: int = 2,
              baseline: bool = False) -> dict:
    """The gate's numbers on held-out teacher clips, as ``pih3_distill.held_eval`` computes them (same clips, same
    paired noise, LPIPS of the whole clip against the teacher's DWD pixels, ``hf`` as a fraction of the teacher's,
    latent MSE), plus ``div`` (relative RMS distance between two draws ``u`` of the same clip's latent) and
    ``seconds`` (median end to end, first clip skipped; in the training process, so a rough number).

    ``baseline``: LoRA off, dial 0: the untrained fast path (``pih3_distill``'s step-0 row)."""
    from accel_h3 import lpips_clip
    from pih3_cascade import hf_energy, sheet
    from pih3_distill import paired_noise, u8
    from pih3_h3 import audio_noise
    from train_dwd_h3 import load_text

    dit = model.dit
    was = dit.training
    dit.eval()
    rows = {"lpips": [], "hf": [], "lat": [], "mean_diff": [], "std_ratio": [], "div": [], "seconds": []}
    try:
        for path in items:
            item = torch.load(path, weights_only=False)
            width, height = item["size"]
            latent_t = item["latent_t"]
            text = load_text(item["te"]).cuda()
            z, z_lr = paired_noise(latent_t, height, width, item["seed"])
            za = audio_noise(latent_t, item["seed"] + 1)
            u = eval_noise(z.shape, item["seed"] + 2, z.device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = model.sample(text, z, z_lr, za, u, baseline=baseline)
            px = head.decode(out["h"], out["temb"], (latent_t, height // 32, width // 32))
            torch.cuda.synchronize()
            rows["seconds"].append(time.perf_counter() - start)
            px = px.float()
            ref = item["pixels"].cuda().permute(3, 0, 1, 2).float().div(127.5).sub(1)[None]
            rows["lpips"].append(lpips_clip(lp, px, ref))
            rows["hf"].append(hf_energy(px) / hf_energy(ref))
            # colour and contrast against the teacher, as pih3_distill.held_eval: |mean difference| per RGB channel
            # ([-1, 1] units) and the ratio of the standard deviations (1.0 = the teacher's contrast)
            rows["mean_diff"].append(float((px.mean(dim=(0, 2, 3, 4)) - ref.mean(dim=(0, 2, 3, 4))).abs().mean()))
            rows["std_ratio"].append(float(px.std(dim=(0, 2, 3, 4)).mean() / ref.std(dim=(0, 2, 3, 4)).mean()))
            rows["lat"].append(float(torch.nn.functional.mse_loss(out["latent"], item["latent"].cuda().float())))
            if not baseline:
                with model.student_mode():
                    other = model.student(out["z_t"], eval_noise(z.shape, item["seed"] + 3, z.device), text,
                                          out["ya"])
                rows["div"].append(float((other - out["latent"]).pow(2).mean().sqrt()
                                         / out["latent"].pow(2).mean().sqrt().clamp_min(1e-8)))
            if sheet_path is not None and len(rows["lat"]) <= sheets:
                sheet({"teacher 4-step": item["pixels"], "baseline" if baseline else "PMD student": u8(px)},
                      sheet_path.with_name(f"{sheet_path.stem}_p{len(rows['lat']) - 1}.jpg"), columns=4, tile_w=320)
    finally:
        dit.train(was)
    return summarise(rows)


def summarise(rows: dict) -> dict:
    out = {}
    for key, values in rows.items():
        if not values:
            continue
        if key == "seconds":
            tail = sorted(values[1:]) or values
            out[key] = round(tail[len(tail) // 2], 3)
        else:
            out[key] = round(sum(values) / len(values), 4)
    return out


def _prompts(args):
    from accel_h3 import default_caches, prompt_lists

    return prompt_lists(default_caches(), args.holdout)[0][:args.prompts]


def _load_text(path):
    from train_dwd_h3 import load_text

    return load_text(str(path)).cuda()


def _noise(shape_args, seed: int):
    from pih3_distill import paired_noise
    from pih3_h3 import audio_noise

    latent_t, height, width = shape_args
    z, z_lr = paired_noise(latent_t, height, width, seed)
    return z, z_lr, audio_noise(latent_t, seed + 1)


def _guard() -> None:
    from pih3_cascade import refuse_if_card_busy

    refuse_if_card_busy(4.0)


def _evaluator(args, model, head):
    import lpips

    folder = args.data / args.folder
    held = sorted((folder / "held").glob("*.pt"))[:args.eval_held]
    if not held:
        raise SystemExit(f"no held-out teacher clips in {folder}/held")
    lp = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)

    def run(step: int, baseline: bool = False) -> dict:
        name = f"sheet_{step}" + ("_baseline" if baseline else "")
        return held_eval(model, head, lp, held, sheet_path=args.out / f"{name}.jpg", baseline=baseline)

    return run


def peak_gb(device) -> float:
    device = torch.device(device)
    return round(torch.cuda.max_memory_allocated(device) / 2**30, 1) if device.type == "cuda" else 0.0


def _latent_shape(path) -> tuple[int, ...]:
    from safetensors import safe_open

    with safe_open(str(path), "pt") as handle:
        key = next(k for k in handle.keys() if k.startswith("latent"))
        return tuple(handle.get_slice(key).get_shape())


def _real(args, shape_args, device):
    """The GAN term's real latents at exactly ``(1, 24, latent_frames, H/16, W/16)``, on ``device``. Returns
    ``draw(rng) -> (latent, text)``.

    ``--real cache`` (default): ``pmd_h3``'s source, the training split of the prompt caches (``--real-cache``,
    default ``accel_h3.default_caches()``: ``LOT_H3_STILLS_CACHE`` and ``LOT_H3_CLIPS_CACHE``), read with
    ``train_dir_h3.load_item`` (a random spatial window, a time window starting on a VAE clip start). Only items that
    crop to the exact canvas are kept: clips with at least ``latent_frames`` frames, and stills (repeated over time,
    as ``pmd_h3`` does) only with ``--real-stills``. Refuses when none fit.
    ``--real teacher``: the pi-H3 4-step clips in ``DATA/FOLDER/train`` (GENERATED, not real data)."""
    latent_t, height, width = shape_args
    want = (1, 24, latent_t, height // 16, width // 16)
    if args.real == "teacher":
        files = sorted((args.data / args.folder / "train").glob("*.pt"))
        if not files:
            raise SystemExit(f"--real teacher: no clips in {args.data / args.folder / 'train'}")

        def draw_teacher(rng):
            item = torch.load(rng.choice(files), map_location="cpu", weights_only=False)
            if tuple(item["latent"].shape) != want:
                raise SystemExit(f"--real teacher: latent {tuple(item['latent'].shape)}, the canvas is {want}")
            return item["latent"].to(device, torch.float32), _load_text(item["te"]).to(device)

        print(f"LOT_H3 kind=pmd_pih3_real source=teacher clips={len(files)} generated=1", flush=True)
        return draw_teacher
    import train_h3
    from accel_h3 import default_caches
    from train_dir_h3 import load_item

    train_h3.MAX_LATENT_HW = want[3:]
    train_h3.CROP_MULTIPLE = 2
    caches = [Path(c) for c in args.real_cache] if args.real_cache else default_caches()
    clips, stills, skipped = [], [], {}
    for cache in caches:
        for item in train_h3.list_items(cache, args.holdout)[0]:
            shape = _latent_shape(item[0])
            t, h, w = (1,) + tuple(shape[1:]) if len(shape) == 3 else shape[1:]
            if h < want[3] or w < want[4]:
                key = f"{h}x{w}"
                skipped[key] = skipped.get(key, 0) + 1
            elif t >= latent_t:
                clips.append(item)
            elif t == 1:
                stills.append(item)
    items = clips + (stills if args.real_stills else [])
    print(f"LOT_H3 kind=pmd_pih3_real source=cache caches={[str(c) for c in caches]} clips={len(clips)} "
          f"stills={len(stills)} used={len(items)} no_fit_hw={skipped}", flush=True)
    if not items:
        raise SystemExit(f"no real latents crop to {want[2:]} (T, H, W): {len(clips)} clips, {len(stills)} stills "
                         f"(--real-stills repeats stills over time), H x W that do not fit: {skipped}. docs/PMD_PIH3.md section 8")

    def draw_cache(rng):
        latent, text = load_item(rng.choice(items), rng, device, latent_t)
        if tuple(latent.shape) != want:
            raise SystemExit(f"real latent {tuple(latent.shape)}, the canvas is {want}")
        return latent, text

    return draw_cache


# swappable for the CPU smoke test (test_pmd_pih3.py)
HOOKS = {"guard": _guard, "build": build, "prompts": _prompts, "load_text": _load_text, "noise": _noise,
         "evaluator": _evaluator, "real": _real}


def train(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        from setproctitle import setproctitle

        setproctitle(f"pih3:pmd r{args.rank} {args.drop or 'full'} {args.out.name}"[:60])
    HOOKS["guard"]()
    width, height = P.parse_size(args.size)
    shape_args = (args.latent_frames, height, width)
    prompts = HOOKS["prompts"](args)
    if not prompts:
        raise SystemExit("no training prompts")
    model, network, head = HOOKS["build"](args, train=True)
    dev = model.device
    lora_params = [p for p in network.parameters() if p.requires_grad]
    noise_params = list(model.noise_input.parameters())
    opt = torch.optim.AdamW([{"params": lora_params, "lr": args.lr},
                             {"params": noise_params, "lr": args.noise_lr or args.lr}],
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False)
    base_lrs = [group["lr"] for group in opt.param_groups]
    gan_weight = float(getattr(args, "gan_weight", 0.0))
    disc = d_opt = real = None
    tdev = model.teacher_device
    if gan_weight > 0:
        proj = (model.dit if model.teacher_dit is None else model.teacher_dit).video_patch_proj
        disc = P.DiscHead(getattr(proj, "out_features", None) or proj.weight.shape[0]).to(tdev)
        d_opt = torch.optim.AdamW(disc.parameters(), lr=args.d_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        real = HOOKS["real"](args, shape_args, tdev)
    step0 = 0
    if args.resume:
        saved = load_ckpt(args.resume, model, network)
        step0 = saved["step"]
        if "opt" in saved:
            opt.load_state_dict(saved["opt"])
        if disc is not None:
            if saved.get("disc") is not None:
                disc.load_state_dict(saved["disc"])
                if saved.get("d_opt") is not None:
                    d_opt.load_state_dict(saved["d_opt"])
            else:
                print(f"LOT_H3 kind=pmd_pih3_gan_fresh_d resume={args.resume}: no discriminator in it", flush=True)
    rng = random.Random(args.seed + step0)
    gen = torch.Generator(dev).manual_seed(args.seed + step0)
    # the GAN term's own randomness, so that with it off nothing above changes, and with it on the PMD half of
    # each update draws exactly what it draws without it
    gan_rng = random.Random(args.seed + step0 + 7_000_003)
    gan_gen = torch.Generator(tdev).manual_seed(args.seed + step0 + 7_000_003)
    log = open(args.out / "log.jsonl", "a")
    (args.out / "config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items() if k != "func"}, "sigmas": [model.s0, model.s1],
         "rho": P.rho_of(model.s1), "paper": "arXiv 2611.26629"}, indent=1) + "\n")
    evaluator = None if args.check or not args.eval_every else HOOKS["evaluator"](args, model, head)

    def emit(kind: str, row: dict) -> None:
        print(f"LOT_H3 kind={kind} " + json.dumps(row), flush=True)
        log.write(json.dumps({"kind": kind, **row}) + "\n")
        log.flush()

    def run_eval(step: int) -> dict:
        start = time.time()
        row = {"step": step, "eval": evaluator(step)}
        if step == 0:
            row["baseline"] = evaluator(step, baseline=True)
        row["eval_s"] = round(time.time() - start, 1)
        emit("pmd_pih3_eval", row)
        return row["eval"]

    emit("pmd_pih3_setup", {"sigma1": model.s1, "rho": round(P.rho_of(model.s1), 4), "dial": model.dial,
                            "sigma_mixed": round(P.mix_noise(torch.zeros(1), torch.zeros(1), model.s1, model.dial)[1], 4),
                            "lora_params": sum(p.numel() for p in lora_params), "prompts": len(prompts),
                            "teacher": model.teacher_kind, "drop": args.drop, "start": step0,
                            "gan_weight": gan_weight, "teacher_device": str(tdev)})
    if evaluator is not None and step0 == 0:
        run_eval(0)
    t_last = time.time()
    for step in range(step0, args.steps):
        seed = rng.randrange(1 << 30)
        text = HOOKS["load_text"](prompts[rng.randrange(len(prompts))])
        z, z_lr, za = HOOKS["noise"](shape_args, seed)
        warm = min(1.0, (step + 1) / max(1, args.warmup))
        for group, lr in zip(opt.param_groups, base_lrs):
            group["lr"] = lr * warm
        extra, d_losses, g_norms = None, [], []
        if disc is not None:
            for _d in range(args.d_updates):                # D first, on fresh clips (pmd_h3's order, D:G = 5:1)
                d_text = HOOKS["load_text"](prompts[gan_rng.randrange(len(prompts))])
                dz, dz_lr, dza = HOOKS["noise"](shape_args, gan_rng.randrange(1 << 30))
                with torch.no_grad():
                    fake = model.sample(d_text, dz, dz_lr, dza, eval_noise(dz.shape, gan_rng.randrange(1 << 30),
                                                                           dz.device))["latent"]
                real_latent, real_text = real(gan_rng)
                d_losses.append(d_step(model, disc, d_opt, fake, d_text, real_latent, real_text, block=args.d_block,
                                       generator=gan_gen, rng=gan_rng, clip=args.clip))
            extra = gan_extra(model, disc, text, weight=gan_weight, draws_n=args.draws, block=args.d_block,
                              generator=gan_gen, rng=gan_rng, log=g_norms)
        opt.zero_grad(set_to_none=True)
        loss, stats = pmd_update(model, text, z, z_lr, za, tilts_n=args.tilts, draws_n=args.draws, generator=gen,
                                 factor=args.tilt_factor, radius=args.radius, estimator=args.estimator,
                                 extra_loss=extra)
        norm = float(torch.nn.utils.clip_grad_norm_(lora_params + noise_params, args.clip))
        opt.step()
        if step % args.log_every == 0 or args.check:
            now = time.time()
            every = 1 if args.check else args.log_every
            row = {"step": step, "loss": round(float(loss), 4), "loss_plain": round(stats["loss_plain"], 4),
                   "ess": round(stats["ess"], 3), "w_max": round(stats["w_max"], 3),
                   "shift_ratio": round(stats["shift_ratio"], 5),
                   "target_spread": round(stats["target_spread"], 5),
                   "draw_spread": round(stats["draw_spread"], 5), "grad": round(norm, 4),
                   "s_per_step": round((now - t_last) / every, 2) if step > step0 else None,
                   "peak_gb": peak_gb(dev)}
            if disc is not None:
                # gan: the G loss summed over the draws (weighted); g_grad: |dG/dx| per draw, to set against
                # pmd_grad (|dPMD/dx| per draw; the PMD loss is a SUM over the latent, docs/PMD.md guess 12)
                row.update(gan=round(stats["extra"], 4), d_loss=round(sum(d_losses) / len(d_losses), 4)
                           if d_losses else None, g_grad=round(sum(g_norms) / len(g_norms), 6) if g_norms else None,
                           pmd_grad=round(stats.get("draw_grad", float("nan")), 6),
                           peak_gb_teacher=peak_gb(tdev) if tdev != dev else None)
            emit("pmd_pih3", row)
            t_last = now
        if args.check and step >= step0 + 2:
            print("LOT_H3 kind=pmd_pih3_check ok=1", flush=True)
            return
        done = step + 1
        if done % args.save_every == 0:
            save_ckpt(args.out / f"pmd_{done}.pt", model, network, args, done, disc=disc)
            save_ckpt(args.out / "pmd_last.pt", model, network, args, done, opt=opt, disc=disc, d_opt=d_opt)
        if evaluator is not None and done % args.eval_every == 0:
            row = run_eval(done)
            t_last = time.time()
            if args.stop_at and done == args.stop_at and row.get("hf", 1.0) < args.stop_below_hf:
                emit("pmd_pih3_early_stop", {"step": done, "hf": row.get("hf"), "below": args.stop_below_hf})
                break
    print("LOT_H3 kind=pmd_pih3_done ok=1", flush=True)


@torch.no_grad()
def evaluate(args) -> None:
    """The gate table on the held-out clips: ``baseline`` (the untrained fast path) and each ``--ckpt``; seconds,
    LPIPS, hf, div; contact sheets. ``--adapter merged`` is the deployment form (int8 pi-H3 merged + the LoRA)."""
    import lpips
    from setproctitle import setproctitle

    setproctitle(f"pih3:pmd eval {args.out.name}"[:60])
    _guard()
    args.out.mkdir(parents=True, exist_ok=True)
    model, network, head = build(args, train=False)
    lp = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    held = sorted((args.data / args.folder / "held").glob("*.pt"))[:args.held]
    summary = {"baseline": held_eval(model, head, lp, held, sheet_path=args.out / "sheet_baseline.jpg",
                                     sheets=args.sheets, baseline=True)}
    print("LOT_H3 kind=pmd_pih3_gate variant=baseline " + json.dumps(summary["baseline"]), flush=True)
    for ckpt in args.ckpt or []:
        saved = load_ckpt(Path(ckpt), model, network)
        model.dial = float(saved.get("dial", args.dial)) if args.eval_dial is None else args.eval_dial
        name = f"{Path(ckpt).parent.name}/{Path(ckpt).stem}"
        summary[name] = held_eval(model, head, lp, held, sheet_path=args.out / f"sheet_{Path(ckpt).stem}.jpg",
                                  sheets=args.sheets)
        print(f"LOT_H3 kind=pmd_pih3_gate variant={name} " + json.dumps(summary[name]), flush=True)
    (args.out / "eval.json").write_text(json.dumps({"adapter": args.adapter, "drop": args.drop, "summary": summary},
                                                   indent=1) + "\n")


@torch.no_grad()
def probe(args) -> None:
    """``probe_state`` on the first ``--states`` held-out prompts (their own seeds), the untrained student. Prints
    one line per state and multiplier, then the means. Writes ``probe.json``. No training, nothing else saved."""
    from pih3_distill import paired_noise
    from pih3_h3 import audio_noise
    from setproctitle import setproctitle
    from train_dwd_h3 import load_text

    setproctitle(f"pih3:pmd probe {args.out.name}"[:60])
    _guard()
    args.out.mkdir(parents=True, exist_ok=True)
    model, _network, _head = build(args, train=False)
    held = sorted((args.data / args.folder / "held").glob("*.pt"))[:args.states]
    gen = torch.Generator(model.device).manual_seed(args.seed)
    rows = []
    for index, path in enumerate(held):
        item = torch.load(path, weights_only=False)
        width, height = item["size"]
        z, z_lr = paired_noise(item["latent_t"], height, width, item["seed"])
        za = audio_noise(item["latent_t"], item["seed"] + 1)
        for row in probe_state(model, load_text(item["te"]).cuda(), z, z_lr, za, scales=args.scales,
                               tilts_n=args.tilts, draws_n=args.draws, generator=gen, factor=args.tilt_factor,
                               radius=args.radius, estimator=args.estimator):
            row = {"state": index, **{k: round(v, 5) for k, v in row.items()}}
            rows.append(row)
            print("LOT_H3 kind=pmd_pih3_probe " + json.dumps(row), flush=True)
    means = []
    for scale in args.scales:
        group = [r for r in rows if r["scale"] == float(scale)]
        means.append({k: round(sum(r[k] for r in group) / len(group), 5) for k in group[0] if k != "state"})
        print("LOT_H3 kind=pmd_pih3_probe_mean " + json.dumps(means[-1]), flush=True)
    (args.out / "probe.json").write_text(json.dumps({"radius": args.radius, "dial": args.dial, "drop": args.drop,
                                                     "teacher": args.teacher, "rows": rows, "means": means},
                                                    indent=1) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--data", type=Path, required=True, help="pih3_distill.py teacher's --out (the held-out clips)")
        q.add_argument("--folder", default="768x512_f39")
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--rank", type=int, default=32)
        q.add_argument("--drop", default="", help="h3tiny, or block indices joined by '-' (student and LR step)")
        q.add_argument("--up", type=Path, default=Path(UP_DEFAULT))
        q.add_argument("--dial", type=float, default=DEFAULT_DIAL, help="noise dial of the full-resolution step")
        q.add_argument("--lora-lr-step", action="store_true",
                       help="LoRA also on in the half-resolution step (default: that step is frozen pi-H3)")

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--adapter", choices=("bypass",), default="bypass")
    t.add_argument("--teacher", choices=("base", "pih3"), default="base",
                   help="base: H3 with the pi-H3 adapter off (CFG-distilled, the closest to the paper's teacher)")
    t.add_argument("--teacher-device", default=None,
                   help="e.g. cuda:1: load base H3 a second time there for the teacher calls, the discriminator and "
                        "its feature passes (default: the student's DiT with its switches, one card)")
    t.add_argument("--gan-weight", type=float, default=0.0,
                   help="the GAN term's weight (paper: 0.75); 0 = PMD only (needs real latents, see --real)")
    t.add_argument("--d-updates", type=int, default=5, help="discriminator updates per update (paper D:G = 5:1)")
    t.add_argument("--d-lr", type=float, default=5e-5, help="discriminator learning rate (paper 5e-5)")
    t.add_argument("--d-block", type=int, default=25, help="teacher block whose output the discriminator reads")
    t.add_argument("--real", choices=("cache", "teacher"), default="cache",
                   help="cache: pmd_h3's real latents (the prompt caches); teacher: the pi-H3 4-step clips in "
                        "DATA/FOLDER/train (generated, not real)")
    t.add_argument("--real-cache", nargs="*", default=None, help="--real cache: the caches (default: stills + clips)")
    t.add_argument("--real-stills", action="store_true",
                   help="--real cache: also use stills, repeated over time (a static 'real' clip)")
    t.add_argument("--size", default="768x512")
    t.add_argument("--latent-frames", type=int, default=12, help="12 latent frames = 39 pixel frames")
    t.add_argument("--prompts", type=int, default=500, help="the first N training prompts (the teacher clips' set)")
    t.add_argument("--holdout", type=int, default=8)
    t.add_argument("--tilts", type=int, default=4)
    t.add_argument("--draws", type=int, default=4)
    t.add_argument("--radius", type=float, nargs=2, default=[0.25, 0.5])
    t.add_argument("--tilt-factor", type=int, nargs=3, default=[4, 8, 8])
    t.add_argument("--estimator", choices=tuple(P.ESTIMATORS), default="jk")
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--noise-lr", type=float, default=None)
    t.add_argument("--warmup", type=int, default=50)
    t.add_argument("--clip", type=float, default=10.0)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--resume", type=Path, default=None)
    t.add_argument("--save-every", type=int, default=250)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--eval-held", type=int, default=8)
    t.add_argument("--stop-at", type=int, default=0, help="with --stop-below-hf: stop after this step's eval")
    t.add_argument("--stop-below-hf", type=float, default=0.9)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--check", action="store_true")
    e = sub.add_parser("eval")
    common(e)
    e.add_argument("--adapter", choices=("merged", "bypass"), default="merged")
    e.add_argument("--ckpt", nargs="*", default=None)
    e.add_argument("--eval-dial", type=float, default=None, help="override the checkpoint's dial at inference")
    e.add_argument("--held", type=int, default=16)
    e.add_argument("--sheets", type=int, default=4)
    b = sub.add_parser("probe", help="no training: ESS, draw spread and teacher shift per tilt-radius multiplier")
    common(b)
    b.add_argument("--adapter", choices=("bypass",), default="bypass")
    b.add_argument("--teacher", choices=("base", "pih3"), default="base")
    b.add_argument("--states", type=int, default=3)
    b.add_argument("--scales", type=float, nargs="+", default=[1.0, 4.0, 16.0])
    b.add_argument("--tilts", type=int, default=4)
    b.add_argument("--draws", type=int, default=4)
    b.add_argument("--radius", type=float, nargs=2, default=[0.25, 0.5])
    b.add_argument("--tilt-factor", type=int, nargs=3, default=[4, 8, 8])
    b.add_argument("--estimator", choices=tuple(P.ESTIMATORS), default="jk")
    b.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    {"train": train, "eval": evaluate, "probe": probe}[args.cmd](args)


if __name__ == "__main__":
    main()
