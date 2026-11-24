"""CPU tests for ``pmd_pih3``: PMD on the full-resolution step of the pi-H3 fast path. A toy DiT (with activation
checkpointing, as H3 trains) stands in for H3, a toy switch for the pi-H3 adapter; the LoRA modules are
musubi-tuner's own. No weights, no GPU.

    CUDA_VISIBLE_DEVICES="" python lot/test_pmd_pih3.py
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import sys
import tempfile
import types
import warnings
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party"))

import pmd_h3 as P  # noqa: E402
import pmd_pih3 as F  # noqa: E402
import test_pmd_h3 as T  # noqa: E402  (the toy DiT classes musubi's LoRA targets)

torch.set_default_dtype(torch.float32)
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
SIGMAS = [0.993766725063324, 0.8653398752212524, 0.0]            # pih3_sigmas(2)


def toy_forward(dit, latent, t, text, *, video_extra=None, audio=None, t_audio=None, out=None, **_unused):
    """``h3_musubi.forward_video``'s contract as the fast path uses it: the video velocity, and in ``out`` the audio
    velocity, the block stack's output and the timestep embedding. Blocks are checkpointed in training mode."""
    from h3 import patchify, unpatchify

    dit.calls.append({"t": float(t), "extra": video_extra is not None, "rows": latent.shape[2:],
                      "drop": sum("forward" in b.__dict__ for b in dit.blocks), "grad": torch.is_grad_enabled()})
    tokens = patchify(latent.float())
    shape = tokens.shape
    h = dit.video_patch_proj(tokens.flatten(1, 3))
    if video_extra is not None:
        h = h + video_extra
    h = h + float(t) * dit.t_vec + 0.1 * text.float().mean()
    if audio is not None:
        h = h + 0.05 * audio.float().mean() + 0.01 * float(t_audio)
    for block in dit.blocks:
        if dit.training and torch.is_grad_enabled():
            h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
        else:
            h = block(h)
    if out is not None:
        out["audio"] = torch.zeros_like(audio) + 0.3 * h.detach().mean() if audio is not None else None
        out["hidden"] = h[0]
        out["temb_video"] = float(t) * dit.t_vec
    return unpatchify(dit.head(h).view(shape))


class ToyAdapter:
    """``pih3_distill.BypassAdapter``'s switch: one tensor swapped between its base and its adapter value."""

    def __init__(self, dit):
        self.base = dit.head.weight.data
        self.tuned = self.base + 0.2 * torch.randn(self.base.shape, generator=torch.Generator().manual_seed(21))
        self.target, self.enabled = dit.head.weight, True
        self.set(True)

    def set(self, on: bool) -> None:
        self.enabled = on
        self.target.data = self.tuned if on else self.base


def toy(dial=0.3, drop=(1,), adapter=True, teacher="base", lora_lr_step=False, train=True, randomize=True,
        separate=False):
    """``separate``: the teacher is a second copy of the base DiT (``--teacher-device``), made before the LoRA and
    the adapter exist, instead of the student's DiT under the teacher's switches."""
    import copy

    from h3_musubi import make_lora

    dit = T.ToyDit(blocks=4)
    teacher_dit = copy.deepcopy(dit) if separate else None
    g = torch.Generator().manual_seed(31)
    z = torch.randn(1, 24, 2, 4, 8, generator=g)
    text, za = torch.ones(1, 3, 5), torch.randn(1, 32, 2, 7, generator=g)
    z_lr = 2.0 * nn.functional.avg_pool3d(z, (1, 2, 2))
    base = {"teacher": lambda x, audio: toy_forward(dit, x, 1 - SIGMAS[1], text, **audio).clone()}
    net = make_lora(dit, 4)
    if randomize:
        with torch.no_grad():
            for m in net.unet_loras:
                m.lora_up.weight.normal_(0, 0.3)
    net.requires_grad_(True)
    dit.train(train)
    if teacher_dit is not None:
        teacher_dit.train(train)
    model = F.FastPMD(dit, nn.Upsample(scale_factor=(1, 2, 2)), sigmas=SIGMAS, dial=dial, noise_input=P.NoiseInput(16),
                      lora=net, adapter=ToyAdapter(dit) if adapter else None, drop=drop, forward=toy_forward,
                      teacher=teacher, lora_lr_step=lora_lr_step, teacher_dit=teacher_dit)
    return model, net, (text, z, z_lr, za), base


def params_of(model, net):
    return [p for p in net.parameters() if p.requires_grad] + [model.noise_input.proj.weight]


def test_rho_and_mixed_level_at_sigma1():
    assert abs(P.rho_of(SIGMAS[1]) - SIGMAS[1] ** 2 / (1 - SIGMAS[1])) < 1e-12 and 5.5 < P.rho_of(SIGMAS[1]) < 5.6
    _y, st = P.mix_noise(torch.zeros(2), torch.zeros(2), SIGMAS[1], F.DEFAULT_DIAL)
    a = 0.7 * (1 - SIGMAS[1])
    s = math.sqrt(0.49 * SIGMAS[1] ** 2 + 0.09)
    assert abs(st - s / (a + s)) < 1e-12 and SIGMAS[1] < st < 1.0      # mixing in noise raises the level


def test_baseline_sample_is_the_fast_path():
    """LoRA off, no mixing, no NoiseInput: ``sample(baseline=True)`` is ``pih3_distill.fast_path`` with the blocks
    dropped, number for number (so the step-0 baseline row is the other table's untrained row)."""
    import h3_musubi
    import pih3_distill
    from pih3_cascade import dropped

    model, net, (text, z, z_lr, za), _ = toy()
    u = torch.randn(z.shape, generator=torch.Generator().manual_seed(1))
    got = model.sample(text, z, z_lr, za, u, baseline=True)
    real = h3_musubi.forward_video
    h3_musubi.forward_video = toy_forward
    try:
        for m in net.unet_loras:
            m.enabled = False
        with dropped(model.dit, (1,)), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want = pih3_distill.fast_path(model.dit, model.up, text, z, z_lr, za, SIGMAS)
    finally:
        h3_musubi.forward_video = real
        for m in net.unet_loras:
            m.enabled = True
    assert torch.allclose(got["latent"], want["latent"], atol=1e-6)
    assert torch.allclose(got["h"], want["h"], atol=1e-6) and torch.allclose(got["temb"], want["temb"])
    assert not torch.allclose(model.sample(text, z, z_lr, za, u)["latent"], want["latent"])    # the student differs


def test_switches_teacher_is_base_and_states_restore():
    model, net, (text, z, _z_lr, za), base = toy()
    assert model.state == {"drop": False, "lora": True, "adapter": True}
    ya = SIGMAS[0] * za
    with torch.no_grad():
        model.adapter.set(False)
        for m in net.unet_loras:
            m.enabled = False
        want = z + SIGMAS[1] * base["teacher"](z, model._audio(ya))            # base DiT, 4 blocks, no LoRA
        model._apply(model.state)
    with model.student_mode():
        assert model.state == {"drop": True, "lora": True, "adapter": True}
        assert "forward" in model.dit.blocks[1].__dict__ and model.adapter.enabled
        got = model.teacher(z, text, ya)
        assert model.dit.calls[-1]["drop"] == 0 and not model.dit.calls[-1]["extra"]
        assert abs(model.dit.calls[-1]["t"] - (1 - SIGMAS[1])) < 1e-9
        assert model.state == {"drop": True, "lora": True, "adapter": True}    # back in the student's mode
        assert "forward" in model.dit.blocks[1].__dict__ and all(m.enabled for m in net.unet_loras)
        assert model.adapter.enabled and torch.equal(model.dit.head.weight.data, model.adapter.tuned)
    assert torch.equal(got, want)
    assert model.state == {"drop": False, "lora": True, "adapter": True}
    assert "forward" not in model.dit.blocks[1].__dict__
    try:
        with model.student_mode():
            raise KeyError("boom")
    except KeyError:
        pass
    assert model.state["drop"] is False and "forward" not in model.dit.blocks[1].__dict__
    with torch.no_grad(), model.student_mode():                                 # --teacher pih3 keeps the adapter
        model.teacher_kind = "pih3"
        assert not torch.allclose(model.teacher(z, text, ya), want)
        assert model.adapter.enabled


def test_merged_adapter_has_no_base_teacher():
    model, _net, (text, z, _z_lr, za), _ = toy(adapter=False)
    model.sample(text, z, 2.0 * nn.functional.avg_pool3d(z, (1, 2, 2)), za, torch.randn_like(z))   # inference works
    try:
        model.teacher(z, text, za)
        raise AssertionError("a merged adapter cannot be the base teacher")
    except RuntimeError:
        pass
    assert model.state == {"drop": False, "lora": True, "adapter": True}


def test_lr_step_is_frozen_and_on_policy():
    """The half-resolution step runs pi-H3 without the LoRA (its weights do not change ``z_t``), without gradients,
    at half resolution with the blocks dropped; ``z_t`` is the upsampled estimate re-noised with ``z`` to sigma_1."""
    model, net, (text, z, z_lr, za), _ = toy()
    with model.student_mode():
        z_t, ya = model.lr_state(text, z, z_lr, za)
        call = model.dit.calls[-1]
        assert tuple(call["rows"]) == (2, 2, 4) and call["drop"] == 1 and not call["grad"]
        assert abs(call["t"] - (1 - SIGMAS[0])) < 1e-9
        assert all(m.enabled for m in net.unet_loras)                           # restored after the step
        with torch.no_grad():
            for m in net.unet_loras:
                m.lora_up.weight.mul_(3.0)
        again, _ = model.lr_state(text, z, z_lr, za)
    assert torch.equal(z_t, again) and not z_t.requires_grad and not ya.requires_grad
    assert z_t.shape == z.shape and ya.shape == za.shape
    clean = (z_t - SIGMAS[1] * z) / (1 - SIGMAS[1])                             # the upsampled estimate
    assert torch.allclose(clean[..., 0::2, 0::2], clean[..., 1::2, 1::2], atol=1e-4)
    other, net2, data, _ = toy(lora_lr_step=True)
    with other.student_mode():
        a, _ = other.lr_state(*data)
        with torch.no_grad():
            for m in net2.unet_loras:
                m.lora_up.weight.mul_(3.0)
        b, _ = other.lr_state(*data)
    assert not torch.allclose(a, b)                                             # --lora-lr-step: the LoRA is in it


def test_student_is_stochastic_from_step_zero():
    """Dial 0 and the zero-init NoiseInput: the student is the plain full-resolution step whatever ``u`` is. Dial
    0.3: the same model on the mixed latent at the mixed level, different for each ``u``."""
    model, _net, (text, z, z_lr, za), _ = toy(dial=0.0)
    g = torch.Generator().manual_seed(2)
    u1, u2 = torch.randn(z.shape, generator=g), torch.randn(z.shape, generator=g)
    with torch.no_grad(), model.student_mode():
        z_t, ya = model.lr_state(text, z, z_lr, za)
        plain = model.student(z_t, u1, text, ya, plain=True)
        assert torch.equal(model.student(z_t, u1, text, ya), plain) and torch.equal(model.student(z_t, u2, text, ya), plain)
        assert model.dit.calls[-1]["extra"] and not model.dit.calls[-3]["extra"]
        model.dial = 0.3
        y, st = P.mix_noise(z_t, u1, SIGMAS[1], 0.3)
        a = model.student(z_t, u1, text, ya)
        assert abs(model.dit.calls[-1]["t"] - (1 - st)) < 1e-9 and model.dit.calls[-1]["drop"] == 1
        want = y + st * toy_forward(model.dit, y, 1 - st, text, **model._audio(ya))
        assert torch.allclose(a, want, atol=1e-6)
        assert not torch.allclose(a, model.student(z_t, u2, text, ya))
        model.noise_input.proj.weight.normal_(0, 0.1)                           # once trained it reads the raw z_t
        assert not torch.allclose(model.student(z_t, u1, text, ya), a)


def direct_grads(model, net, data, seed, tilts_n=2, draws_n=4, scale=10.0):
    """The same update by plain autograd, no checkpointing, all M graphs alive: the reference."""
    text, z, z_lr, za = data
    gen = torch.Generator().manual_seed(seed)
    params = params_of(model, net)
    for p in params:
        p.grad = None
    model.dit.eval()                                                            # no checkpointing in the toy forward
    with model.student_mode():
        z_t, ya = model.lr_state(text, z, z_lr, za)
        tilts = [scale * t for t in P.sample_tilts(z_t, tilts_n, factor=(1, 2, 2), radius=(0.25, 0.5), generator=gen)]
        noises = [torch.randn(z_t.shape, generator=gen) for _ in range(draws_n)]
        targets = [model.teacher(z_t + P.rho_of(model.s1) * t, text, ya)[0] for t in tilts]
        draws = torch.stack([model.student(z_t, u, text, ya)[0] for u in noises])
        loss = P.pmd_loss(draws, [t[0] for t in tilts], targets)
        loss.backward()
    model.dit.train()
    grads = [None if p.grad is None else p.grad.clone() for p in params]
    for p in params:
        p.grad = None
    return loss.detach(), grads


def test_update_gradients_with_checkpointing_and_dropped_blocks():
    """``pmd_update`` (checkpointed blocks, replayed draws, teacher calls with other switches in between) gives the
    gradients of plain autograd. The dropped block and the half-resolution step get none."""
    model, net, data, _ = toy()
    with torch.no_grad():
        model.noise_input.proj.weight.normal_(0, 0.05)
    params = params_of(model, net)
    real = P.sample_tilts
    P.sample_tilts = lambda *a, **k: [10.0 * t for t in real(*a, **k)]          # tiny toy: non-uniform weights
    try:
        gen = torch.Generator().manual_seed(7)
        loss, stats = F.pmd_update(model, *data, tilts_n=2, draws_n=4, generator=gen, factor=(1, 2, 2))
    finally:
        P.sample_tilts = real
    got = [None if p.grad is None else p.grad.clone() for p in params]
    assert model.state == {"drop": False, "lora": True, "adapter": True}
    want_loss, want = direct_grads(model, net, data, 7)
    assert torch.allclose(loss, want_loss, atol=1e-5)
    seen = 0
    for a, b in zip(got, want):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-3), float((a - b).abs().max())
            seen += float(a.abs().sum()) > 0
    assert seen > 4 and float(got[-1].abs().sum()) > 0                          # LoRA and NoiseInput both move
    dropped = [p for m in net.unet_loras if "blocks_1_" in m.lora_name for p in m.parameters()]
    assert dropped and all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in dropped)
    assert 1.0 <= stats["ess"] <= 4.0 and stats["shift_ratio"] > 0 and stats["draw_spread"] > 0
    lr_calls = [c for c in model.dit.calls if tuple(c["rows"]) == (2, 2, 4)]
    assert lr_calls and not any(c["grad"] for c in lr_calls)                    # the LR step never has a graph


def test_backward_under_other_switches_is_wrong():
    """Why ``pmd_update`` holds ``student_mode()`` across the backward: activation checkpointing recomputes each
    block during backward with whatever switches are set THEN. Here the same replay is back-propagated under the
    teacher's switches (LoRA off, nothing dropped): it raises or gives other gradients. (``pih3_distill``'s first
    runs had their backward outside the drop context; in this toy, with non-reentrant checkpointing, a block that
    returned its input is not recomputed, so the drop switch alone does not change the gradients; the LoRA switch
    does.)"""
    model, net, data, _ = toy()
    text, z, z_lr, za = data
    params = params_of(model, net)
    g = torch.Generator().manual_seed(9)
    with model.student_mode():
        z_t, ya = model.lr_state(text, z, z_lr, za)
        u = torch.randn(z_t.shape, generator=g)
        weight = torch.randn(z_t.shape, generator=g)
        inside = (model.student(z_t, u, text, ya) * weight).sum()
        inside.backward()
        good = [None if p.grad is None else p.grad.clone() for p in params]
        for p in params:
            p.grad = None
        other = (model.student(z_t, u, text, ya) * weight).sum()
        assert torch.allclose(inside, other)
        try:
            with model.teacher_mode():
                other.backward()
            bad = [None if p.grad is None else p.grad.clone() for p in params]
            wrong = any((a is None) != (b is None) or (a is not None and not torch.allclose(a, b, atol=1e-6))
                        for a, b in zip(good, bad))
        except RuntimeError:
            wrong = True
    assert wrong


def test_train_loop_saves_stops_early_and_resumes_on_cpu():
    saved = dict(F.HOOKS)
    out = Path(tempfile.mkdtemp())
    built = {}
    evals = []

    def build(args, train):
        model, net, data, _ = toy(randomize=False)
        built.update(model=model, net=net, data=data)
        return model, net, None

    def evaluator(args, model, head):
        def run(step, baseline=False):
            evals.append((step, baseline))
            return {"lpips": 0.5, "hf": 0.8 if step >= 4 else 0.99}
        return run

    F.HOOKS.update(guard=lambda: None, build=build, prompts=lambda args: ["a", "b"],
                   load_text=lambda path: torch.ones(1, 3, 5),
                   noise=lambda shape, seed: (lambda g: (lambda z: (z, 2.0 * nn.functional.avg_pool3d(z, (1, 2, 2)),
                                                                   torch.randn(1, 32, 2, 7, generator=g)))(
                       torch.randn(1, 24, 2, 4, 8, generator=g)))(torch.Generator().manual_seed(seed)),
                   evaluator=evaluator)

    def args_for(steps, **kw):
        ns = types.SimpleNamespace(out=out, rank=4, drop="1", size="128x64", latent_frames=2, tilts=2, draws=4,
                                   radius=[0.25, 0.5], tilt_factor=[1, 2, 2], estimator="jk", lr=1e-3, noise_lr=None,
                                   warmup=2, clip=10.0, steps=steps, resume=None, save_every=2, eval_every=2,
                                   stop_at=4, stop_below_hf=0.9, log_every=1, seed=0, check=False, dial=0.3, up="none")
        ns.__dict__.update(kw)
        return ns

    try:
        F.train(args_for(8))                                                    # stops at 4: hf 0.8 < 0.9
        rows = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
        steps = [r["step"] for r in rows if r["kind"] == "pmd_pih3"]
        assert steps == [0, 1, 2, 3] and all(math.isfinite(r["loss"]) for r in rows if r["kind"] == "pmd_pih3")
        assert evals == [(0, False), (0, True), (2, False), (4, False)]
        assert rows[-1]["kind"] == "pmd_pih3_early_stop" and (out / "pmd_4.pt").exists()
        last = torch.load(out / "pmd_last.pt", weights_only=True)
        assert last["step"] == 4 and "opt" in last and last["dial"] == 0.3 and last["adapter"] == "bypass"
        assert float(last["noise_input"]["proj.weight"].abs().sum()) > 0        # the new input half moved
        assert sum(float(v.abs().sum()) for k, v in last["lora"].items() if "lora_up" in k) > 0
        F.train(args_for(6, resume=out / "pmd_last.pt", stop_at=0))             # resumes at 4, runs to 6
        rows = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
        assert [r["step"] for r in rows if r["kind"] == "pmd_pih3"][-2:] == [4, 5]
        assert torch.load(out / "pmd_last.pt", weights_only=True)["step"] == 6
        before = len(rows)
        F.train(args_for(3, check=True, eval_every=2))                          # --check: 3 updates, no eval, no save
        rows = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
        assert [r["step"] for r in rows[before:] if r["kind"] == "pmd_pih3"] == [0, 1, 2]
        assert not any(r["kind"] == "pmd_pih3_eval" for r in rows[before:])
    finally:
        F.HOOKS.clear()
        F.HOOKS.update(saved)


def test_probe_state_scales_the_same_tilts():
    """The probe changes nothing, uses one set of draws and directions for every multiplier, and reports what the
    radius does: the input shift grows linearly, the weights sharpen (ESS falls), the teacher moves further."""
    model, net, data, _ = toy()
    before = [p.detach().clone() for p in params_of(model, net)]
    rows = F.probe_state(model, *data, scales=[1.0, 4.0, 16.0, 400.0], tilts_n=3, draws_n=4,
                         generator=torch.Generator().manual_seed(5), factor=(1, 2, 2))
    assert [r["scale"] for r in rows] == [1.0, 4.0, 16.0, 400.0]
    assert all(torch.equal(a, b) for a, b in zip(before, params_of(model, net)))
    assert all(p.grad is None for p in params_of(model, net))
    assert model.state == {"drop": False, "lora": True, "adapter": True}
    assert len({r["draw_spread"] for r in rows}) == 1 and rows[0]["draw_spread"] > 0
    for a, b in zip(rows, rows[1:]):
        assert abs(b["shift_ratio"] / a["shift_ratio"] - b["scale"] / a["scale"]) < 1e-3
        assert b["ess"] <= a["ess"] + 1e-6 and b["teacher_shift"] > a["teacher_shift"]
        assert b["tilt_over_mean"] > a["tilt_over_mean"]
    assert rows[0]["ess"] > 3.9 and rows[-1]["ess"] < 2.0 and 1.0 <= rows[-1]["ess"]
    assert all(math.isfinite(v) for r in rows for v in r.values())
    with torch.no_grad(), model.student_mode():                                 # teacher_shift is |D(z+rho l)-D(z)|/|D(z)|
        z_t, ya = model.lr_state(*data)
        tilt = P.sample_tilts(z_t, 3, factor=(1, 2, 2), generator=torch.Generator().manual_seed(5))
        d0 = model.teacher(z_t, data[0], ya)
        want = sum(float((model.teacher(z_t + P.rho_of(model.s1) * t, data[0], ya) - d0).norm()) for t in tilt) / 3
    assert abs(rows[0]["teacher_shift"] - want / float(d0.norm())) < 1e-5


def test_checkpoint_roundtrip():
    model, net, data, _ = toy()
    with torch.no_grad():
        model.noise_input.proj.weight.normal_(0, 0.1)
    out = Path(tempfile.mkdtemp()) / "pmd_1.pt"
    args = types.SimpleNamespace(rank=4, drop="1", up="none")
    F.save_ckpt(out, model, net, args, 1)
    other, net2, _d, _ = toy(randomize=False)
    saved = F.load_ckpt(out, other, net2)
    assert saved["step"] == 1 and saved["rank"] == 4 and saved["lora_lr_step"] is False
    u = torch.randn(data[1].shape, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(model.sample(*data, u)["latent"], other.sample(*data, u)["latent"], atol=1e-6)


# ------------------------------------------------------------------------------------------- the GAN term

def old_pmd_update(model, text, z, z_lr, za, *, tilts_n, draws_n, generator, factor=(4, 8, 8), radius=(0.25, 0.5),
                   estimator="jk", scale=1.0):
    """``pmd_update`` as it was before the GAN term (commit 401fb63's body, verbatim)."""
    with model.student_mode():
        z_t, ya = model.lr_state(text, z, z_lr, za)
        tilts = P.sample_tilts(z_t, tilts_n, factor=tuple(factor), radius=tuple(radius), generator=generator)
        noises = [torch.randn(z_t.shape, device=z_t.device, generator=generator) for _ in range(draws_n)]
        loss, stats = P.pmd_step(lambda s, u, _k: model.student(s, u, text, ya), lambda s: model.teacher(s, text, ya),
                                 z_t, model.s1, 1, tilts, noises, estimator=estimator, scale=scale)
    return loss, stats


def grads_of(params):
    out = [None if p.grad is None else p.grad.clone() for p in params]
    for p in params:
        p.grad = None
    return out


def run_update(model, net, data, seed, **kw):
    loss, stats = F.pmd_update(model, *data, tilts_n=2, draws_n=4, generator=torch.Generator().manual_seed(seed),
                               factor=(1, 2, 2), **kw)
    return loss, stats, grads_of(params_of(model, net))


def test_gan_off_is_bit_identical_to_plain_pmd():
    """``--gan-weight 0`` (``extra_loss=None``): the update is the pre-GAN update, bit for bit, and the training loop
    with the GAN flags at 0 writes the same LoRA as a loop that has never heard of them."""
    model, net, data, _ = toy()
    loss_a, stats_a, got = run_update(model, net, data, 4)
    loss_b, stats_b = old_pmd_update(model, *data, tilts_n=2, draws_n=4, generator=torch.Generator().manual_seed(4),
                                     factor=(1, 2, 2))
    want = grads_of(params_of(model, net))
    assert torch.equal(loss_a, loss_b) and stats_a == stats_b
    assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(got, want))
    lora = []
    for gan in (None, 0.0):
        kw = {} if gan is None else {"gan_weight": gan, "d_updates": 5, "d_lr": 5e-5, "d_block": 2, "real": "cache",
                                     "real_cache": None, "real_stills": False, "teacher_device": None}
        out = Path(tempfile.mkdtemp())
        with train_hooks(real=lambda *a: (_ for _ in ()).throw(AssertionError("no real latents with gan 0"))):
            F.train(loop_args(out, 3, eval_every=0, **kw))
        lora.append(torch.load(out / "pmd_last.pt", weights_only=True))
    assert "disc" not in lora[1]
    for key in ("lora", "noise_input"):
        assert all(torch.equal(lora[0][key][k], lora[1][key][k]) for k in lora[0][key])


def direct_gan_grads(model, net, data, disc, seed, gan_seed, weight, block):
    """The G term alone by plain autograd: eval mode (no checkpointing), every graph alive, one backward."""
    text, z, z_lr, za = data
    gen = torch.Generator().manual_seed(seed)
    ggen = torch.Generator().manual_seed(gan_seed)
    rng = random.Random(gan_seed)
    params = params_of(model, net)
    grads_of(params)
    model.dit.eval()
    if model.teacher_dit is not None:
        model.teacher_dit.eval()
    with model.student_mode():
        z_t, ya = model.lr_state(text, z, z_lr, za)
        P.sample_tilts(z_t, 2, factor=(1, 2, 2), radius=(0.25, 0.5), generator=gen)       # the same draws of gen
        noises = [torch.randn(z_t.shape, generator=gen) for _ in range(4)]
        total = 0.0
        for u in noises:
            clean = model.student(z_t, u, text, ya)
            with model.feature_view().teacher_modes():
                f = F.disc_features(model, clean, text, block=block, generator=ggen, audio_seed=rng.randrange(1 << 30))
            total = total + weight / 4 * P.g_loss(disc(f))
        total.backward()
    model.dit.train()
    if model.teacher_dit is not None:
        model.teacher_dit.train()
    return grads_of(params)


def gan_only(model, net, data, disc, *, seed=7, gan_seed=99, weight=0.75, block=2):
    """``pmd_update`` with the G term and the PMD loss zeroed (train mode, checkpointed): the G term's own gradients
    (a difference of two updates would bury them in float32 cancellation under PMD's much larger gradients)."""
    log = []
    extra = F.gan_extra(model, disc, data[0], weight=weight, draws_n=4, block=block,
                        generator=torch.Generator().manual_seed(gan_seed), rng=random.Random(gan_seed), log=log)
    P.ESTIMATORS["zero"] = lambda draws, _tilt, _target: 0.0 * draws.sum()
    try:
        _l, stats, grads = run_update(model, net, data, seed, extra_loss=extra, estimator="zero")
    finally:
        del P.ESTIMATORS["zero"]
    assert len(log) == 4 and all(v > 0 for v in log) and stats["extra"] > 0 and stats["draw_grad"] == 0
    return grads, stats


def test_gan_term_reaches_the_trainable_params_only():
    """With ``--gan-weight`` > 0 the G term's gradient equals plain autograd through teacher features, D and the
    student's draws, although H3-style checkpointing recomputes the teacher's blocks during the backward (the
    gradient is taken under the teacher's switches). It reaches the LoRA and NoiseInput; not the frozen DiT, not
    the dropped block's LoRA, not the discriminator."""
    model, net, data, _ = toy()
    with torch.no_grad():
        model.noise_input.proj.weight.normal_(0, 0.05)
    torch.manual_seed(5)
    disc = P.DiscHead(16)
    got, stats = gan_only(model, net, data, disc)
    assert all(p.grad is None for p in disc.parameters())                      # the G term leaves D alone
    want = direct_gan_grads(model, net, data, disc, 7, 99, 0.75, 2)
    assert model.state == {"drop": False, "lora": True, "adapter": True}
    moved = 0
    for a, b in zip(got, want):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-3), float((a - b).abs().max())
            moved += float(b.abs().sum()) > 0
    assert moved > 4 and float(got[-1].abs().sum()) > 0                          # LoRA and NoiseInput both move
    dropped = [p for m in net.unet_loras if "blocks_1_" in m.lora_name for p in m.parameters()]
    assert dropped and all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in dropped)
    assert all(p.grad is None for p in model.dit.parameters() if not p.requires_grad)
    assert all(not p.requires_grad for p in T.ToyDit(blocks=4).parameters())     # the toy base is frozen


def test_separate_teacher_equals_the_switched_teacher():
    """``--teacher-device``: a second base DiT as the teacher gives the same teacher output, the same PMD update
    and the same G term as the student's DiT under the teacher's switches (here both on the CPU)."""
    shared, net_a, data, _ = toy()
    separate, net_b, _d, _ = toy(separate=True)
    assert separate.teacher_dit is not None and separate.teacher_device == shared.teacher_device
    with torch.no_grad(), shared.student_mode(), separate.student_mode():
        z_t, ya = shared.lr_state(*data[:1], *data[1:])
        assert torch.equal(shared.teacher(z_t, data[0], ya), separate.teacher(z_t, data[0], ya))
    assert separate.adapter.enabled and separate.state == {"drop": False, "lora": True, "adapter": True}
    _l, _s, a = run_update(shared, net_a, data, 3)
    _l, _s, b = run_update(separate, net_b, data, 3)
    assert all((x is None and y is None) or torch.allclose(x, y, atol=1e-6) for x, y in zip(a, b))
    torch.manual_seed(5)
    disc = P.DiscHead(16)
    ga, _s = gan_only(shared, net_a, data, disc)
    gb, _s = gan_only(separate, net_b, data, disc)
    assert all((x is None and y is None) or torch.allclose(x, y, atol=1e-6) for x, y in zip(ga, gb))
    try:
        toy(separate=True, teacher="pih3")
        raise AssertionError("a separate teacher is base H3 only")
    except ValueError:
        pass


def test_d_step_and_disc_checkpoint_roundtrip():
    """A D update lowers D's loss on a fixed pair; D and its optimizer state go through ``save_ckpt``; a GAN run
    resumed from ``pmd_last.pt`` starts from the saved D (not a fresh one)."""
    model, net, data, _ = toy()
    torch.manual_seed(5)
    disc = P.DiscHead(16)
    d_opt = torch.optim.AdamW(disc.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(1)
    fake, real = torch.randn(1, 24, 2, 4, 8, generator=g), 2.0 + torch.randn(1, 24, 2, 4, 8, generator=g)
    losses = [F.d_step(model, disc, d_opt, fake, data[0], real, data[0], block=2,
                       generator=torch.Generator().manual_seed(2), rng=random.Random(2), clip=10.0) for _ in range(30)]
    assert losses[-1] < losses[0] and model.state == {"drop": False, "lora": True, "adapter": True}
    out = Path(tempfile.mkdtemp()) / "pmd_last.pt"
    F.save_ckpt(out, model, net, types.SimpleNamespace(rank=4, drop="1", up="none", gan_weight=0.75), 3,
                opt=torch.optim.AdamW(params_of(model, net)), disc=disc, d_opt=d_opt)
    saved = torch.load(out, weights_only=True)
    torch.manual_seed(6)
    disc2 = P.DiscHead(16)
    d_opt2 = torch.optim.AdamW(disc2.parameters(), lr=1e-2)
    disc2.load_state_dict(saved["disc"])
    d_opt2.load_state_dict(saved["d_opt"])
    assert saved["gan_weight"] == 0.75
    assert all(torch.equal(a, b) for a, b in zip(disc.state_dict().values(), disc2.state_dict().values()))
    s1, s2 = d_opt.state_dict()["state"], d_opt2.state_dict()["state"]
    assert s1.keys() == s2.keys() and all(torch.equal(s1[k][n], s2[k][n]) for k in s1 for n in ("exp_avg", "exp_avg_sq"))
    no_opt = Path(tempfile.mkdtemp()) / "pmd_3.pt"
    F.save_ckpt(no_opt, model, net, types.SimpleNamespace(rank=4, drop="1", up="none"), 3, disc=disc, d_opt=d_opt)
    assert "disc" in torch.load(no_opt, weights_only=True) and "d_opt" not in torch.load(no_opt, weights_only=True)

    run = Path(tempfile.mkdtemp())
    gan = {"gan_weight": 0.75, "d_updates": 2, "d_lr": 1e-3, "d_block": 2, "real": "cache", "real_cache": None,
           "real_stills": False, "teacher_device": None}
    real_draws = []

    def real_hook(args, shape_args, device):
        def draw(rng):
            real_draws.append(1)
            gr = torch.Generator().manual_seed(rng.randrange(1 << 30))
            return 1.5 + torch.randn(1, 24, 2, 4, 8, generator=gr), torch.ones(1, 3, 5)
        return draw

    with train_hooks(real=real_hook):
        F.train(loop_args(run, 4, eval_every=0, **gan))
    rows = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    steps = [r for r in rows if r["kind"] == "pmd_pih3"]
    assert [r["step"] for r in steps] == [0, 1, 2, 3] and len(real_draws) == 8
    assert all(math.isfinite(r["d_loss"]) and r["gan"] > 0 and r["g_grad"] > 0 and r["pmd_grad"] > 0 for r in steps)
    last = torch.load(run / "pmd_last.pt", weights_only=True)
    assert last["step"] == 4 and "disc" in last and "d_opt" in last and "disc" in torch.load(run / "pmd_2.pt",
                                                                                             weights_only=True)
    seen = []
    real_d_step = F.d_step

    def spy(model, disc, *a, **k):
        if not seen:
            seen.append({n: v.detach().clone() for n, v in disc.state_dict().items()})
        return real_d_step(model, disc, *a, **k)

    F.d_step = spy
    try:
        with train_hooks(real=real_hook):
            F.train(loop_args(run, 6, eval_every=0, resume=run / "pmd_last.pt", **gan))
    finally:
        F.d_step = real_d_step
    assert all(torch.equal(seen[0][n], last["disc"][n]) for n in last["disc"])
    assert torch.load(run / "pmd_last.pt", weights_only=True)["step"] == 6


@contextlib.contextmanager
def train_hooks(**extra):
    """``pmd_pih3.HOOKS`` on the toy: no guard, two prompts, toy noise, no evaluator unless asked."""
    saved = dict(F.HOOKS)

    def build(args, train):
        model, net, _data, _ = toy(randomize=False)
        return model, net, None

    def noise(shape, seed):
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(1, 24, 2, 4, 8, generator=g)
        return z, 2.0 * nn.functional.avg_pool3d(z, (1, 2, 2)), torch.randn(1, 32, 2, 7, generator=g)

    F.HOOKS.update(guard=lambda: None, build=build, prompts=lambda args: ["a", "b"],
                   load_text=lambda path: torch.ones(1, 3, 5), noise=noise, **extra)
    try:
        yield
    finally:
        F.HOOKS.clear()
        F.HOOKS.update(saved)


def loop_args(out, steps, **kw):
    ns = types.SimpleNamespace(out=out, rank=4, drop="1", size="128x64", latent_frames=2, tilts=2, draws=4,
                               radius=[0.25, 0.5], tilt_factor=[1, 2, 2], estimator="jk", lr=1e-3, noise_lr=None,
                               warmup=2, clip=10.0, steps=steps, resume=None, save_every=2, eval_every=2,
                               stop_at=0, stop_below_hf=0.9, log_every=1, seed=0, check=False, dial=0.3, up="none")
    ns.__dict__.update(kw)
    return ns


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok {name}", flush=True)
    print(f"ok {len(tests)}")
