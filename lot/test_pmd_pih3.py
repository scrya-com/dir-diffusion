"""CPU tests for ``pmd_pih3``: PMD on the full-resolution step of the pi-H3 fast path. A toy DiT (with activation
checkpointing, as H3 trains) stands in for H3, a toy switch for the pi-H3 adapter; the LoRA modules are
musubi-tuner's own. No weights, no GPU.

    CUDA_VISIBLE_DEVICES="" python lot/test_pmd_pih3.py
"""

from __future__ import annotations

import json
import math
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


def toy(dial=0.3, drop=(1,), adapter=True, teacher="base", lora_lr_step=False, train=True, randomize=True):
    from h3_musubi import make_lora

    dit = T.ToyDit(blocks=4)
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
    model = F.FastPMD(dit, nn.Upsample(scale_factor=(1, 2, 2)), sigmas=SIGMAS, dial=dial, noise_input=P.NoiseInput(16),
                      lora=net, adapter=ToyAdapter(dit) if adapter else None, drop=drop, forward=toy_forward,
                      teacher=teacher, lora_lr_step=lora_lr_step)
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


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok {name}", flush=True)
    print(f"ok {len(tests)}")
