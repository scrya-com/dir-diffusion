"""CPU tests for ``distill_tiny_h3`` and the block-skip toggle (``h3_musubi.blocks_dropped``). No weights, no GPU.

    CUDA_VISIBLE_DEVICES="" python lot/test_distill_tiny_h3.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party"))

import distill_tiny_h3 as D  # noqa: E402
from h3_musubi import blocks_are_dropped, blocks_dropped, drop_blocks, run_blocks, set_blocks_dropped  # noqa: E402

SIGMAS = [1.0, 0.9, 0.8, 0.6]


class ToyBlock(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.a = torch.randn(6, 6, generator=g) * 0.3

    def forward(self, hidden, temb, _segments, _rotation):
        return hidden + torch.tanh(hidden @ self.a) + temb


class ToyDit(torch.nn.Module):
    blocks_to_swap, activation_cpu_offloading = 0, False

    def __init__(self, n=8, checkpointing=False):
        super().__init__()
        self.blocks = torch.nn.ModuleList(ToyBlock(i) for i in range(n))
        self.video_patch_proj = torch.nn.Linear(1, 1)
        self.gradient_checkpointing = checkpointing
        self.training = checkpointing


class ToyDelta(torch.nn.Module):
    """Stands in for the delta LoRA network: one module, ``.enabled``, zero-initialised."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(6, 6))
        self.enabled = True
        self.unet_loras = [self]


class ToyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x0 = torch.nn.Parameter(torch.full((6,), 0.3))
        self.r = torch.nn.Parameter(torch.linspace(-0.1, 0.1, 4))


def toy_step(dit, delta):
    """One "DiR iteration": the block stack on (state, estimate, r), plus the delta when it is on."""
    def step(head, state, estimate, r, _text, _seed):
        hidden = state + estimate * head.x0 + head.r[r]
        out = run_blocks(dit, hidden, torch.zeros(6), None, None)
        if delta.enabled:
            out = out + hidden @ delta.w
        return 0.2 * out
    return step


def make(drop=(2, 3, 5), checkpointing=True):
    dit, delta = ToyDit(checkpointing=checkpointing), ToyDelta()
    teacher = ToyHead().requires_grad_(False)
    student = ToyHead()
    student.load_state_dict(teacher.state_dict())
    distill = D.Distill(dit, drop, teacher, student, delta, SIGMAS, step=toy_step(dit, delta), device="cpu")
    return distill, dit, delta, teacher, student


# ----------------------------------------------------------------------------- block toggle

def test_blocks_dropped_toggle_and_restore():
    dit = ToyDit()
    hidden, temb = torch.randn(1, 5, 6), torch.randn(6) * 0.1
    full = run_blocks(dit, hidden, temb, None, None)
    want = hidden
    for i, block in enumerate(dit.blocks):
        if i not in (1, 4):
            want = block(want, temb, None, None)
    with blocks_dropped(dit, (1, 4)):
        assert blocks_are_dropped(dit, (1, 4)) == [True, True]
        assert torch.allclose(run_blocks(dit, hidden, temb, None, None), want)
        with blocks_dropped(dit, (1, 4), False):          # nested: the full model inside the tiny one
            assert torch.equal(run_blocks(dit, hidden, temb, None, None), full)
        assert blocks_are_dropped(dit, (1, 4)) == [True, True]
    assert blocks_are_dropped(dit, (1, 4)) == [False, False]
    assert torch.equal(run_blocks(dit, hidden, temb, None, None), full)
    try:
        with blocks_dropped(dit, (1,)):
            raise KeyError
    except KeyError:
        pass
    assert blocks_are_dropped(dit, (1,)) == [False]
    drop_blocks(dit, (1, 4))                               # the env path (load_dit) leaves them dropped ...
    assert blocks_are_dropped(dit, (1, 4)) == [True, True]
    with blocks_dropped(dit, (1, 4), False):               # ... and the teacher can still switch them back on
        assert torch.equal(run_blocks(dit, hidden, temb, None, None), full)
    assert blocks_are_dropped(dit, (1, 4)) == [True, True]
    set_blocks_dropped(dit, (1, 4), False)
    try:
        set_blocks_dropped(dit, (8,), True)
        raise AssertionError("out of range must be refused")
    except ValueError:
        pass


# ----------------------------------------------------------------------------- LoRA merge

def test_merge_lora_states_is_exact():
    from musubi_tuner.networks.lora import LoRAModule

    torch.manual_seed(0)
    base = torch.nn.Linear(10, 7)
    copy = torch.nn.Linear(10, 7)
    copy.load_state_dict(base.state_dict())
    first = LoRAModule("m", base, 1.0, 4, alpha=4)          # the published LoRA (alpha = rank)
    first.apply_to()
    second = LoRAModule("m", base, 1.0, 3, alpha=1.5)       # the delta, stacked on it (a scale other than 1 too)
    second.apply_to()
    for module in (first, second):
        torch.nn.init.normal_(module.lora_up.weight)
    x = torch.randn(5, 10)
    stacked = base(x)
    pub = {f"m.{k}": v for k, v in first.state_dict().items()}
    delta = {f"m.{k}": v for k, v in second.state_dict().items()}
    merged = D.merge_lora_states(pub, delta, dtype=torch.float32)
    assert merged["m.lora_down.weight"].shape == (7, 10) and merged["m.lora_up.weight"].shape == (7, 7)
    assert float(merged["m.alpha"]) == 7.0
    single = LoRAModule("m", copy, 1.0, 7, alpha=7)
    single.apply_to()
    single.load_state_dict({k[2:]: v for k, v in merged.items()})
    assert torch.allclose(copy(x), stacked, atol=1e-5)
    second.enabled = False                                  # the delta off = the published LoRA alone
    assert not torch.allclose(base(x), stacked)
    try:
        D.merge_lora_states(pub, {"n.lora_down.weight": delta["m.lora_down.weight"], "n.lora_up.weight": delta["m.lora_up.weight"]})
        raise AssertionError("mismatched modules must be refused")
    except ValueError:
        pass


# ----------------------------------------------------------------------------- the distillation step

def test_student_starts_as_tiny_and_teacher_is_full():
    distill, dit, delta, teacher, student = make()
    z = torch.randn(1, 4, 6)
    t = distill.final(False, z, None, 0)
    s = distill.final(True, z, None, 0)
    with blocks_dropped(dit, (2, 3, 5), False):
        delta.enabled = False
        want_t = distill.rollout(teacher, z, None, 0)[1]
    delta.enabled = True
    assert all(torch.equal(a, b) for a, b in zip(t, want_t))
    with blocks_dropped(dit, (2, 3, 5)):
        want_s = distill.rollout(student, z, None, 0)[1]     # zero delta: the untuned tiny model
    assert all(torch.allclose(a, b) for a, b in zip(s, want_s))
    assert D.mad(s[-1], t[-1]) > 1e-3
    assert blocks_are_dropped(dit, (2, 3, 5)) == [False] * 3 and delta.enabled


def test_sample_loss_zero_without_drop():
    distill, *_ = make(drop=())
    stats = distill.sample(torch.randn(1, 4, 6), None, 0, on_policy=False, weights=D.iteration_weights(4, 2.0))
    assert stats["loss"] < 1e-12


def reference_grads(distill, z, weights, on_policy):
    """Same loss without activation checkpointing and with the targets built by hand."""
    student, teacher, dit, delta = distill.student_head, distill.teacher_head, distill.dit, distill.delta_net
    dit.gradient_checkpointing = False
    with torch.no_grad():
        delta.enabled = False
        t_in, t_out = distill.rollout(teacher, z, None, 0)
        delta.enabled = True
        if on_policy:
            with blocks_dropped(dit, distill.drop):
                s_in, _ = distill.rollout(student, z, None, 0)
            delta.enabled = False
            targets = [distill._step(teacher, *s_in[r], r, None, 0) for r in range(4)]
            delta.enabled = True
            inputs = s_in
        else:
            inputs, targets = t_in, t_out
    params = [delta.w, student.x0, student.r]
    with blocks_dropped(dit, distill.drop):
        loss = sum(w * torch.nn.functional.mse_loss(distill._step(student, *inputs[r], r, None, 0), targets[r])
                   for r, w in enumerate(weights))
    grads = torch.autograd.grad(loss, params)
    dit.gradient_checkpointing = True
    return float(loss.detach()), grads


def test_sample_grads_match_reference_under_checkpointing():
    """The checkpointed backward recomputes the blocks in student mode (dropped), never the teacher's."""
    weights = D.iteration_weights(4, 2.0)
    for on_policy in (False, True):
        distill, dit, delta, teacher, student = make()
        with torch.no_grad():
            delta.w.normal_(0, 0.05)
            student.x0.add_(0.1)
        z = torch.randn(1, 4, 6, generator=torch.Generator().manual_seed(3))
        loss, want = reference_grads(distill, z, weights, on_policy)
        stats = distill.sample(z, None, 0, on_policy=on_policy, weights=weights)
        assert abs(stats["loss"] - loss) < 1e-6 * max(1.0, loss), (stats["loss"], loss)
        for p, g in zip([delta.w, student.x0, student.r], want):
            assert torch.allclose(p.grad, g, atol=1e-6), on_policy
        assert all(p.grad is None for p in teacher.parameters())
        assert blocks_are_dropped(dit, (2, 3, 5)) == [False] * 3 and delta.enabled
        assert (stats["rollout_gap"] is not None) == on_policy and stats["on_policy"] == int(on_policy)


def test_training_closes_the_gap():
    distill, dit, delta, teacher, student = make()
    params = [delta.w, *student.parameters()]
    opt = torch.optim.Adam(params, lr=3e-3)
    weights = D.iteration_weights(4, 2.0)
    zs = [torch.randn(1, 4, 6, generator=torch.Generator().manual_seed(100 + k)) for k in range(4)]
    gap = lambda: sum(D.mad(distill.final(True, z, None, 0)[-1], distill.final(False, z, None, 0)[-1]) for z in zs)  # noqa: E731
    before = gap()
    for step in range(150):
        opt.zero_grad(set_to_none=True)
        for k in range(2):
            z = torch.randn(1, 4, 6, generator=torch.Generator().manual_seed(step * 2 + k))
            distill.sample(z, None, 0, on_policy=step >= 50 and k == 1, weights=weights, loss_scale=0.5)
        opt.step()
    after = gap()
    assert after < 0.7 * before, (before, after)     # a 6x6 linear delta cannot replace three blocks; it must close a good part


# ----------------------------------------------------------------------------- helpers and the pass check

def test_helpers():
    assert D.parse_sizes("32x48, 48X32,24x24") == [(32, 48), (48, 32), (24, 24)]
    for bad in ("32x47", "x48", "0x0", ""):
        try:
            D.parse_sizes(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    w = D.iteration_weights(4, 2.0)
    assert abs(sum(w) - 1) < 1e-12 and w == [0.2, 0.2, 0.2, 0.4]
    assert D.lr_factor(0, 0.0, 10) == 0.0 and abs(D.lr_factor(10, 0.0, 10) - 1.0) < 1e-12
    assert abs(D.lr_factor(100, 1.0, 10) - 0.1) < 1e-12 and abs(D.lr_factor(100, 0.5, 10) - 0.55) < 1e-12
    assert D.lr_factor(100, 2.0, 0) == D.lr_factor(100, 1.0, 0)
    assert D.on_policy_prob(0.1, 0.25, 0.5) == 0.0 and D.on_policy_prob(0.3, 0.25, 0.5) == 0.5


def probe_files(name, offset):
    torch.manual_seed(0)
    pub = {(i, f"pub_r{r}"): torch.randn(1, 24, 2, 4, 4) for i in range(3) for r in range(1, 5)}
    out = {(i, f"{name}_r{r}"): pub[(i, f"pub_r{r}")] + offset * torch.ones(1, 24, 2, 4, 4) for i in range(3) for r in range(1, 5)}
    return pub, out


def js(values):
    return {"items": [{"own_to_dense": [0.0, 0.0, 0.0, v]} for v in values]}


def test_pass_check():
    pub, tiny = probe_files("tiny", 0.6)
    _, good = probe_files("tiny_ft", 0.2)
    _, meh = probe_files("tiny_ft", 0.4)
    evals = [{"step": 0, "select": 0.6, "by_cache": {"a": 0.6, "b": 0.5}},
             {"step": 100, "select": 0.3, "by_cache": {"a": 0.3, "b": 0.3}}]
    full, untuned = js([0.98, 0.97, 0.83]), js([0.94, 0.97, 0.82])
    result = D.pass_check(pub, tiny, good, full, untuned, js([0.97, 0.97, 0.83]), evals)
    assert result["pass"] and abs(result["to_full_tiny_mean"] - 0.6) < 1e-4 and abs(result["to_full_ft_mean"] - 0.2) < 1e-4
    assert result["best_step"] == 100 and result["heldout"]["b"] == {"step0": 0.5, "best": 0.3}
    assert not D.pass_check(pub, tiny, meh, full, untuned, js([0.97, 0.97, 0.83]), evals)["checks"]["to_full"]
    far = D.pass_check(pub, tiny, good, full, untuned, js([1.0, 1.0, 0.9]), evals)     # further from dense20 is not better
    assert not far["checks"]["own_to_dense"] and far["user_literal_own_to_dense_ge_0.925"]
    worse = evals[:1] + [{"step": 100, "select": 0.4, "by_cache": {"a": 0.2, "b": 0.6}}]
    assert not D.pass_check(pub, tiny, good, full, untuned, js([0.97, 0.97, 0.83]), worse)["checks"]["heldout"]
    assert not D.pass_check(pub, tiny, good, full, untuned, js([0.97, 0.97, 0.83]), evals[:1])["checks"]["heldout"]


def test_eval_items_hold_out_the_probe():
    import train_h3

    with tempfile.TemporaryDirectory() as tmp:
        caches = []
        for name, count in (("stills", 20), ("clips", 15)):
            cache = Path(tmp) / name
            cache.mkdir()
            for k in range(count):
                (cache / f"{name}{k:03d}_0512x0768_minimaxh3.safetensors").touch()
                (cache / f"{name}{k:03d}_minimaxh3_te.safetensors").touch()
            caches.append(cache)
        probe, select = D.eval_items(caches, caches[0], 8, 3)
        held = train_h3.list_items(caches[0], 8)[1]
        assert [(c, i) for c, i, _ in probe] == [("stills", 0), ("stills", 1), ("stills", 2)]
        assert [e for _, _, e in probe] == held[:3]
        assert len(select) == 5 + 8 and all((c, i) not in {(p[0], p[1]) for p in probe} for c, i, _ in select)
        train = {e for c in caches for e in train_h3.list_items(c, 8)[0]}
        assert not train & {e for _, _, e in probe + select}


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("ok", len(tests))
