"""CPU tests for ``pmd_h3``: the Tilt=Shift target, the softmax / jackknife estimators, the stochastic input, the
one-DiT teacher, replayed gradients and the any-step sampler. A toy DiT stands in for H3; the LoRA modules are
musubi-tuner's own. No weights, no GPU.

    CUDA_VISIBLE_DEVICES="" python lot/test_pmd_h3.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party"))

import pmd_h3 as P  # noqa: E402

torch.set_default_dtype(torch.float32)


# ------------------------------------------------------------------ a Gaussian world where everything is analytic

class Gauss:
    """x ~ N(mu0, s0^2 I) in d dims, H3's RF corruption z = (1 - sigma) x + sigma eps. The exact denoiser and the
    exact posterior are closed-form: precision 1/s0^2 + (1-sigma)^2/sigma^2."""

    def __init__(self, d=6, mu0=0.3, s0=1.2, sigma=0.7, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.d, self.sigma = d, sigma
        self.mu0 = mu0 + 0.5 * torch.randn(d, generator=g, dtype=torch.float64)
        self.s0 = s0
        a = 1.0 - sigma
        self.v = 1.0 / (1.0 / s0 ** 2 + a * a / sigma ** 2)

    def denoise(self, z):
        a = 1.0 - self.sigma
        return self.v * (self.mu0 / self.s0 ** 2 + a * z / self.sigma ** 2)

    def posterior_draws(self, z, m, g):
        return self.denoise(z) + math.sqrt(self.v) * torch.randn(m, self.d, generator=g, dtype=torch.float64)


def test_rho_is_rectified_flow():
    for s in (0.1, 0.5, 0.947, 0.667):
        assert abs(P.rho_of(s) - s * s / (1 - s)) < 1e-12
    for bad in (0.0, 1.0):
        try:
            P.rho_of(bad)
            raise AssertionError("rho must be refused at the ends")
        except ValueError:
            pass


def test_tilt_equals_shift_gaussian_and_monte_carlo():
    w = Gauss()
    g = torch.Generator().manual_seed(1)
    z = torch.randn(w.d, generator=g, dtype=torch.float64)
    lam = 0.4 * torch.randn(w.d, generator=g, dtype=torch.float64)
    analytic = w.denoise(z) + w.v * lam                       # mean of the tilted Gaussian posterior
    shifted = w.denoise(z + P.rho_of(w.sigma) * lam)          # one teacher call at the shifted input
    assert torch.allclose(shifted, analytic, atol=1e-12), (shifted, analytic)
    draws = w.posterior_draws(z, 400_000, g)                  # self-normalised importance weights exp(lam^T x)
    wts = torch.softmax(draws @ lam, 0)
    assert torch.allclose((wts[:, None] * draws).sum(0), analytic, atol=1e-2)


def test_softmax_mean():
    g = torch.Generator().manual_seed(2)
    draws = torch.randn(4, 3, 2, generator=g)
    tilt = torch.randn(3, 2, generator=g)
    w = torch.softmax(torch.stack([(x * tilt).sum() for x in draws]), 0)
    want = sum(w[m] * draws[m] for m in range(4))
    assert torch.allclose(P.softmax_mean(draws, tilt), want, atol=1e-6)
    assert torch.allclose(P.softmax_mean(draws, torch.zeros_like(tilt)), draws.mean(0), atol=1e-6)
    perm = draws[[2, 0, 3, 1]]
    assert torch.allclose(P.softmax_mean(perm, tilt), want, atol=1e-6)
    big = 1e4 * tilt                                          # one-hot limit: the best-aligned draw
    best = int(torch.argmax(torch.stack([(x * tilt).sum() for x in draws])))
    assert torch.allclose(P.softmax_mean(draws, big), draws[best], atol=1e-4)
    target = torch.randn(3, 2, generator=g)
    assert torch.allclose(P.loss_plain(draws, tilt, target), ((want - target) ** 2).sum(), atol=1e-5)


def test_balanced_splits_and_jackknife_at_zero_tilt():
    assert len(P.balanced_splits(4)) == 3 and len(P.balanced_splits(6)) == 10 and len(P.balanced_splits(5)) == 10
    for a, b in P.balanced_splits(4):
        assert sorted(a + b) == [0, 1, 2, 3]
    g = torch.Generator().manual_seed(3)
    draws, target = torch.randn(4, 5, generator=g), torch.randn(5, generator=g)
    h = draws - target                                        # zero tilt: uniform weights, jackknife = plain means
    want = torch.stack([(h[list(a)].mean(0) * h[list(b)].mean(0)).sum() for a, b in P.balanced_splits(4)]).mean()
    got = P.loss_jackknife(draws, torch.zeros(5), target)
    assert torch.allclose(got, want, atol=1e-6)
    # ...which is the unbiased U-statistic of ||E h||^2 (pairs i != j)
    u = sum((h[i] * h[j]).sum() for i in range(4) for j in range(4) if i != j) / 12
    assert torch.allclose(got, u, atol=1e-5)


def test_loss_zero_when_student_samples_the_teacher_posterior():
    """Student draws = exact posterior samples. The population residual is 0 (plain estimator, 400k draws). With
    M = 4 the jackknife loss (Algorithm 2) averages to ~0: measured 0.053 +- 0.008 against 0.86 for a mean-only
    student, a residual finite-M bias the paper states (unbiased only at zero tilt; 0.014 at M = 8). Plain
    Algorithm 1 at M = 4 averages 1.55: its variance term alone exceeds the mean-only student's loss."""
    w = Gauss(d=4, sigma=0.6)
    g = torch.Generator().manual_seed(4)
    z = torch.randn(w.d, generator=g, dtype=torch.float64)
    lam = 0.5 * torch.randn(w.d, generator=g, dtype=torch.float64)
    target = w.denoise(z + P.rho_of(w.sigma) * lam)
    pay = float(((w.v * lam) ** 2).sum())                     # a deterministic student at the posterior mean
    many = w.posterior_draws(z, 400_000, g)
    assert float(P.loss_plain(many, lam, target)) < 1e-3 * pay + 1e-4
    det = torch.stack([w.denoise(z)] * 4)
    assert abs(float(P.loss_jackknife(det, lam, target)) - pay) < 1e-5 * pay
    assert abs(float(P.loss_plain(det, lam, target)) - pay) < 1e-5 * pay
    jk = sum(float(P.loss_jackknife(w.posterior_draws(z, 4, g), lam, target)) for _ in range(8000)) / 8000
    assert abs(jk) < 0.12 * pay, (jk, pay)
    plain = sum(float(P.loss_plain(w.posterior_draws(z, 4, g), lam, target)) for _ in range(2000)) / 2000
    assert plain > pay and plain > 8 * abs(jk), (plain, jk, pay)


def test_pmd_loss_averages_tilts():
    g = torch.Generator().manual_seed(5)
    draws = torch.randn(4, 3, generator=g)
    tilts = [torch.randn(3, generator=g) for _ in range(4)]
    targets = [torch.randn(3, generator=g) for _ in range(4)]
    want = sum(P.loss_jackknife(draws, t, y) for t, y in zip(tilts, targets)) / 4
    assert torch.allclose(P.pmd_loss(draws, tilts, targets), want, atol=1e-6)


def test_mix_noise_stays_on_the_rf_path():
    g = torch.Generator().manual_seed(6)
    z = torch.randn(1, 24, 2, 4, 4, generator=g)
    y, s = P.mix_noise(z, torch.randn_like(z), 0.8, 0.0)
    assert y is z and s == 0.8
    n = 400_000
    x = 0.7 * torch.ones(n, dtype=torch.float64)
    for sigma, dial in ((0.947, 0.3), (0.857, 0.45), (0.667, 0.15), (1.0, 0.3)):
        zt = (1 - sigma) * x + sigma * torch.randn(n, generator=g, dtype=torch.float64)
        y, s = P.mix_noise(zt, torch.randn(n, generator=g, dtype=torch.float64), sigma, dial)
        assert s >= sigma - 1e-12 and s <= 1.0
        assert abs(float(y.mean()) - (1 - s) * 0.7) < 5e-3, (sigma, dial)     # signal coefficient 1 - s
        assert abs(float(y.std()) - s) < 5e-3, (sigma, dial)                  # noise std s


def test_tilts_are_coarse_unit_norm_times_radius():
    like = torch.zeros(1, 24, 12, 18, 32)
    g = torch.Generator().manual_seed(7)
    tilts = P.sample_tilts(like, 4, factor=(4, 8, 8), radius=(0.25, 0.5), generator=g)
    assert len(tilts) == 4 and all(t.shape == like.shape for t in tilts)
    norms = [float(t.norm()) for t in tilts]
    assert all(0.25 - 1e-5 <= n <= 0.5 + 1e-5 for n in norms), norms
    assert P.coarse_shape((24, 12, 18, 32), (4, 8, 8)) == (24, 3, 3, 4)
    t = tilts[0][0]                                            # neighbours move together: high lag-1 correlation
    corr = torch.corrcoef(torch.stack([t[:, :, :, :-1].flatten(), t[:, :, :, 1:].flatten()]))[0, 1]
    iso = P.sample_tilts(like, 1, factor=(1, 1, 1), generator=g)[0][0]
    corr_iso = torch.corrcoef(torch.stack([iso[:, :, :, :-1].flatten(), iso[:, :, :, 1:].flatten()]))[0, 1]
    assert corr > 0.8 and abs(corr_iso) < 0.05, (corr, corr_iso)
    still = P.sample_tilts(torch.zeros(1, 24, 1, 18, 32), 2, generator=g)
    assert still[0].shape == (1, 24, 1, 18, 32)


def test_renoise():
    g = torch.Generator().manual_seed(8)
    x, eps, fresh = (torch.randn(1, 24, 1, 4, 4, generator=g) for _ in range(3))
    state = 0.1 * x + 0.9 * eps
    assert torch.allclose(P.renoise(x, state, 0.9, 0.5, 1.0, fresh), 0.5 * x + 0.5 * fresh, atol=1e-6)
    assert torch.allclose(P.renoise(x, state, 0.9, 0.5, 0.0, fresh), 0.5 * x + 0.5 * eps, atol=1e-5)
    mid = P.renoise(x, state, 0.9, 0.5, 0.9, fresh)
    a = 0.45 * math.pi
    assert torch.allclose(mid, 0.5 * x + 0.5 * (math.cos(a) * eps + math.sin(a) * fresh), atol=1e-5)


# ----------------------------------------------------------------------------- a toy DiT with musubi LoRAs

class Attn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.qkv_proj, self.out_proj = nn.Linear(h, 3 * h), nn.Linear(h, h)


class Mlp(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(h, 2 * h), nn.Linear(2 * h, h)


class DiTBlock(nn.Module):                                     # the class name musubi's H3 LoRA targets
    def __init__(self, h):
        super().__init__()
        self.attn, self.mlp = Attn(h), Mlp(h)

    def forward(self, x):
        q = self.attn.qkv_proj(x)[..., : x.shape[-1]]
        x = x + self.attn.out_proj(torch.tanh(q))
        return x + self.mlp.fc2(torch.nn.functional.gelu(self.mlp.fc1(x)))


class ToyDit(nn.Module):
    def __init__(self, hidden=16, blocks=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.video_patch_proj = nn.Linear(P.TOKEN_DIM, hidden)
        self.t_vec = nn.Parameter(torch.randn(hidden) * 0.3)
        self.blocks = nn.ModuleList([DiTBlock(hidden) for _ in range(blocks)])
        self.head = nn.Linear(hidden, P.TOKEN_DIM)
        self.requires_grad_(False)
        self.calls = []


def toy_forward(dit, latent, t, text, *, video_extra=None, video_temb_delta=None, audio_generator=None):
    """H3's ``forward_video`` contract: velocity ``x0 - eps`` of ``latent`` at cleanness ``t``."""
    from h3 import patchify, unpatchify

    dit.calls.append({"t": float(t), "extra": video_extra is not None, "delta": video_temb_delta is not None})
    tokens = patchify(latent.float())
    shape = tokens.shape
    h = dit.video_patch_proj(tokens.flatten(1, 3))
    if video_extra is not None:
        h = h + video_extra
    h = h + float(t) * dit.t_vec + 0.1 * text.float().mean()
    if video_temb_delta is not None:
        h = h + video_temb_delta.float().mean()
    for block in dit.blocks:
        h = block(h)
    return unpatchify(dit.head(h).view(shape))


def toy_model(dials=(0.0, 0.3, 0.45, 0.15), lora_last=0, randomize=True):
    from h3_musubi import make_lora

    dit = ToyDit()
    z = torch.randn(1, 24, 2, 4, 6, generator=torch.Generator().manual_seed(11))
    text = torch.ones(1, 3, 5)
    base_v = toy_forward(dit, z, 1.0 - 0.857, text)              # before any LoRA exists
    init_net = make_lora(dit, 4)
    pmd_net = make_lora(dit, 4)
    if randomize:                                                # make both LoRAs do something
        with torch.no_grad():
            for m in init_net.unet_loras + pmd_net.unet_loras:
                m.lora_up.weight.normal_(0, 0.3)
    active = [m for m in pmd_net.unet_loras if lora_last <= 0 or int(m.lora_name.split("_")[3]) >= 3 - lora_last]
    for m in pmd_net.unet_loras:
        m.requires_grad_(m in active)
    noise = P.NoiseInput(16)
    model = P.H3PMD(dit, sigmas=[1.0, 0.947, 0.857, 0.667], dials=list(dials), noise_input=noise, init_net=init_net,
                    pmd_net=pmd_net, active=active, forward=toy_forward)
    return model, z, text, base_v


def test_lora_off_teacher_is_the_base_model():
    model, z, text, base_v = toy_model()
    teacher = model.teacher(z, 0.857, text, 0)
    assert torch.equal(teacher, z + 0.857 * base_v)              # every LoRA off == the DiT before LoRAs
    assert model.dit.calls[-1] == {"t": 1.0 - 0.857, "extra": False, "delta": False}
    assert all(m.enabled for m in model.init_modules + model.pmd_modules)   # modes are restored
    student_v = model.student(z, torch.zeros_like(z), 2, text, 0)
    with model.teacher_modes():
        pass
    assert not torch.allclose(student_v, teacher)                 # the student does run its LoRAs


def test_zero_init_noise_input_leaves_the_starting_model():
    """Dial 0: the student IS the starting model (base + init LoRA, PMD LoRA up = 0) whatever u is. Dial > 0: it is
    the starting model applied to the mixed latent at its effective level (the paper's C.1 wiring)."""
    model, z, text, _ = toy_model(randomize=False)
    with torch.no_grad():
        for m in model.init_modules:
            m.lora_up.weight.normal_(0, 0.3)                       # the init (DiR) LoRA is not identity
    with model.student_modes():
        start = z + 0.947 * toy_forward(model.dit, z, 1.0 - 0.947, text)
    model.dials = [0.0, 0.0, 0.45, 0.15]
    g = torch.Generator().manual_seed(12)
    u1, u2 = torch.randn(z.shape, generator=g), torch.randn(z.shape, generator=g)
    assert torch.allclose(model.student(z, u1, 1, text, 0), start, atol=1e-6)
    assert torch.allclose(model.student(z, u2, 1, text, 0), start, atol=1e-6)
    assert model.dit.calls[-1]["extra"] and abs(model.dit.calls[-1]["t"] - (1 - 0.947)) < 1e-9
    y, st = P.mix_noise(z, u1, 0.857, 0.45)
    with model.student_modes():
        mixed = y + st * toy_forward(model.dit, y, 1.0 - st, text)
    got = model.student(z, u1, 2, text, 0)
    assert torch.allclose(got, mixed, atol=1e-6) and abs(model.dit.calls[-1]["t"] - (1 - st)) < 1e-9
    assert not torch.allclose(got, model.student(z, u2, 2, text, 0))      # stochastic from step 0
    with torch.no_grad():                                         # the new half reads the RAW state once trained
        model.noise_input.proj.weight.normal_(0, 0.1)
    assert not torch.allclose(model.student(z, u1, 2, text, 0), mixed)


def test_dir_head_path_the_default_init():
    """The default init (the published DiR run) sends the head through every student call: x0_proj of a ZERO
    estimate (its bias rows) added to the noise input, and r_embed[k] as the video timestep delta. The teacher sees
    neither. At dial 0 the student is the DiR generator's own call with that estimate."""
    from dir_h3 import DirHead

    model, z, text, _ = toy_model()
    head = DirHead(16, 8, 4)
    with torch.no_grad():
        head.r_embed.normal_(0, 0.5)
        head.x0_proj.bias.normal_(0, 0.5)                          # trained from zero-init: the bias moved
    model.head = head
    model.dials = [0.0, 0.0, 0.0, 0.0]
    seen = {}

    def spy(dit, latent, t, text, *, video_extra=None, video_temb_delta=None, audio_generator=None):
        seen.update(extra=None if video_extra is None else tuple(video_extra.shape),
                    delta=None if video_temb_delta is None else video_temb_delta.clone())
        return toy_forward(dit, latent, t, text, video_extra=video_extra, video_temb_delta=video_temb_delta)

    model.forward = spy
    got = model.student(z, torch.randn_like(z), 2, text, 0)
    assert seen["extra"] == (1, 2 * 2 * 3, 16) and torch.equal(seen["delta"], head.r_embed[2])
    rows = head.x0_proj.bias.view(1, 1, -1).expand(1, 12, 16)
    with model.student_modes():
        want = z + 0.857 * toy_forward(model.dit, z, 1.0 - 0.857, text, video_extra=rows, video_temb_delta=head.r_embed[2])
    assert torch.allclose(got, want, atol=1e-6)
    model.teacher(z, 0.857, text, 0)
    assert seen["extra"] is None and seen["delta"] is None


def test_replayed_gradients_equal_direct_autograd():
    model, z, text, _ = toy_model(lora_last=1)
    with torch.no_grad():
        model.noise_input.proj.weight.normal_(0, 0.05)
    student, teacher_at = model.bind(text, 0)
    g = torch.Generator().manual_seed(13)
    tilts = P.sample_tilts(z, 2, factor=(1, 2, 2), generator=g)
    tilts = [10.0 * t for t in tilts]                              # tiny toy: make the weights non-uniform
    noises = [torch.randn(z.shape, generator=g) for _ in range(4)]
    params = [p for m in model.pmd_modules for p in m.parameters() if p.requires_grad] + [model.noise_input.proj.weight]
    loss, stats = P.pmd_step(student, teacher_at(0.857), z, 0.857, 2, tilts, noises)
    replay = [p.grad.clone() for p in params]
    for p in params:
        p.grad = None
    rho = P.rho_of(0.857)
    with torch.no_grad():
        targets = [teacher_at(0.857)(z + rho * t)[0] for t in tilts]
    draws = torch.stack([student(z, u, 2)[0] for u in noises])
    direct = P.pmd_loss(draws, [t[0] for t in tilts], targets)
    direct.backward()
    assert torch.allclose(loss, direct.detach(), atol=1e-5)
    for a, p in zip(replay, params):
        assert torch.allclose(a, p.grad, atol=1e-5, rtol=1e-4)
    inactive = [p for m in model.pmd_modules if m not in [*model.pmd_modules[-4:]] for p in m.parameters()]
    assert all(p.grad is None for p in inactive)                  # --lora-last 1: earlier blocks untouched
    assert sum(float(p.grad.abs().sum()) for p in params[:-1]) > 0 and float(params[-1].grad.abs().sum()) > 0
    assert 1.0 <= stats["ess"] <= 4.0 and stats["shift_ratio"] > 0


def test_any_step_and_rollout_shapes():
    model, z, text, _ = toy_model()
    student, _t = model.bind(text, 0)
    sig = model.sigmas
    g = torch.Generator().manual_seed(14)
    four = P.any_step(student, z, sig, [0, 1, 2, 3], eta=0.9, generator=g)
    two = P.any_step(student, z, sig, [0, 2], eta=0.9, generator=g)
    assert len(four) == 4 and len(two) == 2 and all(x.shape == z.shape for x in four + two)
    ts = [c["t"] for c in model.dit.calls[-2:]]
    st2 = P.mix_noise(z, z, sig[2], model.dials[2])[1]
    assert abs(ts[0] - 0.0) < 1e-9 and abs(ts[1] - (1 - st2)) < 1e-9   # level 0 (dial 0) then level 2
    for bad in ([1, 2], [0, 0], [0, 4], []):
        try:
            P.any_step(student, z, sig, bad, eta=0.9, generator=g)
            raise AssertionError(f"levels {bad} must be refused")
        except ValueError:
            pass
    state = P.rollout(student, z, sig, 3, eta=0.9, generator=g)
    assert state.shape == z.shape and not state.requires_grad
    assert torch.equal(P.rollout(student, z, sig, 0, eta=0.9, generator=g), z)


def test_gan_pieces():
    head = P.DiscHead(16)
    f = torch.randn(1, 10, 16)
    assert head(f).shape == (1,)
    r, k = torch.tensor([3.0]), torch.tensor([-3.0])
    assert float(P.d_loss(r, k)) < 0.1 and float(P.g_loss(k)) > 3.0
    g = torch.Generator().manual_seed(15)
    assert all(0.001 <= P.gan_level(g, "cpu") <= 0.999 for _ in range(50))
    model, z, text, _ = toy_model()
    feats = P.teacher_features(model, z, 0.5, text, block=1, generator=g, audio_seed=0)
    assert feats.shape == (1, 2 * 2 * 3, 16) and model.dit.calls[-1]["extra"] is False


def test_train_loop_saves_and_resumes_on_cpu():
    """``train`` end to end on the toy model: accumulation, a save, ``pmd_last.pt``, and a resume that continues."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="pmd_test_") as tmp:
        _train_loop(Path(tmp))


def _train_loop(out: Path):
    import argparse
    import json

    built = {}

    def build(args, dev, train):
        model, _z, _text, _ = toy_model(randomize=False)
        for m in model.init_modules:
            m.enabled = False
        built["model"] = model
        pmd_net = type("Net", (), {})()
        pmd_net.unet_loras = model.pmd_modules
        pmd_net.state_dict = lambda: {f"{m.lora_name}.{k}": v for m in model.pmd_modules for k, v in m.state_dict().items()}

        def load_state_dict(sd, strict=False):
            for m in model.pmd_modules:
                m.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith(m.lora_name + ".")})
            return type("R", (), {"unexpected_keys": []})()
        pmd_net.load_state_dict = load_state_dict
        active = [m for m in model.pmd_modules]
        return model, pmd_net, active, {"init_config": {}, "sigmas": model.sigmas, "dials": model.dials}

    saved = dict(P.HOOKS)
    P.HOOKS.update(guard=lambda: None, prompts=lambda a: ([out], ["p0", "p1"]),
                   load_text=lambda _p: torch.ones(1, 3, 5), build_model=build)

    def args_for(steps):
        ns = argparse.Namespace(out=out, check=False, fresh=False, cache=None, holdout=0, size="64x32", frames=2,
                                steps=steps, minutes=0, lr=1e-3, noise_lr=None, betas=[0.9, 0.999], weight_decay=0.0,
                                grad_clip=10.0, accum=2, draws=4, tilts=2, radius=[0.25, 0.5], tilt_factor=[1, 1, 2],
                                estimator="jk", train_levels="1,2,3", level_schedule="mod", eta=0.9, teacher_cfg=1.0,
                                neg_te=None, gan_weight=0.0, d_updates=0, d_lr=5e-5, d_block=1, save_every=2,
                                log_every=1, seed=0, device="cpu", init_run=None, swap=0, rank=4, lora_last=0)
        return ns

    import os
    os.environ["LOT_H3_TRAIN"] = "1"
    try:
        P.train(args_for(3))
        last = torch.load(out / "pmd_last.pt", weights_only=False)
        assert last["step"] == 3 and (out / "pmd_2.pt").exists() and (out / "pmd_final.pt").exists()
        assert "opt" in last and float(last["noise_input"]["proj.weight"].abs().sum()) > 0   # the new half moved
        lines = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
        assert [r["step"] for r in lines] == [1, 2, 3] and all(math.isfinite(r["loss"]) for r in lines)
        P.train(args_for(5))                                       # resumes at 3, stops at 5
        assert torch.load(out / "pmd_last.pt", weights_only=False)["step"] == 5
        assert [json.loads(x)["step"] for x in (out / "log.jsonl").read_text().splitlines()][-2:] == [4, 5]
    finally:
        P.HOOKS.clear()
        P.HOOKS.update(saved)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok {name}", flush=True)
    print(f"ok {len(tests)}")
