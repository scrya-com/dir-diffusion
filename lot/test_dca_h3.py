#!/usr/bin/env python3
"""CPU tests for dca_h3 (no weights, no GPU) on a tiny random MiniMax-H3 built from musubi-tuner's own classes.

    CUDA_VISIBLE_DEVICES="" python3 lot/test_dca_h3.py        # last line "ok N"
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

TESTS = []
DEPTH = 8


def test(fn):
    TESTS.append(fn)
    return fn


def tiny_h3(num_layers: int = DEPTH, dtype=torch.float32):
    """A ``num_layers``-block, 64-wide MiniMax-H3 with random weights (as ``test_lot._tiny_h3``)."""
    import h3_musubi  # noqa: F401  (puts MUSUBI_SRC on sys.path)
    from musubi_tuner.minimax_h3.model import MiniMaxH3Config, MiniMaxH3Model

    config = MiniMaxH3Config(hidden_size=64, num_layers=num_layers, num_attention_heads=2, attention_head_dim=32,
                             ffn_hidden_size=128, text_dim=16, timestep_input_dim=16, time_embed_hidden_size=32,
                             time_embed_dim=16, rope_inv_freq_len=4)
    torch.manual_seed(0)
    dit = MiniMaxH3Model(config, dtype=dtype, device="cpu")
    with torch.no_grad():
        for param in dit.parameters():
            param.normal_(0.0, 0.3)       # large enough that the AdaLN gates let rows talk
    return dit.eval().requires_grad_(False)


def inputs(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 24, 2, 4, 6, generator=g), torch.randn(1, 5, 16, generator=g)


def forward(dit, latent, text, t: float = 0.4):
    """``forward_video``: the velocity and the block stack's output on the video rows."""
    from h3_musubi import forward_video

    out = {}
    velocity = forward_video(dit, latent, t, text, audio_generator=torch.Generator().manual_seed(7), out=out)
    return velocity, out["hidden"]


def randomise(dca, scale: float = 0.05, seed: int = 2):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in dca.parameters():
            param.copy_(torch.randn(param.shape, generator=g) * scale)
    return dca


@test
def test_identity_at_init():
    from dca_h3 import DcaH3, install

    dit = tiny_h3()
    latent, text = inputs()
    with torch.no_grad():
        want_v, want_h = forward(dit, latent, text)
        for k in (0, 1, 2, 4, DEPTH):
            dca = DcaH3.for_dit(dit, k=k)
            install(dit, dca)
            # zero parameters: every mix is the stream itself, the block's own fused attention runs
            got_v, got_h = forward(dit, latent, text)
            assert torch.equal(got_v, want_v) and torch.equal(got_h, want_h), k
            assert dca.stats["general"] is False
            # the general path at zero parameters: the mixes are computed (stream + 0 * entry) and Q, K, V are
            # projected separately (three sliced F.linear calls instead of one fused one)
            dca.force_general = True
            got_v, got_h = forward(dit, latent, text)
            assert dca.stats["general"] is True
            diff_h = float((got_h - want_h).abs().max()) / float(want_h.abs().max())
            diff_v = float((got_v - want_v).abs().max()) / float(want_v.abs().max())
            assert diff_h <= 1e-5 and diff_v <= 1e-5, (k, diff_h, diff_v)
            if k == 2:
                dca.qkv_mode = "batched"       # the whole fused projection on the three inputs, diagonal thirds kept
                b_v, b_h = forward(dit, latent, text)
                print(f"  identity at init, k=2: skip path bit-identical; general path relative max diff "
                      f"hidden {diff_h:.2e} velocity {diff_v:.2e} (sliced QKV), "
                      f"{float((b_h - want_h).abs().max()) / float(want_h.abs().max()):.2e} (batched QKV); tolerance 1e-5")
            install(dit, None)
        again_v, _ = forward(dit, latent, text)
        assert torch.equal(again_v, want_v)                    # taken off again


@test
def test_identity_at_init_bf16():
    """bf16 is the deployment dtype: the general path must not drift there either."""
    from dca_h3 import DcaH3, install

    dit = tiny_h3(dtype=torch.bfloat16)
    latent, text = inputs()
    with torch.no_grad():
        want_v, want_h = forward(dit, latent, text)
        dca = DcaH3.for_dit(dit, k=2)
        install(dit, dca)
        got_v, got_h = forward(dit, latent, text)
        assert torch.equal(got_v, want_v) and torch.equal(got_h, want_h)
        dca.force_general = True
        got_v, got_h = forward(dit, latent, text)
        diff = float((got_h.float() - want_h.float()).abs().max()) / float(want_h.float().abs().max())
        print(f"  bf16: skip path bit-identical; general path relative max diff on the hidden states {diff:.2e}")
        assert diff <= 0.05, diff                              # bf16 has 8 mantissa bits; 8 blocks deep


@test
def test_mix_is_the_papers_grn_v3():
    """``stream + sum (beta + relu(w.G)) G`` equals the paper's ``sum (b + relu(w.G)) G`` with ``b = 1 + beta``
    when the stack entries sum to the stream."""
    from dca_h3 import Grn

    g = torch.Generator().manual_seed(0)
    entries = [torch.randn(1, 7, 16, generator=g, dtype=torch.float64) for _ in range(4)]
    stream = sum(entries)
    grn = Grn(16, 4, 3).double()
    with torch.no_grad():
        grn.beta.copy_(torch.randn(grn.beta.shape, generator=g, dtype=torch.float64) * 0.3)
        grn.w.copy_(torch.randn(grn.w.shape, generator=g, dtype=torch.float64) * 0.3)
        got = grn(stream, list(enumerate(entries)), general=True)
        for m in range(3):
            want = sum((1.0 + grn.beta[m, s] + torch.relu(entry @ grn.w[m]).unsqueeze(-1)) * entry
                       for s, entry in enumerate(entries))
            assert torch.allclose(got[m], want, atol=1e-12), m
    grad = grn(stream, list(enumerate(entries)), general=True)        # the autograd (out-of-place) branch agrees
    assert all(torch.allclose(a, b, atol=1e-12) for a, b in zip(grad, got))


@test
def test_gradients_reach_dca_only():
    from dca_h3 import FIXED_SLOTS, DcaH3, install, slot_names

    dit = tiny_h3()
    latent, text = inputs()
    k = 2
    dca = DcaH3.for_dit(dit, k=k)
    install(dit, dca)
    velocity, _ = forward(dit, latent, text)
    velocity.square().mean().backward()
    assert all(param.grad is None for param in dit.parameters()), "a frozen block got a gradient"
    w_norms, beta_norms = [], []
    for index in range(DEPTH + 1):
        grn = dca.blocks[index] if index < DEPTH else dca.out
        present = sorted(slot_names(index, k))
        assert grn.w.grad is not None and bool((grn.w.grad.abs().amax(dim=1) > 0).all()), f"w of block {index}"
        for slot in range(FIXED_SLOTS + k):
            norm = float(grn.beta.grad[:, slot].abs().max())
            assert (norm > 0) == (slot in present), (index, slot, norm)     # an empty slot gets none
        w_norms.append(float(grn.w.grad.norm()))
        beta_norms.append(float(grn.beta.grad.norm()))
    print(f"  grad norms at init: w {min(w_norms):.2e}..{max(w_norms):.2e}, beta {min(beta_norms):.2e}..{max(beta_norms):.2e};"
          f" DiT parameters with a gradient: 0 of {sum(1 for _ in dit.parameters())}")

    # torch's own ReLU has gradient 0 at 0: w would be stuck at the paper's zero initialisation
    w = torch.zeros(4, requires_grad=True)
    torch.relu(torch.randn(3, 4) @ w).sum().backward()
    assert float(w.grad.abs().max()) == 0.0

    # activation checkpointing (training mode) gives the same gradients
    plain = [param.grad.clone() for param in dca.parameters()]
    dca.zero_grad()
    dit.enable_gradient_checkpointing()
    dit.train()
    velocity, _ = forward(dit, latent, text)
    velocity.square().mean().backward()
    for a, param in zip(plain, dca.parameters()):
        assert torch.allclose(a, param.grad, rtol=1e-4, atol=1e-8)
    install(dit, None)


@test
def test_history_memory_matches_the_formula():
    from dca_h3 import DcaH3, install

    dit = tiny_h3()
    latent, text = inputs()
    with torch.no_grad():
        for k in (0, 1, 2, 4):
            dca = randomise(DcaH3.for_dit(dit, k=k))
            install(dit, dca)
            forward(dit, latent, text)
            stats = dca.stats
            want = dca.history_bytes(stats["tokens"], stats["bytes_per_value"])
            assert stats["history_slots"] == k + 2
            assert stats["history_bytes"] == want == (k + 2) * stats["tokens"] * 64 * 4, (k, stats, want)
            if k == 2:
                print(f"  k=2 history: measured {stats['history_bytes']} bytes = formula (k+2) x {stats['tokens']} tokens"
                      f" x 64 x 4 bytes = {want}")
            install(dit, None)
    # extra parameters: 4 mixes x hidden x (k + 3) per block, plus one output mix
    dca = DcaH3(50, 5376, k=2)
    assert dca.extra_parameters() == 50 * 4 * 5376 * 5 + 5376 * 5 == 5_402_880


@test
def test_dropping_blocks():
    from dca_h3 import DcaH3, install, slot_names

    dit = tiny_h3()
    latent, text = inputs()
    skip = {3, 5}
    with torch.no_grad():
        full_v, _ = forward(dit, latent, text)
        # zero parameters + skip == the plain stack without those blocks
        blocks = dit.blocks
        dit.blocks = torch.nn.ModuleList(block for i, block in enumerate(blocks) if i not in skip)
        want_v, want_h = forward(dit, latent, text)
        dit.blocks = blocks
        dca = DcaH3.for_dit(dit, k=2, skip=skip)
        install(dit, dca)
        got_v, got_h = forward(dit, latent, text)
        assert torch.equal(got_v, want_v) and torch.equal(got_h, want_h)
        assert dca.stats["blocks_run"] == DEPTH - 2 and not torch.equal(got_v, full_v)

        # the window slides over the blocks that ran; an empty slot has no name and no weight
        assert slot_names(0, 2) == {0: "x"}
        assert slot_names(2, 2) == {0: "x", 2: "f1", 3: "f0"}
        assert slot_names(6, 2) == {0: "x", 1: "mid(sum of 4 updates)", 2: "f5", 3: "f4"}
        assert slot_names(6, 2, skip) == {0: "x", 1: "mid(sum of 2 updates)", 2: "f4", 3: "f2"}
        assert slot_names(4, 2, skip) == {0: "x", 1: "mid(sum of 1 updates)", 2: "f2", 3: "f1"}
        weights = dca.slot_weights(4)
        assert set(weights) == {"q", "k", "v", "r"} and all(float(w.min()) == 1.0 for w in weights["q"].values())
        assert set(dca.slot_weights(1)["q"]) == {"x", "f0"}

        # trained-like (random) weights, then drop blocks: finite, and different from the full stack
        randomise(dca)
        dca.skip = frozenset()
        all_v, _ = forward(dit, latent, text)
        dca.skip = frozenset(skip)
        cut_v, _ = forward(dit, latent, text)
        assert bool(torch.isfinite(cut_v).all()) and not torch.equal(cut_v, all_v)
        # weights on slots that are empty for a block do nothing; weights of a dropped block do nothing
        dca.blocks[1].beta[:, 1].add_(3.0)             # block 1 has no "mid" yet
        dca.blocks[0].beta[:, 2:].add_(3.0)            # block 0 has no recent updates
        dca.blocks[3].beta.add_(3.0)                   # dropped
        dca.blocks[3].w.add_(3.0)
        assert torch.equal(forward(dit, latent, text)[0], cut_v)
        dca.blocks[4].beta[:, 2].add_(0.5)             # a slot that is there
        assert not torch.equal(forward(dit, latent, text)[0], cut_v)
        rel = float((cut_v - all_v).abs().mean() / all_v.abs().mean())
        print(f"  dropping blocks {sorted(skip)} of {DEPTH} with random DCA weights: finite, mean relative change {rel:.2f}")
    install(dit, None)


@test
def test_qkv_paths_agree_and_other_models_are_untouched():
    from dca_h3 import DcaH3, install

    dit, other = tiny_h3(), tiny_h3()
    latent, text = inputs()
    with torch.no_grad():
        want_other = forward(other, latent, text)[0]
        dca = randomise(DcaH3.for_dit(dit, k=2))
        install(dit, dca)
        sliced = forward(dit, latent, text)[0]
        dca.qkv_mode = "batched"
        batched = forward(dit, latent, text)[0]
        diff = float((sliced - batched).abs().max()) / float(sliced.abs().max())
        assert diff <= 1e-5, diff
        assert not torch.equal(sliced, forward(other, latent, text)[0])        # random weights change the output
        assert torch.equal(forward(other, latent, text)[0], want_other)        # the patch is per model
    install(dit, None)


if __name__ == "__main__":
    for fn in TESTS:
        fn()
        print("ok", fn.__name__)
    print("ok", len(TESTS))
