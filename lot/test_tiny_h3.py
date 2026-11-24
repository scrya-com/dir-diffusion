#!/usr/bin/env python3
"""CPU tests for the block-drop wiring (h3_musubi.blocks_dropped, tiny_h3.plan): no weights, no GPU.

    CUDA_VISIBLE_DEVICES="" python3 lot/test_tiny_h3.py        # last line "ok N"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def tiny_h3(num_layers: int = 8):
    import h3_musubi  # noqa: F401
    from musubi_tuner.minimax_h3.model import MiniMaxH3Config, MiniMaxH3Model

    config = MiniMaxH3Config(hidden_size=64, num_layers=num_layers, num_attention_heads=2, attention_head_dim=32,
                             ffn_hidden_size=128, text_dim=16, timestep_input_dim=16, time_embed_hidden_size=32,
                             time_embed_dim=16, rope_inv_freq_len=4)
    torch.manual_seed(0)
    dit = MiniMaxH3Model(config, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for param in dit.parameters():
            param.normal_(0.0, 0.3)
    return dit.eval().requires_grad_(False)


def forward(dit):
    from h3_musubi import forward_video

    g = torch.Generator().manual_seed(1)
    latent, text = torch.randn(1, 24, 2, 4, 6, generator=g), torch.randn(1, 5, 16, generator=g)
    return forward_video(dit, latent, 0.4, text, audio_generator=torch.Generator().manual_seed(7))


@test
def test_plans():
    from h3_musubi import H3_TINY_DROP
    from tiny_h3 import SCAN_ORDER, kept, plan

    assert plan("full") == () and plan("") == () and kept("full") == 50
    assert plan("h3tiny") == H3_TINY_DROP == (6, 7, 8, 9, 10, 11, 13, 15) and kept("h3tiny") == 42
    assert kept("least10") == 40 and kept("least15") == 35
    assert plan("least8") == (4, 6, 8, 11, 13, 15, 16, 17)             # the cascade branch's least8 set
    assert set(plan("least10")) < set(plan("least15")) and len(set(SCAN_ORDER)) == len(SCAN_ORDER) == 16
    assert plan("3-1-2-3") == (1, 2, 3)
    for bad in ("least0", "least17", "50", "1-99"):
        try:
            plan(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


@test
def test_dropped_blocks_are_skipped_and_restored():
    from h3_musubi import blocks_are_dropped, blocks_dropped, drop_blocks, set_blocks_dropped

    dit = tiny_h3()
    calls = []
    hooks = [block.register_forward_hook(lambda _m, _i, _o, i=i: calls.append(i)) for i, block in enumerate(dit.blocks)]
    with torch.no_grad():
        want = forward(dit)
        assert calls == list(range(8))
        calls.clear()
        with blocks_dropped(dit, ()):                                  # nothing dropped: the same bits
            assert torch.equal(forward(dit), want)
        assert calls == list(range(8))

        # skipped == the stack with those blocks removed, bit for bit; every index and name stays in place
        blocks = dit.blocks
        dit.blocks = torch.nn.ModuleList(block for i, block in enumerate(blocks) if i not in (2, 5))
        removed = forward(dit)
        dit.blocks = blocks
        with blocks_dropped(dit, (2, 5)):
            assert blocks_are_dropped(dit, range(8)) == [i in (2, 5) for i in range(8)]
            got = forward(dit)
        assert torch.equal(got, removed) and not torch.equal(got, want)
        assert len(dit.blocks) == 8 and blocks_are_dropped(dit, range(8)) == [False] * 8
        assert torch.equal(forward(dit), want)                         # restored

        # nesting restores the outer state; drop_blocks is permanent until undone
        drop_blocks(dit, (1,))
        with blocks_dropped(dit, (1, 3)):
            pass
        assert blocks_are_dropped(dit, (1, 3)) == [True, False]
        with blocks_dropped(dit, (1,), dropped=False):                 # run a dropped block inside
            assert torch.equal(forward(dit), want)
        assert blocks_are_dropped(dit, (1,)) == [True]
        set_blocks_dropped(dit, (1,), False)
        assert torch.equal(forward(dit), want)
        try:
            set_blocks_dropped(dit, (8,), True)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range block accepted")
    for hook in hooks:
        hook.remove()


@test
def test_a_dropped_block_does_no_work():
    from h3_musubi import blocks_dropped

    dit = tiny_h3()
    seen = []
    for i, block in enumerate(dit.blocks):
        block.attn.register_forward_hook(lambda _m, _i, _o, i=i: seen.append(i))
    with torch.no_grad(), blocks_dropped(dit, (0, 3, 7)):
        forward(dit)
    assert seen == [1, 2, 4, 5, 6], seen


@test
def test_env_plan():
    from h3_musubi import H3_TINY_DROP, dropped_blocks

    old = os.environ.pop("LOT_H3_DROP_BLOCKS", None)
    try:
        assert dropped_blocks() == ()                                  # unset: every block runs
        os.environ["LOT_H3_DROP_BLOCKS"] = "h3tiny"
        assert dropped_blocks() == H3_TINY_DROP
        os.environ["LOT_H3_DROP_BLOCKS"] = "9, 3 3,1"
        assert dropped_blocks() == (1, 3, 9)
    finally:
        os.environ.pop("LOT_H3_DROP_BLOCKS", None)
        if old is not None:
            os.environ["LOT_H3_DROP_BLOCKS"] = old


@test
def test_metrics():
    from tiny_h3 import colour_shift, hf_energy

    a = torch.zeros(1, 3, 12, 16, 16)
    assert colour_shift(a, a) == 0.0
    assert abs(colour_shift(a + 0.2, a) - 0.1) < 1e-6                  # 0.2 of [-1, 1] is 0.1 of [0, 1]
    assert hf_energy(a) == 0.0 and hf_energy(torch.randn(1, 3, 12, 16, 16)) > 0.1


if __name__ == "__main__":
    for fn in TESTS:
        fn()
        print("ok", fn.__name__)
    print("ok", len(TESTS))
