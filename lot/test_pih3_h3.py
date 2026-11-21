#!/usr/bin/env python3
"""CPU tests for pih3_h3 (no weights, no GPU): the schedule, the carried audio step, the box paste, the registry.

    python3 lot/test_pih3_h3.py        # last line "ok N"
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@test
def test_sigmas_match_comfyui():
    from pih3_h3 import pih3_sigmas

    # ComfyUI-piFlow's pi-Flow Scheduler on ModelSamplingMiniMaxH3(12, 3), as dumped by pih3_comfy_reference.py
    two = pih3_sigmas(2)
    assert len(two) == 3 and two[-1] == 0.0
    assert abs(two[0] - 0.9938) < 1e-4 and abs(two[1] - 0.8653) < 1e-4, two
    four = pih3_sigmas(4)
    assert len(four) == 5 and all(a > b for a, b in zip(four, four[1:])), four
    assert abs(four[0] - 12 * 0.96 / (1 + 11 * 0.96)) < 1e-6


@test
def test_carried_audio_step_is_euler_on_the_carried_variable():
    from h3_musubi import audio_sigma
    from pih3_h3 import AUDIO_SCALE, carried_audio_step

    # a straight audio path with its exact velocity: the step's direction is d/dsigma_v of y = (s_v / s_a) x_a
    a0, eps = torch.randn(1, 32, 2, 7), torch.randn(1, 32, 2, 7)

    def y_at(s):
        sa = float(audio_sigma(s))
        return (s / sa) * ((1 - sa) * a0 + sa * eps)

    s, ds = 0.7, 1e-4
    numeric = (y_at(s - ds) - y_at(s)) / -ds
    sa = float(audio_sigma(s))
    step = carried_audio_step(y_at(s), a0 - eps, s, s - ds)
    assert torch.allclose((step - y_at(s)) / -ds, numeric, atol=2e-2), float(((step - y_at(s)) / -ds - numeric).abs().max())
    assert AUDIO_SCALE == 4.0 and 0 < sa < s


@test
def test_box_paste_keeps_the_box_and_the_outside():
    from pih3_h3 import box_paste

    lr = torch.zeros(1, 3, 5, 128, 192)                    # 4 x 6 LR tokens (32 px; 64 px at full resolution)
    box = torch.ones(1, 3, 5, 128, 128)                    # LR token rows 0..2, cols 1..3
    out = box_paste(lr, box, (0, 2, 1, 3), feather=16)
    assert out.shape == (1, 3, 5, 256, 384)
    assert float(out[..., :, :64].abs().max()) == 0.0      # outside the box: the LR pixels x2
    assert float(out[..., 20:100, 90:170].min()) == 1.0   # inside, away from the edges: the box
    assert float(out[..., 0, 128].min()) == 1.0            # at the frame's own edge there is no ramp


@test
def test_registry():
    from generators import GENERATORS, generator_module

    assert generator_module("dir") is None and "pih3" in GENERATORS
    try:
        generator_module("nope")
    except SystemExit:
        pass
    else:
        raise AssertionError("unknown generator accepted")


if __name__ == "__main__":
    for fn in TESTS:
        fn()
        print("ok", fn.__name__)
    print("ok", len(TESTS))
