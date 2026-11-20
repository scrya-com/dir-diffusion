"""CPU tests for the refiner's latent-grid path, ``--init`` and ``bench``. No data, no GPU.

    python lot/test_refine_dwd_h3.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_dwd_h3 import CONTEXT_CH, Refiner, frame_context, init_from, neighbours, refine_clip, spec_of  # noqa: E402


def randomise(model: Refiner) -> Refiner:
    """Give every weight a value, the zero-initialised ones included, so equalities are not trivially true."""
    torch.manual_seed(0)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return model


def clip_and_latent(frames: int = 9, h: int = 64, w: int = 96):
    torch.manual_seed(1)
    return torch.rand(1, 3, frames, h, w) * 2 - 1, torch.randn(1, 24, 3, h // 16, w // 16)     # 9 frames = 3 latent


def test_init_reproduces_source():
    clip, latent = clip_and_latent()
    for source_latent in (False, True, CONTEXT_CH):
        src = randomise(Refiner(16, latent=source_latent)).eval()
        saved = {"model": src.state_dict(), "base": 16, "latent": bool(source_latent), "latent_spec": source_latent}
        want = refine_clip(src, clip, latent if source_latent else None)
        new = Refiner(16, latent=CONTEXT_CH, grid=32, grid_blocks=2).eval()
        missing = init_from(new, saved)
        assert missing and all(k.startswith("grid.") for k in missing), missing
        got = refine_clip(new, clip, latent)
        assert torch.allclose(got, want, atol=1e-5), (source_latent, (got - want).abs().max().item())
    old = {"model": randomise(Refiner(16, latent=True)).state_dict(), "base": 16, "latent": True}   # a run E file
    assert spec_of(old) == {"base": 16, "latent": True, "grid": 0, "grid_blocks": 4}
    try:
        init_from(Refiner(24, latent=CONTEXT_CH), old)
    except SystemExit:
        pass
    else:
        raise AssertionError("a different base must be refused")


def test_grid_path_is_used_and_trains():
    clip, latent = clip_and_latent()
    model = Refiner(16, latent=CONTEXT_CH, grid=32, grid_blocks=2)
    x = neighbours(clip[0].permute(1, 0, 2, 3))[:1]                                          # (1, 9, h, w)
    ctx = frame_context(latent[0], [0])
    # out is zero-initialised: the gradient reaches the grid path once out has a value
    torch.nn.init.normal_(model.out.weight, std=0.05)
    model(x, ctx).square().mean().backward()
    assert model.grid.out.weight.grad.abs().sum() > 0
    randomise(model).eval()
    a = model(x, ctx)
    moved = ctx.clone()
    moved[:, :, 1, 2] += 1.0                                                                 # one latent pixel
    d = (model(x, moved) - a).abs().amax(dim=(0, 1))
    assert d[16:32, 32:48].max() > 0
    # where the latent is locally constant and the frame's edge is far, the plain refiner cannot tell the 4x4
    # blocks of a 16x16 latent pixel apart: its strides make it periodic every 4 pixels. The pixel shuffle gives
    # each of those blocks its own weights (nearest-upsampled copy at the input switched off for this check).
    model.inp.weight.data[:, 9:] = 0
    zeros, ones = torch.zeros(1, 9, 320, 320), torch.ones(1, CONTEXT_CH, 20, 20)
    with torch.no_grad():
        cell = model(zeros, ones)[0, :, 144:160, 144:160]
        assert (cell[:, 4:] - cell[:, :-4]).abs().max() > 1e-4
        cell = randomise(Refiner(16, latent=CONTEXT_CH)).eval()(zeros, ones)[0, :, 144:160, 144:160]
        assert (cell[:, 4:] - cell[:, :-4]).abs().max() < 1e-5 and (cell[:, :, 4:] - cell[:, :, :-4]).abs().max() < 1e-5


def test_bench_runs():
    out = subprocess.run([sys.executable, str(Path(__file__).with_name("refine_dwd_h3.py")), "bench", "--device", "cpu",
                          "--base", "8", "--latent-context", "--latent-grid", "16", "--grid-blocks", "1", "--frames", "9",
                          "--height", "32", "--width", "48", "--chunks", "4", "--repeats", "1"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.count("kind=refine_bench amp=") == 2, out.stdout


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
