"""CPU tests for ``dwd_h3``: the token <-> pixel mapping and the head's algebra. No weights, no GPU.

    python lot/test_dwd_h3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party"))

from dwd_h3 import (OUT, PATCH, DWDHead, frames_of, pack_rows, pixel_frame_count, pixels_to_tokens,  # noqa: E402
                    pseudo_huber, psnr, tokens_to_pixels)


def test_frame_counts():
    assert frames_of(12) == [1, 4, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4]
    assert pixel_frame_count(12) == 39 and pixel_frame_count(1) == 1 and pixel_frame_count(22) == 73
    from h3_musubi import pixel_frames
    assert all(pixel_frames(t) == pixel_frame_count(t) for t in (1, 2, 5, 6, 7, 12, 22))


def test_round_trip():
    for latent_t, th, tw in ((1, 3, 2), (12, 2, 3), (7, 1, 1)):
        f = pixel_frame_count(latent_t)
        pixels = torch.rand(1, 3, f, th * PATCH, tw * PATCH) * 2 - 1
        tokens = pixels_to_tokens(pixels, latent_t)
        assert tokens.shape == (latent_t * th * tw, OUT)
        back = tokens_to_pixels(tokens, latent_t, th, tw)
        assert torch.equal(back, pixels), (latent_t, th, tw)


def test_one_frame_tokens_repeat_slot0():
    pixels = torch.rand(1, 3, pixel_frame_count(6), PATCH, PATCH)
    tokens = pixels_to_tokens(pixels, 6).reshape(6, 4, 3, PATCH, PATCH)
    for k in (0, 5):                       # one-frame latent frames
        assert all(torch.equal(tokens[k, 0], tokens[k, s]) for s in (1, 2, 3))
    assert not torch.equal(tokens[1, 0], tokens[1, 1])


def test_token_order_matches_latent_rows():
    """Row n of the pixel targets covers the same (frame, y, x) as row n of the packed latent."""
    latent_t, lh, lw = 7, 4, 6
    th, tw = lh // 2, lw // 2
    k = torch.arange(latent_t).view(latent_t, 1, 1)
    i = torch.arange(lh).view(1, lh, 1) // 2
    j = torch.arange(lw).view(1, 1, lw) // 2
    ident = (k * 1000 + i * 100 + j).float()
    latent = ident.unsqueeze(0).unsqueeze(0).expand(1, 24, -1, -1, -1).contiguous()
    rows = pack_rows(latent)
    from musubi_tuner.minimax_h3.packing import pack_video_rows
    assert torch.equal(rows, pack_video_rows(latent)[0])
    group = torch.tensor([kk for kk, n in enumerate(frames_of(latent_t)) for _ in range(n)]).view(-1, 1, 1)
    y = torch.arange(th * PATCH).view(1, -1, 1) // PATCH
    x = torch.arange(tw * PATCH).view(1, 1, -1) // PATCH
    pid = (group * 1000 + y * 100 + x).float()
    pixels = pid.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1, -1).contiguous()
    tokens = pixels_to_tokens(pixels, latent_t)
    assert torch.equal(tokens[:, 0], rows[:, 0])
    assert torch.equal(tokens.amin(1), tokens.amax(1))


def test_head_and_losses():
    head = DWDHead(hidden=8, state_dim=4, out=6)
    h, s = torch.randn(5, 8), torch.randn(5, 4)
    assert torch.equal(head(h, s), torch.zeros(5, 6))
    head.mean.copy_(torch.randn(12))
    head.linear.bias.data.fill_(0.5)
    assert torch.allclose(head(h, s), torch.full((5, 6), 0.5))
    a = torch.rand(10, 100)
    assert pseudo_huber(a, a).abs().item() < 1e-6
    assert pseudo_huber(a, a + 1).item() > 9.0
    assert psnr(a, a) == float("inf")
    assert abs(psnr(torch.zeros(4), torch.full((4,), 0.2)) - 20.0) < 1e-4


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"ok {len(tests)}")
