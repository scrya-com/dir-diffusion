"""CPU tests for ``dwd_h3``: the token <-> pixel mapping and the head's algebra. No weights, no GPU.

    python lot/test_dwd_h3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party"))

from dwd_h3 import (OUT, PATCH, DWDHead, flicker_split, frames_of, pack_rows, pixel_frame_count,  # noqa: E402
                    pixels_to_tokens, pseudo_huber, psnr, seam_ratio, slot_mask, token_classes, tokens_to_pixels)


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
    head = DWDHead(hidden=8, state_dim=4, out=6, classes=5)
    h, s = torch.randn(5, 8), torch.randn(5, 4)
    cls = torch.tensor([0, 1, 2, 3, 4])
    assert torch.equal(head(h, s, cls), torch.zeros(5, 6))
    head.bias.data[:, :] = torch.arange(5.0)[:, None]
    assert torch.equal(head(h, s, cls)[:, 0], torch.arange(5.0))          # each token uses its class's map
    head.mean.copy_(torch.randn(12))
    head.scale.fill_(2.0)
    head.weight.data[2, 0, 0] = 1.0
    x = (torch.cat([h, s], 1) - head.mean) / 2.0
    assert torch.allclose(head(h, s, cls)[2, 0], x[2, 0] + 2.0)
    a = torch.rand(10, 100)
    assert pseudo_huber(a, a).abs().item() < 1e-6
    assert pseudo_huber(a, a + 1).item() > 9.0
    assert psnr(a, a) == float("inf")
    assert abs(psnr(torch.zeros(4), torch.full((4,), 0.2)) - 20.0) < 1e-4


def test_classes_and_mask():
    cls = token_classes(7, 2, 3)
    assert cls.tolist() == [k % 5 for k in range(7) for _ in range(6)]
    m = slot_mask(cls)
    assert m[cls == 0, OUT // 4:].sum() == 0 and m[cls == 0, :OUT // 4].min() == 1 and m[cls == 1].min() == 1


def test_extended_head():
    from dwd_h3 import blend_tokens, ext_halo, ext_out, ext_slot_mask, pixels_to_tokens_ext
    assert ext_halo(OUT) is None and ext_halo(6) is None and ext_halo(ext_out(4)) == 4 and ext_halo(ext_out(0)) == 0
    t, th, tw = 7, 2, 3
    pixels = torch.rand(1, 3, pixel_frame_count(t), th * PATCH, tw * PATCH)
    for halo in (0, 4):
        ext = pixels_to_tokens_ext(pixels, t, halo)
        assert ext.shape == (t * th * tw, ext_out(halo))
        side = PATCH + 2 * halo                                # the own slots' centre is the plain target
        own = ext.reshape(-1, 6, 3, side, side)[:, 1:5, :, halo:halo + PATCH, halo:halo + PATCH]
        assert torch.equal(own.reshape(ext.shape[0], -1), pixels_to_tokens(pixels, t))
        for blend in (0.0, 0.25, 0.5):                          # exact targets blend back to the picture
            assert (blend_tokens(ext, t, th, tw, halo, blend) - pixels).abs().max() < 1e-5
        m = ext_slot_mask(token_classes(t, th, tw), th * tw, halo).reshape(-1, 6, 3 * side * side)[:, :, 0]
        assert m[0].tolist() == [0, 1, 0, 0, 0, 1] and m[th * tw].tolist() == [1] * 6 and m[-1].tolist() == [1, 1, 1, 1, 1, 0]
    # the chunked gradient of a frozen head equals autograd's
    head = DWDHead(hidden=8, state_dim=4, out=ext_out(0), classes=5)
    head.weight.data.normal_(std=0.1)
    head.bias.data.normal_()
    head.scale.fill_(2.0)
    n = t * th * tw
    h, s, cls = torch.randn(n, 8, requires_grad=True), torch.randn(n, 4), token_classes(t, th, tw)
    y = head.targets(pixels_to_tokens(pixels, t), t, th, tw)
    mask = head.target_mask(cls, th * tw)
    loss = (((head(h, s, cls) - y) ** 2) * mask).sum() / mask.sum()
    loss.backward()
    value, grad = head.masked_mse_grad(h.detach(), s, y, mask, cls, chunk=5)
    assert abs(value - loss.item()) < 1e-5 and torch.allclose(grad, h.grad, atol=1e-6)
    assert head.decode(h.detach(), s, t, th, tw).shape == pixels.shape


def test_metrics():
    t = 6                                                     # frames 1 + 4 + 1 = 6 pixel frames... (1,4,4,4,4,1)
    f = pixel_frame_count(t)
    flat = torch.zeros(1, 3, f, 64, 64)
    assert flicker_split(flat, t) == (0.0, 0.0)
    steps = torch.zeros(1, 3, f, 64, 64)
    group = [k for k, n in enumerate(frames_of(t)) for _ in range(n)]
    for i, g in enumerate(group):
        steps[:, :, i] = g * 0.1                              # constant inside a latent frame, jumps between
    inside, across = flicker_split(steps, t)
    assert inside == 0.0 and abs(across - 0.1) < 1e-6
    grid = torch.zeros(1, 3, 1, 64, 64)
    grid[..., 32:, :] = 1.0                                   # a jump exactly on a token edge
    assert seam_ratio(grid) > 10
    smooth = torch.linspace(0, 1, 64).view(1, 1, 1, 1, 64).expand(1, 3, 1, 64, 64)
    assert abs(seam_ratio(smooth) - 1.0) < 0.05


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"ok {len(tests)}")
