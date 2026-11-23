# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""GPU checks for the H3 alpha head: python -m alpha.test_alpha (ComfyUI python, COMFYUI_DIR set)."""
import torch

from .alpha_math import ea_loss_per_sample, ea_score_to_alpha, layerdiffuse_pad_rgb
from .h3_alpha import AlphaHead, install
from .train_alpha import FRAMES, decode_rgba, encode, load_vae


def test_math():
    a = torch.rand(2, 1, 3, 8, 8)
    loss, _, _ = ea_loss_per_sample(torch.zeros_like(a), a)
    assert loss.shape == (2,)
    assert torch.allclose(ea_score_to_alpha(torch.tensor([-1.0, 1.0])), torch.tensor([0.0, 1.0]), atol=1e-6)
    rgb, al = torch.rand(2, 3, 32, 32), (torch.rand(2, 1, 32, 32) > 0.5).float()
    pad = layerdiffuse_pad_rgb(rgb, al)
    assert torch.allclose(pad * al, rgb * al, atol=1e-5)  # opaque pixels keep their colour


def test_rgb_unchanged(m):
    torch.manual_seed(0)
    x = torch.rand(2, 3, FRAMES, 256, 256, device="cuda")
    z = encode(m, x)
    with torch.no_grad():
        ref = m.decode(z.half()).float().cuda()
        head = AlphaHead().cuda()
        uninstall = install(m, head)
        rgb, score = decode_rgba(m, z)
        uninstall()
        again = m.decode(z.half()).float().cuda()
    assert rgb.shape == ref.shape and score.shape == (2, 1, FRAMES, 256, 256), (rgb.shape, score.shape)
    d = (rgb - ref).abs().max().item()
    assert d == 0.0, f"RGB changed by {d}"
    assert torch.equal(again, ref), "uninstall did not restore decode"
    # a bigger-than-tile clip goes through tiling + 2 temporal chunks with the 4th channel
    z2 = encode(m, torch.rand(1, 3, 61, 384, 384, device="cuda"))
    uninstall = install(m, AlphaHead().cuda())
    with torch.no_grad():
        rgb2, score2 = decode_rgba(m, z2)
    uninstall()
    assert score2.shape[2:] == rgb2.shape[2:] and rgb2.shape[-1] == 384, (rgb2.shape, score2.shape)


def test_grad_reaches_head(m):
    head = AlphaHead().cuda()
    uninstall = install(m, head)
    z = encode(m, torch.rand(1, 3, FRAMES, 256, 256, device="cuda"))
    _rgb, score = decode_rgba(m, z)
    ea_loss_per_sample(score, torch.rand_like(score)).__getitem__(0).sum().backward()
    uninstall()
    assert head.proj_out.weight.grad is not None and head.proj_out.weight.grad.abs().sum() > 0


if __name__ == "__main__":
    test_math()
    m = load_vae()
    test_rgb_unchanged(m)
    test_grad_reaches_head(m)
    print("ALPHA_TESTS_OK")
