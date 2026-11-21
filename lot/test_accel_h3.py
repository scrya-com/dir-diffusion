"""CPU tests for ``accel_h3``: everything that depends on the canvas size or the clip length. No weights, no GPU.

    python lot/test_accel_h3.py
"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party"))

import accel_h3 as A  # noqa: E402
from dwd_h3 import PATCH, DWDHead, ext_out, pack_rows, pixel_frame_count, unpack_rows  # noqa: E402
from refine_dwd_h3 import CONTEXT_CH, Crops, Refiner, crop_side, refine_clip  # noqa: E402

SIZES = ((768, 512), (512, 768), (512, 512), (1024, 576), (1344, 768), (672, 384), (384, 256), (96, 64), (32, 32))


def raises(fn, *args) -> bool:
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def test_sizes():
    assert A.parse_size("768x512") == (768, 512) and A.parse_size(" 1344X768 ") == (1344, 768)
    assert raises(A.parse_size, "768") and raises(A.parse_size, "768x512x3")
    for w, h in SIZES:
        A.check_size(w, h)
    assert raises(A.check_size, 770, 512) and raises(A.check_size, 768, 500) and raises(A.check_size, 0, 512)
    assert raises(A.check_size, 1344, 768, 64) is False and raises(A.check_size, 672, 384, 64)   # half-res tracks: 64
    assert A.parse_job("1024x576", 39) == (1024, 576, 0, 39)
    assert A.parse_job("1024x576:300", 39) == (1024, 576, 300, 39)
    assert A.parse_job("1024x576:300:22", 39) == (1024, 576, 300, 22)
    assert raises(A.parse_job, "1024x576:3:40", 39) and raises(A.parse_job, "1000x576", 39)
    assert A.folder_name(768, 512, 39) == "768x512" and A.folder_name(768, 512, 22) == "768x512_f22"


def test_clip_lengths_match_h3():
    from h3_musubi import latent_frames, pixel_frames

    assert [A.clip_latent_frames(f) for f in (1, 5, 22, 39, 56, 73)] == [1, 2, 7, 12, 17, 22]
    assert A.clip_latent_frames(40) is None and A.clip_latent_frames(2) is None
    for t in (1, 2, 5, 6, 7, 12, 17, 22):
        assert A.clip_latent_frames(pixel_frames(t)) == t
    for frames in (22, 39, 56, 73):                              # the clip lengths H3 itself snaps to
        assert A.clip_latent_frames(frames) == latent_frames(frames)
    for (w, h) in SIZES:
        t, th, tw = A.token_shape(w, h, 39)
        assert (t, th * PATCH, tw * PATCH) == (12, h, w)


def test_unpack_rows_inverts_pack_rows():
    torch.manual_seed(0)
    for t, th, tw in ((1, 1, 1), (2, 3, 5), (12, 16, 24), (7, 24, 42), (5, 8, 6)):
        latent = torch.randn(1, 24, t, 2 * th, 2 * tw)
        rows = pack_rows(latent)
        assert rows.shape == (t * th * tw, 96)
        assert torch.equal(unpack_rows(rows, t, th, tw), latent)


def test_interleave_and_spread():
    assert A.interleave([[1, 2, 3], [10], [20, 21]]) == [1, 10, 20, 2, 21, 3]
    jobs = A.spread([[("a", i) for i in range(100)], [("b", i) for i in range(10)], [("c", i) for i in range(3)]])
    assert len(jobs) == 113 and [j for j in jobs if j[0] == "a"] == [("a", i) for i in range(100)]
    half = jobs[:57]                                             # cut anywhere: every list is equally far along
    for name, total in (("a", 100), ("b", 10), ("c", 3)):
        assert abs(sum(j[0] == name for j in half) / total - 0.5) <= 0.5 / total + 0.02, name
    assert A.spread([[], [1]]) == [1]


def small_head(halo: int | None) -> DWDHead:
    torch.manual_seed(1)
    head = DWDHead(hidden=24, state_dim=96, out=ext_out(halo) if halo is not None else 4 * 3 * PATCH * PATCH)
    with torch.no_grad():
        head.weight.normal_(std=0.02)
        head.bias.normal_(std=0.1)
        head.mean.normal_()
        head.scale.uniform_(0.5, 2.0)
    return head


def test_head_decodes_any_grid_and_clip_length():
    """The head has no size anywhere: every token grid and clip length gives ``(1, 3, F, 32 th, 32 tw)``, and a
    token's pixels do not depend on the grid it sits in (a plain head; an extended head blends neighbours)."""
    plain, extended = small_head(None), small_head(4)
    for frames, th, tw in ((39, 2, 3), (39, 3, 2), (22, 1, 4), (5, 4, 4), (1, 2, 2), (56, 1, 1)):
        t = A.clip_latent_frames(frames)
        n = t * th * tw
        h, s = torch.randn(n, 24), torch.randn(n, 96)
        for head in (plain, extended):
            assert head.decode(h, s, t, th, tw).shape == (1, 3, frames, th * PATCH, tw * PATCH)
    t, th, tw = 2, 2, 3
    h, s = torch.randn(t * th * tw, 24), torch.randn(t * th * tw, 96)
    whole = plain.decode(h, s, t, th, tw)
    rows = torch.arange(t * th * tw).reshape(t, th, tw)[:, :1, 1:].flatten()              # a 1 x 2 sub-grid
    part = plain.decode(h[rows], s[rows], t, 1, 2)
    assert torch.allclose(part, whole[..., :PATCH, PATCH:], atol=1e-6)


def test_half_head_matches():
    head = small_head(4)
    t, th, tw = 2, 2, 2
    h, s = torch.randn(t * th * tw, 24), torch.randn(t * th * tw, 96)
    want = head.decode(h, s, t, th, tw)
    try:
        got = small_head(4).half_weights().decode(h, s, t, th, tw)
    except RuntimeError as error:                                # fp16 matmul missing on this CPU build
        print(f"   (skipped: {str(error)[:60]})")
        return
    assert head.mean.dtype == torch.float32 and got.dtype == torch.float32
    assert (got - want).abs().max() < 2e-2, (got - want).abs().max()


def test_refiner_any_size():
    torch.manual_seed(2)
    model = Refiner(8, latent=CONTEXT_CH, grid=16, grid_blocks=1).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    for frames, h, w in ((5, 64, 96), (5, 96, 64), (9, 32, 32), (5, 48, 112)):
        t = A.clip_latent_frames(frames)
        clip, latent = torch.rand(1, 3, frames, h, w) * 2 - 1, torch.randn(1, 24, t, h // 16, w // 16)
        out = refine_clip(model, clip, latent, chunk=4)
        assert out.shape == clip.shape and torch.isfinite(out).all()
    assert [A.refiner_chunk(h, w) for w, h in ((384, 256), (672, 384), (768, 512), (1024, 576), (1344, 768), (2048, 1152))] \
        == [8, 6, 4, 2, 1, 1]
    assert crop_side(384, 512, 768, 16) == 384 and crop_side(384, 256, 384, 16) == 256
    assert crop_side(384, 250, 400, 16) == 240 and crop_side(256, 768, 1344, 16) == 256


def test_crops_mixed_sizes_batch_together():
    """Clips smaller than the crop still train, and a batch never mixes crop sizes."""
    from train_dwd_h3 import write_mp4

    torch.manual_seed(3)
    with tempfile.TemporaryDirectory() as tmp:
        tmp, pairs = Path(tmp), []
        for k, (h, w) in enumerate(((96, 160), (64, 96), (112, 96), (48, 64))):      # the last is too small: skipped
            frames = 5
            for name in ("dwd", "vae"):
                write_mp4(tmp / f"{k}_{name}.mp4", torch.rand(1, 3, frames, h, w) * 2 - 1)
            saved = {"latent": torch.randn(1, 24, 2, h // 16, w // 16)}
            if k == 1:                                               # a two-track box: its region is the left half
                saved["weight"] = torch.zeros(2, h // 16, w // 16)
                saved["weight"][..., : w // 32] = 1
            torch.save(saved, tmp / f"{k}.lat.pt")
            pairs.append({"dwd": str(tmp / f"{k}_dwd.mp4"), "vae": str(tmp / f"{k}_vae.mp4"), "lat": str(tmp / f"{k}.lat.pt"),
                          "held": False, "folder": f"{w}x{h}"})
        data = Crops(pairs, crop=80, per_clip=6, seed=0, pool=4, context=True, group=3)
        loader = torch.utils.data.DataLoader(data, batch_size=3, num_workers=0)
        seen = set()
        partial = 0
        for i, (x, y, lat, weight) in enumerate(loader):         # default_collate raises on a mixed batch
            c = x.shape[-1]
            assert x.shape == (3, 2, 9, c, c) and y.shape == (3, 2, 3, c, c) and lat.shape == (3, 2, CONTEXT_CH, c // 16, c // 16)
            assert weight.shape == (3, 2, 1, c // 16, c // 16) and weight.min() >= 0 and weight.max() <= 1
            partial += int((weight.flatten(1).mean(1) < 1).sum())
            assert c == 64 or bool((weight == 1).all())              # only the 64 px clip carries a region
            seen.add(c)
            if i >= 12:
                break
        assert seen == {80, 64} and partial > 0, (seen, partial)
        from refine_dwd_h3 import clip_weight, weight_px
        assert clip_weight(pairs[0]) is None and clip_weight(pairs[1]).shape == (2, 4, 6)
        w = weight_px(clip_weight(pairs[1]), [0, 1, 1, 1, 1])
        assert w.shape == (1, 1, 5, 64, 96) and w[..., :32].min() == 1 and w[..., 64:].max() == 0


class ToyBlock(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.a, self.b, self.lora = torch.randn(6, 6, generator=g) * 0.3, torch.randn(6, 6, generator=g) * 0.3, False

    def forward(self, hidden, temb, _segments, _rotation):
        return hidden + torch.tanh(hidden @ (self.a + (self.b if self.lora else 0))) + temb


def test_rerun_blocks_is_a_full_forward_with_the_lora_on():
    from h3_musubi import rerun_blocks, run_blocks

    class Dit:
        blocks_to_swap, gradient_checkpointing, training = 0, False, False

    dit = Dit()
    dit.blocks = [ToyBlock(i) for i in range(6)]
    dit.video_patch_proj = torch.nn.Linear(1, 1)
    hidden, temb = torch.randn(1, 5, 6), torch.randn(6) * 0.1

    @contextlib.contextmanager
    def lora():
        for block in dit.blocks[4:]:
            block.lora = True
        try:
            yield
        finally:
            for block in dit.blocks[4:]:
                block.lora = False

    base = run_blocks(dit, hidden, temb, None, None)
    tap = {"first": 4, "lora": lora}
    assert torch.equal(run_blocks(dit, hidden, temb, None, None, tap=tap), base)       # the first pass is the base
    again = rerun_blocks(dit, tap, temb, None, None)
    with lora():
        want = run_blocks(dit, hidden, temb, None, None)
    assert torch.equal(again, want) and not torch.equal(again, base)
    assert "hidden" not in tap and not any(block.lora for block in dit.blocks)
    dit.blocks_to_swap = 8
    tap = {"first": 4, "lora": lora, "hidden": hidden}
    try:
        rerun_blocks(dit, tap, temb, None, None)
        raise AssertionError("a streamed DiT must be refused")
    except RuntimeError:
        pass


def test_rollout_bookkeeping():
    """``dir_rollout`` keeps each iteration's inputs as the collector stores them and stops at the exit."""
    import dir_h3

    calls = []

    def fake_step(_dit, _head, state, estimate, r, _text, *, sigma, audio_generator=None, **_):
        calls.append(r)
        return 0.5 * state + 0.25 * estimate + r

    real, dir_h3.generator_step = dir_h3.generator_step, fake_step
    try:
        sigmas = [1.0, 0.9, 0.8, 0.6]
        for shape in ((1, 24, 2, 4, 6), (1, 24, 7, 6, 4)):
            z = torch.randn(shape)
            full = A.dir_rollout(None, None, None, z, sigmas, 3, keep=(3, 2, 1))
            assert calls[-4:] == [0, 1, 2, 3] and set(full["inputs"]) == {1, 2, 3}
            for r in (1, 2, 3):                                  # iteration r replayed from its stored inputs
                state, prev = full["inputs"][r]
                nxt = fake_step(None, None, state, prev, r, None, sigma=sigmas[r])
                if r == 3:
                    assert torch.equal(nxt, full["estimate"])
                else:
                    assert torch.equal(nxt, full["inputs"][r + 1][1])
                    assert torch.equal(dir_h3.next_state(state, nxt, sigmas[r], sigmas[r + 1]), full["inputs"][r + 1][0])
            early = A.dir_rollout(None, None, None, z, sigmas, 3, exit_r=2)
            assert calls[-3:] == [0, 1, 2] and torch.equal(early["estimate"], full["inputs"][3][1])
        assert raises(lambda: A.dir_rollout(None, None, None, z, sigmas, 3, exit_r=4))
    finally:
        dir_h3.generator_step = real


def test_region_decode_pastes_only_the_region():
    """The two-track composite without a VAE: outside the region's box the picture is the LR track x2, at any size."""
    from types import SimpleNamespace

    from mixed_res import TwoTrackPlan
    from two_track_h3 import mask_to_latent

    torch.manual_seed(5)
    for (lr_h, lr_w), frames in (((64, 96), 5), ((96, 64), 22)):
        t = A.clip_latent_frames(frames)
        mask = torch.zeros(t, lr_h // 32, lr_w // 32, dtype=torch.bool)
        mask[:, 0, 0] = True                                         # one LR token: 2 x 2 HR tokens
        plan = TwoTrackPlan(mask)
        stack = SimpleNamespace(head=small_head(4), refiner=None, amp=False, chunk=0)
        lr_px = torch.rand(1, 3, frames, lr_h, lr_w) * 2 - 1
        state_hr = torch.randn(1, 24, t, lr_h // 8, lr_w // 8)
        latent = torch.randn(1, 24, t, lr_h // 8, lr_w // 8)
        weight = mask_to_latent(mask, 4, 0.0)
        assert weight.shape == (1, 1, t, lr_h // 8, lr_w // 8)
        out = A.region_decode(stack, torch.randn(plan.hr_count, 24), state_hr, plan.hr_index, latent, weight, lr_px, t,
                              margin=0)
        up = A.upscale_px(lr_px)
        assert out.shape == up.shape == (1, 3, frames, 2 * lr_h, 2 * lr_w)
        assert torch.equal(out[..., 64:, :], up[..., 64:, :]) and torch.equal(out[..., 64:], up[..., 64:])
        assert (out[..., :64, :64] - up[..., :64, :64]).abs().mean() > 0.05
        nothing = A.region_decode(stack, torch.randn(plan.hr_count, 24), state_hr, plan.hr_index, latent,
                                  torch.zeros_like(weight), lr_px, t)
        assert torch.equal(nothing, up)


def test_score_and_tables():
    torch.manual_seed(4)
    for frames, h, w in ((39, 64, 96), (22, 96, 64)):
        t = A.clip_latent_frames(frames)
        clip = torch.rand(1, 3, frames, h, w) * 2 - 1
        same = A.score_clip(clip, clip, t)
        assert same["psnr"] == float("inf") and same["tde"] == 0 and same["flicker_in"] == same["flicker_in_vae"]
        worse = A.score_clip((clip + 0.1 * torch.randn_like(clip)).clamp(-1, 1), clip, t)
        assert 15 < worse["psnr"] < 30 and worse["tde"] > 0
    table = A.quality_table({"1344x768": {"head": {**worse, "lpips": 0.3, "n": 4, "decode_s": 0.01}}})
    assert "| 1344x768 | head | 4 |" in table
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for canvas, total in (("1344x768", 4.0), ("640x384", 1.0)):
            cols = ["dir_vae", "dir_dwd", "dir_dwd_early", "tt_dwd"]
            (root / canvas).mkdir()
            (root / canvas / "meta.json").write_text(json.dumps({
                "canvas": canvas, "frames": 39, "iters": 4, "early_exit": 2, "columns": cols, "prompts": ["a", "b", "c"],
                "hr_share": [0.15, 0.17], "gpu": "test",
                "seconds": {c: [9.0, total, total] for c in cols}, "dit_seconds": {c: [8.0, total / 2, total / 2] for c in cols},
                "decode_seconds": {c: [1.0, total / 2, total / 2] for c in cols},
                "psnr": {c: [None, 30.0, 31.0] if "dwd" in c else [None] * 3 for c in cols},
                "lpips": {c: [None, 0.1, 0.12] if "dwd" in c else [None] * 3 for c in cols},
                "peak_gb": {c: [25.0, 24.0, 24.0] for c in cols}}))
        text = A.bench_table(root)
        assert "| 1344x768 | 39 | DiR, VAE decode | 4.0 | 2.0 | 2.0 | 0.41 |  |  | 24.0 |" in text, text
        assert "| 640x384 | 39 | DiR + DWD | 1.0 | 0.5 | 0.5 | 1.62 | 30.5 | 0.11 | 24.0 |" in text, text
        assert "early exit (after iteration 3 of 4)" in text and "(16% region)" in text
        assert (root / "BENCH.md").read_text() == text


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("ok", len(tests))
