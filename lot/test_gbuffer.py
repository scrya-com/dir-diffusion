#!/usr/bin/env python3
"""CPU tests for gbuffer.py with a fake estimator and synthetic sources (no models, no GPU; needs ffmpeg).

    CUDA_VISIBLE_DEVICES="" python3 lot/test_gbuffer.py        # last line "ok N"
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def rgba_still(h=48, w=64, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    yy, xx = np.mgrid[:h, :w]
    image[..., 3] = np.clip(255 - 8 * np.hypot(yy - h / 2, xx - w / 2), 0, 255).astype(np.uint8)   # soft disc
    return image


def smooth_clip(frames=7, h=48, w=64) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    clip = [np.stack([(xx * 3 + 5 * t) % 256, (yy * 4) % 256, (xx + yy + 9 * t) % 256], -1) for t in range(frames)]
    return np.asarray(clip, np.uint8)


class CountingFake:
    def __init__(self):
        from gbuffer import FakeEstimator

        self.inner, self.calls = FakeEstimator(), 0

    def describe(self):
        return self.inner.describe()

    def __call__(self, frames):
        self.calls += 1
        return self.inner(frames)


def make_sources(tmp: Path):
    from gbuffer import Item, write_mp4, write_png

    write_png(tmp / "a.png", rgba_still())
    write_png(tmp / "b.png", rgba_still(seed=1)[..., :3])
    write_mp4(tmp / "c.mp4", smooth_clip())
    return [Item("a", "still", str(tmp / "a.png")), Item("b", "still", str(tmp / "b.png")),
            Item("set/c", "clip", str(tmp / "c.mp4"))]


@test
def test_normal_encoding_is_unit_length():
    from gbuffer import decode_normals, encode_normals

    rng = np.random.default_rng(0)
    n = rng.normal(size=(5, 16, 16, 3)).astype(np.float32)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    back = decode_normals(encode_normals(n))
    assert np.abs(np.linalg.norm(back, axis=-1) - 1).max() < 1e-5
    angle = np.degrees(np.arccos(np.clip((back * n).sum(-1), -1, 1)))
    assert angle.max() < 0.5, angle.max()                              # 8 bits: under half a degree
    assert encode_normals(np.array([0.0, 0.0, 1.0])).tolist() == [128, 128, 255]      # facing the viewer
    assert decode_normals(np.zeros((2, 3), np.uint8) + 128).tolist() == [[0.0, 0.0, 1.0]] * 2   # nothing stored: +Z


@test
def test_manifest_round_trip_and_alignment():
    from gbuffer import CHANNELS, VERSION, GBufferSet, build

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        items = make_sources(tmp)
        estimator = CountingFake()
        manifest = build("toy", items, estimator, tmp / "out")
        on_disk = json.loads((tmp / "out" / "toy" / "manifest.json").read_text())
        assert on_disk == manifest and manifest["version"] == VERSION and manifest["channels"] == list(CHANNELS)
        assert manifest["source"] == "toy" and set(manifest["estimators"]) == {"albedo", "normals", "roughness", "alpha"}
        by_id = {e["id"]: e for e in manifest["items"]}
        assert [by_id[i]["frames"] for i in ("a", "b", "set/c")] == [1, 1, 7]
        assert all((e["width"], e["height"]) == (64, 48) for e in manifest["items"])
        assert by_id["a"]["has_alpha"] and not by_id["b"]["has_alpha"] and not by_id["set/c"]["has_alpha"]
        assert set(by_id["a"]["files"]) == set(CHANNELS) and set(by_id["b"]["files"]) == set(CHANNELS) - {"alpha"}
        assert by_id["set/c"]["files"]["normals"] == "normals.mp4" and by_id["a"]["files"]["normals"] == "normals.png"
        assert all(e["bytes"] > 0 and e["source_path"] and e["rgb"] == "source" for e in manifest["items"])

        gset = GBufferSet("toy", tmp / "out")
        assert len(gset) == 3 and gset.ids() == ["a", "b", "set/c"]
        for item_id, frames in (("a", 1), ("b", 1), ("set/c", 7)):
            rgb, alpha, albedo, normals, roughness = gset.load(item_id)
            assert rgb.shape == albedo.shape == normals.shape == (frames, 3, 48, 64), item_id
            assert alpha.shape == roughness.shape == (frames, 1, 48, 64), item_id
            assert all(t.dtype == torch.float32 for t in (rgb, alpha, albedo, normals, roughness))
            assert float((normals.norm(dim=1) - 1).abs().max()) < 1e-4, item_id        # unit, stills and video
            assert 0 <= float(albedo.min()) and float(albedo.max()) <= 1

        # per frame, in order: the fake's albedo is the frame itself, so frame t of albedo is frame t of rgb
        rgb, _alpha, albedo, _normals, roughness = gset.load("set/c")
        assert float((albedo - rgb).abs().max()) < 0.06                   # two x264 crf 8 passes apart
        wrong = float((albedo[1:] - rgb[:-1]).abs().mean())
        assert wrong > 5 * float((albedo - rgb).abs().mean())             # off by one frame is far worse
        assert float((roughness[:, 0] - rgb.mean(1)).abs().max()) < 0.06

        # resume: nothing is estimated twice; a missing file is redone
        calls = estimator.calls
        build("toy", items, estimator, tmp / "out")
        assert estimator.calls == calls
        (tmp / "out" / "toy" / "b" / "albedo.png").unlink()
        build("toy", items, estimator, tmp / "out")
        assert estimator.calls == calls + 1

        # a channel that does not match the manifest is an error, not a silent misalignment
        from gbuffer import write_mp4
        write_mp4(tmp / "out" / "toy" / "set" / "c" / "albedo.mp4", smooth_clip(frames=5))
        try:
            GBufferSet("toy", tmp / "out").load("set/c")
        except RuntimeError as error:
            assert "albedo" in str(error)
        else:
            raise AssertionError("a 5-frame albedo for a 7-frame clip loaded")


@test
def test_alpha_passthrough():
    from gbuffer import MATTE, GBufferSet, build, matte, read_image

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        items = make_sources(tmp)
        build("toy", items, CountingFake(), tmp / "out")
        gset = GBufferSet("toy", tmp / "out")
        source = rgba_still()
        rgb, alpha, albedo, _normals, _roughness = gset.load("a")
        assert torch.equal((alpha[0, 0] * 255).round().byte(), torch.from_numpy(source[..., 3]))      # exact, 8 bits
        assert np.array_equal(read_image(tmp / "out" / "toy" / "a" / "alpha.png")[..., 0], source[..., 3])
        assert torch.equal((rgb[0] * 255).round().byte(), torch.from_numpy(source[..., :3]).permute(2, 0, 1))   # straight
        # the estimator saw the still over grey: fully transparent pixels are the matte, opaque ones the colour
        seen = matte(source)
        assert np.array_equal(seen[source[..., 3] == 255], source[..., :3][source[..., 3] == 255])
        assert np.abs(seen[source[..., 3] == 0].astype(int) - round(MATTE * 255)).max() <= 1
        assert float((albedo[0].permute(1, 2, 0) - torch.from_numpy(seen).float() / 255).abs().max()) < 1e-6
        for item_id in ("b", "set/c"):                                  # no alpha in the source: ones
            assert float(gset.load(item_id)[1].min()) == 1.0


@test
def test_sources():
    from PIL import Image
    from safetensors.torch import save_file

    from gbuffer import cache_items, collect_items, cover_crop, t2rgba_items, write_mp4, write_png

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "img" / "x").mkdir(parents=True)
        write_png(tmp / "img" / "x" / "000000.png", rgba_still())
        write_png(tmp / "img" / "x" / "000001.png", rgba_still(seed=1))
        (tmp / "t.jsonl").write_text("".join(json.dumps({"image": f"x/00000{i}.png", "caption": "c"}) + "\n" for i in (0, 1)))
        items = t2rgba_items(0, tmp / "t.jsonl", tmp / "img")
        assert [i.id for i in items] == ["000000", "000001"] and items[0].frames().shape == (1, 48, 64, 4)
        assert len(t2rgba_items(1, tmp / "t.jsonl", tmp / "img")) == 1

        for name, count in (("s1", 3), ("s2", 1)):
            (tmp / "dwd" / name).mkdir(parents=True)
            for k in range(count):
                write_mp4(tmp / "dwd" / name / f"held_{k:04d}_0.mp4", smooth_clip(frames=3))
        clips = collect_items(0, tmp / "dwd", ("s1", "s2"))
        assert [c.id for c in clips] == ["s1/held_0000_0", "s2/held_0000_0", "s1/held_0001_0", "s1/held_0002_0"]
        assert [c.id for c in collect_items(2, tmp / "dwd", ("s1", "s2"))] == ["s1/held_0000_0", "s2/held_0000_0"]
        assert clips[0].frames().shape == (3, 48, 64, 3)

        (tmp / "cache").mkdir()
        (tmp / "src").mkdir()
        for stem in ("u1", "u2"):
            save_file({"latent_3x4": torch.zeros(24, 3, 4)}, str(tmp / "cache" / f"{stem}_0064x0048_minimaxh3.safetensors"))
        Image.fromarray(rgba_still(100, 100)[..., :3]).save(tmp / "src" / "u1.jpg")
        found = cache_items(0, tmp / "cache", tmp / "src")
        assert [(i.id, i.how, i.size) for i in found] == [("u1", "source", (64, 48))]
        assert found[0].frames().shape == (1, 48, 64, 3)               # fitted to the cache's canvas
        both = cache_items(0, tmp / "cache", tmp / "src", decode_missing=True)
        assert [(i.id, i.how) for i in both] == [("u1", "source"), ("u2", "decoded")]
        assert cover_crop(Image.new("RGB", (200, 100)), 64, 48).size == (64, 48)


@test
def test_contact_sheet_and_size_report():
    from PIL import Image

    from gbuffer import GBufferSet, build, contact_sheet

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build("toy", make_sources(tmp), CountingFake(), tmp / "out")
        contact_sheet([GBufferSet("toy", tmp / "out")], 3, tmp / "sheet.jpg", tile=64)
        assert Image.open(tmp / "sheet.jpg").size == (5 * 64, 18 + 3 * 64)


if __name__ == "__main__":
    for fn in TESTS:
        fn()
        print("ok", fn.__name__)
    print("ok", len(TESTS))
