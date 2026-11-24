#!/usr/bin/env python3
"""G-buffer channels for the training sources: albedo, normals, roughness, alpha, per frame, with a manifest.

The channels are ESTIMATES from single-image models, not ground truth, except alpha, which is copied from an
RGBA source (1 everywhere otherwise). docs/GBUFFER.md has the estimators, their licences and the limits.

    albedo      RGB, sRGB, [0, 1]                         Marigold IID-Appearance v1-1
    roughness   1 channel, linear, [0, 1]                 Marigold IID-Appearance v1-1
    normals     camera space, unit; X right, Y up,        Marigold Normals v1-1
                Z toward the viewer; stored (n + 1) / 2
    alpha       1 channel, [0, 1]                         the source's own alpha, or 1

Sources (``--source``):
    t2rgba    the AlphaAnything RGBA stills of alpha training (jsonl + PNGs with true alpha)
    collect   the DiR DWD collect sets' target clips (``collect_w1_{iso,atlas,kc}/*.mp4``)
    cache     an H3 still cache (``build_cache_h3.py``): the source image the latent was encoded from, cover-resized
              and centre-cropped to the cache's size; ``--decode-missing`` decodes the latent when there is none

Layout, under ``LOT_GBUFFER_ROOT`` (default /media/2TB/gbuffer), never in git::

    <source>/manifest.json
    <source>/<item id>/{albedo,normals,roughness,alpha}.png        stills (lossless)
    <source>/<item id>/{albedo,normals,roughness}.mp4               clips (x264 crf 8, 4:4:4: near-lossless)

    cd lot
    python3 -u gbuffer.py size --source t2rgba collect cache                   # item counts and a size estimate
    python3 -u gbuffer.py build --source t2rgba --limit 200                    # GPU; --limit 0 is everything
    python3 -u gbuffer.py sheet --source t2rgba --count 8 --out sheet.jpg

    from gbuffer import GBufferSet
    rgb, alpha, albedo, normals, roughness = GBufferSet("t2rgba").load(item_id)     # aligned (F, C, H, W) tensors

The Marigold pipelines need diffusers >= 0.33. This env's diffusers (0.32) does not import with its transformers,
so a newer one is read from an overlay folder (``LOT_GBUFFER_OVERLAY``, default <root>/.pyoverlay):
``pip install --no-deps --target /media/2TB/gbuffer/.pyoverlay diffusers==0.40.0``. The env is not changed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

VERSION = 1
ROOT = Path(os.environ.get("LOT_GBUFFER_ROOT", "/media/2TB/gbuffer"))
OVERLAY = Path(os.environ.get("LOT_GBUFFER_OVERLAY", str(ROOT / ".pyoverlay")))
T2RGBA_ROOT = Path(os.environ.get("LOT_T2RGBA_ROOT", "/media/2TB/alpha/data/t2rgba"))
T2RGBA_JSONL = Path(os.environ.get("LOT_T2RGBA_JSONL", "/media/2TB/alpha/data/t2rgba_train.jsonl"))
COLLECT_ROOT = Path(os.environ.get("LOT_DWD_COLLECT_ROOT", "/media/2TB/lora-data/h3_lot/dwd"))
COLLECT_SETS = ("collect_w1_iso", "collect_w1_atlas", "collect_w1_kc")
CHANNELS = ("albedo", "normals", "roughness", "alpha")
MATTE = 0.5                                        # grey behind an RGBA still when the estimators look at it

ESTIMATORS = {
    "albedo": {"model": "prs-eth/marigold-iid-appearance-v1-1", "weights_license": "CreativeML Open RAIL++-M",
               "code": "diffusers MarigoldIntrinsicsPipeline", "code_license": "Apache-2.0"},
    "roughness": {"model": "prs-eth/marigold-iid-appearance-v1-1", "weights_license": "CreativeML Open RAIL++-M",
                  "code": "diffusers MarigoldIntrinsicsPipeline", "code_license": "Apache-2.0"},
    "normals": {"model": "prs-eth/marigold-normals-v1-1", "weights_license": "CreativeML Open RAIL++-M",
                "code": "diffusers MarigoldNormalsPipeline", "code_license": "Apache-2.0",
                "frame": "camera space: X right, Y up, Z toward the viewer"},
    "alpha": {"model": None, "note": "copied from the source's alpha channel; 1 when the source has none"},
}


# ------------------------------------------------------------------------------------------- encodings

def encode_normals(normals: np.ndarray) -> np.ndarray:
    """Unit normals ``(..., 3)`` in [-1, 1] -> uint8 ``(n + 1) / 2 * 255``."""
    return np.clip(np.rint((np.asarray(normals, np.float32) + 1.0) * 127.5), 0, 255).astype(np.uint8)


def decode_normals(encoded: np.ndarray) -> np.ndarray:
    """uint8 ``(..., 3)`` -> float32 unit normals. Re-normalised: 8 bits and a video codec leave them off length 1.
    A near-zero vector (mid-grey: nothing stored) decodes to +Z, facing the viewer."""
    n = np.asarray(encoded, np.float32) / 127.5 - 1.0
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.where(length > 0.05, n / np.maximum(length, 0.05), np.array([0.0, 0.0, 1.0], np.float32)).astype(np.float32)


def to_u8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(values, np.float32) * 255.0), 0, 255).astype(np.uint8)


def matte(rgba: np.ndarray, grey: float = MATTE) -> np.ndarray:
    """``(..., 4)`` uint8 straight-alpha RGBA over a flat grey -> ``(..., 3)`` uint8: what the estimators see."""
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return to_u8((rgba[..., :3].astype(np.float32) / 255.0) * alpha + grey * (1.0 - alpha))


def cover_crop(image, width: int, height: int):
    """PIL image cover-resized and centre-cropped to ``width x height`` (as ``build_cache_h3`` fits its canvas)."""
    from PIL import Image

    scale = max(width / image.width, height / image.height)
    resized = image.resize((max(width, round(image.width * scale)), max(height, round(image.height * scale))),
                           Image.BICUBIC)
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


# ------------------------------------------------------------------------------------------- media io

def read_image(path: Path) -> np.ndarray:
    """``(H, W, 3 or 4)`` uint8; the alpha channel is kept when the file has one."""
    from PIL import Image

    image = Image.open(path)
    return np.asarray(image.convert("RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"))


def read_mp4(path: Path) -> np.ndarray:
    """``(F, H, W, 3)`` uint8, every frame."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    w, h = map(int, probe.stdout.strip().split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vsync", "0", "-f", "rawvideo", "-pix_fmt",
                          "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).copy()


def write_mp4(path: Path, frames: np.ndarray, fps: int = 24) -> None:
    """``(F, H, W, 3)`` uint8 -> x264 crf 8, 4:4:4 (no chroma subsampling: normals are not a picture), atomically."""
    f, h, w, _c = frames.shape
    tmp = path.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r",
                    str(fps), "-i", "-", "-c:v", "libx264", "-crf", "8", "-preset", "medium", "-pix_fmt", "yuv444p",
                    str(tmp)], input=np.ascontiguousarray(frames).tobytes(), check=True)
    tmp.replace(path)


def write_png(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(array).save(path, optimize=False, compress_level=6)


# ------------------------------------------------------------------------------------------- sources

@dataclass
class Item:
    id: str                    # folder under <root>/<source>/
    kind: str                  # "still" | "clip"
    path: str                  # the file the frames come from
    how: str = "source"        # "source": the file itself; "decoded": VAE-decoded from a cached latent
    size: tuple | None = None  # (W, H) to fit a still to (the cache's canvas)

    def frames(self) -> np.ndarray:
        """``(F, H, W, 3 or 4)`` uint8."""
        if self.kind == "clip":
            return read_mp4(Path(self.path))
        if self.how == "decoded":
            raise RuntimeError("a decoded item is read through cache_decoder()")
        from PIL import Image

        image = read_image(Path(self.path))
        if self.size is not None and (image.shape[1], image.shape[0]) != tuple(self.size):
            image = np.asarray(cover_crop(Image.fromarray(image), *self.size))
        return image[None]


def t2rgba_items(limit: int = 0, jsonl: Path = T2RGBA_JSONL, root: Path = T2RGBA_ROOT) -> list[Item]:
    items = []
    with open(jsonl) as handle:
        for line in handle:
            rel = json.loads(line)["image"]
            items.append(Item(Path(rel).stem, "still", str(root / rel)))
            if limit and len(items) >= limit:
                break
    return items


def collect_items(limit: int = 0, root: Path = COLLECT_ROOT, sets=COLLECT_SETS) -> list[Item]:
    """The target clips, round-robin over the sets so a small ``limit`` covers all three."""
    per_set = [[Item(f"{name}/{p.stem}", "clip", str(p)) for p in sorted((Path(root) / name).glob("*.mp4"))]
               for name in sets]
    longest = max((len(group) for group in per_set), default=0)
    items = [group[k] for k in range(longest) for group in per_set if k < len(group)]
    return items[:limit] if limit else items


def cache_items(limit: int = 0, cache: Path | None = None, images: Path | None = None,
                decode_missing: bool = False) -> list[Item]:
    """One item per cached latent ``<stem>_<WWWW>x<HHHH>_minimaxh3.safetensors``. The source image
    ``<images>/<stem>.{jpg,png,webp}`` is used when it exists; otherwise the item is skipped, or with
    ``decode_missing`` marked for VAE decoding."""
    from lot_paths import LOT_H3_STILLS_CACHE, LOT_H3_STILLS_CAPTIONS

    cache, images = Path(cache or LOT_H3_STILLS_CACHE), Path(images or LOT_H3_STILLS_CAPTIONS)
    items = []
    for latent in sorted(cache.glob("*_minimaxh3.safetensors")):
        stem, _, size = latent.name[:-len("_minimaxh3.safetensors")].rpartition("_")
        try:
            w, h = (int(v) for v in size.split("x"))
        except ValueError:
            continue
        found = next((images / f"{stem}{ext}" for ext in (".jpg", ".png", ".webp", ".jpeg")
                      if (images / f"{stem}{ext}").is_file()), None)
        if found is not None:
            items.append(Item(stem, "still", str(found), "source", (w, h)))
        elif decode_missing:
            items.append(Item(stem, "still", str(latent), "decoded", (w, h)))
        if limit and len(items) >= limit:
            break
    return items


def cache_decoder():
    """``decode(item) -> (1, H, W, 3) uint8`` for ``how == "decoded"`` items: the H3 VAE on the cached latent."""
    from safetensors import safe_open

    from h3_musubi import load_vae
    from lot_paths import LOT_H3_VAE

    vae = load_vae(LOT_H3_VAE, device="cuda")

    @torch.no_grad()
    def decode(item: Item) -> np.ndarray:
        with safe_open(item.path, "pt") as handle:
            latent = handle.get_tensor(next(k for k in handle.keys() if k.startswith("latent")))
        pixels = vae.decode(latent.unsqueeze(0).unsqueeze(2).to("cuda", next(vae.parameters()).dtype)).float()
        return to_u8(pixels[0, :, 0].clamp(-1, 1).add(1).div(2).permute(1, 2, 0).cpu().numpy())[None]

    return decode


SOURCES = {"t2rgba": t2rgba_items, "collect": collect_items, "cache": cache_items}


# ------------------------------------------------------------------------------------------- estimators

class FakeEstimator:
    """Deterministic stand-in for the tests: albedo = the image, roughness = its luma, normals tilt with the
    luma's gradient. No model."""

    name, version = "fake", "0"

    def describe(self) -> dict:
        return {k: {"model": "fake", "revision": "0"} for k in ("albedo", "normals", "roughness")}

    def __call__(self, frames: np.ndarray) -> dict:
        rgb = frames.astype(np.float32) / 255.0
        luma = rgb.mean(-1)
        gy, gx = np.gradient(luma, axis=(1, 2))
        normals = np.stack([-gx, gy, np.full_like(luma, 0.5)], -1)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        return {"albedo": rgb, "normals": normals.astype(np.float32), "roughness": luma}


class MarigoldEstimator:
    """Marigold IID-Appearance v1-1 (albedo, roughness, metallicity) and Marigold Normals v1-1, frame by frame.

    Every frame of a clip starts from the same noise (one seed), which removes the sampling noise's share of
    frame-to-frame flicker; the models still see one frame at a time."""

    name = "marigold"

    def __init__(self, device="cuda", steps: int = 4, resolution: int = 768, batch: int = 4, seed: int = 0,
                 dtype=torch.float16):
        if OVERLAY.is_dir() and str(OVERLAY) not in sys.path:
            sys.path.insert(0, str(OVERLAY))
        import diffusers

        self.diffusers = diffusers.__version__
        variant = "fp16" if dtype == torch.float16 else None
        self.appearance = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
            ESTIMATORS["albedo"]["model"], variant=variant, torch_dtype=dtype).to(device)
        self.normals = diffusers.MarigoldNormalsPipeline.from_pretrained(
            ESTIMATORS["normals"]["model"], variant=variant, torch_dtype=dtype).to(device)
        for pipe in (self.appearance, self.normals):
            pipe.set_progress_bar_config(disable=True)
        self.device, self.steps, self.resolution, self.batch, self.seed = device, steps, resolution, batch, seed
        props = self.appearance.target_properties
        names = list(props["target_names"])
        self.albedo_index = names.index("albedo")
        self.material_index, self.roughness_channel = next(
            (i, list(props[n]["sub_target_names"]).index("roughness")) for i, n in enumerate(names)
            if "roughness" in (props[n].get("sub_target_names") or []))
        self.targets = len(names)

    def describe(self) -> dict:
        def revision(model: str) -> str:
            """The commit the hub cache holds for ``main`` (no network)."""
            from huggingface_hub import constants

            ref = Path(constants.HF_HUB_CACHE) / ("models--" + model.replace("/", "--")) / "refs" / "main"
            return ref.read_text().strip() if ref.is_file() else ""

        shared = {"diffusers": self.diffusers, "steps": self.steps, "processing_resolution": self.resolution,
                  "seed": self.seed, "ensemble": 1}
        return {"albedo": {**ESTIMATORS["albedo"], "revision": revision(ESTIMATORS["albedo"]["model"]), **shared},
                "roughness": {**ESTIMATORS["roughness"], "revision": revision(ESTIMATORS["albedo"]["model"]), **shared},
                "normals": {**ESTIMATORS["normals"], "revision": revision(ESTIMATORS["normals"]["model"]), **shared}}

    @torch.no_grad()
    def __call__(self, frames: np.ndarray) -> dict:
        from PIL import Image

        out = {"albedo": [], "normals": [], "roughness": []}
        for start in range(0, frames.shape[0], self.batch):
            images = [Image.fromarray(f) for f in frames[start:start + self.batch]]
            common = dict(num_inference_steps=self.steps, processing_resolution=self.resolution,
                          batch_size=len(images), ensemble_size=1)
            gen = lambda: [torch.Generator(self.device).manual_seed(self.seed) for _ in images]   # noqa: E731
            a = self.appearance(images, generator=gen(), **common).prediction             # (N * targets, H, W, 3)
            a = a.reshape(len(images), self.targets, *a.shape[1:])
            out["albedo"].append(a[:, self.albedo_index])
            out["roughness"].append(a[:, self.material_index, :, :, self.roughness_channel])
            out["normals"].append(self.normals(images, generator=gen(), **common).prediction)   # (N, H, W, 3)
        return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


# ------------------------------------------------------------------------------------------- build

def split_alpha(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """``(F, H, W, 3 or 4)`` -> the RGB the estimators see (matted on grey) and the alpha ``(F, H, W)`` or None."""
    if frames.shape[-1] == 4:
        return matte(frames), frames[..., 3].copy()
    return frames, None


def write_item(folder: Path, kind: str, channels: dict, alpha: np.ndarray | None) -> dict:
    """Write one item's channels; returns ``{channel: file name}``. ``channels`` are float arrays as an estimator
    returns them. Alpha is written only when the source has one (the loader fills 1 otherwise)."""
    folder.mkdir(parents=True, exist_ok=True)
    grey3 = lambda a: np.repeat(a[..., None], 3, -1)                                       # noqa: E731
    encoded = {"albedo": to_u8(channels["albedo"]), "normals": encode_normals(channels["normals"]),
               "roughness": to_u8(channels["roughness"])}
    files = {}
    if kind == "still":
        for name, array in encoded.items():
            write_png(folder / f"{name}.png", array[0])
            files[name] = f"{name}.png"
        if alpha is not None:
            write_png(folder / "alpha.png", alpha[0])
            files["alpha"] = "alpha.png"
    else:
        for name, array in encoded.items():
            write_mp4(folder / f"{name}.mp4", array if array.ndim == 4 else grey3(array))
            files[name] = f"{name}.mp4"
        if alpha is not None:
            write_mp4(folder / "alpha.mp4", grey3(alpha))
            files["alpha"] = "alpha.mp4"
    return files


def build(source: str, items: list[Item], estimator, root: Path = ROOT, decode=None, resume: bool = True) -> dict:
    """Estimate and write every item, then ``<root>/<source>/manifest.json``. Items already in the manifest with
    their files on disk are kept (``resume``)."""
    out = Path(root) / source
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text()) if (resume and path.is_file()) else {}
    done = {entry["id"]: entry for entry in manifest.get("items", [])}
    manifest = {"version": VERSION, "source": source, "channels": list(CHANNELS), "estimators": estimator.describe(),
                "alpha": ESTIMATORS["alpha"]["note"], "matte": MATTE,
                "normals_frame": ESTIMATORS["normals"]["frame"], "items": list(done.values())}
    manifest["estimators"]["alpha"] = ESTIMATORS["alpha"]
    for item in items:
        prior = done.get(item.id)
        if prior and all((out / item.id / f).is_file() for f in prior["files"].values()):
            continue
        start = time.perf_counter()
        frames = decode(item) if item.how == "decoded" else item.frames()
        rgb, alpha = split_alpha(frames)
        channels = estimator(rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        estimate_s = time.perf_counter() - start
        for name, array in channels.items():
            if array.shape[:3] != rgb.shape[:3]:
                raise RuntimeError(f"{item.id}: {name} is {array.shape[:3]}, the source is {rgb.shape[:3]}")
        files = write_item(out / item.id, item.kind, channels, alpha)
        entry = {"id": item.id, "kind": item.kind, "source_path": item.path, "rgb": item.how,
                 "width": int(rgb.shape[2]), "height": int(rgb.shape[1]), "frames": int(rgb.shape[0]),
                 "has_alpha": alpha is not None, "files": files,
                 "seconds_per_frame": round(estimate_s / rgb.shape[0], 4),
                 "bytes": sum((out / item.id / f).stat().st_size for f in files.values())}
        done[item.id] = entry
        manifest["items"] = list(done.values())
        path.write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"LOT_H3 kind=gbuffer source={source} item={item.id} frames={entry['frames']} "
              f"size={entry['width']}x{entry['height']} s_per_frame={entry['seconds_per_frame']} "
              f"bytes={entry['bytes']} alpha={entry['has_alpha']}", flush=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


# ------------------------------------------------------------------------------------------- loader

class GBufferSet:
    """The written channels of one source, aligned with the source frames.

    ``load(id)`` returns ``(rgb, alpha, albedo, normals, roughness)``, float32 tensors ``(F, C, H, W)``:
    rgb, albedo in [0, 1] (3 channels); alpha, roughness in [0, 1] (1 channel); normals unit vectors in [-1, 1]
    (camera space, X right, Y up, Z toward the viewer). A still has ``F = 1``. ``rgb`` is the source's own colour
    (straight, not matted), ``alpha`` its own alpha or ones.
    """

    def __init__(self, source: str, root: Path = ROOT, decode=None):
        self.dir = Path(root) / source
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        if self.manifest["version"] != VERSION:
            raise RuntimeError(f"{self.dir}: manifest version {self.manifest['version']}, this code reads {VERSION}")
        self.items = {entry["id"]: entry for entry in self.manifest["items"]}
        self.decode = decode

    def __len__(self) -> int:
        return len(self.items)

    def ids(self) -> list[str]:
        return list(self.items)

    def _read(self, entry: dict, name: str) -> np.ndarray:
        path = self.dir / entry["id"] / entry["files"][name]
        array = read_mp4(path) if path.suffix == ".mp4" else read_image(path)[None]
        return array

    def source_frames(self, entry: dict) -> np.ndarray:
        item = Item(entry["id"], entry["kind"], entry["source_path"], entry["rgb"], (entry["width"], entry["height"]))
        if item.how == "decoded":
            if self.decode is None:
                raise RuntimeError(f"{item.id} was VAE-decoded: pass decode=cache_decoder()")
            return self.decode(item)
        return item.frames()

    def load(self, item_id: str) -> tuple:
        entry = self.items[item_id]
        shape = (entry["frames"], entry["height"], entry["width"])
        frames = self.source_frames(entry)
        rgb = frames[..., :3]
        albedo, normals, roughness = (self._read(entry, n) for n in ("albedo", "normals", "roughness"))
        alpha = self._read(entry, "alpha") if "alpha" in entry["files"] else None
        if roughness.ndim == 4:
            roughness = roughness[..., 0]
        if alpha is not None and alpha.ndim == 4:
            alpha = alpha[..., 0]
        for name, array in (("rgb", rgb), ("albedo", albedo), ("normals", normals), ("roughness", roughness)):
            if tuple(array.shape[:3]) != shape:
                raise RuntimeError(f"{item_id}: {name} is {tuple(array.shape[:3])}, the manifest says {shape}")
        if alpha is not None and tuple(alpha.shape) != shape:
            raise RuntimeError(f"{item_id}: alpha is {tuple(alpha.shape)}, the manifest says {shape}")
        image = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).permute(0, 3, 1, 2)   # noqa: E731
        plane = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(1)          # noqa: E731
        return (image(rgb), plane(alpha) if alpha is not None else torch.ones(shape[0], 1, *shape[1:]),
                image(albedo), torch.from_numpy(decode_normals(normals)).permute(0, 3, 1, 2), plane(roughness))


# ------------------------------------------------------------------------------------------- commands

def contact_sheet(sets: list[GBufferSet], count: int, path: Path, tile: int = 192) -> None:
    """One row per item: rgb (over a checkerboard where it has alpha), alpha, albedo, normals, roughness."""
    from PIL import Image, ImageDraw

    rows = []
    for gset in sets:
        for item_id in gset.ids()[:count]:
            rgb, alpha, albedo, normals, roughness = gset.load(item_id)
            f = rgb.shape[0] // 2
            h, w = rgb.shape[2:]
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            check = (((yy // 16 + xx // 16) % 2).float() * 0.3 + 0.5).expand(3, h, w)
            tiles = [rgb[f] * alpha[f] + check * (1 - alpha[f]), alpha[f].expand(3, -1, -1), albedo[f],
                     (normals[f] + 1) / 2, roughness[f].expand(3, -1, -1)]
            row = Image.new("RGB", (tile * len(tiles), tile), (32, 32, 32))
            for k, t in enumerate(tiles):
                image = Image.fromarray(to_u8(t.permute(1, 2, 0).numpy()))
                image.thumbnail((tile, tile), Image.LANCZOS)
                row.paste(image, (k * tile + (tile - image.width) // 2, (tile - image.height) // 2))
            ImageDraw.Draw(row).text((4, 2), f"{gset.manifest['source']}:{item_id}"[:40], fill=(255, 255, 0))
            rows.append(row)
    header = Image.new("RGB", (rows[0].width, 18), (0, 0, 0))
    for k, label in enumerate(("rgb", "alpha", "albedo", "normals (camera)", "roughness")):
        ImageDraw.Draw(header).text((k * tile + 4, 3), label, fill=(255, 255, 255))
    out = Image.new("RGB", (rows[0].width, 18 + tile * len(rows)))
    out.paste(header, (0, 0))
    for k, row in enumerate(rows):
        out.paste(row, (0, 18 + k * tile))
    out.save(path, quality=90)


def list_items(source: str, limit: int, decode_missing: bool = False) -> list[Item]:
    if source == "cache":
        return cache_items(limit, decode_missing=decode_missing)
    return SOURCES[source](limit)


def size_report(sources: list[str], root: Path = ROOT) -> list[dict]:
    """Item counts, and the size of the full set scaled from what is already written (bytes per frame)."""
    rows = []
    for source in sources:
        items = list_items(source, 0, decode_missing=True)
        row = {"source": source, "items": len(items), "kind": items[0].kind if items else ""}
        path = Path(root) / source / "manifest.json"
        if path.is_file():
            entries = json.loads(path.read_text())["items"]
            frames = sum(e["frames"] for e in entries)
            row.update(written=len(entries), written_gb=round(sum(e["bytes"] for e in entries) / 1e9, 3),
                       frames_per_item=round(frames / max(1, len(entries)), 1),
                       seconds_per_frame=round(sum(e["seconds_per_frame"] * e["frames"] for e in entries) / max(1, frames), 3))
            if entries:
                row["full_gb"] = round(sum(e["bytes"] for e in entries) / len(entries) * len(items) / 1e9, 1)
                row["full_hours"] = round(row["seconds_per_frame"] * frames / len(entries) * len(items) / 3600, 1)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", choices=tuple(SOURCES), required=True)
    b.add_argument("--limit", type=int, default=200, help="items to do; 0 is every item (check `size` first)")
    b.add_argument("--steps", type=int, default=4)
    b.add_argument("--resolution", type=int, default=768, help="longer side the estimators work at")
    b.add_argument("--batch", type=int, default=4)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--decode-missing", action="store_true", help="cache: VAE-decode latents that have no source image")
    b.add_argument("--root", type=Path, default=ROOT)
    s = sub.add_parser("size")
    s.add_argument("--source", nargs="+", choices=tuple(SOURCES), default=list(SOURCES))
    s.add_argument("--root", type=Path, default=ROOT)
    c = sub.add_parser("sheet")
    c.add_argument("--source", nargs="+", choices=tuple(SOURCES), default=list(SOURCES))
    c.add_argument("--count", type=int, default=6)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.cmd == "size":
        for row in size_report(args.source, args.root):
            print("LOT_H3 kind=gbuffer_size " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        return
    if args.cmd == "sheet":
        sets = [GBufferSet(source, args.root) for source in args.source if (args.root / source / "manifest.json").is_file()]
        contact_sheet(sets, args.count, args.out)
        print(f"LOT_H3 kind=gbuffer_sheet out={args.out}", flush=True)
        return
    from setproctitle import setproctitle

    from gpu_guard import refuse_if_busy

    setproctitle(f"gbuffer:build {args.source} n={args.limit} s{args.steps} r{args.resolution}")
    refuse_if_busy("gbuffer.py")
    if "5090" in torch.cuda.get_device_name(0) and os.environ.get("LOT_GBUFFER_ANY_CARD") != "1":
        raise SystemExit("this is the RTX 5090 (LOT_GBUFFER_ANY_CARD=1 to allow it)")
    items = list_items(args.source, args.limit, args.decode_missing)
    estimator = MarigoldEstimator(steps=args.steps, resolution=args.resolution, batch=args.batch, seed=args.seed)
    decode = cache_decoder() if any(i.how == "decoded" for i in items) else None
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    manifest = build(args.source, items, estimator, args.root, decode)
    mine = [e for e in manifest["items"] if e["id"] in {i.id for i in items}]
    frames = sum(e["frames"] for e in mine)
    print(f"LOT_H3 kind=gbuffer_done source={args.source} items={len(mine)} frames={frames} "
          f"wall_s={time.perf_counter() - start:.0f} "
          f"s_per_frame={sum(e['seconds_per_frame'] * e['frames'] for e in mine) / max(1, frames):.3f} "
          f"gb={sum(e['bytes'] for e in mine) / 1e9:.3f} peak_gb={torch.cuda.max_memory_allocated() / 2**30:.1f} "
          f"gpu={torch.cuda.get_device_name(0)!r} ok=1", flush=True)


if __name__ == "__main__":
    main()
