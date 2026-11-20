#!/usr/bin/env python3
"""A small pixel-space refiner on top of the DWD head: sharpen the linear decoder's output toward the VAE's.

The linear head (``dwd_h3.py``) plus a fine-tuned LoRA levels off around LPIPS 0.26 against the VAE decode:
about the VAE output blurred by 1.5-2 px. A per-token linear map cannot restore that detail; a few
convolutions that see neighbouring pixels and frames can. The refiner is a small U-Net (about 1.5 M parameters)
that reads the DWD frame and its two neighbours and predicts a residual for the middle frame.

    # 1. GPU: render the DWD decode of training and held-out clips (DiT + DWD LoRA + head) as near-lossless mp4s
    python3 lot/refine_dwd_h3.py cache --run lot/runs/dir_turbo_clip --data DATA... --lora LORA --head HEAD \\
        --out OUT --train-per-folder 500

    # 2. train the refiner on (DWD, VAE) frame pairs, random crops, L1 + LPIPS; no DiT
    LOT_H3_TRAIN=1 python3 lot/refine_dwd_h3.py train --out OUT --minutes 240

Outputs in OUT: cache/ (mp4 pairs and pairs.json), refiner_best.pt (lowest held-out LPIPS), refiner_last.pt
(crash-safe, overwritten), refine.log-style lines on stdout, videos/ (VAE | DWD | DWD + refiner).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dwd_h3 import flicker_split, frames_of, pixel_frame_count, psnr, seam_ratio  # noqa: E402
from train_dwd_h3 import load, load_head, read_mp4, samples, to_uint8, write_mp4  # noqa: E402


# ------------------------------------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.silu(self.c1(F.silu(x))))


class LatentPath(nn.Module):
    """The latent input on its own grid: ``(N, C, H/16, W/16)`` -> ``(N, out, H/4, W/4)``, added at the U-Net's
    bottleneck. A pixel shuffle gives each of the 4x4 positions inside a latent pixel its own weights, which the
    nearest-upsampled copy at the input cannot; and a few wide blocks here see +-``2 * blocks + 1`` latent pixels
    for about 1/256 of their cost at full resolution. Zero-initialised output."""

    def __init__(self, in_ch: int, width: int, blocks: int, out_ch: int):
        super().__init__()
        self.inp = nn.Conv2d(in_ch, width, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.out = nn.Conv2d(width, out_ch * 16, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, lat):
        return F.pixel_shuffle(self.out(F.silu(self.blocks(self.inp(lat)))), 4)


class Refiner(nn.Module):
    """``(N, 9, H, W)`` (previous, current, next frame) -> residual for the current frame. Zero-initialised
    output, so an untrained refiner returns the DWD frame unchanged."""

    def __init__(self, base: int = 48, latent: bool | int = False, grid: int = 0, grid_blocks: int = 4):
        super().__init__()
        b = base
        # latent: False, True (the frame's 24-channel latent) or a channel count (CONTEXT_CH: neighbouring latent
        # frames and the sub-frame phase, see ``frame_context``); always at 1/16 resolution
        self.latent_ch = 0 if not latent else (24 if latent is True else int(latent))
        self.latent = self.latent_ch > 0
        self.inp = nn.Conv2d(9 + self.latent_ch, b, 3, padding=1)
        self.e1 = Block(b)
        self.d1 = nn.Conv2d(b, 2 * b, 4, stride=2, padding=1)
        self.e2 = Block(2 * b)
        self.d2 = nn.Conv2d(2 * b, 4 * b, 4, stride=2, padding=1)
        self.mid = nn.Sequential(Block(4 * b), Block(4 * b))
        self.grid_width, self.grid_blocks = (grid, grid_blocks) if grid else (0, 0)
        if grid and not self.latent:
            raise ValueError("the latent-grid path needs a latent input")
        self.grid = LatentPath(self.latent_ch, grid, grid_blocks, 4 * b) if grid else None
        self.u2 = nn.ConvTranspose2d(4 * b, 2 * b, 4, stride=2, padding=1)
        self.m2 = Block(2 * b)
        self.u1 = nn.ConvTranspose2d(2 * b, b, 4, stride=2, padding=1)
        self.m1 = Block(b)
        self.out = nn.Conv2d(b, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, lat=None):
        """``lat`` ``(N, 24, H/16, W/16)``: the current frame's latent, for a ``latent=True`` refiner."""
        inp = x if not self.latent else torch.cat([x, F.interpolate(lat.to(x.dtype), scale_factor=16, mode="nearest")], 1)
        h1 = self.e1(self.inp(inp))
        h2 = self.e2(self.d1(h1))
        h = self.d2(h2)
        if self.grid is not None:
            h = h + self.grid(lat.to(x.dtype))
        h = self.mid(h)
        h = self.m2(self.u2(h) + h2)
        h = self.m1(self.u1(h) + h1)
        return x[:, 3:6] + self.out(h)


CONTEXT_CH = 3 * 24 + 4


def spec_of(saved: dict) -> dict:
    """Constructor arguments of a saved refiner (``refiner_best.pt`` or ``refiner_last.pt``), old files included."""
    return {"base": saved.get("base", 48), "latent": saved.get("latent_spec", saved.get("latent", False)),
            "grid": saved.get("grid", 0), "grid_blocks": saved.get("grid_blocks", 4)}


def init_from(model: Refiner, saved: dict) -> list[str]:
    """Start ``model`` from another refiner's weights so that it computes the same function at step 0.

    The source may lack the latent-grid path (left at its zero-output initialisation) and may read fewer latent
    channels: a plain refiner (9 inputs) or a ``--latent`` one (9 + 24) widens to ``--latent-context`` (9 + 76)
    with its 24 latent channels placed on the pixel frame's own latent frame and zeros elsewhere. Returns the
    names it did not find in the source."""
    src, own = saved["model"], model.state_dict()
    missing = []
    for name, value in own.items():
        if name not in src:
            missing.append(name)
            continue
        old = src[name]
        if name == "inp.weight" and old.shape != value.shape:
            if old.shape[0] != value.shape[0] or old.shape[2:] != value.shape[2:]:
                raise SystemExit(f"--init: {name} is {tuple(old.shape)}, the model wants {tuple(value.shape)} (base?)")
            new = torch.zeros_like(value)
            new[:, :9] = old[:, :9]
            if old.shape[1] == 9 + 24 and value.shape[1] == 9 + CONTEXT_CH:
                # frame_context interleaves: channel 3c + j is latent channel c of the frame before (j=0), own, after
                new[:, 9 + 1:9 + 72:3] = old[:, 9:]
            elif old.shape[1] != 9:
                raise SystemExit(f"--init: cannot widen {old.shape[1]} input channels to {value.shape[1]}")
            own[name] = new
        elif old.shape != value.shape:
            raise SystemExit(f"--init: {name} is {tuple(old.shape)}, the model wants {tuple(value.shape)} (base?)")
        else:
            own[name] = old
    extra = [k for k in src if k not in own]
    if extra:
        raise SystemExit(f"--init: the source has weights this model lacks: {extra[:4]}")
    if any(not k.startswith("grid.") for k in missing):
        raise SystemExit(f"--init: the source lacks {missing[:4]}")
    model.load_state_dict(own)
    return missing


def frame_context(lat: torch.Tensor, ts: list[int]) -> torch.Tensor:
    """``lat`` ``(24, T, h, w)`` -> ``(len(ts), 76, h, w)`` for pixel frames ``ts``: the latent frames before, at and
    after the pixel frame's own (edges repeated), and a one-hot of its place among that latent frame's 1 or 4 pixel
    frames. The VAE's output depends on neighbouring latent frames, and the four pixel frames of one latent frame
    differ; a single latent per pixel frame gives the refiner neither."""
    t_lat = lat.shape[1]
    group = frame_groups(t_lat)
    first = {}
    for i, g in enumerate(group):
        first.setdefault(g, i)
    out = []
    for t in ts:
        g = group[t]
        cur = lat[:, [max(g - 1, 0), g, min(g + 1, t_lat - 1)]].flatten(0, 1)            # (72, h, w)
        phase = torch.zeros(4, *lat.shape[2:], dtype=lat.dtype)
        phase[t - first[g]] = 1
        out.append(torch.cat([cur, phase]))
    return torch.stack(out)


def neighbours(frames: torch.Tensor) -> torch.Tensor:
    """``(F, 3, H, W)`` -> ``(F, 9, H, W)``: each frame with the one before and after (edges repeated)."""
    prev = torch.cat([frames[:1], frames[:-1]])
    nxt = torch.cat([frames[1:], frames[-1:]])
    return torch.cat([prev, frames, nxt], dim=1)


def latent_frames(pixel_frames: int) -> int:
    """Latent frames of a clip with this many pixel frames (39 -> 12)."""
    for t in range(1, pixel_frames + 1):
        if pixel_frame_count(t) == pixel_frames:
            return t
    raise ValueError(f"{pixel_frames} pixel frames is not a whole number of latent frames")


def frame_groups(latent_t: int) -> list[int]:
    """Latent frame index of every pixel frame."""
    return [k for k, n in enumerate(frames_of(latent_t)) for _ in range(n)]


@torch.no_grad()
def refine_clip(model: Refiner, clip: torch.Tensor, latent: torch.Tensor | None = None, chunk: int = 8) -> torch.Tensor:
    """``(1, 3, F, H, W)`` in [-1, 1] -> refined clip, same shape. H and W must be multiples of 4 (16 with a
    latent). ``latent`` ``(1, 24, T, H/16, W/16)`` for a ``latent=True`` refiner."""
    frames = clip[0].permute(1, 0, 2, 3)
    x = neighbours(frames)
    lat = None
    if model.latent_ch == CONTEXT_CH:
        lat = frame_context(latent[0].float().cpu(), list(range(frames.shape[0]))).to(frames.device)
    elif model.latent:
        lat = latent[0][:, frame_groups(latent.shape[2])].permute(1, 0, 2, 3).float()          # (F, 24, h, w)
    out = torch.cat([model(x[i:i + chunk], None if lat is None else lat[i:i + chunk])
                     for i in range(0, x.shape[0], chunk)])
    return out.clamp(-1, 1).permute(1, 0, 2, 3).unsqueeze(0)


def read_mp4_u8(path: Path) -> torch.Tensor:
    """``(F, H, W, 3)`` uint8, every frame (``read_mp4`` without the float copy)."""
    import subprocess

    import numpy as np

    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    w, h = map(int, probe.stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-"], capture_output=True, check=True).stdout
    return torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).copy())


def clip_latent(pair: dict) -> torch.Tensor:
    """The clip's stored final latent ``(1, 24, T, H/16, W/16)`` (beside its VAE target)."""
    return torch.load(Path(pair["vae"]).with_suffix(".pt"), map_location="cpu", mmap=True, weights_only=True)["latent"]


# ------------------------------------------------------------------------------------------- cache

def cache(args) -> None:
    """Render the DWD decode (DiT with the DWD LoRA on at the final iteration, then the head) to mp4."""
    from gpu_guard import refuse_if_busy

    refuse_if_busy("refine_dwd_h3.py")
    from dir_h3 import generator_step, load_dir_run, use_networks
    from h3_musubi import load_dit, make_lora

    dev = torch.device("cuda")
    dit = load_dit(device=dev, swap=0)
    gen_head, net, _ = load_dir_run(args.run, dit)
    saved = torch.load(args.lora, map_location="cpu", weights_only=True)
    lora = make_lora(dit, saved["rank"])
    lora.load_state_dict(saved["lora"], strict=False)
    lora.to(dev, torch.float32)
    lora.requires_grad_(False)
    for module in lora.unet_loras:
        module.enabled = any(k.startswith(module.lora_name + ".") for k in saved["lora"])
    head = load_head(args.head, dev)
    store: dict = {}
    dit.final_layer.video_out.register_forward_pre_hook(lambda _m, a: store.__setitem__("h", a[0][0]))
    out = args.out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    jobs = []
    for folder in args.data:
        held = [p for p in samples(folder, True) if p.with_suffix(".mp4").exists()]
        train = [p for p in samples(folder, False) if p.with_suffix(".mp4").exists()]
        jobs += [(p, True) for p in held] + [(p, False) for p in rng.sample(train, min(args.train_per_folder, len(train)))]
    pairs = []
    keys = ("state", "prev", "text", "audio_seed", "sigma", "r", "s", "shape")
    t0 = time.time()
    with torch.no_grad():
        for i, (path, held) in enumerate(jobs):
            dst = out / f"{path.parent.name}__{path.stem}.mp4"
            if not dst.exists():
                d = load(path, dev, keys)
                with use_networks(net, None, gen=True, fake=False):
                    generator_step(dit, gen_head, d["state"], d["prev"], d["r"], d["text"], sigma=d["sigma"],
                                   audio_generator=torch.Generator(dev).manual_seed(d["audio_seed"]))
                write_mp4(dst, head.decode(store["h"].float(), d["s"], *d["shape"]))
            pairs.append({"dwd": str(dst), "vae": str(path.with_suffix(".mp4")), "held": held,
                          "folder": path.parent.name})
            if (i + 1) % 100 == 0:
                print(f"LOT_H3 kind=refine_cache done={i + 1}/{len(jobs)} min={(time.time() - t0) / 60:.1f}", flush=True)
    (out / "pairs.json").write_text(json.dumps(pairs, indent=1))
    print(f"LOT_H3 kind=refine_cache_done pairs={len(pairs)} out={out} ok=1", flush=True)


# ------------------------------------------------------------------------------------------- train

class Crops(torch.utils.data.IterableDataset):
    """Endless random crops of two consecutive frames: ``x`` ``(2, 9, c, c)`` uint8 (each frame with its
    neighbours), ``y`` ``(2, 3, c, c)`` uint8, and with ``latent`` the two frames' latents ``(2, 24, c/16, c/16)``.

    Two frames per sample let the loss compare the refined frame-to-frame change with the VAE's. Decoding is the
    cost (about 0.25 s a pair), so each decoded clip gives ``per_clip`` crops, cut while still uint8, and a
    shuffle pool mixes clips so a batch is not two clips' worth of crops.
    """

    def __init__(self, pairs: list[dict], crop: int, per_clip: int, seed: int, pool: int = 128, latent: bool = False,
                 context: bool = False):
        self.pairs, self.crop, self.per_clip, self.seed, self.pool = pairs, crop, per_clip, seed, pool
        self.latent, self.context = latent or context, context

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0) * 9973 + int(time.time()))
        c, grid, pool = self.crop, (16 if self.latent else 4), []
        while True:
            p = rng.choice(self.pairs)
            a, b = read_mp4_u8(Path(p["dwd"])), read_mp4_u8(Path(p["vae"]))          # (F, H, W, 3)
            f, h, w, _ = a.shape
            if a.shape != b.shape or f < 2 or h < c or w < c:
                continue
            lat = group = None
            if self.latent:
                lat = clip_latent(p)[0]                                               # (24, T, h/16, w/16)
                if pixel_frame_count(lat.shape[1]) != f:
                    continue
                group = frame_groups(lat.shape[1])
            for _ in range(self.per_clip):
                t = rng.randrange(f - 1)
                top, left = rng.randrange(0, h - c + 1, grid), rng.randrange(0, w - c + 1, grid)
                index = [max(t - 1, 0), t, t + 1, min(t + 2, f - 1)]
                win = a[index, top:top + c, left:left + c].permute(0, 3, 1, 2)       # (4, 3, c, c)
                x = torch.stack([win[0:3].reshape(9, c, c), win[1:4].reshape(9, c, c)])
                y = b[t:t + 2, top:top + c, left:left + c].permute(0, 3, 1, 2).contiguous()
                item = (x, y)
                if self.context:
                    item += (frame_context(lat[:, :, top // 16:(top + c) // 16, left // 16:(left + c) // 16].float(),
                                           [t, t + 1]),)
                elif self.latent:
                    item += (lat[:, [group[t], group[t + 1]], top // 16:(top + c) // 16,
                                 left // 16:(left + c) // 16].permute(1, 0, 2, 3).float(),)
                pool.append(item)
            while len(pool) > self.pool:
                yield pool.pop(rng.randrange(len(pool)))


def evaluate(model: Refiner, pairs: list[dict], net, dev, limit: int = 2) -> tuple[dict, list]:
    """Full held-out clips: PSNR, LPIPS (every 6th frame), flicker inside / across latent frames, seam ratio and
    the error of the frame-to-frame change (``tde``), for the refined output and for the plain DWD output it
    starts from. Also returns ``(name, frames)`` of the first ``limit`` clips, VAE | DWD | refined, for videos."""
    rows, base, keep = {}, {}, []
    model.eval()
    for k, p in enumerate(pairs):
        dwd = read_mp4(Path(p["dwd"])).to(dev)
        vae = read_mp4(Path(p["vae"])).to(dev)
        ref = refine_clip(model, dwd, clip_latent(p).to(dev) if model.latent else None)
        t = latent_frames(dwd.shape[2])
        dv = vae[:, :, 1:] - vae[:, :, :-1]
        for name, clip, store in (("refined", ref, rows), ("dwd", dwd, base)):
            a, b = clip[0].permute(1, 0, 2, 3)[::6], vae[0].permute(1, 0, 2, 3)[::6]
            with torch.no_grad():
                lp = sum(net(a[i:i + 1], b[i:i + 1]).item() for i in range(a.shape[0])) / a.shape[0]
            fi, fa = flicker_split(clip, t)
            tde = (clip[:, :, 1:] - clip[:, :, :-1] - dv).abs().mean().item()
            store.setdefault(p["folder"], []).append({"psnr": psnr(clip, vae), "lpips": lp, "flicker_in": fi,
                                                      "flicker_across": fa, "seam": seam_ratio(clip), "tde": tde})
        if k < limit:
            wide = torch.cat([vae, dwd, ref], dim=-1)[0]
            keep.append((f"{p['folder']}__{Path(p['dwd']).stem.split('__')[-1]}",
                         to_uint8(wide).permute(1, 2, 3, 0).cpu().numpy()))
    model.train()

    def mean(d):
        per = {f: {k: sum(r[k] for r in v) / len(v) for k in v[0]} for f, v in d.items()}
        overall = {k: sum(m[k] for m in per.values()) / len(per) for k in next(iter(per.values()))}
        for f, m in per.items():
            overall[f"psnr@{f}"], overall[f"lpips@{f}"] = m["psnr"], m["lpips"]
        return overall
    return {"refined": mean(rows), "dwd": mean(base)}, keep


def write_videos(keep: list, videos: Path) -> None:
    import imageio

    videos.mkdir(parents=True, exist_ok=True)
    for name, frames in keep:
        imageio.mimwrite(videos / f"{name}.mp4", list(frames), fps=24, quality=8, macro_block_size=16)
        imageio.imwrite(videos / f"{name}_f20.jpg", frames[min(20, len(frames) - 1)])


def train(args) -> None:
    if os.environ.get("LOT_H3_TRAIN") != "1":
        raise SystemExit("train is gated: set LOT_H3_TRAIN=1")
    import lpips

    dev = torch.device(args.device)
    pairs = json.loads((args.out / "cache" / "pairs.json").read_text())
    train_pairs = [p for p in pairs if not p["held"]]
    held = [p for p in pairs if p["held"]]
    latent_spec = CONTEXT_CH if args.latent_context else args.latent
    args.latent = bool(args.latent or args.latent_context)
    model = Refiner(args.base, latent=latent_spec, grid=args.latent_grid, grid_blocks=args.grid_blocks).to(dev)
    spec = {"base": args.base, "latent": latent_spec, "grid": model.grid_width, "grid_blocks": model.grid_blocks}
    net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    last_path = args.out / "refiner_last.pt"
    step, best, elapsed, log = 0, math.inf, 0.0, []
    if last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        if spec_of(last) != spec:
            raise SystemExit(f"{last_path} is a {spec_of(last)} refiner, not {spec}")
        model.load_state_dict(last["model"])
        opt.load_state_dict(last["opt"])
        for group in opt.param_groups:                            # the saved optimiser carries its own lr: --lr wins
            group["lr"] = args.lr
        step, best, elapsed, log = last["step"], last["best"], last["elapsed"], last["log"]
        print(f"LOT_H3 kind=refine_resume step={step} elapsed_min={elapsed:.1f} best_lpips={best:.4f}", flush=True)
    else:
        if args.init:
            missing = init_from(model, torch.load(args.init, map_location="cpu", weights_only=False))
            print(f"LOT_H3 kind=refine_init from={args.init} new={len(missing)}", flush=True)
        # with --init the refined row of this step-0 evaluation must equal the source's own evaluation
        m, _keep = evaluate(model, held, net, dev, limit=0)
        if args.init:
            print("LOT_H3 kind=refine_eval_init step=0 " + " ".join(f"{k}={v:.4f}" for k, v in m["refined"].items()),
                  flush=True)
        log.append({"step": 0, **m})
        print("LOT_H3 kind=refine_eval step=0 " + " ".join(f"{k}={v:.4f}" for k, v in m["dwd"].items()), flush=True)
    ema_model = None
    if args.ema > 0:                                              # averaged weights: evaluated and saved
        import copy
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)       # after any load: starts at the weights
        if last_path.exists() and last.get("ema_model") is not None:
            ema_model.load_state_dict(last["ema_model"])
    print(f"LOT_H3 kind=refine params={sum(p.numel() for p in model.parameters())} train_pairs={len(train_pairs)} "
          f"held={len(held)}", flush=True)
    loader = torch.utils.data.DataLoader(Crops(train_pairs, args.crop, args.per_clip, seed=step, pool=args.pool,
                                               latent=args.latent, context=args.latent_context), batch_size=args.batch,
                                         num_workers=args.workers, persistent_workers=True, prefetch_factor=4)
    t_start = time.time() - elapsed * 60
    next_eval = (math.floor(elapsed / args.eval_minutes) + 1) * args.eval_minutes * 60
    ema, skipped = None, 0

    def checkpoint():
        tmp = last_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "elapsed": (time.time() - t_start) / 60, "log": log, "base": args.base,
                    "latent": args.latent, "latent_spec": latent_spec, "grid": model.grid_width,
                    "grid_blocks": model.grid_blocks,
                    "ema_model": None if ema_model is None else ema_model.state_dict()}, tmp)
        tmp.replace(last_path)

    for batch in loader:
        if time.time() - t_start > args.minutes * 60:
            break
        x = batch[0].to(dev, non_blocking=True).flatten(0, 1).float() / 127.5 - 1      # (2B, 9, c, c): frame pairs
        y = batch[1].to(dev, non_blocking=True).flatten(0, 1).float() / 127.5 - 1
        lat = batch[2].to(dev, non_blocking=True).flatten(0, 1) if args.latent else None
        pred = model(x, lat)
        l1 = (pred - y).abs().mean()
        lp = net(pred.clamp(-1, 1), y).mean()
        pp, yy = pred.unflatten(0, (-1, 2)), y.unflatten(0, (-1, 2))
        # the refined frame-to-frame change should be the VAE's: detail invented per frame would flicker
        td = ((pp[:, 1] - pp[:, 0]) - (yy[:, 1] - yy[:, 0])).abs().mean()
        loss = l1 + args.lpips_weight * lp + args.temporal_weight * td
        if not torch.isfinite(loss) or (ema is not None and step > 50 and loss.item() > 20 * ema):
            skipped += 1
            print(f"LOT_H3 kind=refine_skip step={step} loss={loss.item():.4g} skipped={skipped}", flush=True)
            if skipped > 20:
                raise SystemExit("too many skipped steps")
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        if ema_model is not None:
            with torch.no_grad():
                for pe, pm in zip(ema_model.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - args.ema)
        step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % args.log_every == 0:
            print(f"LOT_H3 kind=refine_train step={step} loss={loss.item():.4f} ema={ema:.4f} l1={l1.item():.4f} "
                  f"lpips={lp.item():.4f} td={td.item():.4f} grad_norm={grad_norm:.3f} min={(time.time() - t_start) / 60:.1f} "
                  f"peak_gb={torch.cuda.max_memory_allocated(dev) / 2**30:.1f}", flush=True)
        if time.time() - t_start > next_eval:
            next_eval += args.eval_minutes * 60
            judged = ema_model if ema_model is not None else model          # with --ema the EMA is what counts
            m, keep = evaluate(judged, held, net, dev)
            log.append({"step": step, **m})
            print(f"LOT_H3 kind=refine_eval step={step} " + " ".join(f"{k}={v:.4f}" for k, v in m["refined"].items()),
                  flush=True)
            if m["refined"]["lpips"] < best:
                best = m["refined"]["lpips"]
                torch.save({"model": judged.state_dict(), "base": args.base, "latent": args.latent,
                            "latent_spec": latent_spec, "grid": model.grid_width,
                            "grid_blocks": model.grid_blocks, "step": step}, args.out / "refiner_best.pt")
                write_videos(keep, args.out / "videos")
                print(f"LOT_H3 kind=refine_save step={step} lpips={best:.4f} out={args.out}", flush=True)
            del keep
            checkpoint()
    checkpoint()
    print(f"LOT_H3 kind=refine_done steps={step} best_lpips={best:.4f} out={args.out} ok=1", flush=True)


def bench(args) -> None:
    """Seconds per clip of ``refine_clip`` on random input: float32 and bfloat16 autocast, a few chunk sizes."""
    dev = torch.device(args.device)
    if args.ckpt:
        saved = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = Refiner(**spec_of(saved))
        model.load_state_dict(saved["model"])
    else:
        model = Refiner(args.base, latent=CONTEXT_CH if args.latent_context else args.latent, grid=args.latent_grid,
                        grid_blocks=args.grid_blocks)
    model = model.to(dev).eval()
    t = latent_frames(args.frames)
    clip = torch.rand(1, 3, args.frames, args.height, args.width, device=dev) * 2 - 1
    latent = torch.randn(1, 24, t, args.height // 16, args.width // 16, device=dev) if model.latent else None

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    print(f"LOT_H3 kind=refine_bench params={sum(p.numel() for p in model.parameters())} frames={args.frames} "
          f"size={args.height}x{args.width} device={dev}", flush=True)
    for amp in (False, True):
        for chunk in args.chunks:
            times = []
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            for _ in range(args.repeats + 1):                      # the first run is the warm-up
                sync()
                t0 = time.time()
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    out = refine_clip(model, clip, latent, chunk=chunk)
                sync()
                times.append(time.time() - t0)
            peak = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0
            print(f"LOT_H3 kind=refine_bench amp={int(amp)} chunk={chunk} s_per_clip={min(times[1:]):.4f} "
                  f"mean={sum(times[1:]) / len(times[1:]):.4f} peak_gb={peak:.2f} out={tuple(out.shape)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="phase", required=True)
    c = sub.add_parser("cache", help="GPU: render DWD decodes to mp4")
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--data", type=Path, nargs="+", required=True)
    c.add_argument("--lora", type=Path, required=True)
    c.add_argument("--head", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--train-per-folder", type=int, default=500)
    t = sub.add_parser("train", help="train the refiner on cached pairs")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--minutes", type=float, default=240)
    t.add_argument("--eval-minutes", type=float, default=20)
    t.add_argument("--base", type=int, default=48)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--batch", type=int, default=8, help="frame pairs per step (two frames each)")
    t.add_argument("--crop", type=int, default=256)
    t.add_argument("--per-clip", type=int, default=24, help="crops taken from each decoded clip pair")
    t.add_argument("--pool", type=int, default=128, help="shuffle pool per worker, in samples")
    t.add_argument("--temporal-weight", type=float, default=1.0,
                   help="weight of the L1 between the refined and the VAE frame-to-frame change")
    t.add_argument("--latent", action="store_true",
                   help="also feed each frame's stored latent (24 channels at 1/16 resolution) to the refiner")
    t.add_argument("--latent-context", action="store_true",
                   help="feed the latent frames before, at and after each pixel frame's own, plus its sub-frame phase")
    t.add_argument("--latent-grid", type=int, default=0, metavar="WIDTH",
                   help="also run the latent input through a WIDTH-wide path on the latent grid, pixel-shuffled into "
                        "the bottleneck (e.g. 256); needs --latent or --latent-context")
    t.add_argument("--grid-blocks", type=int, default=4, help="residual blocks of the latent-grid path")
    t.add_argument("--init", type=Path, default=None,
                   help="start a new run from another refiner's weights (refiner_best.pt), same base; new inputs and "
                        "the latent-grid path start at zero, so step 0 reproduces the source")
    t.add_argument("--ema", type=float, default=0.0, help="weight EMA decay (e.g. 0.999); the EMA is evaluated and saved")
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--lpips-weight", type=float, default=0.5)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--device", default="cuda")
    b = sub.add_parser("bench", help="time refine_clip on random input")
    b.add_argument("--ckpt", type=Path, default=None, help="a saved refiner; otherwise the flags below build one")
    b.add_argument("--base", type=int, default=48)
    b.add_argument("--latent", action="store_true")
    b.add_argument("--latent-context", action="store_true")
    b.add_argument("--latent-grid", type=int, default=0)
    b.add_argument("--grid-blocks", type=int, default=4)
    b.add_argument("--frames", type=int, default=39)
    b.add_argument("--height", type=int, default=512)
    b.add_argument("--width", type=int, default=768)
    b.add_argument("--chunks", type=int, nargs="+", default=[8, 13, 39])
    b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        from setproctitle import setproctitle
        setproctitle(f"dwd:refine-{args.phase} {args.out.name if hasattr(args, 'out') else ''}")
    except ImportError:
        pass
    {"cache": cache, "train": train, "bench": bench}[args.phase](args)


if __name__ == "__main__":
    main()
