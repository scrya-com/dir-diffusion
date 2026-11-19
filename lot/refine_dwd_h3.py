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


class Refiner(nn.Module):
    """``(N, 9, H, W)`` (previous, current, next frame) -> residual for the current frame. Zero-initialised
    output, so an untrained refiner returns the DWD frame unchanged."""

    def __init__(self, base: int = 48, latent: bool = False):
        super().__init__()
        b = base
        self.latent = latent                                   # also read the frame's 24-channel latent (1/16 res)
        self.inp = nn.Conv2d(9 + (24 if latent else 0), b, 3, padding=1)
        self.e1 = Block(b)
        self.d1 = nn.Conv2d(b, 2 * b, 4, stride=2, padding=1)
        self.e2 = Block(2 * b)
        self.d2 = nn.Conv2d(2 * b, 4 * b, 4, stride=2, padding=1)
        self.mid = nn.Sequential(Block(4 * b), Block(4 * b))
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
        h = self.mid(self.d2(h2))
        h = self.m2(self.u2(h) + h2)
        h = self.m1(self.u1(h) + h1)
        return x[:, 3:6] + self.out(h)


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
    if model.latent:
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

    def __init__(self, pairs: list[dict], crop: int, per_clip: int, seed: int, pool: int = 128, latent: bool = False):
        self.pairs, self.crop, self.per_clip, self.seed, self.pool, self.latent = pairs, crop, per_clip, seed, pool, latent

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
                if self.latent:
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
    model = Refiner(args.base, latent=args.latent).to(dev)
    net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    last_path = args.out / "refiner_last.pt"
    step, best, elapsed, log = 0, math.inf, 0.0, []
    if last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        if last.get("base", args.base) != args.base or last.get("latent", False) != args.latent:
            raise SystemExit(f"{last_path} is a base={last.get('base')} latent={last.get('latent', False)} refiner")
        model.load_state_dict(last["model"])
        opt.load_state_dict(last["opt"])
        step, best, elapsed, log = last["step"], last["best"], last["elapsed"], last["log"]
        print(f"LOT_H3 kind=refine_resume step={step} elapsed_min={elapsed:.1f} best_lpips={best:.4f}", flush=True)
    else:
        m, _keep = evaluate(model, held, net, dev, limit=0)
        log.append({"step": 0, **m})
        print("LOT_H3 kind=refine_eval step=0 " + " ".join(f"{k}={v:.4f}" for k, v in m["dwd"].items()), flush=True)
    print(f"LOT_H3 kind=refine params={sum(p.numel() for p in model.parameters())} train_pairs={len(train_pairs)} "
          f"held={len(held)}", flush=True)
    loader = torch.utils.data.DataLoader(Crops(train_pairs, args.crop, args.per_clip, seed=step, pool=args.pool,
                                               latent=args.latent), batch_size=args.batch,
                                         num_workers=args.workers, persistent_workers=True, prefetch_factor=4)
    t_start = time.time() - elapsed * 60
    next_eval = (math.floor(elapsed / args.eval_minutes) + 1) * args.eval_minutes * 60
    ema, skipped = None, 0

    def checkpoint():
        tmp = last_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "elapsed": (time.time() - t_start) / 60, "log": log, "base": args.base,
                    "latent": args.latent}, tmp)
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
        step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % args.log_every == 0:
            print(f"LOT_H3 kind=refine_train step={step} loss={loss.item():.4f} ema={ema:.4f} l1={l1.item():.4f} "
                  f"lpips={lp.item():.4f} td={td.item():.4f} grad_norm={grad_norm:.3f} min={(time.time() - t_start) / 60:.1f} "
                  f"peak_gb={torch.cuda.max_memory_allocated(dev) / 2**30:.1f}", flush=True)
        if time.time() - t_start > next_eval:
            next_eval += args.eval_minutes * 60
            m, keep = evaluate(model, held, net, dev)
            log.append({"step": step, **m})
            print(f"LOT_H3 kind=refine_eval step={step} " + " ".join(f"{k}={v:.4f}" for k, v in m["refined"].items()),
                  flush=True)
            if m["refined"]["lpips"] < best:
                best = m["refined"]["lpips"]
                torch.save({"model": model.state_dict(), "base": args.base, "latent": args.latent, "step": step},
                           args.out / "refiner_best.pt")
                write_videos(keep, args.out / "videos")
                print(f"LOT_H3 kind=refine_save step={step} lpips={best:.4f} out={args.out}", flush=True)
            del keep
            checkpoint()
    checkpoint()
    print(f"LOT_H3 kind=refine_done steps={step} best_lpips={best:.4f} out={args.out} ok=1", flush=True)


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
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--lpips-weight", type=float, default=0.5)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        from setproctitle import setproctitle
        setproctitle(f"dwd:refine-{args.phase} {args.out.name}")
    except ImportError:
        pass
    {"cache": cache, "train": train}[args.phase](args)


if __name__ == "__main__":
    main()
