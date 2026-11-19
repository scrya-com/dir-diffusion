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

from dwd_h3 import flicker_split, psnr, seam_ratio  # noqa: E402
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

    def __init__(self, base: int = 48):
        super().__init__()
        b = base
        self.inp = nn.Conv2d(9, b, 3, padding=1)
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

    def forward(self, x):
        h1 = self.e1(self.inp(x))
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


@torch.no_grad()
def refine_clip(model: Refiner, clip: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """``(1, 3, F, H, W)`` in [-1, 1] -> refined clip, same shape. H and W must be multiples of 4."""
    frames = clip[0].permute(1, 0, 2, 3)
    x = neighbours(frames)
    out = torch.cat([model(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)])
    return out.clamp(-1, 1).permute(1, 0, 2, 3).unsqueeze(0)


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
    """Endless random (DWD triple, VAE middle frame) crops; one mp4 pair decoded per ``per_clip`` crops."""

    def __init__(self, pairs: list[dict], crop: int, per_clip: int, seed: int):
        self.pairs, self.crop, self.per_clip, self.seed = pairs, crop, per_clip, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0) * 9973 + int(time.time()))
        while True:
            p = rng.choice(self.pairs)
            a = read_mp4(Path(p["dwd"]))[0].permute(1, 0, 2, 3)          # (F, 3, H, W)
            b = read_mp4(Path(p["vae"]))[0].permute(1, 0, 2, 3)
            if a.shape != b.shape:
                continue
            x = neighbours(a)
            f, _, h, w = x.shape
            for _ in range(self.per_clip):
                t = rng.randrange(f)
                top, left = rng.randrange(0, h - self.crop + 1, 4), rng.randrange(0, w - self.crop + 1, 4)
                yield (x[t, :, top:top + self.crop, left:left + self.crop],
                       b[t, :, top:top + self.crop, left:left + self.crop])


def evaluate(model: Refiner, pairs: list[dict], net, dev, videos: Path | None = None, limit: int = 2) -> dict:
    """Full held-out clips: PSNR, LPIPS (every 6th frame), flicker inside / across latent frames, seam ratio;
    for the refined output and, once, for the plain DWD output it starts from."""
    rows, base = {}, {}
    model.eval()
    for k, p in enumerate(pairs):
        dwd = read_mp4(Path(p["dwd"])).to(dev)
        vae = read_mp4(Path(p["vae"])).to(dev)
        ref = refine_clip(model, dwd)
        t = 12                                                         # 39 frames = 12 latent frames
        for name, clip, store in (("refined", ref, rows), ("dwd", dwd, base)):
            a, b = clip[0].permute(1, 0, 2, 3)[::6], vae[0].permute(1, 0, 2, 3)[::6]
            with torch.no_grad():
                lp = sum(net(a[i:i + 1], b[i:i + 1]).item() for i in range(a.shape[0])) / a.shape[0]
            fi, fa = flicker_split(clip, t)
            store.setdefault(p["folder"], []).append({"psnr": psnr(clip, vae), "lpips": lp, "flicker_in": fi,
                                                      "flicker_across": fa, "seam": seam_ratio(clip)})
        if videos is not None and k < limit:
            videos.mkdir(parents=True, exist_ok=True)
            wide = torch.cat([vae, dwd, ref], dim=-1)[0]
            frames = to_uint8(wide).permute(1, 2, 3, 0).cpu().numpy()
            import imageio
            imageio.mimwrite(videos / f"{p['folder']}__{Path(p['dwd']).stem.split('__')[-1]}.mp4", list(frames),
                             fps=24, quality=8, macro_block_size=16)
            imageio.imwrite(videos / f"{p['folder']}__{Path(p['dwd']).stem.split('__')[-1]}_f20.jpg", frames[20])
    model.train()

    def mean(d):
        per = {f: {k: sum(r[k] for r in v) / len(v) for k in v[0]} for f, v in d.items()}
        overall = {k: sum(m[k] for m in per.values()) / len(per) for k in next(iter(per.values()))}
        for f, m in per.items():
            overall[f"psnr@{f}"], overall[f"lpips@{f}"] = m["psnr"], m["lpips"]
        return overall
    return {"refined": mean(rows), "dwd": mean(base)}


def train(args) -> None:
    if os.environ.get("LOT_H3_TRAIN") != "1":
        raise SystemExit("train is gated: set LOT_H3_TRAIN=1")
    import lpips

    dev = torch.device(args.device)
    pairs = json.loads((args.out / "cache" / "pairs.json").read_text())
    train_pairs = [p for p in pairs if not p["held"]]
    held = [p for p in pairs if p["held"]]
    model = Refiner(args.base).to(dev)
    net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    last_path = args.out / "refiner_last.pt"
    step, best, elapsed, log = 0, math.inf, 0.0, []
    if last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(last["model"])
        opt.load_state_dict(last["opt"])
        step, best, elapsed, log = last["step"], last["best"], last["elapsed"], last["log"]
        print(f"LOT_H3 kind=refine_resume step={step} elapsed_min={elapsed:.1f} best_lpips={best:.4f}", flush=True)
    else:
        m = evaluate(model, held, net, dev)
        log.append({"step": 0, **m})
        print("LOT_H3 kind=refine_eval step=0 " + " ".join(f"{k}={v:.4f}" for k, v in m["dwd"].items()), flush=True)
    print(f"LOT_H3 kind=refine params={sum(p.numel() for p in model.parameters())} train_pairs={len(train_pairs)} "
          f"held={len(held)}", flush=True)
    loader = torch.utils.data.DataLoader(Crops(train_pairs, args.crop, args.per_clip, seed=step), batch_size=args.batch,
                                         num_workers=args.workers, persistent_workers=True, prefetch_factor=4)
    t_start = time.time() - elapsed * 60
    next_eval = (math.floor(elapsed / args.eval_minutes) + 1) * args.eval_minutes * 60
    ema, skipped = None, 0

    def checkpoint():
        tmp = last_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "elapsed": (time.time() - t_start) / 60, "log": log, "base": args.base}, tmp)
        tmp.replace(last_path)

    for x, y in loader:
        if time.time() - t_start > args.minutes * 60:
            break
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        pred = model(x)
        l1 = (pred - y).abs().mean()
        lp = net(pred.clamp(-1, 1), y).mean()
        loss = l1 + args.lpips_weight * lp
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
                  f"lpips={lp.item():.4f} grad_norm={grad_norm:.3f} min={(time.time() - t_start) / 60:.1f} "
                  f"peak_gb={torch.cuda.max_memory_allocated(dev) / 2**30:.1f}", flush=True)
        if time.time() - t_start > next_eval:
            next_eval += args.eval_minutes * 60
            m = evaluate(model, held, net, dev)
            log.append({"step": step, **m})
            print(f"LOT_H3 kind=refine_eval step={step} " + " ".join(f"{k}={v:.4f}" for k, v in m["refined"].items()),
                  flush=True)
            if m["refined"]["lpips"] < best:
                best = m["refined"]["lpips"]
                torch.save({"model": model.state_dict(), "base": args.base, "step": step}, args.out / "refiner_best.pt")
                evaluate(model, held, net, dev, videos=args.out / "videos")
                print(f"LOT_H3 kind=refine_save step={step} lpips={best:.4f} out={args.out}", flush=True)
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
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--crop", type=int, default=256)
    t.add_argument("--per-clip", type=int, default=8, help="crops taken from each decoded clip pair")
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
