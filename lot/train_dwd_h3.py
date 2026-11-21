#!/usr/bin/env python3
"""Train a DWD pixel head (``dwd_h3.py``) on a DiR generator's final iteration.

    # 1. GPU: run DiR, catch the last-layer tokens of the final iteration, decode x0_hat with the VAE.
    #    Also stores the final iteration's inputs (for `ft`); on files collected before that, it replays the
    #    first R-1 iterations to add them (no VAE).
    python3 lot/train_dwd_h3.py collect --run lot/runs/dir_turbo_clip --out DATA --items 192 --seeds 2

    # 2. least squares: standardised features, one head per latent frame position (k % 5); no DiT
    python3 lot/train_dwd_h3.py ridge --data DATA --out RUN

    # 3a. refine the head alone (pseudo-Huber), no DiT
    LOT_H3_TRAIN=1 python3 lot/train_dwd_h3.py sgd --data DATA --out RUN --steps 2000

    # 3b. the paper's lever: a LoRA on the last K blocks, on only at the final iteration, trained with a pixel
    #     loss through the head (head trained jointly at a low rate)
    LOT_H3_TRAIN=1 python3 lot/train_dwd_h3.py ft --run lot/runs/dir_turbo_clip --data DATA --out RUN --minutes 90

The target is the VAE decode of the generator's own final ``x0_hat``. That can match the VAE more cheaply but
cannot beat it; beating it needs real pixels.

Evaluation: a fixed probe set (the held-out prompts, fixed seeds), scored by PSNR against the VAE decode, by
flicker inside a latent frame and across latent frames (each against the VAE clip's), and by the seam ratio at
32-pixel token edges. Not by the training loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dwd_h3 import (CLASSES, HIDDEN, OUT, TOKEN_DIM, DWDHead, ext_out, capture_last_layer, flicker_split,  # noqa: E402
                    pack_rows, pixel_frame_count, pixels_to_tokens, pseudo_huber, psnr, seam_ratio, slot_mask,
                    token_classes, tokens_to_pixels)

INPUT_KEYS = ("state", "prev", "text", "audio_seed", "sigma", "r")


def to_uint8(pixels: torch.Tensor) -> torch.Tensor:
    return ((pixels.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def from_uint8(y: torch.Tensor) -> torch.Tensor:
    return y.float() / 127.5 - 1.0


def sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


# ----------------------------------------------------------------------------------- collect

def collect(args) -> None:
    from gpu_guard import refuse_if_busy

    refuse_if_busy("train_dwd_h3.py")
    import train_h3
    from dir_h3 import generator_step, load_dir_run, next_state, parse_sigmas, use_networks
    from h3_musubi import load_dit, load_vae
    from lot_paths import LOT_H3_VAE
    from train_dir_h3 import load_item

    train_h3.MAX_LATENT_HW = tuple(args.max_latent_hw)
    train_h3.CROP_MULTIPLE = 2
    dev = torch.device("cuda")
    vae_dev = torch.device(args.vae_device)
    train_items, held_items = train_h3.list_items(args.cache or train_h3.CACHE, args.holdout)
    dit = load_dit(device=dev, swap=args.swap)
    head, net, config = load_dir_run(args.run, dit, tag=f"_{args.tag}" if args.tag else "", ema=args.ema)
    R = int(config.get("iters", 4))
    sigmas = parse_sigmas(config.get("sigmas") or "1", R)
    mix = bool(config.get("mix_state", False))
    vae = vae_dtype = None
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "collect.json").write_text(json.dumps({
        "run": str(args.run), "tag": args.tag, "ema": args.ema, "iters": R, "sigmas": sigmas, "mix_state": mix,
        "frames": args.frames, "max_latent_hw": args.max_latent_hw, "cache": str(args.cache or train_h3.CACHE),
        "holdout": args.holdout}, indent=1))

    jobs = [(True, i, it, s) for i, it in enumerate(held_items[:args.held_items]) for s in range(args.held_seeds)]
    jobs += [(False, i, it, s) for i, it in enumerate(train_items[:args.items]) for s in range(args.seeds)]
    store: dict = {}
    with torch.no_grad():
        for held, index, item, seed in jobs:
            name = f"{'held' if held else 'train'}_{index:04d}_{seed}.pt"
            path = args.out / name
            existing = None
            if path.exists() and args.light:
                continue
            if path.exists():
                existing = torch.load(path, map_location="cpu", weights_only=True)
                if all(k in existing for k in INPUT_KEYS):
                    continue
            base = 0 if held else 100000
            x0, text = load_item(item, random.Random(1000 + base + index), dev, args.frames)
            z = torch.randn(x0.shape, device=dev,
                            generator=torch.Generator(dev).manual_seed(5000 + base + 100 * index + seed)).to(x0.dtype)
            audio_seed = 31 * (base + index) + seed
            estimate, state = torch.zeros_like(z), z
            last_r = R - 1 if existing is None else R - 2     # a replay stops before the final iteration
            t0 = time.time()
            with use_networks(net, None, gen=True, fake=False):
                for r in range(last_r + 1):
                    final = r == R - 1
                    with (capture_last_layer(dit, store) if final else _null()):
                        estimate = generator_step(dit, head, state, estimate, r, text, sigma=sigmas[r],
                                                  audio_generator=torch.Generator(dev).manual_seed(audio_seed)).float()
                    if r + 1 < R:
                        state = next_state(state, estimate, sigmas[r], sigmas[r + 1], z if mix else None)
                        prev = estimate
            inputs = {"state": state.float().cpu(), "prev": prev.float().cpu(), "text": text.cpu(),
                      "audio_seed": audio_seed, "sigma": sigmas[R - 1], "r": R - 1}
            if args.light:                                        # about 4 MB: no tokens, no targets, text by path
                latent_t, lat_h, lat_w = estimate.shape[2:]
                torch.save({"state": state.half().cpu(), "prev": prev.half().cpu(), "latent": estimate.half().cpu(),
                            "te": str(item[1]), "audio_seed": audio_seed, "sigma": sigmas[R - 1], "r": R - 1,
                            "shape": (latent_t, lat_h // 2, lat_w // 2), "held": held, "item": index, "seed": seed,
                            "dit_s": time.time() - t0}, path)
                print(f"LOT_H3 kind=dwd_collect sample={name} light=1 dit_s={time.time() - t0:.1f} "
                      f"peak_mb={torch.cuda.max_memory_allocated(dev) / 2**20:.0f}", flush=True)
                continue
            if existing is not None:
                s_rows = pack_rows(state.float()).half().cpu()
                if not torch.equal(s_rows, existing["s"]):
                    err = (s_rows.float() - existing["s"].float()).abs().max().item()
                    print(f"LOT_H3 kind=dwd_collect note=replayed_state_differs sample={name} max_abs={err:.3g}", flush=True)
                existing.update(inputs)
                torch.save(existing, path)
                print(f"LOT_H3 kind=dwd_collect sample={name} added=inputs dit_s={time.time() - t0:.1f}", flush=True)
                continue
            dit_s = time.time() - t0
            h = store["h"]
            latent_t, lat_h, lat_w = estimate.shape[2:]
            if h.shape != (latent_t * (lat_h // 2) * (lat_w // 2), HIDDEN):
                raise RuntimeError(f"captured tokens {tuple(h.shape)} do not match the latent {tuple(estimate.shape)}")
            if vae is None:
                vae = load_vae(LOT_H3_VAE, device=vae_dev)
                vae_dtype = next(vae.parameters()).dtype
            sync(vae_dev)
            t0 = time.time()
            pixels = vae.decode(estimate.to(vae_dev, vae_dtype))
            sync(vae_dev)
            vae_ms = 1000 * (time.time() - t0)
            if pixels.shape[2] != pixel_frame_count(latent_t):
                raise RuntimeError(f"VAE gave {pixels.shape[2]} frames for {latent_t} latent frames")
            y = to_uint8(pixels_to_tokens(pixels.float().cpu(), latent_t))
            torch.save({
                "h": h.half().cpu(), "s": pack_rows(inputs["state"]).half(), "y": y,
                "latent": estimate.half().cpu(), "shape": (latent_t, lat_h // 2, lat_w // 2), "held": held,
                "item": index, "seed": seed, "vae_ms": vae_ms, "dit_s": dit_s, **inputs,
            }, path)
            print(f"LOT_H3 kind=dwd_collect sample={name} tokens={h.shape[0]} dit_s={dit_s:.1f} vae_ms={vae_ms:.0f} "
                  f"peak_mb={torch.cuda.max_memory_allocated(dev) / 2**20:.0f}", flush=True)
    (args.out / "collect.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"LOT_H3 kind=dwd_collect_done out={args.out} samples={len(list(args.out.glob('*.pt')))} ok=1", flush=True)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ------------------------------------------------------------------------------------ eval

def samples(data: Path, held: bool) -> list[Path]:
    return sorted(data.glob(f"{'held' if held else 'train'}_*.pt"))


def load(path: Path, device, keys=None) -> dict:
    d = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "te" in d:                                                 # light sample: fill text, input rows, targets
        d = dict(d)
        d["state"], d["prev"] = d["state"].float(), d["prev"].float()
        d["text"] = load_text(d["te"])
        d["s"] = pack_rows(d["state"]).half()
        if keys is None or "y" in keys:
            d["y"] = to_uint8(pixels_to_tokens(read_mp4(path.with_suffix(".mp4")), d["shape"][0]))
    keys = keys or d.keys()
    return {k: (d[k].to(device) if torch.is_tensor(d[k]) else d[k]) for k in keys if k in d}


def load_text(te_path: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(te_path, "pt") as handle:
        hidden, mask = handle.get_tensor("hidden_states"), handle.get_tensor("attention_mask")
    return hidden[mask].unsqueeze(0).to(torch.bfloat16)


def read_mp4(path: Path) -> torch.Tensor:
    """``(1, 3, F, H, W)`` in [-1, 1], every frame."""
    import subprocess

    import numpy as np

    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    w, h = map(int, probe.stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return torch.from_numpy(a.copy()).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1


def write_mp4(path: Path, pixels: torch.Tensor) -> None:
    """Near-lossless target video (x264 crf 8, 4:4:4), written atomically."""
    import subprocess

    frames = to_uint8(pixels[0]).permute(1, 2, 3, 0).contiguous().cpu().numpy()
    f, h, w, _ = frames.shape
    tmp = path.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "24",
                    "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "8", "-pix_fmt", "yuv444p", str(tmp)],
                   input=frames.tobytes(), check=True)
    tmp.replace(path)


def targets(args) -> None:
    """Render the VAE decode of each light sample's latent to ``<sample>.mp4`` (on the second card), following a
    collection that is still running until it writes ``collect.done``."""
    from h3_musubi import load_vae
    from lot_paths import LOT_H3_VAE

    dev = torch.device(args.device)
    vae = load_vae(LOT_H3_VAE, device=dev)
    dtype = next(vae.parameters()).dtype
    done_count = 0
    while True:
        pending = sorted((p for p in args.data.glob("*.pt") if not p.with_suffix(".mp4").exists()),
                         key=lambda p: (not p.name.startswith("held"), p.name))   # held-out first: evals need them
        for path in pending:
            d = torch.load(path, map_location="cpu", weights_only=True)
            if "latent" not in d or "te" not in d:
                continue
            with torch.no_grad():
                pixels = vae.decode(d["latent"].to(dev, dtype)).float()
            if pixels.shape[2] != pixel_frame_count(d["shape"][0]):
                raise RuntimeError(f"{path.name}: VAE gave {pixels.shape[2]} frames")
            write_mp4(path.with_suffix(".mp4"), pixels)
            done_count += 1
            if done_count % 50 == 0:
                print(f"LOT_H3 kind=dwd_targets done={done_count} pending={len(pending)}", flush=True)
        if (args.data / "collect.done").exists() and not pending:
            break
        if not pending:
            time.sleep(30)
    print(f"LOT_H3 kind=dwd_targets_done out={args.data} rendered={done_count} ok=1", flush=True)


def score(pred: torch.Tensor, target: torch.Tensor, latent_t: int) -> dict:
    inside, across = flicker_split(pred, latent_t)
    v_inside, v_across = flicker_split(target, latent_t)
    return {"psnr": psnr(pred, target), "flicker_in": inside, "flicker_in_vae": v_inside,
            "flicker_across": across, "flicker_across_vae": v_across,
            "seam": seam_ratio(pred), "seam_vae": seam_ratio(target)}


def summarise(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


@torch.no_grad()
def evaluate(head: DWDHead, files: list[Path], device, *, videos: Path | None = None, limit_videos: int = 2,
             tokens_fn=None, lpips_net=None) -> dict:
    """Score the head on the probe set. ``tokens_fn(d)`` overrides where ``h`` comes from (``ft``)."""
    rows = []
    for k, path in enumerate(files):
        d = load(path, device)
        t, th, tw = d["shape"]
        target = tokens_to_pixels(from_uint8(d["y"]), t, th, tw)
        h = d["h"] if tokens_fn is None else tokens_fn(d)
        sync(device)
        t0 = time.time()
        pred = head.decode(h, d["s"], t, th, tw)
        sync(device)
        row = {"file": path.name, **score(pred, target, t), "head_ms": 1000 * (time.time() - t0)}
        if lpips_net is not None:                                 # perceptual distance on every 6th frame
            a = pred[:, :, ::6].permute(0, 2, 1, 3, 4).flatten(0, 1)
            b = target[:, :, ::6].permute(0, 2, 1, 3, 4).flatten(0, 1)
            row["lpips"] = sum(lpips_net(a[i:i + 1].float(), b[i:i + 1].float()).item()   # one frame at a time:
                               for i in range(a.shape[0])) / a.shape[0]                   # VGG at 768 px is large
        rows.append(row)
        if videos is not None and k < limit_videos:
            write_compare(videos / f"{path.stem}.mp4", target, pred)
    return {"mean": summarise(rows), "rows": rows}


def fmt(m: dict) -> str:
    return " ".join(f"{k}={v:.4f}" if k != "psnr" else f"psnr={v:.3f}" for k, v in m.items())


def write_compare(path: Path, vae: torch.Tensor, dwd: torch.Tensor) -> None:
    """VAE | DWD side by side, 24 fps (a still is written as one PNG)."""
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    wide = torch.cat([vae, dwd], dim=-1)[0]
    frames = to_uint8(wide).permute(1, 2, 3, 0).cpu().numpy()
    if frames.shape[0] == 1:
        imageio.imwrite(path.with_suffix(".png"), frames[0])
    else:
        imageio.mimwrite(path, list(frames), fps=24, quality=8, macro_block_size=16)


# ------------------------------------------------------------------------------------ ridge

def ridge(args) -> None:
    """Closed-form least squares per token class on standardised features; ``lambda`` picked on the probe set."""
    dev = torch.device(args.device)
    train, held = samples(args.data, False), samples(args.data, True)
    if not train or not held:
        raise SystemExit(f"{args.data}: need train_*.pt and held_*.pt from collect")
    C = 1 if args.shared else CLASSES
    d = HIDDEN + TOKEN_DIM
    out = OUT if args.halo < 0 else ext_out(args.halo)             # --halo >= 0: overlapping output patches
    shape_head = DWDHead(classes=1, out=out, hidden=1, state_dim=1)  # only for its targets()
    # Pass 1: per-dimension mean and spread. Centring and scaling BEFORE the second moments are accumulated
    # matters: the last-layer dimensions have means far larger than their spread, and E[xx] - n m m^T then
    # loses the spread to cancellation.
    s1 = torch.zeros(d, dtype=torch.float64, device=dev)
    s2 = torch.zeros(d, dtype=torch.float64, device=dev)
    total = 0
    for path in train:
        s = load(path, dev, ("h", "s"))
        x = torch.cat([s["h"].float(), s["s"].float()], dim=1).double()
        s1 += x.sum(0)
        total += x.shape[0]
    mean = s1 / total
    for path in train:
        s = load(path, dev, ("h", "s"))
        x = torch.cat([s["h"].float(), s["s"].float()], dim=1).double() - mean
        s2 += (x * x).sum(0)
    std = (s2 / total).sqrt()
    # floor the spread: a dimension that barely varies would otherwise get a huge weight, and fine-tuning the
    # blocks underneath would then swing the pixels through it (seen: grad norm 1e3 and a 1e12 loss spike)
    scale = std.clamp_min(args.scale_floor * std.median().item())
    # Pass 2: per-class moments of the standardised features z = (x - mean) / scale.
    xtx = torch.zeros(C, d, d, dtype=torch.float64, device=dev)
    xty = torch.zeros(C, d, out, dtype=torch.float64, device=dev)
    sx = torch.zeros(C, d, dtype=torch.float64, device=dev)
    sy = torch.zeros(C, out, dtype=torch.float64, device=dev)
    n = torch.zeros(C, dtype=torch.float64, device=dev)
    for path in train:
        s = load(path, dev, ("h", "s", "y", "shape"))
        z = (torch.cat([s["h"].float(), s["s"].float()], dim=1).double() - mean) / scale
        y = shape_head.targets(from_uint8(s["y"]), *s["shape"])
        cls = token_classes(*s["shape"], device=dev) % C
        for c in range(C):
            zc, yc = z[cls == c], y[cls == c].double()
            for i in range(0, zc.shape[0], 2048):
                zb, yb = zc[i:i + 2048], yc[i:i + 2048]
                xtx[c] += zb.T @ zb
                xty[c] += zb.T @ yb
                sx[c] += zb.sum(0)
                sy[c] += yb.sum(0)
            n[c] += zc.shape[0]
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"train_samples": len(train), "train_tokens": int(total), "held_samples": len(held), "classes": C,
              "lambdas": {}}
    best = None
    for lam in args.lambdas:
        head = DWDHead(classes=C, out=out).to(dev)
        head.blend = args.blend
        head.mean.copy_(mean.float())
        head.scale.copy_(scale.float())
        for c in range(C):
            mc, my = sx[c] / n[c], sy[c] / n[c]                    # class means, standardised space
            cxx = xtx[c] - n[c] * torch.outer(mc, mc)
            cxy = xty[c] - n[c] * torch.outer(mc, my)
            w = torch.linalg.solve(cxx + lam * n[c] * torch.eye(d, dtype=torch.float64, device=dev), cxy)  # (d, OUT)
            b = my - mc @ w                                         # forward: z @ w + b
            head.weight.data[c].copy_(w.T.float())
            head.bias.data[c].copy_(b.float())
        result = evaluate(head, held, dev)
        report["lambdas"][str(lam)] = result["mean"]
        print(f"LOT_H3 kind=dwd_ridge classes={C} lambda={lam:g} {fmt(result['mean'])}", flush=True)
        if best is None or result["mean"]["psnr"] > best[0]:
            best = (result["mean"]["psnr"], lam, head)
    _p, lam, head = best
    torch.save(head_state(head), args.out / "dwd_head_ridge.pt")
    final = evaluate(head, held, dev, videos=args.out / "ridge_videos")
    report.update({"best_lambda": lam, "best": final["mean"], "rows": final["rows"]})
    (args.out / "ridge.json").write_text(json.dumps(report, indent=1))
    print(f"LOT_H3 kind=dwd_ridge_done classes={C} best_lambda={lam:g} {fmt(final['mean'])} out={args.out} ok=1",
          flush=True)


def refit(args) -> None:
    """Ridge refit of the head on features from the DiT WITH a fine-tuned LoRA on (light or full samples).

    One DiT pass over the training samples: the mean and spread come from the first ``--stat-samples``, the
    per-class moments of the standardised features from all of them (inputs in float64, the wide target sums
    in float32 so the DiT still fits beside them). Held-out features are kept on the CPU; the solve and the
    choice of lambda happen after the DiT is freed. Writes OUT/dwd_head_ridge.pt (overwritten).
    """
    from gpu_guard import refuse_if_busy

    refuse_if_busy("train_dwd_h3.py")
    from dir_h3 import generator_step, load_dir_run, use_networks
    from h3_musubi import load_dit, make_lora

    dev = torch.device("cuda")
    train = [p for d in args.data for p in samples(d, False)]
    held = [p for d in args.data for p in samples(d, True)]
    if "te" in torch.load(train[0], mmap=True, weights_only=True):
        train = [p for p in train if p.with_suffix(".mp4").exists()]
        held = [p for p in held if p.with_suffix(".mp4").exists()]
    if args.max_samples:
        train = random.Random(0).sample(train, min(args.max_samples, len(train)))
    dit = load_dit(device=dev, swap=args.swap)
    gen_head, net, _config = load_dir_run(args.run, dit)
    saved = torch.load(args.lora, map_location="cpu", weights_only=True)
    lora = make_lora(dit, saved["rank"])
    if lora.load_state_dict(saved["lora"], strict=False).unexpected_keys:
        raise SystemExit(f"{args.lora}: keys that are not in this LoRA")
    lora.to(dev, torch.float32)
    lora.requires_grad_(False)
    for module in lora.unet_loras:
        module.enabled = any(k.startswith(module.lora_name + ".") for k in saved["lora"])
    store: dict = {}
    dit.final_layer.video_out.register_forward_pre_hook(lambda _m, a: store.__setitem__("h", a[0][0]))

    @torch.no_grad()
    def tokens(d: dict) -> torch.Tensor:
        with use_networks(net, None, gen=True, fake=False):
            generator_step(dit, gen_head, d["state"], d["prev"], d["r"], d["text"], sigma=d["sigma"],
                           audio_generator=torch.Generator(dev).manual_seed(d["audio_seed"]))
        return store["h"].float()

    C = CLASSES
    d_in = HIDDEN + TOKEN_DIM
    out = OUT if args.halo < 0 else ext_out(args.halo)
    shape_head = DWDHead(classes=1, out=out, hidden=1, state_dim=1)
    t0 = time.time()
    s1 = torch.zeros(d_in, dtype=torch.float64, device=dev)
    s2 = torch.zeros(d_in, dtype=torch.float64, device=dev)
    count = 0
    stat = train[:args.stat_samples]
    cached: dict = {}
    for path in stat:                                            # pass 1 on a subset: mean and spread
        d = load(path, dev)
        x = torch.cat([tokens(d), d["s"].float()], dim=1).double()
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
        count += x.shape[0]
    mean = s1 / count
    std = (s2 / count - mean * mean).clamp_min(0).sqrt()
    scale = std.clamp_min(args.scale_floor * std.median().item())
    print(f"LOT_H3 kind=dwd_refit stats_samples={len(stat)} min={(time.time() - t0) / 60:.1f}", flush=True)
    xtx = torch.zeros(C, d_in, d_in, dtype=torch.float64, device=dev)
    xty = torch.zeros(C, d_in, out, dtype=torch.float32, device=dev)
    sx = torch.zeros(C, d_in, dtype=torch.float64, device=dev)
    sy = torch.zeros(C, out, dtype=torch.float64, device=dev)
    n = torch.zeros(C, dtype=torch.float64, device=dev)
    for i, path in enumerate(train):                             # pass 2: per-class moments
        d = load(path, dev)
        z = (torch.cat([tokens(d), d["s"].float()], dim=1).double() - mean) / scale
        y = shape_head.targets(from_uint8(d["y"]), *d["shape"])
        cls = token_classes(*d["shape"], device=dev)
        for c in range(C):
            zc, yc = z[cls == c], y[cls == c]
            for j in range(0, zc.shape[0], 1024):
                zb, yb = zc[j:j + 1024], yc[j:j + 1024]
                xtx[c] += zb.T @ zb
                xty[c] += zb.float().T @ yb.float()
                sx[c] += zb.sum(0)
                sy[c] += yb.double().sum(0)
            n[c] += zc.shape[0]
        if (i + 1) % 100 == 0:
            print(f"LOT_H3 kind=dwd_refit samples={i + 1}/{len(train)} min={(time.time() - t0) / 60:.1f} "
                  f"peak_gb={torch.cuda.max_memory_allocated(dev) / 2**30:.1f}", flush=True)
    for path in held:                                            # held-out features, kept on the CPU
        d = load(path, dev)
        cached[str(path)] = tokens(d).half().cpu()
    args.out.mkdir(parents=True, exist_ok=True)
    moments = args.out / "refit_moments.pt"
    tmp = moments.with_suffix(".tmp")
    torch.save({"xtx": xtx.cpu(), "xty": xty.cpu(), "sx": sx.cpu(), "sy": sy.cpu(), "n": n.cpu(), "mean": mean.cpu(),
                "scale": scale.cpu(), "held": cached, "out": out, "train_samples": len(train)}, tmp)
    tmp.replace(moments)                                          # the DiT pass is never lost to a failed solve
    print(f"LOT_H3 kind=dwd_refit_moments out={moments} min={(time.time() - t0) / 60:.1f}", flush=True)


def refit_solve(args) -> None:
    """Solve the head from ``refit_moments.pt`` in a process with no DiT on the card."""
    dev = torch.device("cuda")
    m = torch.load(args.out / "refit_moments.pt", map_location="cpu", weights_only=False)
    xtx, xty, sx, sy, n = m["xtx"].to(dev), m["xty"], m["sx"].to(dev), m["sy"].to(dev), m["n"].to(dev)
    mean, scale, cached, out = m["mean"].to(dev), m["scale"].to(dev), m["held"], m["out"]
    C, d_in = xtx.shape[0], xtx.shape[1]
    held = [Path(k) for k in cached]
    folders = {dd.name: [p for p in held if p.parent == dd] for dd in args.data}
    best, report = None, {"train_samples": m["train_samples"], "lora": str(args.lora), "lambdas": {}}
    for lam in args.lambdas:
        head = DWDHead(classes=C, out=out).to(dev)
        head.blend = args.blend
        head.mean.copy_(mean.float())
        head.scale.copy_(scale.float())
        for c in range(C):
            mc, my = sx[c] / n[c], sy[c] / n[c]
            cxx = xtx[c] - n[c] * torch.outer(mc, mc)
            cxy = xty[c].to(dev).double() - n[c] * torch.outer(mc, my)
            w = torch.linalg.solve(cxx + lam * n[c] * torch.eye(d_in, dtype=torch.float64, device=dev), cxy)
            head.weight.data[c].copy_(w.T.float())
            head.bias.data[c].copy_((my - mc @ w).float())
            del cxy, w
            torch.cuda.empty_cache()
        parts = {}
        for name, files in folders.items():
            rows = [evaluate(head, [path], dev, tokens_fn=lambda _d, p=path: cached[str(p)].to(dev))["mean"]
                    for path in files]
            parts[name] = summarise(rows)
        mean_psnr = sum(v["psnr"] for v in parts.values()) / len(parts)
        report["lambdas"][str(lam)] = parts
        print(f"LOT_H3 kind=dwd_refit lambda={lam:g} psnr={mean_psnr:.3f} "
              + " ".join(f"psnr@{k}={v['psnr']:.3f}" for k, v in parts.items()), flush=True)
        if best is None or mean_psnr > best[0]:
            best = (mean_psnr, lam, head)
        else:
            del head
        torch.cuda.empty_cache()
    _p, lam, head = best
    torch.save(head_state(head), args.out / "dwd_head_ridge.pt")
    report["best_lambda"] = lam
    (args.out / "refit.json").write_text(json.dumps(report, indent=1))
    print(f"LOT_H3 kind=dwd_refit_done best_lambda={lam:g} psnr={_p:.3f} out={args.out} ok=1", flush=True)


def head_state(head: DWDHead) -> dict:
    """The head's state dict plus its temporal blend weight, so a saved head decodes as it was evaluated."""
    return {**head.state_dict(), "blend": torch.tensor(float(head.blend))}


def load_head(path: Path, device) -> DWDHead:
    state = torch.load(path, map_location=device, weights_only=True)
    blend = state.pop("blend", None)
    head = DWDHead(classes=state["weight"].shape[0], out=state["weight"].shape[1]).to(device)
    head.load_state_dict(state)
    if blend is not None:
        head.blend = float(blend)
    return head


# -------------------------------------------------------------------------------------- sgd

def sgd(args) -> None:
    """Refine the ridge head with a per-token pseudo-Huber loss. No DiT."""
    if os.environ.get("LOT_H3_TRAIN") != "1":
        raise SystemExit("sgd is gated: set LOT_H3_TRAIN=1")
    dev = torch.device(args.device)
    train, held = samples(args.data, False), samples(args.data, True)
    init = args.init or args.out / "dwd_head_ridge.pt"
    head = load_head(init, dev)
    if head.halo is not None:
        raise SystemExit("sgd trains a plain head only (ridge without --halo)")
    opt = torch.optim.AdamW([head.weight, head.bias], lr=args.lr, weight_decay=0.0)
    rng = random.Random(0)
    base = evaluate(head, held, dev)["mean"]
    best, log = base["psnr"], [{"step": 0, **base}]
    print(f"LOT_H3 kind=dwd_eval step=0 {fmt(base)}", flush=True)
    step = skipped = 0
    while step < args.steps:
        d = load(rng.choice(train), dev, ("h", "s", "y", "shape"))
        cls = token_classes(*d["shape"], device=dev)
        y = from_uint8(d["y"])
        for _ in range(args.batches_per_sample):
            idx = torch.randint(0, y.shape[0], (args.batch,), device=dev)
            mask = slot_mask(cls[idx])
            pred = head(d["h"][idx], d["s"][idx], cls[idx])
            loss = pseudo_huber(pred * mask, y[idx] * mask)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                print(f"LOT_H3 kind=dwd_nonfinite step={step} loss={loss.item()} skipped={skipped}", flush=True)
                if skipped > args.max_skips:
                    raise SystemExit(f"{skipped} non-finite losses: stopping")
                continue
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_([head.weight, head.bias], args.grad_clip))
            if not math.isfinite(grad_norm):
                skipped += 1
                print(f"LOT_H3 kind=dwd_nonfinite step={step} grad_norm={grad_norm} skipped={skipped}", flush=True)
                if skipped > args.max_skips:
                    raise SystemExit(f"{skipped} non-finite gradients: stopping")
                continue
            opt.step()
            step += 1
            if step % args.log_every == 0:
                print(f"LOT_H3 kind=dwd_train step={step} loss={loss.item():.5f} grad_norm={grad_norm:.4f}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(head, held, dev)["mean"]
                log.append({"step": step, **m})
                print(f"LOT_H3 kind=dwd_eval step={step} {fmt(m)}", flush=True)
                if m["psnr"] > best:
                    best = m["psnr"]
                    torch.save(head_state(head), args.out / "dwd_head.pt")
            if step >= args.steps:
                break
    (args.out / "sgd.json").write_text(json.dumps({"init": str(init), "log": log, "best_psnr": best,
                                                   "ridge_psnr": base["psnr"]}, indent=1))
    print(f"LOT_H3 kind=dwd_sgd_done ridge_psnr={base['psnr']:.2f} best_psnr={best:.2f} out={args.out} ok=1", flush=True)


# --------------------------------------------------------------------------------------- ft

def ft(args) -> None:
    """A LoRA on the last K DiT blocks, switched on only for the final DiR iteration, trained with a pixel MSE
    through the DWD head (which trains jointly at a lower rate). Strength 0 (the LoRA off) is the original
    sampler: a disabled LoRA module returns its original forward."""
    if os.environ.get("LOT_H3_TRAIN") != "1":
        raise SystemExit("ft is gated: set LOT_H3_TRAIN=1")
    from gpu_guard import refuse_if_busy

    refuse_if_busy("train_dwd_h3.py")
    from dir_h3 import generator_step, load_dir_run, use_networks
    from h3_musubi import load_dit, make_lora

    dev = torch.device("cuda")
    train = [p for d in args.data for p in samples(d, False)]
    held = [p for d in args.data for p in samples(d, True)]
    missing = [p.name for p in train[:1] + held[:1] if "prev" not in torch.load(p, mmap=True, weights_only=True)]
    light = "te" in torch.load(train[0], mmap=True, weights_only=True)
    if light:                                                     # only samples whose target video is rendered
        train = [p for p in train if p.with_suffix(".mp4").exists()]
        held = [p for p in held if p.with_suffix(".mp4").exists()]
        if not train or not held:
            raise SystemExit("no rendered targets yet; run the `targets` phase")
    if missing:
        raise SystemExit(f"{missing}: no final-iteration inputs; run collect again to add them")
    dit = load_dit(device=dev, swap=0)
    gen_head, net, _config = load_dir_run(args.run, dit, tag=f"_{args.tag}" if args.tag else "", ema=False)
    lora = make_lora(dit, args.rank)
    lora.to(dev, torch.float32)
    lora.requires_grad_(False)
    dit.enable_gradient_checkpointing()
    dit.train()
    blocks = len(dit.blocks)
    trained = []
    for module in lora.unet_loras:
        block = int(re.search(r"blocks_(\d+)_", module.lora_name).group(1))
        module.enabled = block >= blocks - args.k
        if module.enabled:
            module.requires_grad_(True)
            trained.append(module)
    lora_params = [p for m in trained for p in m.parameters() if p.requires_grad]
    head = load_head(args.init or args.out / "dwd_head_ridge.pt", dev)
    head.blend = args.blend
    if args.resume is not None:                                    # continue a saved ft LoRA (same k and rank)
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["k"] != args.k or saved["rank"] != args.rank:
            raise SystemExit(f"{args.resume}: k={saved['k']} rank={saved['rank']}, not k={args.k} rank={args.rank}")
        if lora.load_state_dict(saved["lora"], strict=False).unexpected_keys:
            raise SystemExit(f"{args.resume}: keys that are not in this LoRA")
    groups = [{"params": lora_params, "lr": args.lr}]
    if args.head_lr > 0:
        groups.append({"params": [head.weight, head.bias], "lr": args.head_lr})
    else:
        head.requires_grad_(False)
    opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
    store: dict = {}
    hook = dit.final_layer.video_out.register_forward_pre_hook(lambda _m, a: store.__setitem__("h", a[0][0]))
    lpips_net = None
    if args.lpips_weight > 0 or args.eval_lpips:
        import lpips

        lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval().requires_grad_(False)
    if args.lpips_weight > 0 and args.head_lr > 0:
        raise SystemExit("--lpips-weight needs a frozen head (--head-lr 0)")
    held_by_folder = {d.name: [p for p in held if p.parent == d] for d in args.data}
    if args.select == "lpips" and lpips_net is None:
        raise SystemExit("--select lpips needs --eval-lpips")

    def score_of(m: dict) -> float:                             # higher is better
        return m["psnr"] if args.select == "psnr" else -m["lpips"]

    def set_lora(on: bool) -> None:
        for module in trained:
            module.enabled = on

    def tokens(d: dict) -> torch.Tensor:
        with use_networks(net, None, gen=True, fake=False):
            generator_step(dit, gen_head, d["state"], d["prev"], d["r"], d["text"], sigma=d["sigma"],
                           audio_generator=torch.Generator(dev).manual_seed(d["audio_seed"]))
        return store["h"]

    def probe(on: bool = True) -> dict:
        """Mean over held-out folders (each folder weighted equally), with each folder's PSNR and LPIPS too."""
        torch.cuda.empty_cache()
        set_lora(on)
        try:
            parts = {name: evaluate(head, files, dev, tokens_fn=tokens, lpips_net=lpips_net)["mean"]
                     for name, files in held_by_folder.items() if files}
        finally:
            set_lora(True)
        mean = {k: sum(m[k] for m in parts.values()) / len(parts) for k in next(iter(parts.values()))}
        for name, m in parts.items():
            mean[f"psnr@{name}"] = m["psnr"]
            if "lpips" in m:
                mean[f"lpips@{name}"] = m["lpips"]
        return mean

    keys = ("state", "prev", "text", "audio_seed", "sigma", "r", "s", "y", "shape", "h")
    with torch.no_grad():                                          # the replay must reproduce the stored tokens
        d0 = load(held[0], dev, keys if not light else None)
        set_lora(False)
        replay = tokens(d0).float()
        set_lora(True)
        rel = ((replay - d0["h"].float()).norm() / d0["h"].float().norm()).item() if "h" in d0 else 0.0
    print(f"LOT_H3 kind=dwd_ft trainable_modules={len(trained)} params={sum(p.numel() for p in lora_params)} "
          f"replay_rel_err={rel:.2e}", flush=True)
    if rel > 1e-2:
        raise SystemExit(f"replaying the final iteration gives tokens {rel:.2e} away from the stored ones")
    args.out.mkdir(parents=True, exist_ok=True)
    last_path = args.out / "dwd_ft_last.pt"
    rng = random.Random(0)
    step = skipped = 0
    ema = None
    elapsed = 0.0                                                 # training minutes done before this process
    if args.auto_resume and last_path.exists():                   # crash-safe: continue where the last run stopped
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        lora.load_state_dict(last["lora"], strict=False)
        opt.load_state_dict(last["opt"])
        if last.get("head") is not None:
            head.load_state_dict({k: v for k, v in last["head"].items() if k != "blend"})
        step, best, log, elapsed, ema = last["step"], last["best"], last["log"], last["elapsed"], last["ema"]
        rng.setstate(last["rng"])
        base = log[0]
        print(f"LOT_H3 kind=dwd_ft_resume step={step} elapsed_min={elapsed:.1f} best_{args.select}_score={best:.4f}", flush=True)
    else:
        base = probe()
        best, log = score_of(base), [{"step": 0, **base}]
        print(f"LOT_H3 kind=dwd_ft_eval step=0 {fmt(base)}", flush=True)
    t_start = time.time() - elapsed * 60
    next_eval = (math.floor(elapsed / args.eval_minutes) + 1) * args.eval_minutes * 60
    next_ckpt = time.time() + args.checkpoint_minutes * 60

    def checkpoint() -> None:
        state = {k: v for k, v in lora.state_dict().items() if any(k.startswith(m.lora_name + ".") for m in trained)}
        tmp = last_path.with_suffix(".tmp")
        torch.save({"lora": state, "opt": opt.state_dict(), "head": head_state(head) if args.head_lr > 0 else None,
                    "step": step, "best": best, "log": log, "elapsed": (time.time() - t_start) / 60, "ema": ema,
                    "rng": rng.getstate(), "k": args.k, "rank": args.rank}, tmp)
        tmp.replace(last_path)                                    # atomic: a crash mid-save keeps the old one
        print(f"LOT_H3 kind=dwd_ft_checkpoint step={step} out={last_path}", flush=True)

    while time.time() - t_start < args.minutes * 60:
        d = load(rng.choice(train), dev, keys)
        cls = token_classes(*d["shape"], device=dev)
        mask = head.target_mask(cls, d["shape"][1] * d["shape"][2])
        y = head.targets(from_uint8(d["y"]), *d["shape"])
        h = tokens(d)
        if args.head_lr > 0:
            loss = (((head(h, d["s"], cls) - y) ** 2) * mask).sum() / mask.sum()
            h_grad = None
        else:                                                      # frozen head: no (N, out) graph to hold
            value, h_grad = head.masked_mse_grad(h, d["s"], y, mask, cls)
            loss = torch.tensor(value)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            skipped += 1
            print(f"LOT_H3 kind=dwd_nonfinite step={step} loss={loss.item()} skipped={skipped}", flush=True)
            if skipped > args.max_skips:
                raise SystemExit(f"{skipped} non-finite losses: stopping")
            continue
        if ema is not None and step >= args.spike_warmup and loss.item() > args.spike_factor * ema:
            skipped += 1
            print(f"LOT_H3 kind=dwd_spike step={step} loss={loss.item():.4g} ema={ema:.4g} skipped={skipped}", flush=True)
            if skipped > args.max_skips:
                raise SystemExit(f"{skipped} skipped steps (spikes or non-finite): stopping")
            continue
        lp = None
        if args.lpips_weight > 0:
            lp, rows, g = lpips_h_grad(head, h.detach(), d["s"], d["y"], d["shape"], lpips_net, rng, args.lpips_frames)
            h_grad = h_grad.clone()
            h_grad[rows] += args.lpips_weight * g.to(h_grad.dtype)
        if h_grad is None:
            loss.backward()
        else:
            h.backward(h_grad.to(h.dtype))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip))
        if not math.isfinite(grad_norm):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            print(f"LOT_H3 kind=dwd_nonfinite step={step} grad_norm={grad_norm} skipped={skipped}", flush=True)
            if skipped > args.max_skips:
                raise SystemExit(f"{skipped} non-finite gradients: stopping")
            continue
        opt.step()
        step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % args.log_every == 0:
            print(f"LOT_H3 kind=dwd_ft_train step={step} loss={loss.item():.5f} ema={ema:.5f} grad_norm={grad_norm:.4f} "
                  + (f"lpips={lp:.4f} " if lp is not None else "") +
                  f"min={(time.time() - t_start) / 60:.1f} peak_gb={torch.cuda.max_memory_allocated(dev) / 2**30:.1f}",
                  flush=True)
        if time.time() - t_start > next_eval:
            next_eval += args.eval_minutes * 60
            if light:                                             # targets still rendering: pick up new ones
                fresh = [p for d in args.data for p in samples(d, False) if p.with_suffix(".mp4").exists()]
                if len(fresh) > len(train):
                    print(f"LOT_H3 kind=dwd_ft_data step={step} train={len(fresh)} (was {len(train)})", flush=True)
                    train = fresh
            with torch.no_grad():
                m = probe()
            log.append({"step": step, **m})
            print(f"LOT_H3 kind=dwd_ft_eval step={step} {fmt(m)}", flush=True)
            if (args.gate_minutes and (time.time() - t_start) / 60 >= args.gate_minutes
                    and not args.gate_min <= m.get(args.gate_key, math.nan) <= args.gate_max):
                checkpoint()
                print(f"LOT_H3 kind=dwd_ft_gate_failed step={step} {args.gate_key}={m.get(args.gate_key)} "
                      f"needed={args.gate_min}..{args.gate_max} out={args.out}", flush=True)
                raise SystemExit(3)
            if score_of(m) > best:
                best = score_of(m)
                save_ft(args.out, lora, trained, head, args, step)
                if args.best_videos > 0:                          # something to watch at every new best
                    with torch.no_grad():
                        evaluate(head, held[:args.best_videos], dev, tokens_fn=tokens,
                                 videos=args.out / "best_videos", limit_videos=args.best_videos)
                    print(f"LOT_H3 kind=dwd_ft_videos step={step} out={args.out / 'best_videos'}", flush=True)
        if time.time() > next_ckpt:
            next_ckpt = time.time() + args.checkpoint_minutes * 60
            checkpoint()
    checkpoint()
    with torch.no_grad():
        final, off = probe(), probe(on=False)
    log.append({"step": step, **final})
    if score_of(final) > best:
        best = score_of(final)
        save_ft(args.out, lora, trained, head, args, step)
    hook.remove()
    (args.out / "ft.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "log": log,
                                                  "best_psnr": best, "start_psnr": base["psnr"],
                                                  "lora_off": off}, indent=1))
    print(f"LOT_H3 kind=dwd_ft_done steps={step} start_psnr={base["psnr"]:.3f} best_{args.select}_score={best:.4f} "
          f"lora_off_psnr={off['psnr']:.3f} out={args.out} ok=1", flush=True)


def lpips_h_grad(head: DWDHead, h: torch.Tensor, s: torch.Tensor, y: torch.Tensor, shape, net,
                 rng: random.Random, frames: int = 1) -> tuple[float, torch.Tensor, torch.Tensor]:
    """LPIPS of one random latent frame's decoded frames and its gradient with respect to ``h``.

    Autograd runs through the frozen head only for that latent frame and its two neighbours (whose overlaps
    blend into it); every other row is a constant. Returns ``(loss, rows, grad of those rows)``.
    """
    from dwd_h3 import blend_tokens, frames_of

    t, th, tw = shape
    per = th * tw
    k = rng.randrange(t)
    lo, hi = max(0, k - 1), min(t, k + 2)
    rows = torch.arange(lo * per, hi * per, device=h.device)
    cls = token_classes(t, th, tw, h.device)
    with torch.no_grad():
        full = head(h, s, cls)
    hr = h[rows].detach().float().requires_grad_(True)
    full = full.index_copy(0, rows, head(hr, s[rows], cls[rows]))
    if head.halo is not None:
        pixels = blend_tokens(full, t, th, tw, head.halo, head.blend)
    else:
        pixels = tokens_to_pixels(full, t, th, tw)
    start = sum(frames_of(t)[:k])
    n = frames_of(t)[k]
    pick = sorted(rng.sample(range(start, start + n), min(frames, n)))   # VGG at 768 px: a frame or two fits
    pred = pixels[:, :, pick].clamp(-1, 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
    with torch.no_grad():
        target = tokens_to_pixels(from_uint8(y), t, th, tw)[:, :, pick].permute(0, 2, 1, 3, 4).flatten(0, 1)
    loss = net(pred, target).mean()
    (grad,) = torch.autograd.grad(loss, hr)
    return loss.item(), rows, grad


def save_ft(out: Path, lora, trained, head: DWDHead, args, step: int) -> None:
    state = {k: v for k, v in lora.state_dict().items()
             if any(k.startswith(m.lora_name + ".") for m in trained)}
    torch.save({"lora": state, "k": args.k, "rank": args.rank, "step": step}, out / "dwd_ft_lora.pt")
    if args.head_lr > 0:                                          # a frozen head is its init file; no 1-3 GB copy
        torch.save(head_state(head), out / "dwd_ft_head.pt")
    else:
        (out / "dwd_ft_head.json").write_text(json.dumps({"head": str(args.init or out / "dwd_head_ridge.pt"),
                                                          "blend": head.blend}))
    print(f"LOT_H3 kind=dwd_ft_save step={step} out={out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="phase", required=True)

    c = sub.add_parser("collect", help="GPU: DiR rollouts, last-layer tokens, VAE targets, final-iteration inputs")
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--tag", default="")
    c.add_argument("--ema", action="store_true")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--cache", type=Path, default=None)
    c.add_argument("--holdout", type=int, default=8)
    c.add_argument("--items", type=int, default=8, help="training prompts")
    c.add_argument("--seeds", type=int, default=2, help="seeds per training prompt")
    c.add_argument("--held-items", type=int, default=4)
    c.add_argument("--held-seeds", type=int, default=1)
    c.add_argument("--frames", type=int, default=12)
    c.add_argument("--max-latent-hw", type=int, nargs=2, default=[32, 48], metavar=("H", "W"))
    c.add_argument("--swap", type=int, default=4)
    c.add_argument("--vae-device", default="cuda:1" if torch.cuda.device_count() > 1 else "cuda")
    c.add_argument("--light", action="store_true",
                   help="store only the final iteration's inputs and latent (~4 MB); `targets` renders the VAE mp4s")

    r = sub.add_parser("ridge", help="least squares on the collected tokens")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--lambdas", type=float, nargs="+", default=[1e-8, 1e-6, 1e-4, 1e-3])
    r.add_argument("--shared", action="store_true", help="one map for every token class")
    r.add_argument("--halo", type=int, default=-1,
                   help="overlapping output patches: halo pixels around each token, plus the frame before and "
                        "after (-1: plain per-token patches; 4 removes the seams)")
    r.add_argument("--blend", type=float, default=0.25, help="temporal cross-fade weight of an extended head")
    r.add_argument("--scale-floor", type=float, default=1.0,
                   help="floor on each feature's spread, as a fraction of the median spread")
    r.add_argument("--device", default="cuda")

    s = sub.add_parser("sgd", help="refine the head (pseudo-Huber)")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--init", type=Path, default=None, help="default: OUT/dwd_head_ridge.pt")
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--lr", type=float, default=1e-5)
    s.add_argument("--batch", type=int, default=2048, help="tokens per step")
    s.add_argument("--batches-per-sample", type=int, default=8)
    s.add_argument("--grad-clip", type=float, default=1.0)
    s.add_argument("--max-skips", type=int, default=20, help="stop after this many non-finite losses or gradients")
    s.add_argument("--log-every", type=int, default=50)
    s.add_argument("--eval-every", type=int, default=250)
    s.add_argument("--device", default="cuda")

    q = sub.add_parser("refit", help="ridge refit of the head on features with a fine-tuned LoRA on")
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--data", type=Path, nargs="+", required=True)
    q.add_argument("--lora", type=Path, required=True, help="a dwd_ft_lora.pt")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--halo", type=int, default=4)
    q.add_argument("--blend", type=float, default=0.25)
    q.add_argument("--lambdas", type=float, nargs="+", default=[1e-6, 1e-4])
    q.add_argument("--scale-floor", type=float, default=1.0)
    q.add_argument("--stat-samples", type=int, default=256)
    q.add_argument("--max-samples", type=int, default=0, help="0: every training sample")
    q.add_argument("--swap", type=int, default=0)
    q.add_argument("--solve-only", action="store_true", help="solve from OUT/refit_moments.pt (no DiT)")

    t = sub.add_parser("targets", help="VAE-decode light samples' latents to mp4 targets (follows a running collect)")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else "cuda")
    t.add_argument("--out", type=Path, default=Path("targets"), help=argparse.SUPPRESS)

    f = sub.add_parser("ft", help="final-iteration LoRA on the last K blocks, pixel loss through the head")
    f.add_argument("--run", type=Path, required=True)
    f.add_argument("--tag", default="")
    f.add_argument("--data", type=Path, nargs="+", required=True, help="one or more collect folders")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--init", type=Path, default=None, help="head to start from (default: OUT/dwd_head_ridge.pt)")
    f.add_argument("--resume", type=Path, default=None, help="a dwd_ft_lora.pt to continue from")
    f.add_argument("--blend", type=float, default=0.25, help="temporal cross-fade weight of an extended head")
    f.add_argument("--k", type=int, default=4, help="last K DiT blocks get the LoRA")
    f.add_argument("--rank", type=int, default=32)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--head-lr", type=float, default=1e-5)
    f.add_argument("--minutes", type=float, default=90, help="total training minutes, counted across resumes")
    f.add_argument("--checkpoint-minutes", type=float, default=30, help="save dwd_ft_last.pt (LoRA, optimiser, step)")
    f.add_argument("--auto-resume", action="store_true", help="continue from OUT/dwd_ft_last.pt when it exists")
    f.add_argument("--best-videos", type=int, default=2, help="held-out clips rendered at each new best (0: none)")
    f.add_argument("--lpips-weight", type=float, default=0.0,
                   help="add LPIPS (VGG) on one random latent frame per step; needs a frozen head")
    f.add_argument("--lpips-frames", type=int, default=1, help="frames of the chosen latent frame put through LPIPS")
    f.add_argument("--eval-lpips", action="store_true", help="report held-out LPIPS at every eval")
    f.add_argument("--gate-minutes", type=float, default=0, help="at the first eval after this, stop (exit 3) unless")
    f.add_argument("--gate-key", default="psnr", help="... this eval value (e.g. psnr@collect_w1_iso) ...")
    f.add_argument("--gate-min", type=float, default=-math.inf, help="... is at least this")
    f.add_argument("--gate-max", type=float, default=math.inf, help="... and at most this (e.g. for lpips)")
    f.add_argument("--select", choices=["psnr", "lpips"], default="psnr",
                   help="what a new best means: higher PSNR, or lower LPIPS")
    f.add_argument("--eval-minutes", type=float, default=15)
    f.add_argument("--grad-clip", type=float, default=1.0)
    f.add_argument("--max-skips", type=int, default=20)
    f.add_argument("--spike-factor", type=float, default=20.0, help="skip a step whose loss exceeds this times the running mean")
    f.add_argument("--spike-warmup", type=int, default=10)
    f.add_argument("--log-every", type=int, default=20)

    args = parser.parse_args()
    try:
        from setproctitle import setproctitle
        tag = {"collect": lambda: f"items={args.items}x{args.seeds}", "ridge": lambda: "", "targets": lambda: "", "refit": lambda: "",
               "sgd": lambda: f"lr={args.lr:g}", "ft": lambda: f"k={args.k} r={args.rank} lr={args.lr:g}"}[args.phase]()
        setproctitle(" ".join(x for x in (f"dwd:{args.phase}", tag, args.out.name) if x))
    except ImportError:
        pass
    {"collect": collect, "ridge": ridge, "sgd": sgd, "ft": ft, "targets": targets,
     "refit": lambda a: (refit_solve if a.solve_only else refit)(a)}[args.phase](args)


if __name__ == "__main__":
    main()
