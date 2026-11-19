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

from dwd_h3 import (CLASSES, HIDDEN, OUT, TOKEN_DIM, DWDHead, capture_last_layer, flicker_split,  # noqa: E402
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
    keys = keys or d.keys()
    return {k: (d[k].to(device) if torch.is_tensor(d[k]) else d[k]) for k in keys if k in d}


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
             tokens_fn=None) -> dict:
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
        rows.append({"file": path.name, **score(pred, target, t), "head_ms": 1000 * (time.time() - t0)})
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
    xty = torch.zeros(C, d, OUT, dtype=torch.float64, device=dev)
    sx = torch.zeros(C, d, dtype=torch.float64, device=dev)
    sy = torch.zeros(C, OUT, dtype=torch.float64, device=dev)
    n = torch.zeros(C, dtype=torch.float64, device=dev)
    for path in train:
        s = load(path, dev, ("h", "s", "y", "shape"))
        z = (torch.cat([s["h"].float(), s["s"].float()], dim=1).double() - mean) / scale
        y = from_uint8(s["y"])
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
        head = DWDHead(classes=C).to(dev)
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
    torch.save(head.state_dict(), args.out / "dwd_head_ridge.pt")
    final = evaluate(head, held, dev, videos=args.out / "ridge_videos")
    report.update({"best_lambda": lam, "best": final["mean"], "rows": final["rows"]})
    (args.out / "ridge.json").write_text(json.dumps(report, indent=1))
    print(f"LOT_H3 kind=dwd_ridge_done classes={C} best_lambda={lam:g} {fmt(final['mean'])} out={args.out} ok=1",
          flush=True)


def load_head(path: Path, device) -> DWDHead:
    state = torch.load(path, map_location=device, weights_only=True)
    head = DWDHead(classes=state["weight"].shape[0]).to(device)
    head.load_state_dict(state)
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
                    torch.save(head.state_dict(), args.out / "dwd_head.pt")
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
    train, held = samples(args.data, False), samples(args.data, True)
    missing = [p.name for p in train[:1] + held[:1] if "prev" not in torch.load(p, mmap=True, weights_only=True)]
    if missing:
        raise SystemExit(f"{missing}: no final-iteration inputs; run collect on {args.data} again to add them")
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
    groups = [{"params": lora_params, "lr": args.lr}]
    if args.head_lr > 0:
        groups.append({"params": [head.weight, head.bias], "lr": args.head_lr})
    else:
        head.requires_grad_(False)
    opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
    store: dict = {}
    hook = dit.final_layer.video_out.register_forward_pre_hook(lambda _m, a: store.__setitem__("h", a[0][0]))

    def set_lora(on: bool) -> None:
        for module in trained:
            module.enabled = on

    def tokens(d: dict) -> torch.Tensor:
        with use_networks(net, None, gen=True, fake=False):
            generator_step(dit, gen_head, d["state"], d["prev"], d["r"], d["text"], sigma=d["sigma"],
                           audio_generator=torch.Generator(dev).manual_seed(d["audio_seed"]))
        return store["h"]

    def probe(on: bool = True) -> dict:
        set_lora(on)
        try:
            return evaluate(head, held, dev, tokens_fn=tokens)["mean"]
        finally:
            set_lora(True)

    keys = ("state", "prev", "text", "audio_seed", "sigma", "r", "s", "y", "shape", "h")
    with torch.no_grad():                                          # the replay must reproduce the stored tokens
        d0 = load(held[0], dev, keys)
        set_lora(False)
        replay = tokens(d0).float()
        set_lora(True)
        rel = ((replay - d0["h"].float()).norm() / d0["h"].float().norm()).item()
    print(f"LOT_H3 kind=dwd_ft trainable_modules={len(trained)} params={sum(p.numel() for p in lora_params)} "
          f"replay_rel_err={rel:.2e}", flush=True)
    if rel > 1e-2:
        raise SystemExit(f"replaying the final iteration gives tokens {rel:.2e} away from the stored ones")
    args.out.mkdir(parents=True, exist_ok=True)
    base = probe()
    best, log = base["psnr"], [{"step": 0, **base}]
    print(f"LOT_H3 kind=dwd_ft_eval step=0 {fmt(base)}", flush=True)
    rng = random.Random(0)
    step = skipped = 0
    t_start = time.time()
    next_eval = args.eval_minutes * 60
    ema = None
    while time.time() - t_start < args.minutes * 60:
        d = load(rng.choice(train), dev, keys)
        cls = token_classes(*d["shape"], device=dev)
        mask = slot_mask(cls)
        y = from_uint8(d["y"])
        pred = head(tokens(d), d["s"], cls)
        loss = (((pred - y) ** 2) * mask).sum() / mask.sum()
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
        loss.backward()
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
                  f"min={(time.time() - t_start) / 60:.1f} peak_gb={torch.cuda.max_memory_allocated(dev) / 2**30:.1f}",
                  flush=True)
        if time.time() - t_start > next_eval:
            next_eval += args.eval_minutes * 60
            with torch.no_grad():
                m = probe()
            log.append({"step": step, **m})
            print(f"LOT_H3 kind=dwd_ft_eval step={step} {fmt(m)}", flush=True)
            if m["psnr"] > best:
                best = m["psnr"]
                save_ft(args.out, lora, trained, head, args, step)
    with torch.no_grad():
        final, off = probe(), probe(on=False)
    log.append({"step": step, **final})
    if final["psnr"] > best:
        best = final["psnr"]
        save_ft(args.out, lora, trained, head, args, step)
    hook.remove()
    (args.out / "ft.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "log": log,
                                                  "best_psnr": best, "start_psnr": base["psnr"],
                                                  "lora_off": off}, indent=1))
    print(f"LOT_H3 kind=dwd_ft_done steps={step} start_psnr={base['psnr']:.3f} best_psnr={best:.3f} "
          f"lora_off_psnr={off['psnr']:.3f} out={args.out} ok=1", flush=True)


def save_ft(out: Path, lora, trained, head: DWDHead, args, step: int) -> None:
    state = {k: v for k, v in lora.state_dict().items()
             if any(k.startswith(m.lora_name + ".") for m in trained)}
    torch.save({"lora": state, "k": args.k, "rank": args.rank, "step": step}, out / "dwd_ft_lora.pt")
    torch.save(head.state_dict(), out / "dwd_ft_head.pt")
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

    r = sub.add_parser("ridge", help="least squares on the collected tokens")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--lambdas", type=float, nargs="+", default=[1e-8, 1e-6, 1e-4, 1e-3])
    r.add_argument("--shared", action="store_true", help="one map for every token class")
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

    f = sub.add_parser("ft", help="final-iteration LoRA on the last K blocks, pixel loss through the head")
    f.add_argument("--run", type=Path, required=True)
    f.add_argument("--tag", default="")
    f.add_argument("--data", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--init", type=Path, default=None, help="head to start from (default: OUT/dwd_head_ridge.pt)")
    f.add_argument("--k", type=int, default=4, help="last K DiT blocks get the LoRA")
    f.add_argument("--rank", type=int, default=32)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--head-lr", type=float, default=1e-5)
    f.add_argument("--minutes", type=float, default=90)
    f.add_argument("--eval-minutes", type=float, default=15)
    f.add_argument("--grad-clip", type=float, default=1.0)
    f.add_argument("--max-skips", type=int, default=20)
    f.add_argument("--spike-factor", type=float, default=20.0, help="skip a step whose loss exceeds this times the running mean")
    f.add_argument("--spike-warmup", type=int, default=10)
    f.add_argument("--log-every", type=int, default=20)

    args = parser.parse_args()
    try:
        from setproctitle import setproctitle
        tag = {"collect": lambda: f"items={args.items}x{args.seeds}", "ridge": lambda: "",
               "sgd": lambda: f"lr={args.lr:g}", "ft": lambda: f"k={args.k} r={args.rank} lr={args.lr:g}"}[args.phase]()
        setproctitle(" ".join(x for x in (f"dwd:{args.phase}", tag, args.out.name) if x))
    except ImportError:
        pass
    {"collect": collect, "ridge": ridge, "sgd": sgd, "ft": ft}[args.phase](args)


if __name__ == "__main__":
    main()
