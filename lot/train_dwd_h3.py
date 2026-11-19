#!/usr/bin/env python3
"""Train a DWD pixel head (``dwd_h3.py``) on a DiR generator's final iteration, in three phases.

    # 1. GPU: run DiR, catch the last-layer tokens of the final iteration, decode x0_hat with the VAE
    python3 lot/train_dwd_h3.py collect --run lot/runs/dir_turbo_clip --ema --out DATA --items 8 --seeds 2

    # 2. least squares (the paper's init, and the linear-probe measurement); no DiT
    python3 lot/train_dwd_h3.py ridge --data DATA --out RUN

    # 3. refine the head from the ridge solution with a pseudo-Huber pixel loss; no DiT
    LOT_H3_TRAIN=1 python3 lot/train_dwd_h3.py sgd --data DATA --out RUN --steps 2000

The target is the VAE decode of the generator's own final ``x0_hat``: the head learns to stand in for the
decoder (data-free, like the DiR trainer). That can match the VAE more cheaply but cannot beat it; beating
it, as the paper does, needs real pixels and fine-tuned last blocks, which are not here yet.

Evaluation uses a fixed probe set: the held-out prompts with fixed seeds, scored every time by PSNR against
the VAE decode and by flicker (mean frame-to-frame change) against the VAE clip's, not by the training loss.

``collect`` writes one ``.pt`` per sample: ``h`` (fp16 last-layer tokens), ``s`` (fp16 packed input
state), ``y`` (uint8 per-token pixel targets), ``latent`` (fp16 final x0_hat), ``shape``, ``held``,
``vae_ms``. A 12 x 32 x 48 latent (39 frames, 512 x 768) is about 110 MB.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dwd_h3 import (HIDDEN, OUT, TOKEN_DIM, DWDHead, capture_last_layer, flicker, pack_rows,  # noqa: E402
                    pixel_frame_count, pixels_to_tokens, pseudo_huber, psnr, tokens_to_pixels)


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
    vae = load_vae(LOT_H3_VAE, device=vae_dev)
    vae_dtype = next(vae.parameters()).dtype
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
            if (args.out / name).exists():
                continue
            base = 0 if held else 100000
            x0, text = load_item(item, random.Random(1000 + base + index), dev, args.frames)
            z = torch.randn(x0.shape, device=dev,
                            generator=torch.Generator(dev).manual_seed(5000 + base + 100 * index + seed)).to(x0.dtype)
            audio_seed = 31 * (base + index) + seed
            estimate, state = torch.zeros_like(z), z
            t0 = time.time()
            with use_networks(net, None, gen=True, fake=False):
                for r in range(R):
                    last = r == R - 1
                    ctx = capture_last_layer(dit, store) if last else _null()
                    with ctx:
                        if last:
                            final_state = state
                        estimate = generator_step(dit, head, state, estimate, r, text, sigma=sigmas[r],
                                                  audio_generator=torch.Generator(dev).manual_seed(audio_seed)).float()
                    if r + 1 < R:
                        state = next_state(state, estimate, sigmas[r], sigmas[r + 1], z if mix else None)
            dit_s = time.time() - t0
            h = store["h"]
            latent_t, lat_h, lat_w = estimate.shape[2:]
            if h.shape != (latent_t * (lat_h // 2) * (lat_w // 2), HIDDEN):
                raise RuntimeError(f"captured tokens {tuple(h.shape)} do not match the latent {tuple(estimate.shape)}")
            sync(vae_dev)
            t0 = time.time()
            pixels = vae.decode(estimate.to(vae_dev, vae_dtype))
            sync(vae_dev)
            vae_ms = 1000 * (time.time() - t0)
            if pixels.shape[2] != pixel_frame_count(latent_t):
                raise RuntimeError(f"VAE gave {pixels.shape[2]} frames for {latent_t} latent frames")
            y = to_uint8(pixels_to_tokens(pixels.float().cpu(), latent_t))
            torch.save({
                "h": h.half().cpu(), "s": pack_rows(final_state.float()).half().cpu(), "y": y,
                "latent": estimate.half().cpu(), "shape": (latent_t, lat_h // 2, lat_w // 2), "held": held,
                "item": index, "seed": seed, "vae_ms": vae_ms, "dit_s": dit_s,
            }, args.out / name)
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


def load(path: Path, device) -> dict:
    return torch.load(path, map_location=device, weights_only=True)


@torch.no_grad()
def evaluate(head: DWDHead, files: list[Path], device, *, videos: Path | None = None, limit_videos: int = 2) -> dict:
    """PSNR against the VAE decode, flicker against the VAE clip's, and head decode time, per probe sample."""
    rows = []
    for k, path in enumerate(files):
        d = load(path, device)
        t, th, tw = d["shape"]
        target = tokens_to_pixels(from_uint8(d["y"]), t, th, tw)
        sync(device)
        t0 = time.time()
        pred = head.decode(d["h"], d["s"], t, th, tw)
        sync(device)
        rows.append({"file": path.name, "psnr": psnr(pred, target), "flicker": flicker(pred),
                     "flicker_vae": flicker(target), "head_ms": 1000 * (time.time() - t0), "vae_ms": d["vae_ms"]})
        if videos is not None and k < limit_videos:
            write_compare(videos / f"{path.stem}.mp4", target, pred)
    mean = {key: sum(r[key] for r in rows) / len(rows) for key in ("psnr", "flicker", "flicker_vae", "head_ms", "vae_ms")}
    return {"mean": mean, "rows": rows}


def write_compare(path: Path, vae: torch.Tensor, dwd: torch.Tensor) -> None:
    """VAE | DWD side by side, 24 fps (a still is written as one PNG)."""
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    wide = torch.cat([vae, dwd], dim=-1)[0]                      # (3, F, H, 2W)
    frames = to_uint8(wide).permute(1, 2, 3, 0).cpu().numpy()
    if frames.shape[0] == 1:
        imageio.imwrite(path.with_suffix(".png"), frames[0])
    else:
        imageio.mimwrite(path, list(frames), fps=24, quality=8, macro_block_size=16)


# ------------------------------------------------------------------------------------ ridge

def ridge(args) -> None:
    """Closed-form least squares on centred features, ``lambda`` picked on the held-out probe set."""
    dev = torch.device(args.device)
    train, held = samples(args.data, False), samples(args.data, True)
    if not train or not held:
        raise SystemExit(f"{args.data}: need train_*.pt and held_*.pt from collect")
    d = HIDDEN + TOKEN_DIM
    xtx = torch.zeros(d, d, dtype=torch.float64, device=dev)
    xty = torch.zeros(d, OUT, dtype=torch.float64, device=dev)
    sx = torch.zeros(d, dtype=torch.float64, device=dev)
    sy = torch.zeros(OUT, dtype=torch.float64, device=dev)
    n = 0
    for path in train:
        s = load(path, dev)
        x = torch.cat([s["h"].float(), s["s"].float()], dim=1)
        y = from_uint8(s["y"])
        for i in range(0, x.shape[0], 2048):
            xb, yb = x[i:i + 2048].double(), y[i:i + 2048].double()
            xtx += xb.T @ xb
            xty += xb.T @ yb
            sx += xb.sum(0)
            sy += yb.sum(0)
        n += x.shape[0]
    mx, my = sx / n, sy / n
    cxx = xtx - n * torch.outer(mx, mx)
    cxy = xty - n * torch.outer(mx, my)
    scale = cxx.diagonal().mean()
    best = None
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"train_samples": len(train), "train_tokens": n, "held_samples": len(held), "lambdas": {}}
    for lam in args.lambdas:
        a = cxx + lam * scale * torch.eye(d, dtype=torch.float64, device=dev)
        w = torch.linalg.solve(a, cxy)                               # (d, OUT)
        head = DWDHead().to(dev)
        head.mean.copy_(mx.float())
        head.linear.weight.data.copy_(w.T.float())
        head.linear.bias.data.copy_(my.float())
        result = evaluate(head, held, dev)
        report["lambdas"][str(lam)] = result["mean"]
        print(f"LOT_H3 kind=dwd_ridge lambda={lam:g} " + " ".join(f"{k}={v:.3f}" for k, v in result["mean"].items()),
              flush=True)
        if best is None or result["mean"]["psnr"] > best[0]:
            best = (result["mean"]["psnr"], lam, head)
    _psnr, lam, head = best
    torch.save(head.state_dict(), args.out / "dwd_head_ridge.pt")
    final = evaluate(head, held, dev, videos=args.out / "ridge_videos")
    report.update({"best_lambda": lam, "best": final["mean"], "rows": final["rows"]})
    (args.out / "ridge.json").write_text(json.dumps(report, indent=1))
    print(f"LOT_H3 kind=dwd_ridge_done best_lambda={lam:g} psnr={final['mean']['psnr']:.2f} "
          f"flicker={final['mean']['flicker']:.4f} flicker_vae={final['mean']['flicker_vae']:.4f} "
          f"head_ms={final['mean']['head_ms']:.1f} vae_ms={final['mean']['vae_ms']:.0f} out={args.out} ok=1", flush=True)


# -------------------------------------------------------------------------------------- sgd

def sgd(args) -> None:
    """Refine the ridge head with a per-token pseudo-Huber loss (and LPIPS on whole frames when installed)."""
    if os.environ.get("LOT_H3_TRAIN") != "1":
        raise SystemExit("sgd is gated: set LOT_H3_TRAIN=1")
    dev = torch.device(args.device)
    train, held = samples(args.data, False), samples(args.data, True)
    head = DWDHead().to(dev)
    init = args.init or args.out / "dwd_head_ridge.pt"
    head.load_state_dict(torch.load(init, map_location=dev, weights_only=True))
    lpips_net = None
    if args.lpips_weight > 0:
        try:
            import lpips
        except ImportError as exc:
            raise SystemExit("--lpips-weight needs the lpips package (pip install lpips)") from exc
        lpips_net = lpips.LPIPS(net="vgg").to(dev).eval().requires_grad_(False)
    opt = torch.optim.AdamW(head.linear.parameters(), lr=args.lr, weight_decay=0.0)
    rng = random.Random(0)
    base = evaluate(head, held, dev)["mean"]
    best = base["psnr"]
    log = [{"step": 0, **base}]
    print(f"LOT_H3 kind=dwd_eval step=0 " + " ".join(f"{k}={v:.3f}" for k, v in base.items()), flush=True)
    step = skipped = 0
    while step < args.steps:
        d = load(rng.choice(train), dev)
        t, th, tw = d["shape"]
        y = from_uint8(d["y"])
        for _ in range(args.batches_per_sample):
            idx = torch.randint(0, y.shape[0], (args.batch,), device=dev)
            loss = pseudo_huber(head(d["h"][idx], d["s"][idx]), y[idx])
            if lpips_net is not None:
                frame = rng.randrange(pixel_frame_count(t))
                pred = tokens_to_pixels(head(d["h"], d["s"]), t, th, tw)[:, :, frame]
                target = tokens_to_pixels(y, t, th, tw)[:, :, frame]
                loss = loss + args.lpips_weight * lpips_net(pred, target).mean()
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                print(f"LOT_H3 kind=dwd_nonfinite step={step} loss={loss.item()} skipped={skipped}", flush=True)
                if skipped > args.max_skips:
                    raise SystemExit(f"{skipped} non-finite losses: stopping")
                continue
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.linear.parameters(), args.grad_clip))
            if not torch.isfinite(torch.tensor(grad_norm)):
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
                print(f"LOT_H3 kind=dwd_eval step={step} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
                if m["psnr"] > best:
                    best = m["psnr"]
                    torch.save(head.state_dict(), args.out / "dwd_head.pt")
            if step >= args.steps:
                break
    (args.out / "sgd.json").write_text(json.dumps({"init": str(init), "log": log, "best_psnr": best,
                                                   "ridge_psnr": base["psnr"]}, indent=1))
    if not (args.out / "dwd_head.pt").exists():
        print("LOT_H3 kind=dwd_sgd note=no step beat the ridge head on the probe set; dwd_head.pt not written", flush=True)
    else:
        head.load_state_dict(torch.load(args.out / "dwd_head.pt", map_location=dev, weights_only=True))
        evaluate(head, held, dev, videos=args.out / "sgd_videos")
    print(f"LOT_H3 kind=dwd_sgd_done ridge_psnr={base['psnr']:.2f} best_psnr={best:.2f} out={args.out} ok=1", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="phase", required=True)

    c = sub.add_parser("collect", help="GPU: DiR rollouts, last-layer tokens and VAE targets")
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
    r.add_argument("--lambdas", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    r.add_argument("--device", default="cuda")

    s = sub.add_parser("sgd", help="refine the head (pseudo-Huber, optional LPIPS)")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--init", type=Path, default=None, help="default: OUT/dwd_head_ridge.pt")
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--lr", type=float, default=1e-5)
    s.add_argument("--batch", type=int, default=2048, help="tokens per step")
    s.add_argument("--batches-per-sample", type=int, default=8)
    s.add_argument("--lpips-weight", type=float, default=0.0)
    s.add_argument("--grad-clip", type=float, default=1.0)
    s.add_argument("--max-skips", type=int, default=20, help="stop after this many non-finite losses or gradients")
    s.add_argument("--log-every", type=int, default=50)
    s.add_argument("--eval-every", type=int, default=250)
    s.add_argument("--device", default="cuda")

    args = parser.parse_args()
    try:
        from setproctitle import setproctitle
        tags = {"collect": f"items={args.items}x{args.seeds}" if args.phase == "collect" else "",
                "ridge": "", "sgd": f"lr={args.lr:g}" if args.phase == "sgd" else ""}[args.phase]
        setproctitle(f"dwd:{args.phase} {tags} {args.out.name}".replace("  ", " ").strip())
    except ImportError:
        pass
    {"collect": collect, "ridge": ridge, "sgd": sgd}[args.phase](args)


if __name__ == "__main__":
    main()
