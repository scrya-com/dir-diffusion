# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""P1: train the H3 VAE alpha head on clean latents of RGBA pseudo-videos (no DiT).

    python -m alpha.train_alpha split   --data /media/2TB/alpha/data
    python -m alpha.train_alpha train   --data /media/2TB/alpha/data --out /media/2TB/alpha/runs/p1 --steps 4000
    python -m alpha.train_alpha eval    --data /media/2TB/alpha/data --out /media/2TB/alpha/runs/p1

Each step:
1. makes B pseudo-video clips (``data.make_clips``);
2. encodes their padded RGB with the frozen H3 VAE encoder;
3. decodes through the VAE's own decode path with the head installed;
4. applies the Endpoint-Aware alpha loss.

Gate (decided before a DiT LoRA is worth training): on the soft-heavy eval split, the alpha recovered from clean
latents must be close to the truth in soft regions (``soft_mae``), and the grids must look right on smoke, glass
and hair. Run with ComfyUI on ``sys.path`` (``COMFYUI_DIR``) and its python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

COMFY = os.environ.get("COMFYUI_DIR", "/run/media/johndpope/2TB/ComfyUI")
FRAMES = 22  # 21..33 input frames -> 7 latents -> 22 decoded frames, decoded frame k == input frame k
VAE_FILE = "models/vae/minimax_h3_video_vae_fp16.safetensors"


def load_vae():
    sys.path.insert(0, COMFY)
    cwd = os.getcwd()
    os.chdir(COMFY)
    try:
        import comfy.sd
        import comfy.utils
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE_FILE))
    finally:
        os.chdir(cwd)
    m = vae.first_stage_model.cuda().eval().requires_grad_(False)
    return m


def rows_root(data):
    return Path(data) / "t2rgba"


def encode(m, padded):
    with torch.no_grad():
        return m.encode((padded * 2 - 1).half())


def decode_rgba(m, z):
    b, _, lt, lh, lw = z.shape
    shape = m.decode_output_shape(z.shape)
    buf = torch.empty(shape, device=z.device, dtype=torch.float32)
    out = m.decode(z.half(), output_buffer=buf)
    return out[:, :3], out[:, 3:4]


def metrics(score, alpha):
    from .alpha_math import ea_score_to_alpha
    from .data import SOFT_HI, SOFT_LO
    pred = ea_score_to_alpha(score)
    err = (pred - alpha).abs()
    soft = (alpha > SOFT_LO) & (alpha < SOFT_HI)
    mp = lambda v: torch.nn.functional.max_pool3d(v, (1, 9, 9), 1, (0, 4, 4))
    edge = soft | ((mp(alpha) + mp(-alpha)) > 0.5)  # dilate - erode: within 4 px of a hard edge
    return {"mse": float((err ** 2).mean()), "mae": float(err.mean()),
            "soft_mae": float(err[soft].mean()) if soft.any() else float("nan"),
            "edge_mae": float(err[edge].mean()) if edge.any() else float("nan"),
            "soft_px": int(soft.sum())}


def grid_image(padded, alpha, pred, path, frames=(0, 11, 21)):
    """Rows per clip: padded RGB | true alpha | predicted alpha | composite on checker with predicted alpha."""
    from PIL import Image
    b, _, t, h, w = padded.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    checker = (((yy // 16 + xx // 16) % 2).float() * 0.35 + 0.45).to(padded.device)[None]
    rows = []
    for i in range(b):
        cells = []
        for f in frames:
            p, a, q = padded[i, :, f], alpha[i, :, f], pred[i, :, f]
            comp = p * q + checker * (1 - q)
            cells += [p, a.expand(3, -1, -1), q.expand(3, -1, -1), comp]
        rows.append(torch.cat(cells, 2))
    img = torch.cat(rows, 1).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save(path, quality=90)


def offset_latent(m, offset, padded, alpha):
    """z_rgb (+ delta from the offset net when ``offset`` is given), plus z_rgb itself."""
    z_rgb = encode(m, padded)
    if offset is None:
        return z_rgb, z_rgb
    z_a = encode(m, alpha.expand(-1, 3, -1, -1, -1))
    return z_rgb + offset(z_rgb, z_a), z_rgb


def psnr_premul(rgb, ref, alpha):
    mse = (((rgb - ref) * alpha) ** 2).mean().clamp_min(1e-10)
    return float(-10 * torch.log10(mse))


def evaluate(m, head, split, data, out, step, batch=4, n_grid=8, log=print, offset=None):
    """Alpha metrics on the eval split (overall, per category, per motion mode). With an offset net, also the
    delta = 0 ablation (``zero:``) and the premultiplied-RGB PSNR of decode(z + delta) against decode(z)."""
    from .alpha_math import ea_score_to_alpha
    from .data import RGBAStills, make_clips
    from .h3_alpha import install
    head.eval()
    if offset is not None:
        offset.eval()
    ds = RGBAStills(rows_root(data), split["eval"], train=False)
    gen = torch.Generator().manual_seed(1234)
    uninstall = install(m, head)
    agg, cats, psnrs = {}, {}, []

    def add(key, k, v):
        if not math.isnan(v):
            cats.setdefault(key, {}).setdefault(k, []).append(v)

    try:
        with torch.no_grad():
            for s in range(0, len(ds), batch):
                items = [ds[i] for i in range(s, min(s + batch, len(ds)))]
                stills = torch.stack([x for x, _ in items]).cuda()
                padded, alpha, modes = make_clips(stills, FRAMES, gen, return_modes=True)
                z, z_rgb = offset_latent(m, offset, padded, alpha)
                rgb, score = decode_rgba(m, z)
                if offset is not None:
                    rgb0, score0 = decode_rgba(m, z_rgb)
                    psnrs.append(psnr_premul(rgb, rgb0, alpha))
                for j, (_x, idx) in enumerate(items):
                    mt = metrics(score[j:j + 1], alpha[j:j + 1])
                    cat = split["eval"][idx]["category"]
                    for k in ("mae", "soft_mae"):
                        add("mode:" + modes[j], k, mt[k])
                    for k in ("mse", "mae", "soft_mae", "edge_mae"):
                        if not math.isnan(mt[k]):
                            agg.setdefault(k, []).append(mt[k])
                        add(cat, k, mt[k])
                    if offset is not None:
                        m0 = metrics(score0[j:j + 1], alpha[j:j + 1])
                        add("zero:all", "soft_mae", m0["soft_mae"])
                        add("zero:all", "mae", m0["mae"])
                if s // batch < max(1, n_grid // batch):
                    grid_image(padded, alpha, ea_score_to_alpha(score), Path(out) / f"grid_s{step:05d}_{s // batch}.jpg")
    finally:
        uninstall()
        head.train()
        if offset is not None:
            offset.train()
    res = {k: float(np.mean(v)) for k, v in agg.items()}
    if psnrs:
        res["premul_psnr"] = float(np.mean(psnrs))
    res["by_category"] = {c: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for c, d in cats.items()}
    log(f"ALPHA eval step={step} " + " ".join(f"{k}={v:.4f}" for k, v in res.items() if k != "by_category"))
    log("ALPHA eval modes " + json.dumps({k: v for k, v in res["by_category"].items() if ":" in k}))
    log("ALPHA eval by_category " + json.dumps(res["by_category"]))
    return res


def train(args):
    from .alpha_math import ea_loss_per_sample
    from .data import RGBAStills, make_clips
    from .h3_alpha import AlphaHead, AlphaOffset, install

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    split = json.loads((Path(args.data) / "split.json").read_text())
    m = load_vae()
    head = AlphaHead().cuda()
    offset = AlphaOffset().cuda() if args.offset else None
    params = list(head.parameters()) + (list(offset.parameters()) if offset is not None else [])
    log(f"ALPHA head params {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M taps {head.taps} "
        f"offset {sum(p.numel() for p in offset.parameters()) / 1e6 if offset is not None else 0:.1f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    start = 0
    ck = out / "head_last.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cuda")
        head.load_state_dict(st["head"])
        if offset is not None and "offset" in st:
            offset.load_state_dict(st["offset"])
        opt.load_state_dict(st["opt"])
        start = st["step"]
        log(f"ALPHA resumed at step {start}")
    elif args.init:
        head.load_state_dict(torch.load(args.init, map_location="cuda")["head"])
        log(f"ALPHA head warm-started from {args.init}")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))) * 0.9 + 0.1,
        last_epoch=start - 1)
    ds = RGBAStills(rows_root(args.data), split["train"])
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         drop_last=True, persistent_workers=True)
    gen = torch.Generator().manual_seed(args.seed + start)
    trunk_grad = offset is not None
    uninstall = install(m, head, trunk_grad=trunk_grad)
    step, t0 = start, time.time()
    keys = ("loss", "ea", "rgb", "delta")
    run = dict.fromkeys(keys, 0.0) | {"n": 0}
    while step < args.steps:
        for stills, _idx in loader:
            padded, alpha = make_clips(stills.cuda(non_blocking=True), FRAMES, gen)
            if offset is not None:
                z_rgb = encode(m, padded)
                z_a = encode(m, alpha.expand(-1, 3, -1, -1, -1))
                delta = offset(z_rgb, z_a)
                with torch.no_grad():
                    rgb0, _ = decode_rgba(m, z_rgb)
                rgb, score = decode_rgba(m, z_rgb + delta)
                l_rgb = (((rgb - rgb0) * alpha) ** 2).mean()
                l_delta = delta.square().mean()
            else:
                _rgb, score = decode_rgba(m, encode(m, padded))
                l_rgb = l_delta = torch.zeros((), device="cuda")
            ea, _tr, _op = ea_loss_per_sample(score.float(), alpha)
            ea = ea.mean()
            loss = ea + args.rgb_weight * l_rgb + args.delta_weight * l_delta
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in zip(keys, (loss, ea, l_rgb, l_delta)):
                run[k] += float(v)
            run["n"] += 1
            if step % 50 == 0:
                n = run["n"]
                log(f"ALPHA step={step} loss={run['loss'] / n:.4f} ea={run['ea'] / n:.4f} rgb={run['rgb'] / n:.2e} "
                    f"delta={run['delta'] / n:.2e} lr={sched.get_last_lr()[0]:.2e} s/step={(time.time() - t0) / n:.2f} "
                    f"mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G")
                run, t0 = dict.fromkeys(keys, 0.0) | {"n": 0}, time.time()
            if step % args.save_every == 0 or step == args.steps:
                st = {"head": head.state_dict(), "opt": opt.state_dict(), "step": step, "config": {"taps": head.taps}}
                if offset is not None:
                    st["offset"] = offset.state_dict()
                torch.save(st, ck)
            if step % args.eval_every == 0 or step == args.steps:
                uninstall()
                res = evaluate(m, head, split, args.data, out, step, log=log, offset=offset)
                (out / f"eval_{step:05d}.json").write_text(json.dumps(res, indent=1))
                uninstall = install(m, head, trunk_grad=trunk_grad)
                t0 = time.time()
            if step >= args.steps:
                break
    uninstall()
    final = {"head": head.state_dict(), "step": step, "config": {"taps": head.taps}}
    if offset is not None:
        final["offset"] = offset.state_dict()
    torch.save(final, out / "alpha_head.pt")
    log("ALPHA_TRAIN_DONE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["split", "train", "eval"])
    p.add_argument("--data", default="/media/2TB/alpha/data")
    p.add_argument("--out", default="/media/2TB/alpha/runs/p1")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offset", action="store_true", help="train a latent-transparency offset with the head")
    p.add_argument("--init", default=None, help="warm-start the head from this checkpoint")
    p.add_argument("--rgb-weight", type=float, default=20.0, help="premultiplied RGB harmlessness weight")
    p.add_argument("--delta-weight", type=float, default=1e-3, help="delta L2 weight")
    args = p.parse_args()
    if args.cmd == "split":
        from .data import split_rows
        s = split_rows(Path(args.data) / "t2rgba_train.jsonl", rows_root(args.data), Path(args.data) / "split.json")
        print("train", len(s["train"]), "eval", len(s["eval"]))
    elif args.cmd == "train":
        train(args)
    else:
        from .h3_alpha import AlphaHead, AlphaOffset
        st = torch.load(Path(args.out) / "head_last.pt", map_location="cuda")
        head = AlphaHead().cuda()
        head.load_state_dict(st["head"])
        offset = None
        if "offset" in st:
            offset = AlphaOffset().cuda()
            offset.load_state_dict(st["offset"])
        evaluate(load_vae(), head, json.loads((Path(args.data) / "split.json").read_text()), args.data, args.out, -1,
                 offset=offset)


if __name__ == "__main__":
    main()
