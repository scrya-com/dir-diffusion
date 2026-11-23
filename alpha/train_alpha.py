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


def evaluate(m, head, split, data, out, step, batch=4, n_grid=8, log=print):
    from .alpha_math import ea_score_to_alpha
    from .data import RGBAStills, make_clips
    from .h3_alpha import install
    head.eval()
    ds = RGBAStills(rows_root(data), split["eval"], train=False)
    gen = torch.Generator().manual_seed(1234)
    uninstall = install(m, head)
    agg, cats = {}, {}
    try:
        with torch.no_grad():
            for s in range(0, len(ds), batch):
                items = [ds[i] for i in range(s, min(s + batch, len(ds)))]
                stills = torch.stack([x for x, _ in items]).cuda()
                padded, alpha = make_clips(stills, FRAMES, gen)
                _rgb, score = decode_rgba(m, encode(m, padded))
                for j, (_x, idx) in enumerate(items):
                    mt = metrics(score[j:j + 1], alpha[j:j + 1])
                    cat = split["eval"][idx]["category"]
                    for k in ("mse", "mae", "soft_mae", "edge_mae"):
                        if not math.isnan(mt[k]):
                            agg.setdefault(k, []).append(mt[k])
                            cats.setdefault(cat, {}).setdefault(k, []).append(mt[k])
                if s // batch < max(1, n_grid // batch):
                    grid_image(padded, alpha, ea_score_to_alpha(score), Path(out) / f"grid_s{step:05d}_{s // batch}.jpg")
    finally:
        uninstall()
        head.train()
    res = {k: float(np.mean(v)) for k, v in agg.items()}
    res["by_category"] = {c: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for c, d in cats.items()}
    log(f"ALPHA eval step={step} " + " ".join(f"{k}={v:.4f}" for k, v in res.items() if k != "by_category"))
    log("ALPHA eval by_category " + json.dumps(res["by_category"]))
    return res


def train(args):
    from .alpha_math import ea_loss_per_sample
    from .data import RGBAStills, make_clips
    from .h3_alpha import AlphaHead, install

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
    log(f"ALPHA head params {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M taps {head.taps}")
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    start = 0
    ck = out / "head_last.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cuda")
        head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        start = st["step"]
        log(f"ALPHA resumed at step {start}")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))) * 0.9 + 0.1,
        last_epoch=start - 1)
    ds = RGBAStills(rows_root(args.data), split["train"])
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         drop_last=True, persistent_workers=True)
    gen = torch.Generator().manual_seed(args.seed + start)
    uninstall = install(m, head)
    step, t0 = start, time.time()
    run = {"loss": 0.0, "n": 0}
    while step < args.steps:
        for stills, _idx in loader:
            padded, alpha = make_clips(stills.cuda(non_blocking=True), FRAMES, gen)
            z = encode(m, padded)
            _rgb, score = decode_rgba(m, z)
            loss, _tr, _op = ea_loss_per_sample(score.float(), alpha)
            loss = loss.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run["loss"] += float(loss)
            run["n"] += 1
            if step % 50 == 0:
                log(f"ALPHA step={step} loss={run['loss'] / run['n']:.4f} lr={sched.get_last_lr()[0]:.2e} "
                    f"s/step={(time.time() - t0) / run['n']:.2f} mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G")
                run, t0 = {"loss": 0.0, "n": 0}, time.time()
            if step % args.save_every == 0 or step == args.steps:
                torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "step": step,
                            "config": {"taps": head.taps}}, ck)
            if step % args.eval_every == 0 or step == args.steps:
                uninstall()
                res = evaluate(m, head, split, args.data, out, step, log=log)
                (out / f"eval_{step:05d}.json").write_text(json.dumps(res, indent=1))
                uninstall = install(m, head)
                t0 = time.time()
            if step >= args.steps:
                break
    uninstall()
    torch.save({"head": head.state_dict(), "step": step, "config": {"taps": head.taps}}, out / "alpha_head.pt")
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
    args = p.parse_args()
    if args.cmd == "split":
        from .data import split_rows
        s = split_rows(Path(args.data) / "t2rgba_train.jsonl", rows_root(args.data), Path(args.data) / "split.json")
        print("train", len(s["train"]), "eval", len(s["eval"]))
    elif args.cmd == "train":
        train(args)
    else:
        from .h3_alpha import AlphaHead
        head = AlphaHead().cuda()
        head.load_state_dict(torch.load(Path(args.out) / "head_last.pt", map_location="cuda")["head"])
        evaluate(load_vae(), head, json.loads((Path(args.data) / "split.json").read_text()), args.data, args.out, -1)


if __name__ == "__main__":
    main()
