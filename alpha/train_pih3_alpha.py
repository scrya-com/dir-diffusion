# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""Stage A: train an alpha head on pi-H3's DWD refiner (2-step by default, ``--pih3-steps``), the DiT and DWD frozen.

Each step:
1. a RGBA still from the AlphaAnything set is placed on a 768x512 canvas, set in motion over 39 frames and
   composited over a background like the ones pi-H3 draws (``data_pih3``);
2. the H3 VAE encodes the clip, noised to the sigma of pi-H3's last step (default schedule or ``max_raw_t`` 1.0,
   the camera fix), so the DiT sees what it sees at its last step;
3. one frozen pi-H3 forward with the still's caption; its DWD head decodes RGB, and the alpha head reads the
   refiner's tokens (``pih3_alpha``);
4. the Endpoint-Aware loss against the true alpha.

Eval also reports ``rgb_psnr``: pi-H3's DWD RGB against the clip it was shown, the check that the teacher-forced
input lines up with the alpha target.

    python -m alpha.train_pih3_alpha pool                                 # scene backgrounds from ComfyUI renders
    python -m alpha.train_pih3_alpha train --out /media/2TB/alpha/runs/a1 --steps 4000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

LOT = Path(__file__).resolve().parents[1] / "lot"
DATA = Path("/media/2TB/alpha/data")
TE = Path("/media/2TB/alpha/te")
SCENES = Path("/media/2TB/alpha/scenes_768x512.pt")
W, H, FRAMES = 768, 512, 39
EVAL_ALIGNED_SIGMA = 0.6  # half the eval clips: a level where the redraw matches the clip


def te_path(image: str) -> Path:
    return TE / (Path(image).stem + "_minimaxh3_te.safetensors")


def setup(steps: int = 2):
    sys.path.insert(0, str(LOT))
    from accel_h3 import token_shape
    from h3_musubi import forward_video, load_vae
    from lot_paths import LOT_H3_VAE
    from pih3_h3 import load_pih3, pih3_sigmas
    from train_dwd_h3 import load_text

    model = load_pih3(steps, merge="comfy")
    vae = load_vae(LOT_H3_VAE, decoder=False)
    sigmas = [pih3_sigmas(steps)[-2], pih3_sigmas(steps, max_raw_t=1.0)[-2]]
    return dict(model=model, vae=vae, sigmas=sigmas, forward_video=forward_video, load_text=load_text,
                token_shape=token_shape)


def sample_clip(env, row, gen, rng, scenes, train=True, kind=None, mode=None):
    from .data import load_rgba
    from .data_pih3 import make_clip, place

    still = place(load_rgba(DATA / "t2rgba" / row["image"], 768), (H, W), rng, train=train).cuda()
    return make_clip(still, FRAMES, gen, scenes=scenes, kind=kind, mode=mode)


@torch.no_grad()
def dit_tokens(env, clip, image, sigma, seed):
    """Encode, noise to ``sigma``, one frozen pi-H3 forward: (h, temb, token grid, latent)."""
    z0 = env["vae"].encode((clip * 2 - 1)[None].half()).float()
    g = torch.Generator("cuda").manual_seed(seed)
    x = (1 - sigma) * z0 + sigma * torch.randn(z0.shape, device="cuda", generator=g)
    store: dict = {}
    text = env["load_text"](str(te_path(image))).cuda()
    env["forward_video"](env["model"]["dit"], x, 1.0 - sigma, text, out=store,
                         audio_generator=torch.Generator("cuda").manual_seed(seed + 1))
    _b, _c, lt, lh, lw = z0.shape
    return store["hidden"], store["temb_video"], (lt, lh // 2, lw // 2)


def band_mask(alpha):
    """Soft pixels plus a 4-pixel band around hard edges (``metrics``' edge set)."""
    from .data import SOFT_HI, SOFT_LO
    soft = (alpha > SOFT_LO) & (alpha < SOFT_HI)
    mp = lambda v: torch.nn.functional.max_pool3d(v, (1, 9, 9), 1, (0, 4, 4))
    return soft | ((mp(alpha) + mp(-alpha)) > 0.5)


def band_loss(score, alpha):
    """The EA loss with its two means taken over the soft + edge band only."""
    from .alpha_math import alpha_to_ea_coordinate, ea_pointwise_gap
    m = band_mask(alpha).float()
    if float(m.sum()) < 1:
        return score.sum() * 0
    gap = ea_pointwise_gap(score, alpha_to_ea_coordinate(alpha))
    a = alpha.float().clamp(0, 1)
    n = m.sum()
    return torch.sqrt(((1 - a) * gap * m).sum() / n + 1e-12) + torch.sqrt((a * gap * m).sum() / n + 1e-12)


def metrics(score, alpha):
    from .alpha_math import ea_score_to_alpha
    from .data import SOFT_HI, SOFT_LO
    pred = ea_score_to_alpha(score)
    err = (pred - alpha).abs()
    soft = (alpha > SOFT_LO) & (alpha < SOFT_HI)
    mp = lambda v: torch.nn.functional.max_pool3d(v, (1, 9, 9), 1, (0, 4, 4))
    edge = soft | ((mp(alpha) + mp(-alpha)) > 0.5)
    nan = float("nan")
    return {"mae": float(err.mean()), "soft_mae": float(err[soft].mean()) if soft.any() else nan,
            "edge_mae": float(err[edge].mean()) if edge.any() else nan}


def grid(clip, alpha, rgb, pred, path, frames=(0, 19, 38)):
    from PIL import Image
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    checker = (((yy // 24 + xx // 24) % 2).float() * 0.35 + 0.45).cuda()[None]
    rows = []
    for f in frames:
        f = min(f, pred.shape[1] - 1)
        c, a, r, q = clip[:, f], alpha[:, f], (rgb[:, f] + 1) / 2, pred[:, f]
        rows.append(torch.cat([c, r, a.expand(3, -1, -1), q.expand(3, -1, -1), r * q + checker * (1 - q)], 2))
    img = torch.nn.functional.interpolate(torch.cat(rows, 1)[None], scale_factor=0.5, mode="area")[0]
    Image.fromarray((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(path, quality=88)


@torch.no_grad()
def evaluate(env, head, rows, out, step, scenes, n=60, log=print):
    from .alpha_math import ea_score_to_alpha
    from .pih3_alpha import rgba_decode
    head.eval()
    kinds = ("dark", "light", "gradient", "scene", "flat", "padded")
    modes = ("affine", "drift", "static", "fade")
    acc: dict = {}
    psnrs = []
    for i, row in enumerate(rows[:n]):
        gen = torch.Generator().manual_seed(10_000 + i)
        rng = random.Random(10_000 + i)
        kind, mode = kinds[i % len(kinds)], modes[(i // (2 * len(kinds))) % len(modes)]
        clip, alpha, kind, mode = sample_clip(env, row, gen, rng, scenes, train=False, kind=kind, mode=mode)
        sig = env["sigmas"][0] if (i // len(kinds)) % 2 == 0 else EVAL_ALIGNED_SIGMA   # every background at both
        h, temb, g = dit_tokens(env, clip, row["image"], sig, 10_000 + i)
        rgb, score = rgba_decode(env["model"]["head"], head, h, temb, g)
        f = min(score.shape[2], alpha.shape[1])
        rgb, score, a, c = rgb[0, :, :f], score[0, :, :f], alpha[:, :f], clip[:, :f]
        mse = float((((rgb + 1) / 2 - c) ** 2).mean())
        psnrs.append(-10 * math.log10(max(mse, 1e-10)))
        m = metrics(score[None], a[None])
        m["rgb_psnr"] = psnrs[-1]
        for key in ("all", "bg:" + kind, "mode:" + mode, row["category"], f"sigma:{sig:.3f}"):
            for k, v in m.items():
                if not math.isnan(v):
                    acc.setdefault(key, {}).setdefault(k, []).append(v)
        if i < 6:
            grid(c, a, rgb, ea_score_to_alpha(score), Path(out) / f"grid_s{step:05d}_{i}.jpg")
    head.train()
    res = {k: {m: round(float(np.mean(v)), 4) for m, v in d.items()} for k, d in acc.items()}
    res["rgb_psnr"] = round(float(np.mean(psnrs)), 2)
    log(f"ALPHA eval step={step} rgb_psnr={res['rgb_psnr']} " + " ".join(f"{k}={v}" for k, v in res["all"].items()))
    log("ALPHA eval by " + json.dumps({k: v for k, v in res.items() if k not in ("all", "rgb_psnr")}))
    (Path(out) / f"eval_{step:05d}.json").write_text(json.dumps(res, indent=1))
    return res


def train(args):
    from setproctitle import setproctitle
    from .alpha_math import ea_loss_per_sample
    from .pih3_alpha import AlphaDWDHead, rgba_decode
    setproctitle(f"alpha:pih3-A {Path(args.out).name} {args.steps}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    subset = json.loads((TE / "subset.json").read_text())
    rows = {json.loads(l)["image"]: json.loads(l) for l in open(DATA / "t2rgba_train.jsonl")}
    train_rows = [rows[i] for i in subset["train"] if te_path(i).is_file()]
    eval_rows = [rows[i] for i in subset["eval"] if te_path(i).is_file()]
    log(f"ALPHA rows train={len(train_rows)} eval={len(eval_rows)}")
    scenes = torch.load(SCENES) if SCENES.is_file() else None
    env = setup(args.pih3_steps)
    log(f"ALPHA pi-H3 {args.pih3_steps}-step; last-step sigmas={[round(s, 4) for s in env['sigmas']]}")
    head = AlphaDWDHead().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    step = 0
    ck = out / "head_last.pt"
    if ck.is_file():
        st = torch.load(ck, map_location="cuda")
        head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))),
        last_epoch=step - 1)
    if step == 0:
        evaluate(env, head, eval_rows, out, 0, scenes, n=args.eval_n, log=log)
    rng = random.Random(args.seed + step)
    gen = torch.Generator().manual_seed(args.seed + step)
    t0 = time.time()
    while step < args.steps:
        row = rng.choice(train_rows)
        clip, alpha, kind, mode = sample_clip(env, row, gen, rng, scenes)
        sigma = (rng.choice(env["sigmas"]) if rng.random() < args.real_sigma_frac
                 else rng.uniform(*args.train_sigma))  # lower noise: pi-H3 redraws what it was shown, so alpha lines up
        h, temb, g = dit_tokens(env, clip, row["image"], sigma, rng.randrange(1 << 30))
        n_out = sum(1 + 4 * (min(5, g[0] - b) - 1) for b in range(0, g[0], 5))  # DWD frames for this grid
        idx = torch.tensor(sorted(rng.sample(range(min(n_out, alpha.shape[1])), args.train_frames)), device="cuda")
        _rgb, score = rgba_decode(env["model"]["head"], head, h, temb, g, frames=idx)
        target = alpha[:, idx][None]                                        # (1, 1, k, H, W)
        loss = ea_loss_per_sample(score, target)[0].mean()
        if args.band_weight > 0:                                            # soft pixels are a few % of the frame
            loss = loss + args.band_weight * band_loss(score, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 25 == 0:
            log(f"ALPHA step={step} loss={loss.item():.4f} bg={kind} lr={sched.get_last_lr()[0]:.2e} "
                f"s/step={(time.time() - t0) / 25:.2f} mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G")
            t0 = time.time()
        if step % 250 == 0 or step == args.steps:
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "step": step}, ck)
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(env, head, eval_rows, out, step, scenes, n=args.eval_n, log=log)
    torch.save({"head": head.state_dict(), "step": step, "adapter": env["model"]["adapter"]}, out / "alpha_dwd_head.pt")
    log("ALPHA_TRAIN_DONE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["pool", "train", "eval"])
    p.add_argument("--out", default="/media/2TB/alpha/runs/a1")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pih3-steps", type=int, default=2, choices=(2, 4), help="which pi-H3 adapter's DWD decoder")
    p.add_argument("--train-sigma", type=float, nargs=2, default=[0.45, 0.75],
                   help="noise range for most steps: low enough that pi-H3's last step redraws the clip it is shown")
    p.add_argument("--real-sigma-frac", type=float, default=0.25, help="share of steps at the real last-step sigma")
    p.add_argument("--band-weight", type=float, default=1.0, help="EA loss on the soft + edge band, added to the frame's")
    p.add_argument("--train-frames", type=int, default=8, help="frames per step through the pixel refiner")
    args = p.parse_args()
    if args.cmd == "pool":
        from .data_pih3 import build_scene_pool
        build_scene_pool(Path("/run/media/johndpope/2TB/ComfyUI/output"), SCENES)
    elif args.cmd == "train":
        train(args)
    else:
        from .pih3_alpha import AlphaDWDHead
        rows = {json.loads(l)["image"]: json.loads(l) for l in open(DATA / "t2rgba_train.jsonl")}
        subset = json.loads((TE / "subset.json").read_text())
        env = setup(args.pih3_steps)
        head = AlphaDWDHead().cuda()
        head.load_state_dict(torch.load(Path(args.out) / "alpha_dwd_head.pt", map_location="cuda")["head"])
        evaluate(env, head, [rows[i] for i in subset["eval"]], args.out, -1,
                 torch.load(SCENES) if SCENES.is_file() else None, n=args.eval_n)


if __name__ == "__main__":
    main()
