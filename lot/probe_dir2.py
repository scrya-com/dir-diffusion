#!/usr/bin/env python3
"""Can the published DiR run in 2 forward passes? Training-free schedule probe on held-out prompts, 768x512x39.

Variants (same noise z per prompt):
- full:   r0 @ 1, r1 @ 0.947, r2 @ 0.857, r3 @ 0.667 (the published 4 iterations)
- exit2:  stop after r1 (the measured early exit; ACCEL.md: LPIPS 0.462)
- skip03: r0 @ 1, one Euler jump to 0.667, then r3 @ 0.667 (finish the way DiR finishes)
- skip02: r0 @ 1, jump to 0.857, then r2 @ 0.857
Each final estimate is VAE-decoded and scored against the full rollout's decode (LPIPS vgg, PSNR).

    python3 lot/probe_dir2.py --prompts 4 --swap 20 --out /media/2TB/alpha/dir2_probe
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


S = [1.0, 0.9473684210526315, 0.8571428571428571, 0.6666666666666666]
PLANS = {"full": [(0, S[0]), (1, S[1]), (2, S[2]), (3, S[3])], "exit2": [(0, S[0]), (1, S[1])],
         "skip03": [(0, S[0]), (3, S[3])], "skip02": [(0, S[0]), (2, S[2])]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=int, default=4)
    p.add_argument("--swap", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--score", action="store_true", help="second process: VAE-decode the saved latents and score")
    args = p.parse_args()
    if args.score:
        return score(args, PLANS)
    from setproctitle import setproctitle
    setproctitle("dir:2-step probe")
    import lpips
    from PIL import Image
    from accel_h3 import default_caches, prompt_lists
    from dir_h3 import generator_step, load_dir_run, next_state, use_networks
    from dir_two_track_h3 import resolve_run
    from h3_musubi import load_dit, load_vae
    from lot_paths import LOT_H3_VAE
    from train_dwd_h3 import load_text

    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    dit = load_dit(device=dev, swap=args.swap)
    head, net, config = load_dir_run(str(resolve_run("johndpope/dir-diffusion-h3")), dit)
    plans = PLANS
    _train, held = prompt_lists(default_caches(), 8)
    latents = {}
    with torch.no_grad(), use_networks(net, None, gen=True, fake=False):
        for i, te in enumerate(held[:args.prompts]):
            text = load_text(str(te)).to(dev)
            z = torch.randn(1, 24, 12, 32, 48, device=dev, generator=torch.Generator(dev).manual_seed(100 + i))
            for name, plan in plans.items():
                state, est = z, torch.zeros_like(z)
                for k, (r, sig) in enumerate(plan):
                    est = generator_step(dit, head, state, est, r, text, sigma=sig,
                                         audio_generator=torch.Generator(dev).manual_seed(31 * i)).float()
                    if k + 1 < len(plan):
                        state = next_state(state, est, sig, plan[k + 1][1])
                latents[(i, name)] = est.half().cpu()
            print(f"PROBE prompt {i} done", flush=True)
    torch.save({f"{i}:{n}": v for (i, n), v in latents.items()}, args.out / "latents.pt")
    print("PROBE latents saved", flush=True)


def score(args, plans):
    import lpips
    from PIL import Image
    from h3_musubi import load_vae
    from lot_paths import LOT_H3_VAE
    dev = torch.device("cuda")
    raw = torch.load(args.out / "latents.pt")
    latents = {(int(k.split(":")[0]), k.split(":")[1]): v for k, v in raw.items()}
    n_prompts = 1 + max(i for i, _ in latents)
    vae = load_vae(LOT_H3_VAE, device=dev)
    lp = lpips.LPIPS(net="vgg", verbose=False).cuda().eval()
    res, sheets = {}, []
    with torch.no_grad():
        for i in range(n_prompts):
            px = {n: vae.decode(latents[(i, n)].to(dev, next(vae.parameters()).dtype)).float()[0].clamp(-1, 1) for n in plans}
            ref = px["full"]
            f = ref.shape[1]
            for n in plans:
                a, b = px[n].transpose(0, 1), ref.transpose(0, 1)
                l = float(torch.cat([lp(a[j:j + 4], b[j:j + 4]).flatten() for j in range(0, f, 4)]).mean())
                mse = float(((a - b) / 2).pow(2).mean())
                res.setdefault(n, []).append((l, -10 * math.log10(max(mse, 1e-10))))
            mid = f // 2
            sheets.append(np.concatenate([((px[n][:, mid].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
                                          for n in plans], 1))
    summ = {n: {"lpips_vs_full": round(float(np.mean([r[0] for r in v])), 4),
                "psnr_vs_full": round(float(np.mean([r[1] for r in v])), 2), "forwards": len(plans[n])} for n, v in res.items()}
    print("PROBE_SUMMARY", json.dumps(summ), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summ, indent=1))
    Image.fromarray(np.concatenate(sheets, 0)).resize((len(plans) * 384, len(sheets) * 256)).save(
        args.out / "sheet_full_exit2_skip03_skip02.jpg", quality=88)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
