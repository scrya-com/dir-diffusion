"""Measure eDMD's calibration table for H3 on a student sigma grid, from the teacher alone.

    python3 lot/calibrate_edmd_h3.py --sigmas turbo --frames 12 --max-latent-hw 32 48 --prompts 32 \
        --out lot/runs/edmd_calibration_turbo_f12.json
    python3 lot/train_dir_h3.py ... --tau-floor lot/runs/edmd_calibration_turbo_f12.json

After ``edmd/cli/calibrate.py`` of eDMD (arXiv 2611.17194, Appendix D), at guidance 1 (this repo samples H3
unguided, so the guided and the conditional answer are the same and only one test function is left).

For each training prompt, ``X`` is the teacher's own sample (``--steps`` Euler steps from noise). For every landing
point ``s = sigmas[k + 1]`` of the grid and level ``t = s + (0.99 - s) f``, a Markov pair
``y_s = (1 - s) X + s xi``, ``y_t = a y_s + sqrt(t^2 - (a s)^2) z`` with ``a = (1 - t) / (1 - s)`` gives
``Delta = A(y_t, t) - A(y_s, s)`` (``A``: the teacher's one-call x0 answer) and ``err_s = E|A(y_s, s) - X|^2``.
The table's ``d`` is

    d_s(t) = (E<Delta, A(y_t, t)>)^2 / (E|A(y_t, t)|^2 err_s)

a lower bound on how far the answer drifts between the two levels in units of the teacher's own error at ``s``;
``d_total = E|Delta|^2 / err_s``. Means run over the latent's elements and E pools prompts (ratios of means).
``train_dir_h3.py --tau-floor TABLE.json`` then reads iteration ``k`` before the last at
``t ~ exp(-d_s(t))`` on ``[s, 0.99)``.

Cost per prompt: ``--steps`` forwards for the sample plus 14 per landing point. Not in ``gpu_guard``'s list:
pick a free card with ``CUDA_VISIBLE_DEVICES``. The table is rewritten after every prompt.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: A landing point's levels ``s + (top - s) f``, dense near ``s`` (eDMD's; f = 0 is the landing point, d = 0).
FRACTIONS = (0.0, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 3 / 16, 1 / 4, 5 / 16, 3 / 8, 1 / 2, 5 / 8, 3 / 4, 7 / 8, 1.0)
TOP = 0.99


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sigmas", default="turbo", help="the student's sigma grid, as train_dir_h3.py --sigmas")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--max-latent-hw", type=int, nargs=2, default=[48, 48], metavar=("H", "W"))
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20, help="Euler steps of the teacher sample")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--swap", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import train_h3
    from dir_h3 import parse_sigmas, real_x0
    from h3_musubi import load_dit, predict, sigma_schedule
    from train_dir_h3 import ReadLaw, load_item

    train_h3.MAX_LATENT_HW = tuple(args.max_latent_hw)
    train_h3.CROP_MULTIPLE = 2
    dev = torch.device("cuda")
    sigmas = [float(s) for s in parse_sigmas(args.sigmas, args.iterations)]
    landing = sigmas[1:]
    if min(landing) >= 1.0:
        raise SystemExit("--sigmas has no landing point below 1")
    levels = [[s + (TOP - s) * f for f in FRACTIONS] for s in landing]
    train, _held = train_h3.list_items(args.cache or train_h3.CACHE, args.holdout)
    rng = random.Random(args.seed)
    chosen = rng.sample(train, min(args.prompts, len(train)))
    dit = load_dit(device=dev, swap=args.swap)
    grid = [float(s) for s in sigma_schedule(args.steps)]
    noise = torch.Generator(dev).manual_seed(args.seed)
    count = len(FRACTIONS)
    ip = [[0.0] * count for _ in landing]       # sum over prompts of mean(Delta * A_t)
    nn = [[0.0] * count for _ in landing]       # ... of mean(A_t^2)
    dd = [[0.0] * count for _ in landing]       # ... of mean(Delta^2)
    err = [0.0 for _ in landing]                # ... of mean((A_s - X)^2)

    def draw(like):
        return torch.randn(like.shape, device=dev, generator=noise).to(like.dtype)

    def table(done: int) -> dict:
        d = [[0.0 if j == 0 else ip[k][j] ** 2 / max(nn[k][j] * err[k], 1e-12) for j in range(count)]
             for k in range(len(landing))]
        total = [[dd[k][j] / max(err[k], 1e-12) for j in range(count)] for k in range(len(landing))]
        read = []
        for k, s in enumerate(landing):
            law = ReadLaw(levels[k], d[k], max(s, 0.02), TOP)
            read.append({"landing": round(s, 6), "mean_read": round(law.mean, 4), "uniform_mean": round((s + TOP) / 2, 4)})
        return {"model": "minimax_h3", "cfg": 1.0, "sigmas": sigmas, "frames": args.frames,
                "max_latent_hw": list(args.max_latent_hw), "teacher_steps": args.steps, "prompts": done,
                "seed": args.seed, "landing": landing, "levels": levels, "d": d, "d_total": total,
                "err": [e / max(done, 1) for e in err], "readings": read}

    with torch.no_grad():
        for index, item in enumerate(chosen):
            x0, text = load_item(item, rng, dev, args.frames)
            x, audio = draw(x0), torch.Generator(dev).manual_seed(args.seed + index)
            for s, s_next in zip(grid[:-1], grid[1:]):           # the teacher's own sample X
                clean = x + s * predict(dit, x, 1.0 - s, text, audio_generator=audio).float()
                x = ((s_next / s) * x + (1.0 - s_next / s) * clean).to(x0.dtype)
            sample = x.float()
            for k, s in enumerate(landing):
                y_s = ((1.0 - s) * sample + s * draw(x0).float()).to(x0.dtype)
                a_s = real_x0(dit, y_s, s, text, audio_generator=torch.Generator(dev).manual_seed(args.seed + index))
                err[k] += float((a_s - sample).pow(2).mean())
                for j in range(1, count):
                    t = levels[k][j]
                    scale = (1.0 - t) / (1.0 - s)
                    spread = max(t * t - (scale * s) ** 2, 0.0) ** 0.5
                    y_t = (scale * y_s.float() + spread * draw(x0).float()).to(x0.dtype)
                    a_t = real_x0(dit, y_t, t, text, audio_generator=torch.Generator(dev).manual_seed(args.seed + index))
                    delta = a_t - a_s
                    ip[k][j] += float((delta * a_t).mean())
                    nn[k][j] += float(a_t.pow(2).mean())
                    dd[k][j] += float(delta.pow(2).mean())
            result = table(index + 1)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=1))
            print(f"LOT_H3 kind=edmd_calibrate prompt={index + 1}/{len(chosen)} readings={json.dumps(result['readings'])}",
                  flush=True)
    print(f"LOT_H3 kind=edmd_calibrate_done out={args.out} prompts={len(chosen)} ok=1", flush=True)


if __name__ == "__main__":
    main()
