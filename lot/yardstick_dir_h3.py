"""Seed-spread yardstick: latent statistics of a sampler over held-out prompts x seeds, to compare against H3.

    python3 lot/yardstick_dir_h3.py --name dense --dense --out lot/runs/yardstick      # 20-step H3, once
    python3 lot/yardstick_dir_h3.py --name turbo --run lot/runs/turbo4_init_run --out lot/runs/yardstick
    python3 lot/yardstick_dir_h3.py --name s50 --run lot/runs/dir_clip --tag 50 [--ema] --out lot/runs/yardstick
    python3 lot/yardstick_dir_h3.py --out lot/runs/yardstick --report dense turbo s50   # table, paired with the 2nd

DMD matches distributions, not noise-to-sample pairs, so the distance to the same-noise 20-step clip is a weak
measure (it falls whenever an output shrinks or blurs). This scores what a sampler produces over ``--seeds`` seeds
of each of the ``--holdout`` prompts, in the latent, with no VAE:

* ``motion``: ``train_dir_h3.motion`` (mean |difference| between VAE clip-start latent frames);
* ``detail``: mean |difference| between neighbouring latent pixels, horizontal and vertical;
* ``rms``: root mean square of the latent.

Every name uses the same noise and audio seed per (prompt, seed), so ``--report`` can pair them: it prints each
name's means and, for every name after the second, the mean paired difference against the second name with its
standard error. A run directory needs ``lora_gen*.safetensors``, ``dir_head*.pt`` and ``config.json`` (see the
cheatsheet for wrapping a converted Turbo LoRA). Not in ``gpu_guard``'s list: pick a free card.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

KEYS = ("motion", "detail", "rms")


def stats(x: torch.Tensor, motion) -> dict:
    x = x.float()
    detail = ((x[..., 1:] - x[..., :-1]).abs().mean() + (x[..., 1:, :] - x[..., :-1, :]).abs().mean()) / 2
    return {"rms": float(x.pow(2).mean().sqrt()), "motion": float(motion(x) or 0.0), "detail": float(detail)}


def report(out: Path, names: list[str]) -> None:
    rows = {name: json.loads((out / f"{name}.json").read_text()) for name in names}
    for name, items in rows.items():
        print(f"{name:12} n={len(items):3d}  " + "  ".join(f"{k} {statistics.mean(r[k] for r in items):.3f}" for k in KEYS))
    if len(names) < 3:
        return
    base = {(r["item"], r["seed"]): r for r in rows[names[1]]}
    for name in names[2:]:
        for key in KEYS:
            diff = [r[key] - base[(r["item"], r["seed"])][key] for r in rows[name] if (r["item"], r["seed"]) in base]
            if len(diff) > 1:
                print(f"{name:12} {key:7} vs {names[1]}: {statistics.mean(diff):+.3f} +- "
                      f"{statistics.pstdev(diff) / len(diff) ** 0.5:.3f} (n={len(diff)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", nargs="+", default=None, metavar="NAME", help="print a table of saved names and exit")
    parser.add_argument("--name", default=None)
    parser.add_argument("--dense", action="store_true", help="the base model's --steps Euler sampler")
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--sigmas", default=None, help="default: the run's config")
    parser.add_argument("--mix-state", action="store_true", default=None)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--max-latent-hw", type=int, nargs=2, default=[32, 48], metavar=("H", "W"))
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--swap", type=int, default=4)
    args = parser.parse_args()
    if args.report:
        return report(args.out, args.report)
    if args.name is None or (args.run is None) == (not args.dense):
        raise SystemExit("give --name and exactly one of --dense / --run")

    import train_h3
    from dir_h3 import generator_step, load_dir_run, next_state, parse_sigmas, use_networks
    from h3_musubi import load_dit, predict, sigma_schedule
    from train_dir_h3 import load_item, motion

    train_h3.MAX_LATENT_HW = tuple(args.max_latent_hw)
    train_h3.CROP_MULTIPLE = 2
    dev = torch.device("cuda")
    _train, held = train_h3.list_items(args.cache or train_h3.CACHE, args.holdout)
    dit = load_dit(device=dev, swap=args.swap)
    R = args.iterations
    if args.run is not None:
        head, net, config = load_dir_run(args.run, dit, tag=f"_{args.tag}" if args.tag else "", ema=args.ema)
        sigmas = parse_sigmas(args.sigmas or config.get("sigmas") or "1", R)
        mix = bool(config.get("mix_state", False)) if args.mix_state is None else bool(args.mix_state)
    grid = [float(s) for s in sigma_schedule(args.steps)]
    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, item in enumerate(held):
            x0, text = load_item(item, random.Random(1000 + index), dev, args.frames)
            for seed in range(args.seeds):
                z = torch.randn(x0.shape, device=dev,
                                generator=torch.Generator(dev).manual_seed(5000 + 100 * index + seed)).to(x0.dtype)
                audio_seed = 31 * index + seed
                if args.dense:
                    x, audio = z, torch.Generator(dev).manual_seed(audio_seed)
                    for s, s_next in zip(grid[:-1], grid[1:]):
                        clean = x + s * predict(dit, x, 1.0 - s, text, audio_generator=audio).float()
                        x = ((s_next / s) * x + (1.0 - s_next / s) * clean).to(x0.dtype)
                    final = x
                else:
                    estimate, state = torch.zeros_like(z), z
                    with use_networks(net, None, gen=True, fake=False):
                        for r in range(R):
                            estimate = generator_step(dit, head, state, estimate, r, text, sigma=sigmas[r],
                                                      audio_generator=torch.Generator(dev).manual_seed(audio_seed)).float()
                            if r + 1 < R:
                                state = next_state(state, estimate, sigmas[r], sigmas[r + 1], z if mix else None)
                    final = estimate
                rows.append({"item": index, "seed": seed, **stats(final, motion)})
                (args.out / f"{args.name}.json").write_text(json.dumps(rows))
            print(f"LOT_H3 kind=yardstick name={args.name} prompt={index + 1}/{len(held)} "
                  + " ".join(f"{k}={statistics.mean(r[k] for r in rows):.3f}" for k in KEYS), flush=True)
    print(f"LOT_H3 kind=yardstick_done name={args.name} n={len(rows)} ok=1", flush=True)


if __name__ == "__main__":
    main()
