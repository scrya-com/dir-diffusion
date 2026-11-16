"""Time and memory of DiR sampling on H3 against the dense sampler, DiT only (no text encoder, no VAE).

    python3 lot/bench_dir_h3.py --run lot/runs/train_dir_h3 --swap 4 --canvas 768x768 --canvas 1344x768 \
        --out lot/runs/bench_5090_swap4.json

Per canvas (stills, one latent frame) it reports, after a warm-up pass on the same shapes:

* ``forward_ms``: median of ``--reps`` single forwards, for base H3 and for one DiR iteration (the run's
  generator LoRA, head and sigma; no ``--run``: a zero head and no LoRA);
* ``sample_s``: one whole sampling pass, wall clock: base H3 at ``--steps`` Euler steps, and R DiR iterations;
* ``peak_mb``: ``torch.cuda.max_memory_allocated`` / ``max_memory_reserved`` over the DiR pass and the dense pass.

Not in ``gpu_guard``'s list: pick a free card with ``CUDA_VISIBLE_DEVICES``; another job on the same card
makes the numbers meaningless.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--sigmas", default=None)
    parser.add_argument("--mix-state", action="store_true", default=None)
    parser.add_argument("--canvas", action="append", default=None, metavar="WxH", help="repeatable (default 768x768)")
    parser.add_argument("--steps", type=int, default=20, help="dense sampler steps")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--swap", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=None, help="H3 cache for one held-out prompt's text embedding")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import train_h3
    from dir_h3 import DirHead, generator_step, load_dir_run, next_state, parse_sigmas, probe_temb_dim, use_networks
    from h3_musubi import load_dit, predict, sigma_schedule

    dev = torch.device("cuda")
    train_h3.MAX_LATENT_HW = (16, 16)
    _train, held = train_h3.list_items(args.cache or train_h3.CACHE, 8)
    _x0, text = train_h3.load_item(held[0], random.Random(1000), dev)
    dit = load_dit(device=dev, swap=args.swap)
    R = args.iterations
    config, net, rank = {}, None, 0
    if args.run is not None:
        head, net, config = load_dir_run(args.run, dit, tag=f"_{args.tag}" if args.tag else "")
        rank = int(net.unet_loras[0].lora_dim)
    else:
        head = DirHead(dit.config.hidden_size, probe_temb_dim(dit), R).to(dev).eval()
    sigmas = parse_sigmas(args.sigmas or config.get("sigmas") or "1", R)
    mix = bool(config.get("mix_state", False)) if args.mix_state is None else bool(args.mix_state)
    grid = [float(s) for s in sigma_schedule(args.steps)]

    def clock(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - start

    def dense(z):
        x, generator = z.clone(), torch.Generator(dev).manual_seed(1)
        with use_networks(net, None, gen=False, fake=False):
            for s, s_next in zip(grid[:-1], grid[1:]):
                x0 = x + s * predict(dit, x, 1.0 - s, text, audio_generator=generator).float()
                x = (s_next / s) * x + (1.0 - s_next / s) * x0
        return x

    def dir_pass(z):
        estimate, state = torch.zeros_like(z), z
        with use_networks(net, None, gen=True, fake=False):
            for r in range(R):
                estimate = generator_step(dit, head, state, estimate, r, text, sigma=sigmas[r],
                                          audio_generator=torch.Generator(dev).manual_seed(1))
                if r + 1 < R:
                    state = next_state(state, estimate, sigmas[r], sigmas[r + 1], z if mix else None)
        return estimate

    result = {"gpu": torch.cuda.get_device_name(0), "swap": args.swap, "run": None if args.run is None else str(args.run),
              "tag": args.tag, "rank": rank, "sigmas": sigmas, "mix_state": mix, "steps": args.steps, "iterations": R,
              "reps": args.reps, "canvas": {}}
    with torch.no_grad():
        for canvas in args.canvas or ["768x768"]:
            width, height = (int(v) for v in canvas.lower().split("x"))
            z = torch.randn(1, 24, 1, height // 16, width // 16, device=dev)
            with use_networks(net, None, gen=False, fake=False):
                predict(dit, z, 0.4, text)                                  # warm-up: kernels for this shape
                base_ms = [1000 * clock(lambda: predict(dit, z, 0.4, text)) for _ in range(args.reps)]
            with use_networks(net, None, gen=True, fake=False):
                generator_step(dit, head, z, z, 1, text, sigma=sigmas[1])
                dir_ms = [1000 * clock(lambda: generator_step(dit, head, z, z, 1, text, sigma=sigmas[1]))
                          for _ in range(args.reps)]
            torch.cuda.reset_peak_memory_stats()
            dir_s = clock(lambda: dir_pass(z))
            dir_peak = (torch.cuda.max_memory_allocated() >> 20, torch.cuda.max_memory_reserved() >> 20)
            torch.cuda.reset_peak_memory_stats()
            dense_s = clock(lambda: dense(z))
            dense_peak = (torch.cuda.max_memory_allocated() >> 20, torch.cuda.max_memory_reserved() >> 20)
            row = {"tokens": (height // 32) * (width // 32),
                   "forward_ms": {"base": round(statistics.median(base_ms), 1), "dir": round(statistics.median(dir_ms), 1)},
                   "sample_s": {f"dense{args.steps}": round(dense_s, 2), f"dir{R}": round(dir_s, 2),
                                "speedup": round(dense_s / dir_s, 2)},
                   "peak_mb": {"dir_allocated": dir_peak[0], "dir_reserved": dir_peak[1],
                               "dense_allocated": dense_peak[0], "dense_reserved": dense_peak[1]}}
            result["canvas"][canvas] = row
            print(f"LOT_H3 kind=bench_dir canvas={canvas} {json.dumps(row)}", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
    print(f"LOT_H3 kind=bench_dir_done gpu={result['gpu']!r} swap={args.swap} rank={rank} ok=1", flush=True)


if __name__ == "__main__":
    main()
