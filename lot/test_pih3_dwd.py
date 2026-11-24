#!/usr/bin/env python3
"""GPU parity test for ``pih3_dwd.refine_train`` against ComfyUI-piFlow's streamed ``SolH3DWDHead.refine``.

    python3 lot/test_pih3_dwd.py [--steps 2]

1. the refiner's weight type (int8 would mean no backward through int8 attention);
2. window coverage: every row gets exactly one attention output per layer;
3. a: ``refine`` (fused kernels) vs b: ``refine`` under ``eager()``: the kernel floor;
4. b vs c: ``refine_train``: must sit at or below that floor (the restructuring is exact);
5. a backward pass on a small grid; warm timing of ``refine`` at a fixed shape.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def psnr(a, b):
    mse = float(((a.float() - b.float()) / 2).pow(2).mean())
    return -10 * torch.log10(torch.tensor(max(mse, 1e-20))).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2)
    args = p.parse_args()
    from safetensors.torch import load_file
    from pih3_h3 import PiH3Head, adapter_path
    from pih3_dwd import build_plan, eager, refine_train

    sd = load_file(str(adapter_path(args.steps)))
    dwd = {k: v for k, v in sd.items() if k.startswith("dwd_")}
    del sd
    head = PiH3Head(dwd, 1e-6, device="cuda")
    ref = head.refiner
    w = ref.blocks[0].attn.to_qkv.weight
    print(f"PIH3DWD weight type={type(w).__name__} dtype={getattr(w, 'dtype', None)} layers={len(ref.blocks)}", flush=True)

    torch.manual_seed(0)
    grid = (12, 16, 24)                                   # 768x512, 39 frames
    h = torch.randn(grid[0] * grid[1] * grid[2], head.norm_w.shape[0] // 2, device="cuda")
    temb = torch.randn(head.norm_w.shape[1], device="cuda")
    feats = head.features(h, temb)[None]
    with torch.no_grad():
        x0, geo, shape = ref.expand(feats, grid)
        plan = build_plan(ref, geo, x0.device)
        n = x0.shape[0]
        for s, (wins, ids) in enumerate(zip(plan["windows"], plan["shifts"])):
            idx = torch.cat([rq.reshape(-1) for rq, _ in wins] + [g for g, _ in plan["globs"]])
            ok = idx.numel() == n and torch.equal(idx.sort().values, torch.arange(n, device=idx.device))
            print(f"PIH3DWD coverage shift={s} rows={n} outputs={idx.numel()} permutation={ok}", flush=True)
        a = ref.refine(x0.clone(), geo, shape)
        with eager():
            b = ref.refine(x0.clone(), geo, shape)
        c = refine_train(ref, x0.clone(), geo, shape, checkpoint=False)
    print(f"PIH3DWD shapes a={tuple(a.shape)} c={tuple(c.shape)}", flush=True)
    print(f"PIH3DWD fused_vs_eager rel={rel(a, b):.2e} psnr={psnr(a, b):.1f}", flush=True)
    print(f"PIH3DWD eager_vs_train rel={rel(c, b):.2e} psnr={psnr(c, b):.1f}", flush=True)
    print(f"PIH3DWD fused_vs_train rel={rel(c, a):.2e} psnr={psnr(c, a):.1f}", flush=True)

    # backward on a small grid
    small = (5, 8, 8)
    hs = torch.randn(small[0] * small[1] * small[2], h.shape[1], device="cuda")
    fs = head.features(hs, temb)[None]
    xs, gs, ss = ref.expand(fs, small)
    xs = xs.detach().clone().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    out = refine_train(ref, xs, gs, ss, checkpoint=True)
    out.float().square().mean().backward()
    print(f"PIH3DWD backward grad_norm={float(xs.grad.norm()):.3e} finite={bool(torch.isfinite(xs.grad).all())} "
          f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)

    # warm timing at the fixed 768x512x39 shape: expand (geometry) and refine separately
    with torch.no_grad():
        for _ in range(2):
            ref.refine(*ref.expand(feats, grid))
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            xg = ref.expand(feats, grid)
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(5):
            ref.refine(xg[0].clone(), xg[1], xg[2])
        torch.cuda.synchronize()
        t2 = time.time()
    print(f"PIH3DWD timing expand={(t1 - t0) / 5:.3f}s refine={(t2 - t1) / 5:.3f}s", flush=True)
    print("PIH3DWD_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
