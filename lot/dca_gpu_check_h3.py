#!/usr/bin/env python3
"""GPU check of dca_h3 on the real H3 DiT: identity at init, added time, peak memory.

    cd lot && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 -u dca_gpu_check_h3.py \
        --sizes 512x288x12,768x512x12 --swap 16 --out /tmp/dca.json
    ... dca_gpu_check_h3.py --mix-only 4897,37700        # the mixes alone, no DiT

Sizes are ``WxHxT`` with ``T`` latent frames (12 = 39 pixel frames). Text is random states, as in ``time_lot_h3.py``:
this measures arithmetic, not pictures. One forward per measurement at t = 0.5; times are medians of
``--repeats`` warm runs with ``torch.cuda.synchronize``; peaks are ``torch.cuda.max_memory_allocated``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from setproctitle import setproctitle


def timed(fn, repeats: int):
    fn()                                                    # warm (kernel autotune, block streaming order)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return result, statistics.median(times), torch.cuda.max_memory_allocated() / 2**30


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max()) / float(b.float().abs().max())


def mix_only(tokens: list[int], k: int, repeats: int) -> list[dict]:
    """The four block mixes alone (general path, random parameters) on bf16 ``(1, tokens, 5376)`` tensors."""
    from dca_h3 import MIXES, Grn

    rows = []
    for count in tokens:
        grn = Grn(5376, 2 + k, len(MIXES)).cuda()
        with torch.no_grad():
            grn.beta.normal_(0, 0.01)
            grn.w.normal_(0, 0.01)
            stack = [(slot, torch.randn(1, count, 5376, device="cuda", dtype=torch.bfloat16)) for slot in range(2 + k)]
            stream = torch.randn(1, count, 5376, device="cuda", dtype=torch.bfloat16)
            _out, seconds, peak = timed(lambda: grn(stream, stack, True), repeats)
        entry_gb = count * 5376 * 2 / 2**30
        rows.append({"tokens": count, "k": k, "ms_per_block": 1e3 * seconds, "entry_gb": entry_gb,
                     "peak_gb": peak, "extra_peak_gb": peak - (3 + k) * entry_gb})
        print(f"DCA_H3 kind=mix_only tokens={count} k={k} ms_per_block={1e3 * seconds:.2f} "
              f"entry_gb={entry_gb:.3f} peak_gb={peak:.2f}", flush=True)
        del stack, stream, grn
        torch.cuda.empty_cache()
    return rows


def resident(dit, latent, text, args) -> dict:
    """Clean per-block times: the first ``args.resident`` blocks (on the card, never streamed) run as a
    stack of their own on the inputs a real forward gives the block stack. With block streaming in the loop
    the forward waits on host-to-card copies, which hide added GPU work."""
    import types

    import h3_musubi
    from dca_h3 import DcaH3, run_blocks_dca

    captured = {}
    plain_run = getattr(h3_musubi.run_blocks, "plain", h3_musubi.run_blocks)

    def capture(dit_, hidden, temb, segments, rotation, tap=None):
        captured.update(hidden=hidden, temb=temb, segments=segments, rotation=rotation)
        return plain_run(dit_, hidden, temb, segments, rotation, tap=tap)

    h3_musubi.run_blocks = capture
    try:
        with torch.no_grad():
            h3_musubi.forward_video(dit, latent, 0.5, text, audio_generator=torch.Generator("cuda").manual_seed(1))
    finally:
        h3_musubi.run_blocks = plain_run
    dit.prepare_block_swap_before_forward()
    torch.cuda.synchronize()
    on_card = [index for index, block in enumerate(dit.blocks)
               if all(p.device.type == "cuda" for p in block.parameters())
               and all(b.device.type == "cuda" for b in block.buffers())]
    blocks = [dit.blocks[index] for index in on_card[:args.resident]]
    count = len(blocks)
    if count < 8:
        raise SystemExit(f"only {count} blocks are on the card")
    sub = types.SimpleNamespace(blocks=blocks, blocks_to_swap=0, gradient_checkpointing=False, training=False,
                                video_patch_proj=dit.video_patch_proj)
    hidden, temb, segments, rotation = (captured[key] for key in ("hidden", "temb", "segments", "rotation"))
    row = {"blocks": count, "first": on_card[0], "last": on_card[count - 1], "tokens": hidden.shape[1]}
    with torch.no_grad():
        want, seconds, row["plain_peak_gb"] = timed(lambda: plain_run(sub, hidden, temb, segments, rotation), args.repeats)
        row["plain_ms_per_block"] = 1e3 * seconds / count
        dca = DcaH3(count, hidden.shape[-1], args.k).cuda()
        got, seconds, row["zero_peak_gb"] = timed(lambda: run_blocks_dca(sub, dca, hidden, temb, segments, rotation), args.repeats)
        row["zero_ms_per_block"], row["zero_bit_identical"] = 1e3 * seconds / count, bool(torch.equal(got, want))
        dca.force_general = True
        got, seconds, row["general_peak_gb"] = timed(lambda: run_blocks_dca(sub, dca, hidden, temb, segments, rotation), args.repeats)
        row["general_ms_per_block"], row["general_bit_identical"] = 1e3 * seconds / count, bool(torch.equal(got, want))
        dca.qkv_mode = "batched"
        got, seconds, _peak = timed(lambda: run_blocks_dca(sub, dca, hidden, temb, segments, rotation), args.repeats)
        row["general_batched_ms_per_block"], row["batched_bit_identical"] = 1e3 * seconds / count, bool(torch.equal(got, want))
        dca.qkv_mode = "auto"
        for scale in (1e-6, 1e-4):
            for param in dca.parameters():
                param.normal_(0.0, scale)
            dca.force_general = False
            got, seconds, _peak = timed(lambda: run_blocks_dca(sub, dca, hidden, temb, segments, rotation), args.repeats)
            row[f"active_{scale:g}_ms_per_block"] = 1e3 * seconds / count
            row[f"active_{scale:g}_rel_change"] = rel(got, want)
        row["stream_rms_in"] = float(hidden.float().square().mean().sqrt())
        row["stream_rms_out"] = float(want.float().square().mean().sqrt())
        row["stream_absmax_out"] = float(want.float().abs().max())
        # only the R mix active (Q, K, V stay fused): what the residual mix alone costs
        dca = DcaH3(count, hidden.shape[-1], args.k).cuda()
        for grn in dca.blocks:
            grn.beta[3].normal_(0.0, 1e-6)
        got, seconds, _peak = timed(lambda: run_blocks_dca(sub, dca, hidden, temb, segments, rotation), args.repeats)
        row["r_only_ms_per_block"] = 1e3 * seconds / count
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="512x288x12,768x512x12")
    parser.add_argument("--swap", type=int, default=16)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--text", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mix-only", default="")
    parser.add_argument("--resident", type=int, default=0,
                        help="time this many blocks that stay on the card, run directly: no streaming in the times")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    setproctitle(f"dca:check {args.mix_only or args.sizes} swap={args.swap} k={args.k}")
    result = {"gpu": torch.cuda.get_device_name(0), "k": args.k, "swap": args.swap, "rows": []}
    if "5090" in result["gpu"]:
        raise SystemExit("this is the RTX 5090: set CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0")
    if args.mix_only:
        result["rows"] = mix_only([int(v) for v in args.mix_only.split(",")], args.k, args.repeats)
        if args.out:
            args.out.write_text(json.dumps(result, indent=1))
        return

    from dca_h3 import DcaH3, install
    from gpu_guard import refuse_if_busy
    from h3_musubi import forward_video, load_dit

    refuse_if_busy("dca_gpu_check_h3.py")
    start = time.perf_counter()
    dit = load_dit(swap=args.swap)
    base_params = sum(p.numel() for p in dit.parameters())
    dca = DcaH3.for_dit(dit, k=args.k)
    result.update(base_params=base_params, dca_params=dca.extra_parameters(),
                  dca_param_fraction=dca.extra_parameters() / base_params, blocks=len(dit.blocks),
                  resident_gb=torch.cuda.memory_allocated() / 2**30)
    print(f"DCA_H3 kind=load gpu={result['gpu']!r} swap={args.swap} load_s={time.perf_counter() - start:.0f} "
          f"base_params={base_params} dca_params={dca.extra_parameters()} "
          f"fraction={100 * dca.extra_parameters() / base_params:.4f}% resident_gb={result['resident_gb']:.2f}", flush=True)

    for size in args.sizes.split(","):
        width, height, latent_t = (int(v) for v in size.split("x"))
        g = torch.Generator("cuda").manual_seed(0)
        latent = torch.randn(1, 24, latent_t, height // 16, width // 16, device="cuda", generator=g)
        text = torch.randn(1, args.text, dit.config.text_dim, device="cuda", generator=g).to(torch.bfloat16)
        if args.resident:
            row = resident(dit, latent, text, args)
            row["size"] = size
            result["rows"].append(row)
            print("DCA_H3 kind=resident " + " ".join(
                f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()), flush=True)
            if args.out:
                args.out.write_text(json.dumps(result, indent=1))
            continue

        def forward():
            out = {}
            velocity = forward_video(dit, latent, 0.5, text, audio_generator=torch.Generator("cuda").manual_seed(1), out=out)
            return velocity, out["hidden"]

        row = {"size": size}
        try:
            with torch.no_grad():
                install(dit, None)
                (plain_v, plain_h), row["plain_s"], row["plain_peak_gb"] = timed(forward, args.repeats)
                again_v, again_h = forward()
                row["plain_rerun_bit_identical"] = bool(torch.equal(again_v, plain_v) and torch.equal(again_h, plain_h))

                dca = DcaH3.for_dit(dit, k=args.k)
                install(dit, dca)
                (v, h), row["dca_zero_s"], row["dca_zero_peak_gb"] = timed(forward, args.repeats)
                row["tokens"] = dca.stats["tokens"]
                row["zero_bit_identical"] = bool(torch.equal(v, plain_v) and torch.equal(h, plain_h))
                row["zero_rel_hidden"], row["zero_rel_velocity"] = rel(h, plain_h), rel(v, plain_v)

                dca.force_general = True                    # mixes computed, Q/K/V projected separately, all weights 0
                (v, h), row["dca_general_zero_s"], row["dca_general_zero_peak_gb"] = timed(forward, args.repeats)
                row["general_zero_bit_identical"] = bool(torch.equal(v, plain_v) and torch.equal(h, plain_h))
                row["general_zero_rel_hidden"], row["general_zero_rel_velocity"] = rel(h, plain_h), rel(v, plain_v)
                row["history_gb"] = dca.stats["history_bytes"] / 2**30
                row["history_formula_gb"] = dca.history_bytes(row["tokens"]) / 2**30

                dca.qkv_mode = "batched"
                (bv, bh), row["dca_general_zero_batched_s"], _peak = timed(forward, args.repeats)
                row["general_zero_batched_rel_hidden"] = rel(bh, plain_h)
                dca.qkv_mode = "auto"

                for param in dca.parameters():              # a trained DCA would have nonzero weights everywhere
                    param.normal_(0.0, 1e-3)
                dca.force_general = False
                (v, h), row["dca_active_s"], row["dca_active_peak_gb"] = timed(forward, args.repeats)
                row["active_rel_hidden"] = rel(h, plain_h)
                row["active_finite"] = bool(torch.isfinite(v).all())

                dca.skip = frozenset(range(4, 50, 5))       # 50 -> 40 blocks; untrained, so this times, not judges
                (_v, _h), row["dca_active_40_blocks_s"], _peak = timed(forward, args.repeats)
                dca.skip = frozenset(i for i in range(50) if i % 10 in (2, 5, 8))       # 50 -> 35
                (_v, _h), row["dca_active_35_blocks_s"], _peak = timed(forward, args.repeats)
                install(dit, None)
                install(dit, DcaH3.for_dit(dit, k=args.k, skip=range(4, 50, 5)))
                (_v, _h), row["plain_40_blocks_s"], _peak = timed(forward, args.repeats)   # zero DCA: plain blocks
                install(dit, None)
        except torch.OutOfMemoryError as error:
            row["oom"] = str(error).splitlines()[0]
            install(dit, None)
            torch.cuda.empty_cache()
        result["rows"].append(row)
        print("DCA_H3 kind=check " + " ".join(
            f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()), flush=True)
        if args.out:
            args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
