#!/usr/bin/env python3
"""pi-H3 2-step DWD with sparse attention, on the authors' ComfyUI path: dense vs Spark-Attn vs Kijai's Sol-Attn,
with and without H3-Tiny's 8 blocks skipped, scored against dense pi-H3 4-step from the same noise.

pi-H3 is by Hansheng Chen, Shengqu Cai, Jiaming Song, Gordon Wetzstein, Xinchen Yan and Leonidas Guibas.
Spark-Attn: github.com/zechengtang/Spark-H3 (ComfyUI-Spark-H3 0.2.0, SM120 build). Kijai's Sol-Attn:
github.com/kijai/ComfyUI-SolAttn_triton. Both nodes are set as in ComfyUI-piFlow 1.4.2's own workflows
(``pi-H3-DWD-Spark-SM120.json``, ``pi-H3-DWD-KJ-Sol.json``).

    python3 pih3_sparse_bench.py --sizes 768x512 1344x768 --frames 39 124 --prompts 3 --out OUT

Per (size, frames): ``OUT/<W>x<H>_f<F>/results.json`` and ``sheet.jpg`` (prompt 1: dense 4-step, then every
variant); ``OUT/SPARSE.md`` holds the table. Times: sampling with the DWD decode inside the last step, pixels on the
GPU's host side as ComfyUI returns them, no text encoder or audio decode; medians over prompts after the first.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

H3_TINY_DROP = (6, 7, 8, 9, 10, 11, 13, 15)
SPARK = dict(mode="speed", steps=20, warmup_mode="warmup_ratio", warmup_ratio=0.0, warmup_steps=4,
             topk_mode="topk_ratio", topk_ratio=0.2, topk_blocks=228, dense_layers=0, min_tokens=12288, strict=True,
             dense_backend="inherit", tail_granularity="query", reblock_layout="q_reuse_k",
             sparse_scope="target-video", producer_chunk=16384, chunk_path="none")
KJ_SOL = dict(tau=1.0, start_percent=0.0, end_percent=1.0, min_tokens=0, int8_qk=False,
              sink_conditioning="exact_kv_and_rows", morton=False, morton_curve="2d_frame", int8_pv=False,
              verbose=False, use_tma=False, dense_blocks="0-1")
VARIANTS = [("dense", ""), ("dense", "h3tiny"), ("spark", ""), ("spark", "h3tiny"), ("kjsol", ""), ("kjsol", "h3tiny")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", nargs="+", default=["768x512", "1344x768"])
    p.add_argument("--frames", type=int, nargs="+", default=[39, 124])
    p.add_argument("--prompts", type=int, default=3, help="held-out prompts; the first is a warm-up for times")
    p.add_argument("--holdout", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from pih3_comfy_reference import ADAPTER, CHECKPOINT, FINAL_STEP_SIZE_SCALE, MAX_RAW_T, FixedNoise, call, \
        comfy_setup
    mappings = comfy_setup()
    import lpips
    import torch
    from setproctitle import setproctitle

    from accel_h3 import default_caches, lpips_clip, prompt_lists
    from pih3_cascade import hf_energy, refuse_if_card_busy, sheet
    from train_dwd_h3 import load_text
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent

    setproctitle("pih3:sparse bench")
    refuse_if_card_busy(4.0)
    _train, held = prompt_lists(default_caches(), args.holdout)
    prompts = held[:args.prompts]
    lp = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    (base,) = call(mappings, "UNETLoader", unet_name=CHECKPOINT, weight_dtype="default")

    def adapter(steps):
        (model,) = call(mappings, "Load LakonLab Adapter", model=base, adapter_name=ADAPTER.format(steps),
                        adapter_strength=1.0)
        for head in model.get_additional_models_with_key("lakonlab_dwd"):    # the dtype fix of the reference driver
            for module in head.model.modules():
                if hasattr(module, "comfy_cast_weights"):
                    module.comfy_cast_weights = True
        (model,) = call(mappings, "MiniMaxH3SigmaShift", model=model, shift_video=12.0, shift_audio=3.0)
        return model

    def with_attention(model, attention: str, drop: str):
        if attention == "spark":
            (model,) = call(mappings, "MiniMaxH3SparkAttentionSM120", model=model, **SPARK)
        elif attention == "kjsol":
            (model,) = call(mappings, "SolAttnPatch", model=model, **KJ_SOL)
        if drop:
            model = model.clone()
            for index in H3_TINY_DROP:                 # the block returns its input: ComfyUI's own replace patch
                model.set_model_patch_replace(lambda args, extra: {"img": args["img"]}, "dit", "double_block", index)
        return model

    models = {4: adapter(4), 2: adapter(2)}
    sigmas = {s: call(mappings, "pi-Flow Scheduler", model=models[s], steps=s, min_raw_t=0.0, max_raw_t=MAX_RAW_T[s],
                      final_step_size_scale=FINAL_STEP_SIZE_SCALE)[0] for s in (2, 4)}
    (sampler,) = call(mappings, "KSamplerSelect", sampler_name="euler")

    def generate(model, steps, text, width, height, frames, seed):
        cond = [[text, {"minimax_token_tags": torch.ones(text.shape[1], dtype=torch.long)}]]
        latent, _count = _empty_av_latent(width, height, frames)
        (guider,) = call(mappings, "BasicGuider", model=model, conditioning=cond)
        (noise,) = call(mappings, "RandomNoise", noise_seed=seed)
        drawn = noise.generate_noise(latent)
        torch.cuda.synchronize()
        start = time.perf_counter()
        (samples, _d) = call(mappings, "SamplerCustomAdvanced", noise=FixedNoise(drawn, seed), guider=guider,
                             sampler=sampler, sigmas=sigmas[steps], latent_image=latent)
        (pixels,) = call(mappings, "DWD Decode", samples=samples)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        return pixels.mul(255).round().clamp(0, 255).byte(), seconds          # (F, H, W, 3) uint8, CPU

    def to_clip(u8):
        return u8.cuda().permute(3, 0, 1, 2).float().div(127.5).sub(1)[None]

    def score(u8, ref_u8):
        a, b = to_clip(u8), to_clip(ref_u8)
        return {"lpips": lpips_clip(lp, a, b), "hf": hf_energy(a) / hf_energy(b),
                "mean_diff": float((a.mean(dim=(0, 2, 3, 4)) - b.mean(dim=(0, 2, 3, 4))).abs().mean()),
                "std_ratio": float(a.std(dim=(0, 2, 3, 4)).mean() / b.std(dim=(0, 2, 3, 4)).mean())}

    table = []
    for size in args.sizes:
        width, height = (int(v) for v in size.split("x"))
        for frames in args.frames:
            name = f"{width}x{height}_f{frames}"
            folder = args.out / name
            folder.mkdir(parents=True, exist_ok=True)
            rows = {"4-step dense": {"seconds": [], "oom": 0}}
            rows.update({f"2-step {a}{' +h3tiny' if d else ''}": {"seconds": [], "oom": 0, "vs4": [], "vs2": []}
                         for a, d in VARIANTS})
            sheet_clips = {}
            texts = [load_text(str(te)) for te in prompts]
            refs4, refs2 = {}, {}
            # variant-major: each model runs all its prompts back to back, so a ComfyUI reload (weights repatched
            # when the model changes) lands in the first prompt, which the time medians leave out
            for index, text in enumerate(texts):
                try:
                    refs4[index], s4 = generate(models[4], 4, text, width, height, frames, index)
                    rows["4-step dense"]["seconds"].append(s4)
                except torch.OutOfMemoryError:
                    rows["4-step dense"]["oom"] += 1
                    torch.cuda.empty_cache()
            for attention, drop in VARIANTS:
                label = f"2-step {attention}{' +h3tiny' if drop else ''}"
                model = with_attention(models[2], attention, drop)
                for index, text in enumerate(texts):
                    if index not in refs4:
                        continue
                    try:
                        u8, seconds = generate(model, 2, text, width, height, frames, index)
                    except torch.OutOfMemoryError:
                        rows[label]["oom"] += 1
                        torch.cuda.empty_cache()
                        continue
                    rows[label]["seconds"].append(seconds)
                    rows[label]["vs4"].append(score(u8, refs4[index]))
                    if attention == "dense" and not drop:
                        refs2[index] = u8
                    elif index in refs2:
                        rows[label]["vs2"].append(score(u8, refs2[index]))
                    if index == min(1, len(texts) - 1):
                        sheet_clips[label] = u8
                    print(f"LOT_H3 kind=pih3_sparse size={name} prompt={index} variant={label} seconds={seconds:.2f} "
                          + " ".join(f"{k}={v:.4f}" for k, v in rows[label]["vs4"][-1].items()), flush=True)
                del model
                torch.cuda.empty_cache()
            k = min(1, len(texts) - 1)
            if k in refs4:
                sheet({"4-step dense": refs4[k], **sheet_clips}, folder / "sheet.jpg", columns=4, tile_w=320)
            del refs4, refs2
            (folder / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
            for label, r in rows.items():
                times = r["seconds"][1:] or r["seconds"]
                row = {"case": name, "variant": label, "frames": frames,
                       "seconds": round(statistics.median(times), 2) if times else None, "oom": r["oom"]}
                for key in ("vs4", "vs2"):
                    for metric in ("lpips", "hf", "mean_diff", "std_ratio"):
                        vals = [d[metric] for d in r.get(key, [])]
                        row[f"{key}_{metric}"] = round(sum(vals) / len(vals), 4) if vals else None
                table.append(row)
                print("LOT_H3 kind=pih3_sparse_row " + json.dumps(row), flush=True)
            write_table(args.out, table)
    print("LOT_H3 kind=pih3_sparse_done ok=1", flush=True)


def write_table(out: Path, table: list) -> None:
    lines = ["| Case | Variant | s / clip | × real time | LPIPS vs 4-step | detail vs 4-step | contrast vs 4-step | "
             "colour shift vs 4-step | LPIPS vs 2-step dense | detail vs 2-step dense |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        rt = round(r["frames"] / 24.0 / r["seconds"], 2) if r["seconds"] else None
        cells = [r["case"], r["variant"] + (f" ({r['oom']} out of memory)" if r["oom"] else ""), r["seconds"], rt,
                 r.get("vs4_lpips"), r.get("vs4_hf"), r.get("vs4_std_ratio"), r.get("vs4_mean_diff"),
                 r.get("vs2_lpips"), r.get("vs2_hf")]
        lines.append("| " + " | ".join("" if c is None else str(c) for c in cells) + " |")
    (out / "SPARSE.md").write_text("\n".join(lines) + "\n\nMedians of the prompts after the first (times); means of "
                                   "all prompts (scores). Same noise for every row of a case.\n")


if __name__ == "__main__":
    main()
