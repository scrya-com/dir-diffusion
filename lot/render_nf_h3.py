#!/usr/bin/env python3
"""Stream a video block by block with Neighbor Forcing and ConvKV Memory (``nf_h3``, arXiv 2603.11746).

    python3 lot/render_nf_h3.py --untrained --blocks 3 --size 256 256      # base H3 under the causal mask
    python3 lot/render_nf_h3.py --run lot/runs/nf_stage2 --blocks 8         # a trained run, 62 latent frames

The inference pipeline of the paper's fig. 3 (right): each block of latent frames (6, then 8) is
denoised from noise, reading the keys and values that earlier blocks wrote at the SAME denoising step.
With ``--memory`` (the default) the context from the third block on is 2 reference frames, 2 ConvKV
long-term slots and 2 short-term frames, so the cache does not grow with the length.

``--untrained`` is the base H3 with no LoRA and the mean-initialised ConvKV (the paper's fig. 1
zero-shot setting, plus an untrained memory). ``--run`` loads ``lora_gen`` and, for a Stage 2 run,
``convkv`` and its 3-step schedule; for a DiR-generator run (``train_nf_h3.py --dir-init``) also its
``DirHead``, and each block then runs the run's R DiR iterations. ``--dir-run`` is the zero-shot arm of
that: a ``train_dir_h3.py`` run rendered under the causal mask as it is.

Prompts and reference frames come from the held-out clips of the run's cache (text embedding, and the
clip's first latent frame as the FL2VA ``first`` condition; ``--ref none`` drops it). ``--audio driving``
also feeds the clip's clean audio latent as conditioning (it covers 22 latent frames, so 3 blocks).

Steps: ``--steps 0`` (default) uses the few-step schedule (``1, 0.75, 0.5`` under H3's shift, the
release's re-noising update); ``--steps N`` uses H3's own N-step schedule with Euler steps. One
key/value cache is kept per step: at 512x768 that is about 7.7 GB per step, so ``--cache-device cpu``
(the default) keeps them in host memory.

Decoding: H3's VAE decodes whole 5-latent-frame clips, so the latent is cut to the longest ``5n + 2``
frames before decoding. Writes ``p{i}.mp4`` (24 fps), ``p{i}_sheet.png`` and ``meta.json`` to ``--out``
(default ``lot/runs/render_nf``, gitignored).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA init

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_CHECKPOINT, LOT_H3_CLIPS_CACHE, LOT_H3_VAE  # noqa: E402

from gpu_guard import refuse_if_busy  # noqa: E402

FPS = 24


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="a train_nf_h3.py run")
    source.add_argument("--untrained", action="store_true", help="base H3, no LoRA, mean-initialised ConvKV")
    source.add_argument("--dir-run", type=Path,
                        help="a train_dir_h3.py (DiR) run, or a Hugging Face repo id with lora_gen.safetensors + dir_head.pt + "
                             "config.json (johndpope/dir-diffusion-h3), rendered under the causal mask with no further training: "
                             "its generator LoRA + DirHead, R iterations per block, mean-initialised ConvKV")
    parser.add_argument("--dir-ema", action="store_true", help="--dir-run: load the *_ema generator files")
    parser.add_argument("--tag", default="", help="checkpoint suffix, e.g. _200 (default: final)")
    parser.add_argument("--blocks", type=int, default=3, help="blocks to stream (6 + 8 (N - 1) latent frames)")
    parser.add_argument("--text-reads-first", action="store_true",
                        help="the text prefix also reads chunk 1 (default: whatever the run was trained with)")
    parser.add_argument("--memory", action=argparse.BooleanOptionalAction, default=True,
                        help="ConvKV context from the third block on (--no-memory: every earlier frame, Stage 1)")
    parser.add_argument("--steps", type=int, default=0, help="0 = the 3-step schedule; N = H3's N-step Euler schedule")
    parser.add_argument("--ref", choices=["first", "none"], default=None,
                        help="default: first, or what the run was trained with; none for a DiR generator (text-to-video)")
    parser.add_argument("--holdout", type=int, default=None, help="held-out split size (default: the run's, else 8)")
    parser.add_argument("--audio", choices=["silent", "driving"], default=None,
                        help="driving: the held-out clip's clean H3 audio latent as conditioning (covers 3 blocks); "
                             "default: what the run was trained with, else silent")
    parser.add_argument("--prompts", type=int, default=1, help="held-out clips to render")
    parser.add_argument("--text-file", type=Path, nargs="+", default=None,
                        help="*_minimaxh3_te.safetensors prompt embeddings used in place of each held-out clip's text: "
                             "one per block, the last repeating (one file = the same prompt for every block)")
    parser.add_argument("--cache", type=Path, default=None, help="clip cache (default: the run's, else LOT_H3_CLIPS_CACHE)")
    parser.add_argument("--size", type=int, nargs=2, default=[512, 768], metavar=("W", "H"), help="pixels, multiples of 32")
    parser.add_argument("--cache-device", default="cpu", help="where the per-step key/value caches live (cpu or cuda)")
    parser.add_argument("--no-decode", action="store_true", help="skip the VAE (latents only: finite check and timing)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap", type=int, default=0, help="blocks streamed from CPU (0 = all resident)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "render_nf")
    args = parser.parse_args()

    refuse_if_busy("render_nf_h3.py")
    import train_h3
    from dir_h3 import use_networks
    from h3_musubi import load_dit, load_vae, sigma_schedule
    from nf_h3 import (ConvKV, NFConfig, NoiseBank, few_step_sigmas, load_nf_dir_head, load_nf_run,
                       nf_dir_stream_sample, nf_stream_sample)
    from train_nf_h3 import list_clips, load_audio, load_clip

    width, height = args.size
    if width % 32 or height % 32:
        raise SystemExit("--size must be multiples of 32")
    lat_h, lat_w = height // 16, width // 16
    model = load_dit(LOT_H3_CHECKPOINT, swap=args.swap)
    network, convkv, meta, head = None, None, {}, None
    if args.run is not None:
        network, convkv, meta = load_nf_run(args.run, model, tag=args.tag)
        if meta.get("dir"):
            head = load_nf_dir_head(args.run, meta, tag=args.tag)
    elif args.dir_run is not None:
        from dir_h3 import load_dir_run

        dir_run = args.dir_run
        if not dir_run.is_dir():        # a Hugging Face repo id, e.g. johndpope/dir-diffusion-h3 (no tag, no fake model)
            from huggingface_hub import snapshot_download

            dir_run = Path(snapshot_download(str(args.dir_run), allow_patterns=["lora_gen.safetensors", "dir_head.pt",
                                                                                "config.json"]))
        head, network, dir_meta = load_dir_run(dir_run, model, tag=args.tag, ema=args.dir_ema)
        meta = {"dir": True, "sigmas": dir_meta["sigmas"], "mix_state": dir_meta.get("mix_state", False), "ref": "none"}
    args.ref = args.ref or meta.get("ref") or "first"
    blocks = meta.get("blocks", [6, 8])
    config = NFConfig(blocks[0], blocks[1], memory=args.memory,
                      text_reads_first=bool(args.text_reads_first or meta.get("text_reads_first")))
    if args.memory and convkv is None:
        convkv = ConvKV.for_dit(model).cuda().eval()          # the mean: an untrained memory
    if head is not None:
        if args.steps:
            raise SystemExit("--steps does not apply to a DiR generator (R iterations from its config)")
        sigmas, sampler = [float(s) for s in meta["sigmas"]], "dir"
    elif args.steps:
        sigmas, sampler = [float(s) for s in sigma_schedule(args.steps)], "euler"
    else:
        sigmas, sampler = meta.get("sigmas") or few_step_sigmas(), meta.get("sampler", "renoise")

    cache = args.cache or (Path(meta["cache"]) if meta.get("cache") else LOT_H3_CLIPS_CACHE)
    train_h3.MAX_LATENT_HW, train_h3.CROP_MULTIPLE = (lat_h, lat_w), 2
    holdout = args.holdout if args.holdout is not None else int(meta.get("holdout", 8))
    _train, held = list_clips(cache, meta.get("sources", ["set_nk"]), 22, holdout)
    rng = random.Random(args.seed)
    results = []
    for index, item in enumerate(held[:args.prompts]):
        x0, text = load_clip(item, rng, "cuda", 22)
        if args.text_file is not None:
            from safetensors import safe_open

            texts = []
            for path in args.text_file:
                with safe_open(str(path), "pt") as handle:
                    hidden, mask = handle.get_tensor("hidden_states"), handle.get_tensor("attention_mask")
                texts.append(hidden[mask].unsqueeze(0).to(device=text.device, dtype=text.dtype))
            text = texts[0] if len(texts) == 1 else (lambda block, texts=texts: texts[min(block, len(texts) - 1)])
        if tuple(x0.shape[-2:]) != (lat_h, lat_w) and args.ref != "none":
            raise SystemExit(f"clip latent {tuple(x0.shape[-2:])} is smaller than --size {lat_h}x{lat_w} latent")
        first = x0[:, :, :1].contiguous() if args.ref == "first" else None
        audio_mode = args.audio or meta.get("audio", "silent")
        audio = load_audio(item, "cuda", 22) if audio_mode == "driving" else None
        block_ms, mark = [], [time.perf_counter()]

        def on_block(_index, _latent):
            torch.cuda.synchronize()
            now = time.perf_counter()
            block_ms.append(1000 * (now - mark[0]))
            mark[0] = now

        with torch.inference_mode(), use_networks(network, None, gen=True, fake=False):
            common = dict(lat_h=lat_h, lat_w=lat_w, config=config, sigmas=sigmas, noise=NoiseBank(args.seed + index, "cuda"),
                          convkv=convkv if args.memory else None, first=first, cache_device=args.cache_device,
                          on_block=on_block)
            if head is not None:
                latents = nf_dir_stream_sample(model, head, text, args.blocks, mix_state=bool(meta.get("mix_state")), **common)
            else:
                latents = nf_stream_sample(model, text, args.blocks, sampler=sampler, audio=audio, **common)
        finite = bool(torch.isfinite(latents).all())
        results.append({"clip": item[0].name, "latents": latents.float().cpu(), "block_ms": block_ms, "finite": finite,
                        "latent_frames": int(latents.shape[2]), "abs_mean": float(latents.float().abs().mean()),
                        "real_abs_mean": float(x0.abs().mean())})
        print(f"LOT_H3 kind=render_nf prompt={index} latent_frames={latents.shape[2]} "
              f"steps={len(sigmas) - (0 if head is not None else 1)} "
              f"sampler={sampler} memory={int(args.memory)} audio={audio_mode} block_ms={','.join(f'{ms:.0f}' for ms in block_ms)} "
              f"abs_mean={results[-1]['abs_mean']:.3f} real_abs_mean={results[-1]['real_abs_mean']:.3f} "
              f"finite={int(finite)} peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_decode:
        import imageio

        from render_stream_h3 import frames_of, sheet

        del model, network
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        decoder = load_vae(LOT_H3_VAE)
        dtype = next(decoder.parameters()).dtype
        for index, result in enumerate(results):
            total = result["latent_frames"]
            keep = total - (total - 2) % 5            # whole VAE clips: the longest 5n + 2
            with torch.inference_mode():
                pixels = decoder.decode(result["latents"][:, :, :keep].to("cuda", dtype))
            frames = frames_of(pixels)
            imageio.mimwrite(args.out / f"p{index}.mp4", list(frames), fps=FPS, quality=8, macro_block_size=16)
            sheet(frames).save(args.out / f"p{index}_sheet.png")
            result["decoded_latent_frames"], result["pixel_frames"] = keep, int(frames.shape[0])
    record = {"paper": "arXiv 2603.11746", "run": None if args.run is None else str(args.run), "tag": args.tag,
              "untrained": args.untrained, "dir_run": None if args.dir_run is None else str(args.dir_run), "dir_ema": args.dir_ema,
              "size": f"{width}x{height}", "blocks": args.blocks, "block_sizes": blocks,
              "memory": args.memory, "sigmas": sigmas, "sampler": sampler, "ref": args.ref, "seed": args.seed, "fps": FPS,
              "note": "block_ms is wall time per block (steps forwards of that block's rows; the first includes warm-up)",
              "prompts": [{k: v for k, v in r.items() if k != "latents"} for r in results]}
    (args.out / "meta.json").write_text(json.dumps(record, indent=2) + "\n")
    ok = int(all(r["finite"] for r in results))
    print(f"LOT_H3 kind=render_nf_done out={args.out} decoded={int(not args.no_decode)} ok={ok}", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
