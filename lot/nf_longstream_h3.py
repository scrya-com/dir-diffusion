"""Long-stream Neighbor Forcing test: does the ConvKV memory hold a character over 16 chunks, and does Stage 2 help?

Streams (``nf_dir_stream_sample``, one KV cache per DiR iteration, about 12 GB at 512x512, kept on the GPU with
most DiT blocks streamed from CPU; with memory the cache is a fixed window, so it does not grow with length) every
(arm, clip, seed) at 512x512 (Stage 2's training size; 960x544 needs about 23 GB of cache and did not fit), ``--blocks`` chunks
(16 = 126 latent frames, about 21 s), and saves the clean latent. Arms:

  fix        DiR-Turbo (dir_turbo_clip_iter, step 100 EMA, the Stage 2 init), untrained mean ConvKV, text_reads_first
  nolong     as fix with ``drop_long``: only the reference and short-term frames are read (memory ablation)
  s2_<tag>   the Stage 2 run's checkpoint <tag> (its LoRA, DiR head and trained ConvKV), text_reads_first

usage:
  nf_longstream_h3.py sample OUT S2_RUN --arms fix nolong s2_200 s2_100 [--seeds 0 1 2] [--blocks 16]
  nf_longstream_h3.py decode OUT          (writes OUT/<arm>/<stem>_s<seed>.mp4 and OUT/chunks.json)

Sampling skips latents already on disk, so a killed run resumes.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

HELD = Path("/media/2TB/lora-data/h3_lot/cache_known_chars/held.json")
DIR_RUN = "/run/media/johndpope/2TB/lora-data/h3_lot/runs/dir_turbo_clip_iter"
# six held-out clips, six different characters (all had faces in every frame of the held-out test)
STEMS = ("MiniMax_H3_01080", "MiniMax_H3_00620", "MiniMax_H3_00653", "MiniMax_H3_01408", "MiniMax_H3_01449",
         "MiniMax_H3_01571")


def text_of(path):
    from safetensors import safe_open

    with safe_open(path, "pt") as h:
        hidden, mask = h.get_tensor("hidden_states"), h.get_tensor("attention_mask")
    return hidden[mask].unsqueeze(0).to("cuda", torch.bfloat16)


def sample(args) -> None:
    from dir_h3 import load_dir_run
    from h3_musubi import load_dit
    from lot_paths import LOT_H3_CHECKPOINT
    from nf_h3 import ConvKV, NFConfig, NoiseBank, load_nf_dir_head, load_nf_run, nf_dir_stream_sample

    W, H = args.size
    held = {item["stem"]: item for item in json.loads(HELD.read_text())}
    items = [held[stem] for stem in (args.stems or STEMS)]
    for arm in args.arms:
        todo = [(item, seed) for item in items for seed in args.seeds
                if not (args.out / arm / f"{item['stem']}_s{seed}.pt").is_file()]
        if not todo:
            continue
        dit = load_dit(LOT_H3_CHECKPOINT, swap=args.swap)
        if arm.startswith("s2_"):
            tag = "" if arm == "s2_final" else "_" + arm[3:]
            network, convkv, meta = load_nf_run(args.s2_run, dit, tag=tag)
            head = load_nf_dir_head(args.s2_run, meta, tag=tag)
            config = NFConfig(memory=True, text_reads_first=True)
        else:
            head, network, meta = load_dir_run(DIR_RUN, dit, tag="_100", ema=True)
            convkv = ConvKV.for_dit(dit).cuda().eval()
            config = NFConfig(memory=True, text_reads_first=True, drop_long=arm == "nolong")
        (args.out / arm).mkdir(parents=True, exist_ok=True)
        print(f"{arm}: model loaded gpu={torch.cuda.memory_allocated() / 2**30:.1f}G", flush=True)
        for item, seed in todo:
            t0 = time.time()
            times = []

            def on_block(index, _latent, start=t0):
                times.append(round(time.time() - start, 1))
                if index < 4 or index % 4 == 3:
                    print(f"  block {index} t={times[-1]}s gpu={torch.cuda.memory_allocated() / 2**30:.1f}G "
                          f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)

            with torch.inference_mode():
                x = nf_dir_stream_sample(dit, head, text_of(item["te"]), args.blocks, lat_h=H // 16, lat_w=W // 16,
                                         config=config, sigmas=meta["sigmas"], noise=NoiseBank(seed, "cuda"),
                                         mix_state=bool(meta.get("mix_state", False)), convkv=convkv,
                                         cache_device=args.cache_device, on_block=on_block)
            path = args.out / arm / f"{item['stem']}_s{seed}.pt"
            torch.save({"latent": x.to(torch.float16).cpu(), "character": item["character"], "seed": seed,
                        "config": repr(config), "block_times": times}, path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)
            print(f"sampled {arm} {item['stem']} s{seed} {item['character']} {time.time() - t0:.0f}s "
                  f"finite={bool(torch.isfinite(x).all())}", flush=True)
        del dit, head, convkv, network
        gc.collect()
        torch.cuda.empty_cache()
    print("LONGSTREAM_SAMPLED", flush=True)


def decode(args) -> None:
    import imageio

    from h3_musubi import load_vae, pixel_frames
    from lot_paths import LOT_H3_VAE
    from nf_h3 import NFConfig
    from render_stream_h3 import frames_of

    vae = load_vae(LOT_H3_VAE)
    dtype = next(vae.parameters()).dtype
    chunks = None
    for path in sorted(args.out.glob("*/*.pt")):
        mp4 = path.with_suffix(".mp4")
        x = torch.load(path)["latent"].float()
        frames = x.shape[2]
        if chunks is None:
            config = NFConfig(memory=True)
            chunks = [[pixel_frames(start), pixel_frames(stop)] for start, stop in config.spans(frames)]
            (args.out / "chunks.json").write_text(json.dumps({"latent_frames": frames, "pixel_chunks": chunks}))
        if mp4.is_file():
            continue
        keep = frames if (frames - 2) % 5 == 0 else frames - (frames - 2) % 5 + 5
        x = torch.cat([x, x[:, :, -1:].expand(-1, -1, keep - frames, -1, -1)], dim=2)
        with torch.inference_mode():
            video = frames_of(vae.decode(x.to("cuda", dtype)))[:pixel_frames(frames)]
        imageio.mimwrite(mp4, list(video), fps=24, quality=8, macro_block_size=16)
        print("decoded", path.parent.name, path.stem, video.shape, flush=True)
        torch.cuda.empty_cache()
    print("LONGSTREAM_DECODED", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("sample", "decode"))
    parser.add_argument("out", type=Path)
    parser.add_argument("s2_run", type=Path, nargs="?")
    parser.add_argument("--arms", nargs="+", default=["fix", "nolong", "s2_200", "s2_100"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--stems", nargs="+", help="clip stems (default: the six of STEMS)")
    parser.add_argument("--swap", type=int, default=30,
                        help="DiT blocks streamed from CPU (about 0.42 GB each), so the KV caches fit on the GPU")
    parser.add_argument("--cache-device", default="cuda",
                        help="where the per-iteration KV caches live (512x512: about 12 GB for 3 iterations)")
    parser.add_argument("--size", nargs=2, type=int, default=[512, 512], metavar=("W", "H"))
    args = parser.parse_args()
    if args.mode == "sample":
        if any(arm.startswith("s2_") for arm in args.arms) and args.s2_run is None:
            parser.error("s2 arms need S2_RUN")
        sample(args)
    else:
        decode(args)


if __name__ == "__main__":
    main()
