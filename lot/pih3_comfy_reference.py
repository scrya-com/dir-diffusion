#!/usr/bin/env python3
"""Reference pi-H3 clips from the authors' ComfyUI path, run headless, with every tensor parity needs.

pi-H3 is Hansheng Chen, Shengqu Cai, Jiaming Song, Gordon Wetzstein, Xinchen Yan and Leonidas Guibas's 2/4-step
MiniMax H3 with DWD (https://hanshengchen.com/blog/pi-h3/). This script runs their ComfyUI-piFlow nodes exactly as
the ``pi-H3-DWD.json`` workflow wires them, with dense attention (the workflow's Sol node bypassed):

    UNETLoader(int8 ConvRot H3) -> Load LakonLab Adapter(pi_h3_{2,4}step_dwd_adapter_int8_convrot)
    -> ModelSamplingMiniMaxH3(12, 3) -> BasicGuider -> SamplerCustomAdvanced(euler, pi-Flow Scheduler sigmas)
    -> DWD Decode

The text is one of our cached ``_te`` prompt states (Qwen3-VL layer 50, text tags 1), fed in as CONDITIONING, so
our pipeline and this one see the same text without loading the 32B text encoder.

Needs a ComfyUI >= 0.45 tree with ComfyUI-piFlow in its ``custom_nodes`` (``LOT_PIH3_COMFY``, default
/media/2TB/ComfyUI-pih3 = v0.46.2) and its ``extra_model_paths.yaml`` naming the base checkpoint and the adapters.

    python3 pih3_comfy_reference.py --steps 2 4 --size 768x512 --frames 39 --prompts 3 --out OUT

Per clip, ``OUT/<W>x<H>_f<F>_s<steps>/p<i>.pt`` holds: ``noise_video`` ``noise_audio`` (what the sampler drew),
``sigmas``, ``calls`` (every DiT call: video and audio inputs at their own noise levels, timestep, raw outputs),
``h_last`` ``t_emb_last`` ``mod_segments_last`` (the last step's block output and modulation, what the DWD head
reads), ``pixels`` (DWD Decode, ``(F, H, W, 3)`` in [0, 1], float) and ``latent`` (the sampler's final video latent);
plus ``p<i>.mp4`` to look at and ``meta.json`` with the timings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

LOT = Path(__file__).resolve().parent
COMFY = Path(os.environ.get("LOT_PIH3_COMFY", "/media/2TB/ComfyUI-pih3"))
ADAPTER = "pi_h3_{}step_dwd_adapter_int8_convrot.safetensors"
CHECKPOINT = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
MAX_RAW_T = {2: 0.93, 4: 0.96}
FINAL_STEP_SIZE_SCALE = 0.6


def comfy_setup():
    """Import ComfyUI as a library from ``COMFY`` (its ``.pyoverlay`` first: comfy-kitchen 0.2.41 for v0.46)."""
    for path in (str(LOT), ""):
        while path in sys.path:
            sys.path.remove(path)
    sys.path[:0] = [str(COMFY / ".pyoverlay"), str(COMFY)]
    import folder_paths  # noqa: F401
    import nodes
    from utils.extra_config import load_extra_path_config

    load_extra_path_config(str(COMFY / "extra_model_paths.yaml"))
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
    sys.path.append(str(LOT))                      # our own modules last: ComfyUI's names win
    return nodes.NODE_CLASS_MAPPINGS


def call(mappings, name, **kwargs):
    cls = mappings[name]
    if hasattr(cls, "FUNCTION"):
        return getattr(cls(), cls.FUNCTION)(**kwargs)
    return cls.execute(**kwargs).result


class FixedNoise:
    """The sampler's noise, drawn once by ``RandomNoise`` and kept."""

    def __init__(self, noise, seed):
        self.noise, self.seed = noise, seed

    def generate_noise(self, _latent):
        import comfy.nested_tensor

        if getattr(self.noise, "is_nested", False):
            return comfy.nested_tensor.NestedTensor([t.clone() for t in self.noise.unbind()])
        return self.noise.clone()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, nargs="+", default=[2, 4], choices=[2, 4])
    p.add_argument("--size", default="768x512")
    p.add_argument("--frames", type=int, default=39)
    p.add_argument("--prompts", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="prompt i uses seed + i")
    p.add_argument("--holdout", type=int, default=8)
    p.add_argument("--no-dump", action="store_true", help="timings and pixels only (no per-call tensors)")
    p.add_argument("--dump-weights", default="",
                   help="diagnostic: comma-separated DiT layers whose patched weights (dequantized) go to weights.pt")
    p.add_argument("--no-adapter", action="store_true",
                   help="diagnostic: the base H3 on the same schedule (no adapter, no DWD); dumps the DiT calls only")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    mappings = comfy_setup()
    import torch
    from setproctitle import setproctitle

    from accel_h3 import default_caches, prompt_lists
    from gpu_guard import refuse_if_busy
    from train_dwd_h3 import load_text

    setproctitle(f"pih3:comfy_ref {args.size} f{args.frames} s{'+'.join(map(str, args.steps))}")
    refuse_if_busy("pih3_comfy_reference.py")
    import comfy.patcher_extension
    import comfy.utils
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent

    width, height = (int(v) for v in args.size.split("x"))
    _train, held = prompt_lists(default_caches(), args.holdout)
    prompts = held[:args.prompts]
    (base,) = call(mappings, "UNETLoader", unet_name=CHECKPOINT, weight_dtype="default")
    for steps in args.steps:
        if args.no_adapter:
            model = base
        else:
            (model,) = call(mappings, "Load LakonLab Adapter", model=base, adapter_name=ADAPTER.format(steps),
                            adapter_strength=1.0)
        # Local fix: the head's fp16 weights meet the DiT's bf16 tokens, and ComfyUI leaves weight casting off for
        # this additional model, so its LayerNorm fails on dtype. Cast the weights to the activations, as the
        # authors' diffusers pipeline (bf16 throughout) computes it.
        for head in model.get_additional_models_with_key("lakonlab_dwd"):
            for module in head.model.modules():
                if hasattr(module, "comfy_cast_weights"):
                    module.comfy_cast_weights = True
        (model,) = call(mappings, "MiniMaxH3SigmaShift", model=model, shift_video=12.0, shift_audio=3.0)
        (sigmas,) = call(mappings, "pi-Flow Scheduler", model=model, steps=steps, min_raw_t=0.0,
                         max_raw_t=MAX_RAW_T[steps], final_step_size_scale=FINAL_STEP_SIZE_SCALE)
        (sampler,) = call(mappings, "KSamplerSelect", sampler_name="euler")
        record: dict = {"calls": []}

        def diffusion_wrapper(executor, x, timestep, context, transformer_options, *a, **kw):
            out = executor(x, timestep, context, transformer_options, *a, **kw)
            if not args.no_dump:
                record["calls"].append({"video": x[0].detach().float().cpu(), "audio": x[1].detach().float().cpu(),
                                        "timestep": timestep.detach().float().cpu(),
                                        "out_video": out[0].detach().float().cpu(),
                                        "out_audio": out[1].detach().float().cpu()})
            return out

        model = model.clone()
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "pih3_ref", diffusion_wrapper)
        run_blocks = model.get_model_object("diffusion_model._run_blocks")

        def run_blocks_tap(h, t_emb, mod_segments, layout, transformer_options):
            out = run_blocks(h, t_emb, mod_segments, layout, transformer_options)
            if not args.no_dump:
                record["h_last"] = out.detach().cpu()          # the DWD wrapper decoded this very tensor
                record["t_emb_last"] = t_emb.detach().float().cpu()
                record["mod_segments_last"] = [tuple(int(v) for v in s) for s in mod_segments]
                record["layout_segments"] = [tuple(s) for s in layout.segments]
            return out

        model.add_object_patch("diffusion_model._run_blocks", run_blocks_tap)
        folder = args.out / (f"{width}x{height}_f{args.frames}_s{steps}" + ("_base" if args.no_adapter else ""))
        folder.mkdir(parents=True, exist_ok=True)
        meta = {"steps": steps, "size": [width, height], "frames": args.frames, "sigmas": sigmas.tolist(),
                "attention": "dense (ComfyUI optimized_attention)", "gpu": torch.cuda.get_device_name(),
                "comfyui": str(COMFY), "prompts": [], "seconds": [], "dwd_seconds": []}
        for index, te in enumerate(prompts):
            text = load_text(str(te))
            cond = [[text, {"minimax_token_tags": torch.ones(text.shape[1], dtype=torch.long)}]]
            latent, frame_count = _empty_av_latent(width, height, args.frames)
            (guider,) = call(mappings, "BasicGuider", model=model, conditioning=cond)
            seed = args.seed + index
            (noise,) = call(mappings, "RandomNoise", noise_seed=seed)
            drawn = noise.generate_noise(latent)
            record["calls"] = []
            torch.cuda.synchronize()
            start = time.perf_counter()
            (samples, _denoised) = call(mappings, "SamplerCustomAdvanced", noise=FixedNoise(drawn, seed),
                                        guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent)
            torch.cuda.synchronize()
            sampled = time.perf_counter()
            if args.no_adapter:
                pixels = torch.zeros(frame_count, height, width, 3)
            else:
                (pixels,) = call(mappings, "DWD Decode", samples=samples)
            seconds = sampled - start
            print(f"LOT_H3 kind=pih3_ref steps={steps} size={width}x{height} frames={frame_count} prompt={index} "
                  f"seed={seed} seconds={seconds:.2f} pixels={tuple(pixels.shape)}", flush=True)
            meta["prompts"].append(Path(te).name)
            meta["seconds"].append(round(seconds, 3))
            video_noise, audio_noise = drawn.unbind() if drawn.is_nested else (drawn, None)
            final = samples["samples"]
            video_latent = final.unbind()[0] if final.is_nested else final
            dump = {"prompt": Path(te).name, "seed": seed, "sigmas": sigmas.float().cpu(), "frames": frame_count,
                    "noise_video": video_noise.float().cpu(), "noise_audio": audio_noise.float().cpu(),
                    "pixels": pixels.float().cpu(), "latent": video_latent.float().cpu(),
                    **{k: v for k, v in record.items() if k != "calls"}, "calls": record["calls"]}
            torch.save(dump, folder / f"p{index}.pt")
            from two_track_h3 import save_clip

            clip = pixels.float().permute(3, 0, 1, 2).unsqueeze(0).mul(2).sub(1)       # (1, 3, F, H, W) in [-1, 1]
            if not args.no_adapter:
                save_clip(clip.half(), folder / f"p{index}.mp4")
        (folder / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        if args.dump_weights:                               # the weights ComfyUI actually ran, after patching
            weights = {}
            for layer in args.dump_weights.split(","):
                module = comfy.utils.get_attr(model.model.diffusion_model, layer)
                w = module.weight
                weights[layer] = (w.dequantize() if hasattr(w, "dequantize") else w).float().cpu()
                if module.bias is not None:
                    weights[layer + ".bias"] = module.bias.float().cpu()
            torch.save(weights, folder / "weights.pt")
    print("LOT_H3 kind=pih3_ref_done ok=1", flush=True)


if __name__ == "__main__":
    main()
