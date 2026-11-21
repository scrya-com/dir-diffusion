#!/usr/bin/env python3
# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""GPU parity: the DiR H3 Sampler inside ComfyUI against ``lot/dir_h3.dir_sample`` (musubi), same seed and text.

Two ~21 GB copies of H3 do not fit on one card, so each side runs in its own process and writes a ``.pt`` into
``--out``; ``compare`` reads them. ``all`` runs the five phases in order:

1. ``encode`` (ComfyUI's Python): the prompt through ComfyUI's MiniMax H3 text encoder -> ``text.pt``. Both sides
   then use exactly these Qwen3-VL states, so the text encoder is not part of the comparison.
2. ``musubi`` (``lot/``'s Python): ``load_dir_run`` + ``dir_sample`` -> every iteration's state and estimate; a
   zero-head ``dir_sample`` (how much the DiR head moves the result); and the floor step (zero head, no LoRA, sigma 1).
3. ``musubi-alt`` (``lot/``'s Python, ``LOT_H3_SPEED_PATCHES=0``): musubi against itself with a bf16-level change
   (the block RoPE's arithmetic), teacher-forced and end to end. This is the noise reference: with the generator
   LoRA on, the forward amplifies rounding about 7x (one forward at sigma 1 moves ~0.13 rel_rms, against ~0.017
   without the LoRA), and the DiR head's own per-iteration effect is of that size, so neither the no-LoRA floor nor
   the head effect can bound a correct port. A correct port differs from musubi about as much as musubi differs
   from itself.
4. ``comfy`` (ComfyUI's Python): UNETLoader + DiR H3 Loader + ``dir_runtime.sample`` (the DiR H3 Sampler's code)
   end to end; the same iterations teacher-forced on musubi's states and estimates (per-iteration error without
   compounding); and the floor step.
5. ``compare``: rel_rms per iteration against ``musubi-alt``'s (pass: each teacher-forced iteration and the final
   end-to-end estimate within ``NOISE_FACTOR`` x musubi's self-difference). The original criterion (teacher-forced
   <= 3 x the no-LoRA floor, final <= 0.25 x the head effect) is printed as ``ok_strict``; musubi fails it against
   itself. Exact equality of the hook and LoRA arithmetic is checked in fp32 by ``test_cpu.py`` (tiny H3).

    CUDA_VISIBLE_DEVICES=0 python3 tests/parity_gpu.py all --out /tmp/dir_parity \\
        --comfy-python /home/johndpope/miniconda3/envs/sdwebui/bin/python \\
        --musubi-python /home/johndpope/miniconda3/bin/python

Refuses while another LoT GPU job runs (``lot/gpu_guard.py``). Writes only into ``--out``. Prints one
``LOT_H3 kind=dir_comfy_parity ... ok=0|1`` line.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
LOT = Path(os.environ.get("DIR_DIFFUSION_LOT", PACK.parents[1] / "lot"))
COMFY = Path(os.environ.get("COMFYUI_DIR", "/media/2TB/ComfyUI"))
DEFAULT_PROMPT = ("Head-and-shoulders portrait of a woman in a plain grey studio, soft even light. She looks into the "
                  "camera and says quickly: \"Wait, what time is it? Did I miss the train?\" Her lips, jaw, eyes and head "
                  "move naturally with the words. Static locked-off camera, single continuous shot, no cut.")


def title(phase: str) -> None:
    try:
        from setproctitle import setproctitle

        setproctitle(f"dir:comfy-parity {phase}")
    except ImportError:
        pass


def guard() -> None:
    """``lot/gpu_guard.refuse_if_busy``, in the parent of ``all`` or a phase run on its own. The phases ``all`` starts
    skip it: the waiting parent is itself a registered GPU script and would make every child refuse."""
    if os.environ.get("DIR_PARITY_CHILD") == "1":
        return
    sys.path.insert(0, str(LOT))
    from gpu_guard import refuse_if_busy

    refuse_if_busy("parity_gpu.py")


def latent_t(frames: int) -> int:
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2      # ComfyUI's video_latent_t (lot: latent_frames)


def rel_rms(a, b) -> float:
    return float((a.float() - b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp(min=1e-12))


# ---------------------------------------------------------------------------------------------- comfy side

def comfy_setup():
    sys.path.insert(0, str(COMFY))
    os.chdir(COMFY)
    import comfy.cli_args  # noqa: F401
    import importlib.util

    spec = importlib.util.spec_from_file_location(PACK.name, PACK / "__init__.py", submodule_search_locations=[str(PACK)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return sys.modules[f"{PACK.name}.nodes"], sys.modules[f"{PACK.name}.dir_runtime"], sys.modules[f"{PACK.name}.vendor.dir_core"]


def phase_encode(args) -> None:
    title("encode")
    guard()
    comfy_setup()
    import torch
    import nodes

    clip = nodes.CLIPLoader().load_clip(args.clip, type="minimax", device="default")[0]
    with torch.inference_mode():
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
    text, extra = cond[0][0], cond[0][1]
    tags = extra.get("minimax_token_tags")
    torch.save({"text": text.float().cpu(), "tags": None if tags is None else tags.cpu(), "prompt": args.prompt},
               args.out / "text.pt")
    print(f"encode: text {tuple(text.shape)} {text.dtype}, tags {'none' if tags is None else 'all 1' if bool((tags == 1).all()) else 'MIXED'}")


def phase_comfy(args) -> None:
    title("comfy")
    guard()
    nodes_mod, runtime, core = comfy_setup()
    import torch

    import comfy.model_management
    import nodes
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent

    saved = torch.load(args.out / "text.pt", weights_only=False)
    ref = torch.load(args.out / "musubi.pt", weights_only=False)
    unet = nodes.UNETLoader().load_unet(args.unet, "default")[0]
    model, bundle = nodes_mod.DiRH3Loader.execute(unet, args.run, args.lora_mode, 1.0).result
    extra = {} if saved["tags"] is None else {"minimax_token_tags": saved["tags"]}
    positive = [[saved["text"], extra]]
    latent, _ = _empty_av_latent(args.width, args.height, args.frames)
    sigmas, mix = bundle.sigmas, bundle.mix_state
    device = comfy.model_management.get_torch_device()
    out = {}
    with torch.inference_mode():
        result = runtime.sample(model, bundle.head, positive, latent, seed=args.seed, sigmas=sigmas, mix_state=mix,
                                noise_device="gpu", disable_pbar=True)
        out["z"] = result["z"].float().cpu()
        out["estimates"] = [e.float().cpu() for e in result["estimates"]]
        out["latent"] = result["latent"]["samples"].unbind()[0].float().cpu()

        states, prevs = ref["states"], [torch.zeros_like(ref["states"][0])] + ref["estimates"][:-1]

        def forced(step, z, iters, sigmas_, mix_):
            estimates = []
            for r in range(iters):
                state = states[r].to(device)
                o = step(r, state, prevs[r].to(device), sigmas_[r])
                estimates.append(state.float() + sigmas_[r] * o.float())
            return estimates[-1], estimates, states[-1].to(device)

        tf = runtime.sample(model, bundle.head, positive, latent, seed=args.seed, sigmas=sigmas, mix_state=mix,
                            noise_device="gpu", disable_pbar=True, loop=forced)
        out["forced"] = [e.float().cpu() for e in tf["estimates"]]
        zero = core.DirHead(int(bundle.meta["hidden"]), int(bundle.meta["temb_dim"]), bundle.iters)
        floor = runtime.sample(unet, zero, positive, latent, seed=args.seed, sigmas=[1.0], mix_state=False,
                               noise_device="gpu", disable_pbar=True)
        out["floor"] = floor["estimate"].float().cpu()
    torch.save(out, args.out / "comfy.pt")
    print(f"comfy: {len(out['estimates'])} iterations, latent {tuple(out['latent'].shape)}, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


# --------------------------------------------------------------------------------------------- musubi side

def phase_musubi(args) -> None:
    title("musubi")
    guard()
    sys.path.insert(0, str(LOT))
    import torch

    from dir_h3 import DirHead, dir_sample, generator_step, load_dir_run, next_state, parse_sigmas, use_networks
    from h3_musubi import load_dit

    saved = torch.load(args.out / "text.pt", weights_only=False)
    if saved["tags"] is not None and not bool((saved["tags"] == 1).all()):
        raise SystemExit("ComfyUI's text tags are not all 1 (an image token in the prompt?); musubi uses all-ones tags")
    dev = torch.device("cuda")
    text = saved["text"].to(dev, torch.bfloat16)
    dit = load_dit(device=dev, swap=args.swap)
    run = args.run if Path(args.run).is_dir() else _hf_run(args.run)
    head, network, meta = load_dir_run(run, dit, device=dev)
    iters = int(meta["iters"])
    sigmas = parse_sigmas(meta.get("sigmas") or "1", iters)
    mix = bool(meta.get("mix_state", False))
    t = latent_t(args.frames)
    kw = dict(height=args.height, width=args.width, latent_t=t, iters=iters, seed=args.seed, sigmas=sigmas,
              mix_state=mix, device=dev)
    with torch.no_grad():
        final, estimates = dir_sample(dit, head, text, **kw)
        z = torch.randn(1, 24, t, args.height // 16, args.width // 16, device=dev,
                        generator=torch.Generator(dev).manual_seed(args.seed))
        states = [z]
        for r in range(iters - 1):
            states.append(next_state(states[-1], estimates[r], sigmas[r], sigmas[r + 1], z if mix else None))
        zero = DirHead(int(meta["hidden"]), int(meta["temb_dim"]), iters).to(dev)
        zero_final, _ = dir_sample(dit, zero, text, **kw)
        with use_networks(network, None, gen=False, fake=False):
            floor = generator_step(dit, zero, z, torch.zeros_like(z), 0, text, sigma=1.0,
                                   audio_generator=torch.Generator(dev).manual_seed(args.seed + 1))
    torch.save({"z": z.float().cpu(), "states": [s.float().cpu() for s in states],
                "estimates": [e.float().cpu() for e in estimates], "zero_head_final": zero_final.float().cpu(),
                "floor": floor.float().cpu(), "sigmas": sigmas, "mix_state": mix}, args.out / "musubi.pt")
    print(f"musubi: {iters} iterations, sigmas {sigmas}, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


def _hf_run(repo: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo, allow_patterns=["config.json", "lora_gen*.safetensors", "dir_head*.pt"])


def phase_musubi_alt(args) -> None:
    """musubi again with ``LOT_H3_SPEED_PATCHES=0`` (element-wise vs matmul RoPE: bf16 rounding only)."""
    title("musubi-alt")
    guard()
    os.environ["LOT_H3_SPEED_PATCHES"] = "0"
    sys.path.insert(0, str(LOT))
    import torch

    from dir_h3 import dir_sample, generator_step, load_dir_run
    from h3_musubi import load_dit

    saved = torch.load(args.out / "text.pt", weights_only=False)
    ref = torch.load(args.out / "musubi.pt", weights_only=False)
    dev = torch.device("cuda")
    text = saved["text"].to(dev, torch.bfloat16)
    dit = load_dit(device=dev, swap=args.swap)
    run = args.run if Path(args.run).is_dir() else _hf_run(args.run)
    head, _network, meta = load_dir_run(run, dit, device=dev)
    iters, sigmas, mix = int(meta["iters"]), ref["sigmas"], ref["mix_state"]
    prevs = [torch.zeros_like(ref["states"][0])] + ref["estimates"][:-1]
    with torch.no_grad():
        forced = [generator_step(dit, head, ref["states"][r].to(dev), prevs[r].to(dev), r, text, sigma=sigmas[r],
                                 audio_generator=torch.Generator(dev).manual_seed(args.seed + 1)).float().cpu()
                  for r in range(iters)]
        _final, estimates = dir_sample(dit, head, text, height=args.height, width=args.width,
                                       latent_t=latent_t(args.frames), iters=iters, seed=args.seed, sigmas=sigmas,
                                       mix_state=mix, device=dev)
    torch.save({"forced": forced, "estimates": [e.float().cpu() for e in estimates]}, args.out / "musubi_alt.pt")
    print(f"musubi-alt: {iters} iterations, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


# ------------------------------------------------------------------------------------------------ compare

NOISE_FACTOR = 1.5


def phase_compare(args) -> None:
    import torch

    m = torch.load(args.out / "musubi.pt", weights_only=False)
    c = torch.load(args.out / "comfy.pt", weights_only=False)
    floor = rel_rms(c["floor"], m["floor"])
    z_equal = bool(torch.equal(c["z"], m["z"]))
    forced = [rel_rms(a, b) for a, b in zip(c["forced"], m["estimates"])]
    e2e = [rel_rms(a, b) for a, b in zip(c["estimates"], m["estimates"])]
    final = e2e[-1]
    head_effect = rel_rms(m["zero_head_final"], m["estimates"][-1])
    latent_ok = rel_rms(c["latent"], c["estimates"][-1]) < 1e-3
    print(f"noise z identical: {z_equal}")
    print(f"floor (one forward at sigma 1, zero head, no LoRA, ComfyUI vs musubi): rel_rms {floor:.4f}")
    print("teacher-forced per iteration (same inputs each side): " + ", ".join(f"{e:.4f}" for e in forced))
    print("end to end per iteration:                             " + ", ".join(f"{e:.4f}" for e in e2e))
    print(f"musubi final with a zero DiR head vs trained head: rel_rms {head_effect:.4f} (what the hooks are worth)")
    tol_step = max(3.0 * floor, 0.02)
    ok_strict = z_equal and latent_ok and max(forced) <= tol_step and final <= 0.25 * head_effect
    alt = torch.load(args.out / "musubi_alt.pt", weights_only=False)
    self_forced = [rel_rms(a, b) for a, b in zip(alt["forced"], m["estimates"])]
    self_final = rel_rms(alt["estimates"][-1], m["estimates"][-1])
    print("musubi vs itself (bf16-level change), teacher-forced:  " + ", ".join(f"{e:.4f}" for e in self_forced))
    print(f"musubi vs itself, end to end final: {self_final:.4f}")
    ratios = [c_ / max(s_, 1e-12) for c_, s_ in zip(forced, self_forced)]
    ok = (z_equal and latent_ok and max(ratios) <= NOISE_FACTOR and final <= NOISE_FACTOR * self_final)
    print(f"LOT_H3 kind=dir_comfy_parity floor={floor:.4g} forced_max={max(forced):.4g} "
          f"self_forced_max={max(self_forced):.4g} forced_ratio_max={max(ratios):.3g} final={final:.4g} "
          f"self_final={self_final:.4g} head_effect={head_effect:.4g} z_equal={int(z_equal)} "
          f"ok_strict={int(ok_strict)} ok={int(ok)}")


# ------------------------------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["all", "encode", "musubi", "musubi-alt", "comfy", "compare"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", default=str(LOT / "runs" / "dir_turbo_clip"), help="run folder or HF repo id")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--frames", type=int, default=39, help="17k+5 pixel frames (39 -> 12 latent frames)")
    parser.add_argument("--unet", default="minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    parser.add_argument("--clip", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    parser.add_argument("--lora-mode", default="bypass (as trained)", help="DiR H3 Loader lora_mode")
    parser.add_argument("--swap", type=int, default=0, help="musubi block swap (0 fits a 32 GB card at 12x48x32)")
    parser.add_argument("--comfy-python", default=sys.executable)
    parser.add_argument("--musubi-python", default=sys.executable)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "all":
        guard()
        common = ["--out", str(args.out), "--run", args.run, "--prompt", args.prompt, "--seed", str(args.seed),
                  "--width", str(args.width), "--height", str(args.height), "--frames", str(args.frames),
                  "--unet", args.unet, "--clip", args.clip, "--lora-mode", args.lora_mode, "--swap", str(args.swap)]
        for phase, python in (("encode", args.comfy_python), ("musubi", args.musubi_python),
                              ("musubi-alt", args.musubi_python), ("comfy", args.comfy_python),
                              ("compare", sys.executable)):
            print(f"== {phase} ({python})", flush=True)
            subprocess.run([python, "-u", str(Path(__file__).resolve()), phase, *common], check=True,
                           env=dict(os.environ, DIR_PARITY_CHILD="1"))
        return
    {"encode": phase_encode, "musubi": phase_musubi, "musubi-alt": phase_musubi_alt, "comfy": phase_comfy,
     "compare": phase_compare}[args.phase](args)


if __name__ == "__main__":
    main()
