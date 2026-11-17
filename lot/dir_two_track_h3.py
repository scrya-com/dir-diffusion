#!/usr/bin/env python3
"""DiR iterations inside MPEG-Diffusion's two-track sampler: a few-step generator at mixed resolution.

    python3 lot/dir_two_track_h3.py --run RUN --out DIR [--tag _50] [--frames 39] [--switch 2]
    python3 lot/dir_two_track_h3.py --run RUN --check              # GPU parity of the hooked forward and the sampler

``RUN`` is a ``train_dir_h3.py`` output directory, or a Hugging Face path ``owner/repo[/sub/folder]`` holding
the same files (``config.json``, ``lora_gen{tag}.safetensors``, ``dir_head{tag}.pt`` or ``.safetensors``).
The K/V maps and the region head default to johndpope/MiniMax-H3-MPEG-Diffusion.

Papers: DiR, arXiv 2610.12102 (Lin, Peng, Wang, Fan); MPEG-Diffusion, arXiv 2611.12069 (Menn et al.).
Neither paper combines the two. This is a demo of the combination, not a trained method: the generator
was distilled at one resolution, and the maps and region head were fitted on base H3.

Sampler, for a run with ``R`` iterations and sigmas ``s_0 .. s_{R-1}`` (from its ``config.json``):

1. Iterations ``0 .. k-1`` (``--switch k``) run the generator on the whole frame at half resolution,
   exactly as ``dir_h3.dir_sample`` does there.
2. After iteration ``k-1`` the region ``M`` is chosen on the LR token grid (region head, or ``--box``).
3. The LR estimate is upscaled to give the HR track its first estimate, and the HR state is
   ``s_k * z_hr + (1 - s_k) * U(x0_lr)`` with fresh noise ``z_hr`` that stays fixed from then on.
4. Iterations ``k .. R-1`` run one packed sequence, LR grid plus HR tokens on ``M``, through
   ``two_track_h3.two_track_forward``. Both tracks get the run's ``x0_proj`` of their own estimate and
   the same ``r`` embedding, and each follows the run's state law (``--mix-state`` or Euler).
5. The HR estimate is pasted over ``M`` on the upscaled LR estimate, as in ``two_track_sample``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_guard import auto_swap, refuse_if_busy  # noqa: E402
from lot_paths import LOT_H3_CHECKPOINT  # noqa: E402

MPEG_REPO = "johndpope/MiniMax-H3-MPEG-Diffusion"
MPEG_MAPS = "kv_maps/up_maps_tt.safetensors"
MPEG_REGION = "region_head_v3/region_head_tt.safetensors"
VARIANTS = ("base", "dir", "lr", "two_track")


# ------------------------------------------------------------------------- sampler


def video_extra(head, est_lr: torch.Tensor, est_hr: torch.Tensor | None = None, plan=None) -> torch.Tensor:
    """``x0_proj`` of the previous estimates in ``two_track_forward``'s row order: the LR grid, then
    the region's HR rows."""
    from h3 import patchify

    tokens = patchify(est_lr.float()).flatten(1, 3)
    if plan is not None:
        tokens = torch.cat([tokens, patchify(est_hr.float()).flatten(1, 3)[:, plan.hr_index]], dim=1)
    return head.x0_proj(tokens)


@torch.no_grad()
def dir_two_track_sample(dit, head, text, upscale, *, height: int, width: int, sigmas: list, mix_state: bool,
                         switch: int, seed: int, latent_t: int = 1, maps=None, region=None, box=None,
                         block: int = 25, alpha: float = 0.5, floor=None, feather: float = 4.0,
                         switch_upscale: str = "crop", final: str = "pixel"):
    """The sampler of the module docstring. Returns ``(latent, info)`` at ``width x height``.

    ``info`` has ``mask``, ``tokens`` (video tokens per forward), ``dit_s`` (seconds inside the DiT),
    ``lr`` (the final LR estimate) and, for ``final="pixel"``, ``weight`` for ``two_track_h3.pixel_compose``.
    ``switch == len(sigmas)`` never opens the region: the result is DiR at half resolution, upscaled.
    """
    from dir_h3 import next_state
    from mixed_res import TwoTrackPlan, region_mask
    from two_track_h3 import box_mask, crop_box, mask_to_latent, two_track_forward

    iters = len(sigmas)
    if not 1 <= switch <= iters:
        raise ValueError(f"switch {switch} is outside 1..{iters}")
    lat_h, lat_w = height // 32, width // 32            # LR latent; also the HR token grid
    tok_h, tok_w = lat_h // 2, lat_w // 2               # LR token grid
    info = {"tokens": [], "dit_s": 0.0}

    def forward(state_lr, state_hr, plan, r, extra, capture=None):
        torch.cuda.synchronize()
        start = time.perf_counter()
        # same silent-audio noise at every iteration, as dir_sample
        audio = torch.Generator("cuda").manual_seed(seed + 1)
        out = two_track_forward(dit, state_lr, state_hr, plan, 1.0 - sigmas[r], text, maps=maps,
                                audio_generator=audio, capture=capture, video_extra=extra,
                                video_temb_delta=head.r_embed[r])
        torch.cuda.synchronize()
        info["dit_s"] += time.perf_counter() - start
        return out

    z_lr = torch.randn(1, 24, latent_t, lat_h, lat_w, device="cuda", generator=torch.Generator("cuda").manual_seed(seed))
    est_lr, state_lr, states = torch.zeros_like(z_lr), z_lr, None
    for r in range(switch):
        capture = {"block": block} if region is not None and r == switch - 1 else None
        v_lr, _ = forward(state_lr, None, None, r, video_extra(head, est_lr), capture)
        if capture is not None:
            states = capture["states"]
        est_lr = state_lr.float() + sigmas[r] * v_lr.float()
        info["tokens"].append(tok_h * tok_w * latent_t)
        if r + 1 < iters:
            state_lr = next_state(state_lr, est_lr, sigmas[r], sigmas[r + 1], z_lr if mix_state else None)

    if region is not None:
        mask = region_mask(region(states).float().cpu().reshape(latent_t, tok_h, tok_w), alpha=alpha, floor=floor)
    else:
        mask = box_mask(box or (0.25, 0.25, 0.75, 0.75), latent_t, tok_h, tok_w)
    info.update(mask=mask, lr=est_lr)
    if switch == iters or not mask.any():
        for r in range(switch, iters):                  # no region: finish at LR
            v_lr, _ = forward(state_lr, None, None, r, video_extra(head, est_lr))
            est_lr = state_lr.float() + sigmas[r] * v_lr.float()
            info["tokens"].append(tok_h * tok_w * latent_t)
            if r + 1 < iters:
                state_lr = next_state(state_lr, est_lr, sigmas[r], sigmas[r + 1], z_lr if mix_state else None)
        info["lr"] = est_lr
        return upscale(est_lr), info

    plan = TwoTrackPlan(mask).to("cuda")
    # The HR track's first estimate: U(x0_lr). ``crop`` round-trips only the region's bounding box through
    # the VAE and fills the rest with a latent x2, which the HR track never reads.
    if switch_upscale == "vae":
        est_hr = upscale(est_lr)
    else:
        est_hr = F.interpolate(est_lr.float(), scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        bounds = crop_box(mask, lat_h, lat_w, margin=2) if switch_upscale == "crop" else None
        if bounds is not None:
            y0, y1, x0, x1 = bounds
            est_hr[..., 2 * y0:2 * y1, 2 * x0:2 * x1] = upscale(est_lr[..., y0:y1, x0:x1].contiguous())
    z_hr = torch.randn(1, 24, latent_t, 2 * lat_h, 2 * lat_w, device="cuda",
                       generator=torch.Generator("cuda").manual_seed(seed + 2))
    state_hr = sigmas[switch] * z_hr + (1.0 - sigmas[switch]) * est_hr
    for r in range(switch, iters):
        v_lr, v_hr = forward(state_lr, state_hr, plan, r, video_extra(head, est_lr, est_hr, plan))
        est_lr = state_lr.float() + sigmas[r] * v_lr.float()
        est_hr = state_hr.float() + sigmas[r] * v_hr.float()          # v_hr is zero outside the region
        info["tokens"].append(plan.lr_count + plan.hr_count)
        if r + 1 < iters:
            state_lr = next_state(state_lr, est_lr, sigmas[r], sigmas[r + 1], z_lr if mix_state else None)
            state_hr = next_state(state_hr, est_hr, sigmas[r], sigmas[r + 1], z_hr if mix_state else None)
    info["lr"] = est_lr
    weight = mask_to_latent(mask, 4, feather).to("cuda")
    if final == "pixel":
        # The caller decodes both tracks and pastes in pixels (pixel_compose). Outside the region the HR
        # latent is filled with a latent x2 of LR so the VAE sees a plausible neighbourhood at the edge.
        inside = mask_to_latent(mask, 4).to("cuda")
        context = F.interpolate(est_lr.float(), scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        info["weight"] = weight
        return inside * est_hr + (1.0 - inside) * context, info
    return weight * est_hr + (1.0 - weight) * upscale(est_lr), info


# ------------------------------------------------------------------------- loading


def resolve_run(spec: str) -> Path:
    """A local run directory, or ``owner/repo[/sub/folder]`` on Hugging Face (downloaded to the hub cache)."""
    path = Path(spec)
    if path.is_dir():
        return path
    parts = spec.strip("/").split("/")
    if len(parts) < 2:
        raise SystemExit(f"--run {spec}: not a directory and not owner/repo[/sub/folder]")
    from huggingface_hub import snapshot_download

    repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
    root = Path(snapshot_download(repo, allow_patterns=[f"{sub}/*"] if sub else None))
    return root / sub if sub else root


def load_generator(run: Path, dit, tag: str = "", ema: bool = False):
    """``(head, network, config)``: the generator LoRA applied to ``dit`` and its ``DirHead`` on the GPU."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    from dir_h3 import DirHead, probe_temb_dim
    from h3_musubi import make_lora

    config = json.loads((run / "config.json").read_text())
    sfx = tag + ("_ema" if ema else "")
    lora = run / f"lora_gen{sfx}.safetensors"
    if not lora.is_file():
        raise SystemExit(f"{lora} not found (saved tags: "
                         f"{sorted(p.name[8:-12] for p in run.glob('lora_gen*.safetensors'))})")
    with safe_open(str(lora), "pt") as handle:
        key = next(k for k in handle.keys() if k.endswith("lora_down.weight"))
        rank = int(handle.get_slice(key).get_shape()[0])
    network = make_lora(dit, rank)
    missing = network.load_weights(str(lora)).missing_keys
    if missing:
        raise RuntimeError(f"{lora}: missing {len(missing)} LoRA keys")
    network.requires_grad_(False)
    network.to(device="cuda", dtype=torch.bfloat16).eval()
    head = DirHead(dit.video_patch_proj.out_features, probe_temb_dim(dit), int(config.get("iters", 4)))
    published = run / f"dir_head{sfx}.safetensors"
    state = load_file(str(published)) if published.is_file() else \
        torch.load(run / f"dir_head{sfx}.pt", map_location="cpu", weights_only=True)
    head.load_state_dict(state)
    return head.cuda().eval().requires_grad_(False), network, config


def mpeg_file(value: str | None, env: str, default: str) -> Path | None:
    """``--maps`` / ``--region``: a path, ``none``, or (unset) ``$env``, else the published file."""
    value = value or os.environ.get(env)
    if value == "none":
        return None
    if value:
        return Path(value)
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(MPEG_REPO, default))


def load_prompts(args, config: dict):
    from render_h3 import load_prompts as cached
    from train_h3 import CACHE, list_items

    if args.te:
        return cached(args.prompts, list(args.te))
    cache = Path(config["cache"][0]) if config.get("cache") else CACHE
    held = [te for _latent, te in list_items(cache, int(config.get("holdout", 24)))[1]]
    if args.pick:
        held = [held[int(index)] for index in args.pick.split(",")]
    return cached(args.prompts, held)


def schedule_of(args, config: dict):
    from dir_h3 import parse_sigmas

    if str(config.get("schedule") or "dense") != "dense":
        raise SystemExit("this demo runs dense iterations; the run has a LoT layout schedule")
    sigmas = parse_sigmas(config.get("sigmas") or "1", int(config.get("iters", 4)))
    mix_state = bool(config.get("mix_state", False))
    switch = args.switch if args.switch is not None else max(1, len(sigmas) - 2)
    return sigmas, mix_state, switch


# -------------------------------------------------------------------------- render


@torch.no_grad()
def render(args) -> None:
    from PIL import Image, ImageDraw

    from dir_h3 import dir_sample, use_networks
    from h3_musubi import latent_frames, load_dit, sample
    from mixed_res import RegionHead
    from render_h3 import to_image
    from two_track_h3 import load_maps, load_region, load_upscaler, outline, pixel_compose, save_clip

    run = resolve_run(args.run)
    latent_t = latent_frames(args.frames)
    dit = load_dit(LOT_H3_CHECKPOINT, device="cuda", swap=auto_swap(args.swap, inference=True, latent_t=latent_t))
    head, network, config = load_generator(run, dit, args.tag, args.ema)
    sigmas, mix_state, switch = schedule_of(args, config)
    upscale = load_upscaler()
    maps_path = mpeg_file(args.maps, "LOT_TT_MAPS", MPEG_MAPS)
    maps = load_maps(maps_path) if maps_path else None
    region = block = None
    if args.box is None:
        region_path = mpeg_file(args.region, "LOT_TT_REGION", MPEG_REGION)
        saved = load_region(region_path)
        region = RegionHead(saved["hidden"]).cuda().eval()
        region.load_state_dict(saved["state"])
        block = saved["block"]
    width, height = args.size
    columns = [name for name in VARIANTS if name in args.variants.split(",")]
    meta = {"run": args.run, "tag": args.tag, "ema": args.ema, "sigmas": sigmas, "mix_state": mix_state, "switch": switch,
            "canvas": f"{width}x{height}", "frames": args.frames, "latent_t": latent_t, "base_steps": args.steps,
            "maps": str(maps_path), "region": "box" if args.box else str(region_path), "box": args.box,
            "switch_upscale": args.switch_upscale, "final": args.final, "columns": columns, "prompts": [],
            "seconds": {c: [] for c in columns}, "dit_seconds": {c: [] for c in columns},
            "tokens": {c: [] for c in columns}, "hr_share": [],
            "note": "seconds include the VAE decode; dit_seconds is the time inside DiT forwards"}
    args.out.mkdir(parents=True, exist_ok=True)
    full, half = (height // 32) * (width // 32) * latent_t, (height // 64) * (width // 64) * latent_t
    rows = []
    for index, (caption, embeds) in enumerate(load_prompts(args, config)):
        embeds = embeds.to("cuda", torch.bfloat16)
        seed = args.seed + index
        meta["prompts"].append(caption)
        row = []
        for name in columns:
            mask = info = None
            torch.cuda.synchronize()
            start = time.perf_counter()
            with use_networks(network, None, gen=name != "base", fake=False):
                if name == "base":
                    latent = sample(dit, embeds, height=height, width=width, steps=args.steps, seed=seed, latent_t=latent_t)
                    tokens = [full] * args.steps
                elif name == "dir":
                    latent, _ = dir_sample(dit, head, embeds, height=height, width=width, latent_t=latent_t,
                                           iters=len(sigmas), seed=seed, sigmas=sigmas, mix_state=mix_state)
                    tokens = [full] * len(sigmas)
                else:
                    latent, info = dir_two_track_sample(
                        dit, head, embeds, upscale, height=height, width=width, sigmas=sigmas, mix_state=mix_state,
                        switch=len(sigmas) if name == "lr" else switch, seed=seed, latent_t=latent_t, maps=maps,
                        region=region if name == "two_track" else None, box=args.box, block=block or 25,
                        alpha=args.alpha, floor=args.floor, feather=args.feather,
                        switch_upscale=args.switch_upscale, final=args.final)
                    tokens = info["tokens"]
                    if name == "two_track":
                        mask = info["mask"]
                        meta["hr_share"].append(round(float(mask.float().mean()), 4))
            torch.cuda.synchronize()
            dit_s = info["dit_s"] if info is not None else time.perf_counter() - start
            if info is not None and "weight" in info:
                pixels = pixel_compose(upscale.vae, latent, info["lr"], info["weight"])
            else:
                pixels = upscale.vae.decode(latent.to("cuda", upscale.dtype)).clamp(-1, 1).to("cpu", torch.float16)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            meta["seconds"][name].append(round(seconds, 2))
            meta["dit_seconds"][name].append(round(dit_s, 2))
            meta["tokens"][name].append(tokens)
            if latent_t > 1:
                save_clip(pixels, args.out / f"p{index}_{name}.mp4")
                pixels = pixels[:, :, pixels.shape[2] // 2:pixels.shape[2] // 2 + 1]     # the grid shows the middle frame
            image = to_image(pixels)
            if mask is not None and args.outline:
                image = outline(image, mask[mask.shape[0] // 2:mask.shape[0] // 2 + 1])
            image.save(args.out / f"p{index}_{name}.png")
            row.append(image)
            print(f"LOT_H3 kind=dir_two_track_render prompt={index} variant={name} seconds={seconds:.2f} "
                  f"dit_seconds={dit_s:.2f} token_forwards={sum(tokens)}"
                  + (f" hr_share={meta['hr_share'][-1]}" if mask is not None else ""), flush=True)
        rows.append(row)
    cell = (width // 2, height // 2)
    sheet = Image.new("RGB", (cell[0] * len(columns), 28 + cell[1] * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    labels = {"base": f"base H3, {args.steps} steps", "dir": f"DiR, {len(sigmas)} iterations, full res",
              "lr": f"DiR, {len(sigmas)} iterations, half res, upscaled",
              "two_track": f"DiR two-track: {switch} LR + {len(sigmas) - switch} mixed"}
    for col, name in enumerate(columns):
        draw.text((col * cell[0] + 6, 8), labels[name], fill="black")
    for r, images in enumerate(rows):
        for col, image in enumerate(images):
            sheet.paste(image.resize(cell, Image.LANCZOS), (col * cell[0], 28 + r * cell[1]))
    sheet.save(args.out / "grid.jpg", quality=90)
    meta["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    meta["gpu"] = torch.cuda.get_device_name()
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"LOT_H3 kind=dir_two_track_render_done out={args.out} ok=1", flush=True)


# --------------------------------------------------------------------------- check


def rel_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    left, right = left.float(), right.float()
    return float((left - right).pow(2).mean().sqrt() / right.pow(2).mean().sqrt().clamp_min(1e-12))


@torch.no_grad()
def check(args) -> None:
    """GPU parity on a small clip. (a) With no region the hooked ``two_track_forward`` is ``forward_video``
    with the same hooks. (b) With a region, zero hooks change nothing. (c) The live hooks do move the HR
    rows. (d) ``switch = R`` reproduces ``dir_sample`` at half resolution. The floor is the gap between
    the two forwards with no hooks at all (they are separate implementations of the base model); the
    limit is max(1e-4, 2 x floor), and R times that for (d), which compounds R forwards."""
    from dir_h3 import dir_sample
    from h3_musubi import forward_video, load_dit
    from mixed_res import TwoTrackPlan
    from two_track_h3 import box_mask, two_track_forward

    run = resolve_run(args.run)
    dit = load_dit(LOT_H3_CHECKPOINT, device="cuda", swap=auto_swap(args.swap, inference=True, latent_t=3))
    head, _network, config = load_generator(run, dit, args.tag, args.ema)
    sigmas, mix_state, _switch = schedule_of(args, config)
    _caption, text = load_prompts(args, config)[0]
    text = text.to("cuda", torch.bfloat16)
    width, height, latent_t = 768, 512, 3
    gen = torch.Generator("cuda").manual_seed(args.seed)
    lr = torch.randn(1, 24, latent_t, height // 32, width // 32, device="cuda", generator=gen)
    hr = torch.randn(1, 24, latent_t, height // 16, width // 16, device="cuda", generator=gen)
    est_lr, est_hr = 0.5 * torch.randn_like(lr), 0.5 * torch.randn_like(hr)
    r, sigma = len(sigmas) - 1, sigmas[-1]
    delta = head.r_embed[r]

    def audio():
        return torch.Generator("cuda").manual_seed(args.seed + 1)

    extra = video_extra(head, est_lr)
    reference = forward_video(dit, lr, 1.0 - sigma, text, video_extra=extra, video_temb_delta=delta, audio_generator=audio())
    bare = two_track_forward(dit, lr, None, None, 1.0 - sigma, text, audio_generator=audio())[0]
    floor = rel_rms(bare, forward_video(dit, lr, 1.0 - sigma, text, audio_generator=audio()))
    limit = max(1e-4, 2.0 * floor)
    hooked, _ = two_track_forward(dit, lr, None, None, 1.0 - sigma, text, audio_generator=audio(),
                                  video_extra=extra, video_temb_delta=delta)
    gap_a = rel_rms(hooked, reference)
    moved_a = rel_rms(hooked, bare)

    plan = TwoTrackPlan(box_mask((0.25, 0.25, 0.75, 0.75), latent_t, height // 64, width // 64)).to("cuda")
    plain = two_track_forward(dit, lr, hr, plan, 1.0 - sigma, text, audio_generator=audio())
    zeros = two_track_forward(dit, lr, hr, plan, 1.0 - sigma, text, audio_generator=audio(),
                              video_extra=torch.zeros_like(video_extra(head, est_lr, est_hr, plan)),
                              video_temb_delta=torch.zeros_like(delta))
    gap_b = max(rel_rms(a, b) for a, b in zip(zeros, plain))
    live = two_track_forward(dit, lr, hr, plan, 1.0 - sigma, text, audio_generator=audio(),
                             video_extra=video_extra(head, est_lr, est_hr, plan), video_temb_delta=delta)
    moved_c = rel_rms(live[1], plain[1])

    _final, estimates = dir_sample(dit, head, text, height=height // 2, width=width // 2, latent_t=latent_t,
                                   iters=len(sigmas), seed=args.seed, sigmas=sigmas, mix_state=mix_state)
    _latent, info = dir_two_track_sample(dit, head, text, lambda x: x, height=height, width=width, sigmas=sigmas,
                                         mix_state=mix_state, switch=len(sigmas), seed=args.seed, latent_t=latent_t)
    gap_d = rel_rms(info["lr"], estimates[-1])
    ok = int(gap_a <= limit and gap_b <= limit and gap_d <= len(sigmas) * limit and moved_c > 10 * limit)
    print(f"LOT_H3 kind=dir_two_track_check run={args.run} tag={args.tag} floor={floor:.6g} a={gap_a:.6g} "
          f"hooks_move_lr={moved_a:.6g} b={gap_b:.6g} hooks_move_hr={moved_c:.6g} "
          f"d={gap_d:.6g} ok={ok}", flush=True)
    if not ok:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="train_dir_h3 run directory, or owner/repo[/sub/folder] on Hugging Face")
    parser.add_argument("--tag", default="", help="checkpoint suffix, e.g. _50")
    parser.add_argument("--ema", action="store_true", help="load the *_ema generator files")
    parser.add_argument("--check", action="store_true", help="GPU parity instead of a render")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--variants", default="dir,two_track", help=f"comma list of {','.join(VARIANTS)}")
    parser.add_argument("--switch", type=int, default=None,
                        help="iterations at half resolution before the region opens (default: R - 2)")
    parser.add_argument("--size", type=int, nargs=2, default=(1344, 768), metavar=("W", "H"))
    parser.add_argument("--frames", type=int, default=39, help="pixel frames (1 = a still)")
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--pick", default=None, help="comma list of held-out indices, e.g. 0,2")
    parser.add_argument("--te", type=Path, nargs="+", default=None,
                        help="*_minimaxh3_te.safetensors prompt embeddings (default: the run's held-out prompts)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20, help="steps of the base column")
    parser.add_argument("--swap", default="auto", help="DiT blocks streamed from CPU, or auto (by free memory)")
    parser.add_argument("--maps", default=None, help=f"up_maps_tt file, or none (default: $LOT_TT_MAPS, else {MPEG_REPO})")
    parser.add_argument("--region", default=None, help=f"region_head_tt file (default: $LOT_TT_REGION, else {MPEG_REPO})")
    parser.add_argument("--box", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                        help="a fixed region instead of the region head, as fractions of the frame")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--floor", type=float, default=None)
    parser.add_argument("--feather", type=float, default=4.0)
    parser.add_argument("--switch-upscale", choices=("vae", "crop", "latent"), default="crop")
    parser.add_argument("--final", choices=("latent", "pixel"), default="pixel")
    parser.add_argument("--outline", action="store_true", help="draw the region boundary on the two-track image")
    args = parser.parse_args()
    if args.size[0] % 64 or args.size[1] % 64:
        raise SystemExit("--size sides must be multiples of 64 (the LR latent needs whole 2x2 patches)")
    if not args.check and args.out is None:
        raise SystemExit("--out is required for a render")
    refuse_if_busy("dir_two_track_h3.py")
    check(args) if args.check else render(args)


if __name__ == "__main__":
    main()
