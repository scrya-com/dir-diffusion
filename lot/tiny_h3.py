#!/usr/bin/env python3
"""H3-Tiny's block drop on our fast paths (pi-H3 and DiR), training-free: quality against the 50-block model and
per-block timings. No DCA, nothing trained.

H3-Tiny (https://github.com/fuyi25/MiniMax-H3-Tiny, code Apache-2.0, weights under the MiniMax H3 Community
License) removes blocks 6-11, 13 and 15 of the 50 (``h3_musubi.H3_TINY_DROP``). Here the same blocks are skipped
on our int8 ConvRot checkpoint; nothing of theirs is downloaded. Two deeper plans come from the per-block scan
branch ``exp/pih3-cascade`` already ran (``pih3_cascade.py dropscan``, pi-H3 4-step, 2 prompts).

    cd lot
    python3 -u tiny_h3.py quality --generator pih3 --steps 2 --size 768x512 --prompts 3 --swap 24 --out OUT
    python3 -u tiny_h3.py quality --generator dir --size 768x512 --prompts 3 --swap 24 --out OUT
    python3 -u tiny_h3.py resident --sizes 512x288x12,768x512x12 --swap 16 --out OUT/resident.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from h3_musubi import H3_TINY_DROP, blocks_dropped  # noqa: E402

#: Least-sensitive-first block order of ``pih3_cascade.py dropscan`` (branch exp/pih3-cascade; pi-H3 4-step,
#: 768x512, 2 held-out prompts; /media/2TB/lora-data/h3_lot/pih3/cascade/dropscan_s4/scan.json). The first 16.
SCAN_ORDER = (16, 17, 6, 4, 15, 8, 11, 13, 23, 19, 10, 47, 26, 20, 40, 46)
BLOCKS = 50


def plan(spec: str) -> tuple[int, ...]:
    """Blocks to skip: ``full`` (none), ``h3tiny`` (42 kept), ``leastK`` (the K least sensitive of ``SCAN_ORDER``;
    ``least10`` keeps 40, ``least15`` keeps 35), or indices joined by ``-``."""
    if spec in ("", "full"):
        return ()
    if spec == "h3tiny":
        return tuple(H3_TINY_DROP)
    if spec.startswith("least"):
        count = int(spec[5:])
        if not 0 < count <= len(SCAN_ORDER):
            raise ValueError(f"{spec}: the scan order has {len(SCAN_ORDER)} blocks")
        return tuple(sorted(SCAN_ORDER[:count]))
    indices = tuple(sorted({int(x) for x in spec.split("-")}))
    if indices and not 0 <= indices[0] <= indices[-1] < BLOCKS:
        raise ValueError(f"{spec}: block indices are 0..{BLOCKS - 1}")
    return indices


def kept(spec: str, blocks: int = BLOCKS) -> int:
    return blocks - len(plan(spec))


# --------------------------------------------------------------------------------------------- metrics

def hf_energy(pixels: torch.Tensor, every: int = 6) -> float:
    """Mean |luma - its 5x5 box blur| on every ``every``-th frame: detail (as ``pih3_cascade.hf_energy``)."""
    luma = pixels[0].float().mean(0)[::every].unsqueeze(1)
    return float((luma - F.avg_pool2d(luma, 5, 1, 2, count_include_pad=False)).abs().mean())


def colour_shift(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean over frames and channels of |mean colour of the frame, pred - target|, pixels in [-1, 1] rescaled to
    [0, 1]. 0.1 is a tenth of the range: a visible cast."""
    a, b = pred[0].float().mean(dim=(2, 3)), target[0].float().mean(dim=(2, 3))        # (3, F)
    return float((a - b).abs().mean() / 2.0)


def to_u8(pixels: torch.Tensor) -> torch.Tensor:
    """(1, 3, F, H, W) in [-1, 1] -> (F, H, W, 3) uint8 on the CPU."""
    return pixels[0].float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 3, 0).cpu()


def sheet(clips: dict, path: Path, columns: int, tile_w: int) -> None:
    """Contact sheet: one row per variant, ``columns`` frames spread over the clip, the label on the first tile."""
    from PIL import Image, ImageDraw

    rows = []
    for label, u8 in clips.items():
        count = u8.shape[0]
        picks = [count // 2] if columns == 1 else [round(k * (count - 1) / (columns - 1)) for k in range(columns)]
        row = None
        for k, f in enumerate(picks):
            image = Image.fromarray(u8[f].numpy())
            image = image.resize((tile_w, round(tile_w * image.height / image.width)), Image.LANCZOS)
            if k == 0:
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 8 + 7 * len(label), 16), fill=(0, 0, 0))
                draw.text((4, 2), label, fill=(255, 255, 0))
                row = Image.new("RGB", (tile_w * len(picks), image.height))
            row.paste(image, (k * tile_w, 0))
        rows.append(row)
    out = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
    top = 0
    for row in rows:
        out.paste(row, (0, top))
        top += row.height
    out.save(path, quality=90)


# --------------------------------------------------------------------------------------------- loading

def load_pih3_swap(steps: int, swap: int) -> dict:
    """pi-H3 with ``swap`` blocks streamed: the adapter is merged on the CPU, then blocks go to the card
    (as ``pih3_cascade.load`` on branch exp/pih3-cascade). ``swap=0`` is ``pih3_h3.load_pih3``."""
    from pih3_h3 import PiH3Head, adapter_path, apply_adapter, load_pih3, pih3_sigmas

    if not swap:
        return load_pih3(steps)
    from h3_musubi import load_dit
    from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig

    dit = load_dit(device="cpu", swap=0)
    dwd, counts = apply_adapter(dit, adapter_path(steps), "exact")       # musubi's quantizer runs on the CPU
    device = torch.device("cuda")
    dit.enable_block_swap(swap, BlockSwapConfig(device=device, supports_backward=False, h2d_only=True,
                                                use_pinned_memory=True))
    dit.move_to_device_except_swap_blocks(device)
    dit.prepare_block_swap_before_forward()
    head = PiH3Head(dwd, dit.final_layer.norm.eps, device=device)
    print(f"LOT_H3 kind=pih3_load swap={swap} " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
    return {"dit": dit, "head": head, "steps": steps, "sigmas": pih3_sigmas(steps),
            "adapter": str(adapter_path(steps)), "merge": "exact"}


def refuse_other_card() -> str:
    name = torch.cuda.get_device_name(0)
    if "5090" in name:
        raise SystemExit("this is the RTX 5090: set CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0")
    return name


# --------------------------------------------------------------------------------------------- quality

@torch.no_grad()
def quality(args) -> None:
    import lpips
    from setproctitle import setproctitle

    from accel_h3 import default_caches, lpips_clip, parse_job, prompt_lists, token_shape
    from gpu_guard import refuse_if_busy
    from train_dwd_h3 import load_text

    setproctitle(f"tiny:quality {args.generator} s{args.steps} {args.size} swap{args.swap}")
    gpu = refuse_other_card()
    refuse_if_busy("tiny_h3.py")
    width, height, _count, frames = parse_job(args.size, args.frames)
    latent_t, _th, _tw = token_shape(width, height, frames)
    plans = {name: plan(name) for name in args.plans.split(",")}
    if list(plans)[0] != "full":
        raise SystemExit("--plans must start with full (the reference)")
    args.out.mkdir(parents=True, exist_ok=True)
    vae = None
    if args.generator == "pih3":
        from pih3_h3 import audio_noise, decode, pih3_sample

        model = load_pih3_swap(args.steps, args.swap)
        dit = model["dit"]
    else:
        from accel_h3 import DEFAULT_RUN, dir_rollout
        from dir_h3 import parse_sigmas
        from dir_two_track_h3 import load_generator, resolve_run
        from h3_musubi import load_dit

        dit = load_dit(device="cuda", swap=args.swap)
        gen_head, _net, config = load_generator(resolve_run(args.run or DEFAULT_RUN), dit, "", False)
        iters = int(config.get("iters", 4))
        sigmas = parse_sigmas(config.get("sigmas") or "1", iters)
        mix = bool(config.get("mix_state", False))
    if args.generator == "dir" or args.vae:
        from h3_musubi import load_vae
        from lot_paths import LOT_H3_VAE

        vae = load_vae(LOT_H3_VAE, device="cuda")
    net = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    _train, held = prompt_lists(default_caches(), 8)
    rows = {name: {"blocks": list(indices), "kept": len(dit.blocks) - len(indices), "lpips_full": [], "colour": [],
                   "detail": [], "latent_rel": [], "lpips_vae": [], "seconds": []} for name, indices in plans.items()}
    rows["floor"] = {"blocks": [], "kept": len(dit.blocks), "lpips_full": [], "colour": [], "detail": [],
                     "latent_rel": [], "lpips_vae": [], "seconds": []}

    def render(indices, z, za, text, seed):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with blocks_dropped(dit, indices):
            if args.generator == "pih3":
                out = pih3_sample(model, text, z, za)
                latent = out["latent"]
                pixels = decode(model, out, tuple(z.shape)).float().clamp(-1, 1)
            else:
                latent = dir_rollout(dit, gen_head, text, z, sigmas, seed + 1, mix_state=mix)["estimate"]
                pixels = None
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        vae_pixels = None
        if vae is not None:
            vae_pixels = vae.decode(latent.to("cuda", next(vae.parameters()).dtype)).float().clamp(-1, 1)
        return latent.float(), (pixels if pixels is not None else vae_pixels), vae_pixels, seconds

    for index, te in enumerate(held[:args.prompts]):
        text = load_text(str(te)).cuda()
        seed = args.seed + index
        z = torch.randn(1, 24, latent_t, height // 16, width // 16, device="cuda",
                        generator=torch.Generator("cuda").manual_seed(seed))
        za = None
        if args.generator == "pih3":
            za = audio_noise(latent_t, seed + 1)
        clips, full = {}, None
        order = list(plans.items()) + [("floor", ())]
        for name, indices in order:
            # the floor: the same model from the same noise rounded to bf16 (how far a no-op change moves the clip)
            zz = z.bfloat16().float() if name == "floor" else z
            latent, pixels, vae_pixels, seconds = render(indices, zz, za, text, seed)
            row = rows[name]
            row["seconds"].append(round(seconds, 2))
            if name == "full":
                full = (latent, pixels, hf_energy(pixels))
            else:
                row["lpips_full"].append(round(lpips_clip(net, pixels, full[1]), 4))
                row["colour"].append(round(colour_shift(pixels, full[1]), 4))
                row["latent_rel"].append(round(float((latent - full[0]).norm() / full[0].norm()), 4))
            row["detail"].append(round(hf_energy(pixels) / full[2], 3))
            if args.generator == "pih3" and vae_pixels is not None:
                row["lpips_vae"].append(round(lpips_clip(net, pixels, vae_pixels), 4))
            clips[f"{name} ({row['kept']})"] = to_u8(pixels)
            print(f"LOT_H3 kind=tiny_quality gen={args.generator} prompt={index} plan={name} kept={row['kept']} "
                  + " ".join(f"{k}={v[-1]}" for k, v in row.items() if isinstance(v, list) and v and k != "blocks"),
                  flush=True)
            del latent, pixels, vae_pixels
        sheet(clips, args.out / f"{args.generator}_s{args.steps}_p{index}_strip.jpg", 4, width // 2)
        result = {"gpu": gpu, "generator": args.generator, "steps": args.steps, "size": args.size, "frames": frames,
                  "swap": args.swap, "prompts": index + 1, "rows": rows}
        (args.out / f"quality_{args.generator}_s{args.steps}.json").write_text(json.dumps(result, indent=1) + "\n")
    print("LOT_H3 kind=tiny_quality_mean " + " | ".join(
        f"{name} kept={row['kept']} " + " ".join(f"{k}={statistics.mean(v):.3f}" for k, v in row.items()
                                                  if isinstance(v, list) and v and k not in ("blocks", "seconds"))
        for name, row in rows.items()), flush=True)
    print(f"LOT_H3 kind=tiny_quality_done out={args.out} ok=1", flush=True)


# --------------------------------------------------------------------------------------------- timing

def timed(fn, repeats: int):
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times), torch.cuda.max_memory_allocated() / 2**30


@torch.no_grad()
def resident(args) -> None:
    """Per-block times with no streaming in them: the blocks that stay on the card run as a stack of their own on
    the inputs a real forward gives the block stack. Then the same stack with every other block skipped: what a
    skipped block still costs."""
    import types

    import h3_musubi
    from gpu_guard import refuse_if_busy
    from setproctitle import setproctitle

    setproctitle(f"tiny:resident {args.sizes} swap{args.swap}")
    gpu = refuse_other_card()
    refuse_if_busy("tiny_h3.py")
    dit = h3_musubi.load_dit(swap=args.swap)
    result = {"gpu": gpu, "swap": args.swap, "rows": []}
    plain_run = h3_musubi.run_blocks
    for size in args.sizes.split(","):
        width, height, latent_t = (int(v) for v in size.split("x"))
        g = torch.Generator("cuda").manual_seed(0)
        latent = torch.randn(1, 24, latent_t, height // 16, width // 16, device="cuda", generator=g)
        text = torch.randn(1, args.text, dit.config.text_dim, device="cuda", generator=g).to(torch.bfloat16)
        captured = {}

        def capture(dit_, hidden, temb, segments, rotation, tap=None):
            captured.update(hidden=hidden, temb=temb, segments=segments, rotation=rotation)
            return plain_run(dit_, hidden, temb, segments, rotation, tap=tap)

        h3_musubi.run_blocks = capture
        try:
            h3_musubi.forward_video(dit, latent, 0.5, text, audio_generator=torch.Generator("cuda").manual_seed(1))
        finally:
            h3_musubi.run_blocks = plain_run
        dit.prepare_block_swap_before_forward()
        torch.cuda.synchronize()
        on_card = [i for i, block in enumerate(dit.blocks) if all(p.device.type == "cuda" for p in block.parameters())]
        blocks = [dit.blocks[i] for i in on_card[:args.blocks]]
        sub = types.SimpleNamespace(blocks=blocks, blocks_to_swap=0, gradient_checkpointing=False, training=False,
                                    video_patch_proj=dit.video_patch_proj)
        hidden, temb, segments, rotation = (captured[k] for k in ("hidden", "temb", "segments", "rotation"))
        run = lambda: plain_run(sub, hidden, temb, segments, rotation)                  # noqa: E731
        seconds, peak = timed(run, args.repeats)
        row = {"size": size, "tokens": hidden.shape[1], "blocks": len(blocks),
               "ms_per_block": 1e3 * seconds / len(blocks), "peak_gb": peak}
        half = list(range(0, len(blocks), 2))
        with blocks_dropped(sub, half):
            seconds_half, _peak = timed(run, args.repeats)
        ran = len(blocks) - len(half)
        row["ms_per_block_half_skipped"] = 1e3 * seconds_half / ran
        row["ms_per_skipped_block"] = 1e3 * (seconds_half - ran * seconds / len(blocks)) / len(half)
        result["rows"].append(row)
        print("LOT_H3 kind=tiny_resident " + " ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quality")
    q.add_argument("--generator", choices=("pih3", "dir"), default="pih3")
    q.add_argument("--steps", type=int, default=2, help="pi-H3 adapter: 2 or 4 (DiR ignores it)")
    q.add_argument("--run", default="", help="DiR run (default: the published one)")
    q.add_argument("--size", default="768x512")
    q.add_argument("--frames", type=int, default=39)
    q.add_argument("--prompts", type=int, default=3)
    q.add_argument("--plans", default="full,h3tiny,least10,least15")
    q.add_argument("--swap", type=int, default=24)
    q.add_argument("--seed", type=int, default=7000)
    q.add_argument("--vae", action="store_true", help="pi-H3: also decode the latent with the H3 VAE (DWD vs VAE LPIPS)")
    q.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("resident")
    r.add_argument("--sizes", default="512x288x12,768x512x12")
    r.add_argument("--swap", type=int, default=16)
    r.add_argument("--blocks", type=int, default=30)
    r.add_argument("--text", type=int, default=128)
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    {"quality": quality, "resident": resident}[args.cmd](args)


if __name__ == "__main__":
    main()
