#!/usr/bin/env python3
"""Experiments on pi-H3 (Hansheng Chen, Shengqu Cai, Jiaming Song, Gordon Wetzstein, Xinchen Yan, Leonidas Guibas;
https://huggingface.co/Lakonik/pi-H3): the cascade sampler, block dropping, and both against two-track.

``run``: one table per canvas of sampler variants, same prompts and seeds:
- ``full``      pi-H3 at full resolution (``pih3_h3.pih3_sample``), the reference of its step count
- ``floor``     ``full`` from the same noise rounded to bf16: the DiT's chaos floor for same-seed scores
- ``half``      half resolution + DWD, pixels x2 (bicubic)
- ``tt:K``      MPEG-Diffusion two-track (``pih3_h3.pih3_two_track``), region opens after K steps
- ``cas:K``     the cascade: K steps on the whole clip at half resolution, the clean estimate x2 (latent), re-noised
                to the next sigma, then the remaining steps on the whole clip at full resolution. No region head,
                no K/V maps, no paste: every forward is a plain pi-H3 forward, and DWD decodes a full token grid.
                Same LR, audio and HR noise as ``tt:K`` (seed, seed + 1, seed + 2).
Any variant takes ``@BLOCKS`` to skip DiT blocks (``h3tiny`` or indices joined by ``-``), e.g. ``full@h3tiny``.

Scores (not timed): ``lpips_full`` against ``full`` of this run, ``lpips_ref`` against another run's ``full``
(``--ref``, e.g. the 4-step clips for a 2-step run), ``lpips_vae`` (``--score``) of the DWD pixels against the VAE
decode of the same final latent, and ``hf``: mean |luma - 5x5 box blur|, a crude detail measure, as a ratio to
``full``. Same-seed LPIPS is bounded below by ``floor``; LR-started variants are other videos by construction.

``dropscan``: which blocks pi-H3 can skip. One forward per dropped block at the first and the last sigma of a
dense rollout (relative error of the velocity and of the DWD head's input), then sets of the least sensitive
blocks end to end. ``--swap`` streams blocks from the CPU so it fits a 24 GB card (quality only, not speed).

    python3 pih3_cascade.py run --steps 4 --sizes 768x512 --variants full floor half tt:2 cas:2 cas:3 --out OUT
    python3 pih3_cascade.py dropscan --steps 4 --size 768x512 --swap 12 --out OUT
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

from h3_musubi import H3_TINY_DROP, audio_sigma  # noqa: E402
from pih3_h3 import adapter_path, apply_adapter, audio_noise, carried_audio_step, load_pih3, pih3_sample, \
    pih3_sigmas, pih3_two_track, PiH3Head  # noqa: E402


def refuse_if_card_busy(limit_gb: float) -> None:
    """One job per card: refuse when other processes hold more than ``limit_gb`` on this card (a desktop's few GB
    pass). ``gpu_guard.refuse_if_busy`` is machine-wide and would stop a job on the second card."""
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 2**30
    if used > limit_gb:
        raise SystemExit(f"pih3_cascade.py: {used:.1f} GB already in use on {torch.cuda.get_device_name()}")


def load(steps: int, swap: int = 0) -> dict:
    """``load_pih3``, or with ``swap`` the same model merged on the CPU and ``swap`` blocks streamed to the card."""
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


def parse_drop(spec: str) -> tuple[int, ...]:
    if not spec:
        return ()
    return tuple(H3_TINY_DROP) if spec == "h3tiny" else tuple(sorted({int(x) for x in spec.split("-")}))


class dropped:
    """Skip blocks ``indices`` inside the ``with`` (they return their input), then run them again."""

    def __init__(self, dit, indices):
        self.blocks = [dit.blocks[i] for i in indices]

    def __enter__(self):
        import types

        for block in self.blocks:
            block.forward = types.MethodType(lambda _self, hidden, *_a, **_k: hidden, block)

    def __exit__(self, *exc):
        for block in self.blocks:
            block.__dict__.pop("forward", None)


@torch.no_grad()
def pih3_cascade(model: dict, text: torch.Tensor, *, height: int, width: int, latent_t: int, seed: int,
                 switch: int, up="tri", first: str = "lr") -> dict:
    """The cascade (module docstring). Returns what ``pih3_sample`` returns, at full resolution.

    ``up``: how the clean LR estimate becomes an HR latent: ``tri`` (trilinear on the latent, two-track's),
    ``nearest``, ``bicubic``, or a callable (a VAE round trip, a learned upsampler).
    ``first`` (controls for the handoff): ``lr`` is the cascade; ``self`` runs the first stage at full resolution
    too, so only the fresh re-noising differs from ``full``; ``selfblur`` also sends that estimate through
    area x0.5 and ``up``."""
    from h3_musubi import forward_video

    dit, sigmas = model["dit"], model["sigmas"]
    steps = len(sigmas) - 1
    if not 1 <= switch < steps:
        raise ValueError(f"switch {switch} must be in 1..{steps - 1}")
    lat_h, lat_w = height // 32, width // 32
    scale = 1 if first == "lr" else 2
    z_lr = torch.randn(1, 24, latent_t, scale * lat_h, scale * lat_w, device="cuda",
                       generator=torch.Generator("cuda").manual_seed(seed))
    x, y = sigmas[0] * z_lr, sigmas[0] * audio_noise(latent_t, seed + 1)
    out: dict = {}

    def upsample(latent: torch.Tensor) -> torch.Tensor:
        if callable(up):
            return up(latent).float()
        if up == "tri":
            return F.interpolate(latent, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        frames = latent[0].transpose(0, 1)
        kw = {} if up == "nearest" else {"align_corners": False}
        return F.interpolate(frames, scale_factor=2, mode=up, **kw).transpose(0, 1).unsqueeze(0)

    for i in range(steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        if i == switch:                          # the last LR step left x at sigma, with est its clean estimate
            if first == "selfblur":
                est = F.avg_pool3d(est, (1, 2, 2))
            if first != "self":
                est = upsample(est)
            z_hr = torch.randn(1, 24, latent_t, 2 * lat_h, 2 * lat_w, device="cuda",
                               generator=torch.Generator("cuda").manual_seed(seed + 2))
            x = sigma * z_hr + (1.0 - sigma) * est
        sigma_a = float(audio_sigma(sigma))
        store: dict = {}
        v = forward_video(dit, x, 1.0 - sigma, text, audio=y * (sigma_a / sigma), t_audio=1.0 - sigma_a,
                          out=store).float()
        if i == steps - 1:
            out["h"], out["temb"] = store["hidden"], store["temb_video"]
        est = x + sigma * v
        y = carried_audio_step(y, store["audio"].float(), sigma, sigma_next)
        x = x + v * (sigma - sigma_next)
    out.update(latent=x, audio=y)
    return out


def hf_energy(pixels: torch.Tensor, every: int = 6) -> float:
    luma = pixels[0].float().mean(0)[::every].unsqueeze(1)
    return float((luma - F.avg_pool2d(luma, 5, 1, 2, count_include_pad=False)).abs().mean())


def to_u8(pixels: torch.Tensor) -> torch.Tensor:
    """(1, 3, F, H, W) in [-1, 1] -> (F, H, W, 3) uint8 on the CPU."""
    return pixels[0].float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 3, 0).cpu()


def from_u8(u8: torch.Tensor) -> torch.Tensor:
    return u8.cuda().permute(3, 0, 1, 2).float().div(127.5).sub(1).unsqueeze(0)


def sheet(clips: dict, path: Path, columns: int, tile_w: int) -> None:
    """Contact sheet: one row per variant, ``columns`` frames spread over the clip, labels on the left tile."""
    from PIL import Image, ImageDraw

    rows = []
    for label, u8 in clips.items():
        count = u8.shape[0]
        picks = [count // 2] if columns == 1 else [round(k * (count - 1) / (columns - 1)) for k in range(columns)]
        tiles = []
        for k, f in enumerate(picks):
            image = Image.fromarray(u8[f].numpy())
            image = image.resize((tile_w, round(tile_w * image.height / image.width)), Image.LANCZOS)
            if k == 0:
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 8 + 7 * len(label), 16), fill=(0, 0, 0))
                draw.text((4, 2), label, fill=(255, 255, 0))
            tiles.append(image)
        row = Image.new("RGB", (tile_w * len(tiles), tiles[0].height))
        for k, tile in enumerate(tiles):
            row.paste(tile, (k * tile_w, 0))
        rows.append(row)
    out = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
    top = 0
    for row in rows:
        out.paste(row, (0, top))
        top += row.height
    out.save(path, quality=90)


@torch.no_grad()
def run(args) -> None:
    import lpips

    from accel_h3 import default_caches, lpips_clip, parse_job, prompt_lists, sync, token_shape, upscale_px
    from pih3_h3 import decode
    from setproctitle import setproctitle
    from train_dwd_h3 import load_text
    from two_track_h3 import save_clip

    setproctitle(f"pih3:cascade s{args.steps} {' '.join(args.sizes)} f{args.frames}"[:60])
    refuse_if_card_busy(args.busy_gb)
    model = load(args.steps, args.swap)
    two_track = None
    if any(v.startswith("tt:") for v in args.variants):
        from accel_h3 import load_two_track

        two_track = load_two_track(args)
    vae = None
    if args.score:
        from h3_musubi import load_vae
        from lot_paths import LOT_H3_VAE

        vae = load_vae(LOT_H3_VAE, device="cpu")
    net = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    ups: dict = {}
    if any(v.split("@")[0].endswith(":vae") for v in args.variants):       # diagnostic: the VAE round trip (slow)
        from two_track_h3 import load_upscaler

        vae_up = load_upscaler()
        vae_up.vae.to("cpu")

        def through_vae(latent):
            vae_up.vae.to("cuda")
            result = vae_up(latent)
            vae_up.vae.to("cpu")
            return result

        ups["vae"] = through_vae
    if args.upsampler:
        from latent_up_h3 import load_up

        ups["net"] = load_up(args.upsampler)
    _train, held = prompt_lists(default_caches(), 8)
    for spec in args.sizes:
        width, height, _count, frames = parse_job(spec, args.frames)
        latent_t, _th, _tw = token_shape(width, height, frames)
        out_dir = args.out / f"{width}x{height}_f{frames}_s{args.steps}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: dict = {v: [] for v in args.variants}
        for index, te in enumerate(held[:args.prompts]):
            text = load_text(str(te)).cuda()
            seed = args.seed + index
            clips: dict = {}
            full = full_hf = None
            ref = None
            if args.ref and (args.ref / f"{width}x{height}_f{frames}" / f"p{index}.pt").is_file():
                ref = torch.load(args.ref / f"{width}x{height}_f{frames}" / f"p{index}.pt")
            for variant in args.variants:
                name, _, drop = variant.partition("@")
                kind, k, mode = (name.split(":") + ["", ""])[:3]
                row = {"prompt": index}
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    with dropped(model["dit"], parse_drop(drop)):
                        sync()
                        start = time.perf_counter()
                        tt = None
                        if kind in ("full", "floor", "half"):
                            w, h = (width // 2, height // 2) if kind == "half" else (width, height)
                            z = torch.randn(1, 24, latent_t, h // 16, w // 16, device="cuda",
                                            generator=torch.Generator("cuda").manual_seed(seed))
                            za = audio_noise(latent_t, seed + 1)
                            if kind == "floor":
                                z, za = z.bfloat16().float(), za.bfloat16().float()
                            out = pih3_sample(model, text, z, za)
                            shape = tuple(z.shape)
                        elif kind in ("cas", "self", "selfblur"):
                            out = pih3_cascade(model, text, height=height, width=width, latent_t=latent_t, seed=seed,
                                               switch=int(k), up=ups.get(mode, mode or "tri"),
                                               first={"cas": "lr"}.get(kind, kind))
                            shape = tuple(out["latent"].shape)
                        elif kind == "tt":
                            tt = out = pih3_two_track(model, text, height=height, width=width, latent_t=latent_t,
                                                      seed=seed, two_track=two_track, switch=int(k),
                                                      up=ups.get(mode))
                            row["share"] = round(tt["share"], 3)
                        else:
                            raise SystemExit(f"unknown variant {variant}")
                        sync()
                        sampled = time.perf_counter()
                        if tt is not None:
                            pixels = tt["pixels"]
                        else:
                            pixels = decode(model, out, shape)
                            if kind == "half":
                                pixels = upscale_px(pixels)
                        sync()
                        end = time.perf_counter()
                    row.update(seconds=round(end - start, 3), decode_s=round(end - sampled, 3),
                               peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
                    pixels = pixels.float().clamp(-1, 1)
                    latent = out["latent"]
                    del out, tt
                    row["hf"] = hf_energy(pixels)
                    if variant == "full":
                        full, full_hf = to_u8(pixels), row["hf"]
                        if args.save_ref:
                            (args.save_ref / f"{width}x{height}_f{frames}").mkdir(parents=True, exist_ok=True)
                            torch.save(full, args.save_ref / f"{width}x{height}_f{frames}" / f"p{index}.pt")
                    elif full is not None:
                        row["lpips_full"] = round(lpips_clip(net, pixels, from_u8(full)), 4)
                    if full_hf:
                        row["hf"] = round(row["hf"] / full_hf, 3)
                    if ref is not None:
                        row["lpips_ref"] = round(lpips_clip(net, pixels, from_u8(ref)), 4)
                    clips[variant] = to_u8(pixels)
                    if vae is not None and kind != "half":
                        pixels = pixels.cpu()
                        torch.cuda.empty_cache()
                        vae.to("cuda")
                        target = vae.decode(latent.to("cuda", next(vae.parameters()).dtype)).clamp(-1, 1).float().cpu()
                        vae.to("cpu")
                        torch.cuda.empty_cache()
                        row["lpips_vae"] = round(lpips_clip(net, pixels.cuda(), target.cuda()), 4)
                        del target
                    del pixels, latent
                    if args.save and index == min(1, args.prompts - 1):
                        save_clip(from_u8(clips[variant]).half().cpu(), out_dir / f"p{index}_{variant.replace(':', '_')}.mp4")
                except torch.OutOfMemoryError:
                    row["oom"] = 1
                    if vae is not None:
                        vae.to("cpu")
                rows[variant].append(row)
                print(f"LOT_H3 kind=pih3_cascade size={width}x{height} frames={frames} steps={args.steps} "
                      f"variant={variant} " + " ".join(f"{a}={b}" for a, b in row.items()), flush=True)
            if clips:
                tile = 768 if width <= 768 else 672
                sheet(clips, out_dir / f"p{index}_mid.jpg", 1, min(width, 1344))
                sheet(clips, out_dir / f"p{index}_strip.jpg", 4, tile // 2)
            del clips, full, ref
        keys = ("seconds", "decode_s", "share", "hf", "lpips_full", "lpips_ref", "lpips_vae", "peak_gb")
        lines = [f"{width}x{height}, {frames} frames, pi-H3 {args.steps}-step. Medians of prompts after the first "
                 f"(warm-up); {torch.cuda.get_device_name()}" + (f", {args.swap} blocks streamed" if args.swap else ""),
                 "", "| Variant | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for variant, items in rows.items():
            warm = items[1:] or items
            cells = []
            for key in keys:
                values = [r[key] for r in warm if key in r]
                cells.append(f"{statistics.median(values):.3f}".rstrip("0").rstrip(".") if values else
                             ("oom" if key == "seconds" else ""))
            lines.append(f"| {variant} | " + " | ".join(cells) + " |")
        (out_dir / "TABLE.md").write_text("\n".join(lines) + "\n")
        (out_dir / "rows.json").write_text(json.dumps({"sigmas": model["sigmas"], "rows": rows}, indent=1) + "\n")
        print("\n".join(lines), flush=True)
    print(f"LOT_H3 kind=pih3_cascade_done out={args.out} ok=1", flush=True)


@torch.no_grad()
def dropscan(args) -> None:
    import lpips

    from accel_h3 import default_caches, lpips_clip, parse_job, prompt_lists, token_shape
    from h3_musubi import forward_video
    from pih3_h3 import decode
    from setproctitle import setproctitle
    from train_dwd_h3 import load_text

    setproctitle(f"pih3:dropscan s{args.steps} {args.size} swap{args.swap}")
    refuse_if_card_busy(args.busy_gb)
    model = load(args.steps, args.swap)
    dit, sigmas = model["dit"], model["sigmas"]
    net = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    width, height, _count, frames = parse_job(args.size, args.frames)
    latent_t, _th, _tw = token_shape(width, height, frames)
    _train, held = prompt_lists(default_caches(), 8)
    args.out.mkdir(parents=True, exist_ok=True)
    blocks = len(dit.blocks)

    def rel(a, b) -> float:
        return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))

    def one(x, y, sigma, text):
        sigma_a = float(audio_sigma(sigma))
        store: dict = {}
        v = forward_video(dit, x, 1.0 - sigma, text, audio=y * (sigma_a / sigma), t_audio=1.0 - sigma_a, out=store)
        return v.float(), store["hidden"].float()

    # 1. one forward per dropped block, at the first sigma and at the last one of the dense rollout
    scan = {b: {"v_first": [], "v_last": [], "h_last": []} for b in range(blocks)}
    cases = []
    for index, te in enumerate(held[:args.prompts]):
        text = load_text(str(te)).cuda()
        seed = args.seed + index
        z = torch.randn(1, 24, latent_t, height // 16, width // 16, device="cuda",
                        generator=torch.Generator("cuda").manual_seed(seed))
        za = audio_noise(latent_t, seed + 1)
        x, y = sigmas[0] * z, sigmas[0] * za
        first = (x, y, sigmas[0])
        for i in range(len(sigmas) - 2):                   # the dense rollout up to the last step's input
            sigma_a = float(audio_sigma(sigmas[i]))
            store: dict = {}
            v = forward_video(dit, x, 1.0 - sigmas[i], text, audio=y * (sigma_a / sigmas[i]), t_audio=1.0 - sigma_a,
                              out=store).float()
            y = carried_audio_step(y, store["audio"].float(), sigmas[i], sigmas[i + 1])
            x = x + v * (sigmas[i] - sigmas[i + 1])
        last = (x, y, sigmas[-2])
        v0, _ = one(*first, text)
        v1, h1 = one(*last, text)
        for b in range(blocks):
            with dropped(dit, (b,)):
                a, _ = one(*first, text)
                c, h = one(*last, text)
            scan[b]["v_first"].append(rel(a, v0))
            scan[b]["v_last"].append(rel(c, v1))
            scan[b]["h_last"].append(rel(h, h1))
            print(f"LOT_H3 kind=pih3_dropscan prompt={index} block={b} v_first={scan[b]['v_first'][-1]:.4f} "
                  f"v_last={scan[b]['v_last'][-1]:.4f} h_last={scan[b]['h_last'][-1]:.4f}", flush=True)
        cases.append((text, z, za))
    mean = {b: {k: sum(v) / len(v) for k, v in scan[b].items()} for b in range(blocks)}
    order = sorted(range(blocks), key=lambda b: max(mean[b]["v_first"], mean[b]["v_last"]))
    (args.out / "scan.json").write_text(json.dumps({"mean": mean, "order": order}, indent=1) + "\n")
    print("LOT_H3 kind=pih3_dropscan_order " + " ".join(map(str, order)), flush=True)

    # 2. sets end to end: the least sensitive k blocks, and H3-Tiny's plan, against dense and its floor
    sets = {"dense": (), "floor": (), "h3tiny": tuple(H3_TINY_DROP)}
    sets.update({f"least{k}": tuple(sorted(order[:k])) for k in args.sets})
    results = {name: {"blocks": list(s), "lpips_dense": [], "hf": [], "latent_rel": []} for name, s in sets.items()}
    for index, (text, z, za) in enumerate(cases):
        clips: dict = {}
        dense = dense_latent = dense_hf = None
        for name, indices in sets.items():
            zz, zza = (z.bfloat16().float(), za.bfloat16().float()) if name == "floor" else (z, za)
            with dropped(dit, indices):
                out = pih3_sample(model, text, zz, zza)
                pixels = decode(model, out, tuple(z.shape)).float().clamp(-1, 1)
            if name == "dense":
                dense, dense_latent, dense_hf = pixels, out["latent"], hf_energy(pixels)
            else:
                results[name]["lpips_dense"].append(round(lpips_clip(net, pixels, dense), 4))
                results[name]["latent_rel"].append(round(rel(out["latent"], dense_latent), 4))
            results[name]["hf"].append(round(hf_energy(pixels) / dense_hf, 3))
            clips[name] = to_u8(pixels)
            print(f"LOT_H3 kind=pih3_dropset prompt={index} set={name} blocks={list(indices)} "
                  + " ".join(f"{k}={v[-1]}" for k, v in results[name].items() if k != "blocks" and v), flush=True)
        sheet(clips, args.out / f"p{index}_mid.jpg", 1, width)
        sheet(clips, args.out / f"p{index}_strip.jpg", 4, width // 2)
    (args.out / "sets.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"LOT_H3 kind=pih3_dropscan_done out={args.out} ok=1", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sizes", nargs="+", default=["768x512"])
    r.add_argument("--variants", nargs="+", default=["full", "floor", "half", "tt:1", "cas:1"])
    r.add_argument("--score", action="store_true", help="lpips_vae: DWD pixels against the VAE decode of the same latent")
    r.add_argument("--save", action="store_true", help="an mp4 per variant for the second prompt")
    r.add_argument("--save-ref", type=Path, default=None, help="store the full clips here (another run's --ref)")
    r.add_argument("--ref", type=Path, default=None, help="full clips of another run (e.g. 4-step) to score against")
    r.add_argument("--upsampler", type=Path, default=None, help="latent_up_h3.py weights, for cas:K:net")
    r.add_argument("--maps", default=None)
    r.add_argument("--region", default=None)
    d = sub.add_parser("dropscan")
    d.add_argument("--size", default="768x512")
    d.add_argument("--sets", type=int, nargs="+", default=[4, 8, 12, 16])
    for q in (r, d):
        q.add_argument("--steps", type=int, default=4, choices=(2, 4))
        q.add_argument("--frames", type=int, default=39)
        q.add_argument("--prompts", type=int, default=3)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--swap", type=int, default=0, help="DiT blocks streamed from the CPU (a 24 GB card: 12)")
        q.add_argument("--busy-gb", type=float, default=8.0, help="refuse when the card already holds more than this")
        q.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    {"run": run, "dropscan": dropscan}[args.cmd](args)


if __name__ == "__main__":
    main()
