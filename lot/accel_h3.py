#!/usr/bin/env python3
"""DiR + DWD end to end on H3, at any resolution: one decoder stack, one sampler, and the phases of the
acceleration run (collect, cache, quality, bench, check).

    # GPU parity of the DWD tail (blocks K.. run twice) against a full forward with the LoRA on
    python3 accel_h3.py check --lora LORA --head HEAD

    # multi-resolution rollouts with VAE targets; one folder per size and per exit iteration
    python3 accel_h3.py collect --out DATA --sizes 768x512:600 1344x768:300 672x384:400 --exits 3 2

    # DWD decodes (and the latent the refiner will read) for refiner training: OUT/cache/pairs.json
    python3 accel_h3.py cache --data DATA/r3/* --lora LORA --head HEAD --out OUT

    # quality against the VAE decode of the same latent: OUT/QUALITY.md
    python3 accel_h3.py quality --sizes 768x512 1024x576 1344x768 --lora LORA --head HEAD --refiner REF --out OUT

    # speed, DiT and decode split, per size and variant: OUT/BENCH.md (scripts/bench_accel.sh)
    python3 accel_h3.py bench --sizes 768x512 1344x768 --lora LORA --head HEAD --refiner REF --out OUT

Nothing here assumes a canvas. A size is ``WxH`` in pixels, any multiple of 32 (64 for the half-resolution and
two-track variants, whose LR latent needs whole 2x2 patches); a clip is any number of pixel frames H3 accepts.

Exit iteration. A DWD stack decodes the tokens of one DiR iteration ``r`` (``--exit``; default the last). With an
earlier ``r`` the remaining iterations never run: the stack is trained to give the pixels the VAE would give for
the FULL rollout's latent, so the reference of an early exit is still the R-iteration VAE decode.

Decode modes.
- ``two_pass`` (as the published stack was trained): the iteration runs as the generator's own, then the last K
  blocks run again with the DWD LoRA on (``h3_musubi.rerun_blocks``). The refiner reads the generator's own
  estimate. Costs ``K / blocks`` of an iteration extra (8 / 50).
- ``single``: the DWD LoRA is on in the only forward, and the refiner reads that forward's estimate. No extra
  blocks; the refiner must have been trained on it (``cache --mode single``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dwd_h3 import capture_last_layer, flicker_split, pack_rows, pixel_frame_count, psnr, seam_ratio  # noqa: E402

DEFAULT_RUN = "johndpope/dir-diffusion-h3"
MODES = ("two_pass", "single")
VARIANTS = ("dir_vae", "dir_dwd", "dir_dwd_early", "lr_vae", "lr_dwd", "tt_vae", "tt_dwd", "tt_dwd2")
LABEL = {"dir_vae": "DiR, VAE decode", "dir_dwd": "DiR + DWD", "dir_dwd_early": "DiR + DWD, early exit",
         "lr_vae": "DiR half res, VAE decode, x2", "lr_dwd": "DiR half res + DWD, x2",
         "tt_vae": "two-track fast path, VAE", "tt_dwd": "two-track, DWD on the LR track, VAE on the region",
         "tt_dwd2": "two-track, DWD on both tracks"}
NEEDS_VAE = ("dir_vae", "lr_vae", "tt_vae", "tt_dwd")


# ------------------------------------------------------------------------------------------- sizes

def parse_size(spec: str) -> tuple[int, int]:
    """``"768x512"`` -> ``(768, 512)``: width, height in pixels."""
    match = re.fullmatch(r"(\d+)x(\d+)", str(spec).strip().lower())
    if not match:
        raise ValueError(f"size {spec!r}: expected WxH, e.g. 768x512")
    return int(match.group(1)), int(match.group(2))


def check_size(width: int, height: int, multiple: int = 32) -> None:
    """H3 needs whole tokens: one token is 32x32 pixels. The half-resolution tracks need ``multiple=64``."""
    if width <= 0 or height <= 0 or width % multiple or height % multiple:
        raise ValueError(f"size {width}x{height}: both sides must be positive multiples of {multiple}")


def parse_job(spec: str, frames: int) -> tuple[int, int, int, int]:
    """``"WxH[:count[:frames]]"`` -> ``(width, height, count, pixel frames)``; ``count`` defaults to 0."""
    parts = str(spec).split(":")
    if len(parts) > 3:
        raise ValueError(f"{spec!r}: expected WxH[:count[:frames]]")
    width, height = parse_size(parts[0])
    check_size(width, height)
    count = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    frames = int(parts[2]) if len(parts) > 2 else frames
    if clip_latent_frames(frames) is None:
        raise ValueError(f"{spec!r}: {frames} pixel frames is not a clip length H3 decodes to (1, 5, 22, 39, 56, ...)")
    return width, height, count, frames


def clip_latent_frames(frames: int) -> int | None:
    """Latent frames of a clip with exactly ``frames`` pixel frames, or ``None`` when no latent decodes to that."""
    for t in range(1, frames + 1):
        n = pixel_frame_count(t)
        if n == frames:
            return t
        if n > frames:
            return None
    return None


def folder_name(width: int, height: int, frames: int, default_frames: int = 39) -> str:
    return f"{width}x{height}" + ("" if frames == default_frames else f"_f{frames}")


def token_shape(width: int, height: int, frames: int) -> tuple[int, int, int]:
    """``(latent frames, token rows, token columns)`` of a clip: what ``DWDHead.decode`` takes."""
    check_size(width, height)
    latent_t = clip_latent_frames(frames)
    if latent_t is None:
        raise ValueError(f"{frames} pixel frames is not a clip length H3 decodes to")
    return latent_t, height // 32, width // 32


def interleave(lists: list[list]) -> list:
    """Round-robin merge: ``[[a, b, c], [x], [p, q]]`` -> ``[a, x, p, b, q, c]``."""
    out = []
    for i in range(max((len(x) for x in lists), default=0)):
        out += [x[i] for x in lists if i < len(x)]
    return out


def spread(groups: list[list]) -> list:
    """Merge job lists so every list advances at the same relative pace (a run cut short by its time budget
    leaves every size equally complete). Stable inside a list."""
    keyed = [((i + 0.5) / len(group), g, i, job) for g, group in enumerate(groups) for i, job in enumerate(group)]
    return [job for _key, _g, _i, job in sorted(keyed, key=lambda k: k[:3])]


# ------------------------------------------------------------------------------------------- stack

def _block_of(lora_name: str) -> int:
    return int(re.search(r"blocks_(\d+)_", lora_name).group(1))


def load_refiner(path: Path, device):
    from refine_dwd_h3 import Refiner, spec_of

    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = Refiner(**spec_of(saved))
    model.load_state_dict(saved["model"])
    return model.to(device).eval().requires_grad_(False)


def refiner_chunk(height: int, width: int, budget: float = 1.6e6) -> int:
    """Frames per refiner batch for a ``height x width`` clip: as many as fit ``budget`` pixels, between 1 and 8.
    Measured on an RTX 5090 at 768x512: 4 frames per batch take 0.26 s a clip and 2.2 GB, 8 the same time and
    3.6 GB, so memory is what the batch costs. 1344x768 gets 1 (8 would need about 9.5 GB beside the DiT)."""
    return int(max(1, min(8, budget // (height * width))))


class DwdStack:
    """The DWD decoder of one exit iteration: the last-K-blocks LoRA (applied to ``dit``, off until ``on()``),
    the linear pixel head and, optionally, the refiner."""

    def __init__(self, dit, lora: Path, head: Path, refiner: Path | None = None, *, exit_r: int | None = None,
                 mode: str = "two_pass", half_head: bool = True, amp: bool = True, chunk: int = 0,
                 channels_last: bool = False, device="cuda"):
        from h3_musubi import make_lora
        from train_dwd_h3 import load_head

        if mode not in MODES:
            raise ValueError(f"mode {mode!r} is not one of {MODES}")
        saved = torch.load(lora, map_location="cpu", weights_only=True)
        self.network = make_lora(dit, saved["rank"])
        if self.network.load_state_dict(saved["lora"], strict=False).unexpected_keys:
            raise SystemExit(f"{lora}: keys that are not in this LoRA")
        self.network.requires_grad_(False)
        self.modules = [m for m in self.network.unet_loras if any(k.startswith(m.lora_name + ".") for k in saved["lora"])]
        for module in self.network.unet_loras:
            module.enabled = False                                # the other blocks' modules stay off, and on the CPU
        for module in self.modules:
            module.to(device, torch.float32)
        blocks = sorted({_block_of(m.lora_name) for m in self.modules})
        if not blocks or blocks != list(range(blocks[0], len(dit.blocks))):
            raise SystemExit(f"{lora}: the LoRA must cover the last K blocks, got blocks {blocks[:3]}..{blocks[-3:]}")
        self.first = blocks[0]
        self.head = load_head(head, "cpu").requires_grad_(False)
        if half_head:
            self.head.half_weights()                              # before the move: the fp32 map is 3 GB
        self.head.to(device)
        self.refiner = load_refiner(refiner, device) if refiner else None
        if self.refiner is not None and channels_last:
            self.refiner.to(memory_format=torch.channels_last)
        self.exit_r, self.mode, self.amp, self.chunk, self.channels_last = exit_r, mode, amp, chunk, channels_last
        self.paths = {"lora": str(lora), "head": str(head), "refiner": str(refiner) if refiner else None}

    def to(self, device) -> "DwdStack":
        """Move the head and the refiner (about 1.6 GB with an fp16 head). A 32 GB card holds the DiT with either
        this stack or the VAE (4.9 GB) beside a 1344x768 clip, not both."""
        self.head.to(device)
        if self.refiner is not None:
            self.refiner.to(device)
        return self

    @contextlib.contextmanager
    def on(self):
        for module in self.modules:
            module.enabled = True
        try:
            yield
        finally:
            for module in self.modules:
                module.enabled = False

    @torch.no_grad()
    def decode(self, h: torch.Tensor, state: torch.Tensor, latent: torch.Tensor | None,
               shape: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """``(pixels, head pixels)``, each ``(1, 3, F, H, W)`` in [-1, 1]. ``h`` are the exit iteration's DWD tokens,
        ``state`` the latent state that iteration read, ``latent`` the estimate the refiner reads."""
        from refine_dwd_h3 import refine_clip

        latent_t, token_h, token_w = shape
        rows = state if state.ndim == 2 else pack_rows(state.float())
        head_px = self.head.decode(h, rows, latent_t, token_h, token_w)
        if self.refiner is None:
            return head_px, head_px
        clip = head_px.contiguous(memory_format=torch.channels_last_3d) if self.channels_last else head_px
        with torch.autocast(head_px.device.type, dtype=torch.bfloat16, enabled=self.amp):
            out = refine_clip(self.refiner, clip, latent, chunk=self.chunk or refiner_chunk(*head_px.shape[-2:]))
        return out.float(), head_px


@torch.no_grad()
def exit_step(dit, gen_head, stack: DwdStack, state: torch.Tensor, prev: torch.Tensor, r: int, text: torch.Tensor,
              sigma: float, audio_seed: int, mode: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """DiR iteration ``r`` with the DWD tokens: ``(h, estimate)``. ``h`` is ``(N, hidden)`` fp32. In ``two_pass``
    mode ``estimate`` is the generator's own ``x0_hat[r + 1]``; in ``single`` mode it is the LoRA-on forward's."""
    from dir_h3 import estimate_extra
    from h3_musubi import forward_video

    mode = mode or stack.mode
    kwargs = dict(video_extra=estimate_extra(gen_head, prev, None), video_temb_delta=gen_head.r_embed[r],
                  audio_generator=torch.Generator(state.device).manual_seed(int(audio_seed)))
    if mode == "two_pass":
        tap = {"first": stack.first, "lora": stack.on}
        out = forward_video(dit, state, 1.0 - sigma, text, dwd=tap, **kwargs)
        h = tap["h"]
    else:
        store: dict = {}
        with stack.on(), capture_last_layer(dit, store):
            out = forward_video(dit, state, 1.0 - sigma, text, **kwargs)
        h = store["h"].float()
    return h, state.float() + sigma * out.float()


@torch.no_grad()
def dir_rollout(dit, gen_head, text: torch.Tensor, z: torch.Tensor, sigmas: list, audio_seed: int, *,
                mix_state: bool = False, stack: DwdStack | None = None, exit_r: int | None = None,
                keep: tuple = ()) -> dict:
    """DiR iterations ``0 .. exit_r`` from noise ``z`` (default: all of them), as ``dir_h3.dir_sample`` runs them.

    Returns ``estimate`` (the last iteration's), ``estimates`` (every iteration's), ``inputs`` (``{r: (state,
    previous estimate)}`` for ``r`` in ``keep``) and, with a ``stack``, ``h`` and ``state`` of the exit iteration
    for ``DwdStack.decode``."""
    from dir_h3 import generator_step, next_state

    last = len(sigmas) - 1 if exit_r is None else int(exit_r)
    if not 0 <= last < len(sigmas):
        raise ValueError(f"exit iteration {last} is outside 0..{len(sigmas) - 1}")
    estimate, state, out = torch.zeros_like(z), z, {"inputs": {}, "estimates": []}
    for r in range(last + 1):
        if r in keep:
            out["inputs"][r] = (state, estimate)
        if r == last and stack is not None:
            out["state"] = state
            out["h"], estimate = exit_step(dit, gen_head, stack, state, estimate, r, text, sigmas[r], audio_seed)
        else:
            estimate = generator_step(dit, gen_head, state, estimate, r, text, sigma=sigmas[r],
                                      audio_generator=torch.Generator(z.device).manual_seed(int(audio_seed))).float()
        out["estimates"].append(estimate)
        if r < last:
            state = next_state(state, estimate, sigmas[r], sigmas[r + 1], z if mix_state else None)
    out["estimate"] = estimate
    return out


# ------------------------------------------------------------------------------------------- loading

def gpu_setup(name: str) -> None:
    from gpu_guard import refuse_if_busy

    refuse_if_busy(name)


def load_models(args) -> dict:
    """The DiT with every block on the card (the DWD tail and a stacked LoRA need that), and the generator."""
    from dir_h3 import parse_sigmas
    from dir_two_track_h3 import load_generator, resolve_run
    from h3_musubi import load_dit

    dit = load_dit(device="cuda", swap=0)
    gen_head, net, config = load_generator(resolve_run(str(args.run)), dit, args.tag, False)
    if str(config.get("schedule") or "dense") != "dense":
        raise SystemExit("DWD reads the dense forward; this run has a LoT layout schedule")
    iters = int(config.get("iters", 4))
    return {"dit": dit, "gen_head": gen_head, "net": net, "config": config, "iters": iters,
            "sigmas": parse_sigmas(config.get("sigmas") or "1", iters), "mix": bool(config.get("mix_state", False))}


def load_stack(args, dit, *, refiner: bool = True, half_head: bool = True, prefix: str = "") -> DwdStack:
    get = lambda name: getattr(args, prefix + name)                                    # noqa: E731
    return DwdStack(dit, get("lora"), get("head"), get("refiner") if refiner else None, exit_r=get("exit"),
                    mode=get("mode"), half_head=half_head, amp=not getattr(args, "fp32", False),
                    chunk=getattr(args, "chunk", 0), channels_last=getattr(args, "channels_last", False))


def prompt_lists(caches: list[Path], holdout: int) -> tuple[list[Path], list[Path]]:
    """Text-embedding files, ``(train, held out)``, round-robin over the caches. Each cache is split exactly as
    ``train_h3.list_items`` splits it, so a prompt held out of the DiR and DWD runs is held out here."""
    from train_h3 import list_items

    train, held = [], []
    for cache in caches:
        a, b = list_items(Path(cache), holdout)
        train.append([te for _latent, te in a])
        held.append([te for _latent, te in b])
    return interleave(train), interleave(held)


def default_caches() -> list[Path]:
    from lot_paths import LOT_H3_CLIPS_CACHE, LOT_H3_STILLS_CACHE

    env = os.environ.get("ACCEL_CACHES")
    if env:
        return [Path(p) for p in env.split(":") if p]
    return [p for p in (LOT_H3_STILLS_CACHE, LOT_H3_CLIPS_CACHE) if p.is_dir()]


def sample_seeds(held: bool, index: int, seed: int) -> tuple[int, int]:
    """``(noise seed, audio seed)`` of a collected sample: the numbering of ``train_dwd_h3.py collect``."""
    base = 0 if held else 100000
    return 5000 + base + 100 * index + seed, 31 * (base + index) + seed


def sync() -> None:
    torch.cuda.synchronize()


# ------------------------------------------------------------------------------------------- collect

@torch.no_grad()
def collect_into(root: Path, models: dict, vae, jobs_spec: list[tuple[int, int, int, int]], *, exits: list[int],
                 caches: list[Path], holdout: int = 8, held: int = 8, seeds: int = 1, minutes: float = 0,
                 min_free_gb: float = 40.0, default_frames: int = 39, run: str = "") -> dict:
    """Rollouts at every size in ``jobs_spec`` (``(width, height, training clips, pixel frames)``), each written as
    a light sample per exit iteration, ``root/r<exit>/<WxH>/<name>.pt``, plus the VAE decode of the FULL
    rollout's latent as ``<name>.mp4`` (one file, hard-linked into every exit folder).

    A sample is ``train_dwd_h3.py collect --light``'s: the exit iteration's inputs (``state``, ``prev``,
    ``sigma``, ``r``), the text by path, and the rollout's estimate after that iteration (``latent`` for the last
    exit, where it is the final latent; ``exit_latent`` for an earlier one). So ``train_dwd_h3.py ft`` /
    ``refit`` and ``cache`` here read these folders unchanged. Existing samples are skipped: a rerun resumes.
    Sizes advance together, so a time budget (``minutes``) leaves each equally complete."""
    from train_dwd_h3 import load_text, write_mp4

    dit, gen_head, net, sigmas, mix = (models[k] for k in ("dit", "gen_head", "net", "sigmas", "mix"))
    from dir_h3 import use_networks

    R = len(sigmas)
    exits = sorted({int(e) for e in exits}, reverse=True)
    if any(not 0 <= e < R for e in exits):
        raise SystemExit(f"--exits {exits}: the run has iterations 0..{R - 1}")
    train_te, held_te = prompt_lists(caches, holdout)
    dev = torch.device("cuda")
    vae_dtype = next(vae.parameters()).dtype
    groups, folders = [], {}
    for size_id, (width, height, count, frames) in enumerate(jobs_spec):
        name = folder_name(width, height, frames, default_frames)
        folders[name] = (width, height, frames)
        group = [(name, True, i, 0) for i in range(min(held, len(held_te)))]
        group += [(name, False, i, s) for i in range(-(-count // seeds)) for s in range(seeds)][:count]
        groups.append(group)
        for e in exits:
            folder = root / f"r{e}" / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "collect.json").write_text(json.dumps({
                "run": run, "iters": R, "sigmas": sigmas, "mix_state": mix, "frames": frames, "size": [width, height],
                "exit": e, "caches": [str(c) for c in caches], "holdout": holdout, "targets": "the full rollout's VAE decode"},
                indent=1))
    jobs = spread(groups)

    def files(name: str, stem: str) -> list[Path]:
        return [root / f"r{e}" / name / f"{stem}{ext}" for e in exits for ext in (".mp4", ".pt")]

    t_start, done, skipped, stopped = time.time(), 0, 0, ""
    for name, is_held, index, seed in jobs:
        stem = f"{'held' if is_held else 'train'}_{index:04d}_{seed}"
        if all(p.exists() for p in files(name, stem)):
            skipped += 1
            continue
        if minutes and time.time() - t_start > minutes * 60:
            stopped = "minutes"
            break
        if shutil.disk_usage(root).free / 2**30 < min_free_gb:
            stopped = "disk"
            break
        width, height, frames = folders[name]
        latent_t, token_h, token_w = token_shape(width, height, frames)
        pool = held_te if is_held else train_te
        # training prompts start at a different place for every size, so the sizes do not share one prompt list
        te = pool[index % len(pool)] if is_held else pool[(index + 997 * list(folders).index(name)) % len(pool)]
        text = load_text(str(te)).to(dev)
        z_seed, audio_seed = sample_seeds(is_held, index, seed)
        z = torch.randn(1, 24, latent_t, height // 16, width // 16, device=dev,
                        generator=torch.Generator(dev).manual_seed(z_seed))
        sync()
        t0 = time.time()
        with use_networks(net, None, gen=True, fake=False):
            out = dir_rollout(dit, gen_head, text, z, sigmas, audio_seed, mix_state=mix, keep=tuple(exits))
        sync()
        dit_s = time.time() - t0
        final = out["estimate"]
        pixels = vae.decode(final.to(dev, vae_dtype)).float()
        if pixels.shape[2] != frames:
            raise RuntimeError(f"{name}/{stem}: the VAE gave {pixels.shape[2]} frames, expected {frames}")
        video = root / f"r{exits[0]}" / name / f"{stem}.mp4"
        write_mp4(video, pixels)
        for e in exits[::-1]:                                     # the last exit's .pt is written last: it marks done
            folder = root / f"r{e}" / name
            if e != exits[0]:
                link = folder / f"{stem}.mp4"
                link.unlink(missing_ok=True)
                try:
                    os.link(video, link)
                except OSError:
                    shutil.copyfile(video, link)
            state, prev = out["inputs"][e]
            sample = {"state": state.half().cpu(), "prev": prev.half().cpu(), "te": str(te), "audio_seed": audio_seed,
                      "sigma": sigmas[e], "r": e, "shape": (latent_t, token_h, token_w), "held": is_held,
                      "item": index, "seed": seed, "size": (width, height), "dit_s": dit_s}
            if e == R - 1:
                sample["latent"] = final.half().cpu()
            else:                                                 # this rollout's own x0_hat[e + 1]: see stored_latent
                sample["exit_latent"] = out["estimates"][e].half().cpu()
            tmp = folder / f"{stem}.tmp"
            torch.save(sample, tmp)
            tmp.replace(folder / f"{stem}.pt")
        done += 1
        if done % 20 == 0 or done <= 2:
            print(f"LOT_H3 kind=accel_collect size={name} sample={stem} done={done} skipped={skipped} of={len(jobs)} "
                  f"dit_s={dit_s:.1f} min={(time.time() - t_start) / 60:.1f} "
                  f"peak_gb={torch.cuda.max_memory_allocated() / 2**30:.1f}", flush=True)
    complete = not stopped
    if complete:
        for e in exits:
            for name in folders:
                (root / f"r{e}" / name / "collect.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    counts = {name: len(list((root / f"r{exits[0]}" / name).glob("train_*.pt"))) for name in folders}
    print(f"LOT_H3 kind=accel_collect_done out={root} new={done} skipped={skipped} of={len(jobs)} complete={int(complete)} "
          f"stopped={stopped or 'no'} min={(time.time() - t_start) / 60:.1f} "
          + " ".join(f"n@{k}={v}" for k, v in counts.items()) + " ok=1", flush=True)
    return {"folders": {e: [root / f"r{e}" / name for name in folders] for e in exits}, "complete": complete}


def collect(args) -> None:
    from h3_musubi import load_vae
    from lot_paths import LOT_H3_VAE

    gpu_setup("accel_h3.py")
    models = load_models(args)
    vae = load_vae(LOT_H3_VAE, device="cuda")
    exits = args.exits if args.exits else [models["iters"] - 1]
    jobs = [parse_job(spec, args.frames) for spec in args.sizes]
    collect_into(args.out, models, vae, jobs, exits=exits, caches=args.caches or default_caches(),
                 holdout=args.holdout, held=args.held, seeds=args.seeds, minutes=args.minutes,
                 min_free_gb=args.min_free_gb, default_frames=args.frames, run=str(args.run))


# ------------------------------------------------------------------------------------------- cache

SAMPLE_KEYS = ("state", "prev", "text", "audio_seed", "sigma", "r", "s", "shape", "latent", "exit_latent")


def stored_latent(d: dict, iters: int) -> torch.Tensor | None:
    """The collected rollout's own estimate after the sample's iteration, the latent a ``two_pass`` refiner reads.

    Why not the replayed one: the DiT runs in bf16 and is numerically chaotic. Replaying an iteration from its
    stored (fp16) inputs gives an estimate about 10% (relative RMS) away from the one the rollout made (``check``
    prints the figure), while the target video is the VAE decode of the rollout's own latent. A refiner fed the
    replayed estimate is scored (and trained) against a target its input does not determine: measured on the
    published stack, 25.4 dB instead of 27.1 dB. At inference the tokens and the estimate come from one forward,
    so there the question does not arise."""
    if d.get("exit_latent") is not None:
        return d["exit_latent"].float()
    if d.get("latent") is not None and d["r"] == iters - 1:
        return d["latent"].float()
    return None


@torch.no_grad()
def cache(args) -> None:
    """Render each sample's DWD head decode to ``OUT/cache/<folder>__<sample>.mp4`` for refiner training, through
    the same ``exit_step`` the renderer uses. Where the refiner's latent is not the stored final one, it is written
    beside the mp4 as ``.lat.pt`` and named in ``pairs.json``: an early exit's own estimate (``stored_latent``),
    or with ``--mode single`` the LoRA-on forward's."""
    import random

    from dir_h3 import use_networks
    from train_dwd_h3 import load, samples, write_mp4

    gpu_setup("accel_h3.py")
    models = load_models(args)
    dit, gen_head, net, R = models["dit"], models["gen_head"], models["net"], models["iters"]
    stack = load_stack(args, dit, refiner=False, half_head=False)
    out = args.out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    groups = []
    for folder in args.data:
        held = [p for p in samples(folder, True) if p.with_suffix(".mp4").exists()][:args.held_per_folder]
        train = [p for p in samples(folder, False) if p.with_suffix(".mp4").exists()]
        groups.append([(p, True) for p in held]
                      + [(p, False) for p in rng.sample(train, min(args.train_per_folder, len(train)))])
    jobs = spread(groups)                                         # a run cut short leaves every size equally done
    pairs, t0, new, late = [], time.time(), 0, False
    for i, (path, held) in enumerate(jobs):
        dst = out / f"{path.parent.name}__{path.stem}.mp4"
        lat = dst.with_suffix(".lat.pt")
        d = None
        if not dst.exists() or not (lat.exists() or (dst.with_suffix(".nolat")).exists()):
            late = late or bool(args.minutes and time.time() - t0 > args.minutes * 60)
            if late:
                continue
            d = load(path, "cuda", SAMPLE_KEYS)
            with use_networks(net, None, gen=True, fake=False):
                h, estimate = exit_step(dit, gen_head, stack, d["state"], d["prev"], d["r"], d["text"], d["sigma"],
                                        d["audio_seed"])
            if stack.mode == "single" or d["r"] != R - 1:
                own = None if stack.mode == "single" else stored_latent(d, R)
                tmp = lat.with_suffix(".tmp")
                torch.save({"latent": (estimate if own is None else own).half().cpu()}, tmp)
                tmp.replace(lat)
            else:
                dst.with_suffix(".nolat").touch()                 # the stored final latent is the refiner's
            write_mp4(dst, stack.head.decode(h, d["s"], *d["shape"]))
            new += 1
        pair = {"dwd": str(dst), "vae": str(path.with_suffix(".mp4")), "held": held, "folder": path.parent.name}
        if lat.exists():
            pair["lat"] = str(lat)
        pairs.append(pair)
        if (i + 1) % 100 == 0:
            print(f"LOT_H3 kind=accel_cache done={i + 1}/{len(jobs)} new={new} min={(time.time() - t0) / 60:.1f}", flush=True)
    complete = len(pairs) == len(jobs)
    (out / "pairs.json").write_text(json.dumps(pairs, indent=1))   # what exists so far is usable as it is
    print(f"LOT_H3 kind=accel_cache_done pairs={len(pairs)} of={len(jobs)} new={new} mode={stack.mode} "
          f"held={sum(p['held'] for p in pairs)} complete={int(complete)} out={out} ok=1", flush=True)
    if not complete:
        raise SystemExit(4)                                       # out of time: a rerun continues


def merge_pairs(args) -> None:
    """Concatenate ``pairs.json`` files (the first wins on a repeated DWD clip) into ``OUT/cache/pairs.json``, so a
    refiner trains on the new sizes together with an earlier cache."""
    seen, merged = set(), []
    for path in args.pairs:
        for pair in json.loads(Path(path).read_text()):
            if pair["dwd"] not in seen and Path(pair["dwd"]).exists() and Path(pair["vae"]).exists():
                seen.add(pair["dwd"])
                merged.append(pair)
    out = args.out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.json").write_text(json.dumps(merged, indent=1))
    folders = sorted({p["folder"] for p in merged})
    print(f"LOT_H3 kind=accel_merge_pairs pairs={len(merged)} held={sum(p['held'] for p in merged)} "
          f"folders={len(folders)} out={out} ok=1", flush=True)


# ------------------------------------------------------------------------------------------- quality

def lpips_clip(net, pred: torch.Tensor, target: torch.Tensor, every: int = 6) -> float:
    """Mean LPIPS over every ``every``-th frame, one frame at a time (VGG at 1344x768 is large)."""
    a, b = pred[0].permute(1, 0, 2, 3)[::every].float(), target[0].permute(1, 0, 2, 3)[::every].float()
    return sum(net(a[i:i + 1], b[i:i + 1]).item() for i in range(a.shape[0])) / a.shape[0]


def score_clip(pred: torch.Tensor, target: torch.Tensor, latent_t: int, net=None) -> dict:
    """PSNR and LPIPS against the VAE clip, and the temporal metrics of ``refine_dwd_h3.evaluate``: flicker inside
    and across latent frames (with the VAE clip's own), the seam ratio at 32 px token edges (with the VAE's) and
    ``tde``, the error of the frame-to-frame change."""
    pred, target = pred.float(), target.float()
    fi, fa = flicker_split(pred, latent_t)
    vi, va = flicker_split(target, latent_t)
    row = {"psnr": psnr(pred, target), "flicker_in": fi, "flicker_in_vae": vi, "flicker_across": fa,
           "flicker_across_vae": va, "seam": seam_ratio(pred), "seam_vae": seam_ratio(target),
           "tde": ((pred[:, :, 1:] - pred[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])).abs().mean().item()
           if pred.shape[2] > 1 else 0.0}
    if net is not None:
        row["lpips"] = lpips_clip(net, pred, target)
    return row


def mean_rows(rows: list[dict]) -> dict:
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


QUALITY_COLUMNS = ("psnr", "lpips", "flicker_in", "flicker_across", "seam", "tde")


def quality_table(results: dict) -> str:
    """Markdown: one row per folder and variant. Flicker and seam show the VAE clip's value in brackets."""
    lines = ["| Clips | Variant | n | PSNR dB | LPIPS | Flicker in (VAE) | Flicker across (VAE) | Seam (VAE) | tde | Decode s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for folder, variants in results.items():
        for name, m in variants.items():
            lines.append(
                f"| {folder} | {name} | {m['n']} | {m['psnr']:.2f} | {m.get('lpips', float('nan')):.4f} | "
                f"{m['flicker_in']:.4f} ({m['flicker_in_vae']:.4f}) | {m['flicker_across']:.4f} ({m['flicker_across_vae']:.4f}) | "
                f"{m['seam']:.3f} ({m['seam_vae']:.3f}) | {m['tde']:.4f} | {m.get('decode_s', float('nan')):.3f} |")
    return "\n".join(lines) + "\n"


@torch.no_grad()
def quality(args) -> None:
    """Score a DWD stack on held-out clips against the VAE decode of the same rollout's final latent."""
    import lpips

    from dir_h3 import use_networks
    from h3_musubi import load_vae
    from lot_paths import LOT_H3_VAE
    from train_dwd_h3 import load, read_mp4, samples, to_uint8

    gpu_setup("accel_h3.py")
    models = load_models(args)
    dit, gen_head, net, R = models["dit"], models["gen_head"], models["net"], models["iters"]
    exit_r = R - 1 if args.exit is None else args.exit
    folders = list(args.data or [])
    vae = None
    if args.sizes or args.vae_exit:
        vae = load_vae(LOT_H3_VAE, device="cuda")
    if args.sizes:                                                # held-out rollouts at these sizes, made here
        jobs = [parse_job(spec, args.frames)[:2] + (0, parse_job(spec, args.frames)[3]) for spec in args.sizes]
        made = collect_into(args.out / "data", models, vae, jobs, exits=[exit_r], held=args.held,
                            caches=args.caches or default_caches(), holdout=args.holdout, default_frames=args.frames,
                            run=str(args.run))
        folders += made["folders"][exit_r]
    if not folders:
        raise SystemExit("quality: give --sizes and/or --data")
    if vae is not None:                                           # the card holds the VAE or the stack, not both
        vae.to("cpu")
        torch.cuda.empty_cache()
    stack = load_stack(args, dit)
    lp = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    args.out.mkdir(parents=True, exist_ok=True)
    results, per_clip, early = {}, {}, []
    for folder in folders:
        files = [p for p in samples(Path(folder), True) if p.with_suffix(".mp4").exists()][:args.held]
        if not files:
            print(f"LOT_H3 kind=accel_quality note=no_held_clips folder={folder}", flush=True)
            continue
        rows: dict = {}
        for k, path in enumerate(files):
            d = load(path, "cuda", SAMPLE_KEYS)
            if d["r"] != exit_r:
                raise SystemExit(f"{path}: collected for exit iteration {d['r']}, the stack decodes {exit_r}")
            target = read_mp4(path.with_suffix(".mp4")).cuda()
            latent_t = d["shape"][0]
            with use_networks(net, None, gen=True, fake=False):
                h, estimate = exit_step(dit, gen_head, stack, d["state"], d["prev"], d["r"], d["text"], d["sigma"],
                                        d["audio_seed"])
            sync()
            t1 = time.perf_counter()
            own = None if stack.mode == "single" else stored_latent(d, R)
            pixels, head_px = stack.decode(h, d["s"], estimate if own is None else own, d["shape"])
            sync()
            decode_s = time.perf_counter() - t1
            rows.setdefault("head", []).append({**score_clip(head_px, target, latent_t, lp), "decode_s": 0.0})
            shown = [target, head_px]
            if stack.refiner is not None:
                rows.setdefault("head + refiner", []).append({**score_clip(pixels, target, latent_t, lp), "decode_s": decode_s})
                shown.append(pixels)
            if args.vae_exit:                                     # scored below, once the VAE is back on the card
                if d.get("exit_latent") is not None:
                    early.append((Path(folder).name, path, "VAE of the exit estimate", d["exit_latent"].cpu(), latent_t))
                if stack.mode == "two_pass":
                    early.append((Path(folder).name, path, "VAE of the replayed iteration", estimate.half().cpu(), latent_t))
            if k < args.videos:
                import imageio

                videos = args.out / "videos"
                videos.mkdir(exist_ok=True)
                frames = to_uint8(torch.cat(shown, dim=-1)[0]).permute(1, 2, 3, 0).cpu().numpy()
                imageio.mimwrite(videos / f"{Path(folder).name}__{path.stem}.mp4", list(frames), fps=24, quality=8,
                                 macro_block_size=16)
                imageio.imwrite(videos / f"{Path(folder).name}__{path.stem}_mid.jpg", frames[len(frames) // 2])
            del target, pixels, head_px, shown
        per_clip[Path(folder).name] = rows
    if early:                                                     # the VAE on the exit iteration's own estimate
        stack.to("cpu")
        torch.cuda.empty_cache()
        vae.to("cuda")
        for name, path, variant, estimate, latent_t in early:
            pixels = vae.decode(estimate.to("cuda", next(vae.parameters()).dtype)).float().clamp(-1, 1)
            target = read_mp4(path.with_suffix(".mp4")).cuda()
            per_clip[name].setdefault(variant, []).append({**score_clip(pixels, target, latent_t, lp), "decode_s": 0.0})
    for name, rows in per_clip.items():
        results[name] = {variant: {**mean_rows(r), "n": len(r)} for variant, r in rows.items()}
        for variant, m in results[name].items():
            print(f"LOT_H3 kind=accel_quality clips={name} variant={variant.replace(' ', '_')} n={m['n']} "
                  + " ".join(f"{k}={m[k]:.4f}" for k in QUALITY_COLUMNS if k in m) + f" decode_s={m['decode_s']:.3f}",
                  flush=True)
    last = "head + refiner" if stack.refiner is not None else "head"
    overall = {k: statistics.mean(v[last][k] for v in results.values()) for k in ("psnr", "lpips")}
    worst = max(v[last]["lpips"] for v in results.values())
    report = {"stack": stack.paths, "exit": exit_r, "mode": stack.mode, "amp": stack.amp, "results": results,
              "overall": overall, "worst_lpips": worst, "variant": last}
    (args.out / "quality.json").write_text(json.dumps(report, indent=1))
    text = (f"Exit iteration {exit_r} of {R} ({stack.mode}), refiner {'bf16' if stack.amp else 'fp32'}. Reference: the "
            f"VAE decode of the full rollout's latent. Held-out prompts, LPIPS (VGG) on every 6th frame.\n\n"
            + quality_table(results))
    (args.out / "QUALITY.md").write_text(text)
    print(text)
    print(f"LOT_H3 kind=accel_quality_done variant={last.replace(' ', '_')} psnr={overall['psnr']:.3f} "
          f"lpips={overall['lpips']:.4f} worst_lpips={worst:.4f} out={args.out} ok=1", flush=True)


# ------------------------------------------------------------------------------------------- check

def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


@torch.no_grad()
def check(args) -> None:
    """GPU parity on one small clip per size. (a) ``dir_rollout`` is ``dir_sample``. (b) The two-pass iteration's
    estimate is the generator's own. (c) Its DWD tokens are those of a full forward with the LoRA on. (d) The
    LoRA does something (the tokens move)."""
    from dir_h3 import dir_sample, use_networks
    from train_dwd_h3 import load_text

    gpu_setup("accel_h3.py")
    models = load_models(args)
    dit, gen_head, net, sigmas, mix = (models[k] for k in ("dit", "gen_head", "net", "sigmas", "mix"))
    stack = load_stack(args, dit, refiner=False)
    _train, held = prompt_lists(args.caches or default_caches(), args.holdout)
    text = load_text(str(held[0])).cuda()
    ok = True
    for spec in args.sizes:
        width, height, _count, frames = parse_job(spec, args.frames)
        latent_t, _th, _tw = token_shape(width, height, frames)
        z = torch.randn(1, 24, latent_t, height // 16, width // 16, device="cuda",
                        generator=torch.Generator("cuda").manual_seed(args.seed))
        with use_networks(net, None, gen=True, fake=False):
            want, _all = dir_sample(dit, gen_head, text, height=height, width=width, latent_t=latent_t,
                                    iters=len(sigmas), seed=args.seed, sigmas=sigmas, mix_state=mix)
            plain = dir_rollout(dit, gen_head, text, z, sigmas, args.seed + 1, mix_state=mix)
            stack.mode = "two_pass"
            two = dir_rollout(dit, gen_head, text, z, sigmas, args.seed + 1, mix_state=mix, stack=stack)
            stack.mode = "single"
            one = dir_rollout(dit, gen_head, text, z, sigmas, args.seed + 1, mix_state=mix, stack=stack)
            store: dict = {}
            with capture_last_layer(dit, store):
                dir_rollout(dit, gen_head, text, z, sigmas, args.seed + 1, mix_state=mix)
            # the same rollout from fp16-rounded noise: how far bf16 rounding alone moves the result
            noisy = dir_rollout(dit, gen_head, text, z.half().float(), sigmas, args.seed + 1, mix_state=mix)
        a, b, c = rel(plain["estimate"], want), rel(two["estimate"], want), rel(two["h"], one["h"])
        moved = rel(one["h"], store["h"])
        good = a <= 1e-4 and b <= 1e-4 and c <= args.tolerance and moved > 10 * max(c, 1e-4)
        ok = ok and good
        print(f"LOT_H3 kind=accel_check size={width}x{height} frames={frames} rollout_vs_dir_sample={a:.3g} "
              f"two_pass_estimate={b:.3g} two_pass_vs_single_tokens={c:.3g} lora_moves_tokens={moved:.3g} "
              f"lora_moves_estimate={rel(one['estimate'], want):.3g} "
              f"fp16_rounded_noise_moves_estimate={rel(noisy['estimate'], want):.3g} ok={int(good)}", flush=True)
    print(f"LOT_H3 kind=accel_check_done ok={int(ok)}", flush=True)
    if not ok:
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------- bench

def upscale_px(pixels: torch.Tensor) -> torch.Tensor:
    """Bicubic x2 per frame of a ``(1, 3, F, h, w)`` clip, 16 frames at a time."""
    frames = pixels[0].transpose(0, 1)
    out = [F.interpolate(frames[i:i + 16].float(), scale_factor=2, mode="bicubic", align_corners=False).clamp(-1, 1)
           for i in range(0, frames.shape[0], 16)]
    return torch.cat(out).transpose(0, 1).unsqueeze(0)


@torch.no_grad()
def region_decode(stack: DwdStack, h_hr: torch.Tensor, state_hr: torch.Tensor, hr_index: torch.Tensor,
                  latent: torch.Tensor, weight: torch.Tensor, lr_px: torch.Tensor, latent_t: int,
                  margin: int = 4) -> torch.Tensor:
    """Two-track composite with no VAE: the region's HR tokens through the DWD head and refiner, pasted on the
    LR track's DWD pixels (x2). ``(1, 3, F, H, W)`` at full resolution.

    ``h_hr`` are the DWD tokens of the region's HR rows (``hr_index`` order), ``state_hr`` the HR state the last
    forward read, ``latent`` the composite HR latent ``dir_two_track_sample`` returns (the region's estimate
    inside, the LR latent x2 outside) and ``weight`` its paste weight on the HR latent grid. Every HR token
    outside the region takes its patch from the x2 LR picture, so the head's overlap-add needs no special case
    at the region's edge. The refiner runs on the region's bounding box only (plus ``margin`` latent pixels)."""
    from dwd_h3 import blend_tokens, frames_of, pixels_to_tokens, pixels_to_tokens_ext, token_classes, tokens_to_pixels
    from refine_dwd_h3 import refine_clip

    head = stack.head
    up = upscale_px(lr_px)
    height, width = up.shape[-2:]
    token_h, token_w = height // 32, width // 32
    rows = pack_rows(state_hr.float())[hr_index]
    pred = head(h_hr, rows, token_classes(latent_t, token_h, token_w, h_hr.device)[hr_index])
    if head.halo is None:
        tokens = pixels_to_tokens(up, latent_t).to(pred.dtype)
        tokens[hr_index] = pred
        px = tokens_to_pixels(tokens, latent_t, token_h, token_w).float().clamp(-1, 1)
    else:
        tokens = pixels_to_tokens_ext(up, latent_t, head.halo).to(pred.dtype)
        tokens[hr_index] = pred
        px = blend_tokens(tokens, latent_t, token_h, token_w, head.halo, head.blend).clamp(-1, 1)
    del tokens
    w = weight[0, 0].float()                                      # (T, H / 16, W / 16)
    ys, xs = torch.nonzero(w.amax(0) > 0, as_tuple=True)
    if len(ys) == 0:
        return up
    y0, y1 = max(0, int(ys.min()) - margin), min(w.shape[1], int(ys.max()) + 1 + margin)
    x0, x1 = max(0, int(xs.min()) - margin), min(w.shape[2], int(xs.max()) + 1 + margin)
    box = (slice(None), slice(None), slice(None), slice(16 * y0, 16 * y1), slice(16 * x0, 16 * x1))
    crop = px[box]
    if stack.refiner is not None:
        with torch.autocast(crop.device.type, dtype=torch.bfloat16, enabled=stack.amp):
            crop = refine_clip(stack.refiner, crop.contiguous(), latent[..., y0:y1, x0:x1].contiguous(),
                               chunk=stack.chunk or refiner_chunk(*crop.shape[-2:])).float()
    frame_of = [k for k, n in enumerate(frames_of(latent_t)) for _ in range(n)]
    w_px = F.interpolate(w[frame_of, y0:y1, x0:x1][:, None], scale_factor=16, mode="bilinear", align_corners=False)
    w_px = w_px.transpose(0, 1).unsqueeze(0)                      # (1, 1, F, h, w)
    out = up.clone()
    out[box] = w_px * crop + (1.0 - w_px) * up[box]
    return out


def bench_table(root: Path) -> str:
    """``root/*/meta.json`` -> one markdown table (``BENCH.md``). Medians over prompts after the first (warm-up)."""
    lines = ["| Canvas | Frames | Variant | Total s | DiT s | Decode s | x real time | PSNR dB | LPIPS | Peak GB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    notes = set()
    for meta_path in sorted(root.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        skip = 1 if len(meta["prompts"]) > 1 else 0
        for name in meta["columns"]:
            def med(key, digits=2):
                values = meta.get(key, {}).get(name, [])
                values = [v for v in (values[skip:] or values) if v is not None]
                return round(statistics.median(values), digits) if values else None
            total = med("seconds")
            label = LABEL[name] + (" (out of memory)" if name in meta.get("oom", []) else "")
            if name == "dir_dwd_early":
                label += f" (after iteration {meta['early_exit'] + 1} of {meta['iters']})"
            if name.startswith("tt_") and meta.get("hr_share"):
                label += f" ({100 * statistics.median(meta['hr_share']):.0f}% region)"
            row = (meta["canvas"], meta["frames"], label, total, med("dit_seconds"), med("decode_seconds"),
                   round(meta["frames"] / 24.0 / total, 2) if total else None, med("psnr"), med("lpips", 4),
                   med("peak_gb", 1))
            lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
        notes.add(meta.get("gpu", ""))
    text = "\n".join(lines) + (
        "\n\nMedians over prompts after the first (warm-up). Total = DiT + decode, pixels left on the GPU. "
        "x real time = clip seconds at 24 fps / total. PSNR and LPIPS: the DWD pixels against the VAE's for the same "
        "latent (the full rollout's latent for an early exit; the same x2 for half res; the VAE composite of the same "
        "two latents for two-track), not timed. "
        f"GPU: {', '.join(sorted(n for n in notes if n))}.\n")
    (root / "BENCH.md").write_text(text)
    return text


@torch.no_grad()
def bench(args) -> None:
    from dir_h3 import use_networks
    from h3_musubi import load_vae
    from lot_paths import LOT_H3_VAE
    from train_dwd_h3 import load_text

    gpu_setup("accel_h3.py")
    models = load_models(args)
    dit, gen_head, net, sigmas, mix, R = (models[k] for k in ("dit", "gen_head", "net", "sigmas", "mix", "iters"))
    wanted = [v for v in VARIANTS if v in args.variants.split(",")]
    unknown = set(args.variants.split(",")) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"--variants: {sorted(unknown)} not in {VARIANTS}")
    if "dir_dwd_early" in wanted and not args.early_lora:
        wanted.remove("dir_dwd_early")
    stack = load_stack(args, dit) if any("dwd" in v for v in wanted if v != "dir_dwd_early") else None
    early = load_stack(args, dit, prefix="early_") if "dir_dwd_early" in wanted else None
    vae = load_vae(LOT_H3_VAE, device="cpu")
    vae_dtype = next(vae.parameters()).dtype
    two_track = None
    if any(v.startswith("tt_") for v in wanted):
        from dir_two_track_h3 import MPEG_MAPS, MPEG_REGION, mpeg_file
        from mixed_res import RegionHead
        from two_track_h3 import load_maps, load_region

        saved = load_region(mpeg_file(args.region, "LOT_TT_REGION", MPEG_REGION))
        region = RegionHead(saved["hidden"]).cuda().eval()
        region.load_state_dict(saved["state"])
        two_track = {"maps": load_maps(mpeg_file(args.maps, "LOT_TT_MAPS", MPEG_MAPS)), "region": region,
                     "block": saved["block"]}
    lp = None
    if args.score:
        import lpips

        lp = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    _train, held = prompt_lists(args.caches or default_caches(), args.holdout)
    prompts = held[:args.prompts]

    def vae_decode(latent: torch.Tensor) -> torch.Tensor:
        return vae.decode(latent.to("cuda", vae_dtype)).clamp(-1, 1)

    def place(vae_on: bool, stack_on, dwd: bool) -> None:
        """Not timed: a 32 GB card holds the DiT with the VAE or with a DWD stack. ``tt_dwd`` needs both."""
        for other in (stack, early):
            if other is not None and (other is not stack_on or not dwd):
                other.to("cpu")
        if not vae_on:
            vae.to("cpu")
        torch.cuda.empty_cache()
        if vae_on:
            vae.to("cuda")
        if dwd:
            stack_on.to("cuda")

    def latent_x2(latent: torch.Tensor) -> torch.Tensor:          # only reached when the region is empty
        return F.interpolate(latent.float(), scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)

    for spec in args.sizes:
        width, height, _count, frames = parse_job(spec, args.frames)
        latent_t, token_h, token_w = token_shape(width, height, frames)
        columns = [v for v in wanted if not (v.startswith(("lr_", "tt_")) and (width % 64 or height % 64))]
        meta = {"run": str(args.run), "canvas": f"{width}x{height}", "frames": frames, "latent_t": latent_t, "iters": R,
                "sigmas": sigmas, "columns": columns, "prompts": [], "stack": stack.paths if stack else None,
                "mode": stack.mode if stack else None, "early": early.paths if early else None,
                "early_exit": early.exit_r if early else None, "early_mode": early.mode if early else None,
                "switch": args.switch, "hr_share": [], "oom": [], "gpu": torch.cuda.get_device_name(),
                **{key: {c: [] for c in columns} for key in ("seconds", "dit_seconds", "decode_seconds", "psnr",
                                                              "lpips", "peak_gb")}}
        out_dir = args.out / f"{width}x{height}" if frames == 39 else args.out / f"{width}x{height}_f{frames}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for index, te in enumerate(prompts):
            text = load_text(str(te)).cuda()
            seed = args.seed + index
            meta["prompts"].append(Path(te).name)
            for name in columns:
                lr = name.startswith(("lr_", "tt_"))
                w, h = (width // 2, height // 2) if lr else (width, height)
                shape = (latent_t, h // 32, w // 32)
                z = torch.randn(1, 24, latent_t, h // 16, w // 16, device="cuda",
                                generator=torch.Generator("cuda").manual_seed(seed))
                use = early if name == "dir_dwd_early" else stack
                pixels = reference = scored = info = out = latent = tap = lr_px = None
                try:
                    place(vae_on=name in NEEDS_VAE, stack_on=use, dwd="dwd" in name)
                    torch.cuda.reset_peak_memory_stats()
                    sync()
                    start = time.perf_counter()
                    with use_networks(net, None, gen=True, fake=False):
                        if name in ("dir_vae", "lr_vae"):
                            out = dir_rollout(dit, gen_head, text, z, sigmas, seed + 1, mix_state=mix)
                            sync()
                            sampled = time.perf_counter()
                            pixels = vae_decode(out["estimate"])
                        elif name in ("dir_dwd", "lr_dwd", "dir_dwd_early"):
                            out = dir_rollout(dit, gen_head, text, z, sigmas, seed + 1, mix_state=mix, stack=use,
                                              exit_r=use.exit_r)
                            sync()
                            sampled = time.perf_counter()
                            pixels, _head = use.decode(out["h"], out["state"], out["estimate"], shape)
                            scored = pixels
                        else:
                            from dir_two_track_h3 import dir_two_track_sample, pixel_compose_crop

                            tap = {"first": stack.first, "lora": stack.on} if name != "tt_vae" else None
                            latent, info = dir_two_track_sample(
                                dit, gen_head, text, latent_x2, height=height, width=width, sigmas=sigmas, mix_state=mix,
                                switch=args.switch, seed=seed, latent_t=latent_t, maps=two_track["maps"],
                                region=two_track["region"], block=two_track["block"] or 25, switch_upscale="latent",
                                final="pixel", dwd=tap)
                            sync()
                            sampled = time.perf_counter()
                            lr_px = None
                            if tap is not None:
                                count = shape[0] * shape[1] * shape[2]
                                lr_px, _head = stack.decode(tap["h"][:count], info["state_lr"], info["lr"], shape)
                            if "weight" not in info:              # no region: the LR track is the picture
                                pixels = lr_px if lr_px is not None else vae_decode(info["lr"])
                            elif name == "tt_dwd2":
                                pixels = region_decode(stack, tap["h"][count:], info["state_hr"], info["hr_index"], latent,
                                                       info["weight"], lr_px, latent_t)
                            else:
                                pixels = pixel_compose_crop(vae, latent, info["lr"], info["weight"],
                                                            lr_px=None if lr_px is None else lr_px.to("cpu", torch.float16))
                            scored = pixels if tap is not None else None
                            if index and name == "tt_vae":
                                meta["hr_share"].append(round(float(info["mask"].float().mean()), 4))
                        if name.startswith("lr_") or (name.startswith("tt_") and "weight" not in info):
                            pixels = upscale_px(pixels)
                            scored = pixels if scored is not None else None
                    sync()
                    done = time.perf_counter()
                    dit_s = info["dit_s"] if name.startswith("tt_") else sampled - start
                    peak = round(torch.cuda.max_memory_allocated() / 2**30, 1)
                    if pixels.shape[2:] != (frames, height, width):
                        raise RuntimeError(f"{name}: pixels {tuple(pixels.shape)} for a {frames}-frame {width}x{height} clip")
                    if args.save and index == min(1, len(prompts) - 1):
                        from two_track_h3 import save_clip

                        save_clip(pixels.to("cpu", torch.float16), out_dir / f"p{index}_{name}.mp4")
                    values = {"seconds": round(dit_s + done - sampled, 3), "dit_seconds": round(dit_s, 3),
                              "decode_seconds": round(done - sampled, 3), "peak_gb": peak, "psnr": None, "lpips": None}
                    if lp is not None and scored is not None:
                        scored = scored.float().cpu()             # room for the VAE: the stack leaves the card
                        pixels = None
                        place(vae_on=True, stack_on=None, dwd=False)
                        if name.startswith("tt_") and "weight" in info:      # the VAE path on the same two latents
                            reference = pixel_compose_crop(vae, latent, info["lr"], info["weight"]).float()
                        else:
                            with use_networks(net, None, gen=True, fake=False):
                                if name == "dir_dwd_early":       # the full rollout: what the skipped iterations give
                                    full = dir_rollout(dit, gen_head, text, z, sigmas, seed + 1, mix_state=mix)["estimate"]
                                else:
                                    full = info["lr"] if name.startswith("tt_") else out["estimate"]
                            reference = vae_decode(full).float()
                            if reference.shape != scored.shape:   # the half-resolution rows are scored after the x2
                                reference = upscale_px(reference)
                        scored, reference = scored.cuda(), reference.cuda()
                        values["psnr"] = round(psnr(scored, reference), 3)
                        values["lpips"] = round(lpips_clip(lp, scored, reference), 4)
                    print(f"LOT_H3 kind=accel_bench size={width}x{height} prompt={index} variant={name} "
                          f"seconds={values['seconds']:.2f} dit_s={dit_s:.2f} decode_s={done - sampled:.2f} peak_gb={peak}"
                          + (f" psnr={values['psnr']} lpips={values['lpips']}" if reference is not None else ""), flush=True)
                except torch.OutOfMemoryError:                    # e.g. the DiT, the VAE and a DWD stack at 1344x768
                    values = {key: None for key in ("seconds", "dit_seconds", "decode_seconds", "peak_gb", "psnr", "lpips")}
                    meta["oom"].append(name)
                    print(f"LOT_H3 kind=accel_bench size={width}x{height} prompt={index} variant={name} oom=1", flush=True)
                for key, value in values.items():
                    meta[key][name].append(value)
                del pixels, reference, scored, info, out, latent, tap, lr_px
                torch.cuda.empty_cache()
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    text = bench_table(args.out)
    print(text)
    print(f"LOT_H3 kind=accel_bench_done out={args.out} ok=1", flush=True)


# ------------------------------------------------------------------------------------------- main

def add_stack(p, prefix: str = "", required: bool = True) -> None:
    flag = "--" + prefix.replace("_", "-")
    what = "the early-exit stack's" if prefix else "the stack's"
    p.add_argument(flag + "lora", type=Path, required=required, help=f"{what} dwd_ft_lora.pt")
    p.add_argument(flag + "head", type=Path, required=required, help=f"{what} dwd_head_ridge.pt")
    p.add_argument(flag + "refiner", type=Path, default=None, help=f"{what} refiner_best.pt")
    p.add_argument(flag + "exit", type=int, default=None,
                   help="DiR iteration (0-based) whose tokens the stack decodes (default: the last)")
    p.add_argument(flag + "mode", choices=MODES, default="two_pass", help="see the module docstring")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="phase", required=True)

    def common(p, sizes: bool = True, sizes_required: bool = True):
        p.add_argument("--run", default=DEFAULT_RUN, help="train_dir_h3 run directory, or owner/repo[/sub] on Hugging Face")
        p.add_argument("--tag", default="")
        p.add_argument("--caches", type=Path, nargs="+", default=None,
                       help="H3 caches whose prompts are used (default: $ACCEL_CACHES, else the stills and clips caches)")
        p.add_argument("--holdout", type=int, default=8, help="held-out prompts per cache, as the DiR run split them")
        p.add_argument("--frames", type=int, default=39, help="pixel frames of a clip (1, 5, 22, 39, 56, ...)")
        if sizes:
            p.add_argument("--sizes", nargs="+", required=sizes_required, default=None, metavar="WxH",
                           help="canvases in pixels, each side a multiple of 32, e.g. 768x512 1344x768")

    c = sub.add_parser("collect", help="multi-resolution rollouts with VAE targets")
    common(c)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--exits", type=int, nargs="+", default=None, help="iterations to store inputs for (default: the last)")
    c.add_argument("--held", type=int, default=8, help="held-out clips per size")
    c.add_argument("--seeds", type=int, default=1, help="noise seeds per training prompt")
    c.add_argument("--minutes", type=float, default=0, help="stop cleanly after this long (0: no limit); a rerun resumes")
    c.add_argument("--min-free-gb", type=float, default=40.0, help="stop when the disk has less than this free")

    k = sub.add_parser("cache", help="DWD decodes for refiner training")
    common(k, sizes=False)
    add_stack(k)
    k.add_argument("--data", type=Path, nargs="+", required=True, help="collect folders (one per size)")
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--train-per-folder", type=int, default=500)
    k.add_argument("--held-per-folder", type=int, default=8,
                   help="held-out clips per size the refiner run evaluates on (every eval scores all of them)")
    k.add_argument("--minutes", type=float, default=0, help="stop (exit 4) after this long; a rerun resumes")

    m = sub.add_parser("merge-pairs", help="join pairs.json files for one refiner run")
    m.add_argument("--pairs", type=Path, nargs="+", required=True)
    m.add_argument("--out", type=Path, required=True)

    q = sub.add_parser("quality", help="LPIPS / PSNR / temporal metrics against the VAE decode")
    common(q, sizes_required=False)
    add_stack(q)
    q.add_argument("--data", type=Path, nargs="+", default=None, help="collect folders to score (their held-out clips)")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--held", type=int, default=8, help="held-out clips per size")
    q.add_argument("--videos", type=int, default=1, help="side-by-side videos per size: VAE | head | refined")
    q.add_argument("--vae-exit", action="store_true",
                   help="also score the VAE decode of the exit iteration's estimate (what skipping the later iterations "
                        "costs), and of the same iteration replayed (the DiT's numerical noise: the last exit's row "
                        "would be 0 for a deterministic model)")
    q.add_argument("--fp32", action="store_true", help="refiner in fp32 instead of bf16 autocast")
    q.add_argument("--chunk", type=int, default=0, help="frames per refiner batch (0: by clip size, refiner_chunk)")

    b = sub.add_parser("bench", help="end-to-end speed, DiT and decode split")
    common(b)
    add_stack(b)
    add_stack(b, prefix="early_", required=False)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--variants", default=",".join(VARIANTS),
                   help=f"comma list of {','.join(VARIANTS)}")
    b.add_argument("--prompts", type=int, default=3, help="held-out prompts; the first is warm-up")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--switch", type=int, default=2, help="two-track: iterations at half resolution before the region opens")
    b.add_argument("--maps", default=None)
    b.add_argument("--region", default=None)
    b.add_argument("--score", action="store_true", help="PSNR and LPIPS of each DWD variant against the VAE decode (not timed)")
    b.add_argument("--save", action="store_true", help="write one clip per variant")
    b.add_argument("--fp32", action="store_true")
    b.add_argument("--chunk", type=int, default=0, help="frames per refiner batch (0: by clip size)")
    b.add_argument("--channels-last", action="store_true")

    t = sub.add_parser("table", help="rebuild BENCH.md from the meta.json files under a folder")
    t.add_argument("out", type=Path)

    h = sub.add_parser("check", help="GPU parity of the DWD tail and the sampler")
    common(h, sizes_required=False)
    add_stack(h)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--tolerance", type=float, default=2e-2,
                   help="relative gap allowed between the two-pass and the one-pass tokens (bf16 blocks)")

    args = parser.parse_args()
    if args.phase == "table":
        print(bench_table(args.out))
        return
    if args.phase == "merge-pairs":
        merge_pairs(args)
        return
    if args.phase == "check":
        args.sizes = args.sizes or ["384x256", "512x384"]
        args.frames = 22 if args.frames == 39 else args.frames
    try:
        from setproctitle import setproctitle
        what = " ".join(args.sizes)[:36] if getattr(args, "sizes", None) else ""
        setproctitle(" ".join(x for x in (f"accel:{args.phase}", getattr(args, "out", Path("")).name, what) if x))
    except ImportError:
        pass
    {"collect": collect, "cache": cache, "quality": quality, "bench": bench, "check": check}[args.phase](args)


if __name__ == "__main__":
    main()
