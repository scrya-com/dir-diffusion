"""Inference-only probes of a DiR generator on held-out prompts: does it refine, and does it READ its estimate?

    # references, once per (cache, crop, holdout): base H3 at 20 steps and at R Euler steps, same z as below
    python3 lot/probe_dir_h3.py --out lot/runs/probe --name ref --reference

    # a run (sigmas / mix_state from its config.json), or the untrained generator with no --run
    python3 lot/probe_dir_h3.py --out lot/runs/probe --name step100 --run lot/runs/train_dir_h3 --tag 100
    python3 lot/probe_dir_h3.py --out lot/runs/probe --name base0 --sigmas grid:6 --mix-state

    # decode a picture grid: reference columns, then r = 1..R of each name
    python3 lot/probe_dir_h3.py --out lot/runs/probe --decode base0 step100

Per item it writes the rollout ``x0_hat[1..R]`` (``NAME.pt``) and, in ``NAME.json``:

* ``own_to_dense``: mean |x0_hat[r] - dense20| per iteration (same z; a distance between two unrelated
  images of one prompt is about 0.8, so this says "same picture or not", not quality).
* ``feed``: iteration r (r >= 2) is handed a CONTROLLED estimate instead of its own, and the output's
  distance to the 20-step latent is recorded: ``zeros``, ``dense20`` (the answer), ``wrong`` (another
  item's real latent). A generator that reads its estimate moves toward ``dense20`` when fed it and away
  when fed ``wrong``; one that ignores it returns the same distance for all three. The estimate goes in
  the way the run uses it: through ``x0_proj`` and, with ``--mix-state``, through the state.

``--x0-proj-copy ALPHA`` (no ``--run``) sets ``x0_proj := ALPHA * video_patch_proj``, the obvious
non-zero init of the paper's pinned form.

``--frames T`` (5n+2 latent frames, e.g. 12) probes clips instead of stills: the items give the text and the
crop, the noise is clip-shaped, ``wrong`` is skipped as a still repeated over time, and each row also records
``motion`` (``train_dir_h3.motion``: mean change between VAE clip-start frames; compare with the reference's
``dense20_motion``). ``--decode`` then writes ``probe_<names>_p<i>.mp4`` (columns side by side, 24 fps) and a
sheet of the first, middle and last frame per item.

Not in ``gpu_guard``'s list on purpose: it is meant to run on a second card next to a training job.
Pick the card with ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default=None, help="basename of the .pt / .json written")
    parser.add_argument("--reference", action="store_true", help="write the dense20 / euler columns and stop")
    parser.add_argument("--run", type=Path, default=None, help="a train_dir_h3.py run dir (default: base H3, zero head)")
    parser.add_argument("--tag", default="", help="--save-every step of --run (default: the final save)")
    parser.add_argument("--sigmas", default=None, help="default: the run's config.json sigmas, else 1")
    parser.add_argument("--mix-state", action="store_true", default=None, help="default: the run's config.json mix_state")
    parser.add_argument("--x0-proj-copy", type=float, default=None, metavar="ALPHA")
    parser.add_argument("--ref-name", default="ref", help="where the dense20 latents are (from --reference)")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--items", type=int, default=None, help="first N held-out items (default: all)")
    parser.add_argument("--max-latent-hw", type=int, nargs=2, default=[48, 48])
    parser.add_argument("--frames", type=int, default=1, help="latent frames (1 = stills; 5n+2 for clips)")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--swap", type=int, default=16)
    parser.add_argument("--ema", action="store_true", help="load the run's averaged weights (--ema runs)")
    parser.add_argument("--decode", nargs="+", default=None, metavar="NAME")
    parser.add_argument("--cell", type=int, default=384)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.decode:
        return decode(args)
    if not args.name:
        raise SystemExit("--name is required")

    import train_h3
    from dir_h3 import DirHead, generator_step, load_dir_run, next_state, parse_sigmas, probe_temb_dim, use_networks
    from h3 import patchify
    from h3_musubi import load_dit, predict, sigma_schedule

    train_h3.MAX_LATENT_HW = tuple(args.max_latent_hw)
    train_h3.CROP_MULTIPLE = 2
    dev = torch.device("cuda")
    _train, held = train_h3.list_items(args.cache or train_h3.CACHE, args.holdout)
    count = min(args.items or len(held), len(held))
    dit = load_dit(device=dev, swap=args.swap)
    R = args.iterations
    config, net = {}, None
    if args.run is not None:
        head, net, config = load_dir_run(args.run, dit, tag=f"_{args.tag}" if args.tag else "", ema=args.ema)
    else:
        head = DirHead(dit.config.hidden_size, probe_temb_dim(dit), R).to(dev).eval()
        if args.x0_proj_copy is not None:
            with torch.no_grad():
                head.x0_proj.weight.copy_(args.x0_proj_copy * dit.video_patch_proj.weight.float())
    sigmas = parse_sigmas(args.sigmas or config.get("sigmas") or "1", R)
    mix = bool(config.get("mix_state", False)) if args.mix_state is None else bool(args.mix_state)

    def audio(seed):
        return torch.Generator(dev).manual_seed(seed)

    def mad(a, b):
        return round(float((a.float() - b.float()).abs().mean()), 4)

    def euler(z, text, grid, seed):
        x, generator = z.clone(), audio(seed)
        for s, s_next in zip(grid[:-1], grid[1:]):
            x0 = x + s * predict(dit, x, 1.0 - s, text, audio_generator=generator).float()
            x = (s_next / s) * x + (1.0 - s_next / s) * x0
        return x

    def like(other, x0):
        """Another item's latent at ``x0``'s shape (buckets differ): it only has to be the wrong picture."""
        if other.shape == x0.shape:
            return other
        return torch.nn.functional.interpolate(other[:, :, 0].float(), size=x0.shape[-2:], mode="bilinear")[:, :, None].to(x0.dtype)

    from train_dir_h3 import load_item, motion

    items = [load_item(held[i], random.Random(1000 + i), dev, args.frames) for i in range(count)]
    reference = None if args.reference else torch.load(args.out / f"{args.ref_name}.pt")
    keep, rows = {}, []
    meta = {"run": None if args.run is None else str(args.run), "tag": args.tag, "sigmas": sigmas, "mix_state": mix,
            "x0_proj_copy": args.x0_proj_copy, "max_latent_hw": list(args.max_latent_hw), "holdout": args.holdout, "frames": args.frames,
            "swap": args.swap, "gpu": torch.cuda.get_device_name(0)}
    with torch.no_grad():
        for i, (x0, text) in enumerate(items):
            z = torch.randn(x0.shape, device=dev, generator=torch.Generator(dev).manual_seed(i))
            row = {"item": i}
            if args.reference:
                with use_networks(net, None, gen=False, fake=False):
                    keep[(i, "dense20")] = euler(z, text, [float(s) for s in sigma_schedule(20)], 7 * i).cpu()
                    keep[(i, f"euler{R}")] = euler(z, text, [float(s) for s in sigma_schedule(R)], 7 * i).cpu()
                keep[(i, "real")] = x0.cpu()
                row[f"euler{R}_to_dense"] = mad(keep[(i, f"euler{R}")], keep[(i, "dense20")])
                if args.frames > 1:
                    del keep[(i, "real")]
                    row["dense20_motion"] = round(motion(keep[(i, "dense20")].float()), 4)
                    row[f"euler{R}_motion"] = round(motion(keep[(i, f"euler{R}")].float()), 4)
            else:
                dense = reference[(i, "dense20")].to(dev)
                wrong = like(items[(i + 1) % count][0], x0)
                with use_networks(net, None, gen=True, fake=False):
                    estimate, state, own = torch.zeros_like(z), z, []
                    for r in range(R):
                        estimate = generator_step(dit, head, state, estimate, r, text, sigma=sigmas[r],
                                                  audio_generator=audio(7 * i))
                        own.append(estimate)
                        keep[(i, f"{args.name}_r{r + 1}")] = estimate.cpu()
                        if r + 1 < R:
                            state = next_state(state, estimate, sigmas[r], sigmas[r + 1], z if mix else None)
                    row["own_to_dense"] = [mad(o, dense) for o in own]
                    row["own_change"] = [mad(own[r], own[r - 1]) for r in range(1, R)]
                    if args.frames > 1:
                        row["motion"] = [round(motion(o.float()), 4) for o in own]
                    row["feed"] = {}
                    for r in range(1, R if args.frames == 1 else 1):
                        fed = {}
                        for label, value in (("zeros", torch.zeros_like(z)), ("dense20", dense), ("wrong", wrong)):
                            # with the Euler state there is no state for an arbitrary estimate: the feed then
                            # goes through x0_proj only, on the model's own state for that iteration
                            fed_state = next_state(z, value, 1.0, sigmas[r], z) if mix else own_state(
                                z, own, sigmas, r, next_state)
                            out = generator_step(dit, head, fed_state, value, r, text, sigma=sigmas[r],
                                                 audio_generator=audio(7 * i))
                            fed[label] = {"to_dense": mad(out, dense), "to_wrong": mad(out, wrong)}
                        row["feed"][f"r{r + 1}_sigma{sigmas[r]:.3f}"] = fed
            rows.append(row)
            print(json.dumps(row), flush=True)
            torch.save(keep, args.out / f"{args.name}.pt")
            (args.out / f"{args.name}.json").write_text(json.dumps({"meta": meta, "items": rows}, indent=1))
    print(f"LOT_H3 kind=probe_dir name={args.name} items={count} ok=1", flush=True)


def own_state(z, own, sigmas, r, next_state):
    """The Euler state iteration ``r`` sees in the model's own rollout."""
    state = z
    for k in range(r):
        state = next_state(state, own[k], sigmas[k], sigmas[k + 1])
    return state


def decode(args) -> None:
    from PIL import Image, ImageDraw

    from h3_musubi import load_vae
    from lot_paths import LOT_H3_VAE
    from render_h3 import to_image

    keep = torch.load(args.out / f"{args.ref_name}.pt")
    columns = sorted({c for _i, c in keep if c != "real"}, key=lambda c: (c != "dense20", c))
    for name in args.decode:
        extra = torch.load(args.out / f"{name}.pt")
        keep.update(extra)
        columns += sorted({c for _i, c in extra})
    rows = sorted({i for i, _c in keep})
    cell = args.cell
    decoder = load_vae(LOT_H3_VAE)
    if next(iter(keep.values())).shape[2] > 1:
        return decode_clips(args, keep, columns, rows, decoder)
    sheet = Image.new("RGB", (cell * len(columns), (cell + 18) * len(rows)), "black")
    draw = ImageDraw.Draw(sheet)
    with torch.no_grad():
        for y, i in enumerate(rows):
            for x, column in enumerate(columns):
                if (i, column) not in keep:
                    continue
                image = to_image(decoder.decode(keep[(i, column)].to("cuda", next(decoder.parameters()).dtype)))
                scale = cell / max(image.size)
                image = image.resize((int(image.width * scale), int(image.height * scale)))
                sheet.paste(image, (x * cell, y * (cell + 18) + 18))
                draw.text((x * cell + 3, y * (cell + 18) + 3), column, fill="white")
    path = args.out / ("probe_" + "_".join(args.decode) + ".png")
    sheet.save(path)
    print(f"LOT_H3 kind=probe_dir_decode out={path} columns={len(columns)} rows={len(rows)} ok=1", flush=True)


def decode_clips(args, keep, columns, rows, decoder) -> None:
    import imageio
    import numpy as np
    from PIL import Image, ImageDraw

    from render_stream_h3 import frames_of

    stem = "probe_" + "_".join(args.decode)
    with torch.no_grad():
        for i in rows:
            clips = []
            for column in columns:
                frames = frames_of(decoder.decode(keep[(i, column)].to("cuda", next(decoder.parameters()).dtype)))
                labelled = []
                for frame in frames:
                    image = Image.fromarray(frame)
                    ImageDraw.Draw(image).text((6, 6), column, fill="white")
                    labelled.append(np.asarray(image))
                clips.append(np.stack(labelled))
            wide = np.concatenate(clips, axis=2)
            imageio.mimwrite(args.out / f"{stem}_p{i}.mp4", list(wide), fps=24, quality=8, macro_block_size=16)
            picks = (0, len(wide) // 2, len(wide) - 1)
            Image.fromarray(np.concatenate([wide[k] for k in picks], axis=0)).save(args.out / f"{stem}_p{i}_sheet.jpg", quality=85)
    print(f"LOT_H3 kind=probe_dir_decode out={args.out / stem}_p*.mp4 columns={len(columns)} rows={len(rows)} ok=1", flush=True)


if __name__ == "__main__":
    main()
