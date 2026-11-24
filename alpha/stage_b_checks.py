# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""Two checks for Stage B (docs/ALPHA_DWD.md):

``split``: soft-pixel alpha error split by whether colour can tell subject from background at all. On the sigma-0.6
eval clips (the same clips, noise and backgrounds as ``train_stage_b.evaluate``), a soft pixel is *separable* where
the subject's straight colour F and the background B differ, ``||F - B|| > --sep`` (RGB in [0, 1]), and *ambiguous*
otherwise: there the composite ``a F + (1 - a) B`` barely depends on ``a``, so no matting can recover it. Reports
soft_mae on each part and the ambiguous share, for the a5 head (Stage A: LoRA off) and the Stage B checkpoint.

``deploy``: what a user gets. pi-H3 2-step from noise (``max_raw_t`` 1.0, the camera-lock schedule; the musubi path
has no ``cam_static`` dial) on smoke / glass / person captions; at the last step the RGB pass decodes with the fused
DWD head and the alpha pass gives alpha from the Stage B checkpoint, and the a5 head reads the same RGB pass. Writes an
RGBA contact sheet over a checker, a WebM with alpha per clip, and end-to-end seconds with and without the alpha pass.

    ALPHA_VAE_DEVICE=cuda:1 python -m alpha.stage_b_checks split  --b /media/2TB/alpha/runs/b1/stage_b_last.pt
    python -m alpha.stage_b_checks deploy --b /media/2TB/alpha/runs/b1/stage_b_last.pt --out /media/2TB/alpha/deploy_b1
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .train_pih3_alpha import DATA, EVAL_ALIGNED_SIGMA, FRAMES, LOT, SCENES, TE, sample_clip, setup, te_path

A5 = "/media/2TB/alpha/runs/a5/alpha_dwd_head.pt"


def load_both(env, b_ckpt, k=8, rank=32):
    from .pih3_alpha import AlphaDWDHead
    from .train_stage_b import AlphaPath, TailLoRA
    st = torch.load(b_ckpt, map_location="cuda")
    lora = TailLoRA(env["model"]["dit"], st.get("rank", rank), st.get("k", k))
    lora.load_state_dict(st["lora"])
    head_b = AlphaDWDHead().cuda().eval()
    head_b.load_state_dict(st["head"])
    head_a = AlphaDWDHead().cuda().eval()
    head_a.load_state_dict(torch.load(A5, map_location="cuda")["head"])
    return lora, AlphaPath(env["model"]["head"]), head_a, head_b


@torch.no_grad()
def split(args):
    from .alpha_math import ea_score_to_alpha
    from .data import SOFT_HI, SOFT_LO
    from .train_stage_b import two_pass
    env = setup(2)
    lora, path, head_a, head_b = load_both(env, args.b)
    subset = json.loads((TE / "subset.json").read_text())
    rows = {json.loads(l)["image"]: json.loads(l) for l in open(DATA / "t2rgba_train.jsonl")}
    eval_rows = [rows[i] for i in subset["eval"] if te_path(i).is_file()]
    scenes = torch.load(SCENES) if SCENES.is_file() else None
    kinds = ("dark", "light", "gradient", "scene", "flat", "padded")
    modes = ("affine", "drift", "static", "fade")
    acc = {"a5": {"sep": [], "amb": []}, "b": {"sep": [], "amb": []}}
    counts = {"sep": 0, "amb": 0}
    per_kind: dict = {}
    for i, row in enumerate(eval_rows[:args.n]):
        if (i // len(kinds)) % 2 == 0:                                     # the eval's sigma-0.6 half only
            continue
        gen, rng = torch.Generator().manual_seed(10_000 + i), random.Random(10_000 + i)
        kind, mode = kinds[i % len(kinds)], modes[(i // (2 * len(kinds))) % len(modes)]
        clip, alpha, kind, mode, (fg, bg) = sample_clip(env, row, gen, rng, scenes, train=False, kind=kind, mode=mode,
                                                        layers=True)
        rgb_h, temb, raw, g = two_pass(env, lora, clip, row["image"], EVAL_ALIGNED_SIGMA, 10_000 + i, grad=False)
        rgb = env["model"]["head"].decode(rgb_h, temb, g)
        x = rgb[0].transpose(0, 1)
        preds = {"a5": ea_score_to_alpha(head_a(path.chunks(rgb_h, temb, g, checkpoint=False), x)),   # LoRA off
                 "b": ea_score_to_alpha(head_b(path.chunks(raw, temb, g, checkpoint=False), x))}
        f = min(x.shape[0], alpha.shape[1])
        a = alpha[:, :f].transpose(0, 1)                                    # (F, 1, H, W)
        soft = (a > SOFT_LO) & (a < SOFT_HI)
        sep = ((fg[:, :f] - bg[:, :f]).norm(dim=0, keepdim=True).transpose(0, 1) > args.sep)
        for part, m in (("sep", soft & sep), ("amb", soft & ~sep)):
            n = int(m.sum())
            counts[part] += n
            if n:
                for name, p in preds.items():
                    err = (p[:f] - a).abs()[m]
                    acc[name][part].append((float(err.sum()), n))
                    per_kind.setdefault(f"{name}:{kind}:{part}", []).append((float(err.sum()), n))
    mean = lambda l: sum(s for s, _ in l) / max(1, sum(n for _, n in l))
    res = {name: {part: round(mean(v), 4) for part, v in d.items()} for name, d in acc.items()}
    res["ambiguous_share_of_soft_pixels"] = round(counts["amb"] / max(1, counts["sep"] + counts["amb"]), 3)
    res["by_kind"] = {k: round(mean(v), 4) for k, v in sorted(per_kind.items())}
    res["sep_threshold"] = args.sep
    print("SPLIT", json.dumps(res, indent=1), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "split.json").write_text(json.dumps(res, indent=1))


@torch.no_grad()
def deploy(args):
    from PIL import Image
    from .alpha_math import ea_score_to_alpha
    sys.path.insert(0, str(LOT))
    from h3_musubi import audio_sigma
    from pih3_h3 import AUDIO_SCALE, audio_noise, carried_audio_step, pih3_sigmas
    env = setup(2)
    lora, path, head_a, head_b = load_both(env, args.b)
    dit, fv = env["model"]["dit"], env["forward_video"]
    sigmas = pih3_sigmas(2, max_raw_t=1.0)
    subset = json.loads((TE / "subset.json").read_text())
    rows = {json.loads(l)["image"]: json.loads(l) for l in open(DATA / "t2rgba_train.jsonl")}
    want = {"effect": 3, "translucent": 3, "person": 2}
    picks = []
    for i in subset["eval"]:
        r = rows[i]
        if want.get(r["category"], 0) > 0 and te_path(i).is_file():
            want[r["category"]] -= 1
            picks.append(r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    W, H, F = 768, 512, 39
    lt, th, tw = env["token_shape"](W, H, F)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    checker = (((yy // 24 + xx // 24) % 2).float() * 0.35 + 0.45).cuda()[None]
    rows_img, times = [], {"rgb_only": [], "with_alpha": []}
    for k, row in enumerate(picks):
        text = env["load_text"](str(te_path(row["image"]))).cuda()
        z = torch.randn(1, 24, lt, H // 16, W // 16, device="cuda", generator=torch.Generator("cuda").manual_seed(k))
        za = audio_noise(lt, k + 1)
        for with_alpha in (False, True):
            torch.cuda.synchronize()
            t0 = time.time()
            x, y = sigmas[0] * z.float(), sigmas[0] * za.float()
            for i in range(len(sigmas) - 1):
                sigma, nxt = sigmas[i], sigmas[i + 1]
                sa = float(audio_sigma(sigma))
                store, last = {}, i == len(sigmas) - 2
                tap = {"first": lora.first, "lora": lambda: lora.on(False)} if (last and with_alpha) else None
                v = fv(dit, x, 1.0 - sigma, text, audio=y * (sa / sigma), t_audio=1.0 - sa, out=store, dwd=tap).float()
                y = carried_audio_step(y, store["audio"].float(), sigma, nxt)
                x = x + v * (sigma - nxt)
            g = (lt, th, tw)
            rgb = env["model"]["head"].decode(store["hidden"], store["temb_video"], g)
            if with_alpha:
                xr = rgb[0].transpose(0, 1)
                al_b = ea_score_to_alpha(head_b(path.chunks(tap["raw"], store["temb_video"], g, checkpoint=False), xr))
            torch.cuda.synchronize()
            times["with_alpha" if with_alpha else "rgb_only"].append(time.time() - t0)
        al_a = ea_score_to_alpha(head_a(path.chunks(store["hidden"], store["temb_video"], g, checkpoint=False), xr))
        r01 = ((rgb[0] + 1) / 2).clamp(0, 1)                                 # (3, F, H, W)
        cells = []
        for fr in (0, F // 2, F - 1):
            c = r01[:, fr]
            cells += [c, al_a[fr].expand(3, -1, -1), c * al_a[fr] + checker * (1 - al_a[fr]),
                      al_b[fr].expand(3, -1, -1), c * al_b[fr] + checker * (1 - al_b[fr])]
        rows_img.append(torch.cat(cells, 2))
        d = out / f"{k:02d}_{row['category']}"
        d.mkdir(exist_ok=True)
        rgba = torch.cat([r01, al_b.transpose(0, 1)], 0).permute(1, 2, 3, 0).cpu().numpy()
        for t in range(rgba.shape[0]):
            Image.fromarray((rgba[t] * 255 + 0.5).astype(np.uint8), "RGBA").save(d / f"{t:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(d / "%04d.png"), "-c:v",
                        "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "24", "-auto-alt-ref", "0",
                        str(out / f"{k:02d}_{row['category']}.webm")], check=True)
        print("DEPLOY", k, row["category"], row["caption"][:70], flush=True)
    sheet = torch.cat(rows_img, 1)
    sheet = torch.nn.functional.interpolate(sheet[None], scale_factor=0.25, mode="area")[0]
    Image.fromarray((sheet.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(out / "sheet.jpg", quality=88)
    t_rgb, t_al = np.median(times["rgb_only"][1:]), np.median(times["with_alpha"][1:])
    res = {"rgb_only_s": round(float(t_rgb), 3), "with_alpha_s": round(float(t_al), 3),
           "added_s": round(float(t_al - t_rgb), 3), "columns": "per frame: RGB | a5 alpha | a5 over checker | Stage B alpha | Stage B over checker",
           "rows": [f"{r['category']}: {r['caption'][:90]}" for r in picks]}
    (out / "deploy.json").write_text(json.dumps(res, indent=1))
    print("DEPLOY_DONE", json.dumps(res), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["split", "deploy"])
    p.add_argument("--b", required=True, help="a Stage B stage_b_last.pt")
    p.add_argument("--out", default="/media/2TB/alpha/checks_b1")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--sep", type=float, default=0.1)
    args = p.parse_args()
    if str(LOT) not in sys.path:
        sys.path.insert(0, str(LOT))
    split(args) if args.cmd == "split" else deploy(args)


if __name__ == "__main__":
    main()
