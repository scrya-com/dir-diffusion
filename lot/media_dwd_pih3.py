#!/usr/bin/env python3
"""Comparison images for docs/DWD_PIH3.md: per held-out clip, columns VAE decode (the target) | pi-H3 2-step's decoder
on DiR untrained | Stage 1, rows at three frames; plus one side-by-side mp4. Runs with block swap, so it fits a
24 GB card (no v0.2 here: its two-pass tail needs every block resident).

    python3 lot/media_dwd_pih3.py --stage1 RUN/best.pt --out docs/dwd_pih3_media
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path("/media/2TB/lora-data/h3_lot/dwd")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--swap", type=int, default=20)
    p.add_argument("--data", type=Path, nargs="+", default=[ROOT / f"collect_w1_{d}" for d in ("iso", "atlas", "kc")])
    args = p.parse_args()
    from setproctitle import setproctitle
    setproctitle("dwd:media stage1")
    from PIL import Image, ImageDraw, ImageFont
    from safetensors.torch import load_file
    from dir_h3 import estimate_extra, load_dir_run, use_networks
    from dir_two_track_h3 import resolve_run
    from dwd_h3 import pack_rows
    from h3_musubi import forward_video, load_dit
    from pih3_h3 import adapter_path
    from train_dwd_h3 import load_text, read_mp4
    from train_pih3_dwd import DIR_RUN, DiRDWD

    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    dit = load_dit(device=dev, swap=args.swap)
    gen_head, net, _ = load_dir_run(str(resolve_run(DIR_RUN)), dit)
    sd = load_file(str(adapter_path(2)))
    dwd = {k: v for k, v in sd.items() if k.startswith("dwd_")}
    del sd
    raw = DiRDWD(dwd, dit.final_layer.norm.eps).eval()
    s1 = DiRDWD(dwd, dit.final_layer.norm.eps).eval()
    st = torch.load(args.stage1, map_location="cuda")
    s1.norm.load_state_dict(st["norm"]); s1.expand.load_state_dict(st["expand"]); s1.s_proj.load_state_dict(st["s_proj"])
    font = ImageFont.truetype("/home/johndpope/miniconda3/fonts/DejaVuSans.ttf", 18)
    to8 = lambda t: ((t.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()

    with torch.no_grad():
        for data in args.data:
            path = sorted(data.glob("held_*.pt"))[1]
            s = torch.load(path, map_location="cpu", weights_only=False)
            state, prev = s["state"].to(dev).float(), s["prev"].to(dev).float()
            text = load_text(s["te"]).to(dev)
            r, sigma, seed, grid = int(s["r"]), float(s["sigma"]), int(s["audio_seed"]), tuple(s["shape"])
            store = {}
            with use_networks(net, None, gen=True, fake=False):
                forward_video(dit, state, 1.0 - sigma, text, video_extra=estimate_extra(gen_head, prev, None),
                              video_temb_delta=gen_head.r_embed[r],
                              audio_generator=torch.Generator(dev).manual_seed(seed), out=store)
            srows = pack_rows(state)
            outs = []
            for m in (raw, s1):
                x, geo, shape = m.tokens(store["hidden"], store["temb_video"], srows if m is s1 else srows * 0, grid)
                outs.append(m.ref.refine(x, geo, shape).transpose(1, 2)[0].float())   # (3, F, H, W)
            tgt = read_mp4(path.with_suffix(".mp4"))[0].to(dev).float()
            f = min(tgt.shape[1], *(o.shape[1] for o in outs))
            cols = [tgt[:, :f], outs[0][:, :f], outs[1][:, :f]]
            names = ["VAE decode (target)", "pi-H3 2-step decoder, untrained", "Stage 1 (ours)"]
            H, W = cols[0].shape[-2:]
            rows = []
            for fr in (0, f // 2, f - 1):
                rows.append(np.concatenate([to8(c[:, fr]) for c in cols], 1))
            img = Image.fromarray(np.concatenate(rows, 0))
            bar = Image.new("RGB", (img.width, 34), (18, 18, 22))
            d = ImageDraw.Draw(bar)
            for i, n in enumerate(names):
                d.text((i * W + 10, 7), n, fill=(235, 235, 240), font=font)
            full = Image.new("RGB", (img.width, img.height + 34))
            full.paste(bar, (0, 0)); full.paste(img, (0, 34))
            tag = data.name.replace("collect_w1_", "")
            full.resize((full.width // 2, full.height // 2), Image.LANCZOS).save(args.out / f"{tag}_{path.stem}.jpg", quality=88)
            # side-by-side mp4 of the whole clip
            frames = np.stack([np.concatenate([to8(c[:, k]) for c in cols], 1) for k in range(f)])
            fh, fw = frames.shape[1:3]
            ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fw}x{fh}",
                                   "-r", "24", "-i", "-", "-vf", "scale=iw/2:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                   "-crf", "22", "-movflags", "+faststart", str(args.out / f"{tag}_{path.stem}.mp4")],
                                  stdin=subprocess.PIPE)
            ff.stdin.write(frames.tobytes()); ff.stdin.close(); ff.wait()
            print("MEDIA", tag, path.name, flush=True)
    print("MEDIA_DONE", flush=True)


if __name__ == "__main__":
    main()
