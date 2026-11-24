# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""Stage B deployment check: real pi-H3 2-step generations from noise (``max_raw_t`` 1.0, the camera-lock
schedule; the cam_static dial is a ComfyUI-side LoRA and is not applied here), with the alpha pass on the last
step. Writes, per caption: an RGBA WebM (VP9 with alpha), a ProRes 4444 .mov, an mp4 of the RGBA over a moving
checker, and a contact sheet (RGB | alpha | composite at three frames).

    python -m alpha.deploy_check_b --ckpt /media/2TB/alpha/runs/b1/stage_b_last.pt --out /media/2TB/alpha/runs/b1/deploy
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .train_pih3_alpha import LOT, setup

PROBE = Path("/media/2TB/alpha/probe")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--frames", type=int, default=39)
    p.add_argument("--max-raw-t", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    sys.path.insert(0, str(LOT))
    from setproctitle import setproctitle
    setproctitle("alpha:stageB deploy check")
    from PIL import Image
    import h3_musubi
    from accel_h3 import token_shape
    from pih3_h3 import audio_noise, pih3_sample, pih3_sigmas
    from .alpha_math import ea_score_to_alpha
    from .pih3_alpha import AlphaDWDHead
    from .train_stage_b import AlphaPath, TailLoRA

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = setup(2)
    st = torch.load(args.ckpt, map_location="cuda")
    lora = TailLoRA(env["model"]["dit"], st["rank"], st["k"])
    lora.load_state_dict(st["lora"])
    head = AlphaDWDHead().cuda().eval()
    head.load_state_dict(st["head"])
    path = AlphaPath(env["model"]["head"])
    sigmas = pih3_sigmas(2, max_raw_t=args.max_raw_t)
    W, H, F = args.width, args.height, args.frames
    latent_t, _th, _tw = token_shape(W, H, F)
    orig = h3_musubi.forward_video
    tes = sorted(PROBE.glob("*_minimaxh3_te.safetensors"))
    print(f"DEPLOY ckpt step {st['step']} sigmas {[round(s, 4) for s in sigmas]} prompts {len(tes)}", flush=True)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    for i, te in enumerate(tes):
        stem = te.name.replace("_minimaxh3_te.safetensors", "")
        text = env["load_text"](str(te)).cuda()
        calls = {"n": 0}
        cap: dict = {}

        def fv(*a, **k):                                   # the last pi-H3 call also runs the LoRA tail
            calls["n"] += 1
            if calls["n"] == len(sigmas) - 1:
                tap = {"first": lora.first, "lora": lambda: lora.on(False)}
                k["dwd"] = tap
                r = orig(*a, **k)
                cap["raw"] = tap["raw"]
                return r
            return orig(*a, **k)

        h3_musubi.forward_video = fv
        try:
            with torch.no_grad():
                z = torch.randn(1, 24, latent_t, H // 16, W // 16, device="cuda",
                                generator=torch.Generator("cuda").manual_seed(args.seed + i))
                o = pih3_sample(env["model"], text, z, audio_noise(latent_t, args.seed + i + 1), sigmas=sigmas)
                g = (latent_t, H // 32, W // 32)
                rgb = env["model"]["head"].decode(o["h"], o["temb"], g)                     # (1, 3, F, H, W)
                chunks = path.chunks(cap["raw"], o["temb"], g, checkpoint=False)
                a = ea_score_to_alpha(head(chunks, rgb[0].transpose(0, 1))).transpose(0, 1)  # (1, F, H, W)
        finally:
            h3_musubi.forward_video = orig
        f = min(rgb.shape[2], a.shape[1])
        rgb01 = ((rgb[0, :, :f].float() + 1) / 2).clamp(0, 1)
        a = a[:, :f].float().clamp(0, 1)
        d = out / stem
        d.mkdir(exist_ok=True)
        comps = []
        for t in range(f):
            rgba = torch.cat([rgb01[:, t], a[:, t]], 0).permute(1, 2, 0).cpu().numpy()
            Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), "RGBA").save(d / f"{t:04d}.png")
            checker = (((yy // 32 + xx // 32 + t // 3) % 2).float() * 0.4 + 0.3).cuda()[None]
            comps.append(((rgb01[:, t] * a[:, t] + checker * (1 - a[:, t])).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        for codec, ext, pix in (("libvpx-vp9", "webm", "yuva420p"), ("prores_ks", "mov", "yuva444p10le")):
            extra = ["-b:v", "0", "-crf", "26", "-auto-alt-ref", "0"] if codec == "libvpx-vp9" else ["-profile:v", "4444"]
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(d / "%04d.png"),
                            "-c:v", codec, "-pix_fmt", pix, *extra, str(out / f"{stem}.{ext}")], check=True)
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", "24", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                               str(out / f"{stem}_over_checker.mp4")], stdin=subprocess.PIPE)
        ff.stdin.write(np.stack(comps).tobytes()); ff.stdin.close(); ff.wait()
        rows = []
        for t in (0, f // 2, f - 1):
            rows.append(np.concatenate([(rgb01[:, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
                                        (a[0, t, :, :, None].expand(-1, -1, 3).cpu().numpy() * 255).astype(np.uint8),
                                        comps[t]], 1))
        Image.fromarray(np.concatenate(rows, 0)).resize((W * 3 // 2, H * 3 // 2)).save(out / f"{stem}_sheet.jpg", quality=88)
        print(f"DEPLOY {stem} frames={f} alpha_mean={float(a.mean()):.3f} soft_frac={float(((a > .02) & (a < .98)).float().mean()):.3f}", flush=True)
    print("DEPLOY_DONE", flush=True)


if __name__ == "__main__":
    main()
