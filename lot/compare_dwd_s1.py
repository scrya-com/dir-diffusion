#!/usr/bin/env python3
"""Head-to-head on the same 12 held-out DiR clips (collect_w1_{iso,atlas,kc}, 4 each): our DWD v0.2 (last-8-blocks
LoRA, linear head, v0.2 refiner) against Stage 1 (pi-H3 2-step's decoder with a trained norm, expansion and
input-state branch, no LoRA), both scored against the VAE decode of the same latent. Stage 1 runs pi-H3's own
fused ``refine`` here (the inference path). Decode times are warm, synchronised, decoder only.

    python3 lot/compare_dwd_s1.py --stage1 /media/2TB/lora-data/h3_lot/dwd/pih3_s1/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path("/media/2TB/lora-data/h3_lot/dwd")
V02 = dict(lora=ROOT / "hf_stage/dwd_lora.pt", head=ROOT / "hf_stage/dwd_head.pt",
           refiner=ROOT / "accel/mr/ref/refiner_best.pt")                     # v0.2 refiner, step 121279


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--swap", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("/media/2TB/alpha/compare_dwd_s1.json"))
    args = p.parse_args()
    from setproctitle import setproctitle
    setproctitle("dwd:compare v0.2 vs stage1")
    import lpips
    from safetensors.torch import load_file
    from accel_h3 import DwdStack, exit_step
    from dir_h3 import estimate_extra, load_dir_run, use_networks
    from dwd_h3 import pack_rows
    from h3_musubi import forward_video, load_dit
    from pih3_h3 import adapter_path
    from train_dwd_h3 import load_text, read_mp4
    from train_pih3_dwd import DIR_RUN, DiRDWD, lpips_frames

    dev = torch.device("cuda")
    held = [q for d in ("iso", "atlas", "kc") for q in sorted((ROOT / f"collect_w1_{d}").glob("held_*.pt"))[:4]]
    dit = load_dit(device=dev, swap=args.swap)
    from dir_two_track_h3 import resolve_run
    gen_head, net, _ = load_dir_run(str(resolve_run(DIR_RUN)), dit)
    stack = DwdStack(dit, V02["lora"], V02["head"], V02["refiner"], mode="two_pass")
    sd = load_file(str(adapter_path(2)))
    s1 = DiRDWD({k: v for k, v in sd.items() if k.startswith("dwd_")}, dit.final_layer.norm.eps)
    del sd
    st = torch.load(args.stage1, map_location="cuda")
    s1.norm.load_state_dict(st["norm"]); s1.expand.load_state_dict(st["expand"]); s1.s_proj.load_state_dict(st["s_proj"])
    s1.eval()
    lp = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)

    def timed(fn, reps=3):
        fn(); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            out = fn()
        torch.cuda.synchronize()
        return out, (time.time() - t0) / reps

    rows = []
    with torch.no_grad():
        for path in held:
            s = torch.load(path, map_location="cpu", weights_only=False)
            state, prev = s["state"].to(dev).float(), s["prev"].to(dev).float()
            text = load_text(s["te"]).to(dev)
            r, sigma, seed, grid = int(s["r"]), float(s["sigma"]), int(s["audio_seed"]), tuple(s["shape"])
            tgt = read_mp4(path.with_suffix(".mp4")).to(dev).float()
            res = {"sample": f"{path.parent.name[11:]}/{path.name}"}
            with use_networks(net, None, gen=True, fake=False):
                h2, est = exit_step(dit, gen_head, stack, state, prev, r, text, sigma, seed)
                store = {}
                forward_video(dit, state, 1.0 - sigma, text, video_extra=estimate_extra(gen_head, prev, None),
                              video_temb_delta=gen_head.r_embed[r],
                              audio_generator=torch.Generator(dev).manual_seed(seed), out=store)
            px2, t2 = timed(lambda: stack.decode(h2, state, est, grid)[0])
            srows = pack_rows(state)

            def s1_decode():
                x, geo, shape = s1.tokens(store["hidden"], store["temb_video"], srows, grid)
                return s1.ref.refine(x, geo, shape).transpose(1, 2).float()      # fused int8: the inference path
            px1, t1 = timed(s1_decode)
            for name, px, t in (("v0.2", px2, t2), ("stage1", px1, t1)):
                f = min(px.shape[2], tgt.shape[2])
                l = float(lpips_frames(lp, px[:, :, :f].float(), tgt[:, :, :f], list(range(f))))
                mse = float(((px[:, :, :f].float() - tgt[:, :, :f]) / 2).pow(2).mean())
                res[name] = {"lpips": round(l, 4), "psnr": round(-10 * math.log10(max(mse, 1e-10)), 2), "decode_s": round(t, 4)}
            rows.append(res)
            print("CMP", json.dumps(res), flush=True)
    summ = {n: {k: round(sum(r[n][k] for r in rows) / len(rows), 4) for k in ("lpips", "psnr", "decode_s")} for n in ("v0.2", "stage1")}
    print("CMP_SUMMARY", json.dumps(summ), flush=True)
    args.out.write_text(json.dumps({"rows": rows, "summary": summ}, indent=1))


if __name__ == "__main__":
    main()
