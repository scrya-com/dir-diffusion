#!/usr/bin/env python3
"""Stage 1: pi-H3's DWD decoder on DiR. pi-H3 2-step's 8 Sol decoder blocks stay frozen; trained are its modulated
output norm (``dwd_norm_out``), its token expansion (``feature_expand``), and a new zero-initialised map from DiR's
final-iteration input state ``s`` into the expanded tokens (pi-H3 has none: its last blocks were trained with its
head, DiR's were not, and at sigma 0.667 a third of x0_hat is the input state).

Data: the DiR samples collected for our DWD (``collect_w1_{iso,atlas,kc}``: final-iteration inputs + the VAE decode
of x0_hat as mp4). Each step reruns DiR's final iteration (frozen) to get the block output, then decodes with
``pih3_dwd.refine_train``. Loss: L1 on every frame + LPIPS (vgg) on a few. Eval: LPIPS / PSNR against the VAE decode
on 12 held-out clips (4 per set), the zero-training check's yardstick (untrained: 0.219).

    python3 lot/train_pih3_dwd.py --out /media/2TB/lora-data/h3_lot/dwd/pih3_s1 --steps 2000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path("/media/2TB/lora-data/h3_lot/dwd")
SETS = ("iso", "atlas", "kc")
DIR_RUN = Path(__file__).resolve().parents[2] / "dir-diffusion/lot/runs/dir_turbo_clip"


class SolLoRA(nn.Module):
    """A LoRA beside one frozen (int8) Sol linear: ``y = linear(x) + B(A(x))``, B zero-initialised."""

    def __init__(self, linear, rank: int):
        super().__init__()
        w = linear.weight
        out_f, in_f = w.shape
        self.A = nn.Linear(in_f, rank, bias=False).cuda()
        self.B = nn.Linear(rank, out_f, bias=False).cuda()
        nn.init.normal_(self.A.weight, std=1.0 / rank)
        nn.init.zeros_(self.B.weight)
        orig = linear.forward

        def forward(x, *a, **k):
            return orig(x, *a, **k) + self.B(self.A(x.to(self.A.weight.dtype))).to(x.dtype)

        linear.forward = forward


def add_sol_lora(ref, rank: int) -> nn.ModuleList:
    mods = nn.ModuleList()
    for blk in ref.blocks:
        for lin in (blk.attn.to_qkv, blk.attn.to_out, blk.ff.w1, blk.ff.w2):
            mods.append(SolLoRA(lin, rank))
    return mods


class DiRDWD(nn.Module):
    """pi-H3's DWD head with a trainable norm and token expansion, plus an input-state branch."""

    def __init__(self, dwd_sd: dict, eps: float, token_dim: int = 96):
        super().__init__()
        from pih3_h3 import PiH3Head
        head = PiH3Head(dwd_sd, eps, device="cuda")
        self.ref = head.refiner                                   # frozen blocks, int8
        self.eps = eps
        self.norm = nn.Linear(head.norm_w.shape[1], head.norm_w.shape[0]).cuda()
        with torch.no_grad():
            self.norm.weight.copy_(head.norm_w.float())
            self.norm.bias.copy_(head.norm_b.float())
        fe = self.ref.feature_expand
        w, b = fe.weight, fe.bias
        w = w.dequantize() if hasattr(w, "dequantize") else w
        self.expand = nn.Linear(w.shape[1], w.shape[0]).cuda()
        with torch.no_grad():
            self.expand.weight.copy_(w.float())
            self.expand.bias.copy_(b.float())
        self.s_proj = nn.Linear(token_dim, w.shape[0]).cuda()
        nn.init.zeros_(self.s_proj.weight)
        nn.init.zeros_(self.s_proj.bias)

    def features(self, h, temb):
        scale, shift = self.norm(F.silu(temb.float())).chunk(2, dim=-1)
        return (F.layer_norm(h.float(), h.shape[-1:], eps=self.eps) * (1 + scale) + shift)

    def tokens(self, h, temb, s, grid):
        """Sol tokens in the decoder's block-major rows (out of place), with the geometry."""
        ref = self.ref
        t, hh, ww = grid
        e_h, e_w = ref.expansion
        D, S = ref.dim, ref.num_global
        geo = ref._build_geometry(t, hh * e_h, ww * e_w)
        x = (self.expand(self.features(h, temb)) + self.s_proj(s.float())).to(torch.bfloat16)
        frames = x.view(t, hh * ww, -1)
        glob = torch.cat([ref.register_tokens[0].to(x), x.new_zeros(1, D)], 0)
        parts = []
        for b in range(len(geo["rows"])):
            f = frames[b * ref.block:(b + 1) * ref.block]
            parts += [f.reshape(-1, hh, ww, e_h, e_w, D).permute(0, 1, 3, 2, 4, 5).reshape(-1, D), glob]
        return torch.cat(parts), geo, (t, hh * e_h, ww * e_w)

    def forward(self, h, temb, s, grid, checkpoint=True):
        from pih3_dwd import refine_train
        x, geo, shape = self.tokens(h, temb, s, grid)
        return refine_train(self.ref, x, geo, shape, checkpoint=checkpoint).transpose(1, 2)  # (1, 3, F, H, W)

    def trainable(self):
        st = {"norm": self.norm.state_dict(), "expand": self.expand.state_dict(), "s_proj": self.s_proj.state_dict()}
        if getattr(self, "lora", None) is not None:
            st["sol_lora"] = self.lora.state_dict()
        return st


def lpips_frames(lp, a, b, frames):
    return torch.cat([lp(a[0, :, f].transpose(0, 1)[i:i + 8], b[0, :, f].transpose(0, 1)[i:i + 8]).flatten()
                      for f in [frames] for i in range(0, len(frames), 8)]).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-s", type=float, default=2e-4)
    p.add_argument("--lpips-frames", type=int, default=4)
    p.add_argument("--lpips-crop", type=int, default=320, help="training LPIPS on random square crops (memory)")
    p.add_argument("--lpips-weight", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--adapter-steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", type=Path, default=None, help="a Stage 1 best.pt to start from")
    p.add_argument("--sol-lora", type=int, default=0, help="Stage 2: LoRA rank on every linear of the 8 Sol blocks")
    p.add_argument("--lr-lora", type=float, default=1e-4)
    args = p.parse_args()

    from setproctitle import setproctitle
    setproctitle(f"dwd:pih3-s1 {args.out.name} {args.steps}")
    import lpips
    from safetensors.torch import load_file
    from dir_h3 import estimate_extra, load_dir_run, use_networks
    from dwd_h3 import pack_rows
    from h3_musubi import forward_video, load_dit
    from pih3_h3 import adapter_path
    from train_dwd_h3 import load_text, read_mp4

    args.out.mkdir(parents=True, exist_ok=True)
    logf = open(args.out / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    train = [p for d in SETS for p in sorted((ROOT / f"collect_w1_{d}").glob("train_*.pt"))]
    held = [p for d in SETS for p in sorted((ROOT / f"collect_w1_{d}").glob("held_*.pt"))[:4]]
    log(f"DWDS1 train={len(train)} held={len(held)} adapter={args.adapter_steps}-step")
    sd = load_file(str(adapter_path(args.adapter_steps)))
    dwd_sd = {k: v for k, v in sd.items() if k.startswith("dwd_")}
    del sd
    dev = torch.device("cuda")
    dit = load_dit(device=dev, swap=0)
    dir_head, net, _config = load_dir_run(str(DIR_RUN), dit)
    model = DiRDWD(dwd_sd, dit.final_layer.norm.eps)
    model.lora = None
    if args.init:
        st = torch.load(args.init, map_location="cuda")
        model.norm.load_state_dict(st["norm"]); model.expand.load_state_dict(st["expand"]); model.s_proj.load_state_dict(st["s_proj"])
        log(f"DWDS1 init={args.init} (step {st.get('step')}, lpips {st.get('lpips')})")
    groups = [{"params": list(model.norm.parameters()) + list(model.expand.parameters()), "lr": args.lr},
              {"params": model.s_proj.parameters(), "lr": args.lr_s}]
    if args.sol_lora:
        model.lora = add_sol_lora(model.ref, args.sol_lora)
        groups.append({"params": model.lora.parameters(), "lr": args.lr_lora})
        log(f"DWDS1 sol_lora rank={args.sol_lora} params={sum(q.numel() for q in model.lora.parameters()) / 1e6:.1f}M")
    lp = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))))

    @torch.no_grad()
    def dir_features(path):
        s = torch.load(path, map_location="cpu", weights_only=False)
        state, prev = s["state"].to(dev).float(), s["prev"].to(dev).float()
        text = load_text(s["te"]).to(dev)
        r, sigma = int(s["r"]), float(s["sigma"])
        store = {}
        with use_networks(net, None, gen=True, fake=False):
            forward_video(dit, state, 1.0 - sigma, text, video_extra=estimate_extra(dir_head, prev, None),
                          video_temb_delta=dir_head.r_embed[r],
                          audio_generator=torch.Generator(dev).manual_seed(int(s["audio_seed"])), out=store)
        return store["hidden"], store["temb_video"], pack_rows(state), tuple(s["shape"])

    @torch.no_grad()
    def evaluate(step):
        rows = []
        for path in held:
            h, temb, srows, grid = dir_features(path)
            px = model(h, temb, srows, grid, checkpoint=False)
            tgt = read_mp4(path.with_suffix(".mp4")).to(dev).float()
            f = min(px.shape[2], tgt.shape[2])
            l = float(lpips_frames(lp, px[:, :, :f], tgt[:, :, :f], list(range(f))))
            mse = float(((px[:, :, :f] - tgt[:, :, :f]) / 2).pow(2).mean())
            rows.append((l, -10 * math.log10(max(mse, 1e-10))))
        lpm = sum(r[0] for r in rows) / len(rows)
        ps = sum(r[1] for r in rows) / len(rows)
        log(f"DWDS1 eval step={step} lpips={lpm:.4f} psnr={ps:.2f} s_proj_norm={float(model.s_proj.weight.norm()):.3e}")
        return lpm

    best = evaluate(0)
    rng = random.Random(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        path = rng.choice(train)
        h, temb, srows, grid = dir_features(path)
        tgt = read_mp4(path.with_suffix(".mp4")).to(dev).float()
        px = model(h, temb, srows, grid)
        f = min(px.shape[2], tgt.shape[2])
        px, tgt = px[:, :, :f], tgt[:, :, :f]
        frames = sorted(rng.sample(range(f), min(args.lpips_frames, f)))
        l1 = (px - tgt).abs().mean()
        H, W = px.shape[-2:]
        c = min(args.lpips_crop, H, W)
        y0, x0 = rng.randrange(H - c + 1), rng.randrange(W - c + 1)
        lpl = lpips_frames(lp, px[..., y0:y0 + c, x0:x0 + c], tgt[..., y0:y0 + c, x0:x0 + c], frames)
        loss = l1 + args.lpips_weight * lpl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for g in opt.param_groups for q in g["params"]], 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0:
            log(f"DWDS1 step={step} loss={loss.item():.4f} l1={l1.item():.4f} lpips={lpl.item():.4f} "
                f"s/step={(time.time() - t0) / 25:.2f} mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G")
            t0 = time.time()
        if step % args.eval_every == 0 or step == args.steps:
            lpm = evaluate(step)
            torch.save({"step": step, **model.trainable()}, args.out / "last.pt")
            if lpm < best:
                best = lpm
                torch.save({"step": step, "lpips": lpm, **model.trainable()}, args.out / "best.pt")
    log(f"DWDS1_DONE best_lpips={best:.4f}")


if __name__ == "__main__":
    main()
