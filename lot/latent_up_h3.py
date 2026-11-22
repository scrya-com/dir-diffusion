#!/usr/bin/env python3
"""A learned x2 upsampler of H3 video latents, for the LR -> HR handoff of the cascade and of two-track.

Interpolating the 16x VAE latent (trilinear, nearest, bicubic) gives latents the DiT reads as content: pi-H3 then
draws dots, ghost outlines and text over the clip (``pih3_cascade.py`` ``cas:K`` against ``cas:K:vae``). The VAE
round trip (decode, bicubic x2, encode) is clean but costs seconds. This net is fitted to that round trip:

    input   D(x): the latent of a cached clip, decoded, area x0.5, encoded
    target  U(D(x)): that LR latent decoded, bicubic x2, encoded

so it is a deterministic regression, and it runs in milliseconds. It is a residual on the trilinear upsample.

    python3 latent_up_h3.py --out OUT [--clips 160 --stills 150 --steps 6000]     # pairs, training, held-out error
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))


class Res(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.a = nn.Conv3d(width, width, 3, padding=1, padding_mode="replicate")
        self.b = nn.Conv3d(width, width, 3, padding=1, padding_mode="replicate")

    def forward(self, x):
        return x + self.b(F.silu(self.a(F.silu(x))))


class LatentUp(nn.Module):
    """``(B, 24, T, h, w)`` -> ``(B, 24, T, 2h, 2w)``."""

    def __init__(self, width: int = 128, blocks: int = 6, hr_width: int = 64, hr_blocks: int = 2):
        super().__init__()
        self.config = {"width": width, "blocks": blocks, "hr_width": hr_width, "hr_blocks": hr_blocks}
        self.inp = nn.Conv3d(24, width, 3, padding=1, padding_mode="replicate")
        self.lr = nn.Sequential(*[Res(width) for _ in range(blocks)])
        self.expand = nn.Conv3d(width, 4 * hr_width, 1)
        self.hr = nn.Sequential(*[Res(hr_width) for _ in range(hr_blocks)])
        self.out = nn.Conv3d(hr_width, 24, 3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        h = self.expand(self.lr(self.inp(x)))                                   # (B, 4c, T, h, w)
        b, c, t, hh, ww = h.shape
        h = F.pixel_shuffle(h.transpose(1, 2).reshape(b * t, c, hh, ww), 2)     # (B T, c, 2h, 2w)
        h = h.reshape(b, t, c // 4, 2 * hh, 2 * ww).transpose(1, 2)
        return base + self.out(F.silu(self.hr(h)))


def load_up(path: Path, device="cuda"):
    """The upsampler as a function of a ``(1, 24, T, h, w)`` latent (bf16 autocast)."""
    saved = torch.load(path, map_location="cpu")
    net = LatentUp(**saved["config"]).to(device).eval().requires_grad_(False)
    net.load_state_dict(saved["state"])

    @torch.no_grad()
    def up(latent: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", torch.bfloat16):
            return net(latent.float()).float()

    return up


def rel(a, b) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--clips", type=int, default=160, help="cached clips (the first --held are held out)")
    p.add_argument("--clip-frames", type=int, default=12, help="latent frames used of each clip (2, 7, 12, 17, 22)")
    p.add_argument("--stills", type=int, default=150)
    p.add_argument("--held", type=int, default=12)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--pairs", type=Path, default=None, help="pairs.pt of an earlier run (default: OUT/pairs.pt)")
    args = p.parse_args()
    from lot_paths import LOT_H3_CLIPS_CACHE
    from safetensors.torch import load_file
    from setproctitle import setproctitle
    from two_track_h3 import load_upscaler

    setproctitle(f"pih3:latent_up clips={args.clips} steps={args.steps}")
    args.out.mkdir(parents=True, exist_ok=True)
    pairs_path = args.pairs or args.out / "pairs.pt"
    if pairs_path.is_file():
        pairs = torch.load(pairs_path)
    else:
        up = load_upscaler()
        clip_files = [f for f in sorted(glob.glob(str(LOT_H3_CLIPS_CACHE / "*_minimaxh3.safetensors")))
                      if not f.endswith("_te.safetensors")][:args.clips]
        still_files = [f for f in sorted(glob.glob(str(LOT_H3_CLIPS_CACHE.parent / "cache_iso3d_1344x768" /
                                                       "*_minimaxh3.safetensors")))
                       if not f.endswith("_te.safetensors")][:args.stills]
        pairs = []
        start = time.time()
        with torch.no_grad():
            for n, path in enumerate(clip_files + still_files):
                latent = next(iter(load_file(path).values())).float()
                latent = latent[:, :args.clip_frames] if latent.ndim == 4 else latent[:, None]
                latent = latent[None, ..., :latent.shape[-2] // 2 * 2, :latent.shape[-1] // 2 * 2].cuda()
                lr = up.down(latent)
                pairs.append((lr.half().cpu(), up(lr).half().cpu()))
                if n % 20 == 0:
                    print(f"LOT_H3 kind=latent_up_pairs n={n}/{len(clip_files) + len(still_files)} "
                          f"shape={tuple(lr.shape)} seconds={time.time() - start:.0f}", flush=True)
        torch.save(pairs, pairs_path)
        del up
        torch.cuda.empty_cache()
    held, train = pairs[:args.held], pairs[args.held:]
    net = LatentUp(args.width, args.blocks).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    gen = torch.Generator().manual_seed(0)

    def crop(pair, size: int = 16, frames: int = 6):
        lr, hr = pair
        t, h, w = lr.shape[2:]
        f = min(frames, t)
        t0 = int(torch.randint(0, t - f + 1, (1,), generator=gen)) if t > f else 0
        if t > f and torch.rand(1, generator=gen) < 0.5:
            t0 = 0                                           # the clip's first latent frame is special (causal VAE)
        y = int(torch.randint(0, h - size + 1, (1,), generator=gen))
        x = int(torch.randint(0, w - size + 1, (1,), generator=gen))
        return (lr[0, :, t0:t0 + f, y:y + size, x:x + size],
                hr[0, :, t0:t0 + f, 2 * y:2 * (y + size), 2 * x:2 * (x + size)])

    clips = [q for q in train if q[0].shape[2] > 1]
    stills = [q for q in train if q[0].shape[2] == 1]

    @torch.no_grad()
    def evaluate() -> dict:
        net.eval()
        a = [rel(net(lr.cuda().float()), hr.cuda()) for lr, hr in held]
        b = [rel(F.interpolate(lr.cuda().float(), scale_factor=(1, 2, 2), mode="trilinear"), hr.cuda()) for lr, hr in held]
        net.train()
        return {"net_rel": round(sum(a) / len(a), 4), "trilinear_rel": round(sum(b) / len(b), 4)}

    start = time.time()
    for step in range(args.steps):
        pool = stills if (stills and step % 4 == 3) else clips
        picks = [crop(pool[int(torch.randint(0, len(pool), (1,), generator=gen))]) for _ in range(args.batch)]
        lr = torch.stack([q[0] for q in picks]).cuda().float()
        hr = torch.stack([q[1] for q in picks]).cuda().float()
        loss = F.l1_loss(net(lr), hr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"LOT_H3 kind=latent_up_train step={step} loss={loss.item():.4f} seconds={time.time() - start:.0f} "
                  + json.dumps(evaluate()), flush=True)
    torch.save({"config": net.config, "state": net.state_dict(), "eval": evaluate()}, args.out / "latent_up.pt")
    print(f"LOT_H3 kind=latent_up_done out={args.out / 'latent_up.pt'} ok=1", flush=True)


if __name__ == "__main__":
    main()
