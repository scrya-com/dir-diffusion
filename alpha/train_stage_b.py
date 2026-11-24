# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""Stage B: a LoRA on pi-H3's last K blocks that runs only in a second pass, so pi-H3 itself carries alpha.

Two passes at pi-H3 2-step's last step:
* RGB: the ordinary forward, LoRA off. pi-H3's DWD decodes it exactly as today (RGB untouched by construction).
* alpha: block ``50 - K``'s input from that same forward runs the last K blocks again with the LoRA on
  (``h3_musubi.rerun_blocks``, checkpointed); its raw block output goes through ``dwd_norm_out`` and the frozen Sol
  refiner (``pih3_dwd``'s trainable forward), and the Stage A alpha head reads the refiner's tokens.

Only the LoRA and the alpha head train (EA loss + band loss, as Stage A). The LoRA's B starts at zero, so step 0
is Stage A exactly; ``--check`` asserts the alpha pass's block output equals the RGB pass's when B = 0.
Training is at the aligned sigma range (pi-H3's real last-step sigma redraws a different subject); the real
deployment is checked separately with real generations.

    python -m alpha.train_stage_b train --init /media/2TB/alpha/runs/a5/alpha_dwd_head.pt --out /media/2TB/alpha/runs/b1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from .train_pih3_alpha import (DATA, EVAL_ALIGNED_SIGMA, FRAMES, H, LOT, SCENES, TE, W, band_loss, grid, metrics,
                               sample_clip, setup, te_path)


class TailLoRA:
    """A rank-``rank`` LoRA on qkv/out/fc1/fc2 of the last ``k`` DiT blocks, off except inside ``on()``."""

    def __init__(self, dit, rank: int, k: int, device="cuda"):
        from h3_musubi import make_lora
        self.net = make_lora(dit, rank)
        self.net.requires_grad_(False)
        self.first = len(dit.blocks) - k
        self.mods = [m for m in self.net.unet_loras if int(re.search(r"blocks_(\d+)_", m.lora_name).group(1)) >= self.first]
        for m in self.net.unet_loras:
            m.enabled = False                       # the other blocks' modules stay off, and on the CPU
        for m in self.mods:
            m.to(device, torch.float32)
            m.requires_grad_(True)

    def params(self):
        return [p for m in self.mods for p in m.parameters() if p.requires_grad]

    @contextlib.contextmanager
    def on(self, grad: bool = True):
        """Re-entrant (checkpoint recomputes re-enter it inside the outer pass)."""
        self._depth = getattr(self, "_depth", 0) + 1
        if self._depth == 1:
            for m in self.mods:
                m.enabled = True
        try:
            with torch.enable_grad() if grad else torch.no_grad():
                yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                for m in self.mods:
                    m.enabled = False

    def state_dict(self):
        return {k: v for k, v in self.net.state_dict().items() if any(k.startswith(m.lora_name + ".") for m in self.mods)}

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd, strict=False)


class AlphaPath:
    """Differentiable: the alpha pass's raw block output -> ``dwd_norm_out`` -> frozen Sol refiner -> its final
    token states per 5-latent block (what ``pih3_alpha.capture_refiner_tokens`` catches on the fused path)."""

    def __init__(self, head_pih3):
        self.head, self.ref = head_pih3, head_pih3.refiner
        w = self.ref.feature_expand.weight
        w = w.dequantize() if hasattr(w, "dequantize") else w
        self.expand_w = w.to(torch.bfloat16).detach()
        self.expand_b = self.ref.feature_expand.bias.to(torch.bfloat16).detach()

    def chunks(self, h_raw, temb, grid_shape, checkpoint=True):
        from pih3_dwd import _layer, _layer_eager, build_plan, eager
        hd, ref = self.head, self.ref
        x = h_raw.float()
        scale, shift = F.linear(F.silu(temb.float()), hd.norm_w.float(), hd.norm_b.float()).chunk(2, dim=-1)
        feats = (F.layer_norm(x, x.shape[-1:], eps=hd.eps) * (1 + scale) + shift).to(torch.bfloat16)
        t, hh, ww = grid_shape
        e_h, e_w = ref.expansion
        D, S = ref.dim, ref.num_global
        geo = ref._build_geometry(t, hh * e_h, ww * e_w)
        tok = F.linear(feats, self.expand_w, self.expand_b).view(t, hh * ww, -1)
        glob = torch.cat([ref.register_tokens[0].to(tok), tok.new_zeros(1, D)], 0)
        parts = []
        for b in range(len(geo["rows"])):
            f = tok[b * ref.block:(b + 1) * ref.block]
            parts += [f.reshape(-1, hh, ww, e_h, e_w, D).permute(0, 1, 3, 2, 4, 5).reshape(-1, D), glob]
        xs = torch.cat(parts)
        plan = build_plan(ref, geo, xs.device)
        with eager():
            for li in range(len(ref.blocks)):
                if checkpoint and torch.is_grad_enabled():
                    xs = torch.utils.checkpoint.checkpoint(_layer_eager, ref, li, xs, plan, use_reentrant=False)
                else:
                    xs = _layer(ref, li, xs, plan)
        Hs, Ws = hh * e_h, ww * e_w
        return [xs[r0:r1 - S].view(-1, Hs, Ws, D) for r0, r1 in geo["rows"]]


def two_pass(env, lora: TailLoRA, clip, image, sigma, seed, grad=True):
    """Encode, noise, one pi-H3 forward (RGB pass) + the LoRA tail (alpha pass). Returns
    (rgb_hidden, temb, alpha_raw, token grid)."""
    z0 = env["encode"](clip)
    g = torch.Generator("cuda").manual_seed(seed)
    x = (1 - sigma) * z0 + sigma * torch.randn(z0.shape, device="cuda", generator=g)
    text = env["load_text"](str(te_path(image))).cuda()
    store: dict = {}
    tap = {"first": lora.first, "lora": lambda: lora.on(grad), "checkpoint": True}
    with torch.no_grad():
        env["forward_video"](env["model"]["dit"], x, 1.0 - sigma, text, out=store, dwd=tap,
                             audio_generator=torch.Generator("cuda").manual_seed(seed + 1))
    _b, _c, lt, lh, lw = z0.shape
    return store["hidden"], store["temb_video"], tap["raw"], (lt, lh // 2, lw // 2)


def decode_b(env, lora, path: AlphaPath, alpha_head, clip, image, sigma, seed, frames=None, grad=True):
    rgb_h, temb, raw, g = two_pass(env, lora, clip, image, sigma, seed, grad=grad)
    with torch.no_grad():
        rgb = env["model"]["head"].decode(rgb_h, temb, g)                   # pi-H3's own RGB, fused
    with torch.enable_grad() if grad else torch.no_grad():
        chunks = path.chunks(raw, temb, g, checkpoint=grad)
        score = alpha_head(chunks, rgb[0].transpose(0, 1), frames)        # (F or k, 1, H, W)
    return rgb, score.transpose(0, 1)[None], (rgb_h, raw)


@torch.no_grad()
def evaluate(env, lora, path, head, rows, out, step, scenes, n=60, log=print):
    from .alpha_math import ea_score_to_alpha
    head.eval()
    kinds = ("dark", "light", "gradient", "scene", "flat", "padded")
    modes = ("affine", "drift", "static", "fade")
    acc: dict = {}
    for i, row in enumerate(rows[:n]):
        gen, rng = torch.Generator().manual_seed(10_000 + i), random.Random(10_000 + i)
        kind, mode = kinds[i % len(kinds)], modes[(i // (2 * len(kinds))) % len(modes)]
        clip, alpha, kind, mode = sample_clip(env, row, gen, rng, scenes, train=False, kind=kind, mode=mode)
        sig = env["sigmas"][0] if (i // len(kinds)) % 2 == 0 else EVAL_ALIGNED_SIGMA
        rgb, score, _ = decode_b(env, lora, path, head, clip, row["image"], sig, 10_000 + i, grad=False)
        f = min(score.shape[2], alpha.shape[1])
        m = metrics(score[:, :, :f], alpha[None, :, :f])
        for key in ("all", "bg:" + kind, "mode:" + mode, row["category"], f"sigma:{sig:.3f}"):
            for k, v in m.items():
                if not math.isnan(v):
                    acc.setdefault(key, {}).setdefault(k, []).append(v)
        if i < 6:
            grid(clip[:, :f], alpha[:, :f], rgb[0, :, :f], ea_score_to_alpha(score[0, :, :f]),
                 Path(out) / f"grid_s{step:05d}_{i}.jpg")
    head.train()
    res = {k: {m: round(float(np.mean(v)), 4) for m, v in d.items()} for k, d in acc.items()}
    a = res.get(f"sigma:{EVAL_ALIGNED_SIGMA:.3f}", {})
    log(f"STAGEB eval step={step} sigma0.6 soft_mae={a.get('soft_mae')} edge_mae={a.get('edge_mae')} "
        f"all={res['all']}")
    (Path(out) / f"eval_{step:05d}.json").write_text(json.dumps(res, indent=1))
    return res


def train(args):
    from setproctitle import setproctitle
    from .alpha_math import ea_loss_per_sample
    from .pih3_alpha import AlphaDWDHead
    setproctitle(f"alpha:stageB {Path(args.out).name} k{args.k} r{args.rank}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    subset = json.loads((TE / "subset.json").read_text())
    rows = {json.loads(l)["image"]: json.loads(l) for l in open(DATA / "t2rgba_train.jsonl")}
    train_rows = [rows[i] for i in subset["train"] if te_path(i).is_file()]
    eval_rows = [rows[i] for i in subset["eval"] if te_path(i).is_file()]
    scenes = torch.load(SCENES) if SCENES.is_file() else None
    env = setup(args.pih3_steps)
    dit = env["model"]["dit"]
    lora = TailLoRA(dit, args.rank, args.k)
    path = AlphaPath(env["model"]["head"])
    head = AlphaDWDHead().cuda()
    if args.init:
        head.load_state_dict(torch.load(args.init, map_location="cuda")["head"])
    log(f"STAGEB k={args.k} rank={args.rank} lora_params={sum(p.numel() for p in lora.params()) / 1e6:.1f}M "
        f"init={args.init} train={len(train_rows)} eval={len(eval_rows)}")

    # check: with B = 0 the alpha pass's block output is the RGB pass's, so step 0 is Stage A exactly
    row = train_rows[0]
    clip, alpha, _k, _m = sample_clip(env, row, torch.Generator().manual_seed(1), random.Random(1), scenes)
    rgb_h, _t, raw, _g = two_pass(env, lora, clip, row["image"], 0.6, 7, grad=False)
    diff = float((rgb_h.float() - raw.float()).abs().max())
    log(f"STAGEB check B=0: max|alpha_pass - rgb_pass| = {diff:.3e}")
    if diff > 1e-2:
        raise SystemExit("the B = 0 alpha pass does not reproduce the RGB pass")
    # and its decoder tokens match the fused-path tokens Stage A's head was trained on
    from .pih3_alpha import capture_refiner_tokens
    with torch.no_grad():
        with capture_refiner_tokens(env["model"]["head"].refiner) as fused:
            env["model"]["head"].decode(rgb_h, _t, _g)
        ours = path.chunks(raw, _t, _g, checkpoint=False)
        fused = [c.view_as(o) for c, o in zip(fused, ours)]
        rel = max(float((o.float() - c.float()).norm() / c.float().norm()) for o, c in zip(ours, fused))
    log(f"STAGEB check tokens: eager alpha path vs fused decode, max rel = {rel:.3e} (int8 kernel floor ~1e-2)")
    if args.check:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        idx = torch.arange(0, 8, device="cuda")
        _rgb, score, _ = decode_b(env, lora, path, head, clip, row["image"], 0.6, 7, frames=idx)
        loss = ea_loss_per_sample(score, alpha[:, idx][None])[0].mean() + band_loss(score, alpha[:, idx][None])
        loss.backward()
        g_lora = sum(float(p.grad.norm()) for p in lora.params() if p.grad is not None)
        log(f"STAGEB check step: {time.time() - t0:.1f}s peak={torch.cuda.max_memory_allocated() / 2**30:.1f}G "
            f"lora_grad={g_lora:.3e}")
        # inference timing at 768x512x39: one pi-H3 forward + fused decode (today) vs + the alpha tail and path
        z0 = env["encode"](clip)
        x = 0.4 * z0 + 0.6 * torch.randn_like(z0)
        text = env["load_text"](str(te_path(row["image"]))).cuda()

        def rgb_only():
            store = {}
            env["forward_video"](dit, x, 0.4, text, out=store)
            _b, _c, lt, lh, lw = z0.shape
            return env["model"]["head"].decode(store["hidden"], store["temb_video"], (lt, lh // 2, lw // 2))

        def with_alpha():
            store = {}
            tap = {"first": lora.first, "lora": lambda: lora.on(False)}
            env["forward_video"](dit, x, 0.4, text, out=store, dwd=tap)
            _b, _c, lt, lh, lw = z0.shape
            g = (lt, lh // 2, lw // 2)
            rgb = env["model"]["head"].decode(store["hidden"], store["temb_video"], g)
            chunks = path.chunks(tap["raw"], store["temb_video"], g, checkpoint=False)
            return head(chunks, rgb[0].transpose(0, 1))

        with torch.no_grad():
            times = {}
            for name, fn in (("rgb_only", rgb_only), ("with_alpha", with_alpha)):
                fn(); torch.cuda.synchronize()
                t1 = time.time()
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                times[name] = (time.time() - t1) / 3
        log(f"STAGEB timing last step 768x512x39: rgb_only={times['rgb_only']:.3f}s "
            f"with_alpha={times['with_alpha']:.3f}s added={times['with_alpha'] - times['rgb_only']:.3f}s "
            f"(the alpha path here uses the eager refiner; a fused one is faster)")
        return

    opt = torch.optim.AdamW([{"params": lora.params(), "lr": args.lr_lora},
                             {"params": head.parameters(), "lr": args.lr_head}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))))
    evaluate(env, lora, path, head, eval_rows, out, 0, scenes, n=args.eval_n, log=log)
    torch.cuda.empty_cache()                                             # the eval leaves the cache fragmented
    rng = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _a in range(args.accum):
            row = rng.choice(train_rows)
            clip, alpha, kind, _mode = sample_clip(env, row, gen, rng, scenes)
            sigma = rng.uniform(*args.train_sigma)
            idx = torch.tensor(sorted(rng.sample(range(min(FRAMES, alpha.shape[1])), args.train_frames)), device="cuda")
            _rgb, score, _ = decode_b(env, lora, path, head, clip, row["image"], sigma, rng.randrange(1 << 30), frames=idx)
            target = alpha[:, idx][None]
            loss = ea_loss_per_sample(score, target)[0].mean() + args.band_weight * band_loss(score, target)
            (loss / args.accum).backward()
            del score
        torch.nn.utils.clip_grad_norm_(lora.params() + list(head.parameters()), 1.0)
        opt.step()
        sched.step()
        if step % 10 == 0:
            log(f"STAGEB step={step} loss={loss.item():.4f} bg={kind} s/step={(time.time() - t0) / 10:.1f} "
                f"mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G")
            t0 = time.time()
        if step % 125 == 0 or step == args.steps:
            torch.save({"lora": lora.state_dict(), "head": head.state_dict(), "step": step, "k": args.k,
                        "rank": args.rank}, out / "stage_b_last.pt")
        if step % args.eval_every == 0 or step == args.steps:
            torch.cuda.empty_cache()
            res = evaluate(env, lora, path, head, eval_rows, out, step, scenes, n=args.eval_n, log=log)
            torch.cuda.empty_cache()
            soft = res.get(f"sigma:{EVAL_ALIGNED_SIGMA:.3f}", {}).get("soft_mae", 1.0)
            if step == args.stop_at and soft > args.stop_above:
                log(f"STAGEB early stop: sigma0.6 soft_mae {soft} > {args.stop_above} at step {step}")
                break
    log("STAGEB_DONE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train"])
    p.add_argument("--out", required=True)
    p.add_argument("--init", default=None, help="a Stage A alpha_dwd_head.pt (or head_last.pt) to start the head from")
    p.add_argument("--k", type=int, default=8, help="LoRA on the last K of 50 blocks")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=5e-5)
    p.add_argument("--train-sigma", type=float, nargs=2, default=[0.45, 0.75])
    p.add_argument("--train-frames", type=int, default=8)
    p.add_argument("--band-weight", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-n", type=int, default=60)
    p.add_argument("--stop-at", type=int, default=250)
    p.add_argument("--stop-above", type=float, default=0.22, help="early stop if sigma-0.6 soft_mae is above this")
    p.add_argument("--pih3-steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check", action="store_true", help="B=0 parity, one training step's time/memory/grad, then exit")
    args = p.parse_args()
    if str(LOT) not in sys.path:
        sys.path.insert(0, str(LOT))
    train(args)


if __name__ == "__main__":
    main()
