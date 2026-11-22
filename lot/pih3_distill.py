#!/usr/bin/env python3
"""Distil pi-H3 4-step at full resolution into the 2-step fast path: 1 step at half resolution, the learned latent x2
(``latent_up_h3``), 1 step at full resolution. A bypass LoRA (unmerged, on top of the merged int8 pi-H3 adapter).

pi-H3 is by Hansheng Chen, Shengqu Cai, Jiaming Song, Gordon Wetzstein, Xinchen Yan and Leonidas Guibas. The cascade
is not theirs and pi-H3 was never trained for it; MPEG-Diffusion (Dennis Menn, Tianpei Gu, Guande He, Aryan V S,
Yanke Song, Nan Liu, Desai Xie, Xun Huang, Chenfeng Xu, Diana Marculescu) fine-tunes its model for mixed resolution,
which is what this does for the cascade, DiR style: the student is trained to land on the teacher's own video.

Paired noise: the teacher draws full-resolution noise ``z`` (seed); the student's half-resolution noise is
``2 * avgpool2x2(z)`` (unit variance) and its full-resolution re-noising uses ``z`` itself, so the target is a
deterministic function of what the student sees.

    python3 pih3_distill.py teacher --out D --train 500        # pi-H3 4-step clips (latent, DWD frames)
    python3 pih3_distill.py train --data D --out D/run_a [--drop h3tiny] [--check]
    python3 pih3_distill.py eval --data D --run D/run_a --tag 1000 [--drop h3tiny]   # held-out gate table + sheets

Loss: MSE of the final latent against the teacher's, plus LPIPS of the DWD pixels of one native block (5 latent
frames, 17 pixel frames) against the teacher's DWD pixels of the same block. The head is differentiable here
(``TrainHead``: dequantized refiner weights, an out-of-place copy of ComfyUI-piFlow's refine loop).
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from h3_musubi import H3_TINY_DROP, audio_sigma  # noqa: E402
from pih3_h3 import (adapter_path, apply_adapter, audio_noise, carried_audio_step, PiH3Head,  # noqa: E402
                     pih3_sigmas)

BLOCK = 5                                   # latent frames per native Sol block (17 pixel frames for the first)
UP_DEFAULT = "/media/2TB/lora-data/h3_lot/pih3/cascade/latent_up_v2/latent_up.pt"


# ------------------------------------------------------------------------------------------- differentiable head

def refine_block(refiner, x, geo, latent_shape):
    """ComfyUI-piFlow ``SolH3DWDHead.refine`` for a clip of ONE native block, out of place (autograd-safe).
    Same operations in the same order; returns ``(1, F, 3, 16H, 16W)`` in [-1, 1]."""
    from comfy.ldm.modules.attention import optimized_attention

    T, H, W = latent_shape
    heads, dh, D, S = refiner.heads, refiner.dim_head, refiner.dim, refiner.num_global
    rows = geo["rows"]
    if len(rows) != 1:
        raise ValueError("refine_block takes one native block (<= 5 latent frames)")
    dev = x.device
    r0, r1 = rows[0]
    gslot = geo["gslot"].to(dev)
    shifts = [(sh["ids"].to(dev), [[(qi.to(dev), ki.to(dev)) for qi, ki in wins] for wins in sh["windows_at"]])
              for sh in geo["shifts"]]
    import comfy.ops

    def layer(x, li):
        blk = refiner.blocks[li]
        attn, (ids, windows_at) = blk.attn, shifts[li % 2]
        qkv = comfy.ops.linear_input_act(attn.to_qkv, x, "rms_norm", act_weight=blk.norm1)
        qq, kk, vv = qkv.view(1, r1 - r0, heads, 3 * dh).chunk(3, -1)
        qq, kk = attn.rms_rope(qq, kk, refiner.pos_embed(ids[None, r0:r1].to(x.dtype)))
        q, k, v = qq[0], kk[0], vv[0]
        out = x.new_zeros(r1 - r0, D)
        for qi, ki in windows_at[0]:
            out = out.index_put((qi,), attn.attention(*(t.transpose(1, 2) for t in (q[qi], k[ki], v[ki]))))
        gs = gslot[0]
        out = out.index_put((gs,), optimized_attention(*(t[None].transpose(1, 2) for t in (q[gs], k, v)),
                                                       heads, skip_reshape=True)[0])
        return blk.post_attention(out, x)

    for li in range(len(refiner.blocks)):                 # activation checkpointing per layer when training
        if torch.is_grad_enabled() and x.requires_grad:
            x = torch.utils.checkpoint.checkpoint(layer, x, li, use_reentrant=False)
        else:
            x = layer(x, li)
    from pih3_h3 import comfy_lib

    sol = comfy_lib()["sol"]
    n = (r1 - r0 - S) // (H * W)
    y = refiner.norm_out(x[:r1 - S]).view(n, H, W, D)
    first_t = sol.PATCH_T - sol.FRAMES_TO_TRIM
    P = sol.PATCH
    parts = [refiner.first_frame_proj_out(y[:1]).view(H, W, sol.OUT_CH, first_t, P, P).permute(
        3, 2, 0, 4, 1, 5).reshape(first_t, sol.OUT_CH, H * P, W * P)]
    if n > 1:
        parts.append(refiner.proj_out(y[1:]).view(n - 1, H, W, sol.OUT_CH, sol.PATCH_T, P, P).permute(
            0, 4, 3, 1, 5, 2, 6).reshape((n - 1) * sol.PATCH_T, sol.OUT_CH, H * P, W * P))
    pixels = torch.cat(parts).float()
    mean = torch.tensor(sol.COLOR_MEAN, device=dev).view(-1, 1, 1)
    std = torch.tensor(sol.COLOR_STD, device=dev).view(-1, 1, 1)
    return (pixels * std + mean).clamp(-1, 1)[None]


class TrainHead:
    """pi-H3's DWD head with gradients to its input tokens (weights frozen): the int8 refiner weights dequantized to
    bf16, ComfyUI's fused int8 paths off (``in_training``), comfy-kitchen's eager rope (the fused one has no
    autograd), and ``refine_block`` for the loop. Within 51.5 dB of the int8 head on random tokens."""

    def __init__(self, dwd_sd: dict, eps: float, device="cuda"):
        head = PiH3Head(dwd_sd, eps, device=device)
        import comfy.model_management as mm
        import comfy.quant_ops
        import comfy_kitchen.backends.eager.rope as eager_rope
        from comfy.quant_ops import QuantizedTensor

        for module in head.refiner.modules():
            weight = getattr(module, "weight", None)
            if isinstance(weight, QuantizedTensor):
                module.weight = torch.nn.Parameter(weight.dequantize().to(torch.bfloat16), requires_grad=False)
                if hasattr(module, "layout_type"):
                    module.layout_type = None
        mm.in_training = True
        comfy.quant_ops.ck.rms_rope_split_half = eager_rope.rms_rope_split_half
        self.head = head

    def features(self, h, t_emb):
        x = h.to(torch.bfloat16)
        mod = F.linear(F.silu(t_emb.to(torch.bfloat16)), self.head.norm_w.to(torch.bfloat16),
                       self.head.norm_b.to(torch.bfloat16))
        scale, shift = mod.chunk(2, dim=-1)
        return F.layer_norm(x, x.shape[-1:], eps=self.head.eps) * (1 + scale) + shift

    def decode_block(self, h, t_emb, grid, block: int):
        """Pixels ``(1, 3, F, H, W)`` of native block ``block`` (latent frames ``5 b .. 5 b + 4``) decoded alone."""
        t, gh, gw = grid
        a, b = block * BLOCK * gh * gw, min(t, (block + 1) * BLOCK) * gh * gw
        sub_t = min(t, (block + 1) * BLOCK) - block * BLOCK
        feats = self.features(h[a:b], t_emb)[None]
        refiner = self.head.refiner
        x, geo, shape = refiner.expand(feats, (sub_t, gh, gw))
        return refine_block(refiner, x, geo, shape).transpose(1, 2)


# ------------------------------------------------------------------------------------------- the fast path

def paired_noise(latent_t: int, height: int, width: int, seed: int):
    """``(z, z_lr)``: the teacher's full-resolution noise and the student's half-resolution noise from it."""
    z = torch.randn(1, 24, latent_t, height // 16, width // 16, device="cuda",
                    generator=torch.Generator("cuda").manual_seed(seed))
    return z, 2.0 * F.avg_pool3d(z, (1, 2, 2))


def fast_path(dit, up, text, z, z_lr, za, sigmas, *, grad: bool = False) -> dict:
    """1 step at half resolution, ``up`` (the learned latent x2), re-noised with ``z`` to ``sigmas[1]``, 1 step at
    full resolution. Returns ``latent`` (the final clean estimate), ``h`` and ``temb`` (the DWD head's input)."""
    from h3_musubi import forward_video

    s0, s1 = sigmas[0], sigmas[1]
    with torch.set_grad_enabled(grad):
        x, y = s0 * z_lr, s0 * za
        sa = float(audio_sigma(s0))
        store: dict = {}
        v = forward_video(dit, x, 1.0 - s0, text, audio=y * (sa / s0), t_audio=1.0 - sa, out=store).float()
        est = x + s0 * v
        y = carried_audio_step(y, store["audio"].float().detach(), s0, s1)
        with torch.autocast("cuda", torch.bfloat16):
            est_hr = up(est).float()
        x = s1 * z + (1.0 - s1) * est_hr
        sa = float(audio_sigma(s1))
        store = {}
        v = forward_video(dit, x, 1.0 - s1, text, audio=y * (sa / s1), t_audio=1.0 - sa, out=store).float()
        return {"latent": x + s1 * v, "h": store["hidden"], "temb": store["temb_video"]}


def load_up_net(path, device="cuda"):
    from latent_up_h3 import LatentUp

    saved = torch.load(path, map_location="cpu")
    net = LatentUp(**saved["config"]).to(device).eval().requires_grad_(False)
    net.load_state_dict(saved["state"])
    return net


def drop_ctx(dit, spec: str):
    from pih3_cascade import dropped, parse_drop
    import contextlib

    return dropped(dit, parse_drop(spec)) if spec else contextlib.nullcontext()


# ------------------------------------------------------------------------------------------- teacher clips

def prompt_split(holdout: int):
    from accel_h3 import default_caches, prompt_lists

    return prompt_lists(default_caches(), holdout)


def u8(pixels):
    return pixels[0].float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 3, 0).cpu()   # (F,H,W,3)


@torch.no_grad()
def teacher(args) -> None:
    """pi-H3 4-step, full resolution: per clip the seed, the final latent, the DWD pixels of every native block
    decoded alone (the training target) and, for held-out clips, the whole clip's DWD pixels."""
    from accel_h3 import token_shape
    from pih3_cascade import refuse_if_card_busy
    from pih3_h3 import load_pih3, pih3_sample
    from setproctitle import setproctitle
    from train_dwd_h3 import load_text

    setproctitle(f"pih3:distill teacher {args.size}")
    refuse_if_card_busy(4.0)
    width, height = (int(v) for v in args.size.split("x"))
    latent_t, gh, gw = token_shape(width, height, args.frames)
    train, held = prompt_split(args.holdout)
    model = load_pih3(4)
    head = model["head"]
    jobs = [("held", i, te) for i, te in enumerate(held)] + [("train", i, te) for i, te in enumerate(train[:args.train])]
    root = args.out / f"{width}x{height}_f{args.frames}"
    start = time.time()
    for split, index, te in jobs:
        path = root / split / f"{index:04d}.pt"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        seed = (0 if split == "held" else 100000) + 7919 * index + 11
        text = load_text(str(te)).cuda()
        z, _ = paired_noise(latent_t, height, width, seed)
        out = pih3_sample(model, text, z, audio_noise(latent_t, seed + 1))
        blocks = []
        for b in range(math.ceil(latent_t / BLOCK)):
            a0, a1 = b * BLOCK * gh * gw, min(latent_t, (b + 1) * BLOCK) * gh * gw
            blocks.append(u8(head.decode(out["h"][a0:a1], out["temb"], (min(latent_t, (b + 1) * BLOCK) - b * BLOCK,
                                                                        gh, gw))))
        item = {"te": str(te), "seed": seed, "size": [width, height], "frames": args.frames, "latent_t": latent_t,
                "latent": out["latent"].half().cpu(), "blocks": blocks}
        if split == "held":
            item["pixels"] = u8(head.decode(out["h"], out["temb"], (latent_t, gh, gw)))
        torch.save(item, path)
        done = sum(1 for _ in root.glob("*/*.pt"))
        if done % 25 == 0:
            print(f"LOT_H3 kind=pih3_teacher done={done}/{len(jobs)} minutes={(time.time() - start) / 60:.1f}", flush=True)
    print(f"LOT_H3 kind=pih3_teacher_done n={len(jobs)} minutes={(time.time() - start) / 60:.1f} ok=1", flush=True)


# ------------------------------------------------------------------------------------------- training

def load_student(steps: int = 2, *, rank: int = 32, lora: Path | None = None, train: bool = True):
    """pi-H3 ``steps`` (merged, int8) with a bypass LoRA (``h3_musubi.make_lora``, unmerged), the differentiable
    head (``train``) or the inference head."""
    from h3_musubi import load_dit, make_lora
    from safetensors.torch import load_file

    dit = load_dit(device="cuda", swap=0, train=train)
    dwd, _counts = apply_adapter(dit, adapter_path(steps), "comfy")
    network = make_lora(dit, rank)
    network.to("cuda", torch.float32)
    if lora:
        saved = torch.load(lora, map_location="cpu", weights_only=True)
        missing = network.load_state_dict(saved["lora"], strict=False)
        if missing.unexpected_keys:
            raise SystemExit(f"{lora}: unexpected keys")
    network.requires_grad_(train)
    head = TrainHead(dwd, dit.final_layer.norm.eps) if train else PiH3Head(dwd, dit.final_layer.norm.eps)
    return dit, network, head


@torch.no_grad()
def held_eval(dit, head, up, lp, items, sigmas, drop: str) -> dict:
    """The fast path on held-out teacher clips: LPIPS of the whole clip against the teacher's DWD pixels, detail
    (``hf``) as a fraction of the teacher's, latent MSE. Inside training (same DiT, LoRA as it is)."""
    from accel_h3 import lpips_clip
    from pih3_cascade import hf_energy
    from train_dwd_h3 import load_text

    was = dit.training
    dit.eval()
    rows = {"lpips": [], "hf": [], "lat": []}
    for path in items:
        item = torch.load(path, weights_only=False)
        width, height = item["size"]
        latent_t = item["latent_t"]
        z, z_lr = paired_noise(latent_t, height, width, item["seed"])
        with drop_ctx(dit, drop):
            out = fast_path(dit, up, load_text(item["te"]).cuda(), z, z_lr, audio_noise(latent_t, item["seed"] + 1),
                            sigmas)
        px = head.decode(out["h"], out["temb"], (latent_t, height // 32, width // 32)).float()
        ref = item["pixels"].cuda().permute(3, 0, 1, 2).float().div(127.5).sub(1)[None]
        rows["lpips"].append(lpips_clip(lp, px, ref))
        rows["hf"].append(hf_energy(px) / hf_energy(ref))
        rows["lat"].append(float(F.mse_loss(out["latent"], item["latent"].cuda().float())))
    dit.train(was)
    return {k: round(sum(v) / len(v), 4) for k, v in rows.items()}


def train(args) -> None:
    import lpips
    from pih3_cascade import refuse_if_card_busy
    from setproctitle import setproctitle
    from train_dwd_h3 import load_text

    args.out.mkdir(parents=True, exist_ok=True)
    setproctitle(f"pih3:distill train r{args.rank} {args.drop or 'full'} {args.out.name}"[:60])
    refuse_if_card_busy(4.0)
    folder = args.data / args.folder
    items = sorted((folder / "train").glob("*.pt"))
    if not items:
        raise SystemExit(f"no teacher clips in {folder}/train")
    dit, network, head = load_student(rank=args.rank, lora=args.resume)
    params = [p for p in network.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    up = load_up_net(args.up)
    lp = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    sigmas = pih3_sigmas(2)
    rng = random.Random(args.seed)
    log = open(args.out / "log.jsonl", "a")
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1) + "\n")
    step0 = args.start_step
    held = sorted((folder / "held").glob("*.pt"))[:args.eval_held]

    def run_eval(step):
        start = time.time()
        row = {"step": step, "eval": held_eval(dit, head.head, up, lp, held, sigmas, args.drop),
               "eval_s": round(time.time() - start, 1)}
        print("LOT_H3 kind=pih3_distill_eval " + json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    if args.eval_every and not args.check:
        run_eval(step0)
    t_last = time.time()
    for step in range(step0, args.steps):
        item = torch.load(items[rng.randrange(len(items))], weights_only=False)
        width, height = item["size"]
        latent_t = item["latent_t"]
        gh, gw = height // 32, width // 32
        text = load_text(item["te"]).cuda()
        z, z_lr = paired_noise(latent_t, height, width, item["seed"])
        za = audio_noise(latent_t, item["seed"] + 1)
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for group in opt.param_groups:
            group["lr"] = lr
        with drop_ctx(dit, args.drop):
            out = fast_path(dit, up, text, z, z_lr, za, sigmas, grad=True)
            target = item["latent"].cuda().float()
            loss_lat = F.mse_loss(out["latent"], target)
            block = rng.randrange(len(item["blocks"]))
            px = head.decode_block(out["h"], out["temb"], (latent_t, gh, gw), block)          # (1, 3, F, H, W)
            ref = item["blocks"][block].cuda().permute(3, 0, 1, 2).float().div(127.5).sub(1)[None]
            frames = sorted(rng.sample(range(px.shape[2]), min(args.lpips_frames, px.shape[2])))
            c = args.lpips_crop                               # one full-resolution crop, the same on both sides
            cy, cx = rng.randrange(px.shape[3] - c + 1), rng.randrange(px.shape[4] - c + 1)
            crop = (slice(None), slice(None), frames, slice(cy, cy + c), slice(cx, cx + c))
            loss_lp = lp(px[crop][0].transpose(0, 1).float(), ref[crop][0].transpose(0, 1)).mean()
            loss = args.lat_weight * loss_lat + args.lpips_weight * loss_lp
            if args.hf_weight:                                # detail energy below the teacher's, on the same crop
                def energy(v):
                    luma = v[crop][0].mean(0).unsqueeze(1).float()                 # (frames, 1, c, c)
                    return (luma - F.avg_pool2d(luma, 5, 1, 2, count_include_pad=False)).abs().mean()
                loss_hf = F.relu(1.0 - energy(px) / energy(ref).clamp_min(1e-6))
                loss = loss + args.hf_weight * loss_hf
            # inside the context: activation checkpointing recomputes the blocks during backward, and a block that
            # was skipped in the forward must be skipped again (runs before this fix: run_b, run_b2 were not)
            opt.zero_grad(set_to_none=True)
            loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        opt.step()
        if step % 10 == 0 or args.check:
            now = time.time()
            row = {"step": step, "loss": round(float(loss), 5), "lat": round(float(loss_lat), 5),
                   "lpips": round(float(loss_lp), 4), "grad": round(norm, 4), "lr": lr,
                   "s_per_step": round((now - t_last) / (1 if args.check else 10), 2) if step > step0 else None,
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            t_last = now
            print("LOT_H3 kind=pih3_distill " + json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
        if args.check and step >= step0 + 3:
            print("LOT_H3 kind=pih3_distill_check ok=1", flush=True)
            return
        if (step + 1) % args.save_every == 0:
            torch.save({"lora": {k: v.detach().cpu() for k, v in network.state_dict().items()}, "rank": args.rank,
                        "step": step + 1, "drop": args.drop, "up": str(args.up)}, args.out / f"lora_{step + 1}.pt")
            print(f"LOT_H3 kind=pih3_distill_save step={step + 1}", flush=True)
        if args.eval_every and (step + 1) % args.eval_every == 0:
            run_eval(step + 1)
            t_last = time.time()
    print("LOT_H3 kind=pih3_distill_done ok=1", flush=True)


# ------------------------------------------------------------------------------------------- DMD

class BypassAdapter:
    """The pi-H3 adapter on the DiT unmerged and switchable (``off()``), so base H3 can be DMD's real score next to
    the student: the rank-64 LoRA as forward hooks, the replaced int8 layers, AdaLN weights and final layer as
    tensor swaps (both versions kept on the card, about 1.5 GB)."""

    def __init__(self, dit, path: Path):
        import contextlib
        import json as _json

        from safetensors.torch import load_file

        sd = load_file(str(path))
        self.dwd = {k: sd.pop(k) for k in list(sd) if k.startswith("dwd_")}
        modules = dict(dit.named_modules())
        tensors = dict(dit.named_parameters())
        tensors.update(dict(dit.named_buffers()))
        self.enabled = True
        self.swaps = []                                     # (tensor, base value, adapter value)
        used = set()
        for layer in sorted({k[:-len(".comfy_quant")] for k in sd if k.endswith(".comfy_quant")}):
            module = modules[layer]
            spec = _json.loads(bytes(sd[layer + ".comfy_quant"].tolist()).decode())
            if spec.get("convrot_groupsize") != module._convrot_groupsize or module.weight.dtype != torch.int8:
                raise SystemExit(f"{layer}: int8 spec mismatch")
            self.swaps.append((module.weight, module.weight.data, sd[layer + ".weight"].to(module.weight.device)))
            self.swaps.append((module.scale_weight, module.scale_weight.data,
                               sd[layer + ".weight_scale"].to(module.scale_weight.device, torch.float32)
                               .view_as(module.scale_weight)))
            used |= {layer + ".weight", layer + ".weight_scale", layer + ".comfy_quant"}
        for layer in sorted({k.rsplit(".", 2)[0] for k in sd if k.endswith(".lora_A.weight")}):
            a = sd[layer + ".lora_A.weight"].to("cuda", torch.bfloat16)
            b = sd[layer + ".lora_B.weight"].to("cuda", torch.bfloat16)

            def hook(_module, inputs, output, a=a, b=b):
                if not self.enabled:
                    return output
                return output + F.linear(F.linear(inputs[0].to(torch.bfloat16), a), b).to(output.dtype)

            modules[layer].register_forward_hook(hook)
            used |= {layer + ".lora_A.weight", layer + ".lora_B.weight"}
        for key, value in sd.items():
            if key in used:
                continue
            target = tensors[key]
            self.swaps.append((target, target.data, value.to(target.device, target.dtype)))
            used.add(key)
        if set(sd) - used:
            raise SystemExit("adapter tensors not consumed")
        self.set(True)
        self.off = contextlib.contextmanager(self._off)

    def set(self, on: bool) -> None:
        self.enabled = on
        for tensor, base, adapter in self.swaps:
            tensor.data = adapter if on else base

    def _off(self):
        self.set(False)
        try:
            yield
        finally:
            self.set(True)


def dmd_train(args) -> None:
    """Distribution matching (DiR's recipe, ``train_dir_h3.py``) for the fast path: the student's final latent is
    pushed along ``-(x0_fake - x0_real)`` at a random noise level, the real score being base H3 (the pi-H3 adapter
    switched off) and the fake score base H3 plus a fake LoRA trained on the student's samples; the latent MSE to the
    teacher stays as a small anchor (``--lat-weight``)."""
    from dir_h3 import use_networks
    from h3_musubi import forward_video, load_dit, make_lora
    from pih3_cascade import refuse_if_card_busy
    from setproctitle import setproctitle
    from train_dwd_h3 import load_text

    args.out.mkdir(parents=True, exist_ok=True)
    setproctitle(f"pih3:dmd r{args.rank} {args.drop or 'full'} {args.out.name}"[:60])
    refuse_if_card_busy(4.0)
    folder = args.data / args.folder
    items = sorted((folder / "train").glob("*.pt"))
    held = sorted((folder / "held").glob("*.pt"))[:args.eval_held]
    dit = load_dit(device="cuda", swap=0, train=True)
    adapter = BypassAdapter(dit, adapter_path(2))
    gen_net = make_lora(dit, args.rank)
    fake_net = make_lora(dit, args.fake_rank)
    for net in (gen_net, fake_net):
        net.to("cuda", torch.float32)
        net.requires_grad_(True)
    head = PiH3Head(adapter.dwd, dit.final_layer.norm.eps)
    gen_params = [p for p in gen_net.parameters() if p.requires_grad]
    fake_params = [p for p in fake_net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(gen_params, lr=args.lr, betas=(0.0, 0.95), weight_decay=0.0, foreach=False)
    fake_opt = torch.optim.AdamW(fake_params, lr=args.fake_lr, betas=(0.0, 0.95), weight_decay=0.0, foreach=False)
    up = load_up_net(args.up)
    sigmas = pih3_sigmas(2)
    rng = random.Random(args.seed)
    gen = torch.Generator("cuda").manual_seed(args.seed)
    log = open(args.out / "log.jsonl", "a")
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1) + "\n")

    def x0_at(x_tau, tau, text, gen_on, fake_on, adapter_on):
        import contextlib

        ctx = contextlib.nullcontext() if adapter_on else adapter.off()
        with ctx, use_networks(gen_net, fake_net, gen=gen_on, fake=fake_on):
            return x_tau + tau * forward_video(dit, x_tau, 1.0 - tau, text).float()

    def run_eval(step):
        import lpips

        lp = getattr(run_eval, "lp", None) or lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
        run_eval.lp = lp
        with use_networks(gen_net, fake_net, gen=True, fake=False):
            row = {"step": step, "eval": held_eval(dit, head, up, lp, held, sigmas, args.drop)}
        print("LOT_H3 kind=pih3_distill_eval " + json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    if args.eval_every and not args.check:
        run_eval(args.start_step)
    t_last = time.time()
    for step in range(args.start_step, args.steps):
        item = torch.load(items[rng.randrange(len(items))], weights_only=False)
        width, height = item["size"]
        latent_t = item["latent_t"]
        text = load_text(item["te"]).cuda()
        z, z_lr = paired_noise(latent_t, height, width, item["seed"])
        za = audio_noise(latent_t, item["seed"] + 1)
        warm = min(1.0, (step + 1) / args.warmup)
        for group in opt.param_groups:
            group["lr"] = args.lr * warm
        # fake updates first, on a detached student sample (the paper's order: fake tracks the current student)
        with torch.no_grad(), drop_ctx(dit, args.drop), use_networks(gen_net, fake_net, gen=True, fake=False):
            sample = fast_path(dit, up, text, z, z_lr, za, sigmas)["latent"]
        fake_loss = 0.0
        for _ in range(args.fake_updates):
            tau = 0.02 + (args.tau_max - 0.02) * float(torch.rand((), generator=gen, device="cuda"))
            eps = torch.randn(sample.shape, generator=gen, device="cuda")
            x_tau = (1.0 - tau) * sample + tau * eps
            with torch.enable_grad(), use_networks(gen_net, fake_net, gen=False, fake=True), adapter.off():
                v = forward_video(dit, x_tau, 1.0 - tau, text).float()
                loss_f = F.mse_loss(v, sample - eps)
                fake_opt.zero_grad(set_to_none=True)
                loss_f.backward()                       # inside: the checkpoint recompute needs the same switches
            torch.nn.utils.clip_grad_norm_(fake_params, args.clip)
            fake_opt.step()
            fake_loss += float(loss_f) / args.fake_updates
        # generator update
        # the DMD direction from the current student's sample (no grad), then one graph for the update, with its
        # backward inside the same switches (checkpoint recompute)
        with torch.no_grad():
            tau = 0.02 + (args.tau_max - 0.02) * float(torch.rand((), generator=gen, device="cuda"))
            eps = torch.randn(sample.shape, generator=gen, device="cuda")
            # the generator has not changed since ``sample`` (the fake update only moves the fake LoRA)
            x_tau = (1.0 - tau) * sample + tau * eps
            real = x0_at(x_tau, tau, text, False, False, False)
            fake = x0_at(x_tau, tau, text, False, True, False)
            weight = 1.0 / (sample - real).abs().mean().clamp_min(1e-4)                # DiR's --dmd-weight norm
            grad = (fake - real) * weight
            target = (sample - grad).detach()
        with drop_ctx(dit, args.drop), use_networks(gen_net, fake_net, gen=True, fake=False):
            x0_hat = fast_path(dit, up, text, z, z_lr, za, sigmas, grad=True)["latent"]
            loss_dmd = 0.5 * F.mse_loss(x0_hat, target)
            loss_lat = F.mse_loss(x0_hat, item["latent"].cuda().float())
            loss = loss_dmd + args.lat_weight * loss_lat
            opt.zero_grad(set_to_none=True)
            loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(gen_params, args.clip))
        opt.step()
        if step % 10 == 0 or args.check:
            now = time.time()
            row = {"step": step, "dmd": round(float(loss_dmd), 5), "lat": round(float(loss_lat), 5),
                   "fake": round(fake_loss, 5), "gap": round(float((fake - real).abs().mean()), 5),
                   "tau": round(tau, 3), "grad": round(norm, 4),
                   "s_per_step": round((now - t_last) / (1 if args.check else 10), 2) if step > args.start_step else None,
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            t_last = now
            print("LOT_H3 kind=pih3_dmd " + json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
        if args.check and step >= args.start_step + 3:
            print("LOT_H3 kind=pih3_dmd_check ok=1", flush=True)
            return
        if (step + 1) % args.save_every == 0:
            torch.save({"lora": {k: v.detach().cpu() for k, v in gen_net.state_dict().items()}, "rank": args.rank,
                        "step": step + 1, "drop": args.drop, "up": str(args.up), "adapter": "bypass"},
                       args.out / f"lora_{step + 1}.pt")
            torch.save({"lora": {k: v.detach().cpu() for k, v in fake_net.state_dict().items()},
                        "rank": args.fake_rank}, args.out / "fake_last.pt")
        if args.eval_every and (step + 1) % args.eval_every == 0:
            run_eval(step + 1)
            t_last = time.time()
    print("LOT_H3 kind=pih3_dmd_done ok=1", flush=True)


# ------------------------------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(args) -> None:
    """Held-out gate: time, LPIPS against the teacher (whole clip, DWD pixels), high-frequency detail as a fraction of
    the teacher's, per variant; contact sheets. Variants: ``cascade`` (no LoRA, the baseline), ``student`` (each
    ``--run`` checkpoint), ``full2drop`` (2-step full resolution, H3-Tiny's blocks dropped, no LoRA)."""
    import lpips
    from pih3_cascade import hf_energy, refuse_if_card_busy, sheet
    from pih3_h3 import pih3_sample
    from setproctitle import setproctitle
    from train_dwd_h3 import load_text

    setproctitle(f"pih3:distill eval {args.out.name}"[:60])
    refuse_if_card_busy(4.0)
    folder = args.data / args.folder
    items = sorted((folder / "held").glob("*.pt"))[:args.held]
    dit, network, head = load_student(lora=None, train=False)
    up = load_up_net(args.up)
    lp = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    sigmas = pih3_sigmas(2)
    variants = [("cascade", None, args.drop), ("full2drop", None, "h3tiny")]
    for ckpt in args.lora or []:
        variants.append((Path(ckpt).stem + (f"_{args.drop}" if args.drop else ""), ckpt, args.drop))
    rows = {name: {"seconds": [], "lpips": [], "hf": []} for name, _c, _d in variants}
    clips = {}
    for name, ckpt, drop in variants:
        if ckpt:
            saved = torch.load(ckpt, map_location="cpu", weights_only=True)
            network.load_state_dict(saved["lora"], strict=False)
        for module in network.unet_loras:            # off = not computed (multiplier 0 would still cost time)
            module.enabled = bool(ckpt)
        for k, path in enumerate(items):
            item = torch.load(path, weights_only=False)
            width, height = item["size"]
            latent_t = item["latent_t"]
            text = load_text(item["te"]).cuda()
            z, z_lr = paired_noise(latent_t, height, width, item["seed"])
            za = audio_noise(latent_t, item["seed"] + 1)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with drop_ctx(dit, drop):
                if name == "full2drop":
                    model = {"dit": dit, "sigmas": sigmas}
                    out = pih3_sample(model, text, z, za)
                else:
                    out = fast_path(dit, up, text, z, z_lr, za, sigmas)
                px = head.decode(out["h"], out["temb"], (latent_t, height // 32, width // 32))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            ref = item["pixels"].cuda().permute(3, 0, 1, 2).float().div(127.5).sub(1)[None]
            from accel_h3 import lpips_clip

            rows[name]["seconds"].append(seconds)
            rows[name]["lpips"].append(lpips_clip(lp, px.float(), ref))
            rows[name]["hf"].append(hf_energy(px) / hf_energy(ref))
            if k < args.sheets:
                clips.setdefault(k, {"teacher 4-step": item["pixels"]})[name] = u8(px)
        print(f"LOT_H3 kind=pih3_distill_eval variant={name} " + " ".join(
            f"{key}={sorted(v)[len(v) // 2]:.4f}" for key, v in rows[name].items()), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for k, clip_set in clips.items():
        sheet(clip_set, args.out / f"sheet_p{k}.jpg", columns=4, tile_w=320)
    summary = {name: {key: (sorted(v[1:])[len(v[1:]) // 2] if key == "seconds" else sum(v) / len(v))
                      for key, v in r.items()} for name, r in rows.items()}
    (args.out / "eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--size", default="768x512")
    t.add_argument("--frames", type=int, default=39)
    t.add_argument("--train", type=int, default=500)
    t.add_argument("--holdout", type=int, default=8, help="held-out prompts per cache (the bench's held-out set)")
    r = sub.add_parser("train")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--folder", default="768x512_f39")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--rank", type=int, default=32)
    r.add_argument("--lr", type=float, default=1e-4)
    r.add_argument("--warmup", type=int, default=50)
    r.add_argument("--clip", type=float, default=1.0)
    r.add_argument("--steps", type=int, default=3000)
    r.add_argument("--start-step", type=int, default=0)
    r.add_argument("--resume", type=Path, default=None)
    r.add_argument("--save-every", type=int, default=250)
    r.add_argument("--eval-every", type=int, default=250, help="held-out eval in the run (0: off); also at the start")
    r.add_argument("--eval-held", type=int, default=8)
    r.add_argument("--lpips-weight", type=float, default=0.2)
    r.add_argument("--hf-weight", type=float, default=0.0,
                   help="penalty on the crop's high-frequency energy falling below the teacher's (position-free)")
    r.add_argument("--lat-weight", type=float, default=1.0, help="weight of the latent MSE (alone it regresses to a "
                   "blurry mean: run_b at 1.0 / 0.2 had detail 0.41 at step 250)")
    r.add_argument("--lpips-frames", type=int, default=4)
    r.add_argument("--lpips-crop", type=int, default=256, help="LPIPS on one full-resolution crop of this size")
    r.add_argument("--drop", default="", help="h3tiny, or block indices joined by '-'")
    r.add_argument("--up", type=Path, default=Path(UP_DEFAULT))
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--check", action="store_true")
    m = sub.add_parser("dmd", help="distribution matching, base H3 as the real score, the latent MSE as an anchor")
    m.add_argument("--data", type=Path, required=True)
    m.add_argument("--folder", default="768x512_f39")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--rank", type=int, default=32)
    m.add_argument("--fake-rank", type=int, default=32)
    m.add_argument("--lr", type=float, default=2e-5)
    m.add_argument("--fake-lr", type=float, default=2e-5)
    m.add_argument("--fake-updates", type=int, default=1)
    m.add_argument("--tau-max", type=float, default=0.9, help="H3 is CFG-distilled: above about 0.9 its x0 is the baked guidance")
    m.add_argument("--lat-weight", type=float, default=0.05)
    m.add_argument("--warmup", type=int, default=50)
    m.add_argument("--clip", type=float, default=1.0)
    m.add_argument("--steps", type=int, default=1000)
    m.add_argument("--start-step", type=int, default=0)
    m.add_argument("--save-every", type=int, default=250)
    m.add_argument("--eval-every", type=int, default=250)
    m.add_argument("--eval-held", type=int, default=8)
    m.add_argument("--drop", default="")
    m.add_argument("--up", type=Path, default=Path(UP_DEFAULT))
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--check", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--folder", default="768x512_f39")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--lora", nargs="*", default=None)
    e.add_argument("--drop", default="")
    e.add_argument("--held", type=int, default=16)
    e.add_argument("--sheets", type=int, default=4)
    e.add_argument("--up", type=Path, default=Path(UP_DEFAULT))
    args = p.parse_args()
    {"teacher": teacher, "train": train, "dmd": dmd_train, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
