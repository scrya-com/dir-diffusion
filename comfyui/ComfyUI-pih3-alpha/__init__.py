# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""Transparent video from pi-H3 2-step: our alpha Stage B as ComfyUI nodes (research checkpoint).

The pi-H3 model's own forward is untouched (RGB is pi-H3's). At the last sampling step, on the positive pass, the
input of block ``50 - K`` is kept, and after the last block, blocks ``50 - K .. 49`` run again with the alpha LoRA
on; their output goes through pi-H3's DWD decoder (``dwd_norm_out`` + the Sol refiner) and our alpha head reads the
refiner's tokens (with pi-H3's RGB). Needs ComfyUI-piFlow (pi-H3's DWD head classes) and its DWD adapter.

Nodes:
* ``pi-H3 Alpha (Stage B)``: MODEL in (the pi-H3 model, after its adapter), MODEL out with the alpha pass.
* ``pi-H3 Alpha Decode``: the alpha (MASK, F x H x W) of the sampling run that returned the LATENT.
* ``pi-H3 Alpha Refine Sigmas``: SIGMAS [sigma, 0] for the refine-pass mode (re-noise the generated latent to sigma,
  one more step with the alpha model; at sigma ~0.6 pi-H3 redraws what it is shown, the regime the head trained in).
* ``pi-H3 Alpha Solidify``: MASK -> solid matte for solid subjects (holes filled, opaque inside, soft edge).
* ``Save RGBA Video``: IMAGE + MASK -> VP9-with-alpha WebM, ProRes 4444 .mov and/or a PNG sequence.

Trained on the AlphaAnything RGBA set (CC BY-NC 4.0): research use only.
"""

from __future__ import annotations

import collections
import hashlib
import math
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
import comfy.patcher_extension
import comfy.utils
import folder_paths

ALPHA_DIR = os.path.join(folder_paths.models_dir, "pih3_alpha")
os.makedirs(ALPHA_DIR, exist_ok=True)
folder_paths.folder_names_and_paths["pih3_alpha"] = ([ALPHA_DIR], {".pt", ".pth"})

PATCH, PATCH_T, FIRST_T = 16, 4, 1


# ------------------------------------------------------------------------------------------- the alpha head
class _Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))
        return x + self.ff(self.norm2(x))


class AlphaDWDHead(nn.Module):
    """As ``alpha/pih3_alpha.py`` (feat/alpha-dwd): Sol refiner tokens -> alpha patches -> pixel refiner."""

    def __init__(self, sol_dim=2048, dim=384, heads=6, blocks=3, refine_ch=48):
        super().__init__()
        self.norm_in = nn.LayerNorm(sol_dim, elementwise_affine=False)
        self.proj_in = nn.Linear(sol_dim, dim)
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(blocks)])
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, PATCH_T * PATCH * PATCH)
        self.first_proj_out = nn.Linear(dim, FIRST_T * PATCH * PATCH)
        c = refine_ch
        self.refine = nn.Sequential(
            nn.Conv2d(4, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, c, 3, padding=2, dilation=2), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=4, dilation=4), nn.SiLU(), nn.Conv2d(c, c, 3, padding=8, dilation=8), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def coarse(self, chunks):
        out = []
        for y in chunks:
            n, h, w, _d = y.shape
            x = self.proj_in(self.norm_in(y.float())).reshape(1, n * h * w, -1)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm_out(x).view(n, h, w, -1)
            first = self.first_proj_out(x[:1]).view(h, w, FIRST_T, PATCH, PATCH).permute(2, 0, 3, 1, 4)
            out.append(first.reshape(FIRST_T, 1, h * PATCH, w * PATCH))
            if n > 1:
                rest = self.proj_out(x[1:]).view(n - 1, h, w, PATCH_T, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5)
                out.append(rest.reshape((n - 1) * PATCH_T, 1, h * PATCH, w * PATCH))
        return torch.cat(out)

    def forward(self, chunks, rgb, frames_per_chunk: int = 4):
        """Alpha score (F, 1, H, W) on the CPU. The pixel stage runs a few frames at a time: one pass over every
        frame at about 1 MP and 90 frames needs ~17 GB."""
        c = self.coarse(chunks)                                   # (F, 1, H, W), small
        out = []
        for i in range(0, c.shape[0], frames_per_chunk):
            cc = c[i:i + frames_per_chunk]
            rr = rgb[i:i + frames_per_chunk].to(cc.device, torch.float32)
            out.append((cc + self.refine(torch.cat([rr, cc], 1))).cpu())
        return torch.cat(out)


def score_to_alpha(score):
    u = ((score.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    return torch.sin(u * (math.pi / 2.0)).square()


# ------------------------------------------------------------------------------------------- piFlow pieces
def _piflow(name: str):
    for key, mod in sys.modules.items():
        if key.endswith(name) and "piFlow" in key:
            return mod
    raise RuntimeError(f"ComfyUI-piFlow is needed (module {name} not loaded)")


ALPHA = collections.OrderedDict()   # alpha of the latest sampling runs, by the fingerprint of the video latent
ALPHA_KEY = "pih3_alpha_holder"


def _fingerprint(latent):
    return hashlib.sha1(latent.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


class _Holder:
    def __init__(self):
        self.h_in = None
        self.alpha = None


def _outer_sample_wrapper(executor, noise, latent_image, sampler, sigmas, *args, latent_shapes=None, **kwargs):
    holder = _Holder()
    executor.class_obj.model_options.setdefault("transformer_options", {})[ALPHA_KEY] = holder
    output = executor(noise, latent_image, sampler, sigmas, *args, latent_shapes=latent_shapes, **kwargs)
    if holder.alpha is not None:
        video = output if latent_shapes is None or len(latent_shapes) < 2 else comfy.utils.unpack_latents(output, latent_shapes)[0]
        ALPHA[_fingerprint(video)] = holder.alpha
        while len(ALPHA) > 4:
            ALPHA.popitem(last=False)
    return output


class PiH3Alpha:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "alpha_checkpoint": (folder_paths.get_filename_list("pih3_alpha"),),
            "dwd_adapter": (folder_paths.get_filename_list("loras"), {"default": "pi_h3_2step_dwd_adapter_int8_convrot.safetensors"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "pi-H3/alpha"
    DESCRIPTION = "Adds the Stage B alpha pass to a pi-H3 model (after its LakonLab adapter). RGB is unchanged."

    def patch(self, model, alpha_checkpoint, dwd_adapter):
        st = torch.load(folder_paths.get_full_path_or_raise("pih3_alpha", alpha_checkpoint), map_location="cpu", weights_only=False)
        k = int(st["k"])
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", dwd_adapter))
        dwd_sd = {key: v for key, v in sd.items() if key.startswith("dwd_")}
        del sd
        arch = _piflow("modules.architectures.minimax_h3_dwd")
        loader = _piflow("modules.loader_utils")
        dm = model.model.diffusion_model
        dwd = arch.MiniMaxH3DWD(dwd_sd, dm.final_layer.norm.eps, dtype=dwd_sd["dwd_norm_out.linear.weight"].dtype,
                                device="cpu", operations=comfy.ops.mixed_precision_ops())
        loader.load_adapter_module(dwd, dwd_sd)
        for mod in dwd.modules():                      # cast weights to the activations' dtype, as piFlow's patcher does
            if hasattr(mod, "comfy_cast_weights"):
                mod.comfy_cast_weights = True
        head = AlphaDWDHead()
        head.load_state_dict(st["head"])
        n_blocks = len(dm.blocks)
        first = n_blocks - k
        lora = {}
        for key, v in st["lora"].items():
            name, kind = key.split(".", 1)
            lora.setdefault(name, {})[kind] = v
        targets = []
        for name, t in lora.items():
            parts = name[len("lora_unet_blocks_"):].split("_", 1)
            idx, sub = int(parts[0]), parts[1]          # e.g. 42, attn_qkv_proj
            group, attr = sub.split("_", 1)             # attn, qkv_proj | mlp, fc1
            scale = float(t["alpha"]) / t["lora_down.weight"].shape[0] if "alpha" in t else 1.0
            targets.append((idx, group, attr, t["lora_down.weight"], t["lora_up.weight"], scale))
        state = {"dwd": dwd, "head": head, "targets": targets, "dev": None}

        def to_device(dev, dtype):
            if state["dev"] != dev:
                dwd.to(dev); head.to(dev)
                state["lora_t"] = [(i, g, a, d.to(dev, dtype), u.to(dev, dtype), s) for i, g, a, d, u, s in targets]
                state["dev"] = dev

        def is_last(to):
            sig, sample = to.get("sigmas"), to.get("sample_sigmas")
            if sig is None or sample is None or to.get("cond_or_uncond") != [0]:
                return False
            return bool(torch.isclose(sig.flatten()[0].float(), sample[sample > 0][-1].float().to(sig.device)))

        def keep_input(args, extra):
            to = args["transformer_options"]
            holder = to.get(ALPHA_KEY)
            if holder is not None and is_last(to):
                # blocks update the residual stream in place, and ComfyUI allocates per block (scoped async pools):
                # keep a copy off the GPU so it outlives this block's scope
                holder.h_in = args["img"].detach().to("cpu", copy=True)
            return extra["original_block"](args)

        def tail(args, extra):
            out = extra["original_block"](args)
            to = args["transformer_options"]
            holder = to.get(ALPHA_KEY)
            if holder is None or holder.h_in is None or not is_last(to):
                return out
            dev, dtype = out["img"].device, out["img"].dtype
            h = holder.h_in.to(dev)
            holder.h_in = None
            to_device(dev, dtype)
            hooks, forced = [], []
            mods = dm.blocks
            for i, g, a, d, u, s in state["lora_t"]:
                lin = getattr(getattr(mods[i], g), a)
                hooks.append(lin.register_forward_hook(
                    lambda m, inp, o, d=d, u=u, s=s: o + F.linear(F.linear(inp[0].to(d.dtype), d), u).to(o.dtype) * s))
                if not getattr(lin, "comfy_force_forward", False):
                    lin.comfy_force_forward = True      # fused fc2 calls skip forward hooks; the eager path does not
                    forced.append(lin)
            try:
                with torch.no_grad():
                    for i in range(first, n_blocks):
                        h = mods[i](h, args["t_emb"], args["mod_segments"], args["rope_freqs"], transformer_options=to)
                    rgb = dwd.decode(out["img"], args["t_emb"], args["mod_segments"], args["layout"])   # (F, H, W, 3)
                    chunks = []
                    hk = dwd.dwd_pixel_refiner.norm_out.register_forward_pre_hook(lambda m, a_: chunks.append(a_[0]))
                    try:
                        dwd.decode(h, args["t_emb"], args["mod_segments"], args["layout"])
                    finally:
                        hk.remove()
                    _, lt, lh, lw, _ = args["layout"].signature
                    sh, sw = lh // 2 * 2, lw // 2 * 2
                    chunks = [c.view(-1, sh, sw, c.shape[-1]) for c in chunks]
                    rgb_m = rgb.permute(0, 3, 1, 2).mul(2).sub(1)                    # (F, 3, H, W) in [-1, 1], stays where DWD put it
                    del h
                    score = head(chunks, rgb_m)                                       # (F, 1, H, W) on the CPU
                    holder.alpha = score_to_alpha(score)[:, 0].float()                # (F, H, W)
            finally:
                for hk in hooks:
                    hk.remove()
                for lin in forced:
                    lin.comfy_force_forward = False
            return out

        m = model.clone()
        m.set_model_patch_replace(keep_input, "dit", "double_block", first)
        m.set_model_patch_replace(tail, "dit", "double_block", n_blocks - 1)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "pih3_alpha", _outer_sample_wrapper)
        return (m,)


class PiH3AlphaDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "decode"
    CATEGORY = "pi-H3/alpha"
    DESCRIPTION = "The alpha of the sampling run (with a pi-H3 Alpha model) that returned this LATENT."

    def decode(self, samples):
        latent = samples["samples"]
        video = latent.unbind()[0] if latent.is_nested else latent
        a = ALPHA.get(_fingerprint(video))
        if a is None:
            raise RuntimeError("No alpha for this latent: connect the output of a sampler whose model went through "
                               "'pi-H3 Alpha (Stage B)' (the latest 4 runs are kept)")
        nan = float(torch.isnan(a.float()).float().mean())
        if nan > 0:
            print(f"[pih3-alpha] WARNING: decoded alpha is {nan:.1%} NaN ({tuple(a.shape)}); connect it to "
                  "'pi-H3 Alpha Solidify' with a fallback_alpha, or use the last-step alpha", flush=True)
        return (a,)


class PiH3AlphaRefineSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sigma": ("FLOAT", {"default": 0.6, "min": 0.05, "max": 0.99, "step": 0.01})}}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "make"
    CATEGORY = "pi-H3/alpha"
    DESCRIPTION = "Refine-pass mode: one step from this sigma to 0 on the generated latent, for alpha only."

    def make(self, sigma):
        return (torch.tensor([float(sigma), 0.0]),)


class SaveRGBAVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",), "alpha": ("MASK",),
            "filename_prefix": ("STRING", {"default": "pih3_alpha/rgba"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            "format": (["webm (VP9 alpha)", "mov (ProRes 4444)", "png sequence", "all"],),
        }}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "pi-H3/alpha"

    def save(self, images, alpha, filename_prefix, fps, format):
        from PIL import Image
        f = min(images.shape[0], alpha.shape[0])
        rgb = images[:f].float().clamp(0, 1)
        a = alpha[:f].float().clamp(0, 1)
        if a.shape[-2:] != rgb.shape[1:3]:
            a = F.interpolate(a[:, None], size=rgb.shape[1:3], mode="bilinear", align_corners=False)[:, 0]
        out_dir, name, counter, sub, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(),
                                                                          rgb.shape[2], rgb.shape[1])
        stem = os.path.join(out_dir, f"{name}_{counter:05}")
        os.makedirs(stem + "_png", exist_ok=True)
        for t in range(f):
            rgba = torch.cat([rgb[t], a[t, :, :, None]], -1).cpu().numpy()
            Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), "RGBA").save(f"{stem}_png/{t:04d}.png")
        results = []
        jobs = []
        if format in ("webm (VP9 alpha)", "all"):
            jobs.append(("webm", ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "24", "-auto-alt-ref", "0"]))
        if format in ("mov (ProRes 4444)", "all"):
            jobs.append(("mov", ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]))
        for ext, codec in jobs:
            path = f"{stem}.{ext}"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{stem}_png/%04d.png",
                            *codec, path], check=True)
            results.append({"filename": os.path.basename(path), "subfolder": sub, "type": "output"})
        return {"ui": {"text": [f"{f} RGBA frames -> {stem}.*"]}}


class PiH3AlphaSolidify:
    """Solid-subject matte: threshold, close small gaps, fill enclosed holes, drop specks (floor-shadow haze), fully
    opaque inside, the model's soft alpha kept only in a thin band at the silhouette edge. For characters and props;
    not for smoke or glass. ``fix_first``/``fix_last`` copy the neighbour frame's matte into frames given as a
    first/last frame (conditions, not generated: their alpha is broken)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "alpha": ("MASK",),
            "threshold": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05}),
            "feather": ("INT", {"default": 3, "min": 0, "max": 16}),
            "min_area": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 0.1, "step": 0.001,
                                   "tooltip": "drop opaque islands smaller than this fraction of the frame"}),
            "fix_first": ("BOOLEAN", {"default": True}),
            "fix_last": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "fallback_alpha": ("MASK", {"tooltip": "used for frames where 'alpha' is NaN (e.g. the last-step alpha when "
                                                   "the refine pass failed)"}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "solidify"
    CATEGORY = "pi-H3/alpha"
    DESCRIPTION = "Solid matte for solid subjects (holes filled, interior opaque, soft edge)."

    def solidify(self, alpha, threshold, feather, min_area, fix_first, fix_last, fallback_alpha=None):
        from scipy import ndimage as nd
        a_all = alpha.float().cpu().numpy()
        bad = np.isnan(a_all).reshape(len(a_all), -1).mean(1) > 0.01
        if bad.any():
            fb = None
            if fallback_alpha is not None:
                fb = fallback_alpha.float()
                if fb.shape[-2:] != alpha.shape[-2:]:
                    fb = F.interpolate(fb[:, None], size=alpha.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
                fb = fb.cpu().numpy()
            for t in np.flatnonzero(bad):
                a_all[t] = fb[min(t, len(fb) - 1)] if fb is not None else 0.0
            print(f"[pih3-alpha] WARNING: alpha had NaN in {int(bad.sum())}/{len(bad)} frames; "
                  + ("used the fallback alpha there" if fb is not None else "no fallback_alpha connected: those frames are empty"),
                  flush=True)
        a_all = np.nan_to_num(a_all, nan=0.0).clip(0, 1)
        out = np.empty_like(a_all)
        for t, a in enumerate(a_all):
            m = nd.binary_closing(a > threshold, structure=np.ones((5, 5)), iterations=1)
            m = nd.binary_fill_holes(m)
            lab, n = nd.label(m)
            if n:
                sizes = nd.sum(m, lab, range(1, n + 1))
                keep = np.isin(lab, 1 + np.flatnonzero(sizes >= min_area * a.size))
                m = keep if keep.any() else m
            if feather > 0:
                core = nd.binary_erosion(m, iterations=feather)
                band = nd.binary_dilation(m, iterations=feather) & ~core
                soft = np.clip((a - 0.1) / 0.5, 0, 1)
                o = np.where(core, 1.0, 0.0)
                o = np.where(band, np.maximum(soft, nd.gaussian_filter(m.astype(np.float32), feather / 2)), o)
            else:
                o = m.astype(np.float32)
            out[t] = o
        f = len(out)
        if fix_first and f > 1:
            out[0] = out[1]
        if fix_last and f > 1:
            out[-1] = out[-2]
        return (torch.from_numpy(out),)


NODE_CLASS_MAPPINGS = {
    "PiH3AlphaStageB": PiH3Alpha,
    "PiH3AlphaDecode": PiH3AlphaDecode,
    "PiH3AlphaRefineSigmas": PiH3AlphaRefineSigmas,
    "SaveRGBAVideo": SaveRGBAVideo,
    "PiH3AlphaSolidify": PiH3AlphaSolidify,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PiH3AlphaStageB": "pi-H3 Alpha (Stage B)",
    "PiH3AlphaDecode": "pi-H3 Alpha Decode",
    "PiH3AlphaRefineSigmas": "pi-H3 Alpha Refine Sigmas",
    "SaveRGBAVideo": "Save RGBA Video",
    "PiH3AlphaSolidify": "pi-H3 Alpha Solidify",
}
