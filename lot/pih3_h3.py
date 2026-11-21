#!/usr/bin/env python3
"""pi-H3 as a generator in our pipeline, next to DiR: the official 2/4-step MiniMax H3 with DWD.

pi-H3 is by Hansheng Chen, Shengqu Cai, Jiaming Song, Gordon Wetzstein, Xinchen Yan and Leonidas Guibas
(https://hanshengchen.com/blog/pi-h3/, weights https://huggingface.co/Lakonik/pi-H3, code
https://github.com/Lakonik/LakonLab and https://github.com/Lakonik/ComfyUI-piFlow, Apache-2.0). This module runs
their int8 ConvRot DWD adapter on our musubi-tuner H3 (the same pruned int8 checkpoint as DiR), so our two-track
sampler and our benchmarks can use it. It is not affiliated with the authors.

What the adapter changes (``apply_adapter``; every tensor is consumed or the load fails):
- a rank-64 LoRA (alpha = rank) on ``out_proj``, ``fc1``, ``fc2`` of blocks 0-45, merged into the int8 weights;
- full int8 ``out_proj``, ``fc1``, ``fc2`` of blocks 46-49, new AdaLN weights in all 50 blocks, a new final layer;
- ``dwd_norm_out`` and ``dwd_pixel_refiner`` (8 Sol-H3-VAE decoder blocks): the DWD head (``PiH3Head``).

Sampling (``pih3_sample``) is ComfyUI's pi-H3 workflow: Euler on the pi-Flow schedule (``pih3_sigmas``: raw times
``linspace(max_raw_t, ...)`` with ``max_raw_t`` 0.93 / 0.96 for 2 / 4 steps, final step 0.6 of the others, shift
12), starting from ``sigma_0 * noise``, with the audio latent denoised jointly the way ComfyUI carries it (on the
video schedule, scaled by ``sigma_v / sigma_a``). At the last step the block stack's output on the video rows goes
through the DWD head instead of the VAE.

Merges (``--merge``): ``exact`` dequantizes, adds the LoRA in fp32 and requantizes with musubi's deterministic
ConvRot rounding. ``comfy`` reproduces ComfyUI's patching (fp16 merge, comfy-kitchen requantize with the per-key
stochastic-rounding seed), to tell merge rounding apart from real differences in a parity run.

The head is ComfyUI-piFlow's ``SolH3DWDHead`` on ComfyUI's ``NonCausalViT3DDecoder`` (imported from a ComfyUI >=
0.45 tree, ``LOT_PIH3_COMFY``); LakonLab's own head needs mmcv and its CUDA ops.

    python3 pih3_h3.py parity --ref REF/768x512_f39_s2 --steps 2           # against pih3_comfy_reference.py dumps
    python3 pih3_h3.py sample --steps 2 --sizes 768x512 --prompts 3 --out OUT
    python3 accel_h3.py bench --generator pih3 --steps 2 --sizes 768x512 ...   # the benchmark rows (bench_accel.sh)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from h3_musubi import AUDIO_SHIFT, VIDEO_SHIFT, audio_frames, audio_sigma, shift_sigma  # noqa: E402

ADAPTERS = Path(os.environ.get("LOT_PIH3_ADAPTERS", "/media/2TB/models/pi_h3"))
COMFY = Path(os.environ.get("LOT_PIH3_COMFY", "/media/2TB/ComfyUI-pih3"))
PIFLOW = COMFY / "custom_nodes" / "ComfyUI-piFlow"
MAX_RAW_T = {2: 0.93, 4: 0.96}
FINAL_STEP_SIZE_SCALE = 0.6
AUDIO_SCALE = VIDEO_SHIFT / AUDIO_SHIFT           # ComfyUI's ModelSamplingAV.audio_scale
AUDIO_CHANNELS = 32
MERGES = ("exact", "comfy")


def adapter_path(steps: int) -> Path:
    return ADAPTERS / f"pi_h3_{steps}step_dwd_adapter_int8_convrot.safetensors"


def pih3_sigmas(steps: int, max_raw_t: float | None = None, final_step_size_scale: float = FINAL_STEP_SIZE_SCALE,
                shift: float = VIDEO_SHIFT) -> list[float]:
    """``steps + 1`` video sigmas, ComfyUI-piFlow's ``get_piflow_schedule`` (min_raw_t 0) on H3's shift, in fp32."""
    top = MAX_RAW_T[steps] if max_raw_t is None else max_raw_t
    raw = torch.linspace(top, top * final_step_size_scale / (steps - 1 + final_step_size_scale), steps)
    sigmas = [float(shift_sigma(t, shift)) for t in raw.tolist()]
    return torch.tensor(sigmas + [0.0], dtype=torch.float32).tolist()


# ------------------------------------------------------------------------------------------- ComfyUI pieces

_COMFY: dict = {}


def comfy_lib() -> dict:
    """ComfyUI (>= 0.45, ``COMFY``) as a library, and ComfyUI-piFlow's DWD head modules. Imported only when pi-H3
    is used. Pixels stay on the GPU (``intermediate_device``), as our benchmarks leave them."""
    if _COMFY:
        return _COMFY
    for path in (str(COMFY), str(COMFY / ".pyoverlay")):
        if path not in sys.path:
            sys.path.insert(1, path)
    import comfy.model_management as mm
    import comfy.ops

    mm.intermediate_device = lambda: torch.device("cuda")

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    _COMFY.update(ops=comfy.ops, sol=load("pih3_piflow_sol_h3", PIFLOW / "modules/architectures/sol_h3.py"),
                  loader=load("pih3_piflow_loader_utils", PIFLOW / "modules/loader_utils.py"))
    return _COMFY


class PiH3Head:
    """The DWD head of a pi-H3 adapter: ``dwd_norm_out`` (affine-free LayerNorm modulated by
    ``linear(silu(t_emb))``, scale first) and the Sol refiner, as ComfyUI-piFlow's ``MiniMaxH3DWD`` runs them."""

    def __init__(self, dwd_sd: dict, eps: float, device="cuda"):
        lib = comfy_lib()
        self.eps = eps
        self.norm_w = dwd_sd["dwd_norm_out.linear.weight"].to(device)
        self.norm_b = dwd_sd["dwd_norm_out.linear.bias"].to(device)
        refiner_sd = {k[len("dwd_pixel_refiner."):]: v for k, v in dwd_sd.items() if k.startswith("dwd_pixel_refiner.")}
        if set(dwd_sd) - {"dwd_norm_out.linear.weight", "dwd_norm_out.linear.bias"} - {
                "dwd_pixel_refiner." + k for k in refiner_sd}:
            raise SystemExit("pi-H3 adapter: DWD tensors this head does not know (a linear head?)")
        hidden = self.norm_w.shape[0] // 2
        self.refiner = lib["sol"].SolH3DWDHead.from_state_dict(
            refiner_sd, hidden, (32, 32), dtype=self.norm_w.dtype, device=device,
            operations=lib["ops"].mixed_precision_ops())
        lib["loader"].load_adapter_module(self.refiner, refiner_sd)
        for module in self.refiner.modules():              # weights cast to the activations (bf16), as in the reference
            if hasattr(module, "comfy_cast_weights"):
                module.comfy_cast_weights = True
        self.refiner.to(device).eval().requires_grad_(False)

    def to(self, device) -> "PiH3Head":
        self.norm_w, self.norm_b = self.norm_w.to(device), self.norm_b.to(device)
        self.refiner.to(device)
        return self

    @torch.no_grad()
    def features(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """``(N, hidden)`` block output and ``(temb_dim,)`` video timestep embedding -> the head's input features."""
        x = h.to(torch.bfloat16)
        mod = F.linear(F.silu(t_emb.to(torch.bfloat16)), self.norm_w.to(torch.bfloat16), self.norm_b.to(torch.bfloat16))
        scale, shift = mod.chunk(2, dim=-1)
        return F.layer_norm(x, x.shape[-1:], eps=self.eps).mul_(1 + scale).add_(shift)

    @torch.no_grad()
    def decode(self, h: torch.Tensor, t_emb: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """``(1, 3, F, 32 h, 32 w)`` pixels in [-1, 1] of a ``(t, h, w)`` DiT token grid (raster order rows)."""
        feats = self.features(h, t_emb)[None]
        pixels = self.refiner.refine(*self.refiner.expand(feats, grid))       # (1, F, 3, H, W)
        return pixels.transpose(1, 2)


# ------------------------------------------------------------------------------------------- the adapter

def _kitchen_quantize(weight: torch.Tensor, group: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    comfy_lib()
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    qdata, params = TensorWiseINT8Layout.quantize(weight, per_channel=True, convrot=True, convrot_groupsize=group,
                                                  stochastic_rounding=seed)
    return qdata, params.scale.float().reshape(-1, 1)


@torch.no_grad()
def apply_adapter(dit, path: Path, merge: str = "comfy") -> tuple[dict, dict]:
    """Merge a pi-H3 int8 ConvRot adapter into ``dit`` in place. Returns ``(dwd tensors, counts)``."""
    from musubi_tuner.modules.convrot_int8_kernels import dequantize_int8_convrot_weight, quantize_int8_convrot_weight
    from safetensors.torch import load_file

    if merge not in MERGES:
        raise ValueError(f"merge {merge!r} is not one of {MERGES}")
    if merge == "comfy":
        comfy_lib()
    sd = load_file(str(path))
    dwd = {k: sd.pop(k) for k in list(sd) if k.startswith("dwd_")}
    modules = dict(dit.named_modules())
    tensors = dict(dit.named_parameters())
    tensors.update(dict(dit.named_buffers()))
    used: set[str] = set()

    def requantize(module, layer: str, weight: torch.Tensor) -> None:
        group = module._convrot_groupsize
        if merge == "comfy":
            from comfy.utils import string_to_seed

            q, s = _kitchen_quantize(weight.half(), group, string_to_seed(f"diffusion_model.{layer}.weight"))
        else:
            q, s = quantize_int8_convrot_weight(weight.float(), group)
        module.weight.copy_(q)
        module.scale_weight.copy_(s.view_as(module.scale_weight))

    replaced = sorted({k[:-len(".comfy_quant")] for k in sd if k.endswith(".comfy_quant")})
    for layer in replaced:
        spec = json.loads(bytes(sd[layer + ".comfy_quant"].tolist()).decode())
        module = modules[layer]
        group = getattr(module, "_convrot_groupsize", None)
        if module.weight.dtype != torch.int8 or spec.get("format") != "int8_tensorwise" or not spec.get("convrot") \
                or spec.get("convrot_groupsize") != group:
            raise SystemExit(f"{layer}: adapter int8 spec {spec} does not match the DiT's (group {group})")
        q = sd[layer + ".weight"].to(module.weight.device)
        s = sd[layer + ".weight_scale"].to(module.scale_weight.device, torch.float32)
        if merge == "comfy":                 # ComfyUI dequantizes the adapter to bf16, then requantizes it (fp16)
            requantize(module, layer, dequantize_int8_convrot_weight(q, s, group).bfloat16())
        else:
            module.weight.copy_(q)
            module.scale_weight.copy_(s.view_as(module.scale_weight))
        used |= {layer + ".weight", layer + ".weight_scale", layer + ".comfy_quant"}

    loras = sorted({k.rsplit(".", 2)[0] for k in sd if k.endswith((".lora_A.weight", ".lora_B.weight"))})
    for layer in loras:
        module = modules[layer]
        a, b = sd[layer + ".lora_A.weight"], sd[layer + ".lora_B.weight"]
        device = module.weight.device
        delta = b.to(device, torch.float32) @ a.to(device, torch.float32)        # alpha = rank: scale 1
        if module.weight.dtype == torch.int8:
            base = dequantize_int8_convrot_weight(module.weight, module.scale_weight, module._convrot_groupsize)
            if merge == "comfy":                             # LoRA patched onto an fp16 dequant, as ComfyUI does
                weight = base.half() + delta.half()
            else:
                weight = base + delta
            requantize(module, layer, weight)
        else:
            module.weight.copy_((module.weight.float() + delta).to(module.weight.dtype))
        used |= {layer + ".lora_A.weight", layer + ".lora_B.weight"}

    floats = 0
    for key, value in sd.items():
        if key in used:
            continue
        target = tensors.get(key)
        if target is None or tuple(target.shape) != tuple(value.shape):
            raise SystemExit(f"adapter tensor {key} {tuple(value.shape)} has no place in the DiT")
        target.copy_(value.to(target.device, target.dtype))
        used.add(key)
        floats += 1
    if set(sd) - used:
        raise SystemExit(f"adapter tensors not consumed: {sorted(set(sd) - used)[:5]}")
    return dwd, {"lora": len(loras), "int8": len(replaced), "float": floats, "dwd": len(dwd), "merge": merge}


def load_pih3(steps: int, *, merge: str = "comfy", adapter: Path | None = None, device="cuda",
              base: bool = False) -> dict:
    """The pi-H3 generator: our int8 H3 with the adapter merged, its DWD head and its schedule.
    ``base=True`` (diagnostics) leaves the DiT as it is and loads no head."""
    from h3_musubi import load_dit

    dit = load_dit(device=device, swap=0)
    if base:
        return {"dit": dit, "head": None, "steps": steps, "sigmas": pih3_sigmas(steps), "adapter": None, "merge": None}
    path = Path(adapter) if adapter else adapter_path(steps)
    dwd, counts = apply_adapter(dit, path, merge)
    head = PiH3Head(dwd, dit.final_layer.norm.eps, device=device)
    torch.cuda.empty_cache()
    print(f"LOT_H3 kind=pih3_load adapter={path.name} steps={steps} {' '.join(f'{k}={v}' for k, v in counts.items())}",
          flush=True)
    return {"dit": dit, "head": head, "steps": steps, "sigmas": pih3_sigmas(steps), "adapter": str(path),
            "merge": merge}


# ------------------------------------------------------------------------------------------- sampling

def audio_noise(latent_t: int, seed: int, device="cuda") -> torch.Tensor:
    return torch.randn(1, AUDIO_CHANNELS, 2, audio_frames(latent_t), device=device,
                       generator=torch.Generator(device).manual_seed(int(seed)))


def carried_audio_step(y: torch.Tensor, v_audio: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    """One Euler step of the audio ComfyUI carries as ``y = (sigma_v / sigma_a) x_audio`` on the video schedule
    (``MiniMaxH3Model.forward`` in ComfyUI >= 0.45). ``v_audio`` is our ``x0 - eps`` audio velocity."""
    sigma_a = float(audio_sigma(sigma))
    x_a = y * (sigma_a / sigma)
    d = (1.0 - AUDIO_SCALE) * x_a - (1.0 + (AUDIO_SCALE - 1.0) * sigma_a) * v_audio
    return y + d * (sigma_next - sigma)


@torch.no_grad()
def pih3_sample(model: dict, text: torch.Tensor, z: torch.Tensor, za: torch.Tensor, *,
                sigmas: list | None = None) -> dict:
    """pi-H3 from unit noise ``z`` ``(1, 24, T, H, W)`` and audio noise ``za`` ``(1, 32, 2, A)``.

    Returns ``latent`` (the last Euler step: the clean video estimate), ``audio`` (the audio latent), ``h`` and
    ``temb`` (the last forward's block output on the video rows and its timestep embedding, the DWD head's input)."""
    from h3_musubi import forward_video

    dit, sigmas = model["dit"], sigmas or model["sigmas"]
    x, y = sigmas[0] * z.float(), sigmas[0] * za.float()
    out: dict = {}
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        sigma_a = float(audio_sigma(sigma))
        store: dict = {}
        v = forward_video(dit, x, 1.0 - sigma, text, audio=y * (sigma_a / sigma), t_audio=1.0 - sigma_a,
                          out=store).float()
        if i == len(sigmas) - 2:
            out["h"], out["temb"] = store["hidden"], store["temb_video"]
        y = carried_audio_step(y, store["audio"].float(), sigma, sigma_next)
        x = x + v * (sigma - sigma_next)
    out.update(latent=x, audio=y / AUDIO_SCALE)
    return out


def decode(model: dict, out: dict, latent_shape: tuple) -> torch.Tensor:
    """The DWD pixels ``(1, 3, F, H, W)`` of a ``pih3_sample`` result for a ``(1, 24, T, H/16, W/16)`` latent."""
    _b, _c, latent_t, lat_h, lat_w = latent_shape
    return model["head"].decode(out["h"], out["temb"], (latent_t, lat_h // 2, lat_w // 2))


# ------------------------------------------------------------------------------------------- two-track

@torch.no_grad()
def pih3_two_track(model: dict, text: torch.Tensor, *, height: int, width: int, latent_t: int, seed: int,
                   two_track: dict, switch: int = 1, alpha: float = 0.5, floor=None, box=None,
                   timer: dict | None = None) -> dict:
    """MPEG-Diffusion's two-track sampler on pi-H3, DWD on both tracks.

    Steps ``0 .. switch - 1`` denoise the whole clip at half resolution; then the region head (on the LR track's
    block states, MPEG-Diffusion's) picks a region, whose bounding box over the whole clip becomes the HR track:
    the clean LR estimate x2 (latent, trilinear), re-noised with fresh noise to the next sigma. The remaining steps
    run one packed sequence (dual-resolution attention with the K/V maps). Audio is denoised jointly throughout.

    The HR region is a box so the Sol refiner, which needs a full token grid, decodes it from real HR tokens.
    Pixels: the LR track's DWD pixels x2 (bicubic), with the box's DWD pixels pasted in (``box_paste``).
    Returns ``pixels``, ``lr_pixels`` (half resolution), ``latent`` (LR x2 with the region's HR latent inside),
    ``lr`` (LR latent), ``box`` (LR token rows/cols) and ``share`` (HR tokens / full-resolution tokens)."""
    from mixed_res import TwoTrackPlan, region_mask
    from two_track_h3 import two_track_forward

    dit, head, sigmas = model["dit"], model["head"], model["sigmas"]
    steps = len(sigmas) - 1
    if not 1 <= switch < steps:
        raise ValueError(f"switch {switch} must be in 1..{steps - 1}")
    lat_h, lat_w = height // 32, width // 32               # LR latent = HR token grid
    tok_h, tok_w = lat_h // 2, lat_w // 2                  # LR token grid
    timer = timer if timer is not None else {}
    timer.setdefault("dit_s", 0.0)
    z_lr = torch.randn(1, 24, latent_t, lat_h, lat_w, device="cuda", generator=torch.Generator("cuda").manual_seed(seed))
    za = audio_noise(latent_t, seed + 1)
    x_lr, y = sigmas[0] * z_lr, sigmas[0] * za
    est_lr = states = None
    info: dict = {}

    def forward(x_lr, x_hr, plan, i, capture=None):
        sigma = sigmas[i]
        sigma_a = float(audio_sigma(sigma))
        store: dict = {}
        torch.cuda.synchronize()
        start = time.perf_counter()
        v_lr, v_hr = two_track_forward(dit, x_lr, x_hr, plan, 1.0 - sigma, text, maps=two_track["maps"],
                                       capture=capture, audio=y * (sigma_a / sigma), t_audio=1.0 - sigma_a, out=store)
        torch.cuda.synchronize()
        timer["dit_s"] += time.perf_counter() - start
        return v_lr, v_hr, store

    for i in range(switch):
        capture = {"block": two_track["block"] or 25} if i == switch - 1 else None
        v_lr, _, store = forward(x_lr, None, None, i, capture)
        if capture is not None:
            states = capture["states"]
        est_lr = x_lr + sigmas[i] * v_lr.float()
        y = carried_audio_step(y, store["audio"].float(), sigmas[i], sigmas[i + 1])
        x_lr = x_lr + v_lr.float() * (sigmas[i] - sigmas[i + 1])

    if box is None:
        mask = region_mask(two_track["region"](states).float().cpu().reshape(latent_t, tok_h, tok_w), alpha=alpha,
                           floor=floor)
        ys, xs = torch.nonzero(mask.any(0), as_tuple=True)
        if len(ys) == 0:
            ys, xs = torch.tensor([tok_h // 4, (3 * tok_h) // 4 - 1]), torch.tensor([tok_w // 4, (3 * tok_w) // 4 - 1])
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    else:
        y0, y1, x0, x1 = (round(box[1] * tok_h), round(box[3] * tok_h), round(box[0] * tok_w), round(box[2] * tok_w))
    rect = torch.zeros(latent_t, tok_h, tok_w, dtype=torch.bool)
    rect[:, y0:y1, x0:x1] = True
    plan = TwoTrackPlan(rect).to("cuda")
    est_hr = F.interpolate(est_lr, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
    z_hr = torch.randn(1, 24, latent_t, 2 * lat_h, 2 * lat_w, device="cuda",
                       generator=torch.Generator("cuda").manual_seed(seed + 2))
    x_hr = sigmas[switch] * z_hr + (1.0 - sigmas[switch]) * est_hr
    for i in range(switch, steps):
        v_lr, v_hr, store = forward(x_lr, x_hr, plan, i)
        if i == steps - 1:
            info.update(h=store["hidden"], temb=store["temb_video"])
        y = carried_audio_step(y, store["audio"].float(), sigmas[i], sigmas[i + 1])
        x_lr = x_lr + v_lr.float() * (sigmas[i] - sigmas[i + 1])
        x_hr = x_hr + v_hr.float() * (sigmas[i] - sigmas[i + 1])

    torch.cuda.synchronize()
    start = time.perf_counter()
    lr_count = latent_t * tok_h * tok_w
    lr_px = head.decode(info["h"][:lr_count], info["temb"], (latent_t, tok_h, tok_w))
    box_px = head.decode(info["h"][lr_count:], info["temb"], (latent_t, 2 * (y1 - y0), 2 * (x1 - x0)))
    pixels = box_paste(lr_px, box_px, (y0, y1, x0, x1))
    torch.cuda.synchronize()
    timer["decode_s"] = time.perf_counter() - start
    inside = torch.zeros_like(x_hr[:, :1])
    inside[..., 4 * y0:4 * y1, 4 * x0:4 * x1] = 1.0
    latent = inside * x_hr + (1.0 - inside) * F.interpolate(x_lr, scale_factor=(1, 2, 2), mode="trilinear",
                                                            align_corners=False)
    return {"pixels": pixels, "lr_pixels": lr_px, "box_pixels": box_px, "latent": latent, "lr": x_lr,
            "box": (y0, y1, x0, x1), "share": plan.hr_count / (4 * lr_count), "audio": y / AUDIO_SCALE}


def box_paste(lr_px: torch.Tensor, box_px: torch.Tensor, box: tuple, feather: int = 16) -> torch.Tensor:
    """The LR pixels x2 (bicubic) with the box's pixels pasted over LR token rows/cols ``box``, the paste weight
    ramping over ``feather`` pixels inside the box's edges (not at the frame's)."""
    from accel_h3 import upscale_px

    up = upscale_px(lr_px)
    y0, y1, x0, x1 = (64 * v for v in box)                 # an LR token is 64 x 64 pixels at full resolution
    height, width = up.shape[-2:]
    wy = torch.ones(y1 - y0, device=up.device)
    wx = torch.ones(x1 - x0, device=up.device)
    ramp = (torch.arange(feather, device=up.device, dtype=torch.float32) + 0.5) / feather
    if y0 > 0:
        wy[:feather] = ramp
    if y1 < height:
        wy[-feather:] = torch.minimum(wy[-feather:], ramp.flip(0))
    if x0 > 0:
        wx[:feather] = ramp
    if x1 < width:
        wx[-feather:] = torch.minimum(wx[-feather:], ramp.flip(0))
    w = wy[:, None] * wx[None, :]
    out = up.clone()
    out[..., y0:y1, x0:x1] = w * box_px.float() + (1.0 - w) * up[..., y0:y1, x0:x1]
    return out


# ------------------------------------------------------------------------------------------- parity

def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


@torch.no_grad()
def parity(args) -> None:
    """Our pi-H3 against ``pih3_comfy_reference.py``'s dumps, in layers: the head on ComfyUI's own block output,
    one forward per sampler call on ComfyUI's inputs, then the whole clip from ComfyUI's noise."""
    import lpips

    from accel_h3 import lpips_clip
    from dwd_h3 import psnr
    from gpu_guard import refuse_if_busy
    from h3_musubi import forward_video
    from lot_paths import LOT_H3_STILLS_CACHE, LOT_H3_CLIPS_CACHE
    from train_dwd_h3 import load_text

    refuse_if_busy("pih3_h3.py")
    model = load_pih3(args.steps, merge=args.merge, base=args.base)
    net = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    caches = [LOT_H3_STILLS_CACHE, LOT_H3_CLIPS_CACHE]
    rows = []
    for dump_path in sorted(Path(args.ref).glob("p*.pt"))[:args.prompts]:
        ref = torch.load(dump_path, weights_only=False)
        te = next(c / ref["prompt"] for c in caches if (c / ref["prompt"]).is_file())
        text = load_text(str(te)).cuda()
        if [round(s, 5) for s in ref["sigmas"].tolist()] != [round(s, 5) for s in model["sigmas"]]:
            raise SystemExit(f"sigmas differ: ComfyUI {ref['sigmas'].tolist()} ours {model['sigmas']}")
        target = ref["pixels"].permute(3, 0, 1, 2).unsqueeze(0).cuda().mul(2).sub(1)        # (1, 3, F, H, W)
        latent_shape = tuple(ref["noise_video"].shape)
        _b, _c, latent_t, lat_h, lat_w = latent_shape
        row = {"prompt": ref["prompt"]}

        # 1. the head alone, on ComfyUI's block output and timestep embedding
        video_seg = next((a, b) for a, b, kind in ref["layout_segments"] if kind == "video")
        video_row = next(r for a, b, r in ref["mod_segments_last"] if (a, b) == video_seg) // 3
        if not args.base:
            px = model["head"].decode(ref["h_last"][video_seg[0]:video_seg[1]].cuda(),
                                      ref["t_emb_last"][video_row].cuda(), (latent_t, lat_h // 2, lat_w // 2)).clamp(-1, 1)
            row["head_psnr"] = round(psnr(px, target), 2)
            row["head_maxabs"] = round(float((px - target).abs().max()), 4)

        # 2. one forward per sampler call, on ComfyUI's inputs (video, audio at its own level, timestep)
        call_rel = []
        for k, call in enumerate(ref["calls"]):
            sigma = float(call["timestep"].flatten()[0]) / 1000.0
            sigma_a = float(audio_sigma(sigma))
            store: dict = {}
            v = forward_video(model["dit"], call["video"].cuda(), 1.0 - sigma, text, audio=call["audio"].cuda(),
                              t_audio=1.0 - sigma_a, out=store).float()
            entry = {"video": round(rel(-v, call["out_video"].cuda()), 5),
                     "audio": round(rel(-store["audio"].float(), call["out_audio"].cuda()), 5)}
            if k == len(ref["calls"]) - 1:
                entry["h"] = round(rel(store["hidden"], ref["h_last"][video_seg[0]:video_seg[1]].cuda()), 5)
                entry["temb"] = round(rel(store["temb_video"], ref["t_emb_last"][video_row].cuda()), 5)
            call_rel.append(entry)
        row["calls"] = call_rel

        if args.base:
            print("LOT_H3 kind=pih3_parity " + json.dumps(row), flush=True)
            rows.append(row)
            continue
        # 3. end to end from ComfyUI's noise, and the floor: the same with the noise rounded to bf16
        z, za = ref["noise_video"].cuda(), ref["noise_audio"].cuda()
        first = ref["calls"][0]["video"].cuda() if ref["calls"] else None
        if first is not None:
            row["start_rel"] = round(rel(model["sigmas"][0] * z, first), 6)
        out = pih3_sample(model, text, z, za)
        ours = decode(model, out, latent_shape).clamp(-1, 1)
        row["e2e_psnr"] = round(psnr(ours, target), 2)
        row["e2e_lpips"] = round(lpips_clip(net, ours, target), 4)
        row["latent_rel"] = round(rel(out["latent"], ref["latent"].cuda()), 4)
        floor = pih3_sample(model, text, z.bfloat16().float(), za.bfloat16().float())
        floor_px = decode(model, floor, latent_shape).clamp(-1, 1)
        row["floor_psnr"] = round(psnr(floor_px, ours), 2)
        row["floor_lpips"] = round(lpips_clip(net, floor_px, ours), 4)
        print("LOT_H3 kind=pih3_parity " + json.dumps(row), flush=True)
        rows.append(row)
        if args.out:
            from two_track_h3 import save_clip

            args.out.mkdir(parents=True, exist_ok=True)
            save_clip(ours.half().cpu(), args.out / f"{dump_path.stem}_ours_{args.merge}.mp4")
    if args.out:
        (args.out / f"parity_{Path(args.ref).name}_{args.merge}.json").write_text(json.dumps(rows, indent=1) + "\n")
    ok = all(r.get("head_psnr", 99) > 40 for r in rows)
    print(f"LOT_H3 kind=pih3_parity_done ok={int(ok)}", flush=True)


# ------------------------------------------------------------------------------------------- sample

@torch.no_grad()
def sample(args) -> None:
    """Render clips: ``OUT/<W>x<H>_f<F>_s<steps>/p<i>.mp4`` (and two-track ``p<i>_tt.mp4`` with ``--two-track``)."""
    from accel_h3 import default_caches, parse_job, prompt_lists, token_shape
    from gpu_guard import refuse_if_busy
    from setproctitle import setproctitle
    from train_dwd_h3 import load_text
    from two_track_h3 import save_clip

    setproctitle(f"pih3:sample s{args.steps} {' '.join(args.sizes)}")
    refuse_if_busy("pih3_h3.py")
    model = load_pih3(args.steps, merge=args.merge)
    two_track = None
    if args.two_track:
        from accel_h3 import load_two_track

        two_track = load_two_track(args)
    _train, held = prompt_lists(default_caches(), args.holdout)
    for spec in args.sizes:
        width, height, _count, frames = parse_job(spec, args.frames)
        latent_t, _th, _tw = token_shape(width, height, frames)
        folder = args.out / f"{width}x{height}_f{frames}_s{args.steps}"
        folder.mkdir(parents=True, exist_ok=True)
        for index, te in enumerate(held[:args.prompts]):
            text = load_text(str(te)).cuda()
            seed = args.seed + index
            z = torch.randn(1, 24, latent_t, height // 16, width // 16, device="cuda",
                            generator=torch.Generator("cuda").manual_seed(seed))
            out = pih3_sample(model, text, z, audio_noise(latent_t, seed + 1))
            save_clip(decode(model, out, tuple(z.shape)).half().cpu(), folder / f"p{index}.mp4")
            if two_track is not None:
                tt = pih3_two_track(model, text, height=height, width=width, latent_t=latent_t, seed=seed,
                                    two_track=two_track, switch=args.switch)
                save_clip(tt["pixels"].half().cpu(), folder / f"p{index}_tt.mp4")
            print(f"LOT_H3 kind=pih3_sample size={width}x{height} frames={frames} prompt={index}", flush=True)


# ------------------------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("parity", "sample"):
        q = sub.add_parser(name)
        q.add_argument("--steps", type=int, default=2, choices=sorted(MAX_RAW_T))
        q.add_argument("--merge", default="comfy", choices=MERGES)
        q.add_argument("--prompts", type=int, default=3)
        q.add_argument("--out", type=Path, default=None)
    q = sub.choices["parity"]
    q.add_argument("--ref", type=Path, required=True, help="a pih3_comfy_reference.py folder (p*.pt)")
    q.add_argument("--base", action="store_true", help="diagnostic: no adapter, against a --no-adapter reference")
    q = sub.choices["sample"]
    q.add_argument("--sizes", nargs="+", default=["768x512"])
    q.add_argument("--frames", type=int, default=39)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--holdout", type=int, default=8)
    q.add_argument("--two-track", action="store_true")
    q.add_argument("--switch", type=int, default=1)
    q.add_argument("--maps", type=Path, default=None)
    q.add_argument("--region", type=Path, default=None)
    args = p.parse_args()
    from setproctitle import setproctitle

    setproctitle(f"pih3:{args.cmd} s{args.steps} {args.merge}")
    {"parity": parity, "sample": sample}[args.cmd](args)


if __name__ == "__main__":
    main()
