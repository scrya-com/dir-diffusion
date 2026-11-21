# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""DiR (and DWD) inside ComfyUI's own MiniMax H3 forward. Stock ComfyUI files are not edited.

How the ``lot/h3_musubi._lot_forward`` hooks map onto ``comfy.ldm.minimax.model.MiniMaxH3Model._forward``:

* A ``DIFFUSION_MODEL`` wrapper (key ``dir_h3``) runs only while a DiR sampler is active. Around each call it adds a
  ``patches_replace["dit"][("double_block", i)]`` entry for every block (chaining any patch already there) and a
  forward pre-hook / hook on ``final_layer``, and removes the hooks afterwards.
* ``video_extra`` (``DirHead.x0_proj`` of the previous estimate): the block-0 patch rewrites the target-video rows
  as ``video_patch_proj(rows(state fp32)).to(dtype) + extra.to(dtype)``, exactly ``_lot_forward``'s order. Comfy
  casts the sampler latent to the model dtype before the forward; re-embedding from the fp32 state keeps
  ``dir_h3``'s precision.
* Silent audio: the same block-0 patch rewrites the target-audio rows from ``sigma_a * eps`` (one ``eps`` per run,
  ``Generator(seed + 1)``, as ``h3_musubi._silence``), with ``sigma_a`` on H3's 12 / 3 shifts.
* ``video_temb_delta`` (``DirHead.r_embed[r]``): every block patch appends the row ``t_emb[video] + delta`` to the
  timestep table and points the target-video modulation segment at it (``3 * new + 0``); text and audio keep their
  rows (``h3_musubi.video_modulation_rewrite``). The final-layer pre-hook does the same for the video head.
* The final-layer forward hook keeps the fp32 velocity (Comfy rounds the returned velocity to the latent dtype),
  so the clean estimate is ``state + sigma * out`` in fp32, as ``dir_h3.generator_step``.
* DWD: at the final iteration the patch of the first DWD block saves its input; after the forward returns, blocks
  ``K..`` run again from it with the DWD LoRA on (forward hooks, ``RuntimeLora``), and the modulated last-layer
  tokens are what ``final_layer.video_out`` would read. Blocks before ``K`` carry no DWD LoRA, so this equals a
  full final iteration with the LoRA on, at ``(50 - K) / 50`` of the cost; the base pass gives the LATENT.

Tensors that leave the forward are allocated under ``pause_malloc_graph`` (ComfyUI's allocation graph recycles
memory allocated inside a forward).
"""

from __future__ import annotations

import contextlib
import logging

import torch
import torch.nn.functional as F

from .vendor.dir_core import AUDIO_SHIFT, VIDEO_SHIFT, audio_sigma, dir_loop, estimate_extra

log = logging.getLogger(__name__)

WRAPPER_KEY = "dir_h3"


def _pause_malloc_graph():
    try:
        import comfy.model_prefetch

        return comfy.model_prefetch.pause_malloc_graph()
    except (ImportError, AttributeError):
        return contextlib.nullcontext()


# ------------------------------------------------------------------------------------------ runtime LoRA

class RuntimeLora:
    """A musubi-format LoRA applied with forward hooks: ``out + up(down(x)) * multiplier * alpha / rank``.

    Same arithmetic as musubi-tuner's ``LoRAModule.forward``: the input is cast to the LoRA's dtype, the delta is
    added, and the sum is cast back to the layer's output dtype. Only active inside ``active()``. ``fc2`` is
    called through ``comfy.ops.linear_input_act``'s fused path, which bypasses the module call, so
    ``comfy_force_forward`` is set while the hooks are on (as ComfyUI's own bypass LoRA does).
    """

    def __init__(self, state: dict, dtype: torch.dtype = torch.float32, multiplier: float = 1.0):
        self.dtype, self.multiplier = dtype, float(multiplier)
        self.modules: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
        for key, down in state.items():
            if not key.endswith(".lora_down.weight"):
                continue
            name = key[:-len(".lora_down.weight")]
            up = state[name + ".lora_up.weight"]
            rank = down.shape[0]
            alpha = float(state[name + ".alpha"]) if name + ".alpha" in state else float(rank)
            self.modules[name] = (down.to(dtype), up.to(dtype), alpha / rank)
        if not self.modules:
            raise ValueError("no LoRA modules (keys ending in .lora_down.weight) in this file")
        self._on_device: dict = {}

    def block_indices(self) -> list[int]:
        out = set()
        for name in self.modules:
            parts = name.split("_")
            if "blocks" in parts:
                out.add(int(parts[parts.index("blocks") + 1]))
        return sorted(out)

    def targets(self, diffusion_model) -> dict:
        lookup = {"lora_unet_" + name.replace(".", "_"): module for name, module in diffusion_model.named_modules()}
        missing = [name for name in self.modules if name not in lookup]
        if missing:
            raise ValueError(f"{len(missing)} LoRA modules have no layer in this model, e.g. {missing[:3]}")
        return {name: lookup[name] for name in self.modules}

    def _weights(self, name: str, device):
        key = (name, str(device))
        if key not in self._on_device:
            with _pause_malloc_graph():
                down, up, scale = self.modules[name]
                self._on_device[key] = (down.to(device), up.to(device), scale)
        return self._on_device[key]

    def _hook(self, name: str):
        def hook(_module, args, output):
            down, up, scale = self._weights(name, output.device)
            lx = F.linear(F.linear(args[0].to(self.dtype), down), up)
            res = output + lx * (self.multiplier * scale)
            return res if res.dtype == output.dtype else res.to(output.dtype)
        return hook

    @contextlib.contextmanager
    def active(self, diffusion_model):
        handles, forced = [], []
        try:
            for name, module in self.targets(diffusion_model).items():
                handles.append(module.register_forward_hook(self._hook(name)))
                forced.append((module, getattr(module, "comfy_force_forward", False)))
                module.comfy_force_forward = True
            yield
        finally:
            for handle in handles:
                handle.remove()
            for module, previous in forced:
                module.comfy_force_forward = previous


# --------------------------------------------------------------------------------------------- run state

class DirRun:
    """What one DiR sampling run hands the hooks; the sampler function writes it before every model call."""

    def __init__(self, head, eps_audio: torch.Tensor, dwd_lora: RuntimeLora | None = None):
        self.head = head
        self.eps_audio = eps_audio
        self.dwd_lora = dwd_lora
        self.dwd_block = dwd_lora.block_indices()[0] if dwd_lora is not None else None
        self.active = False
        self.model = None
        self.r = self.sigma = None
        self.state = self.extra = self.delta = self.silence = None
        self.capture = False
        self.velocity = None
        self.saved = None
        self.block0_seen = False

    def begin(self, r: int, state: torch.Tensor, estimate: torch.Tensor, sigma: float, capture: bool) -> None:
        self.r, self.sigma, self.capture = r, float(sigma), bool(capture)
        self.state = state.float()
        self.extra = estimate_extra(self.head, estimate)
        self.delta = self.head.r_embed[r].float()
        self.silence = float(audio_sigma(float(sigma))) * self.eps_audio
        self.velocity = self.saved = None
        self.block0_seen = False
        self.active = True

    def end(self) -> torch.Tensor:
        self.active = False
        if not self.block0_seen or self.velocity is None:
            raise RuntimeError("the DiR hooks did not run: another node (a DIFFUSION_MODEL wrapper such as LoT Apply) "
                               "replaced MiniMax H3's forward, or the model is not MiniMax H3")
        return self.velocity

    # The new timestep row: t_emb[video] + r_embed[r] (fp32 add, cast to the table's dtype).
    def extend(self, t_emb: torch.Tensor, row: int) -> tuple[torch.Tensor, int]:
        new = (t_emb[row].float() + self.delta.to(t_emb.device)).to(t_emb.dtype)
        return torch.cat([t_emb, new.unsqueeze(0)], dim=0), t_emb.shape[0]


def _target_spans(layout) -> tuple[tuple[int, int], tuple[int, int]]:
    kinds = {kind for _a, _b, kind in layout.segments}
    if kinds - {"text", "audio", "video"}:
        raise ValueError(f"DiR for H3 is text-to-video only: this conditioning adds {sorted(kinds - {'text', 'audio', 'video'})} "
                         "rows (first/last frame or references). Use a plain prompt")
    video = next((a, b) for a, b, kind in layout.segments if kind == "video")
    audio = next((a, b) for a, b, kind in layout.segments if kind == "audio")
    return video, audio


def _block_patch(run: DirRun, model, index: int, previous):
    from comfy.ldm.minimax.model import pack_audio, patchify_video

    def patch(args, extra):
        img = args["img"]
        (va, vb), (aa, ab) = _target_spans(args["layout"])
        segments = list(args["mod_segments"])
        position = next(i for i, (a, b, _row) in enumerate(segments) if a == va and b == vb)
        row = segments[position][2]
        if not isinstance(row, int) or row % 3:
            raise ValueError("DiR: the target video has per-row timesteps (a denoise mask); not supported")
        t_emb, new = run.extend(args["t_emb"], row // 3)
        segments[position] = (va, vb, 3 * new)
        if index == 0:
            rows = patchify_video(run.state.to(img.device), model.patch_size)
            video = model.video_patch_proj(rows).to(img.dtype)
            if video.shape[0] != vb - va:
                raise RuntimeError(f"DiR: {video.shape[0]} video rows for a {vb - va}-row segment")
            img[va:vb] = video + run.extra.to(img.device, img.dtype)
            audio = model.audio_patch_proj(pack_audio(run.silence.to(img.device, torch.float32)))
            img[aa:ab] = audio.to(img.dtype)
            run.block0_seen = True
        if run.capture and index == run.dwd_block:
            with _pause_malloc_graph():
                run.saved = {"h": img.clone(), "t_emb": t_emb.clone(), "segments": list(segments),
                             "rope": args["rope_freqs"].clone(), "options": dict(args["transformer_options"]),
                             "video": (va, vb), "row": new}
        new_args = dict(args)
        new_args.update(img=img, t_emb=t_emb, mod_segments=segments)
        if previous is not None:
            return previous(new_args, extra)
        return extra["original_block"](new_args)

    return patch


def _final_pre_hook(run: DirRun):
    def hook(module, args):
        h, t_emb, video_seg, *rest = args
        va, vb, row = video_seg
        if not isinstance(row, int):
            raise ValueError("DiR: per-row video timesteps (a denoise mask) are not supported")
        if module.video_out.weight.shape[0] != module.video_dim:
            raise ValueError("DiR: PDD multi-head H3 checkpoints are not supported")
        t_emb, new = run.extend(t_emb, row)
        return (h, t_emb, (va, vb, new), *rest)
    return hook


def _final_hook(run: DirRun):
    def hook(_module, _args, output):
        with _pause_malloc_graph():
            run.velocity = output[0].detach().float().clone()
    return hook


def make_wrapper(run: DirRun):
    def wrapper(executor, x, timestep, context, transformer_options={}, minimax_payload=None,
                denoise_mask=None, audio_denoise_mask=None, **kwargs):
        if not run.active:
            return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload,
                            denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask, **kwargs)
        if denoise_mask is not None or audio_denoise_mask is not None:
            raise ValueError("DiR: denoise masks (inpainting latents) are not supported")
        model = executor.class_obj
        run.model = model
        options = dict(transformer_options)
        # dir_h3 derives the audio timestep on H3's own shifts whatever a SigmaShift node set
        options["minimax_h3_sigma_shift_video"] = VIDEO_SHIFT
        options["minimax_h3_sigma_shift_audio"] = AUDIO_SHIFT
        replace = dict(options.get("patches_replace", {}))
        dit = dict(replace.get("dit", {}))
        for i in range(len(model.blocks)):
            dit[("double_block", i)] = _block_patch(run, model, i, dit.get(("double_block", i)))
        replace["dit"] = dit
        options["patches_replace"] = replace
        handles = [model.final_layer.register_forward_pre_hook(_final_pre_hook(run)),
                   model.final_layer.register_forward_hook(_final_hook(run))]
        try:
            return executor(x, timestep, context, options, minimax_payload=minimax_payload,
                            denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

    return wrapper


@torch.no_grad()
def dwd_tokens(run: DirRun) -> torch.Tensor:
    """Blocks ``K..`` again from the saved input with the DWD LoRA on, then the final layer's video modulation:
    the ``(N, hidden)`` fp32 tokens ``final_layer.video_out`` reads (``dwd_h3.capture_last_layer``)."""
    saved, model = run.saved, run.model
    if saved is None or model is None:
        raise RuntimeError("DWD: the final iteration did not save the block input")
    hidden = saved["h"]
    with run.dwd_lora.active(model):
        for i in range(run.dwd_block, len(model.blocks)):
            hidden = model.blocks[i](hidden, saved["t_emb"], saved["segments"], saved["rope"],
                                     transformer_options=saved["options"])
    final = model.final_layer
    shift, scale = final.adaln_proj(saved["t_emb"])
    va, vb = saved["video"]
    row = saved["row"]
    tokens = final.norm(hidden[va:vb]) * (1.0 + scale[row].to(scale.dtype)) + shift[row].to(shift.dtype)
    run.saved = None
    return tokens.to(torch.float32)


# ------------------------------------------------------------------------------------------------ sampler

def make_sampler_function(run: DirRun, z: torch.Tensor, sigmas: list[float], mix_state: bool, shapes, result: dict,
                          loop=None):
    """A ``comfy.samplers.KSAMPLER`` function running ``dir_h3.dir_sample``'s recursion (``dir_core.dir_loop``, or
    ``loop`` with the same signature: the GPU parity script uses it to feed each iteration fixed inputs)."""
    import comfy.utils
    from comfy.ldm.minimax.model import unpatchify_video

    iters = len(sigmas)
    video_shape, audio_shape = shapes
    _b, _c, latent_t, lat_h, lat_w = video_shape

    def sampler(model, x, sigma_tensor, extra_args=None, callback=None, disable=None):
        extra_args = extra_args or {}
        device = x.device
        run.head.to(device)
        run.eps_audio = run.eps_audio.to(device)
        z_dev = z.to(device)
        zeros_audio = torch.zeros(audio_shape, device=device, dtype=x.dtype)
        s_in = x.new_ones([x.shape[0]])

        def step(r, state, estimate, sigma):
            run.begin(r, state, estimate, sigma, capture=run.dwd_lora is not None and r == iters - 1)
            packed, _ = comfy.utils.pack_latents([state.to(x.dtype), zeros_audio])
            try:
                model(packed, sigma_tensor[r] * s_in, **extra_args)
                velocity = run.end()
            finally:
                run.active = False
            out = unpatchify_video(velocity, latent_t, lat_h // 2, lat_w // 2, 24)
            if run.capture:
                result["dwd_tokens"] = dwd_tokens(run)
            if callback is not None:
                clean, _ = comfy.utils.pack_latents([(state.float() + sigma * out).to(x.dtype), zeros_audio])
                callback({"x": packed, "i": r, "sigma": sigma_tensor[r], "sigma_hat": sigma_tensor[r], "denoised": clean})
            return out

        estimate, estimates, state = (loop or dir_loop)(step, z_dev, iters, sigmas, mix_state)
        result.update(estimate=estimate, estimates=estimates, state=state)
        packed, _ = comfy.utils.pack_latents([estimate.to(x.dtype), zeros_audio])
        return packed

    return sampler


def sample(model_patcher, head, positive, latent: dict, *, seed: int, sigmas: list[float], mix_state: bool,
           noise_device: str = "gpu", dwd_lora: RuntimeLora | None = None, disable_pbar: bool | None = None,
           loop=None) -> dict:
    """Run DiR on ``model_patcher`` (MiniMax H3 with the generator LoRA). Returns ``{"latent": LATENT dict,
    "estimate": (1,24,T,H,W) fp32, "estimates": [...], "state": the final iteration's input state, "z": noise,
    "dwd_tokens": (N, hidden) fp32 when ``dwd_lora`` is given}``."""
    import comfy.model_management
    import comfy.nested_tensor
    import comfy.patcher_extension
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview

    samples = latent["samples"]
    if not getattr(samples, "is_nested", False):
        raise ValueError("DiR needs a MiniMax H3 audio-video latent (Empty MiniMax H3 AV Latent or MiniMax H3 Image to Video)")
    samples = comfy.sample.fix_empty_latent_channels(model_patcher, samples)
    video, audio = samples.unbind()[:2]
    if video.shape[0] != 1:
        raise ValueError("DiR for H3 samples batch size 1")
    if video.shape[1] != 24 or video.shape[3] % 2 or video.shape[4] % 2:
        raise ValueError(f"unexpected H3 video latent {tuple(video.shape)} (width and height must be multiples of 32)")
    _b, _c, latent_t, lat_h, lat_w = video.shape
    from .vendor.dir_core import audio_frames

    if audio.shape[-1] != audio_frames(latent_t):
        log.warning("DiR: audio latent has %d frames, dir_h3 uses %d for %d latent frames", audio.shape[-1],
                    audio_frames(latent_t), latent_t)

    device = comfy.model_management.get_torch_device()
    gen_device = device if noise_device == "gpu" and device.type == "cuda" else torch.device("cpu")
    # dir_h3.dir_sample: z from Generator(seed) and the silent-audio eps from Generator(seed + 1), on the GPU
    z = torch.randn((1, 24, latent_t, lat_h, lat_w), device=gen_device,
                    generator=torch.Generator(gen_device).manual_seed(int(seed)))
    eps = torch.randn((1, 32, 2, audio.shape[-1]), device=gen_device,
                      generator=torch.Generator(gen_device).manual_seed(int(seed) + 1))

    run = DirRun(head, eps, dwd_lora)
    patched = model_patcher.clone()
    patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, make_wrapper(run))

    result: dict = {"z": z}
    shapes = (tuple(video.shape), tuple(audio.shape))
    sampler = comfy.samplers.KSAMPLER(make_sampler_function(run, z, sigmas, mix_state, shapes, result, loop))
    guider = comfy.samplers.CFGGuider(patched)
    guider.inner_set_conds({"positive": positive})
    guider.set_cfg(1.0)
    sigma_tensor = torch.tensor(list(sigmas) + [0.0], dtype=torch.float32)
    noise = comfy.nested_tensor.NestedTensor((torch.zeros_like(video), torch.zeros_like(audio)))
    callback = latent_preview.prepare_callback(patched, len(sigmas))
    if disable_pbar is None:
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    out = guider.sample(noise, samples, sampler, sigma_tensor, callback=callback, disable_pbar=disable_pbar, seed=int(seed))
    out = out.to(comfy.model_management.intermediate_device())
    latent_out = {k: v for k, v in latent.items() if k not in ("downscale_ratio_spacial", "downscale_ratio_temporal")}
    latent_out["samples"] = out
    result["latent"] = latent_out
    return result
