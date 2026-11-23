# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""ComfyUI nodes for DiR (data-space iteration) on MiniMax H3, with optional DWD (decoding without the VAE).

* DiR H3 Loader: a run directory or a Hugging Face repo (``johndpope/dir-diffusion-h3``) -> the generator LoRA on
  MODEL (``lot/dir_h3.load_dir_run``) and the ``DirHead`` (``x0_proj``, ``r_embed``) plus the run's sigmas.
* DiR H3 Sampler: ``lot/dir_h3.dir_sample`` inside ComfyUI's own H3 forward -> LATENT.
* DiR H3 DWD Loader: the final-iteration LoRA, the pixel head and the refiner, by file path (or ``hf://``).
* DiR H3 DWD Sampler: the same sampling, plus ``DWDHead.decode`` and ``refine_clip`` on the final iteration's
  last-layer tokens -> IMAGE, without the VAE. Its LATENT is the plain DiR latent (VAE Decode it to compare).
"""

from __future__ import annotations

import copy
import json
import logging
import os

import torch

import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io

from . import dir_runtime
from .vendor.dir_core import DirHead, parse_sigmas
from .vendor.dwd_core import load_head, load_refiner, pack_rows, refine_clip

log = logging.getLogger(__name__)

DEFAULT_REPO = "johndpope/dir-diffusion-h3"
DIR_FOLDER = os.path.join(folder_paths.models_dir, "dir_h3")
folder_paths.add_model_folder_path("dir_h3", DIR_FOLDER)

DirSpec = io.Custom("DIR_H3")
DwdSpec = io.Custom("DWD_H3")

# One mode. Merging (the stock LoRA loader) re-quantizes the int8 ConvRot weights, which this generator does not
# tolerate (a bf16-level change to the forward moves its output ~13%, see README), and it raised "Cannot set
# version_counter for inference tensor" when loaded under inference mode. The widget stays so saved workflows load.
LORA_MODES = ["bypass (as trained)"]
NOISE_DEVICES = ["gpu", "cpu"]
MIX_MODES = ["from run", "off", "on"]


def resolve_file(spec: str, filename: str | None = None) -> str:
    """A local file, a file inside a local folder, ``models/dir_h3/<spec>``, or ``hf://<owner>/<repo>/<path>``
    (``hf://<owner>/<repo>`` plus ``filename``), downloaded to the Hugging Face cache."""
    spec = spec.strip()
    if spec.startswith("hf://"):
        parts = spec[len("hf://"):].strip("/").split("/")
        repo, path = "/".join(parts[:2]), "/".join(parts[2:])
        if filename:
            path = f"{path}/{filename}" if path else filename
        if not path:
            raise ValueError(f"{spec}: name a file inside the repo")
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo, filename=path)
    for base in (spec, os.path.join(DIR_FOLDER, spec)):
        path = os.path.join(base, filename) if filename else base
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"{spec}{'/' + filename if filename else ''}: not found (a path, a folder in "
                            f"{DIR_FOLDER}, or hf://owner/repo)")


def _run_source(source: str) -> str:
    source = source.strip() or DEFAULT_REPO
    if os.path.isdir(source) or os.path.isdir(os.path.join(DIR_FOLDER, source)) or source.startswith("hf://"):
        return source
    if source.count("/") == 1 and not os.path.exists(source):
        return "hf://" + source
    return source


class DirBundle:
    """A loaded DiR generator: ``DirHead`` (CPU, fp32), its config, sigmas and state-update rule."""

    def __init__(self, head: DirHead, meta: dict, source: str):
        self.head, self.meta, self.source = head, meta, source
        self.iters = int(meta.get("iters", head.iterations))
        self.sigmas = parse_sigmas(meta.get("sigmas") or "1", self.iters)
        self.mix_state = bool(meta.get("mix_state", False))

    def __repr__(self):
        return f"DirBundle({self.source}, iters={self.iters}, sigmas={self.sigmas}, mix_state={self.mix_state})"


def load_dir_bundle(source: str, tag: str = "", ema: bool = False):
    """``lot/dir_h3.load_dir_run`` without the DiT: config, generator LoRA state dict and ``DirHead``."""
    source = _run_source(source)
    tag = (tag or "").strip()
    if ema:
        tag = f"{tag}_ema"
    meta = json.loads(open(resolve_file(source, "config.json"), encoding="utf-8").read())
    if meta.get("lot") or str(meta.get("schedule", "dense")).split(",")[0].strip() not in ("dense", ""):
        raise ValueError(f"{source}: a LoT-layout DiR run; this pack runs dense DiR runs only")
    lora_path = resolve_file(source, f"lora_gen{tag}.safetensors")
    head_path = resolve_file(source, f"dir_head{tag}.pt")
    head = DirHead(int(meta["hidden"]), int(meta["temb_dim"]), int(meta["iters"]))
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    head.eval().requires_grad_(False)
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    return DirBundle(head, meta, source), lora


def apply_generator_lora(model, lora: dict, strength: float, mode: str):
    """The generator LoRA on MODEL. Every ``lora_down`` must map to a layer (``load_dir_run`` refuses missing keys)."""
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map)
    expected = sum(1 for k in lora if k.endswith(".lora_down.weight"))
    if len(loaded) != expected:
        raise RuntimeError(f"generator LoRA: {len(loaded)} of {expected} modules map onto this model; is it MiniMax H3?")
    if not mode.startswith("bypass"):
        raise ValueError(f"lora_mode {mode!r}: only bypass is supported (merging re-quantizes the int8 weights)")
    return comfy.sd.load_bypass_lora_for_models(model, None, lora, strength, 0)[0]


def _check_h3(model) -> None:
    import comfy.model_base

    if not isinstance(model.model, comfy.model_base.MiniMaxH3):
        raise ValueError(f"DiR needs a MiniMax H3 model, got {type(model.model).__name__}")


def _mix(bundle: DirBundle, mode: str) -> bool:
    return bundle.mix_state if mode == "from run" else mode == "on"


def _sigmas(bundle: DirBundle, spec: str) -> list[float]:
    return bundle.sigmas if not spec.strip() else parse_sigmas(spec.strip(), bundle.iters)


class DiRH3Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiRH3Loader",
            display_name="DiR H3 Loader",
            category="model/minimax/dir",
            description=("Load a DiR few-step generator for MiniMax H3: applies its generator LoRA to the model and "
                         "holds the DiR head (previous-estimate projection and iteration embedding). Source: a run "
                         "folder (config.json, lora_gen.safetensors, dir_head.pt), a folder in models/dir_h3, or a "
                         "Hugging Face repo id."),
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 (the pruned int8 FL2VA DiT the run was trained on)"),
                io.String.Input("source", default=DEFAULT_REPO,
                                tooltip="run folder, models/dir_h3/<name>, or a Hugging Face repo id"),
                io.Combo.Input("lora_mode", options=LORA_MODES, default=LORA_MODES[0],
                               tooltip="bypass: the LoRA runs beside the int8 weights, as in training (the only mode; "
                                       "merging would re-quantize the int8 weights)"),
                io.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.01),
                io.String.Input("tag", default="", optional=True, advanced=True,
                                tooltip="intermediate save, e.g. _3 for lora_gen_3.safetensors"),
                io.Boolean.Input("ema", default=False, optional=True, advanced=True,
                                 tooltip="load the *_ema copy"),
            ],
            outputs=[io.Model.Output(), DirSpec.Output(display_name="dir")],
        )

    @classmethod
    def execute(cls, model, source, lora_mode, strength, tag="", ema=False) -> io.NodeOutput:
        _check_h3(model)
        bundle, lora = load_dir_bundle(source, tag, ema)
        log.info("DiR H3: %r", bundle)
        return io.NodeOutput(apply_generator_lora(model, lora, strength, lora_mode), bundle)


def _sampler_inputs():
    return [
        io.Model.Input("model", tooltip="MODEL from DiR H3 Loader"),
        DirSpec.Input("dir"),
        io.Conditioning.Input("positive", tooltip="MiniMax H3 text conditioning, or MiniMax H3 Reference to Video (Ref2VA) conditioning; no first/last frame"),
        io.Latent.Input("latent", tooltip="MiniMax H3 AV latent: sets width, height and length"),
        io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
        io.Combo.Input("noise_device", options=NOISE_DEVICES, default="gpu",
                       tooltip="gpu: the noise lot/dir_h3.dir_sample draws for this seed. cpu: device-independent noise."),
        io.String.Input("sigmas", default="", advanced=True,
                        tooltip="blank: the run's schedule. Else one value per iteration, from 1 (e.g. "
                                "1,0.947,0.857,0.667), 1 (pinned), grid[:SHIFT] or turbo[:SHIFT]"),
        io.Combo.Input("mix_state", options=MIX_MODES, default="from run", advanced=True,
                       tooltip="state update between iterations: Euler (off) or re-noise the estimate with the same z (on)"),
    ]


class DiRH3Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiRH3Sampler",
            display_name="DiR H3 Sampler",
            category="sampling/minimax/dir",
            description=("Data-space iteration: the generator refines its own clean estimate from one noise sample, "
                         "R forward passes (4 for the published run), no CFG. Text-to-video only."),
            inputs=_sampler_inputs(),
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, dir, positive, latent, seed, noise_device, sigmas="", mix_state="from run") -> io.NodeOutput:
        _check_h3(model)
        result = dir_runtime.sample(model, copy.deepcopy(dir.head), positive, latent, seed=seed,
                                    sigmas=_sigmas(dir, sigmas), mix_state=_mix(dir, mix_state),
                                    noise_device=noise_device)
        return io.NodeOutput(result["latent"])


class DwdBundle:
    """DWD weights: the final-iteration LoRA (hooks), the pixel head (CPU, mmap) and an optional refiner."""

    def __init__(self, lora: dir_runtime.RuntimeLora, head, refiner, paths: dict):
        self.lora, self.head, self.refiner, self.paths = lora, head, refiner, paths

    def __repr__(self):
        return (f"DwdBundle(blocks {self.lora.block_indices()}, head classes={self.head.classes} "
                f"halo={self.head.halo}, refiner={'yes' if self.refiner is not None else 'no'})")


def load_dwd_bundle(lora_path: str, head_path: str, refiner_path: str = "", strength: float = 1.0) -> DwdBundle:
    saved = torch.load(resolve_file(lora_path), map_location="cpu", weights_only=True)
    state = saved["lora"] if isinstance(saved, dict) and "lora" in saved else saved
    lora = dir_runtime.RuntimeLora(state, dtype=torch.float32, multiplier=strength)
    head = load_head(resolve_file(head_path))
    refiner = load_refiner(resolve_file(refiner_path)) if refiner_path.strip() else None
    return DwdBundle(lora, head, refiner, {"lora": lora_path, "head": head_path, "refiner": refiner_path})


class DiRH3DWDLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiRH3DWDLoader",
            display_name="DiR H3 DWD Loader (experimental)",
            category="model/minimax/dir",
            description=("Decoding without the VAE: the final-iteration LoRA (last blocks), the linear pixel head and "
                         "the pixel refiner. Paths are local files or hf://owner/repo/file."),
            inputs=[
                io.String.Input("lora_path", default="/media/2TB/lora-data/h3_lot/dwd/wB/dwd_ft_lora.pt",
                                tooltip="train_dwd_h3.py ft output: {'lora', 'rank', 'k'}"),
                io.String.Input("head_path", default="/media/2TB/lora-data/h3_lot/dwd/wB_head/dwd_head_ridge.pt",
                                tooltip="DWDHead state (ridge / refit); 3.15 GB for the extended head, memory-mapped"),
                io.String.Input("refiner_path", default="/media/2TB/lora-data/h3_lot/dwd/wG/refiner_best.pt",
                                tooltip="refine_dwd_h3.py refiner; blank skips the refiner"),
                io.Float.Input("lora_strength", default=1.0, min=0.0, max=2.0, step=0.01, advanced=True),
            ],
            outputs=[DwdSpec.Output(display_name="dwd")],
        )

    @classmethod
    def execute(cls, lora_path, head_path, refiner_path, lora_strength=1.0) -> io.NodeOutput:
        bundle = load_dwd_bundle(lora_path, head_path, refiner_path, lora_strength)
        log.info("DiR H3: %r", bundle)
        return io.NodeOutput(bundle)


HEAD_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@torch.no_grad()
def dwd_decode(dwd: DwdBundle, tokens: torch.Tensor, state: torch.Tensor, estimate: torch.Tensor, *,
               refine: bool = True, head_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Head + refiner -> IMAGE ``(F, H, W, 3)`` in [0, 1]. ``state`` is the final iteration's input state (``s``),
    ``estimate`` the final DiR latent the refiner reads (``clip_latent`` in ``refine_dwd_h3.py``)."""
    device = comfy.model_management.get_torch_device()
    _b, _c, latent_t, lat_h, lat_w = estimate.shape
    head = dwd.head
    per_class = head.weight.shape[1] * head.weight.shape[2] * torch.finfo(head_dtype).bits // 8
    outputs = tokens.shape[0] * head.weight.shape[1] * 4
    comfy.model_management.free_memory(per_class + 3 * outputs + (512 << 20), device)
    s = pack_rows(state.float()).to(device)
    pixels = head.decode(tokens.to(device), s, latent_t, lat_h // 2, lat_w // 2, dtype=head_dtype)
    if refine and dwd.refiner is not None:
        dwd.refiner.to(device)
        try:
            pixels = refine_clip(dwd.refiner, pixels, estimate.to(device))
        finally:
            dwd.refiner.to("cpu")
    image = (pixels[0].permute(1, 2, 3, 0).float() + 1.0) / 2.0
    return image.clamp(0, 1).to(comfy.model_management.intermediate_device())


class DiRH3DWDSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiRH3DWDSampler",
            display_name="DiR H3 DWD Sampler (experimental)",
            category="sampling/minimax/dir",
            description=("DiR sampling that also decodes to pixels without the VAE: at the final iteration the last "
                         "blocks run again with the DWD LoRA, a linear head maps their tokens to pixels and a small "
                         "refiner sharpens them. LATENT is the plain DiR latent (for VAE Decode)."),
            inputs=_sampler_inputs() + [
                DwdSpec.Input("dwd"),
                io.Boolean.Input("refine", default=True, tooltip="run the refiner (if the DWD Loader has one)"),
                io.Combo.Input("head_dtype", options=list(HEAD_DTYPES), default="float32", advanced=True,
                               tooltip="dtype of the pixel head matmul (float32 as measured; bf16 is unchecked)"),
            ],
            outputs=[io.Image.Output(), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, dir, positive, latent, seed, noise_device, dwd, refine=True, head_dtype="float32",
                sigmas="", mix_state="from run") -> io.NodeOutput:
        _check_h3(model)
        result = dir_runtime.sample(model, copy.deepcopy(dir.head), positive, latent, seed=seed,
                                    sigmas=_sigmas(dir, sigmas), mix_state=_mix(dir, mix_state),
                                    noise_device=noise_device, dwd_lora=dwd.lora)
        image = dwd_decode(dwd, result["dwd_tokens"], result["state"], result["estimate"], refine=refine,
                           head_dtype=HEAD_DTYPES[head_dtype])
        return io.NodeOutput(image, result["latent"])


NODES = [DiRH3Loader, DiRH3Sampler, DiRH3DWDLoader, DiRH3DWDSampler]
