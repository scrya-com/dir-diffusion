#!/usr/bin/env python3
# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""CPU-only checks of the DiR H3 node pack. No GPU, no ComfyUI server, writes no file.

    CUDA_VISIBLE_DEVICES= /path/to/comfyui/python tests/test_cpu.py [--real-files]

Run it with ComfyUI's Python (``COMFYUI_DIR`` defaults to /media/2TB/ComfyUI). ``lot/`` (dir-diffusion) is used
as the reference when the pack sits in its repo (``DIR_DIFFUSION_LOT`` overrides). The checks:

1. the pack imports through ``comfy_entrypoint`` and every node schema builds;
2. both example API workflows validate against the real node schemas (classes, inputs, links, socket types, combos);
3. the vendored code equals ``lot/`` (token order, sigmas, ``next_state``, DWD head decode, refiner);
4. end to end on a tiny random H3 (3 blocks) built twice with the same weights, as ComfyUI's ``MiniMaxH3Model`` behind
   a real ``ModelPatcher`` + bypass generator LoRA, and as musubi's model with ``lot/h3_musubi.make_lora``:
   ``DiR H3 Sampler``'s path (``dir_runtime.sample``: guider, conds, hooks) against ``lot/dir_h3.dir_sample``, for
   both state updates; and the DWD last-layer tokens against ``lot``'s capture with the DWD LoRA on;
5. ``--real-files``: the published run (``lot/runs/dir_turbo_clip``) and the local DWD files load, their LoRA keys
   all map onto a 50-block H3, and the DWD head + refiner decode a tiny random token grid.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # before torch: never touch a GPU

PACK = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("COMFYUI_DIR", "/media/2TB/ComfyUI"))
LOT = Path(os.environ.get("DIR_DIFFUSION_LOT", PACK.parents[1] / "lot"))
WORKFLOWS = [PACK.parent / "dir_h3_video_api.json", PACK.parent / "dir_dwd_h3_video_api.json"]

sys.path.insert(0, str(COMFY))
import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True
import torch  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def load_pack():
    spec = importlib.util.spec_from_file_location(PACK.name, PACK / "__init__.py", submodule_search_locations=[str(PACK)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------- 1. import, schemas

def test_import(pack):
    nodes = asyncio.run(asyncio.run(pack.comfy_entrypoint()).get_node_list())
    names = sorted(n.__name__ for n in nodes)
    check("entrypoint lists 4 nodes", names == ["DiRH3DWDLoader", "DiRH3DWDSampler", "DiRH3Loader", "DiRH3Sampler"], str(names))
    for node in nodes:
        info = node.GET_NODE_INFO_V1()
        check(f"schema {node.__name__}", bool(info["output"]) and "required" in info["input"], f"outputs={info['output']}")
    return nodes


# ------------------------------------------------------------------------------------- 2. workflows

def node_classes(pack_nodes) -> dict:
    import nodes as comfy_nodes
    from comfy_extras import nodes_images, nodes_minimax_h3

    classes = dict(comfy_nodes.NODE_CLASS_MAPPINGS)
    for module in (nodes_minimax_h3, nodes_images):
        ext = asyncio.run(module.comfy_entrypoint())
        for node in asyncio.run(ext.get_node_list()):
            classes[node.GET_SCHEMA().node_id] = node
    for node in pack_nodes:
        classes[node.GET_SCHEMA().node_id] = node
    return classes


def input_spec(cls) -> tuple[dict, dict]:
    spec = cls.INPUT_TYPES()
    return spec.get("required", {}), spec.get("optional", {})


def output_types(cls) -> list[str]:
    if hasattr(cls, "GET_NODE_INFO_V1"):
        return list(cls.GET_NODE_INFO_V1()["output"])
    return list(cls.RETURN_TYPES)


FILE_COMBOS = {"unet_name", "clip_name", "vae_name"}


def test_workflows(pack_nodes):
    classes = node_classes(pack_nodes)
    for path in WORKFLOWS:
        workflow = json.loads(path.read_text())
        problems = []
        for node_id, node in workflow.items():
            cls = classes.get(node["class_type"])
            if cls is None:
                problems.append(f"{node_id}: unknown class {node['class_type']}")
                continue
            required, optional = input_spec(cls)
            given = node["inputs"]
            for name in required:
                if name not in given:
                    problems.append(f"{node_id} {node['class_type']}: missing {name}")
            for name, value in given.items():
                spec = required.get(name, optional.get(name))
                if spec is None:
                    problems.append(f"{node_id} {node['class_type']}: unknown input {name}")
                    continue
                kind = spec[0]
                if isinstance(value, list):
                    src, index = workflow.get(value[0]), value[1]
                    if src is None:
                        problems.append(f"{node_id}.{name}: link to missing node {value[0]}")
                        continue
                    outs = output_types(classes[src["class_type"]])
                    if index >= len(outs) or (isinstance(kind, str) and outs[index] != kind and kind != "*"):
                        problems.append(f"{node_id}.{name}: {src['class_type']}[{index}] is "
                                        f"{outs[index] if index < len(outs) else '?'}, wants {kind}")
                elif isinstance(kind, list) or kind == "COMBO":
                    options = kind if isinstance(kind, list) else spec[1].get("options", [])
                    if name in FILE_COMBOS:
                        continue                    # depends on this machine's model folders (checked below)
                    if value not in options:
                        problems.append(f"{node_id}.{name}: {value!r} not in {options}")
        check(f"workflow {path.name}", not problems, "; ".join(problems))
        import folder_paths

        missing = []
        for node in workflow.values():
            for name, folder in (("unet_name", "diffusion_models"), ("clip_name", "text_encoders"), ("vae_name", "vae")):
                value = node["inputs"].get(name)
                if isinstance(value, str) and value not in folder_paths.get_filename_list(folder):
                    missing.append(value)
        print(f"note  {path.name}: model files {'all present in ' + str(COMFY) if not missing else 'missing: ' + ', '.join(missing)}")


# ------------------------------------------------------------------------------------- 3. vendored == lot

def test_vendored(pack):
    sys.path.insert(0, str(LOT))
    import dir_h3
    import dwd_h3
    import refine_dwd_h3
    from h3 import patchify as lot_patchify

    from comfy.ldm.minimax.model import patchify_video

    core = sys.modules[f"{PACK.name}.vendor.dir_core"]
    dwd = sys.modules[f"{PACK.name}.vendor.dwd_core"]
    lat = torch.randn(1, 24, 3, 4, 6)
    a = lot_patchify(lat).flatten(1, 3)[0]
    check("token order: lot patchify == comfy patchify_video == dwd pack_rows == vendored patchify",
          torch.equal(a, patchify_video(lat)) and torch.equal(a, dwd_h3.pack_rows(lat))
          and torch.equal(a, dwd.pack_rows(lat)) and torch.equal(a, core.patchify(lat).flatten(1, 3)[0]))
    specs = ["1", "1,0.9473684210526315,0.8571428571428571,0.6666666666666666", "grid", "turbo", "turbo:4"]
    check("parse_sigmas matches lot", all(core.parse_sigmas(s, 4) == dir_h3.parse_sigmas(s, 4) for s in specs))
    s, c, z = torch.randn(1, 24, 2, 4, 4), torch.randn(1, 24, 2, 4, 4), torch.randn(1, 24, 2, 4, 4)
    check("next_state matches lot", torch.equal(core.next_state(s, c, 0.9, 0.6), dir_h3.next_state(s, c, 0.9, 0.6))
          and torch.equal(core.next_state(s, c, 0.9, 0.6, z), dir_h3.next_state(s, c, 0.9, 0.6, z)))
    head_v, head_l = core.DirHead(64, 8), dir_h3.DirHead(64, 8)
    check("DirHead state keys match lot", head_v.state_dict().keys() == head_l.state_dict().keys())
    torch.manual_seed(1)
    head_l.paper_init()
    head_v.load_state_dict(head_l.state_dict())
    with torch.no_grad():
        check("estimate_extra matches lot", torch.equal(core.estimate_extra(head_v, lat), dir_h3.estimate_extra(head_l, lat, None)[0]))

    for out, label in ((dwd_h3.OUT, "plain"), (dwd_h3.ext_out(4), "extended halo 4")):
        ref = dwd_h3.DWDHead(hidden=16, out=out, classes=5)
        with torch.no_grad():
            ref.weight.normal_(0, 0.02)
            ref.bias.normal_(0, 0.1)
            ref.mean.normal_()
            ref.scale.uniform_(0.5, 2)
        mine = dwd.DWDHead(hidden=16, out=out, classes=5)
        mine.load_state_dict(ref.state_dict())
        h, s = torch.randn(7 * 2 * 3, 16), torch.randn(7 * 2 * 3, 96)
        want = ref.decode(h, s, 7, 2, 3)
        got = mine.decode(h, s, 7, 2, 3)
        check(f"DWD head decode matches lot ({label})", torch.allclose(got, want, atol=1e-5), f"max {float((got - want).abs().max()):.2e}")

    torch.manual_seed(2)
    ref = refine_dwd_h3.Refiner(16, latent=refine_dwd_h3.CONTEXT_CH, grid=32, grid_blocks=2).eval()
    for p in ref.parameters():
        torch.nn.init.normal_(p, 0, 0.05)
    saved = {"model": ref.state_dict(), "base": 16, "latent": True, "latent_spec": refine_dwd_h3.CONTEXT_CH, "grid": 32,
             "grid_blocks": 2}
    mine = dwd.Refiner(**dwd.spec_of(saved)).eval()
    mine.load_state_dict(saved["model"])
    clip, latent = torch.rand(1, 3, 9, 32, 48) * 2 - 1, torch.randn(1, 24, 3, 2, 3)
    want = refine_dwd_h3.refine_clip(ref, clip, latent)
    got = dwd.refine_clip(mine, clip, latent)
    check("refiner + refine_clip match lot", torch.equal(got, want), f"max {float((got - want).abs().max()):.2e}")


# ------------------------------------------------------------------------------------- 4. tiny H3

TINY = dict(hidden_size=256, num_layers=3, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=128,
            ffn_hidden_size=128, text_dim=32, time_embed_dim=8, adaln_curve_grid=1025)


def tiny_weights(reference) -> dict:
    torch.manual_seed(0)
    sd = {}
    for key, value in reference.state_dict().items():
        if key == "rope.inv_freq":
            sd[key] = 1.0 / (10000 ** (torch.arange(value.shape[0]).float() / value.shape[0]))
        elif key == "adaln_t_table":
            sd[key] = torch.randn_like(value) * 0.5
        elif value.ndim >= 2:
            sd[key] = torch.randn_like(value) / value.shape[-1] ** 0.5
        elif "norm" in key:
            sd[key] = 1 + 0.1 * torch.randn_like(value)
        else:
            sd[key] = 0.1 * torch.randn_like(value)
    return sd


def build_tiny(num_layers: int = 3):
    """``(comfy ModelPatcher, musubi model, weights)`` with identical weights, fp32 on the CPU."""
    import comfy.model_patcher
    import comfy.supported_models

    sys.path.insert(0, str(LOT))
    import h3_musubi  # noqa: F401  (puts musubi on sys.path)
    from musubi_tuner.minimax_h3.model import MiniMaxH3Config, MiniMaxH3Model

    cfg = dict(TINY, num_layers=num_layers)
    model_config = comfy.supported_models.MiniMaxH3(dict(cfg, image_model="minimax_h3"))
    model_config.set_inference_dtype(torch.float32, None)
    base = model_config.get_model({}, device="cpu")
    musubi = MiniMaxH3Model(MiniMaxH3Config(**cfg, time_embed_hidden_size=cfg["hidden_size"]), attn_mode="torch",
                            dtype=torch.float32, device="cpu")
    weights = tiny_weights(musubi)
    musubi.load_state_dict(weights)
    base.diffusion_model.load_state_dict(weights)
    for model in (musubi, base.diffusion_model):
        model.eval().requires_grad_(False)
    patcher = comfy.model_patcher.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    return patcher, musubi


def random_lora(musubi, rank: int, blocks=None, seed: int = 3) -> tuple[object, dict]:
    """A musubi LoRA network on ``musubi`` with random weights; returns it and its state dict (musubi keys),
    restricted to ``blocks`` when given (the other modules disabled, as ``refine_dwd_h3.cache`` does)."""
    from h3_musubi import make_lora

    network = make_lora(musubi, rank)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in network.named_parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * (0.3 / p.shape[-1] ** 0.5))
    network.requires_grad_(False)
    state = {k: v.clone() for k, v in network.state_dict().items()}
    if blocks is not None:
        keep = tuple(f"lora_unet_blocks_{b}_" for b in blocks)
        state = {k: v for k, v in state.items() if k.startswith(keep)}
        for module in network.unet_loras:
            module.enabled = module.lora_name.startswith(keep)
    return network, state


def test_tiny(pack):
    sys.path.insert(0, str(LOT))
    import dir_h3
    from dwd_h3 import capture_last_layer

    import comfy.nested_tensor

    nodes_mod = sys.modules[f"{PACK.name}.nodes"]
    runtime = sys.modules[f"{PACK.name}.dir_runtime"]
    core = sys.modules[f"{PACK.name}.vendor.dir_core"]

    patcher, musubi = build_tiny()
    gen_net, gen_state = random_lora(musubi, rank=4)
    head_l = dir_h3.DirHead(256, 8, 4)
    torch.manual_seed(5)
    with torch.no_grad():
        head_l.x0_proj.weight.normal_(0, 0.05)
        head_l.x0_proj.bias.normal_(0, 0.05)
        head_l.r_embed.normal_(0, 0.3)
    head_v = core.DirHead(256, 8, 4)
    head_v.load_state_dict(head_l.state_dict())

    lora_patcher = nodes_mod.apply_generator_lora(patcher, gen_state, 1.0, "bypass (as trained)")
    text = torch.randn(1, 5, 32, generator=torch.Generator().manual_seed(9))
    T, H, W, seed = 2, 4, 6, 11
    sigmas = core.parse_sigmas("1,0.9473684210526315,0.8571428571428571,0.6666666666666666", 4)
    latent = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, T, H, W),
                                                           torch.zeros(1, 32, 2, core.audio_frames(T))))}
    positive = [[text, {}]]

    for mix in (False, True):
        with torch.inference_mode():
            want, want_all = dir_h3.dir_sample(musubi, head_l, text, height=16 * H, width=16 * W, latent_t=T, iters=4,
                                               seed=seed, sigmas=sigmas, mix_state=mix, device="cpu")
            got = runtime.sample(lora_patcher, copy.deepcopy(head_v), positive, latent, seed=seed, sigmas=sigmas,
                                 mix_state=mix, noise_device="cpu", disable_pbar=True)
        per_iter = [rel(g, w) for g, w in zip(got["estimates"], want_all)]
        out_video = got["latent"]["samples"].unbind()[0]
        check(f"tiny H3: DiR Sampler path == lot dir_sample (mix_state={mix})", max(per_iter) < 1e-4,
              "per-iteration rel " + ", ".join(f"{e:.1e}" for e in per_iter))
        check(f"tiny H3: LATENT output is the final estimate (mix_state={mix})", rel(out_video, want) < 1e-4,
              f"rel {rel(out_video, want):.1e}")

    # the teacher-forced loop of parity_gpu.py: each iteration fed lot's own (state, previous estimate)
    with torch.inference_mode():
        _final, want_all = dir_h3.dir_sample(musubi, head_l, text, height=16 * H, width=16 * W, latent_t=T, iters=4,
                                             seed=seed, sigmas=sigmas, device="cpu")
        z = torch.randn(1, 24, T, H, W, generator=torch.Generator("cpu").manual_seed(seed))
        states = [z]
        for r in range(3):
            states.append(dir_h3.next_state(states[-1], want_all[r], sigmas[r], sigmas[r + 1]))
        prevs = [torch.zeros_like(z)] + want_all[:-1]

        def forced(step, _z, iters, sigmas_, _mix):
            out = [states[r].float() + sigmas_[r] * step(r, states[r], prevs[r], sigmas_[r]).float() for r in range(iters)]
            return out[-1], out, states[-1]

        tf = runtime.sample(lora_patcher, copy.deepcopy(head_v), positive, latent, seed=seed, sigmas=sigmas,
                            mix_state=False, noise_device="cpu", disable_pbar=True, loop=forced)
    per_iter = [rel(g, w) for g, w in zip(tf["estimates"], want_all)]
    check("tiny H3: teacher-forced loop (parity_gpu.py) == lot per iteration", max(per_iter) < 1e-4,
          ", ".join(f"{e:.1e}" for e in per_iter))

    # sensitivity: without the hooks' inputs (zero head) or without the LoRA, the result moves by far more
    with torch.inference_mode():
        zero = runtime.sample(lora_patcher, core.DirHead(256, 8, 4), positive, latent, seed=seed, sigmas=sigmas,
                              mix_state=False, noise_device="cpu", disable_pbar=True)["estimate"]
        no_lora = runtime.sample(patcher, copy.deepcopy(head_v), positive, latent, seed=seed, sigmas=sigmas,
                                 mix_state=False, noise_device="cpu", disable_pbar=True)["estimate"]
        want = dir_h3.dir_sample(musubi, head_l, text, height=16 * H, width=16 * W, latent_t=T, iters=4, seed=seed,
                                 sigmas=sigmas, device="cpu")[0]
    check("tiny H3: the DiR head changes the result (test is sensitive)", rel(zero, want) > 1e-2, f"rel {rel(zero, want):.2e}")
    check("tiny H3: the generator LoRA changes the result", rel(no_lora, want) > 1e-2, f"rel {rel(no_lora, want):.2e}")

    # DWD: last-layer tokens of the final iteration with a DWD LoRA on blocks 1-2, against lot's capture
    dwd_net, dwd_state = random_lora(musubi, rank=4, blocks=(1, 2), seed=4)
    dwd_lora = runtime.RuntimeLora(dwd_state, dtype=torch.float32)
    dwd_modules = [m for m in dwd_net.unet_loras if m.enabled]
    store: dict = {}
    with torch.inference_mode():
        z = torch.randn(1, 24, T, H, W, generator=torch.Generator("cpu").manual_seed(seed))
        estimate, state = torch.zeros_like(z), z
        for module in dwd_modules:
            module.enabled = False
        for r in range(4):
            if r == 3:
                for module in dwd_modules:
                    module.enabled = True
                with capture_last_layer(musubi, store):
                    dir_h3.generator_step(musubi, head_l, state, estimate, r, text, sigma=sigmas[r],
                                          audio_generator=torch.Generator("cpu").manual_seed(seed + 1))
                for module in dwd_modules:
                    module.enabled = False
            estimate = dir_h3.generator_step(musubi, head_l, state, estimate, r, text, sigma=sigmas[r],
                                             audio_generator=torch.Generator("cpu").manual_seed(seed + 1))
            if r < 3:
                state = dir_h3.next_state(state, estimate, sigmas[r], sigmas[r + 1])
        got = runtime.sample(lora_patcher, copy.deepcopy(head_v), positive, latent, seed=seed, sigmas=sigmas,
                             mix_state=False, noise_device="cpu", dwd_lora=dwd_lora, disable_pbar=True)
    check("tiny H3: DWD tokens (blocks 1.. re-run with the DWD LoRA) == lot capture_last_layer",
          rel(got["dwd_tokens"], store["h"]) < 1e-4, f"rel {rel(got['dwd_tokens'], store['h']):.1e}")
    check("tiny H3: DWD run's LATENT is the plain DiR estimate (DWD LoRA off)", rel(got["estimate"], estimate) < 1e-4,
          f"rel {rel(got['estimate'], estimate):.1e}")
    check("tiny H3: DWD input state s == lot's final-iteration state", rel(got["state"], state) < 1e-4,
          f"rel {rel(got['state'], state):.1e}")
    with torch.inference_mode():
        no_dwd_h = {}
        with capture_last_layer(musubi, no_dwd_h):
            dir_h3.generator_step(musubi, head_l, state, got["estimates"][2], 3, text, sigma=sigmas[3],
                                  audio_generator=torch.Generator("cpu").manual_seed(seed + 1))
    check("tiny H3: the DWD LoRA changes the tokens (test is sensitive)", rel(no_dwd_h["h"], store["h"]) > 1e-2,
          f"rel {rel(no_dwd_h['h'], store['h']):.2e}")

    # rejections
    try:
        with torch.inference_mode():
            runtime.sample(lora_patcher, copy.deepcopy(head_v),
                           [[text, {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, H, W)}]}]],
                           latent, seed=seed, sigmas=sigmas, mix_state=False, noise_device="cpu", disable_pbar=True)
        check("tiny H3: keyframe conditioning is refused", False, "no error")
    except ValueError as error:
        check("tiny H3: keyframe conditioning is refused", "keyframes" in str(error), str(error)[:80])

    # style LoRA (a LoKr, as fizgig's isometric adapters) summed with the generator LoRA in one bypass injection;
    # lot's reference: the same LoKr as base_lokr forward hooks on the musubi model
    import tempfile

    from safetensors.torch import save_file

    import base_lokr

    g = torch.Generator().manual_seed(21)
    lokr = {}
    for name in ("blocks.1.attn.qkv_proj", "blocks.1.mlp.fc1", "blocks.2.attn.out_proj"):
        layer = musubi.get_submodule(name)
        o, i = layer.out_features, layer.in_features
        lokr[f"diffusion_model.{name}.lokr_w1"] = torch.randn(2, 2, generator=g) * 0.5
        lokr[f"diffusion_model.{name}.lokr_w2"] = torch.randn(o // 2, i // 2, generator=g) * (0.5 / (i // 2) ** 0.5)
        lokr[f"diffusion_model.{name}.alpha"] = torch.tensor(1.0)
    styled = nodes_mod.apply_generator_lora(patcher, gen_state, 1.0, "bypass (as trained)", lokr, 0.7)
    with tempfile.TemporaryDirectory() as tmp:
        save_file(lokr, f"{tmp}/style.safetensors")
        hooks = base_lokr.apply_base_lokr(musubi, f"{tmp}/style.safetensors:0.7", dtype=torch.float32)
        try:
            with torch.inference_mode():
                want_style = dir_h3.dir_sample(musubi, head_l, text, height=16 * H, width=16 * W, latent_t=T, iters=4,
                                               seed=seed, sigmas=sigmas, device="cpu")[0]
        finally:
            hooks.remove()
    with torch.inference_mode():
        got_style = runtime.sample(styled, copy.deepcopy(head_v), positive, latent, seed=seed, sigmas=sigmas,
                                   mix_state=False, noise_device="cpu", disable_pbar=True)["estimate"]
        plain_again = runtime.sample(lora_patcher, copy.deepcopy(head_v), positive, latent, seed=seed, sigmas=sigmas,
                                     mix_state=False, noise_device="cpu", disable_pbar=True)["estimate"]
    check("tiny H3: style LoKr + generator LoRA (one bypass injection) == lot dir_sample with base_lokr hooks",
          rel(got_style, want_style) < 1e-4, f"rel {rel(got_style, want_style):.1e}")
    check("tiny H3: the style LoKr changes the result (it is applied)", rel(got_style, plain_again) > 1e-2,
          f"rel {rel(got_style, plain_again):.2e}")
    check("tiny H3: after a styled run the plain model is unchanged (no hook left behind)",
          rel(plain_again, want) < 1e-4, f"rel {rel(plain_again, want):.1e}")
    try:
        stacked = patcher.clone()
        stacked.set_injections("bypass_lora", [])
        stacked.injections["bypass_lora"] = [object()]
        nodes_mod.apply_generator_lora(stacked, gen_state, 1.0, "bypass (as trained)")
        check("tiny H3: an upstream bypass LoRA is refused (it would be replaced)", False, "no error")
    except ValueError as error:
        check("tiny H3: an upstream bypass LoRA is refused (it would be replaced)", "style_lora" in str(error),
              str(error)[:80])

    # Ref2VA: a reference image adds clean ref_img rows; the hooks rewrite only the target video and audio
    ref = {"kind": "image", "latent_h": H, "latent_w": W,
           "latent": torch.randn(1, 24, 1, H, W, generator=torch.Generator().manual_seed(3))}
    ref_positive = [[text, {"minimax_refs": [ref]}]]
    with torch.inference_mode():
        a = runtime.sample(lora_patcher, copy.deepcopy(head_v), ref_positive, latent, seed=seed, sigmas=sigmas,
                           mix_state=False, noise_device="cpu", disable_pbar=True)
        b = runtime.sample(lora_patcher, copy.deepcopy(head_v), ref_positive, latent, seed=seed, sigmas=sigmas,
                           mix_state=False, noise_device="cpu", disable_pbar=True, dwd_lora=dwd_lora)
        plain = runtime.sample(lora_patcher, copy.deepcopy(head_v), positive, latent, seed=seed, sigmas=sigmas,
                               mix_state=False, noise_device="cpu", disable_pbar=True)
    check("tiny H3: Ref2VA reference conditioning runs, finite", bool(torch.isfinite(a["estimate"]).all()))
    check("tiny H3: the reference changes the result (refs are read)", rel(a["estimate"], plain["estimate"]) > 1e-3,
          f"rel {rel(a['estimate'], plain['estimate']):.2e}")
    check("tiny H3: Ref2VA + DWD: LATENT unchanged by the DWD capture", rel(b["estimate"], a["estimate"]) < 1e-5,
          f"rel {rel(b['estimate'], a['estimate']):.1e}")
    n_video = T * (H // 2) * (W // 2)
    check("tiny H3: Ref2VA + DWD tokens are the target video rows only",
          tuple(b["dwd_tokens"].shape[:1]) == (n_video,), str(tuple(b["dwd_tokens"].shape)))


# ------------------------------------------------------------------------------------- 5. real files

def test_real_files(pack):
    nodes_mod = sys.modules[f"{PACK.name}.nodes"]
    run = LOT / "runs" / "dir_turbo_clip"
    if not (run / "config.json").is_file():
        print(f"skip  real files: {run} not found")
        return
    bundle, lora = nodes_mod.load_dir_bundle(str(run))
    check("published run loads", bundle.iters == 4 and bundle.mix_state is False and len(bundle.sigmas) == 4
          and tuple(bundle.head.r_embed.shape) == (4, 8) and tuple(bundle.head.x0_proj.weight.shape) == (5376, 96), repr(bundle))
    # musubi fixes scale = alpha / rank when the network is built (alpha = rank in make_lora); ComfyUI and RuntimeLora
    # read alpha from the file, so the two agree only while the saved alpha equals the rank
    ranks = {k[:-len(".lora_down.weight")]: v.shape[0] for k, v in lora.items() if k.endswith(".lora_down.weight")}
    check("generator LoRA alpha == rank (scale 1, as musubi ran it)",
          all(float(lora[n + ".alpha"]) == r for n, r in ranks.items() if n + ".alpha" in lora), f"rank {set(ranks.values())}")
    patcher, _ = build_tiny(num_layers=50)          # tiny widths, H3's 50 block names: key mapping only
    for mode in nodes_mod.LORA_MODES:
        try:
            nodes_mod.apply_generator_lora(patcher, lora, 1.0, mode)
            check(f"generator LoRA keys all map onto H3 ({mode})", True, f"{sum(k.endswith('lora_down.weight') for k in lora)} modules")
        except RuntimeError as error:
            check(f"generator LoRA keys all map onto H3 ({mode})", False, str(error))
    try:
        nodes_mod.apply_generator_lora(patcher, lora, 1.0, "merge (stock LoRA loader)")
        check("merge lora_mode (removed) is refused", False, "no error")
    except ValueError as error:
        check("merge lora_mode (removed) is refused", "bypass" in str(error), str(error)[:80])
    paths = {"lora": "/media/2TB/lora-data/h3_lot/dwd/wB/dwd_ft_lora.pt",
             "head": "/media/2TB/lora-data/h3_lot/dwd/wB_head/dwd_head_ridge.pt",
             "refiner": "/media/2TB/lora-data/h3_lot/dwd/wG/refiner_best.pt"}
    if not all(os.path.isfile(p) for p in paths.values()):
        print("skip  DWD files not found")
        return
    dwd = nodes_mod.load_dwd_bundle(paths["lora"], paths["head"], paths["refiner"])
    check("DWD files load", dwd.lora.block_indices() == list(range(42, 50)) and dwd.head.halo == 4
          and dwd.refiner is not None and dwd.refiner.latent_ch == 76, repr(dwd))
    check("DWD LoRA alpha == rank (scale 1)", all(scale == 1.0 for _d, _u, scale in dwd.lora.modules.values()))
    dwd.lora.targets(patcher.model.diffusion_model)
    check("DWD LoRA modules all exist in a 50-block H3", True)
    dwd_mod = sys.modules[f"{PACK.name}.vendor.dwd_core"]
    tokens, state, latent = torch.randn(2 * 2 * 3, 5376), torch.randn(1, 24, 2, 4, 6), torch.randn(1, 24, 2, 4, 6)
    pixels = dwd.head.decode(tokens, dwd_mod.pack_rows(state), 2, 2, 3)
    refined = dwd_mod.refine_clip(dwd.refiner, pixels, latent)
    check("DWD head + refiner decode a tiny grid on the CPU", tuple(refined.shape) == (1, 3, 5, 64, 96)
          and bool(torch.isfinite(refined).all()), f"{tuple(refined.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-files", action="store_true", help="also load the published run and the DWD files (reads ~1.5 GB)")
    args = parser.parse_args()
    pack = load_pack()
    nodes = test_import(pack)
    test_workflows(nodes)
    if (LOT / "dir_h3.py").is_file():
        test_vendored(pack)
        test_tiny(pack)
    else:
        print(f"skip  lot reference checks: {LOT} not found")
    if args.real_files:
        test_real_files(pack)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
