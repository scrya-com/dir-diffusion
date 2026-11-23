"""A LyCORIS LoKr file (full ``lokr_w1`` / ``lokr_w2`` per layer, e.g. fizgig's ``scrya_iso_scene``) applied to a
musubi H3 DiT as frozen forward hooks: ``y = f(x) + strength * x @ kron(w1, w2).T``, scale ``alpha / dim`` with
``dim`` unset for full factors, i.e. 1 (ComfyUI ``weight_adapter/lokr.py``). The int8 weights are never touched
(merging re-quantizes them, which this generator does not tolerate), and the hooks stay on for every forward:
teacher, generator and fake all see base + adapter.

    handle = apply_base_lokr(dit, "/media/2TB/lora-data/fizgig_h3/loras/scrya_iso_scene/scrya_iso_scene.safetensors")
    ...
    handle.remove()
"""

from __future__ import annotations

from pathlib import Path

import torch

PREFIX = "diffusion_model."


def kron_apply(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """``x @ kron(w1, w2).T`` without building the product: x (..., m*n), w1 (o, m), w2 (p, n) -> (..., o*p)."""
    o, m = w1.shape
    p, n = w2.shape
    xr = x.reshape(*x.shape[:-1], m, n)
    y = torch.matmul(torch.matmul(w1, xr), w2.T)          # (..., o, p)
    return y.reshape(*x.shape[:-1], o * p)


class BaseLoKr:
    def __init__(self, dit, path: str | Path, strength: float = 1.0, dtype: torch.dtype | None = None):
        from safetensors.torch import load_file

        state = load_file(str(path))
        names = sorted({k[: -len(".lokr_w1")] for k in state if k.endswith(".lokr_w1")})
        if not names:
            raise ValueError(f"{path}: no .lokr_w1 keys (not a LoKr file)")
        self.strength = float(strength)
        self.handles = []
        self.factors: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
        missing = []
        for name in names:
            if any(k.startswith(name + ".") and k.split(".")[-1] not in ("lokr_w1", "lokr_w2", "alpha") for k in state):
                raise ValueError(f"{path}: {name} has decomposed or extra LoKr factors; only full w1/w2 is supported")
            module_name = name[len(PREFIX):] if name.startswith(PREFIX) else name
            try:
                module = dit.get_submodule(module_name)
            except AttributeError:
                missing.append(module_name)
                continue
            w1, w2 = state[name + ".lokr_w1"], state[name + ".lokr_w2"]
            out_f, in_f = w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1]
            weight_shape = getattr(module, "out_features", None), getattr(module, "in_features", None)
            if weight_shape != (out_f, in_f):
                raise ValueError(f"{module_name}: LoKr is {out_f}x{in_f}, layer is {weight_shape[0]}x{weight_shape[1]}")
            device = next((t.device for t in module.parameters()), torch.device("cpu"))
            target = dtype or torch.bfloat16
            self.factors[module_name] = (w1.to(device, target), w2.to(device, target), 1.0)
            self.handles.append(module.register_forward_hook(self._hook(module_name)))
        if missing:
            raise ValueError(f"{path}: {len(missing)} LoKr layers have no module in this DiT, e.g. {missing[:3]}")

    def _hook(self, name: str):
        def hook(_module, args, output):
            if self.strength == 0.0:
                return output
            w1, w2, scale = self.factors[name]
            x = args[0]
            if w1.device != x.device:                         # block swap moved the layer; follow it once
                w1, w2 = w1.to(x.device), w2.to(x.device)
                self.factors[name] = (w1, w2, scale)
            delta = kron_apply(x.to(w1.dtype), w1, w2)
            return output + (self.strength * scale) * delta.to(output.dtype)
        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def apply_base_lokr(dit, spec: str, dtype: torch.dtype | None = None) -> BaseLoKr:
    """``spec`` is ``PATH`` or ``PATH:STRENGTH``."""
    path, strength = str(spec), ""
    if not Path(path).exists() and ":" in path:
        path, _, strength = path.rpartition(":")
    return BaseLoKr(dit, path, float(strength) if strength else 1.0, dtype)
