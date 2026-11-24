"""DeepCrossAttention (DCA) retrofitted onto the MiniMax H3 block stack. Research prototype.

Paper: "DeepCrossAttention: Supercharging Transformer Residual Connections" (Heddes et al., ICML 2025,
arXiv 2502.06785). docs/DCA_H3.md has the equations, what is the paper's and what is guessed here.

The paper replaces the residual stream ``h_t = x + sum_i f_i`` by learned mixes of the stack
``G_t = [x, f_0, ..., f_{t-1}]`` (GRN-v3: weights ``b + relu(w^T G)``, ``b`` per feature and per stack entry,
``w`` one vector per mix), with three separate mixes feeding the attention's query, key and value. With
``b = 1, w = 0`` every mix is the plain residual stream, so a pretrained model is unchanged.

H3's block is pre-norm with AdaLN and gated updates::

    h'  = h  + gate_msa * Attn(mod_msa(norm1(h)))
    h'' = h' + gate_mlp * MLP (mod_mlp(norm2(h')))

Here the stack entry of block ``t`` is its gated update ``f_t = gate_msa * attn + gate_mlp * mlp`` (what the
block adds to the stream), and with ``k`` recent entries kept the stack is the paper's first-and-last-k::

    slot 0      x      the embedded rows the stack starts from
    slot 1      mid    the sum of the updates older than the last k
    slot 2..    f_{t-1}, ..., f_{t-k}

Every mix is written as the stream plus a correction, ``mix = h + sum_s (beta_s + relu(w^T G_s)) * G_s`` with
``beta = b - 1`` stored, so zero parameters give ``h`` itself, bit for bit (a sum of the stack entries in bf16
would not reproduce the running sum). Each block has four mixes: Q, K, V (each goes through ``norm1`` and the
MSA modulation on its own) and R, the stream the MLP's input is built on (``norm2(R + gate_msa * attn)``).
One more mix, after the last block, is what the final layer reads.

``install(dit, dca)`` routes ``h3_musubi.run_blocks`` through ``run_blocks_dca`` for that ``dit`` only;
musubi-tuner is not edited. Blocks listed in ``dca.skip`` are not run (the depth-reduction use case).
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
from torch import nn

MIXES = ("q", "k", "v", "r")
FIXED_SLOTS = 2                                    # x and mid


class _Relu1(torch.autograd.Function):
    """ReLU whose gradient at exactly 0 is 1. With ``w = 0`` every score is exactly 0, where torch's ReLU
    has gradient 0: ``w`` would never move from the paper's zero initialisation."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp_min(0)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * (x >= 0).to(grad.dtype)


def relu1(x: torch.Tensor) -> torch.Tensor:
    return _Relu1.apply(x)


class Grn(nn.Module):
    """``mixes`` GRN-v3 instances over a ``slots``-entry stack, as corrections to the residual stream.

    ``beta`` ``(mixes, slots, hidden)`` is the paper's ``b - 1``; ``w`` ``(mixes, hidden)`` is its ``w``.
    Both start at zero.
    """

    def __init__(self, hidden: int, slots: int, mixes: int):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(mixes, slots, hidden))
        self.w = nn.Parameter(torch.zeros(mixes, hidden))

    def active(self) -> list[bool]:
        """Per mix: is any of its parameters nonzero."""
        with torch.no_grad():
            flags = (self.beta.abs().amax(dim=(1, 2)) > 0) | (self.w.abs().amax(dim=1) > 0)
        return flags.tolist()

    def forward(self, stream: torch.Tensor, stack: list[tuple[int, torch.Tensor]], general: bool) -> list[torch.Tensor]:
        """The mixes of ``stream`` ``(B, N, hidden)`` with the stack ``[(slot, tensor)]``. A mix whose
        parameters are all zero is ``stream`` itself (the same object) unless ``general``."""
        count = self.beta.shape[0]
        active = [True] * count if general else self.active()
        if not any(active):
            return [stream] * count
        dtype = stream.dtype
        w = self.w.to(dtype)
        in_place = not torch.is_grad_enabled()
        out = [stream.clone() if (on and in_place) else stream for on in active]
        for slot, entry in stack:
            scores = relu1(entry @ w.t())                              # (B, N, mixes): the input-dependent weight
            for m in range(count):
                if not active[m]:
                    continue
                coefficient = self.beta[m, slot].to(dtype) + scores[..., m:m + 1]
                if in_place:
                    out[m].addcmul_(entry, coefficient)
                else:
                    out[m] = out[m] + entry * coefficient
        return out


class DcaH3(nn.Module):
    """DCA parameters for a ``depth``-block stack of width ``hidden`` keeping ``k`` recent block updates."""

    def __init__(self, depth: int, hidden: int, k: int = 2, skip=()):
        super().__init__()
        if k < 0:
            raise ValueError("k must be >= 0")
        self.depth, self.hidden, self.k = depth, hidden, k
        self.slots = FIXED_SLOTS + k
        self.blocks = nn.ModuleList(Grn(hidden, self.slots, len(MIXES)) for _ in range(depth))
        self.out = Grn(hidden, self.slots, 1)
        self.skip = frozenset(skip)
        self.force_general = False     # run the separate-QKV path even when the parameters are zero
        self.qkv_mode = "auto"         # "auto" | "batched" (see project_qkv)
        self.stats: dict = {}

    @classmethod
    def for_dit(cls, dit, k: int = 2, skip=()):
        hidden = dit.blocks[0].norm1.weight.shape[0]
        return cls(len(dit.blocks), hidden, k, skip).to(dit.blocks[0].norm1.weight.device)

    def extra_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def history_bytes(self, tokens: int, bytes_per_value: int = 2, batch: int = 1) -> int:
        """The stack's size once full: ``x``, ``mid`` and ``k`` updates, each ``batch x tokens x hidden``."""
        return self.slots * batch * tokens * self.hidden * bytes_per_value

    def slot_weights(self, block: int) -> dict:
        """The static weights ``b = 1 + beta`` block ``block`` puts on the stack entries that exist for it
        (with ``self.skip`` applied), per mix: ``{mix: {slot name: (hidden,) tensor}}``. ``block == depth``
        is the output mix."""
        names = slot_names(block, self.k, self.skip)
        grn, mixes = (self.out, ("out",)) if block == self.depth else (self.blocks[block], MIXES)
        return {mix: {name: 1.0 + grn.beta[m, slot].detach() for slot, name in names.items()}
                for m, mix in enumerate(mixes)}


def slot_names(block: int, k: int, skip=()) -> dict[int, str]:
    """``{slot index: name}`` of the stack entries block ``block`` sees. Skipped blocks leave no entry, so
    the last-k window slides over the blocks that ran: weights are tied to slots, not to block numbers, and
    a slot with nothing in it carries no weight."""
    ran = [i for i in range(block) if i not in skip]
    names = {0: "x"}
    if len(ran) > k:
        names[1] = f"mid(sum of {len(ran) - k} updates)"
    for age, index in enumerate(reversed(ran[len(ran) - min(k, len(ran)):])):
        names[FIXED_SLOTS + age] = f"f{index}"
    return names


# ------------------------------------------------------------------------------ the block

def _mod_scale_shift(hidden, shift, scale, segments):
    """musubi's ``_mod_scale_shift``: the same in-place ops on a fresh norm output."""
    for start, stop, row in segments:
        hidden[:, start:stop].mul_(1.0 + scale[row]).add_(shift[row])
    return hidden


def _gated(update, gate, segments):
    """``update * gate`` per modulation segment. ``residual + _gated(...)`` has the bits of musubi's
    ``_mod_gate(residual, update, gate, segments)``: the same product, then the same add."""
    covered = sum(stop - start for start, stop, _row in segments) == update.shape[1]
    out = torch.empty_like(update) if covered else torch.zeros_like(update)
    for start, stop, row in segments:
        out[:, start:stop] = update[:, start:stop] * gate[row]
    return out


def project_qkv(attn, query_in, key_in, value_in, mode: str = "auto"):
    """``attn.qkv_proj``'s three output thirds, each from its own input.

    ``auto`` slices the weight rows when the Linear is a plain float one or musubi's ConvRot int8 one (its
    activation quantisation is per row, so a row slice of the weight is the same arithmetic). Anything else
    (a LoRA-wrapped Linear, ...) and ``mode="batched"`` runs the whole projection on the three inputs stacked
    and keeps the diagonal thirds: three times the projection's work, but it goes through ``forward``.
    """
    proj, inner = attn.qkv_proj, attn.inner_dim
    inputs = (query_in, key_in, value_in)
    patched = proj.__dict__.get("forward")
    if mode == "auto" and patched is None and type(proj) is nn.Linear and proj.weight.is_floating_point():
        bias = proj.bias
        return tuple(F.linear(x, proj.weight[i * inner:(i + 1) * inner],
                              None if bias is None else bias[i * inner:(i + 1) * inner])
                     for i, x in enumerate(inputs))
    if mode == "auto" and patched is not None and hasattr(proj, "_convrot_groupsize"):
        from musubi_tuner.modules import convrot_int8_utils as convrot

        if getattr(patched, "__func__", None) is convrot.convrot_int8_linear_forward_patch:
            bias = proj.bias
            return tuple(convrot.ConvRotInt8LinearFn.apply(
                x, proj.weight[i * inner:(i + 1) * inner], proj.scale_weight[i * inner:(i + 1) * inner],
                None if bias is None else bias[i * inner:(i + 1) * inner],
                proj._convrot_groupsize, proj._convrot_bwd_mode) for i, x in enumerate(inputs))
    batch = query_in.shape[0]
    out = proj(torch.cat(inputs, dim=0))
    return tuple(out[i * batch:(i + 1) * batch, :, i * inner:(i + 1) * inner] for i in range(3))


def cross_attention(attn, query_in, key_in, value_in, rotation, mode: str = "auto"):
    """musubi's ``Attention.forward`` with the query, key and value projected from three inputs."""
    from musubi_tuner.minimax_h3 import model as h3_model
    from musubi_tuner.modules.attention import AttentionParams, attention

    batch, length, _ = query_in.shape
    query, key, value = project_qkv(attn, query_in, key_in, value_in, mode)
    query = attn.q_norm(query.reshape(batch, length, attn.num_heads, attn.head_dim))
    key = attn.k_norm(key.reshape(batch, length, attn.num_heads, attn.head_dim))
    value = value.reshape(batch, length, attn.num_heads, attn.head_dim)
    if not value.is_contiguous():
        value = value.contiguous()
    if rotation is not None:
        query = h3_model._apply_rope_split_half(query, rotation)       # looked up late: speed_patches swaps it
        key = h3_model._apply_rope_split_half(key, rotation)
    query, key = query.to(value), key.to(value)
    params = AttentionParams.create_attention_params(attn.attn_mode, attn.split_attn)
    return attn.out_proj(attention([query, key, value], attn_params=params))


def dca_block(block, stream, mixes, temb, segments, rotation, qkv_mode: str = "auto"):
    """One H3 ``DiTBlock`` with DCA inputs. Returns the next stream and the block's gated update.

    ``mixes`` is ``[q, k, v, r]``. When ``q``, ``k`` and ``v`` are one object the block's own fused attention
    runs; when ``r`` is ``stream`` the MLP reads the plain mid-stream. With all four equal to ``stream`` this is
    ``DiTBlock.forward``, same operations in the same order.
    """
    query_in, key_in, value_in, mlp_base = mixes
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)
    if query_in is key_in and key_in is value_in:
        attn = block.attn(_mod_scale_shift(block.norm1(query_in), shift_msa, scale_msa, segments), rotation)
    else:
        normed = [_mod_scale_shift(block.norm1(x), shift_msa, scale_msa, segments)
                  for x in (query_in, key_in, value_in)]
        attn = cross_attention(block.attn, *normed, rotation, qkv_mode)
    attn = _gated(attn, gate_msa, segments)
    middle = stream + attn
    mlp_in = middle if mlp_base is stream else mlp_base + attn
    mlp = _gated(block.mlp(_mod_scale_shift(block.norm2(mlp_in), shift_mlp, scale_mlp, segments)), gate_mlp, segments)
    return middle + mlp, attn + mlp


# ------------------------------------------------------------------------------ the stack

def _stack(x, mid, recent) -> list[tuple[int, torch.Tensor]]:
    stack = [(0, x)]
    if mid is not None:
        stack.append((1, mid))
    stack.extend((FIXED_SLOTS + age, entry) for age, entry in enumerate(recent))
    return stack


def _held_bytes(stack) -> int:
    seen, total = set(), 0
    for _slot, entry in stack:
        key = entry.untyped_storage().data_ptr()
        if key not in seen:
            seen.add(key)
            total += entry.untyped_storage().nbytes()
    return total


def run_blocks_dca(dit, dca: DcaH3, hidden: torch.Tensor, temb: torch.Tensor, block_segments, rotation) -> torch.Tensor:
    """``h3_musubi.run_blocks`` with DCA mixes in front of every block and of the final layer.

    Block streaming works as in ``run_blocks``. In training mode with activation checkpointing each block
    (mixes included) is one checkpoint. ``dca.stats`` gets the stack's peak size in bytes and entries.
    """
    device = dit.video_patch_proj.weight.device
    training = any(p.requires_grad for p in dca.parameters()) and torch.is_grad_enabled()
    general = dca.force_general or training
    checkpointed = bool(getattr(dit, "gradient_checkpointing", False) and dit.training and training)
    x, stream, mid, recent = hidden, hidden, None, deque()
    peak_bytes, peak_slots = 0, 0
    for index, block in enumerate(dit.blocks):
        if dit.blocks_to_swap:
            dit.offloader.wait_for_block(index)
        if index not in dca.skip:
            stack = _stack(x, mid, recent)
            peak_bytes, peak_slots = max(peak_bytes, _held_bytes(stack)), max(peak_slots, len(stack))
            grn = dca.blocks[index]

            def step(stream, *entries, block=block, grn=grn, slots=[slot for slot, _ in stack]):
                mixes = grn(stream, list(zip(slots, entries)), general)
                return dca_block(block, stream, mixes, temb, block_segments, rotation, dca.qkv_mode)

            entries = [entry for _slot, entry in stack]
            if checkpointed:
                stream, update = torch.utils.checkpoint.checkpoint(step, stream, *entries, use_reentrant=False)
            else:
                stream, update = step(stream, *entries)
            recent.appendleft(update)
            if len(recent) > dca.k:
                oldest = recent.pop()
                mid = oldest if mid is None else mid + oldest
        if dit.blocks_to_swap:
            dit.offloader.submit_move_blocks_forward(dit.blocks, index)
    stack = _stack(x, mid, recent)
    peak_bytes, peak_slots = max(peak_bytes, _held_bytes(stack)), max(peak_slots, len(stack))
    dca.stats = {"history_bytes": peak_bytes, "history_slots": peak_slots, "general": general,
                 "tokens": hidden.shape[1], "bytes_per_value": hidden.element_size(),
                 "blocks_run": len(dit.blocks) - len([i for i in dca.skip if i < len(dit.blocks)])}
    return dca.out(stream, stack, general)[0].to(device)


def install(dit, dca: DcaH3 | None) -> None:
    """Make ``h3_musubi.run_blocks`` (so ``forward_video``, ``predict`` with a splice, the pi-H3 sampler) run
    ``dit`` through ``dca``; ``None`` takes it off again. Other models are untouched."""
    import h3_musubi

    if dca is None:
        dit.__dict__.pop("_dca", None)
        return
    if len(dca.blocks) != len(dit.blocks):
        raise ValueError(f"DCA built for {len(dca.blocks)} blocks, the DiT has {len(dit.blocks)}")
    dit.__dict__["_dca"] = dca                     # __dict__: not a submodule, so not in the DiT's state dict
    if getattr(h3_musubi.run_blocks, "_dca_patch", False):
        return
    plain = h3_musubi.run_blocks

    def run_blocks(dit, hidden, temb, block_segments, rotation, tap=None):
        attached = dit.__dict__.get("_dca") if hasattr(dit, "__dict__") else None
        if attached is None:
            return plain(dit, hidden, temb, block_segments, rotation, tap=tap)
        if tap is not None:
            raise NotImplementedError("the DWD re-run tail (tap=) is not wired through DCA")
        return run_blocks_dca(dit, attached, hidden, temb, block_segments, rotation)

    run_blocks._dca_patch = True
    run_blocks.plain = plain
    h3_musubi.run_blocks = run_blocks
