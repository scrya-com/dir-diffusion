#!/usr/bin/env python3
"""A trainable forward of pi-H3's DWD decoder (ComfyUI-piFlow's ``SolH3DWDHead``), for a DWD on DiR.

``SolH3DWDHead.refine`` is inference-only: it streams each layer over 5-latent blocks through a ring of in-place
q/k/v slots and runs fused kernels with no backward. ``refine_train`` computes the same thing layer by layer:
every row's q/k/v, then the same shifted spatio-temporal windows (from the decoder's own geometry tables, the
ring slots mapped back to rows), the global rows over their block, then the block's out-projection and
feed-forward. Inside a layer every window reads the layer's input, as the streamed version does, so the two
agree to rounding (``test_pih3_dwd.py``). It runs the decoder's own modules with ComfyUI's ``in_training`` eager
path and an eager rms-rope, and checkpoints each layer.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

PATCH, PATCH_T, OUT_CH, FRAMES_TO_TRIM = 16, 4, 3, 3
COLOR_MEAN = (-0.03, -0.088, -0.188)
COLOR_STD = (0.458, 0.448, 0.45)


@contextlib.contextmanager
def eager():
    """ComfyUI's differentiable path: ``in_training`` (eager linear + act), comfy_kitchen's eager rms-rope, and SDPA
    in place of int8 attention."""
    import comfy.model_management as mm
    from comfy.ldm.minimax import vae as h3vae
    from comfy_kitchen.backends.eager import rope as eager_rope

    def rms_rope(attn, query, key, rotary_pos_emb):
        return eager_rope.rms_rope_split_half(query, key, rotary_pos_emb, attn.qk_norm_scale.to(query.device),
                                              epsilon=attn.norm_q.eps, rot_dim=rotary_pos_emb.shape[-3] * 2)

    def attention(attn, query, key, value):
        # int8 attention has no backward: plain SDPA on the same q/k/v (training only; inference keeps int8)
        return torch.nan_to_num(optimized_attention(query, key, value, attn.heads, skip_reshape=True))

    from comfy.ldm.modules.attention import optimized_attention
    prev = (mm.in_training, h3vae.Attention.rms_rope, h3vae.Attention.attention)
    mm.in_training, h3vae.Attention.rms_rope, h3vae.Attention.attention = True, rms_rope, attention
    try:
        yield
    finally:
        mm.in_training, h3vae.Attention.rms_rope, h3vae.Attention.attention = prev


def _layer(ref, li: int, x: torch.Tensor, plan) -> torch.Tensor:
    from comfy.ldm.modules.attention import optimized_attention
    import comfy.ops

    blk = ref.blocks[li]
    attn = blk.attn
    heads, dh, D = ref.heads, ref.dim_head, ref.dim
    ids, windows, globs = plan["shifts"][li % 2], plan["windows"][li % 2], plan["globs"]
    n = x.shape[0]
    qkv = comfy.ops.linear_input_act(attn.to_qkv, x[None], "rms_norm", act_weight=blk.norm1)
    q, k, v = qkv.view(1, n, heads, 3 * dh).chunk(3, -1)
    q, k = attn.rms_rope(q, k, ref.pos_embed(ids[None].to(x.dtype)))
    q, k, v = q[0], k[0], v[0]                                   # (n, heads, dh)
    idx, outs = [], []
    for rq, rk in windows:                                        # (nw, Lq), (nw, Lk) clip rows
        outs.append(attn.attention(q[rq].transpose(1, 2), k[rk].transpose(1, 2), v[rk].transpose(1, 2)).reshape(-1, D))
        idx.append(rq.reshape(-1))
    for g, (r0, r1) in globs:                                     # global rows read their whole block
        o = optimized_attention(q[g][None].transpose(1, 2), k[r0:r1][None].transpose(1, 2),
                                v[r0:r1][None].transpose(1, 2), heads, skip_reshape=True)[0]
        outs.append(o)
        idx.append(g)
    attn_out = x.new_zeros(n, D).index_put((torch.cat(idx),), torch.cat(outs).to(x.dtype))
    return blk.post_attention(attn_out, x)


def _layer_eager(ref, li, x, plan):
    with eager():                                                 # also during checkpoint recompute
        return _layer(ref, li, x, plan)


def build_plan(ref, geo, device) -> dict:
    """The decoder's geometry with ring slots mapped to clip rows: windows per shift, global rows per block."""
    rows, slot_rows, R = geo["rows"], geo["slot_rows"], ref.ring
    nb = len(rows)
    maps = []
    for b in range(nb):
        m = torch.full((R * slot_rows,), -1, dtype=torch.long)
        for c in range(max(0, b - R + 1), b + 1):
            r0, r1 = rows[c]
            p = (c % R) * slot_rows
            m[p:p + r1 - r0] = torch.arange(r0, r1)
        maps.append(m.to(device))
    shifts, windows = [], []
    for sh in geo["shifts"]:
        shifts.append(sh["ids"].to(device))
        wins = []
        for b in range(nb):
            for qi, ki in sh["windows_at"][b]:
                rq, rk = maps[b][qi.to(device)], maps[b][ki.to(device)]
                assert int(rq.min()) >= 0 and int(rk.min()) >= 0, "a window slot maps to no row"
                wins.append((rq, rk))
        windows.append(wins)
    S = ref.num_global
    globs = [(torch.arange(r1 - S, r1, device=device), (r0, r1)) for r0, r1 in rows]
    return {"shifts": shifts, "windows": windows, "globs": globs}


def refine_train(ref, x: torch.Tensor, geo, latent_shape, checkpoint: bool = True) -> torch.Tensor:
    """``SolH3DWDHead.refine`` without in-place streaming: (1, F, 3, 16H, 16W) pixels in [-1, 1], differentiable."""
    T, H, W = latent_shape
    S, D = ref.num_global, ref.dim
    plan = build_plan(ref, geo, x.device)
    with eager():
        for li in range(len(ref.blocks)):
            if checkpoint and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(_layer_eager, ref, li, x, plan, use_reentrant=False)
            else:
                x = _layer(ref, li, x, plan)
        first_t = PATCH_T - FRAMES_TO_TRIM
        frames = []
        for r0, r1 in geo["rows"]:
            n = (r1 - r0 - S) // (H * W)
            y = ref.norm_out(x[r0:r1 - S]).view(n, H, W, D)
            frames.append(ref.first_frame_proj_out(y[:1]).view(H, W, OUT_CH, first_t, PATCH, PATCH).permute(
                3, 2, 0, 4, 1, 5).reshape(first_t, OUT_CH, H * PATCH, W * PATCH))
            if n > 1:
                frames.append(ref.proj_out(y[1:]).view(n - 1, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(
                    0, 4, 3, 1, 5, 2, 6).reshape((n - 1) * PATCH_T, OUT_CH, H * PATCH, W * PATCH))
    pixels = torch.cat(frames).float()
    mean = torch.tensor(COLOR_MEAN, device=pixels.device).view(-1, 1, 1)
    std = torch.tensor(COLOR_STD, device=pixels.device).view(-1, 1, 1)
    return (pixels * std + mean).clamp(-1, 1)[None]
