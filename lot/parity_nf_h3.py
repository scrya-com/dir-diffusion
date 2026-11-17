#!/usr/bin/env python3
"""Neighbor Forcing / ConvKV plumbing parity on one small clip, on the real H3 (arXiv 2603.11746, ``nf_h3.py``).

Checks, on the pruned int8 checkpoint with block streaming (writes no file, refuses while another LoT
GPU script is alive):

(a) ``nf_forward`` with every row visible equals the stock ``dit(...)`` call: T2VA, and FL2VA-first with a
    reference frame. This is the "step 0 equals the base model" statement, and it holds for mask off only:
    the causal mask is itself a change to the model (the paper's fig. 1).
(b) the block-causal mask changes the output (an O(1) control), and perturbing the last block leaves every
    earlier block's output unchanged.
(c) a zero-initialised generator LoRA under the causal mask equals the causal forward without it.
(d) block-by-block inference with per-step key/value caches (``NFStream``) equals the one-pass masked
    forward, Stage 1 context (every earlier block).
(e) the same with the ConvKV context (mean-initialised and a perturbed convolution), where the third
    block reads 2 reference + 2 long-term + 2 short-term frame slots.
(f) a 4-block, 3-step stream with ConvKV is finite and its cache does not grow after block 1.

(a)-(c) use ``parity_h3.py``'s thresholds: rel_rms <= max(1e-4, 2 x run-to-run floor), max_abs <= 1e-2.
(d)-(e) compare two different batchings of the same arithmetic (a block alone against the block inside the
whole clip): equal in exact arithmetic (``test_lot.py`` on the CPU in fp32, 1e-5) but not guaranteed to the
bf16 bit. Measured: 6.6e-7. ``STREAM_BOUND`` (1e-3) is a ceiling over that, read against the controls
``control_memory`` (the third block under the Stage 1 context instead of ConvKV, 0.14) and ``control_conv``
(a perturbed convolution, 0.08).

    python3 lot/parity_nf_h3.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_CHECKPOINT  # noqa: E402

from gpu_guard import refuse_if_busy  # noqa: E402

CHECKPOINT = LOT_H3_CHECKPOINT
STREAM_BOUND = 1e-3


def rel_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).pow(2).mean().sqrt()
    scale = right.float().pow(2).mean().sqrt().clamp(min=1e-8)
    return float(delta / scale)


def main() -> None:
    refuse_if_busy("parity_nf_h3.py")
    if not CHECKPOINT.is_file():
        raise SystemExit(f"missing checkpoint {CHECKPOINT}")
    from dir_h3 import use_networks
    from h3_musubi import audio_frames, audio_sigma, load_dit, make_lora, predict
    from musubi_tuner.minimax_h3.packing import build_h3_layout
    from nf_h3 import (ConvKV, Geometry, NFConfig, NFStream, NoiseBank, few_step_sigmas, nf_forward,
                       nf_stream_sample, silence)

    model = load_dit(CHECKPOINT, swap=32)
    torch.manual_seed(0)
    frames, height, width = 22, 16, 16
    z = torch.randn(1, 24, frames, height, width, device="cuda", dtype=torch.float32)
    first = torch.randn(1, 24, 1, height, width, device="cuda", dtype=torch.float32)
    text = torch.randn(1, 4, model.config.text_dim, device="cuda", dtype=torch.bfloat16)
    t = 0.6
    # ``predict`` and ``nf_forward`` round t to float32 before the audio sigma; so must the audio made here.
    # (With the double-precision sigma the silent audio differs in its last bit and the 50-block stack turns
    # that into 2.4% of the output: measured on the first run of this script.)
    sigma_a = float(audio_sigma(1.0 - float(torch.as_tensor(t).reshape(-1)[0])))
    audio = silence(audio_frames(frames), sigma_a, "cuda", torch.Generator("cuda").manual_seed(0))
    causal, memory = NFConfig(), NFConfig(memory=True)

    def forward(config=None, **kw):
        with torch.inference_mode():
            return nf_forward(model, kw.pop("latent", z), t, text, audio=kw.pop("audio", audio),
                              **({} if config is None else {"config": config}), **kw)

    with torch.inference_mode():
        base = predict(model, z, t, text, audio_generator=torch.Generator("cuda").manual_seed(0))
        again = predict(model, z, t, text, audio_generator=torch.Generator("cuda").manual_seed(0))
    floor = rel_rms(again, base)
    limit = max(1e-4, 2.0 * floor)

    def within(out, ref):
        return rel_rms(out, ref) <= limit and float((out.float() - ref.float()).abs().max()) <= 1e-2

    # (a) mask off == stock, T2VA and FL2VA-first
    dense = forward()
    gap_a, ok_a = rel_rms(dense, base), within(dense, base)
    layout = build_h3_layout(task="fl2va", text_length=4, target_video=(frames, height, width),
                             target_audio_frames=audio_frames(frames), visual_conditions=[(1, height, width)],
                             condition_roles=("first",))
    with torch.inference_mode():
        stock_first = model(video_latents=z, audio_latents=audio, text_hidden_states=text,
                            text_token_tags=torch.ones(1, 4, dtype=torch.int64), layout=layout, model_t_video=t,
                            model_t_audio=1.0 - sigma_a, visual_condition_latents=[first]).video
    dense_first = forward(first=first)
    gap_a2, ok_a2 = rel_rms(dense_first, stock_first), within(dense_first, stock_first)

    # (b) the mask matters, and nothing flows backwards
    masked = forward(causal)
    moved = rel_rms(masked, dense)
    start = causal.spans(frames)[-1][0]
    shaken = z.clone()
    shaken[:, :, start:] += torch.randn_like(shaken[:, :, start:])
    out = forward(causal, latent=shaken)
    leak = rel_rms(out[:, :, :start], masked[:, :, :start])
    changed = rel_rms(out[:, :, start:], masked[:, :, start:])
    ok_b = moved > 10 * limit and leak <= limit and changed > 10 * limit

    # (d), (e) block-by-block with per-step caches == the one-pass masked forward
    def streamed(config, convkv, ref):
        stream = NFStream(model, lat_h=height, lat_w=width, steps=1, config=config, convkv=convkv, first=ref,
                          total_frames=frames)
        parts = []
        for index, (a, b) in enumerate(config.spans(frames)):
            a0, a1 = Geometry.audio_span(a, b, frames)
            parts.append(stream.velocity(index, 0, z[:, :, a:b], t, text, audio[..., a0:a1]))
        return torch.cat(parts, dim=2)

    gaps, controls, ok_stream = {}, {}, True
    convkv = ConvKV.for_dit(model).cuda()
    rough = ConvKV.for_dit(model).cuda()
    with torch.no_grad():
        for layer in rough.layers:
            layer.memory_proj_k.weight.mul_(1.0 + 0.5 * torch.randn_like(layer.memory_proj_k.weight))
    cases = (("d", causal, None, None), ("d_first", causal, None, first), ("e", memory, convkv, None),
             ("e_first", memory, convkv, first), ("e_rough", memory, rough, None))
    whole_by_case = {}
    for name, config, conv, ref in cases:
        whole = forward(config, convkv=conv, first=ref)
        whole_by_case[name] = whole
        gaps[name] = rel_rms(streamed(config, conv, ref), whole)
    # controls: the third block under a different context is a different function by O(1)
    controls["memory_vs_stage1"] = rel_rms(whole_by_case["e"][:, :, start:], whole_by_case["d"][:, :, start:])
    controls["rough_vs_mean"] = rel_rms(whole_by_case["e_rough"][:, :, start:], whole_by_case["e"][:, :, start:])
    same_early = rel_rms(whole_by_case["e"][:, :, :start], whole_by_case["d"][:, :, :start])
    ok_stream = (all(gap <= STREAM_BOUND for gap in gaps.values()) and same_early <= limit
                 and controls["memory_vs_stage1"] > 4 * max(gaps.values()))

    # (f) a 4-block, 3-step stream with ConvKV
    sizes = []
    torch.cuda.synchronize()
    begun = time.perf_counter()
    stream_out = nf_stream_sample(model, text, 4, lat_h=height, lat_w=width, config=memory, sigmas=few_step_sigmas(),
                                  noise=NoiseBank(0, "cuda"), convkv=convkv, first=first,
                                  on_block=lambda index, _x: sizes.append(index))
    torch.cuda.synchronize()
    stream_ms = 1000 * (time.perf_counter() - begun)
    probe = NFStream(model, lat_h=height, lat_w=width, steps=1, config=memory, convkv=convkv, total_frames=None)
    rows = []
    for index in range(4):
        a, b = memory.span(index)
        a0, a1 = probe.audio_span(index)
        probe.velocity(index, 0, torch.randn(1, 24, b - a, height, width, device="cuda"), t, text,
                       silence(a1 - a0, sigma_a, "cuda"))
        rows.append(probe.cache_rows())
    ok_f = (bool(torch.isfinite(stream_out).all()) and tuple(stream_out.shape) == (1, 24, 30, height, width)
            and rows[1] == rows[2] == rows[3] and sizes == [0, 1, 2, 3])

    # (c) zero-init LoRA under the causal mask (applied last: it wraps the Linears for good)
    network = make_lora(model, 4)
    network.to(device="cuda", dtype=torch.bfloat16)
    with use_networks(network, None, gen=True, fake=False):
        with_lora = forward(causal)
    gap_c, ok_c = rel_rms(with_lora, masked), within(with_lora, masked)

    ok = int(ok_a and ok_a2 and ok_b and ok_c and ok_stream and ok_f and floor <= 1e-3)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(
        "LOT_H3 kind=nf_parity "
        f"frames={frames} latent_hw={height}x{width} blocks={causal.spans(frames)} rel_rms_floor={floor:.6g} "
        f"a_t2va={gap_a:.6g}/{int(ok_a)} a_first={gap_a2:.6g}/{int(ok_a2)} "
        f"b_mask_moved={moved:.6g} b_leak={leak:.6g} b_changed={changed:.6g}/{int(ok_b)} "
        f"c_lora0={gap_c:.6g}/{int(ok_c)} "
        + " ".join(f"{name}={gap:.6g}" for name, gap in gaps.items())
        + f" same_early={same_early:.6g} control_memory={controls['memory_vs_stage1']:.6g} "
        f"control_conv={controls['rough_vs_mean']:.6g} stream_ok={int(ok_stream)} "
        f"f_cache_rows={rows} f_stream_ms={stream_ms:.0f}/{int(ok_f)} "
        f"peak_mb={peak_mb:.1f} dtype=bf16 ok={ok}"
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
