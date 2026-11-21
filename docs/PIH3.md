# pi-H3 in this pipeline

**pi-H3** is the official 2- and 4-step MiniMax H3 with DWD (decoding without a decoder), by **Hansheng Chen, Shengqu Cai, Jiaming Song, Gordon Wetzstein, Xinchen Yan and Leonidas Guibas** (Stanford University; part of the work at Luma AI). They also wrote DWD, with Jan Ackermann and Rayhan Zirvi.
- Blog: [hanshengchen.com/blog/pi-h3](https://hanshengchen.com/blog/pi-h3/)
- Weights: [Lakonik/pi-H3](https://huggingface.co/Lakonik/pi-H3), under the MiniMax H3 Community License
- Code: [LakonLab](https://github.com/Lakonik/LakonLab) and [ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow), Apache-2.0

This branch (`feat/pi-h3`) runs their int8 ConvRot DWD adapters as a **generator** next to DiR, so our benchmarks and our MPEG-Diffusion two-track sampler can use it. The port is ours and is not affiliated with the authors.

**It is the reference to beat.** pi-H3 is faster than our DiR + DWD and needs no generator of ours. Its pixels come from the authors' own DWD head, trained with their generator. See the numbers below.

## How it runs here

| Piece | Where | What it does |
|---|---|---|
| DiT | `lot/h3_musubi.py` (musubi-tuner) | Our int8 ConvRot H3 (`LOT_H3_CHECKPOINT`, the same checkpoint DiR uses) with the adapter merged at load (`pih3_h3.apply_adapter`). |
| Adapter | `Lakonik/pi-H3`, `pi_h3_{2,4}step_dwd_adapter_int8_convrot.safetensors` | A rank-64 LoRA on out_proj/fc1/fc2 of blocks 0–45, merged into the int8 weights. Full int8 out_proj/fc1/fc2 for blocks 46–49. New AdaLN weights in all 50 blocks and a new final layer. Every tensor is consumed, or the load fails. |
| Sampler | `pih3_h3.pih3_sample` | Euler on the pi-Flow schedule: raw times from `max_raw_t` (0.93 for 2 steps, 0.96 for 4), final step 0.6 of the others, shift 12. It starts at `σ0·noise`. Audio is denoised jointly, the way ComfyUI carries it (`carried_audio_step`). |
| DWD head | `pih3_h3.PiH3Head` | The last forward's block output on the video rows. It goes through an affine-free LayerNorm modulated by `dwd_norm_out`, then the Sol refiner (8 Sol-H3-VAE decoder blocks). The refiner is **imported** from ComfyUI-piFlow, on ComfyUI ≥ 0.45's `NonCausalViT3DDecoder`. |
| Two-track | `pih3_h3.pih3_two_track` | MPEG-Diffusion's sampler (`two_track_h3.two_track_forward`), DWD on both tracks. |
| Bench | `accel_h3.py bench --generator pih3`, `scripts/bench_accel.sh` (`GENERATOR=pih3`), `scripts/bench_pih3.sh` | The same table and conventions as DiR's rows (`lot/generators.py` is the switch). |

**Why the head comes from ComfyUI-piFlow and not LakonLab.** LakonLab's head needs `mmcv` and LakonLab's CUDA ops. LakonLab's pipeline also needs the 40 GB bf16 base, which this machine does not have room for. ComfyUI-piFlow's head is the same module, written on ComfyUI's Sol decoder, and it runs on our int8 checkpoint.

**Attention.** Dense PyTorch SDPA in every block, here and in the ComfyUI reference. SageAttention, FlashAttention 3 and Sol are not installed. The blog's numbers use Sol sparse attention and FP8.

## Setup

```bash
# 1. the adapters (2.2 GB each)
hf download Lakonik/pi-H3 pi_h3_2step_dwd_adapter_int8_convrot.safetensors --local-dir /media/2TB/models/pi_h3
hf download Lakonik/pi-H3 pi_h3_4step_dwd_adapter_int8_convrot.safetensors --local-dir /media/2TB/models/pi_h3
export LOT_PIH3_ADAPTERS=/media/2TB/models/pi_h3

# 2. a ComfyUI >= 0.45 tree with ComfyUI-piFlow, for the DWD head. A separate worktree leaves your ComfyUI alone.
git -C /path/to/ComfyUI worktree add --detach /media/2TB/ComfyUI-pih3 v0.46.2
git clone https://github.com/Lakonik/ComfyUI-piFlow /media/2TB/ComfyUI-pih3/custom_nodes/ComfyUI-piFlow
pip install --no-deps --target /media/2TB/ComfyUI-pih3/.pyoverlay comfy-kitchen==0.2.41   # v0.46 pins it
export LOT_PIH3_COMFY=/media/2TB/ComfyUI-pih3
# extra_model_paths.yaml there must name the base checkpoint (diffusion_models) and the adapters (loras);
# only the reference script needs that

python3 lot/test_pih3_h3.py          # CPU, last line "ok 4"
```

`LOT_H3_CHECKPOINT` is the pruned int8 ConvRot FL2VA checkpoint, as for DiR. The VAE (`LOT_H3_VAE`) is only used for the benchmark's VAE row and its scores.

## Run it

```bash
python3 lot/pih3_h3.py sample --steps 2 --sizes 768x512 --prompts 3 --out runs/pih3            # mp4s
python3 lot/pih3_h3.py sample --steps 4 --sizes 768x512 --two-track --switch 2 --out runs/pih3  # + two-track
GENERATOR=pih3 STEPS=2 SWITCH=1 SIZES=768x512 bash scripts/bench_accel.sh speed                  # pi-H3 rows
bash scripts/bench_pih3.sh                                                                         # pi-H3 vs DiR + DWD
```

From Python:

```python
from pih3_h3 import load_pih3, pih3_sample, decode, audio_noise
model = load_pih3(2)                                     # or 4
z = torch.randn(1, 24, latent_t, H // 16, W // 16, device="cuda")
out = pih3_sample(model, text, z, audio_noise(latent_t, seed))     # text: (1, L, 5120) Qwen3-VL layer-50 states
pixels = decode(model, out, tuple(z.shape))             # (1, 3, F, H, W) in [-1, 1]; out["audio"] is the audio latent
```

**Prompts** come from our caches of encoded captions, the same held-out prompts as the DiR rows. Text-only (T2VA); first- and last-frame conditioning are not wired here.

## Parity with the authors' ComfyUI path

`lot/pih3_comfy_reference.py` runs the authors' `pi-H3-DWD.json` workflow headless, with dense attention: ComfyUI v0.46.2 and ComfyUI-piFlow 1.4.0 on the same int8 checkpoint. It takes the same cached text and dumps every tensor: noise, sigmas, each DiT call's inputs and outputs, the last block output, and the DWD pixels. `pih3_h3.py parity` then compares in layers:

```bash
python3 lot/pih3_comfy_reference.py --steps 2 4 --size 768x512 --frames 39 --prompts 3 --out REF
python3 lot/pih3_h3.py parity --steps 2 --ref REF/768x512_f39_s2
```

Three held-out prompts per row, 768×512, 39 frames. Seeds 0, 1 and 2, ComfyUI's own noise.

| | Head on ComfyUI's tokens, dB | Video rel. error per DiT call | Ours vs ComfyUI, PSNR / LPIPS | Floor: ours vs ours (bf16 noise), PSNR / LPIPS |
|---|---|---|---|---|
| 2-step, p0 | 58.9 | 21% → 8% | 18.3 / 0.249 | 19.4 / 0.234 |
| 2-step, p1 | 64.6 | 3% → 3% | 35.2 / 0.044 | 40.0 / 0.038 |
| 2-step, p2 | 58.7 | 12% → 8% | 21.3 / 0.295 | 19.9 / 0.297 |
| 4-step, p0 | 59.3 | 14% → 3% | 19.4 / 0.278 | 19.9 / 0.273 |
| 4-step, p1 | 63.6 | 4% → 2% | 38.2 / 0.074 | 24.1 / 0.104 |
| 4-step, p2 | 59.1 | 10% → 3% | 22.9 / 0.222 | 22.1 / 0.247 |

Base model without the adapter, same check: 3–5% per call. Audio: 2–6% per call.

- **The head is exact.** Fed ComfyUI's own block output and timestep embedding, our head gives ComfyUI's pixels to 59–65 dB.
- **The weights are ComfyUI's.** ComfyUI merges the LoRA into fp16, then requantizes to int8 with per-key stochastic rounding. `--merge comfy` (the default) does the same. Every layer compared matched ComfyUI's patched weights to the bf16 floor of the dump (0.16–0.17% relative, the same as an untouched layer).
  - `--merge exact` merges in fp32 and rounds deterministically. It lands 1.3% away per layer. That is ordinary int8 requantization noise, but it is as large as the LoRA's whole effect (1.7%): merging any LoRA into int8 is lossy at this scale.
- **The forward differs by 3–5% per call on the base model.** That is two W8A8 implementations: musubi's triton kernels against comfy-kitchen's. pi-H3's first step (an x0 jump from almost pure noise) amplifies it to 3–21%.
- **End to end the two clips sit at the DiT's chaos floor.** Ours against ComfyUI is as far apart as ours against itself from the same noise rounded to bf16 (the "floor" columns). Same scene, layout and motion; different fine detail.

## Results (RTX 5090, dense attention, medians of the warm prompts)

All rows: 39 frames unless stated, 3 held-out prompts, the first a warm-up. DiR rows were re-measured in the same session with the v0.2 DWD stack.

| Canvas, frames | Generator | Total s | DiT s | Decode s | × real time | LPIPS vs VAE, same latent | Peak GB |
|---|---|---|---|---|---|---|---|
| 768×512, 39 | DiR + VAE | 5.42 | 3.64 | 1.77 | 0.30 | | 27.2 |
| 768×512, 39 | DiR + our DWD v0.2 | 4.12 | 3.86 | 0.26 | 0.39 | 0.112 | 24.9 |
| 768×512, 39 | **pi-H3 2-step + its DWD** | **1.55** | 1.47 | 0.07 | **1.05** | (0.396, see below) | 21.5 |
| 768×512, 39 | **pi-H3 4-step + its DWD** | **3.06** | 2.98 | 0.08 | 0.53 | **0.067** | 21.5 |
| 768×512, 39 | DiR two-track, DWD on both tracks | 1.67 | 1.45 | 0.21 | 0.97 | 0.206 | 24.7 |
| 768×512, 39 | pi-H3 2-step two-track, DWD on both (46% region) | 0.80 | 0.72 | 0.07 | 2.03 | | 21.4 |
| 768×512, 39 | pi-H3 4-step two-track, DWD on both (40% region) | 1.46 | 1.39 | 0.07 | 1.11 | 0.270 | 21.3 |
| 768×512, 39 | DiR half res + DWD, ×2 | 0.98 | 0.92 | 0.06 | 1.66 | 0.200 | 23.6 |
| 768×512, 39 | pi-H3 2-step half res + DWD, ×2 | 0.36 | 0.33 | 0.03 | 4.51 | | 21.0 |
| 768×512, 39 | pi-H3 4-step half res + DWD, ×2 | 0.73 | 0.69 | 0.03 | 2.23 | 0.140 | 21.0 |
| 1344×768, 39 | DiR + VAE | 16.56 | 12.37 | 4.19 | 0.10 | | 29.4 |
| 1344×768, 39 | DiR + our DWD v0.2 | 13.73 | 13.05 | 0.68 | 0.12 | 0.114 | 28.4 |
| 1344×768, 39 | **pi-H3 2-step + its DWD** | **5.45** | 5.25 | 0.20 | 0.30 | | 23.4 |
| 1344×768, 39 | **pi-H3 4-step + its DWD** | **10.76** | 10.56 | 0.20 | 0.15 | **0.087** | 23.4 |
| 1344×768, 39 | DiR two-track, DWD on both | 4.47 | 3.99 | 0.48 | 0.36 | 0.173 | 26.4 |
| 1344×768, 39 | pi-H3 2-step two-track (52% region) | 2.63 | 2.47 | 0.16 | 0.62 | | 23.3 |
| 1344×768, 39 | pi-H3 4-step two-track (29% region) | 3.86 | 3.75 | 0.11 | 0.42 | 0.311 | 22.8 |
| 1344×768, 39 | pi-H3 2-step half res + DWD, ×2 | 0.96 | 0.91 | 0.05 | 1.69 | | 22.2 |
| 1344×768, 39 | pi-H3 4-step half res + DWD, ×2 | 1.91 | 1.86 | 0.05 | 0.85 | 0.111 | 22.2 |
| 1344×768, **124** (5.2 s) | DiR + our DWD v0.2 | out of memory (the DWD tail needs every block resident) | | | | | |
| 1344×768, **124** (5.2 s) | **pi-H3 2-step + its DWD** | **30.95** | 30.32 | 0.62 | 0.17 | | 27.0 |
| 1344×768, **124** (5.2 s) | **pi-H3 4-step + its DWD** | **61.42** | 60.79 | 0.62 | 0.08 | | 27.0 |

Every row with PSNR and the VAE rows: [`pih3_bench_raw.md`](pih3_bench_raw.md).

How to read it:
- **Total** = DiT + decode, pixels left on the GPU, no text encoder, no audio decode, in every row. **× real time** = clip seconds at 24 fps / total.
- **LPIPS vs VAE** compares the DWD pixels with the VAE decode of the same run's final latent.
  - At **4 steps**, pi-H3's DWD agrees with the VAE better than ours does: 0.067 / 0.087 against our 0.112 / 0.114.
  - At **2 steps** (blank above, 0.40 in the raw table) it is not a DWD error. The last 2-step latent is one jump from σ 0.87, and its VAE decode is blocky; the DWD pixels are clean. The non-DWD adapters exist for VAE decoding.
  - For two-track rows the reference is the VAE on the composite latent.
- **The 124-frame rows are 15–20× the 39-frame rows for 3.2× the frames.** Dense attention is quadratic in 37k tokens; the blog uses Sol sparse attention here.
- **The ComfyUI reference** (the authors' path, same card, dense) takes **1.37 s** for 2 steps and **2.64 s** for 4 steps at 768×512 × 39 frames, DWD included. Ours is 13% (2-step) and 16% (4-step) slower: musubi's int8 kernels against comfy-kitchen's. The ComfyUI path is the faster way to run pi-H3 on this card today.

### The authors' numbers (theirs, pi-H3 0.9, not measured here)

From the blog. B200, 1344×768, 5 s with audio, timed from text encoding to decoded video and audio. Sol attention, FP8 W8A8:
- pi-H3 2-step: **3.81 s**; 4-step: **7.30 s**
- Sol-H3 4-step: 9.48 s

H100, 768p, 5 s, sums of per-stage medians:

| Model | Time |
|---|---|
| pi-H3 2-step DWD | 6.19 s |
| PDMD 2-step + H3 VAE | 9.87 s |
| pi-H3 4-step DWD | 12.02 s |
| Sol-H3 4-step + Sol-H3-VAE | 12.61 s |
| LightX2V 4-step + LightVAE-Lite | 12.99 s |

Quality claims (visual galleries, no metrics):
- 4-step: the best overall among the 4–8-step H3 models compared.
- 2-step: better than TaoMate-H3 (3 steps) and PDMD (2 steps), with better audio.

pi-H3 1.0 (Nov 21), with the transformer refiner used here, is said to be slightly faster; it has no published numbers.

## Two-track (MPEG-Diffusion) on pi-H3

`pih3_two_track`, DWD on both tracks with pi-H3's own head and refiner; nothing is trained:
1. The first `switch` steps run the whole clip at half resolution (2-step: 1; 4-step: 2).
2. MPEG-Diffusion's region head (trained on base H3's states) picks a region. Its bounding box, over the whole clip, becomes the HR track: the LR estimate ×2, re-noised.
3. The remaining steps run one mixed sequence with the K/V maps.
4. Pixels: the LR track's DWD pixels ×2, with the box's DWD pixels pasted in, feathered over 16 px.

The box is a rectangle because the Sol refiner needs a full token grid of real HR tokens. Upsampled LR tokens decode as tiled copies.

| | 768×512 | 1344×768 |
|---|---|---|
| pi-H3 2-step, full res | 1.55 s | 5.45 s |
| pi-H3 2-step two-track | **0.80 s** (1.9×) | **2.63 s** (2.1×) |
| pi-H3 4-step, full res | 3.06 s | 10.76 s |
| pi-H3 4-step two-track | **1.46 s** (2.1×) | **3.86 s** (2.8×) |
| LPIPS two-track vs full res, same seed: whole frame (2-step / 4-step) | 0.45 / 0.48 | 0.69 / 0.68 |
| LPIPS two-track vs full res, same seed: the HR box only (2-step / 4-step) | 0.49 / 0.49 | 0.73 / 0.74 |

Medians of the two warm prompts.

**It is a different video, not the same one faster**, as with DiR ([`TWO_TRACK.md`](TWO_TRACK.md)). The LR track starts from other noise, so 0.45–0.74 LPIPS against full resolution is mostly content, and the box scores no closer than the whole frame.

By eye (`p1_*.mp4` in the bench folders):
- **2 steps (one mixed step):** the box disagrees with the LR track around it. There is a ghost silhouette at the box edge and a different outfit.
- **4 steps (two mixed steps):** coherent, but with an odd detail on the head. The half-resolution ×2 row (4-step: 0.73 s at 768×512, LPIPS 0.14 against its own VAE) is cleaner and faster.

**What would make it work:** the region head and the K/V maps were fitted on base H3, not pi-H3, and pi-H3 was not distilled for mixed sequences. Our DiR two-track had the same untrained gap.

## Files

| Path | What |
|---|---|
| [`lot/pih3_h3.py`](../lot/pih3_h3.py) | Adapter merge, sampler, DWD head, two-track, `parity`, `sample`, the bench rows |
| [`lot/pih3_comfy_reference.py`](../lot/pih3_comfy_reference.py) | The authors' ComfyUI workflow, run headless, with tensor dumps |
| [`lot/generators.py`](../lot/generators.py) | `--generator dir\|pih3`: the plug-in seam of the benchmarks |
| [`lot/test_pih3_h3.py`](../lot/test_pih3_h3.py) | CPU tests: schedule, carried audio step, box paste, registry |
| [`scripts/bench_pih3.sh`](../scripts/bench_pih3.sh) | pi-H3 2/4-step and two-track against DiR + DWD v0.2, one table |
| `lot/h3_musubi.py`, `lot/two_track_h3.py` | `audio=`, `t_audio=`, `out=` on the forwards (a joint sampler's audio in; audio velocity, block output, timestep embedding out). Defaults change nothing. |

## Citation

```
@misc{chen2026pih3,
  title        = {{pi-H3: Video Generation in 2–4 Steps Without a VAE Decoder}},
  author       = {Hansheng Chen and Shengqu Cai and Jiaming Song and Gordon Wetzstein and Xinchen Yan and Leonidas Guibas},
  year         = {2026},
  howpublished = {\url{https://hanshengchen.com/blog/pi-h3/}}
}
```

DWD: Hansheng Chen, Shengqu Cai, Jan Ackermann, Rayhan Zirvi, Gordon Wetzstein, Xinchen Yan, Leonidas Guibas, *Decoding Without a Decoder: Direct Pixel Prediction from Diffusion Transformers*, [arXiv 2611.22287](https://arxiv.org/abs/2611.22287). MPEG-Diffusion: Dennis Menn, Tianpei Gu, Guande He, Aryan V S, Yanke Song, Nan Liu, Desai Xie, Xun Huang, Chenfeng Xu, Diana Marculescu, [arXiv 2611.12069](https://arxiv.org/abs/2611.12069).
