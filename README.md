# dir-diffusion

Few-step video generation for the **MiniMax H3** DiT: **DiR** (data-space iteration, 4 steps instead of 20) and **DWD** (decoding pixels straight from the transformer instead of the VAE).

These are independent ports to H3 of:
- **DiR:** ByteDance Seed's ["Few-Step Generation via Data-Space Iteration"](https://arxiv.org/abs/2610.12102), by Shanchuan Lin, Yansong Peng, Fu-Yun Wang and Haoqi Fan.
- **DWD:** ["Decoding Without a Decoder"](https://arxiv.org/abs/2611.22287), by Hansheng Chen, Shengqu Cai, Jan Ackermann, Rayhan Zirvi, Gordon Wetzstein, Xinchen Yan and Leonidas Guibas.

The ports are not affiliated with either team or with MiniMax.

| | Time per 39-frame 512×768 clip, RTX 5090 |
|---|---|
| H3, 20 steps + VAE | 16.6 s |
| DiR, 4 iterations + VAE | 5.2 s |
| **DiR + DWD** | **3.7 s** |

Explainers: [scrya.com/dir_diffusion](https://scrya.com/dir_diffusion) and [scrya.com/dwd_diffusion](https://scrya.com/dwd_diffusion).

## News

- **2026-11-21: DWD weights** on Hugging Face: [johndpope/dir-diffusion-h3-dwd](https://huggingface.co/johndpope/dir-diffusion-h3-dwd). Run them with the `feat/accel` branch (below).
- **2026-11-20: DiR + MPEG two-track.** About 2× faster end to end (768×512: 2.7 s against 5.5 s per 39-frame clip), as a different video, not the same one sooner. [`docs/TWO_TRACK.md`](docs/TWO_TRACK.md).
- **2026-11-20: DWD.** LPIPS against the VAE goes 0.466 → 0.115, at 0.27 s per clip against the VAE's 1.8 s. [Results with pictures](https://github.com/scrya-com/dir-diffusion/issues/6) · [`docs/DWD.md`](docs/DWD.md).
- **2026-11-19: Neighbor Forcing + ConvKV memory** for streaming H3. [`docs/SOULX_LIVEACT.md`](docs/SOULX_LIVEACT.md) · [issue #5](https://github.com/scrya-com/dir-diffusion/issues/5).
- **2026-11-17: DiR research checkpoint** on Hugging Face (below).

## The best model

| Model | Weights | What it is |
|---|---|---|
| **DiR generator** | [johndpope/dir-diffusion-h3](https://huggingface.co/johndpope/dir-diffusion-h3) | A rank-128 LoRA + DiR head on H3, 4 iterations. Sharp, with motion at the teacher's level; busier detail than 20-step H3. |
| **DWD decoder** | [johndpope/dir-diffusion-h3-dwd](https://huggingface.co/johndpope/dir-diffusion-h3-dwd) (v0.1) | A final-iteration LoRA, pixel head and refiner. Replaces the VAE decode: LPIPS 0.114 against the VAE, about 6–7× faster. Decodes only the DiR checkpoint above, on a 32 GB card. |

These are research weights: one run each, judged on held-out prompts. Read [`docs/DIR.md`](docs/DIR.md) for what works and what doesn't.

## Honest numbers: against H3 Turbo 4-step

[H3 Turbo](https://github.com/ModelTC/Minimax-H3-Turbo) (LightX2V's 4-step LoRA) is the baseline that matters, because it costs the same 4 transformer forwards as DiR. The table is latent statistics on 8 held-out prompts × 4 seeds; closer to the 20-step teacher is better.

| | Forwards | Transformer time, 12-frame clip, RTX 5090 | Motion | Spatial detail | Size (rms) |
|---|---|---|---|---|---|
| H3, 20 steps (the teacher) | 20 | 14.8 s | 0.25 | 0.35 | 1.08 |
| H3, 4 Euler steps | 4 | 3.0 s | – | – | – (soft, garbled people) |
| **H3 Turbo, 4 steps** | 4 | not timed here (4 forwards, like DiR) | 0.47 | 0.43 | 1.04 |
| **DiR checkpoint** (this repo) | 4 | 3.4–3.6 s | **0.26** | 0.45 | 1.00 |

**Where DiR is better:** motion. It fixes Turbo's over-motion; Turbo moves about twice as much as the teacher.

**Where DiR is not better:**
- **Detail:** slightly busier than Turbo, and both are further from the teacher than we'd like. Edges are 40–70% stronger than Turbo's.
- **Composition:** for the same noise, neither matches the 20-step picture.
- **Speed:** no gain over Turbo; the speed-up is the 4-vs-20 step ratio that Turbo already has.
- **People:** one held-out prompt gains a second person.

**Where the real time saving is:** DWD, at the decoder end (1.8 s VAE → 0.27 s). It currently works only on top of this DiR checkpoint.

## How these models were made, and the vanilla alternative

The published models are **not the paper's main-line training.** They are stacked:

```
H3 (frozen) ─► Turbo 4-step LoRA ─► DiR, DMD from Turbo's weights (100 steps) ─► DWD, final-iteration LoRA + head + refiner
               (LightX2V)            = johndpope/dir-diffusion-h3                (trained on that DiR model's outputs)
```

So changing the generator means retraining DWD. You can also train DiR from scratch:

| Recipe | Command (see [`DIR_CHEATSHEET.md`](DIR_CHEATSHEET.md)) | Pros | Cons |
|---|---|---|---|
| **Vanilla, the paper:** base H3, σ pinned to 1, DMD | `train_dir_h3.py --swap 4 --max-latent-hw 48 48 --accum 2 --steps 1000` (defaults are the paper's) | Faithful to the paper. No Turbo dependency. | **On H3 the iterations do not refine:** the trained generator ignores the estimate it is fed, so it is a one-step generator run four times. The paper used batch 256 and ~50M samples; one card manages batch 2. |
| **Base H3 + mixed state** (not in the paper) | `--rank 128 --fake-rank 64 --zero-init --sigmas grid:6 --mix-state --dmd-draws 4 --fake-warmup 100 --accum 2` | Refines: 4 iterations give sharp images from base H3 in ~200 steps. No Turbo dependency. | Departs from the paper. Unstable between checkpoints (a white-background collapse at step 100). The stills model barely moves on clips, and the clip version was only tested to step 50. |
| **Stacked on Turbo** (the published model) | `--init lot/runs/turbo4_init --sigmas turbo --zero-init --frames 12 --fake-source iteration --tau-max 0.9` | Step 0 is already Turbo-quality. It fixes Turbo's over-motion in 100 steps. The cheapest route to a usable model. | Inherits Turbo's look and adds detail. Depends on Turbo's LoRA (Apache-2.0). One run, batch 2. |
| **DWD** (stacked on the DiR checkpoint) | [`docs/DWD.md`](docs/DWD.md), recipe | Replaces the 1.8 s VAE decode with 0.27 s. | Tied to the generator it was trained on. Its target is the VAE, so it can match the VAE but not beat it. |

## Setup

```bash
git clone https://github.com/scrya-com/dir-diffusion && cd dir-diffusion
pip install -r requirements.txt                       # install torch for your CUDA first
hf download Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors --local-dir models
hf download johndpope/dir-diffusion-h3 --local-dir lot/runs/dir_turbo_clip
export LOT_H3_CHECKPOINT=$PWD/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
export LOT_H3_VAE=/path/to/minimax_h3_video_vae_fp16.safetensors
python3 lot/test_lot.py                               # CPU check, last line "ok N"
```

Every path variable is listed in [`lot/lot_paths.py`](lot/lot_paths.py). A 24 GB card needs block swap (`--swap 12` or more); a 32 GB card runs without it.

## Run it

**ComfyUI:** DiR and DWD nodes are written but not yet tested on a GPU; they will land in [`comfyui/`](comfyui) once they pass.

**From the command line** (prompts come from a cache of encoded captions; see [`DIR_CHEATSHEET.md`](DIR_CHEATSHEET.md), sections 2–3):

```bash
python3 lot/probe_dir_h3.py --out runs/probe --frames 12 --max-latent-hw 32 48 --items 3 --name dir --run lot/runs/dir_turbo_clip
python3 lot/probe_dir_h3.py --out runs/probe --decode dir          # mp4s and frame sheets
bash scripts/bench_two_track.sh                                     # DiR vs two-track timings -> runs/bench_two_track/BENCH.md
```

**DiR + DWD** (32 GB card, no block swap; the end-to-end code is on the `feat/accel` branch until it is merged):

```bash
git checkout feat/accel
hf download johndpope/dir-diffusion-h3-dwd --local-dir dwd
python3 lot/accel_h3.py check --run johndpope/dir-diffusion-h3 --lora dwd/dwd_lora.pt --head dwd/dwd_head.pt --refiner dwd/dwd_refiner.pt
LORA=dwd/dwd_lora.pt HEAD=dwd/dwd_head.pt REFINER=dwd/dwd_refiner.pt SIZES=768x512 bash scripts/bench_accel.sh speed
```

DWD decodes only the published DiR checkpoint's final iteration: with base H3, Turbo or another checkpoint the output is noise.

## Train

| What | Guide |
|---|---|
| DiR generator (DMD distillation) | [`DIR_CHEATSHEET.md`](DIR_CHEATSHEET.md): samples, cache, the recipe that works, renders |
| DWD decoder | [`docs/DWD.md`](docs/DWD.md), recipe section: collect, ridge, final-iteration LoRA, refit, refiner |
| Streaming (experimental) | [`DIR_CHEATSHEET.md`](DIR_CHEATSHEET.md) section 6; Neighbor Forcing in [`docs/SOULX_LIVEACT.md`](docs/SOULX_LIVEACT.md) |

## Docs

- [`docs/DIR.md`](docs/DIR.md): how DiR works on H3, every result (what refines, what doesn't, video, Turbo, eDMD), speed and memory, status.
- [`docs/DWD.md`](docs/DWD.md): decoding without the VAE: design, results, speed, recipe.
- [`docs/TWO_TRACK.md`](docs/TWO_TRACK.md): DiR inside MPEG-Diffusion's two-track sampler: one-command benchmark, results.
- [`docs/SOULX_LIVEACT.md`](docs/SOULX_LIVEACT.md): Neighbor Forcing and ConvKV memory.
- [`docs/FILES.md`](docs/FILES.md): what every script does.
- [`lot/CHEATSHEET.md`](lot/CHEATSHEET.md): Level-of-Token diffusion for H3.

## Weights and license

- **Code** in `lot/` and `comfyui/`: Apache-2.0 ([`LICENSE`](LICENSE)).
- **`third_party/musubi_tuner/`:** Apache-2.0, from musubi-tuner, which adapts it from Hugging Face Diffusers; see [`NOTICE`](NOTICE).
- **MiniMax H3 weights,** and anything trained from them, are under the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE):
  - **not for use in the EU, UK, South Korea or USA;**
  - keep its `LICENSE` and `NOTICE` with any copy.

## Citation

- **Data-space iteration:** Shanchuan Lin, Yansong Peng, Fu-Yun Wang, Haoqi Fan. *Few-Step Generation via Data-Space Iteration*. [arXiv 2610.12102](https://arxiv.org/abs/2610.12102).
- **Decoding without a decoder:** Hansheng Chen, Shengqu Cai, Jan Ackermann, Rayhan Zirvi, Gordon Wetzstein, Xinchen Yan, Leonidas Guibas. *Decoding Without a Decoder: Direct Pixel Prediction from Diffusion Transformers*. [arXiv 2611.22287](https://arxiv.org/abs/2611.22287).
- **MPEG-Diffusion:** Dennis Menn, Tianpei Gu, Guande He, Aryan V S, Yanke Song, Nan Liu, Desai Xie, Xun Huang, Chenfeng Xu, Diana Marculescu. *MPEG-Diffusion: Mixed-Resolution Video Generation*. [arXiv 2611.12069](https://arxiv.org/abs/2611.12069).
- **Level-of-Token Diffusion:** [arXiv 2610.05816](https://arxiv.org/abs/2610.05816).
