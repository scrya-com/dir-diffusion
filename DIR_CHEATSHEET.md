# DiR on MiniMax H3: getting started

> **Research code. One 1,000-step run exists. Its speed is real; its pictures are not done.** Progress, renders and losses: [MiniMax-H3 issue #3](https://github.com/johndpope/MiniMax-H3/issues/3). Explainer: [scrya.com/dir_diffusion](https://scrya.com/dir_diffusion).

**DiR (data-space iteration) is ByteDance Seed's method**: Shanchuan Lin, Yansong Peng, Fu-Yun Wang and Haoqi Fan, ["Few-Step Generation via Data-Space Iteration"](https://arxiv.org/abs/2610.12102), arXiv 2610.12102. MiniMax H3 is MiniMax's model. This repo is an independent port to H3 and is not affiliated with either company.

**What it does.** A few-step generator refines its own clean estimate, `x̂0⁽ʳ⁺¹⁾ = G(x̂0⁽ʳ⁾, z, r)`. It uses the same noise `z` every time, with no schedule and no re-noising. It takes R = 4 iterations instead of 20 sampler steps. It is trained with DMD against the frozen H3 and needs **text prompts only, no clean images**. Each step costs one H3 forward:

| 768×1152 still, RTX PRO 4000 Blackwell 24 GB | DiT time |
|---|---|
| dense H3, 20 steps | 35.8 s |
| DiR, 4 iterations | 6.6 s (5.4×) |
| DiR, 1 iteration | 1.65 s |

Run everything from the repo root. Only one GPU job runs at a time: the scripts refuse to start next to another one and never kill it.

---

## 0. Setup (once)

```bash
pip install -r requirements.txt
hf download Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors --local-dir models
export LOT_H3_CHECKPOINT=$PWD/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
export LOT_H3_VAE=/path/to/minimax_h3_video_vae_fp16.safetensors          # renders and caching
export LOT_H3_TEXT_ENCODER=/path/to/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors   # caching prompts only
```

All paths live in `lot/lot_paths.py`, and each one can be overridden by an environment variable. H3 itself (int8 loader, block swap, LoRA, VAE) comes from the unmodified musubi-tuner subset in `third_party/`.

**Hardware:** one 24 GB NVIDIA card. Training peaks at about 20.5 GB with `--swap 4` (4 blocks streamed from CPU). A Mac (MPS) port is on the `mac` branch; it is untested on real hardware and much slower.

## 1. Check the install

```bash
python3 lot/test_lot.py                 # CPU, no weights: every test prints, last line "ok N"
python3 lot/parity_dir_h3.py            # GPU: the untrained generator equals H3's one-step prediction -> ok=1
python3 lot/parity_stream_h3.py --swap 4   # GPU: streaming forward == stock T2VA / FL2VA, and window timing -> ok=1
```

Do not train if a parity check fails. With zero-initialised new parts, the untrained generator *is* H3, and these tests prove the plumbing.

## 2. Prepare samples

DMD learns from **prompts**. The image is only used for its shape, and as the first frame when you train streaming video. So a training set is a folder of pictures, each with a same-stem caption:

```
my_stills/
  kitchen_001.jpg   kitchen_001.txt   <- "a photorealistic isometric kitchen at dusk, warm light"
  cafe_002.png      cafe_002.txt
  talk_003.mp4      talk_003.txt      <- clips are fine too (24 fps)
```

- **Captions:** write what you want the generator to be good at, in H3's long descriptive style. Variety beats count; run 1 used 801 isometric 3D stills.
- **Stills:** any size. `--size W H` cover-resizes and centre-crops (multiples of 32), and training then takes random crops (`--max-latent-hw`).
- **Clips (optional):** 24 fps. The first `--frames` frames are kept, on the 17n+5 grid (22 → 7 latent frames). To cut one:

```bash
ffmpeg -v error -y -i talk.mp4 -vf "fps=24,scale=768:768:force_original_aspect_ratio=increase,crop=768:768" \
  -frames:v 22 -an -c:v libx264 -crf 16 -pix_fmt yuv420p my_stills/talk_003.mp4
```

## 3. Save latents and encoded prompts (the cache)

```bash
python3 lot/build_cache_h3.py media --src my_stills --out my_cache --size 768 1152   # H3 video VAE -> latents
python3 lot/build_cache_h3.py text  --out my_cache                                   # Qwen3-VL-32B -> prompt states
```

- `media` writes `<stem>_<W>x<H>_minimaxh3.safetensors` (key `latent_<T>x<H>x<W>`) and merges the captions into `captions.json`.
- `text` loads the text encoder **once**, writes `<stem>_minimaxh3_te.safetensors` (`hidden_states`, `attention_mask`) and is resumable. It takes about 9.5 s per caption, so 800 prompts is about 2 h. Run it alone on the GPU.
- Training uses every pair that has both files. A fixed-seed shuffle sets aside `--holdout` items (default 8) for evals and renders.

Already have H3 latents (`z.pt`)? `build_cache_h3.py latents` skips the VAE (see `lot/CHEATSHEET.md`, section 2).

**Sanity check:** `ls my_cache/*_minimaxh3_te.safetensors | wc -l` should equal the number of samples.

## 4. Train the still-image generator

Dry pass first. It runs one generator update and one fake update with gradients, prints the gradient sums and memory, and writes nothing:

```bash
python3 lot/train_dir_h3.py --check --cache my_cache --swap 4 --max-latent-hw 48 48
```

Every line must end `ok=1`. Then train. **Every default is the paper's recipe** (appendix A, Table 4), so only the H3 and budget settings need flags:

```bash
LOT_H3_TRAIN=1 setsid nohup python3 lot/train_dir_h3.py --cache my_cache \
  --swap 4 --max-latent-hw 48 48 --accum 2 \
  --steps 1000 --eval-every 50 --save-every 100 --out lot/runs/my_dir > /tmp/my_dir.log 2>&1 &
```

| Setting | Default (the paper) | Run 1 and Turbo arms A–C (before 2026-10-16) |
|---|---|---|
| Added-parameter init | `--paper-init`: Xavier `x0_proj`, no bias; `r_embed`, `fake_r_embed` ~ N(0, 0.02) | `--zero-init` (step 0 == H3; the estimate pathway starts dead, so r = 4 ≈ r = 1) |
| Iteration trained per step | `--r-schedule mod` (`r = step mod R`) | `random` |
| Adam betas | `--betas 0 0.95` (β1 = 0 helped at every NFE, Table 1c) | `0.9 0.95` |
| EMA | `--ema 0.9999`, with warm-up, saved as `*_ema` | `0.999` (run 1), `0` (Turbo arms) |
| DMD weight | `--dmd-weight paper`: w(τ) = 1, `grad = (x0_fake − x0_real)/τ` | same |
| τ density | `--tau-shift beta:2.5` | same |
| Fake updates per generator update | `--fake-updates 4` | same |
| Gradient clip | `--grad-clip 0` (off) | same |
| Flow time | `--sigmas 1` (t = 1 for every iteration) | same |
| Learning rates | `--lr 1e-4`, `--fake-lr 1e-4` | same |

Not paper settings, chosen for H3 on one card:

| Flag | Value | Why |
|---|---|---|
| `--accum 2` | 2 | samples per update; H3 runs one sample at a time (the paper's batch is 256) |
| `--max-latent-hw 48 48` | on | 768×768 crops keep the step at about 20.5 GB |
| `--rank` | 16 (128 with the Turbo init) | LoRA instead of the paper's full fine-tune |
| `--head-lr` | = `--lr` | optional: a higher learning rate for `x0_proj` / `r_embed` to make up for the smaller budget |

**Resuming or reproducing an older run:** pass its old settings explicitly (from its `config.json`), e.g. run 1 used `--zero-init --r-schedule random --betas 0.9 0.95 --ema 0.999`. `--resume-step` loads the saved head, so the init flag doesn't matter there.

**Cost:** about 68 s per step at rank 16 with `--swap 4` on the 24 GB card (1,000 steps in about 19 h). The rank-128 Turbo arms logged about 240 s per step at 48×48 and about 410 s at 1344×768 (their swap setting was not recorded; `config.json` records it now).

**Watch:**

```bash
grep "kind=dir step" /tmp/my_dir.log | tail -3        # dmd, fake_loss, grad_norm, s_per_step
grep '"eval"' lot/runs/my_dir/log.jsonl | tail -2      # held-out gap to the real model, change per iteration
```

**Pause and resume:** stop the process by PID (never `pkill -f`). Trim `log.jsonl` to steps ≤ N, then rerun the same command with `--resume-step N`. That loads the step-N save. The optimizer restarts, and with β2 = 0.95 it re-warms within about 20 steps.

**Saves** are written to `--out` every `--save-every` steps: `lora_gen_N`, `dir_head_N`, `lora_fake_N`, `dir_fake_head_N`, and with `--ema` also `*_N_ema`.

### Warm start from Minimax-H3-Turbo (experimental)

[LightX2V's Turbo LoRAs](https://github.com/ModelTC/Minimax-H3-Turbo) (Apache-2.0) are far ahead of run 1 ([dir-diffusion#1](https://github.com/scrya-com/dir-diffusion/issues/1)). Converting one gives DiR a much better starting point than base H3's one-step blur:

```bash
hf download lightx2v/Minimax-h3-Turbo minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors --local-dir turbo
python3 lot/turbo_lora.py turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors --rank 128 --out lot/runs/turbo4_init
LOT_H3_TRAIN=1 python3 lot/train_dir_h3.py --init lot/runs/turbo4_init --rank 128 --fake-rank 64 --swap 12 --ema 0 \
  --max-latent-hw 48 48 --accum 2 --steps 1000 --eval-every 50 --save-every 100 --out lot/runs/dir_turbo
```

- **What converts exactly:** out_proj, fc1 and fc2.
- **What is approximated:** Turbo's fused qkv (rank 384) is cut to rank 128 by SVD, which keeps 92–99.9% of its energy per block.
- **What is dropped:** the 2 token-refiner blocks.
- **Memory:** rank 128 on a 24 GB card needs `--swap 12` and `--ema 0`. The `--check` peak is about 20.5 GB before optimizer state. `--fake-rank` keeps the fake model smaller.

### Clip run from Turbo on a 48 GB card (RTX 6000 class): prerequisites and commands

**Status (2026-11-17): no run from Turbo has beaten Turbo yet.** Four 50–125 step runs on a 32 GB card at 2 samples per update are in the README ("Video, part 4"). This section is the recipe those runs point to, at a batch a larger card allows. **None of it has been run on a 48 GB or 96 GB card**: measure first (step 3), and report what you get in [issue #4](https://github.com/scrya-com/dir-diffusion/issues/4).

**1. Prerequisites**

| What | Where it comes from | Needed for |
|---|---|---|
| H3 DiT, pruned int8 (`LOT_H3_CHECKPOINT`) | `hf download Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | everything |
| Turbo 4-step LoRA | `hf download lightx2v/Minimax-h3-Turbo minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | the start point |
| A prompt cache (`--cache`, or `LOT_H3_STILLS_CACHE`) | sections 2–3; the runs here used 801 stills with long captions | training and the 8 held-out prompts |
| H3 text encoder (`LOT_H3_TEXT_ENCODER`) | only to build the cache (about 9.5 s per caption) | caching |
| H3 video VAE (`LOT_H3_VAE`) | `hf download Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_fp16.safetensors` | renders only; training and the yardstick never decode |

Training reads prompts and latent shapes only, so a stills cache is enough for clip-shaped runs (`--frames 12`).

**2. Convert Turbo into a start point** (CPU, a few seconds):

```bash
python3 lot/turbo_lora.py turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors \
  --rank 128 --out lot/runs/turbo4_init --run-dir
```

`--run-dir` adds a zero `dir_head.pt` and a `config.json` with Turbo's sigmas, so the directory is also an untrained run: Turbo's own 4-step sampler, for the yardstick and the renders.

**3. Measure the card before the long run** (about 10 minutes):

```bash
C=(--init lot/runs/turbo4_init --sigmas turbo --zero-init --frames 12 --rank 128 --fake-rank 64 --max-latent-hw 32 48)
LOT_H3_TRAIN=1 python3 lot/train_dir_h3.py $C --swap 0 --check                  # prints the peak; ok=1
LOT_H3_TRAIN=1 python3 lot/train_dir_h3.py $C --swap 0 --fake-warmup 0 --accum 2 --dmd-draws 2 \
  --steps 5 --eval-every 0 --save-every 0 --out lot/runs/timing                 # s_per_step and peak_mb
```

- `--swap 0` keeps all 50 blocks on the card and turns gradient checkpointing off. On a 32 GB card the same shapes peak at 29.8 GB with `--swap 6` (checkpointing on). If `--swap 0` runs out of memory, use `--swap 1`: checkpointing comes back and one block streams from CPU.
- Streaming is what makes a 32 GB card slow: one forward is 0.08 s at `--swap 0` and 0.35 s at `--swap 4`. A step there is 72 s for 2 samples.
- Pick `--accum` from the timing: samples per update = `--accum`. eDMD uses 16, DiR 256; the runs so far used 2.

**4. The run**

```bash
LOT_H3_TRAIN=1 python3 lot/train_dir_h3.py $C --swap 0 \
  --fake-source iteration --tau-max 0.9 --dmd-weight norm --dmd-draws 2 \
  --lr 6e-6 --lr-warmup 100 --grad-clip 3.0 --fake-lr 2e-5 --fake-grad-clip 1.0 --weight-decay 0.01 --ema 0.999 \
  --fake-warmup 100 --accum 8 --steps 400 --eval-every 25 --save-every 50 --eval-anchors --holdout 8 --seed 0 \
  --out lot/runs/dir_turbo_clip
```

Why each choice (measurements in the README, part 4):

- `--fake-source iteration`: the DiR paper's fake model, conditioned on the iteration. eDMD's endpoint fake model (`final`) never sees a first-iteration estimate and inflated the first iteration in two runs.
- `--tau-max 0.9`: H3 is CFG-distilled, and above about 0.9 its one-call answer is dominated by the baked guidance. eDMD's read levels (`--tau-floor`) sit in that range and pushed every run off.
- `--lr 6e-6 --lr-warmup 100 --grad-clip 3.0`, `--fake-lr 2e-5 --fake-grad-clip 1.0`, `--weight-decay 0.01`, `--ema 0.999`: eDMD's optimizer settings. **Untested here**: the runs so far used `--lr 2e-5 --grad-clip 1.0 --ema 0.99`.
- `--accum 8`: the one thing the 32 GB runs could not afford. Raise it if the timing allows.

**5. Watch it.** `kind=dir_eval` prints every 25 steps; `log.jsonl` has one line per step (written in batches of ten).

- Stop if the first iteration inflates: eval `hat_real[0]` at 0.21 or more (0.166 at step 0), or `hat_rms` on `r = 0` steps trending above 1.12.
- `motion` in the eval should fall from 0.44 (final iteration) toward the 20-step sampler's 0.23. In the latest run it reached 0.34 at step 50 and 0.23 at step 75. A falling score is not proof: check the renders for blur and stills.

**6. Score it.** The same-noise distance to a 20-step clip falls whenever an output shrinks, so use the seed-spread yardstick (8 held-out prompts × 4 seeds, latent only):

```bash
Y=(--out lot/runs/yardstick --swap 0)
python3 lot/yardstick_dir_h3.py $Y --name dense --dense
python3 lot/yardstick_dir_h3.py $Y --name turbo --run lot/runs/turbo4_init
python3 lot/yardstick_dir_h3.py $Y --name s200 --run lot/runs/dir_turbo_clip --tag 200 --ema
python3 lot/yardstick_dir_h3.py --out lot/runs/yardstick --report dense turbo s200
```

| 32 GB card, 12-frame 32×48 latents | Motion | Detail | Size (rms) |
|---|---|---|---|
| H3, 20 steps | 0.25 | 0.35 | 1.08 |
| Turbo 4-step | 0.47 | 0.43 | 1.04 |
| Latest run, step 100, averaged weights | 0.26 | 0.45 | 1.00 |

**Score every name on the same card with the same `--swap`.** The same noise gives visibly different clips on two cards (one prompt's Turbo motion: 0.06 on the 5090 at `--swap 4`, 0.50 on a PRO 4000 at `--swap 18`), so paired differences across cards are not paired.

A run is worth publishing when motion is below Turbo's by more than two standard errors, detail and size have not moved further from the 20-step row, and the renders (`probe_dir_h3.py`, commands above) are as sharp and on-prompt as Turbo's by eye. No run meets all three yet.

**Reproduce the latest run exactly** (`dir_turbo_clip_iter`, 2026-11-17, RTX 5090 32 GB, 72 s per step, 29.8 GB peak):

```bash
LOT_H3_TRAIN=1 python3 lot/train_dir_h3.py --init lot/runs/turbo4_init --sigmas turbo --zero-init --frames 12 \
  --fake-source iteration --tau-max 0.9 --dmd-weight norm --grad-clip 1.0 --ema 0.99 --lr 2e-5 \
  --rank 128 --fake-rank 64 --max-latent-hw 32 48 --swap 6 --dmd-draws 4 --fake-warmup 100 --accum 2 \
  --steps 100 --eval-every 25 --save-every 25 --eval-anchors --holdout 8 --seed 0 --out lot/runs/dir_turbo_clip_iter
```

Held-out eval of that run, final iteration: motion 0.44 → 0.37 → 0.34 → 0.23 at steps 0 / 25 / 50 / 75 (20-step H3: 0.23), gap to the teacher's answer 0.117 → 0.129 → 0.133 → 0.133. Same-card yardstick (5090, `--swap 4`): motion 0.45 / 0.43 / 0.35 / 0.22 at steps 25 / 50 / 75 / 100 and 0.26 with step 100's averaged weights (Turbo 0.47, 20-step H3 0.25); detail 0.44–0.46 throughout (Turbo 0.43, H3 0.35). The renders are sharp with less motion than Turbo's, but busier and more tightly framed. Its weights are not published; with the same cache, seed and commit the command above retrains them in about 2.5 hours.

**A 96 GB card** (RTX PRO 6000) runs the same commands. Use the headroom for `--max-latent-hw 48 48` or a larger `--accum`, not a larger rank.

### Why the paper's init matters (r = 4 ≈ r = 1 before)

In every run before run D, r = 4 looked like r = 1 ([dir-diffusion#2](https://github.com/scrya-com/dir-diffusion/issues/2)). The main cause was ours, not the paper's: we initialised the added parameters at **zero** (now `--zero-init`), so that step 0 equals H3 exactly. The previous estimate's only path, `DirHead.x0_proj`, then starts dead and grows at most about one learning rate per Adam step. The paper initialises it like SiT's input projection (Xavier, no bias) and the r embeddings from N(0, 0.02). That's now the default (`--paper-init`), together with the paper's `--r-schedule mod` and `--betas 0 0.95`. The table in section 4 lists every default.

With the Turbo init on 24 GB, rank 128 still needs `--ema 0`.

- **`--head-lr`** (not in the paper) gives `x0_proj` and `r_embed` their own learning rate. It compensates for our budget: LoRA at batch 2 for about 1,000 steps, against the paper's full fine-tune at batch 256 for 40 epochs.
- **`--paper-init` only touches parameters that aren't loaded.** A `--init` run with a `dir_head.pt`, or a `--resume-step`, keeps its saved values. The Turbo init has no head, so it gets the paper init.
- **Expect modest gains even when it works.** The paper's own FID by iteration is 3.85 / 2.68 / 2.47 / 2.46: most of the gain is r = 1 → 2.

### Flow-map baseline (not DiR): `--sigmas` (experimental)

**`--sigmas` below 1 departs from the paper.** It builds states at intermediate flow times, which is the paper's *flow-map* baseline (Table 3: FID 2.55 at 4 NFE, against DiR's 2.46). DiR's point is to remove that schedule. Keep `--sigmas 1` to reproduce the paper, and use a grid only as a comparison or a starting point.

`--sigmas` gives each iteration its own flow time. Iteration r runs on a deterministic Euler state built from the same `z` and its earlier estimates, still with no renoising. With a zero head, the generator at step 0 is its LoRA's own 4-step sampler, so later iterations refine from the start.

| `--sigmas` | σ per iteration (R = 4) | Use with |
|---|---|---|
| `1` (default) | 1, 1, 1, 1 | the paper's DiR, unchanged |
| `grid` / `grid:SHIFT` | 1, 0.973, 0.923, 0.8 (H3's shift 12) | any init: base H3, a fresh LoRA, a pinned-σ run |
| `turbo` / `turbo:SHIFT` | 1, 0.947, 0.857, 0.667 (Turbo's 768p shift 6) | the Turbo warm start |
| `1,0.9,0.7,0.5` | explicit, non-increasing, starting at 1 | anything |

```bash
# step 0, no training: what the schedule does to an init (reuse an earlier render's dense column)
python3 lot/render_dir_h3.py --run lot/runs/turbo4_init_run --sigmas turbo --dense-from lot/runs/render_turbo_s0 \
  --prompts 3 --swap 8 --size 768 1152 --out lot/runs/render_turbo_s0_sig
# train with it (the Turbo warm start above, plus --sigmas)
LOT_H3_TRAIN=1 python3 lot/train_dir_h3.py --init lot/runs/turbo4_init --rank 128 --fake-rank 64 --swap 12 --ema 0 \
  --max-latent-hw 48 48 --accum 2 --sigmas turbo --steps 1000 --eval-every 50 --save-every 100 --out lot/runs/dir_turbo_sig
```

**`--mix-state` (experimental, not in the paper).** With `--sigmas` below 1, iteration r sees `sigma_r * z + (1 - sigma_r) * x0_hat[r]`: the newest estimate re-noised with the same `z`, in place of the Euler state. The iteration is then a function of `(x0_hat[r], z, r)` only, which is DiR's form, and the estimate reaches the model through its pretrained input path. An inference-only test on runs 1 and E showed the trained generators ignore an estimate fed through `x0_proj`; untrained base H3 follows one fed this way. `render_dir_h3.py` reads the setting from the run's `config.json`.

```bash
# the one run that refines so far: base H3 start, 32 GB card (24.9 GB peak), 21.5 s per step
LOT_H3_TRAIN=1 python3 lot/train_dir_h3.py --rank 128 --fake-rank 64 --zero-init --sigmas grid:6 --mix-state \
  --max-latent-hw 48 48 --swap 4 --dmd-draws 4 --fake-warmup 100 --ema 0 --accum 2 \
  --steps 300 --eval-every 20 --save-every 100 --eval-anchors --out lot/runs/dir_mix
# check it: references once, then a checkpoint, then the picture grid
python3 lot/probe_dir_h3.py --out lot/runs/probe --name ref --reference
python3 lot/probe_dir_h3.py --out lot/runs/probe --name step200 --run lot/runs/dir_mix --tag 200
python3 lot/probe_dir_h3.py --out lot/runs/probe --decode step200
python3 lot/bench_dir_h3.py --run lot/runs/dir_mix --swap 0 --canvas 768x768 --canvas 1344x768
```

**Clips: `--frames 12`.** The same recipe on clip-shaped noise (5n+2 latent frames; 12 is 39 pixel frames). Needs a 32 GB card: 28.2 GB peak and 64 s per step at `--max-latent-hw 32 48`. `--eval-frames` sets the eval's clip length, and the eval logs a `motion` score next to the 20-step anchor.

```bash
LOT_H3_TRAIN=1 python3 lot/train_dir_h3.py --frames 12 --rank 128 --fake-rank 64 --zero-init --sigmas grid:6 --mix-state \
  --max-latent-hw 32 48 --swap 4 --dmd-draws 4 --fake-warmup 100 --ema 0 --accum 2 \
  --steps 250 --eval-every 25 --save-every 50 --eval-anchors --out lot/runs/dir_clip
python3 lot/probe_dir_h3.py --out lot/runs/probe_clip --frames 12 --max-latent-hw 32 48 --items 3 --name ref --reference
python3 lot/probe_dir_h3.py --out lot/runs/probe_clip --frames 12 --max-latent-hw 32 48 --items 3 --name s50 --run lot/runs/dir_clip --tag 50
python3 lot/probe_dir_h3.py --out lot/runs/probe_clip --decode s50          # side-by-side mp4s and frame sheets
# a cache from clip latents you already have (z.pt + meta.json with a prompt), then --cache it
python3 lot/build_clip_cache_h3.py --src /path/to/clips --out /path/to/cache_clips
```

`--tau-floor sigma` and `--fake-source final` (after eDMD, arXiv 2611.17194): the first port of the floor pushed a run off the prompt, and the endpoint fake model alone is stable but no closer to the teacher at step 50. The rest of eDMD is behind `--tau-floor TABLE.json` (table from `lot/calibrate_edmd_h3.py`), `--precise-teacher 4`, `--fake-loss-weight edmd`, `--fake-grad-clip`, `--weight-decay`, `--lr-warmup`, `--fake-first`, meant for a Turbo start (`--init <turbo run> --sigmas turbo --zero-init`), not yet evaluated; see the README, "Video, part 3".

Trained result (one run, 300 steps): iteration 4 is sharp on all 8 held-out prompts where base H3 at 4 steps is blurred, and the generator still follows a fed estimate. Checkpoints differ a lot (step 100 collapsed three prompts to a cutaway on white, step 200 did not), so save often and pick by the probe grid. Tables and pictures are in the [README](README.md#results-2026-11-17-what-works-on-h3-and-what-it-costs).

- **Step-0 result** (Turbo init, untrained): the held-out gap goes 0.217, 0.193, 0.175, 0.160 across r = 1 to 4. With pinned σ it's 0.217 at every r. The untrained r = 4 already beats every trained pinned-σ run, and it has Turbo's wide framing.
- **The run records it:** `config.json` stores `sigmas`, and `render_dir_h3.py` reads it, so renders use the run's schedule unless `--sigmas` overrides it.
- **Dense layouts only:** a LoT `--schedule` needs `--sigmas 1`.
- **Cost:** none in memory. DiR iterations rendered at about 7.5 s each, against 3.5 s pinned, at 768×1152. Not yet explained.
- **Training on it (arm C in #2):** by step 10 the per-iteration improvement had shrunk (gap r=1..4 0.247 / 0.250 / 0.237 / 0.211). DMD gives every iteration the same target, so it doesn't preserve the schedule's built-in refinement.
- **Making a run dir for an untrained init:** copy the converted `lora_gen.safetensors` and a training run's `config.json` next to a zero `dir_head.pt`, e.g. `torch.save(DirHead(5376, 8, 4).state_dict(), 'dir_head.pt')`.

### Real-data term: `--priming-weight` (experimental)

After Priming Models ([arXiv 2610.20331](https://arxiv.org/abs/2610.20331); Park, Kim, Du, Chen, Han; Seoul National University and Harvard). DMD only learns from the frozen teacher. This adds their stage-2 loss, `F = D(a, y) − D(a, sg[a'])`, measured in latent space:
- `y` is the cache item's own clean latent, one real sample per prompt.
- `a'` are `--priming-refs` extra rollouts at the same prompt and r, run without gradients.

With one live sample per prompt, the optimal-transport plan is uniform, so the gradient is closed-form:

```
s2         = 0.5 * (mean (a − y)^2 + mean_k mean (a − a'_k)^2)      # the paper's pair-RMS scale
grad_prime = (mean_k a'_k − y) / s2                                 # added as --priming-weight * grad_prime
```

It moves every sample by the same displacement (the generator's mean toward `y`), so it doesn't collapse diversity the way plain regression to `y` would.

```bash
python3 lot/train_dir_h3.py --check ... --priming-weight 0.1     # prints grad_rms, prime_rms, weighted_ratio
LOT_H3_TRAIN=1 python3 lot/train_dir_h3.py ... --priming-weight 0.1 --priming-refs 1
```

- **Choosing the weight:** run `--check` and look at `weighted_ratio`, the weighted priming gradient over the DMD gradient. At weight 1 it's about 4.7, which swamps the teacher; 0.1 gives about 0.5.
- **Paired A/B:** the references draw from their own random generator, so a run with the term sees exactly the same prompts, `z`, r, τ and ε as one without it.
- **Cost:** one extra rollout per sample without gradients (r + 1 forwards), about 8–11% per step. No extra peak memory.
- **Result so far** (20 steps, weight 0.1, pinned σ): wider framing, closer to Turbo 4-step, but one duplicated person, and a held-out gap of 0.229 against the control's 0.202. Not conclusive; details in #2.
- **What's missing vs the paper:** their 16 + 16 samples per prompt, and DINO/SigLIP features. These need more GPU memory, e.g. an RTX 6000.

## 5. Render and compare

```bash
python3 lot/render_dir_h3.py --run lot/runs/my_dir --tag _1000 --prompts 3 --swap 4 --size 768 1152
python3 lot/render_dir_h3.py --run lot/runs/my_dir --tag _1000 --ema --prompts 3 --swap 4 --size 768 1152
```

The output is `grid.png`. Columns are dense 20-step H3, then DiR after 1, 2, 3 and 4 iterations, all from the same seed. `meta.json` holds the DiT milliseconds and how much each iteration changed the image. The prompts are held out (`--holdout` from the run's config).

- **`--sigmas`** overrides the run's iteration schedule (default: the run's `config.json`, else `1`).
- **`--dense-from DIR`** reuses an earlier render's dense 20-step column instead of sampling it again. That's about 73 s per prompt at 768×1152, most of a render. The prompts, seed, canvas and steps must match; the script checks.

## 6. Streaming video (experimental, not yet trained)

One window of 7 latent frames slides by one VAE clip (17 frames) and is never re-noised. Each window's first frame is the previous window's frame 5, fed as H3's own FL2VA condition. See `lot/stream_dir_h3.py`.

```bash
# warm start from a still run: stream 4 clips from a dense first frame -> p*.mp4 + contact sheets
python3 lot/render_stream_h3.py --run lot/runs/my_dir --tag _1000 --clips 4 --prompts 2 --swap 4

# train the stream (windows start from your cache's stills as first frames)
python3 lot/train_stream_dir_h3.py --check --init lot/runs/my_dir --init-tag _1000 --cache my_cache
LOT_H3_TRAIN=1 python3 lot/train_stream_dir_h3.py --init lot/runs/my_dir --init-tag _1000 \
  --cache my_cache --max-latent-hw 32 32 --steps 500 --out lot/runs/my_stream
```

- **Window cost:** a 7×48×48 window forward takes 3.3 s (3.8 s with 16 memory keyframes) at about 20.6 GB.
- **Known limit:** the DMD teachers see the first frame but not the long-range memory, so training doesn't yet reward long-range consistency.

## Gotchas

- **Render the live weights for early checkpoints.** An EMA started without warm-up is mostly untrained weights for hundreds of steps. Warm-up EMA saves (`ema_warmup: true` in `config.json`) are fine.
- **Wait on a PID, not a pattern.** A `while pgrep -f X` loop matches its own command line and never ends. `kill` by PID.
- **Only one GPU job at a time.** Rendering needs the GPU, so pause training first.
- **No LPIPS or GAN loss is used**, on purpose. The paper uses exactly two losses: DMD and the fake model's flow loss.
- **Differences from the paper:** rank-16 LoRAs instead of a full fine-tune, batch 2 instead of 256, and text-to-image instead of class-conditional ImageNet. Runs before 2026-10-16 also used zero init and a random `r`. Details: [issue #3](https://github.com/johndpope/MiniMax-H3/issues/3).
