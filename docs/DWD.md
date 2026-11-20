# DWD: a linear pixel head on DiR's last-layer tokens

After "Decoding Without a Decoder" (Chen et al.): a linear head on a diffusion transformer's last-layer tokens predicts pixels directly, so inference can skip the VAE decoder. This branch (`feat/dwd`) adds that head to the DiR generator on H3. Status: experimental. See [Results](#results) for what has actually been run.

## What the head reads and writes

- **Input:** `[h, s]` per token.
  - `h` (5376 wide) is the modulated last-layer state that `final_layer.video_out` reads, caught with a forward pre-hook. Only the dense forward has it: LoT never calls `video_out`.
  - `s` (96 wide) is the iteration's input state, packed like the video rows. DiR's estimate is `state + σ·video_out(h)`, so the latent is exactly linear in `[h, s]`. The published checkpoint's last iteration runs at σ = 0.667, so a third of its estimate is the input state.
- **Output:** 12288 values per token: 4 frame slots × 3 × 32 × 32 pixels.
  - One token covers 32×32 pixels (2×2 latent pixels × 16).
  - Latent frame `k` covers 1 pixel frame when `k % 5 == 0` and 4 otherwise. For fitting, a one-frame token's slots 1–3 repeat slot 0; decoding drops them.
  - `test_dwd_h3.py` checks that the mapping round-trips exactly and matches `pack_video_rows`.

## Training

`lot/train_dwd_h3.py` runs in three phases:

1. **`collect` (GPU).** Run DiR rollouts with a trained generator, catch `h` at the final iteration, and decode the final estimate with the VAE (on the second card by default). It writes one `.pt` per sample, about 110 MB for 12×32×48 latents. Held-out prompts use the yardstick's seeds.
2. **`ridge`.** Closed-form least squares on centred features, the paper's initialisation. λ is chosen by PSNR on the held-out probe set. This also measures how linearly decodable the tokens already are.
3. **`sgd`.** Refines the head from the ridge solution with a per-token pseudo-Huber loss, plus optional LPIPS.

The target is the VAE decode of the generator's own estimate. So the head can at best match the VAE more cheaply. Beating the VAE, as the paper reports, needs real pixels and fine-tuned last blocks.

## Ideas taken from nano-diffusion ("From 1000 Steps to 1", liqq010)

- **Pseudo-Huber loss** (iCT, TDM). The per-token gradient is bounded, so no token dominates the update.
- **A fixed probe set instead of the loss value.** Every evaluation scores the same held-out (prompt, seed) pairs by PSNR against the VAE decode and by flicker against the VAE clip's.
- **Not used yet: DMD2's GAN term.** With a VAE target a discriminator would push away from the target; it belongs with a real-pixel target.

## Results

Status thread with pictures (including known-character prompts): [issue #6](https://github.com/scrya-com/dir-diffusion/issues/6).

### Summary (2026-11-20)

Everything is scored against the VAE decode of the same latent, on held-out clips: iso (isometric scenes) and atlas (portraits), 39 frames at 512×768. LPIPS is VGG on every 6th frame.

| Stage | LPIPS (all) | iso PSNR / LPIPS | atlas PSNR / LPIPS |
|---|---|---|---|
| Linear head on base DiR, no LoRA | 0.466 | 21.76 / 0.467 | 25.08 / 0.466 |
| + final-iteration LoRA, K=8, rank 64, MSE, 5 h | 0.365 | 23.49 / 0.381 | 28.70 / 0.348 |
| + head refit on those features (`refit`) | 0.321 | 23.88 / 0.346 | 29.49 / 0.296 |
| + LoRA continued with LPIPS | 0.267 | 23.82 / 0.302 | 29.34 / 0.231 |
| + plain refiner (`refine_dwd_h3.py`) | 0.171 | 24.38 / 0.196 | 30.59 / 0.146 |
| + refiner that also reads the latent (`--latent`) | 0.138 | 25.56 / 0.161 | 31.65 / 0.115 |
| **+ latent context and latent-grid path (`--latent-context --latent-grid 256`), still training** | **0.115** | **27.02 / 0.135** | **33.09 / 0.096** |

**Temporal:**
- Flicker across latent frames: 0.044, against the VAE's ~0.047.
- Error in frame-to-frame change: 0.026.
- Seam ratio: 1.05, against the VAE's ~1.08.

**Known-character prompts** (8 held out; not in the refiner's training data): PSNR 24.57 → 27.74 dB, LPIPS 0.247 → 0.119.

**Speed** (RTX 5090, one 39-frame 512×768 clip, measured with `refine_dwd_h3.py bench`):

| Decoder | Time |
|---|---|
| VAE decode | 1.8 s |
| Linear head | 13 ms fp32, 4 ms fp16 |
| Refiner, `--latent` | 0.34 s fp32, 0.21 s bf16 |
| Refiner, context + grid | 0.40 s fp32, 0.25 s bf16 |

So the whole path (head + refiner, bf16) takes about 0.27 s, roughly 6–7× faster than the VAE. Refiner accuracy in bf16 has not been re-checked.

**LPIPS calibration** (the VAE's own frames after simple degradations):

| Degradation | LPIPS |
|---|---|
| blur σ0.5 | 0.033 |
| half resolution, upscaled | 0.121 |
| blur σ1 | 0.167 |
| blur σ2 | 0.289 |
| quarter resolution, upscaled | 0.295 |

**Recipe**, as run in `train_dwd_h3.py` and `refine_dwd_h3.py`:
1. Run `collect --light`, plus `targets` for the VAE mp4s.
2. Run `ridge --halo 4`.
3. Run `ft` with K=8, rank 64, MSE.
4. Run `refit`.
5. Run `ft` again with `--lpips-weight 0.05`.
6. Run `refine_dwd_h3.py cache`, then `train --latent`, then `train --init <that> --latent-context --latent-grid 256 --ema 0.999 --crop 384`.

**What did not help:**
- Teacher-forced targets from existing H3 clips: +5 dB teacher-forced, −4.6 dB on real rollouts.
- `sgd` on a frozen-feature head.
- More LoRA training after the refit.

**Weights:** [johndpope/dir-diffusion-h3-dwd](https://huggingface.co/johndpope/dir-diffusion-h3-dwd) (v0.1: run B LoRA, ridge head in fp16, run G refiner at step 40416). Run them with the `feat/accel` branch's `lot/accel_h3.py`; see the README.

**Not done:**
- Real-pixel targets: needed to beat the VAE rather than match it.
- A renderer `--dwd` path.
- A bf16 accuracy check.


### Run 1 (2026-11-19)

**Setup**
- **Generator:** the published DiR checkpoint (`johndpope/dir-diffusion-h3`; per-iteration fake model, step 100, averaged weights). 4 iterations, sigmas 1 / 0.947 / 0.857 / 0.667.
- **Data:** 12×32×48 latents (39 frames, 512×768) from the isometric-stills prompts, in clip-shaped noise.
- **Samples:** 384 for training (192 prompts × 2 seeds), 8 held out with the yardstick's seeds.
- **Hardware:** RTX 5090 with no block swap (22.7 GB peak); VAE on the RTX PRO 4000.
- **Time:** collection about 10 s per sample, about 65 min in total.

**Ridge** (PSNR against the VAE decode on the 8 held-out clips):

| λ (relative to the mean feature variance) | PSNR | Flicker (VAE: 0.044) |
|---|---|---|
| 1e-6 | **21.04 dB** | 0.043 |
| 1e-5 | 20.71 | 0.043 |
| 1e-4 | 20.44 | 0.043 |
| 1e-3 | 20.26 | 0.043 |
| 1e-2 | 20.09 | 0.042 |

**SGD:** 3000 steps from the ridge head, pseudo-Huber loss, lr 3e-5, 2048 tokens per step.
- Gradient norm stayed between 0.27 and 0.32 (clip 1.0).
- No non-finite steps.
- Held-out PSNR stayed between 20.98 and 21.02, so the ridge head was kept (`dwd_head.pt` not written).

**Decode time:** the head takes 15 ms per clip; the VAE takes 3.7 s.

**Reading**
- **Recognisable but soft.** The ridge head gives a frame with the right layout, people and colours, but it is soft and shows faint seams on the 32 px token grid.
- **Underfitting, not overfitting.** PSNR falls steadily as λ grows, so a linear head on single tokens has hit its capacity limit.
- **Flicker matches the VAE's.** The head adds no frame-to-frame flicker of its own.
- **SGD adds nothing.** Pseudo-Huber on top of the least-squares optimum gives nothing on PSNR, which is MSE-based.

**What would raise the ceiling (untested)**
- **Context across tokens.** For example, a small conv refiner in pixel space after the linear head, or features from neighbouring tokens.
- **The paper's own lever:** fine-tuning the last few blocks with a pixel loss, behind a LoRA that is on only at the final iteration.
- **Real-pixel targets,** needed to go beyond matching the VAE.

Data: `/media/2TB/lora-data/h3_lot/dwd/collect_v1`. Head, reports and comparison videos: `/media/2TB/lora-data/h3_lot/dwd/run_v1`. Log: `/media/2TB/lora-data/h3_lot/dwd/full_v1.log`.
