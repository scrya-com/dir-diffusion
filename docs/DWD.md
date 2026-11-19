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

## Not done

- Fine-tuning the last K blocks. The generator LoRA is shared by every iteration, so this needs a separate small LoRA that is on only at the final iteration, and a check that it is bit-identical when switched off.
- A real-pixel target from the clip cache.
- A `--dwd` decode path in the renderers.

## Results

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
