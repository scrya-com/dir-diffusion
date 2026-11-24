# PMD on the pi-H3 2-step fast path

> **Result: gate FAILED at step 250 (detail 0.47 of the teacher, contrast 0.82); the run was stopped there.** Code and CPU tests done (`CUDA_VISIBLE_DEVICES="" python3 lot/test_pmd_pih3.py`, 15 tests; `lot/test_pmd_h3.py`, 16 tests). GPU results are in section 7. **PMD + the paper's GAN term is wired and CPU-tested but has never run on a GPU: it is a help-wanted experiment, section 8.**

**Credit.**
- **PMD (Posterior Matching Distillation)** is the method of Hidir Yesiltepe, Julius Berner, Arash Vahdat and Pinar Yanardag (arXiv 2611.26629). Their code is not released. `lot/pmd_h3.py` is our independent port (see `docs/PMD.md`: the method, the Tilt=Shift identity, ρ for rectified flow, every guess). An author confirmed the coarse-only tilt choice in the repo's issue #1.
- **pi-H3** is by Hansheng Chen, Shengqu Cai, Jiaming Song, Gordon Wetzstein, Xinchen Yan and Leonidas Guibas (`docs/PIH3.md`).
- **MiniMax H3** is the base model. **H3-Tiny** (github.com/fuyi25/H3-Tiny) is the 8-block depth plan used by `--drop h3tiny`.
- The fast path, the latent ×2 and this combination are ours, and are not affiliated with any of the authors.

Files: `lot/pmd_pih3.py` (trainer, sampler, eval), `lot/test_pmd_pih3.py` (CPU tests). `lot/pih3_distill.py` is imported, not changed.

## 1. What is tested

The fast path (`pih3_distill.fast_path`, `docs/PIH3_TRAIN.md`) is:

1. one pi-H3 step at half resolution from `2·avgpool2x2(z)`;
2. the learned latent ×2 (`latent_up_h3`);
3. re-noise with `z` to σ1 = 0.8653;
4. one pi-H3 step at full resolution;
5. the DWD head.

Untrained it is soft (detail 0.82 of the pi-H3 4-step teacher, LPIPS 0.473). Regressing its latent onto the 4-step teacher's blurred it (detail 0.41–0.52): the 4-step teacher is a chaotic function of the noise, and a regression averages over what the student cannot predict. PMD is tried here as the objective instead, next to the DMD run (`pih3_distill.py dmd`). It needs no teacher clips, only one teacher call per tilt, and it matches a distribution, not one mean.

## 2. The method on the fast path

PMD trains the **full-resolution step only**.

| Piece | What it is here |
|---|---|
| State `z_t` | The upsampled half-resolution estimate re-noised with `z` to σ1. On-policy (made by the fast path itself), no gradients, detached. |
| Pure-noise step | The half-resolution step. No PMD loss on it, as in the paper (their FLUX.2 setting has none at pure noise; ρ is infinite at σ = 1). |
| Student | pi-H3 (adapter on) + a trainable bypass LoRA (rank 32, all blocks) + the zero-init `NoiseInput`. It reads `z_t` mixed with fresh noise `u` at the noise dial (`pmd_h3.mix_noise`) at the mixed level σ̃, and the raw `z_t` through `NoiseInput`. `x̂ = y + σ̃·v`. With `--drop h3tiny` it runs 42 of the 50 blocks. |
| Teacher | Base H3: the same DiT with the pi-H3 adapter switched **off**, the LoRA off, no `NoiseInput`, all 50 blocks, at σ1. One call per tilt at `z_t + ρλ`, with `ρ = σ1²/(1 − σ1) = 5.56`. |
| Loss | `pmd_h3.pmd_step`: K = 4 coarse tilts, M = 4 student draws shared by all tilts, Algorithm 2 (jackknife, cross-fitted), gradients by cached-draw replay (one student graph alive at a time). |

One DiT is in memory. `pih3_distill.BypassAdapter` holds the pi-H3 adapter unmerged, so it can be switched off for the teacher. `FastPMD` keeps three switches (dropped blocks, LoRA, adapter) and sets them together:

| Call | Dropped blocks | LoRA | pi-H3 adapter |
|---|---|---|---|
| Half-resolution step | yes | off | on |
| Student (full resolution) | yes | on | on |
| Teacher | no | off | off |

**The backward runs inside the student's switches.** H3 trains with activation checkpointing, which recomputes each block during backward with whatever switches are set at that moment. `pmd_update` holds `student_mode()` across the rollout, the teacher calls (which flip the switches and restore them), the draws, and every replay's backward. `pih3_distill`'s first regression runs had their backward outside the drop context; this code does not repeat that. Two tests cover it: `test_update_gradients_with_checkpointing_and_dropped_blocks` (the update's gradients equal plain autograd without checkpointing) and `test_backward_under_other_switches_is_wrong` (a backward under the teacher's switches raises or gives other gradients).

## 3. Choices

**The half-resolution step stays frozen.** No LoRA in it, no gradients through it or through the upsampler. Reasons:
- PMD's loss is defined on a state at a noise level with a finite ρ. The half-resolution step starts from pure noise, where the paper applies no loss. Training it would need a second objective (a regression or a gradient through the re-noising), which is not PMD.
- With the LoRA off in that step, `z_t` does not move while the LoRA trains. The state distribution is fixed, so the test isolates one question: does PMD sharpen the full-resolution step without changing the scene?
- It is cheaper: no half-resolution graph, and the LoRA costs nothing in that step at inference.

`--lora-lr-step` puts the LoRA in the half-resolution step too (still without gradients, as in the paper's on-policy rollout). It is not the default, because the LoRA would then change a step it gets no training signal for.

**The teacher is base H3 with the adapter off.** That is the closest thing to the paper's teacher (the undistilled base model, guidance 1, one evaluation per tilt). **It is CFG-distilled**: H3's published checkpoint has guidance baked into its one-call output, so it is not an exact posterior-mean denoiser, and Tilt=Shift holds only approximately. `--teacher pih3` keeps the adapter on instead (a 2-step policy model's one-call output is even further from a posterior mean; it is there for comparison only).

**The teacher does not drop blocks.** The dropped blocks are the student's architecture, not the teacher's.

**Prompts, not clips.** Training draws a prompt from the first 500 training prompts (the same set the teacher clips were made from; the 16 held-out prompts are excluded) and a fresh seed each update.

## 4. Guesses (each is a flag)

Everything in `docs/PMD.md` section 3 still applies. New or changed here:

1. **Noise dial 0.30** for the full-resolution step (`--dial`). The paper's FLUX.2 table is {0, 0.30, 0.45, 0.15} over four levels; 0.30 is its first level after pure noise, which is where σ1 sits. It gives σ̃ = 0.878 against σ1 = 0.865.
2. **pi-H3 at the mixed level.** The student runs the pi-H3 adapter at σ̃, a level its 2-step schedule never visits. Step 0 is therefore not the untrained fast path: it is the untrained fast path on the mixed latent. The eval reports both (`baseline` and step 0).
3. **Audio.** H3's audio rows are carried through the fast path as `fast_path` carries them. The student and the teacher get the same carried audio at σ1's audio level; it is not mixed with noise.
4. **Tilt grid** (4, 8, 8) in (T, H, W): 3 × 4 × 6 cells on the 12 × 32 × 48 latent, unit L2 norm over the whole latent (`--tilt-factor`).
   - **Tilt radius U[1, 2] in the run, 4× the paper's U[0.25, 0.5]** (`--radius 1.0 2.0`; the code's default is still the paper's). Chosen from the probe in section 7: it is the smallest of the tried radii where the teacher's response to the tilt rises above its noise floor while the softmax weights are neither flat nor one-hot. U[1, 2] is also the paper's own larger-radius ablation row.
5. **Learning rate 2e-5**, warm-up 50 updates, AdamW (0.9, 0.999), gradient clip 10, batch 1.
6. **Inference noise.** `u` is one fresh Gaussian per clip. The held-out eval fixes it by the clip's seed.
7. **Trained on the unmerged adapter, deployed on the merged int8 one.** `eval --adapter merged` measures the deployment form.

## 5. Commands

```bash
cd /home/johndpope/Documents/GitHub/dir-diffusion-pmd-pih3/lot
export CUDA_VISIBLE_DEVICES=GPU-022a724e-9bc8-f221-8451-ea58c21a8235      # the RTX 5090 only
D=/media/2TB/lora-data/h3_lot/pih3/distill

CUDA_VISIBLE_DEVICES="" python3 test_pmd_pih3.py                          # CPU tests
python3 pmd_pih3.py train --data $D --out $D/pmd_check --drop h3tiny --check          # 3 updates: s/step, peak GB
python3 pmd_pih3.py probe --data $D --out $D/pmd_probe --drop h3tiny --scales 0.25 1 4 16   # tilt radius, no training
python3 pmd_pih3.py train --data $D --out $D/pmd_a --drop h3tiny --steps 1000 --radius 1.0 2.0 \
    --stop-at 500 --stop-below-hf 0.9                                     # eval at 0, 250, 500, ...; sheets
python3 pmd_pih3.py eval --data $D --out $D/pmd_a/eval --drop h3tiny --adapter merged \
    --ckpt $D/pmd_a/pmd_250.pt $D/pmd_a/pmd_500.pt                        # the gate table, 16 clips, timing
```

The trainer refuses to start when more than 4 GB is already in use on its card (`pih3_cascade.refuse_if_card_busy`).

**Log lines** (`kind=pmd_pih3`): `loss` (Algorithm 2, signed), `loss_plain` (Algorithm 1), `ess` (softmax effective sample size, 1 to 4), `w_max`, `shift_ratio` (‖ρλ‖/‖z_t‖), `target_spread` (how much the tilted teacher targets differ), `draw_spread` (draw-to-draw spread over the mean), `grad`, `s_per_step`, `peak_gb`.

**Eval lines** (`kind=pmd_pih3_eval`), on the held-out teacher clips, computed as `pih3_distill.held_eval` computes them: `lpips` (whole clip against the pi-H3 4-step DWD pixels), `hf` (detail as a fraction of the teacher's), `lat` (latent MSE), `mean_diff` and `std_ratio` (colour and contrast against the teacher, `pih3_distill`'s formulas), `div` (relative RMS distance between two draws `u` for the same `z`), `seconds` (median, end to end with the head, in the training process). Step 0 also has `baseline`: LoRA off, no mixing, which is `fast_path` itself (`test_baseline_sample_is_the_fast_path`).

## 6. Gate

At 768 × 512, 39 frames, with `--drop h3tiny`:

- ≤ 1.0 s end to end;
- detail ≥ 0.95 of the teacher;
- LPIPS below the untrained 0.473;
- the same scene by eye.

The baseline to beat is pi-H3 2-step at full resolution with the 8 blocks dropped: 1.31 s, sharp.

## 7. Results

All on one RTX 5090, 768 × 512, 39 frames, `--drop h3tiny`, rank 32, K = 4, M = 4, dial 0.30. Files are in `/media/2TB/lora-data/h3_lot/pih3/distill/` (`pmd_check/`, `pmd_probe/`, `pmd_a/`, `pmd_a.log`).

### 7.1 `--check`

24.0 s per update, peak 28.5 GB, `ok=1`.

### 7.2 Tilt-radius probe (no training)

3 held-out states, the untrained student, the same tilt directions and the same 4 draws at every radius. Means:

| Radius | ESS (of 4) | `w_max` | Teacher shift `‖D(z+ρλ) − D(z)‖ / ‖D(z)‖` | Teacher std ratio | Tilt part / mean residual |
|---|---|---|---|---|---|
| ×0.25 | 3.97 | 0.29 | 0.026 | 1.000 | 0.05 |
| ×1 (the paper's U[0.25, 0.5]) | 3.58 | 0.41 | 0.027 | 1.000 | 0.05 |
| ×4 (U[1, 2]) | 2.18 | 0.82 | 0.047 | 1.000 | 0.09 |
| ×16 | 1.36 | 1.00 | 0.129 | 0.999 | 0.25 |

- **The teacher's response does not scale below ×1.** The shift is 0.026–0.027 at both ×0.25 and ×1. That is a floor: we take it to be the int8 / bf16 teacher's sensitivity to any small input change (not confirmed by running the teacher twice on one input). At the paper's radius the tilt part of the target is therefore noise.
- **×4 is the first radius above the floor** with weights that are neither flat nor one-hot. The teacher stays in range at every radius (std ratio 1.00).
- **×16 collapses the softmax to one draw**, which leaves nothing to average at M = 4.
- **The mean residual dominates at every radius.** The student's mean is about 0.5 RMS per latent value away from base H3's one-call output at σ1 (loss about 120,000 summed over 442,368 values). The tilt part is 9% of that at ×4. So the gradient is mostly "move the mean to base H3's posterior mean".
- The draws differ by 0.23 of their mean's norm (0.14–0.30): the student is stochastic from the start at dial 0.30.

### 7.3 The run (`pmd_a`, radius U[1, 2], learning rate 2e-5)

Held-out eval on 8 clips against the pi-H3 4-step teacher, in the training process (unmerged adapter):

| Row | LPIPS | Detail (hf) | Contrast (`std_ratio`) | `mean_diff` | Latent MSE | `div` | Seconds |
|---|---|---|---|---|---|---|---|
| Untrained fast path (`baseline`, this run) | 0.4771 | 0.8129 | 1.009 | 0.109 | 1.176 | n/a | 0.93 |
| Untrained fast path (the other table) | 0.473 | 0.82 | | | | | |
| Latent regression, step 250 (the other table) | 0.445 | 0.41–0.52 | | | | | |
| **PMD student, step 0** | 0.4799 | 0.7945 | 1.000 | 0.111 | 1.204 | 0.370 | 1.22 |
| **PMD student, step 250** | 0.4726 | **0.4742** | **0.818** | 0.074 | 0.950 | 0.165 | 1.24 |
| PMD student, step 500 | not run | | | | | | |

The run was stopped after the step-250 eval by the early-stop rule (detail below 0.8, or contrast below 0.85, or LPIPS above 0.473): detail and contrast both failed. Step 500 was not run.

Training log: the loss fell from about 136,000 to about 5,000 by step 240 (the student's mean reached the teacher's), `draw_spread` fell from 0.30 to 0.07, ESS stayed between 1.3 and 2.6.

Contact sheets: `pmd_a/sheet_0_p0.jpg`, `sheet_0_p1.jpg`, `sheet_0_baseline_p*.jpg`, `sheet_250_p0.jpg`, `sheet_250_p1.jpg`. By eye at step 250 the scene and motion are those of the step-0 student, but the clips are blurred and washed out: fine texture is gone (parquet, rug pattern, faces), blacks are grey, contrast is low.

### 7.4 Gate: FAIL

| Gate | Result |
|---|---|
| ≤ 1.0 s end to end | **Fail** as measured: 1.24 s in the training process (the untrained path is 0.93 s there). The unmerged rank-32 LoRA and the noise input add about 0.3 s. The merged-adapter eval was not run, since the run failed on quality. |
| Detail ≥ 0.95 of the teacher | **Fail**: 0.47 (untrained 0.81) |
| LPIPS below 0.473 | 0.4726: level with the untrained path, not an improvement |
| Same scene by eye | The student's own scene is kept, but blurred and washed out |

### 7.5 What it means

- PMD on this step did what the probe predicted. With a one-call teacher at σ = 0.865, the target is base H3's posterior mean, which is a blurry, low-contrast image. The loss is dominated by the mean residual, and the tilts add a small correction to it. The student's mean was pulled onto the blurry mean, and its draws collapsed together (`draw_spread` 0.30 → 0.07, `div` 0.37 → 0.16) instead of staying sharp and diverse around it.
- This is the same failure as the latent regression (detail 0.41–0.52), reached by another route, plus the drift toward base H3's flatter look that the DMD run showed (contrast 0.82 here).
- The paper reports PMD alone as its most diverse and least faithful setting and adds a GAN term for fidelity (`docs/PMD.md`). Nothing here tests PMD with that term.

### 7.6 Not tried

- The GAN term. It is wired into `pmd_pih3.py` now (section 8), but has not been run.
- A lower σ for the full-resolution step, where the posterior mean is sharper.
- More draws (M = 8), a larger LoRA, a longer run, the paper's radius, `--teacher pih3`.
- The merged-adapter timing.

## 8. GAN term (paper 0.75): help wanted

**Status: code and CPU tests only. Never run on a GPU.** We are not running it on our cards. If you have the hardware, please run it and report back (section 8.6). It tests what section 7.5 left open: the paper's own fix for PMD alone (PMD alone is the paper's most diverse and least faithful setting; the paper adds a GAN term at weight 0.75 for fidelity). Here PMD alone blurred the student toward base H3's posterior mean.

### 8.1 What the term is

It reuses `pmd_h3.py`'s pieces, imported rather than copied: `DiscHead`, `d_loss`, `g_loss`, `gan_level`, `teacher_features` and `pmd_step`'s `extra_loss`. The paper's settings and our guesses are in `docs/PMD.md` (sections 2 and 3, guess 14).

| Piece | Here |
|---|---|
| Discriminator | `DiscHead` (LayerNorm → Linear 256 → SiLU → Linear 1, mean over video tokens; the head's shape is our guess). It reads the **frozen teacher's** block-`--d-block` (25 of 50) video rows of a clean latent noised to a level k/1000, with k uniform in 1..999. |
| Which model D reads | The same frozen teacher as the PMD targets: by default **base H3**, which is the shared DiT with the pi-H3 adapter off, the LoRA off and no blocks dropped. `--teacher pih3` makes it the pi-H3 2-step model, with the adapter on. With `--teacher-device` it is a second copy of base H3 on that card. Real and generated clips get the same silent audio from a seeded generator. They do not get the fast path's carried audio, which D could use to tell the two apart. |
| D update | `--d-updates` (default 5, the paper's D:G = 5:1) per update, before the G step. The fake is a fresh prompt, noise `z` and draw `u` through the whole fast path (`FastPMD.sample`), with no gradients. The real is one real latent at exactly the training canvas (8.3). The loss is non-saturating logistic. AdamW with lr `--d-lr` 5e-5 (the paper's value), clipped at `--clip`. |
| G term | `--gan-weight / M · g_loss(D(features(x_m)))` is added to each of the M = 4 replayed draws through `pmd_step`'s `extra_loss`. |
| Checkpoints | D's weights go into `pmd_N.pt` and `pmd_last.pt`. D's optimizer goes into `pmd_last.pt`. `--resume` restores both. A non-GAN checkpoint resumes with a fresh D and logs `pmd_pih3_gan_fresh_d`. |
| Randomness | The GAN term draws from its own RNGs. With it off, the PMD part draws the same tilts and noise as before. |

**One correctness point** (`pmd_pih3.g_term`). H3 checkpoints activations, so a backward recomputes each block under whatever switches are set at that moment (section 2). The G loss's gradient with respect to the draw is therefore taken *inside* the teacher's switches, with `torch.autograd.grad` on a detached copy. It then goes back through the student's replay under the student's switches, as a surrogate `(x · dG/dx).sum()` whose value is the G loss. A plain backward of the G loss after the teacher's switches are restored fails in the toy: H3-style checkpointing raises `CheckpointError`. (`pmd_h3.py`'s own GAN path has this flaw: there, `teacher_features`' graph is back-propagated under the student's LoRA state. That path was never run, and it is not fixed here.)

**Loss scale (a guess to watch).** Our PMD loss is a *sum* over the latent (`docs/PMD.md` guess 12). The G loss is a mean logit, so at 0.75 the G term may be small next to PMD. The training log prints both gradients on the draws: `pmd_grad` is |dPMD/dx| per draw and `g_grad` is |dG/dx| per draw. On the CPU toy they differ by about 4 orders of magnitude. If the ratio on H3 is like that, 0.75 is not the paper's balance. In that case report the ratio. A second run with `--gan-weight` scaled up by that ratio is the obvious follow-up. It would no longer be the paper's number.

### 8.2 Hardware

The H3 path has only ever been run on an RTX 5090 (32 GB, sm_120) with the int8 ConvRot kernels from `docs/PIH3.md`. Other GPUs are untested. All figures below are **estimates** built on one measured fact: **PMD without the GAN term peaked at 28.5 GB at 24.0 s per update** (section 7.1).

| Setup | Student card | Teacher card | Fits? |
|---|---|---|---|
| One GPU (default) | 28.5 GB + the teacher's feature graph during each G-term backward (checkpointed boundaries of 50 blocks, about 2.7 GB, plus one block's recompute, while the student's replay graph is alive) ≈ **31–33 GB** | — | Probably **not** on a 32 GB card. Needs a 40–48 GB+ card. |
| Two GPUs, `--teacher-device cuda:1` | About 28.5 GB as now (the student, the switchable pi-H3 adapter and the up-net; the teacher's activations move off this card) | A second base H3 (about 22 GB of int8 weights) + the G-term graph (about 3–4 GB) + D (5 MB) ≈ **26–27 GB** | Two 32 GB cards. |

Time per update, counted in full-resolution forward-equivalents (fe). The measured 24 s is about 20 fe, so 1 fe ≈ 1.2 s:

| Part | fe |
|---|---|
| PMD as now: the half-resolution step, 4 teacher calls, 4 draws, 4 replays | ≈ 20 |
| 5 D updates × (one fast-path sample ≈ 1.1 + 2 feature passes, all 50 blocks) | ≈ 15.5 |
| 4 G terms × (feature forward + recompute + backward ≈ 3.5–4) | ≈ 15 |
| **Total** | **≈ 50 → about 60 s per update (2.5× the PMD-only update)** |

- `--d-updates 1` brings it to about 38 fe, or about 45 s.
- Two GPUs do not overlap the work, so they are no faster; they only move memory. The tensors moved between cards are about 2 MB per call.
- Allow about 4 h for 250 updates and 8–9 h for 500, plus about 2 min per held-out eval.
- If memory runs out, the first fix is to stop the feature forward at `--d-block`. Today it runs all 50 blocks and keeps only block 25. That fix is not implemented.

### 8.3 Real latents: what exists and what to pass

The canvas is fixed: 768 × 512, 12 latent frames, a `(1, 24, 12, 32, 48)` latent. `pmd_h3.py` takes its real latents from the prompt caches (`--real cache`, the default: `LOT_H3_STILLS_CACHE` and `LOT_H3_CLIPS_CACHE`, or `--real-cache DIR ...`). The trainer keeps only items that crop to exactly that shape. It prints the counts (`kind=pmd_pih3_real`) and refuses to start if none fit. On our machine:

- `cache_atlas_clips`: 1,373 clips of 22 latent frames, but **portrait 512 × 768** (latent 48 × 32). **None fit** the landscape canvas.
- `cache_iso3d`: 403 landscape **stills** fit (out of 801). They are used only with `--real-stills`, which repeats each still over the 12 frames, as `pmd_h3.py` does. D may then learn that "real" means "static" and penalise motion.

**So there are no real video latents at this canvas.** The options, smallest first:

1. **`--real teacher`**: the 500 pi-H3 4-step clips in `$D/768x512_f39/train`, which you make anyway in 8.4 step 1. They are exactly `(1, 24, 12, 32, 48)` with their prompts. They are **generated, not real**: this is a GAN toward the gate's own reference model, not the paper's setup. Say so when you report.
2. **`--real-cache DIR`** with your own landscape H3 latents of at least 12 latent frames: real 768 × 512 (or larger, landscape) videos encoded with the H3 video VAE, in `train_h3.list_items`' layout (`<stem>_<W>x<H>_minimaxh3.safetensors` + `<stem>_minimaxh3_te.safetensors`; `lot/build_clip_cache_h3.py` writes it from `z.pt` + `meta.json` folders). This is the paper's setup.
3. `--real-stills` (above). Last resort.

### 8.4 Setup and commands

Paths the trainer reads. Every one is an environment variable or a flag:

| What | Variable / flag | Where it comes from |
|---|---|---|
| musubi-tuner source (H3 model, int8 loader, LoRA) | `MUSUBI_SRC` | Defaults to the vendored `third_party/` |
| H3 DiT, int8 ConvRot, pruned FL2VA | `LOT_H3_CHECKPOINT` | `hf download Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| pi-H3 2-step adapter | `LOT_PIH3_ADAPTERS` (a folder) | `hf download Lakonik/pi-H3 pi_h3_2step_dwd_adapter_int8_convrot.safetensors` (and the 4-step one for the teacher clips) |
| ComfyUI ≥ 0.45 with ComfyUI-piFlow (pi-H3's DWD head) | `LOT_PIH3_COMFY` | `docs/PIH3.md` setup |
| Prompt caches (Qwen3-VL text states; also the `--real cache` source) | `ACCEL_CACHES` (`:`-separated), else `LOT_H3_STILLS_CACHE` + `LOT_H3_CLIPS_CACHE` | Encoded with `LOT_H3_TEXT_ENCODER`, `lot/lot_paths.py` |
| The learned latent ×2 | `--up` | `python3 latent_up_h3.py --out $D/latent_up --width 160 --blocks 8 --steps 24000` (`docs/PIH3_TRAIN.md` step 0) |
| Teacher clips (held-out eval; `--real teacher`) | `--data $D` (`--folder 768x512_f39`) | `python3 pih3_distill.py teacher --out $D --size 768x512 --frames 39 --train 500 --holdout 8` |
| Real latents | `--real`, `--real-cache`, `--real-stills` | Section 8.3 |
| Must be **unset** | `LOT_H3_DROP_BLOCKS` | `load_dit` would drop blocks from the teacher copy too |

Also needed: `pip install lpips setproctitle`. The trainer refuses to start if more than 4 GB is already in use on the student's card. It does not check the teacher card.

```bash
cd lot
D=/path/to/distill                       # holds 768x512_f39/{train,held} and latent_up/latent_up.pt
UP=$D/latent_up/latent_up.pt

# CPU tests first
CUDA_VISIBLE_DEVICES="" python3 test_pmd_pih3.py && CUDA_VISIBLE_DEVICES="" python3 test_pmd_h3.py

# --check: 3 updates, prints s_per_step, peak_gb (student card), peak_gb_teacher, gan, d_loss, g_grad, pmd_grad
#   one GPU (needs > 32 GB, 8.2):
python3 pmd_pih3.py train --data $D --up $UP --out $D/pmd_gan_check --drop h3tiny --radius 1.0 2.0 \
    --gan-weight 0.75 --real teacher --check
#   two GPUs:
CUDA_VISIBLE_DEVICES=0,1 python3 pmd_pih3.py train --data $D --up $UP --out $D/pmd_gan_check --drop h3tiny \
    --radius 1.0 2.0 --gan-weight 0.75 --real teacher --teacher-device cuda:1 --check

# the run: 768x512, 39 frames, the 2-step pi-H3 fast path, gan 0.75; eval at 0, 250, 500; stops after 250 if detail < 0.8
CUDA_VISIBLE_DEVICES=0,1 python3 pmd_pih3.py train --data $D --up $UP --out $D/pmd_gan --drop h3tiny \
    --radius 1.0 2.0 --gan-weight 0.75 --real teacher --teacher-device cuda:1 \
    --steps 500 --stop-at 250 --stop-below-hf 0.8
#   (one GPU: the same without CUDA_VISIBLE_DEVICES=0,1 and --teacher-device)
#   (real video latents instead: --real cache --real-cache /path/to/landscape_cache)
#   (resume after a crash: add --resume $D/pmd_gan/pmd_last.pt; D and its optimizer come back too)
```

Everything else is section 7's run: rank 32, K = 4, M = 4, dial 0.30, tilt radius U[1, 2], lr 2e-5, H3-Tiny's blocks dropped. The only change is the GAN term.

### 8.5 Gate and early stop

The gate is section 6's, unchanged: detail ≥ 0.95 of the teacher, LPIPS below 0.473, the same scene by eye (and ≤ 1.0 s end to end, measured with `eval --adapter merged`).

The early-stop rule is section 7.3's: **stop after the step-250 eval if detail is below 0.8, or contrast (`std_ratio`) is below 0.85, or LPIPS is above 0.473.** The code checks only the detail part (`--stop-at 250 --stop-below-hf 0.8`). Check contrast and LPIPS by hand in the step-250 `pmd_pih3_eval` line, and stop the run if either fails.

### 8.6 What to report back

- Hardware: GPU model(s), and one or two GPUs.
- The `--check` line: `s_per_step`, `peak_gb` (the student's card), `peak_gb_teacher`, `pmd_grad` and `g_grad`.
- The `pmd_pih3_real` line: the source and the counts.
- The eval table at steps 0 (with `baseline`), 250 and 500, from `log.jsonl` (`kind=pmd_pih3_eval`): `hf` (detail against the teacher), `std_ratio` (contrast), `lpips`, `mean_diff`, `div`, `seconds`.
- Training: `s_per_step` and `peak_gb` per card, and the trends of `loss`, `gan`, `d_loss`, `draw_spread` and `ess`. A `d_loss` near 0 means D won. One stuck near 1.386 (2·ln 2) means D learns nothing.
- The contact sheets `sheet_250_p*.jpg` and `sheet_500_p*.jpg`, and whether the scene is the same by eye.
- Which `--real` you used.

