# PMD on the pi-H3 2-step fast path

> **Result: gate FAILED at step 250 (detail 0.47 of the teacher, contrast 0.82); the run was stopped there.** Code and CPU tests done (`CUDA_VISIBLE_DEVICES="" python3 lot/test_pmd_pih3.py`, 11 tests; `lot/test_pmd_h3.py`, 16 tests). GPU results are in section 7.

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

- The GAN term (`pmd_h3.py` has the pieces; they were never run on a GPU).
- A lower σ for the full-resolution step, where the posterior mean is sharper.
- More draws (M = 8), a larger LoRA, a longer run, the paper's radius, `--teacher pih3`.
- The merged-adapter timing.
