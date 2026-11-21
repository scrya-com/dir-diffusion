# PMD on MiniMax H3: Posterior Matching Distillation

> **Research code, not yet run on a GPU.** Everything below the paper's own settings is a guess, listed in full in section 3. CPU tests: `CUDA_VISIBLE_DEVICES="" python3 lot/test_pmd_h3.py` (16 tests).

**PMD is the method of Hidir Yesiltepe, Julius Berner, Arash Vahdat and Pinar Yanardag**: ["Posterior Matching Distillation"](https://arxiv.org/abs/2611.26629), arXiv 2611.26629 (Virginia Tech and NVIDIA; project page pmd-distillation.github.io). Their code is not released. This port to H3 is independent and is not affiliated with the authors, with MiniMax, or with ByteDance Seed (DiR).

Files:
- `lot/pmd_h3.py` holds the trainer, the any-step sampler, and the probe and score scripts.
- `lot/test_pmd_h3.py` holds the CPU tests.

## 1. What it is

A few-step **stochastic** student `x̂ = G(z_t, u, t)` gets a noise input `u`. It is trained so that its distribution of clean samples, given a noisy state, matches the teacher's posterior `q(x | z_t)`. This is a stronger target than matching one mean (DMD-style mode seeking).

**How the posterior is matched:**
- It is matched through exponentially tilted means. For a tilt vector `λ`, the student's softmax-weighted mean `Σ_m softmax(λᵀx̂_m) x̂_m` over M draws must equal the teacher's tilted posterior mean.
- **Tilt=Shift identity:** under Gaussian corruption, the tilted mean is one teacher call at a shifted input, `D_t(z_t + ρ_t λ)` with `ρ_t = σ_t²/α_t`. So no teacher sampling is needed.
- Matching over an open ball of tilts identifies the whole posterior (their Posterior Identification Theorem).

**Any-step:** the student learns the posterior at each level of a 4-level grid. So the same weights sample with 4 steps or 2 steps (for example levels 0 and 2), re-noising between steps.

### ρ for H3's rectified flow (derivation)

The paper writes `z_t = α_t x + σ_t ε`, with t = 1 as pure noise. H3 (this repo) uses `z = (1 − σ) x + σ ε`. So `α = 1 − σ` and `σ_t = σ`, which gives:

```
ρ = σ² / (1 − σ)
z + ρλ = (1 − σ) · (x + λ σ²/(1 − σ)²) + σ ε          (same noise, same level, clean signal moved)
D(z) = z + σ · v(z, cleanness 1 − σ)                  (H3's head predicts v = x0 − eps)
```

On the DiR checkpoint's grid this gives:

| σ | ρ |
|---|---|
| 1 | ∞ (no PMD loss at pure noise, as in the paper's FLUX.2 setting) |
| 0.947 | 16.9 |
| 0.857 | 5.14 |
| 0.667 | 1.33 |

`test_tilt_equals_shift_gaussian_and_monte_carlo` checks the identity two ways in this parametrisation:
- exactly against a closed-form Gaussian posterior;
- against 400k self-normalised importance-weighted posterior draws.

## 2. One update (what `pmd_h3.py train` does)

1. **Choose a level.** `k` is drawn uniformly from the levels {1, 2, 3}. Level 0 (σ = 1) only starts the rollout.
2. **On-policy rollout** from Gaussian noise to `z_k`, without gradients (section 4.4). Each clean prediction is re-noised to the next level. Then `z_k` is detached.
3. **K = 4 coarse tilts.** Gaussians on a coarse grid are upsampled to the latent, normalised to unit L2 norm, and multiplied by a radius ~ U[0.25, 0.5].
4. **K teacher calls** at `z_k + ρλ_i`, without gradients.
5. **M = 4 student draws** at `z_k`, without gradients. The same draws serve all 4 tilts (C.1).
6. **Loss.** The default is Algorithm 2: the delete-one jackknife inside each half, cross-fitted over the 3 balanced pair splits. It is signed and not clamped. `--estimator plain` gives Algorithm 1 instead. The loss is averaged over the tilts.
7. **Gradients by replay (C.1).** `∂L/∂x̂_m` is taken on the cached draws. Then each draw is replayed with gradients (same `u`, same audio seed) and back-propagated with `x̂_m · g_m`. Only one student graph is alive at a time. `test_replayed_gradients_equal_direct_autograd` checks this against plain autograd.
8. **Optimiser step.** AdamW with β = (0.9, 0.999), ε = 1e-8, no weight decay, and gradient clip 10.

### The stochastic student (appendix C.1)

The paper concatenates "Gaussian noise-mixed and original latent tokens". It keeps the pretrained input weights on the **mixed** latent and zero-initialises the new weights on the **raw** state. On H3:
- The patch embedding is `video_patch_proj`, a `Linear(96 → 5376)` on 1×2×2 patches of the 24-channel latent.
- Doubling it to 192 inputs equals adding a second `Linear(96 → 5376, bias=False)`, called `NoiseInput`, on the raw state's tokens. Its output goes through `forward_video`'s existing `video_extra` hook. No musubi code changes.
- The mixed latent is run at its effective level `σ̃`, and `x̂ = y + σ̃ · v`.

> **Correction to the brief:** step 0 is **not** the starting model in general. It is the starting model applied to the noise-mixed latent, so it is stochastic from the first update. It equals the starting model exactly only at a level whose noise dial is 0 (the pure-noise level). `test_zero_init_noise_input_leaves_the_starting_model` checks both cases.

### Student and teacher on one copy of H3

There is one DiT in memory.

**The student** is H3 plus:
- the init run's LoRA (the DiR checkpoint, rank 128, frozen);
- a new trainable PMD LoRA (rank 32, all 50 blocks by default; `--lora-last K` restricts it to the last K);
- the DiR head's iteration embedding `r_embed[k]` (r = level index);
- `NoiseInput`.

**The teacher** is the same DiT with **every** LoRA module disabled (`.enabled = False`, so no compute and no graph). It gets no `video_extra` and no `video_temb_delta`, and it runs at the state's own level `σ_k`. That is base H3, one call, guidance 1. `test_lora_off_teacher_is_the_base_model` checks, with real musubi LoRA modules on a toy DiT, that the teacher is bit-identical to the model before any LoRA was attached.

**Which teacher, and what it costs:**
- **The paper's setting is the undistilled base model, at guidance scale 1, one evaluation per tilt** (Tables 3 and 4: "Teacher / student guidance scale 1/1").
- It is *not* CFG and *not* many-step sampling. Tilt=Shift needs a posterior mean, and appendix E.1 notes that a CFG output is not the posterior mean of any distribution. A many-step ODE sample is one posterior *sample*, not the mean.
- So the default teacher costs K = 4 forward passes per update.
- H3's published checkpoint is CFG-distilled (DIR_CHEATSHEET.md), so its one-call output already carries baked-in guidance. That breaks the exact-denoiser assumption somewhat, but it is the closest thing to the paper's teacher.
- `--teacher-cfg S --neg-te EMPTY_te.safetensors` applies real CFG to the teacher. It doubles the teacher cost (8 forwards per update) and departs from the paper. An empty-prompt embedding exists at `/media/2TB/lora-data/h3_lot/cache_iso3d/uncond_minimaxh3_te.safetensors` (not checked for which prompt it encodes).
- The many-step cost belongs to **evaluation** (the 20-step dense20 reference), not to training.

**Starting point.** The default is `--init-run johndpope/dir-diffusion-h3`, the published DiR checkpoint, with its own grid of 1 / 0.947 / 0.857 / 0.667. `--init-run none` gives the paper's setting: base H3 plus a fresh LoRA, on H3's `grid` (1 / 0.973 / 0.923 / 0.8).

## 3. From the paper vs our guesses

**From the paper:**
- Tilt=Shift, with `ρ = σ²/α` (made concrete for RF above).
- Algorithms 1 and 2.
- M = 4 draws, with the same draws reused over 4 tilted teacher queries.
- Coarse tilt probes (8 × coarser than the latent in both of their models), unit direction × radius ~ U[0.25, 0.5].
- On-policy backward simulation without gradients, detached states, and no PMD loss at the pure-noise call (FLUX.2).
- Mixed latent through the pretrained input weights, and the raw state through zero-initialised new weights.
- The cached-draw replay for memory.
- AdamW (0.9, 0.999), ε 1e-8, weight decay 0, gradient clip 10.
- Teacher and student guidance 1, no EMA.
- The GAN term: non-saturating logistic loss, weight 0.75, D:G = 5:1, D learning rate 5e-5, a discriminator head on frozen-teacher caption-conditioned midpoint features, noise levels k/1000 with k uniform in {1..999}.
- Any-step sampling by re-noising, and CPS sampling at η = 0.9 on the RF model (FLUX.2).
- Noise dials {0, 0.30, 0.45, 0.15} (FLUX.2's values; their transfer to H3 is guess 2).

**Guessed (every one is a flag):**
1. **Dial → mixed latent.** `y = ((1−d) z + d u)/(a+s)` with `a = (1−d)(1−σ)` and `s = √((1−d)²σ² + d²)`, giving `σ̃ = s/(a+s)`. This is "noise mixing in RF coordinates" as we read it (`mix_noise`, tested to stay on the RF path).
2. **Dial values for H3.** FLUX.2's table is used unchanged (`--dials`).
3. **Tilt normalisation.** Unit L2 norm over the **whole** latent, then × radius. Per-element normalisation would make `λᵀx̂ ~ r√d` and the softmax one-hot.
4. **Coarse grid for video.** A factor of (4, 8, 8) in (T, H, W): 3 × 3 × 4 cells at 12 × 18 × 32. Coarse in time too, channels independent, trilinear upsampling (`--tilt-factor`).
5. **CPS re-noising formula.** `x' = (1−σ')x̂ + σ'(cos(ηπ/2) ε̂ + sin(ηπ/2) ε)`, where ε̂ is the state's implied noise. η = 0.9 is used for both the training rollout and sampling (`--eta`; 1 = fresh noise as in section 4.5).
6. **LoRA instead of the full fine-tune.** Rank 32 on all 50 blocks.
7. **Learning rate 2e-5.** The paper's 5e-7 is for a full fine-tune. The new input half uses the same rate (`--noise-lr`).
8. **Batch.** 1 sample per update (`--accum`) instead of a global batch of 16.
9. **Level choice.** Uniform random over {1, 2, 3} per sample (`--level-schedule`).
10. **Starting from DiR** instead of the undistilled base:
    - the DiR LoRA is frozen and a new LoRA is stacked on top;
    - `r_embed[k]` is used even though the mixed level σ̃ ≠ σ_k;
    - the DiR estimate projection `x0_proj` is fed zeros, so the student stays a function of `z_t` alone, not of the rollout history.
11. **Teacher.** Base H3, which is CFG-distilled, so not an exact posterior-mean denoiser (section 2).
12. **Loss scale.** Squared norm **summed** over the latent (Algorithm 1's `squared_norm`), averaged over tilts. This interacts with clip 10.
13. **Audio.** H3's silent audio rows use one seed per prompt for every student, replay and teacher call.
14. **GAN.**
    - Head design: LayerNorm → Linear 256 → SiLU → Linear 1, mean-pooled over video tokens.
    - The "midpoint" is the output of block 25 of 50.
    - The G term is applied to the M draws at level k.
    - Real latents come from the cache; a still is repeated over time.
15. **"12 frames" = 12 latent frames** (39 pixel frames), the repo's `--frames` convention.
16. **Evaluation.** LPIPS (AlexNet, 256 × 256 bicubic, as in the paper's image protocol) on every 4th pixel frame.

**Not ported:** `LOT_H3_DROP_BLOCKS` and the LoRA-stacking helper live on `exp/h3-tiny-drop`, not `feat/accel`. Stacking works without them, as two musubi networks on one DiT, which is how `train_dir_h3.py` already runs its gen and fake LoRAs.

## 4. Memory and time per update on one RTX 5090 (estimated, not measured)

**Shape.** 512 × 288 at 12 latent frames is a 24 × 12 × 18 × 32 latent. That packs to 1,728 video rows, plus text and 65 audio frames, about 2.2k rows.

**Memory at `--swap 4`, rank 32:**

| Item | GB |
|---|---|
| 46 resident int8 blocks | 19.3 |
| Non-block weights | about 1 |
| DiR LoRA, rank 128, bf16 | 1.1 |
| PMD LoRA: weights, gradients and AdamW state (138M params, fp32) | 2.2 |
| Checkpointed activations: 50 block boundaries (about 1.2 GB) plus one block's recompute (about 1.5 GB) | 2.7 |
| Allocator and context | about 1.5 |
| **Total** | **about 27–28** |

For comparison, the DiR clip run peaked at 29.8 GB at `--swap 6` with a larger shape and two trained LoRAs. If this runs out of memory, try in order `--swap 6`, `--rank 16`, `--lora-last 8`.

**Time.** Count forward-equivalents per update:

| Step | Forward-equivalents |
|---|---|
| Rollout (mean of 2 student calls) | 2 |
| Teacher calls | 4 |
| No-grad draws | 4 |
| 4 replays × (forward + recompute + backward ≈ 4) | 16 |
| **Total** | **≈ 26** |

- The DiR clip run's 72 s per 2-sample step at about 4.6k rows works out to roughly 1.6 s per forward-equivalent. At about 2.2k rows that is about 0.9–1.2 s.
- So an update is **about 25–45 s**, which is 300 updates in 2–4 h.
- The default `--minutes 120` stops at whichever comes first: the time limit or `--steps`.
- `--gan-weight 0.75` adds about 5 × (4-step rollout + 2 teacher calls) for the D updates, plus 4 teacher forward+backward passes for the G term. That is roughly 3× the update cost.
- `--check` prints the real peak and is the first thing to run.

## 5. First trial

```bash
cd /home/johndpope/Documents/GitHub/dir-diffusion-pmd
export ACCEL_CACHES=...                       # the prompt caches (default: the stills + clips caches)
# 1. one update with gradients: grad sums, ESS, shift ratio, peak memory; ok=1; writes nothing
LOT_H3_TRAIN=1 python3 lot/pmd_h3.py train --check --init-run johndpope/dir-diffusion-h3 --out lot/runs/pmd_check
# 2. the trial (512x288, 12 latent frames, 300 updates or 120 minutes; rerun the same command to resume from pmd_last.pt)
LOT_H3_TRAIN=1 setsid nohup python3 lot/pmd_h3.py train --init-run johndpope/dir-diffusion-h3 \
  --size 512x288 --frames 12 --steps 300 --minutes 120 --swap 4 --rank 32 --lr 2e-5 \
  --save-every 50 --out lot/runs/pmd_trial > /tmp/pmd_trial.log 2>&1 &
```

**Watch** with `grep "kind=pmd step" /tmp/pmd_trial.log | tail`. Each line reports:
- `loss` (Algorithm 2, signed) and `loss_plain` (Algorithm 1, diagnostic);
- `ess`: softmax effective sample size, from 1 (one-hot) to 4 (flat);
- `w_max`;
- `shift_ratio`: ‖ρλ‖/‖z_k‖;
- `draw_spread`: draw-to-draw spread relative to the mean;
- `grad_norm`, `s_per_step` and `peak_mb`.

**Red flags:**
- `ess` stuck near 1: the tilts are too large, so lower `--radius`.
- `ess` at 4.00 together with `shift_ratio` ≈ 0: the tilts are negligible, so raise `--radius` (the paper's U(1, 2) row gave the most diversity).
- `draw_spread` collapsing to 0: the student is ignoring its noise.

**Saves** are written to the run directory:
- `pmd_N.pt` and `pmd_final.pt`: the LoRA and `NoiseInput`;
- `pmd_last.pt`: the same plus the optimiser, step, RNG state and elapsed minutes, written atomically;
- `config.json` and `log.jsonl`.

## 6. Pass check: 2-step PMD vs DiR

Use the same card and the same `--swap` for every name, because the same noise gives different clips on different cards (DIR_CHEATSHEET.md).

```bash
O=(--out lot/runs/probe_pmd --frames 12 --max-latent-hw 18 32 --swap 4)
python3 lot/probe_dir_h3.py "${O[@]}" --name ref --reference                       # dense20 for the 8 held-out prompts, z = seed i
python3 lot/pmd_h3.py probe "${O[@]}" --name dir4 --dir-run johndpope/dir-diffusion-h3 --iters 4 --seeds 4
python3 lot/pmd_h3.py probe "${O[@]}" --name dir2 --dir-run johndpope/dir-diffusion-h3 --iters 2 --seeds 4
python3 lot/pmd_h3.py probe "${O[@]}" --name pmd2 --pmd lot/runs/pmd_trial --levels 0,2 --seeds 4
python3 lot/pmd_h3.py probe "${O[@]}" --name pmd4 --pmd lot/runs/pmd_trial --levels 0,1,2,3 --seeds 4
python3 lot/pmd_h3.py probe "${O[@]}" --name pmd2_eta0 --pmd lot/runs/pmd_trial --levels 0,2 --eta 0 --seeds 4   # same-noise path
python3 lot/pmd_h3.py score --out lot/runs/probe_pmd --names dir4 dir2 pmd2 pmd4 pmd2_eta0 --pass pmd2 dir4 dir2
```

`probe` uses `probe_dir_h3.py`'s held-out items, crops and noise: seed 0 is `z = randn(seed i)`, with audio seed 7i. `score` decodes with the VAE and writes `SCORE.md` and `score.json`. It reports two numbers, each with its standard error:
- **LPIPS to dense20:** seed 0 against the 20-step H3 clip from the same `z`. Lower is better.
- **Seed diversity:** mean pairwise LPIPS between the 4 seeds of one prompt. Higher is better.

**Pass** (`kind=pmd_pass PASS=1`) requires both:
- 2-step PMD's LPIPS to dense20 is ≤ that of **DiR stopped after iteration 2**;
- 2-step PMD's seed diversity is ≥ that of **DiR at 4 iterations**.

PMD-2 against DiR-4 on LPIPS is reported as well. Look at the clips by eye too: `python3 lot/probe_dir_h3.py --out lot/runs/probe_pmd --decode pmd2 dir4 dir2` writes side-by-side mp4s (dense20, then every seed of each name), since `probe` stores its latents in `probe_dir_h3.py`'s format.

**Caveat.** LPIPS to a same-z dense20 favours deterministic samplers. PMD re-noises with fresh noise (CPS at η = 0.9) and draws `u`, so even a perfect PMD student does not reproduce dense20's trajectory from `z`. At η = 0.9 about 99% of the noise is replaced between steps (sin²(0.45π) ≈ 0.976 of the variance), so the PASS line's fidelity half favours DiR almost by construction. `pmd2_eta0` re-noises with the state's own implied noise (η = 0, deterministic between steps, still with student draws `u`), which gives a fidelity number on a same-noise path; read it next to `pmd2`'s diversity. The paper's own trade-off holds too: PMD-only training (no GAN) gives the most diverse but least faithful samples (appendix E.1). A failed fidelity check without the GAN term is therefore expected; it is not proof that the code is wrong.
