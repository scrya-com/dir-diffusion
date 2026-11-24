# DiR with R = 2: "what if it's 2 of 2 iterations?"

Can we train DiR natively for 2 iterations, instead of stopping the published 4-iteration generator after 2 (LPIPS 0.462 against its own full rollout, [`ACCEL.md`](https://github.com/scrya-com/dir-diffusion/blob/feat/accel/docs/ACCEL.md) on `feat/accel`)? Would it match pi-H3 2-step?

**Status (2026-11-25): desk research. Nothing was trained or rendered for this note; both GPUs were busy.** Sources: the paper (arXiv 2610.12102 v1, PDF read in full with appendices A–G), this repo's code and docs on `main`, `feat/accel` and `feat/pi-h3`, the published run's `log.jsonl`, and one CPU-only weight measurement (section 4). Every guess is marked **(guess)**.

## Short answer

- **The trainer already supports it** ✅: `--iterations 2 --sigmas turbo` gives noise levels 1 and 0.857. No code change is needed for a run from Turbo's LoRA.
- **The paper has no R = 2 model.** It trains R = 4 only and reads the 2-step number off that model's prefix.
- **It is cheap to try:** about 69 s per step against 76 s at R = 4; a 50-step gate run is about 1.5 GPU-hours.
- **It will not be as fast as pi-H3 2-step as it stands:** 1.82 s of DiT against 1.47 s at 768×512×39. The gap is the unmerged LoRA, and merging it into the int8 weights erases it (measured, section 4).
- **Verdict:** unlikely to beat pi-H3 2-step on quality (**guess**: under 10%; 15–20% to match by eye), and it loses on speed by about 20%. Worth the 10-minute and 1.5-hour experiments in section 5, not a multi-day run.

## 1. What the paper says about R

**R is fixed at training time, and the paper only ever trains R = 4.**

- Sec. 3.1: "For DMD training, we set R = 4 so that the trained generator supports generation with up to four steps; as shown in our later experiments, using additional steps does not further improve FID."
- Sec. 2.3: "Here, r is the refinement index and R is the maximum number of network evaluations."
- Appendix A: "During training, we sample the refinement index simply as r = i mod R, where i is the current training iteration." The generator and the fake model each have an `r` embedding with R rows, so a model cannot run more iterations than it was trained for.
- Sec. 3.3: "All methods use the same self-rollout training procedure with R = 4."

**A model trained for R runs at fewer iterations by design.** Every number at 1, 2 and 3 NFE in the paper is a prefix of the one R = 4 model:

- Sec. 2.6: "we apply the distribution-matching objective at every step rather than only after convergence. Thus, every prefix of the rollout, up to the trained maximum R, is encouraged to produce a valid sample."
- Sec. 2.3: "Each step is trained to produce a valid sample and can use the previous prediction to improve its output."

**Results by number of iterations** (ImageNet 256×256, SiT-XL/2, FID-50k, lower is better):

| Table | Row | 1 NFE | 2 NFE | 3 NFE | 4 NFE |
|---|---|---|---|---|---|
| 3 / 5a, no CFG | Data-space DMD (DiR) | 3.85 | 2.68 | 2.47 | 2.46 |
| 3 / 5a, no CFG | DMD, flow-renoise, uniform schedule | 4.45 | 3.22 | 2.67 | 2.75 |
| 3 / 5a, no CFG | Phased DMD, flow-map, uniform | | | | 2.55 |
| 3 / 5b, CFG 1.1 on τ ≤ 0.6 | Data-space DMD (DiR) | 3.31 | 2.47 | 2.33 | 2.29 |
| 3 / 5b, CFG | DMD, flow-renoise, uniform | 4.02 | 2.89 | 2.51 | 2.61 |
| 3 / 5b, CFG | Phased DMD, flow-map, uniform / best of 3 | | | | 2.86 / 2.37 |
| 2a | DiR, reads restricted to τ ∈ [t_{r+1}, 1], w = 1 | 44.91 | 4.08 | 2.24 | 2.51 |
| 8 (CIFAR-10, UNet) | Data-space DMD | 12.20 | 8.28 | 8.32 | 8.62 |
| 8 (CIFAR-10) | DMD, flow-renoise | 11.21 | 11.14 | 9.93 | 10.03 |

What the tables say about quality against R:

- **Most of the gain is from 1 to 2 iterations.** 3.85 → 2.68 → 2.47 → 2.46. The third iteration adds 0.21, the fourth 0.01. Appendix B gives a run-to-run standard deviation "below 0.05", so 3 and 4 NFE are tied.
- **2 iterations is 0.2 FID behind 4** (2.68 against 2.46; with CFG 2.47 against 2.29). On CIFAR-10, 2 NFE is the best column.
- **DiR's lead over a scheduled sampler is largest at 2 NFE:** 0.54 FID over flow-renoise at 2 NFE against 0.29 at 4 (no CFG).
- **Restricting an iteration's reads to high noise wrecks the prefixes** (Table 2a, the row matched to Phased DMD): 1 NFE goes to 44.91 and 2 NFE to 4.08. That is the eDMD-style floor rule (`--tau-floor`).
- **Iterations keep changing the picture even where FID is flat.** Table 6, mean L2 between consecutive iterates in pixels: 59.65, 44.25, 35.98. So a large distance between iteration 2 and iteration 4 is expected and is not, by itself, a quality gap.

**The noise schedule per R: there is none.** Sec. 2.5: "The original flow-time input is fixed to t = 1 for every generator evaluation." Sec. 2.3: "Sampling therefore does not require a flow-time schedule." The only schedules in the paper belong to the baselines (Table 2c: `[1.00, 0.75, 0.50, 0.25]` and shifts 0.5 and 2.0), and "the shift parameter does not exhibit a predictable correlation with final performance". The paper gives no guidance for the second noise level of a 2-step schedule.

**What the paper does not contain:**

- No model trained with R = 2 (or 1, 3, or more than 4). Whether a native R = 2 model beats the 2-NFE prefix of an R = 4 model is not tested.
- No table for more than 4 steps. The sentence "using additional steps does not further improve FID" is backed in v1 only by the 3 → 4 NFE columns.
- No text-to-image or video result: "Scaling data-space iteration to text-to-image generation and other modalities is left for future work."

## 2. What our code needs for R = 2

### How the published run sits against the paper

The published checkpoint ([`lot/runs/dir_turbo_clip/config.json`](../lot/runs/dir_turbo_clip/config.json)) is not the paper's pinned form:

| | Paper | Published run |
|---|---|---|
| Flow time per iteration | 1 at every iteration | `sigmas` 1 / 0.947 / 0.857 / 0.667 (`--sigmas turbo`) |
| State fed to iteration r | `z`, plus the estimate through a new projection | the Euler state built from `z` and the earlier estimates (`mix_state: false`) |
| Fake model | conditioned on r | the same: `fake_source: iteration` |
| DMD read level τ | Beta(2.5, 1) on (0, 1) | Beta(2.5, 1), redrawn until τ ≤ 0.9 (`tau_max`); no floor (`tau_floor: none`) |
| Start | the teacher | Turbo 4-step LoRA cut to rank 128, zero DiR head |

Two consequences for the question:

- **The 2-of-4 prefix is already trained as a sample.** With `fake_source: iteration` and no floor, iteration 2's output gets the same DMD loss as iteration 4's, as in the paper. The run's own eval says so: the gap to the teacher's answer at step 100 is 0.176 / 0.151 / 0.145 / 0.143 after iterations 1–4, while each later iteration still moves the latent by 0.55 and 0.38 rms (`change` in `log.jsonl`).
- **So LPIPS 0.462 is a distance between two samples, not a score of the 2-iteration clip.** The DiT is chaotic (`ACCEL.md`: rounding the noise to fp16 moves the final latent by 52–56%), and the paper's own iterates differ as much (Table 6). Nobody has yet judged the published checkpoint's iteration-2 clip on its own. That is experiment 0 below.

The estimate input `x0_proj` is still unused in the published weights: `dir_head.pt` has |weights| ≤ 7.9e-4 and `r_embed` row norms 3e-4 to 7e-4 (read from the file). The estimate reaches the model through the state.

### Flags

Nothing in the trainer is fixed to 4. `--iterations` sizes the generator's and the fake model's `r` embeddings, `parse_sigmas`, the `r = (step − 1) mod R` schedule, the fake warm-up, the eval and the anchors.

```bash
cd lot
LOT_H3_TRAIN=1 python3 train_dir_h3.py --init ../../lot-diffusion/lot/runs/turbo4_init \
  --iterations 2 --sigmas turbo --zero-init --frames 12 \
  --fake-source iteration --tau-max 0.9 --dmd-weight norm --grad-clip 1.0 --ema 0.99 --lr 2e-5 \
  --rank 128 --fake-rank 64 --max-latent-hw 32 48 --swap 6 --dmd-draws 4 --fake-warmup 100 --accum 2 \
  --steps 50 --eval-every 25 --save-every 25 --eval-anchors --holdout 8 --seed 0 \
  --out /media/2TB/lora-data/h3_lot/runs/dir_r2_turbo
```

This is the published run's command ([`DIR_CHEATSHEET.md`](../DIR_CHEATSHEET.md), "Reproduce the latest run exactly") with `--iterations 2` and 50 steps. Run it with `--check` first. Not run here.

### Which noise levels

`parse_sigmas` gives `shift(linspace(1, 0, R + 1)[:R])`:

| `--sigmas` at R = 2 | Levels | Verdict |
|---|---|---|
| `turbo` (shift 6) | **1, 0.857** | **Proposed.** |
| `grid` (shift 12) | 1, 0.923 | No: above 0.9 H3's one-call answer is dominated by its baked guidance (22–31% of the answer at 0.90, [`DIR.md`](DIR.md) part 4). |
| `1,0.667` | 1, 0.667 | The one alternate worth a run. |
| `1` (the paper) | 1, 1 | No: our pinned runs never learned to read `x0_proj`, so this is a one-step generator run twice. |

Why 1 and 0.857:

- **It is on Turbo's own grid.** 0.857 is Turbo's third node, so the start LoRA has seen that level, and the published generator was DMD-trained at it (its iteration 3).
- **pi-H3 2-step uses almost the same pair.** `pih3_sigmas(2)` on `feat/pi-h3` is 0.994, 0.865 (raw times 0.93 and 0.349 under shift 12). The official 2-step distillation of this model landed on the same second level.
- **It is under 0.9,** below the range where the baked guidance takes over.
- **The estimate is visible:** the second iteration's input is 14% estimate, against 5% at 0.947 (the second level of the 4-iteration run).
- **0.667 as the alternate:** the input is 33% estimate and base H3 follows a fed estimate closely there (0.10 from it, `DIR.md`). The risk is that iteration 2 only polishes iteration 1's blur. **(guess)**

**At R = 2 the Euler state and `--mix-state` are the same thing.** The first level is 1, so `next_state` gives `x_1 = σ_1·z + (1 − σ_1)·x̂0⁽¹⁾` either way. A 2-iteration generator on a grid is therefore exactly a function of `(x̂0, z, r)`, the paper's form, with the estimate entering through H3's pretrained input.

### Fake model, DMD, floor rule

- **Fake model:** `--fake-source iteration`. One rank-64 fake LoRA with a 2-row `fake_r_embed`. The endpoint fake model (`final`) inflated the first iteration in two runs (`DIR.md` part 4).
- **Updates per iteration double:** `r = step mod 2`, so each iteration gets half the generator updates (a quarter at R = 4). 50 steps at R = 2 give each iteration what 100 steps gave it in the published run.
- **Floor rule:** keep `--tau-floor none`. The paper's restricted row (2 NFE 4.08 against 2.68) and our two floor runs (off prompt) agree. If a calibrated floor is ever wanted, `edmd_calibration_turbo_f12.json` will not load: its landing levels are 0.947 / 0.857 / 0.667 and `load_read_laws` requires this run's.
- **Data:** unchanged. `cache_iso3d` (793 training prompts, 8 held out), prompts and latent shapes only, `--frames 12 --max-latent-hw 32 48` (768×512, 39 frames).

### Where 4 is assumed

| Where | What | Effect on R = 2 |
|---|---|---|
| `train_dir_h3.py --init` | loads `dir_head.pt` with a strict `load_state_dict` | **A 4-row head fails to load.** Use an init folder that holds only `lora_gen.safetensors`: `turbo4_init`, not `turbo4_init_run`, and not the published folder as it is. |
| `turbo_lora.py --run-dir` | writes a 4-row zero head and `"iters": 4` | as above |
| `probe_dir_h3.py`, `yardstick_dir_h3.py`, `bench_dir_h3.py` | `--iterations` defaults to 4 | pass `--iterations 2`, or `parse_sigmas` rejects the run's 2 levels |
| `render_dir_h3.py`, `dir_two_track_h3.py`, `train_dwd_h3.py`, `accel_h3.py` | read `iters` from `config.json` | fine |
| DWD stack (`feat/accel`), pi-H3 decoder on DiR (`feat/dwd-pih3`) | trained on iteration 4's tokens | must be refitted on the 2-iteration generator |

**Warm start from the published weights** needs no code change either: a folder with only the published `lora_gen.safetensors` (a symlink is enough), then the command above with `--init` on it. The head restarts at zero, which drops values of at most 7.9e-4. What that does to the output was not measured.

## 3. Cost

**Time per step, from the published run's `log.jsonl`** (RTX 5090, `--swap 6`, 12×32×48 latents, 2 samples per update, 4 DMD draws, 4 fake updates; median seconds by the iteration trained):

| r trained | 0 | 1 | 2 | 3 | Mean |
|---|---|---|---|---|---|
| Seconds per step | 64.0 | 74.2 | 75.7 | 92.2 | 76.5 |

- **R = 4:** 76.5 s median (the cheatsheet says 72 s; the run's mean with evals and saves is 85.5 s).
- **R = 2 (estimate from the same log):** the mean of the r = 0 and r = 1 rows, **69 s**. About 10% cheaper.
- **Why so little:** a step is `8r + 24` no-grad forwards plus 10 forward-and-backward passes (counted from `generator_update` and `fake_phase` at these settings). Only the `8r` part is the rollout. At R = 4 the average is 36 + 10, at R = 2 it is 28 + 10.

**What the published run took:** 100 steps in 8,638 s (2.40 h), after 100 fake warm-up updates at 12.0 s (20 min). About 2.75 h plus loading and anchors. The model card says "about 2.5 hours".

**GPU-hours for R = 2 on one RTX 5090** (69 s per step, warm-up about 11 s per update at R = 2 **(guess)**, 10 min of loading, anchors and evals **(guess)**):

| Run | Steps | GPU-hours | Updates per iteration |
|---|---|---|---|
| Gate run | 50 | about 1.5 | 25, the published run's count |
| Same step count as published | 100 | about 2.4 | 50 |
| A real attempt | 200 | about 4.3 | 100 |
| Decoder for it (pi-H3 decoder on DiR, stage 1: 2,000 steps at 1.4 s, plus collecting rollouts) | | about 2 **(guess)** | |

Peak memory should be the published run's 29.8 GB, since the rollout is no-grad. **(guess, not measured)**

## 4. Expected speed

A DiR iteration is one H3 forward, so 2 iterations cost half of 4. Measured rows are from [`PIH3.md`](https://github.com/scrya-com/dir-diffusion/blob/feat/pi-h3/docs/PIH3.md) (same session, RTX 5090, dense attention, no block swap); the DiR R = 2 rows are those numbers halved.

| Canvas, frames | pi-H3 2-step DiT (measured) | DiR 2 iterations DiT (estimate) | Gap |
|---|---|---|---|
| 768×512, 39 | 1.47 s | 3.64 / 2 = **1.82 s** | +24% |
| 1344×768, 39 | 5.25 s | 12.37 / 2 = **6.18 s** | +18% |
| 1344×768, 124 | 30.32 s | **33–38 s (guess)** | +10–24% |

- **The 124-frame row is a guess.** DiR was never timed there; DiR + DWD ran out of memory. The range assumes the LoRA's overhead falls as attention (quadratic in 37k tokens) takes over. Memory is also unmeasured: pi-H3 peaks at 27.0 GB there with its adapter merged, and DiR carries 1.2 GB of unmerged LoRA on top.
- **End to end at 768×512×39,** adding a decoder:

| Stack | Total |
|---|---|
| pi-H3 2-step + its DWD (measured) | 1.55 s |
| DiR R = 2 + pi-H3's decoder refitted on it (0.07 s, [`DWD_PIH3.md`](https://github.com/scrya-com/dir-diffusion/blob/feat/dwd-pih3/docs/DWD_PIH3.md)) | about 1.89 s |
| DiR R = 2 + our DWD v0.2 (0.22 s rerun of the last 8 blocks + 0.26 s) | about 2.30 s |
| DiR R = 2 + VAE (1.77 s) | about 3.59 s |
| DiR R = 2 with no LoRA overhead + pi-H3's decoder | about 1.55 s (hypothetical) |

### The gap is the unmerged LoRA

- At this clip shape, base H3 at 4 Euler steps takes 2.96 s and DiR's 4 iterations 3.6 s (`DIR.md`, video part 1): **0.74 s against 0.90 s per forward**. pi-H3's merged forward is 0.735 s.
- On stills the same comparison is 81 ms against 97 ms, +20% (`DIR.md`, speed).
- So the rank-128 LoRA on qkv, out_proj, fc1 and fc2 of all 50 blocks costs about 0.16 s per forward, 0.33 s over 2 iterations. pi-H3's adapter is rank 64 on out_proj, fc1 and fc2 only, and merged.

### Merging it into the int8 weights does not work as it stands (measured, CPU)

`PIH3.md` found that an exact int8 merge lands 1.3% away per layer, as large as pi-H3's whole LoRA (1.7%). The same check on our weights, with musubi's deterministic ConvRot requantisation, on out_proj and fc2 of blocks 0, 10, 25 and 40:

| LoRA | Size of the LoRA, relative to the layer | int8 codes changed by the merge | Direction kept (cosine) |
|---|---|---|---|
| pi-H3 2-step adapter (rank 64) | 0.6–1.5% | 1.5–16% | 0.32–0.83 |
| **Published DiR generator (rank 128)** | **0.07–0.18%** | **0–0.008%** | **0.000–0.012** |
| Turbo 4-step init (rank 128) | 0.07–0.17% | 0–0.008% | 0.000–0.011 |

- **Our LoRA is a few percent of one int8 step per weight** (estimated from the 1.3% requantisation noise), so deterministic rounding returns the base model's codes. The merged model is base H3. pi-H3's adapter is 5–10 times larger and survives.
- **ComfyUI's stochastic rounding** keeps a small delta on average, but with per-weight noise several times the delta. Whether a generator survives that is untested. **(guess: not without a quality loss)**
- **Not sampled:** fc1, and most blocks. A first pass that included qkv and block 49 gave 0.1–0.2% on blocks 0–40 and 0.5% on block 49.
- **Ways out, none tried (guesses):** a lower rank, or dropping qkv as pi-H3 does, for a cheaper unmerged LoRA; training with the requantised forward in the loop; float weights for the adapted layers (the 40 GB bf16 base does not fit this card).

So "similar speed" does not hold today: 2-iteration DiR is 0.35 s behind pi-H3 2-step at 768×512×39 before any decoder.

## 5. Verdict

**Is a 2-iteration DiR likely to beat or match pi-H3 2-step at similar speed? No.**

- **Speed:** 18–24% slower in the DiT, and the merge that would close it erases the LoRA (measured).
- **Quality (guess: under 10% to beat, 15–20% to match by eye on our prompts).** Reasons:
  - No run from Turbo has beaten Turbo at 4 forwards yet (`DIR.md`, part 4). The published checkpoint trades over-motion for over-detail. Asking for the same quality in half the forwards is harder.
  - The budget is 2 samples per update for 50–200 steps on 793 single-style prompts with a rank-128 LoRA. The paper trains at batch 256 for 40 epochs with full weights.
  - pi-H3 2-step does not have a good latent either: its VAE decode is blocky (LPIPS 0.396 against its own DWD pixels), and its quality comes from a DWD head trained with the generator. A DMD-trained 2-iteration DiR must make the latent itself a clean sample, which is a stricter target, and then still needs a decoder refit.
- **For it:** the paper's lead over a scheduled sampler is largest at 2 NFE, the 1 → 2 step is where most of the paper's gain is, each iteration gets twice the updates, and R = 2 on a grid is exactly the paper's functional form here.

**Is it worth the GPU time?** The 10-minute look and the 1.5-hour gate run, yes. A 200-step run plus a decoder (6–7 hours, **guess**) only if the gate passes.

### The cheapest experiments, in order

**0. No training, about 10 minutes of GPU: what do 2 iterations look like today?**

```bash
P=(--out /media/2TB/lora-data/h3_lot/runs/probe_r2 --frames 12 --max-latent-hw 32 48 --items 3 --swap 4)
python3 lot/probe_dir_h3.py $P --name ref --reference
python3 lot/probe_dir_h3.py $P --name pub4 --run lot/runs/dir_turbo_clip                               # published, all 4: read its r = 2 column
python3 lot/probe_dir_h3.py $P --name pub2 --run lot/runs/dir_turbo_clip --iterations 2 --sigmas 1,0.857
python3 lot/probe_dir_h3.py $P --name turbo2 --run ../lot-diffusion/lot/runs/turbo4_init_run --iterations 2 --sigmas 1,0.857
python3 lot/probe_dir_h3.py $P --decode pub2      # and pub4, turbo2
```

- `pub4`'s second column is the 2-of-4 prefix judged as a clip, which the 0.462 number never did.
- `pub2` and `turbo2` are step 0 of the two candidate starts at the proposed levels.
- Compare by eye with pi-H3 2-step on the same prompts (`pih3_h3.py sample --steps 2` on `feat/pi-h3`).
- Commands are from reading the code, not run. `pub2` uses the published `r_embed` row 1 at 0.857 where it was trained at 0.947; the row's norm is 6e-4, so this should not matter. **(guess)**

**1. The gate run, about 1.5 GPU-hours:** the command in section 2 (50 steps from Turbo), or the same from the published LoRA if `pub2` looks better than `turbo2` in experiment 0.

- Stop early if `hat_rms` on `r = 0` steps trends above 1.12 or eval `hat_real[0]` reaches 0.21 (the cheatsheet's rule).
- Pass, at step 25 or 50: eval `motion[1]` falling from Turbo's level toward 0.23; eval `hat_real[1]` at or under 0.15 (the published run's iteration 2 is 0.151, its iteration 4 is 0.143; the threshold is a **guess**); and the decoded 2-iteration clips as sharp and on-prompt as pi-H3 2-step's on the three probe prompts.
- The published start carries a known risk: its live weights were still losing motion at step 100 (eval motion 0.17 against the teacher's 0.23), so more steps may go too still.

**2. Only if 1 passes:** 200 steps, then refit pi-H3's decoder on the 2-iteration tokens (`train_pih3_dwd.py` on `feat/dwd-pih3`), then the `accel_h3.py bench` row against pi-H3 2-step.

## Guesses, in one place

- The R = 2 step time (69 s) and memory are read off the R = 4 log, not measured at R = 2.
- The warm-up time, the overhead of loading and evals, and the decoder refit time.
- The 124-frame speed and whether it fits in memory.
- That a stochastic-rounding merge would hurt; that a lower-rank LoRA would close the speed gap.
- That 0.667 as the second level would only polish.
- Every probability in the verdict.
- Whether a native R = 2 model beats an R = 4 model's 2-step prefix: the paper does not test it, and neither have we.

## Sources

- Paper: Shanchuan Lin, Yansong Peng, Fu-Yun Wang, Haoqi Fan (ByteDance Seed), "Few-Step Generation via Data-Space Iteration", [arXiv 2610.12102](https://arxiv.org/abs/2610.12102) v1. Sections 2.3–2.6, 3.1–3.4; Tables 1–3, 5, 6, 8; appendices A, B, E.
- Code on `main`: [`lot/dir_h3.py`](../lot/dir_h3.py) (`parse_sigmas`, `next_state`, `generator_step`, `DirHead`), [`lot/train_dir_h3.py`](../lot/train_dir_h3.py), [`lot/turbo_lora.py`](../lot/turbo_lora.py), [`DIR.md`](DIR.md), [`DIR_CHEATSHEET.md`](../DIR_CHEATSHEET.md).
- Published run: [`lot/runs/dir_turbo_clip`](../lot/runs/dir_turbo_clip/config.json) ([Hugging Face](https://huggingface.co/johndpope/dir-diffusion-h3)); its training log at `/media/2TB/lora-data/h3_lot/runs/dir_turbo_clip_iter/log.jsonl`.
- `feat/accel`: `docs/ACCEL.md` (the 0.462 early-exit floor). `feat/pi-h3`: `docs/PIH3.md`, `docs/pih3_bench_raw.md`, `lot/pih3_h3.py`. `feat/dwd-pih3`: `docs/DWD_PIH3.md`.
- Turbo releases: the `lightx2v/Minimax-h3-Turbo` file list has 4-step and 8-step LoRAs only, no 2-step.

## Result: the published DiR at 1, 2 and 4 iterations, judged by eye (25 November)

Same noise per prompt, 4 held-out prompts (isometric scenes and generated portraits), 768x512x39, VAE decode. Columns: **4 iterations | 2 iterations | 1 iteration | pi-H3 2-step** (pi-H3 has its own noise, so its framing differs).

![DiR at 4, 2 and 1 iterations next to pi-H3 2-step](dir_r2_media/dir_1_2_4_vs_pih3.jpg)

Chef clip, side by side (DiR 4 | DiR 2 | DiR 1 | pi-H3 2-step): [`chef_4_2_pih3.mp4`](dir_r2_media/chef_4_2_pih3.mp4).

- **2 iterations is a good clip.** Sharp and coherent, the same scene, people and composition as 4 iterations, a little simpler (fewer small objects, a less detailed chef). This is the paper's prefix property (Sec. 2.6). The LPIPS 0.38 between the 2- and 4-iteration clips (and the older 0.462 in ACCEL.md) measures that they are different valid clips, not that the 2-iteration one is damaged.
- **1 iteration is a usable draft with artefacts:** a different face and wardrobe, a duplicated chef, softer and blurrier fire. Matches the paper's FID 3.85 at 1 NFE against 2.68 at 2.
- **Skipping the middle iterations is worse than stopping early** (2 forwards, LPIPS against the 4-iteration clip): stop after 2 = 0.382, iteration 1 then 3 = 0.441, iteration 1 then 4 = 0.475 (`lot/probe_dir2.py`).

| DiT time, 768x512x39, RTX 5090 | |
|---|---|
| DiR 4 iterations | 3.64 s |
| **DiR 2 iterations** | **~1.82 s** |
| DiR 1 iteration | 0.91 s (measured) |
| pi-H3 2-step | 1.47 s |

**What needs training before DiR's 2-iteration mode is a fast path:** the fast DWD decoder (Stage 1, `feat/dwd-pih3`, 0.07 s) was fitted to iteration-4 tokens; 2-iteration DiR runs with the VAE decode (1.8 s) until it is refitted on iteration-2 tokens (about 2 GPU-hours, estimated). A native 2-iteration training run (section 5) is optional: the published weights already give a good 2-iteration clip.
