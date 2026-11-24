# H3-Tiny's block drop on our fast paths: a training-free probe

**Status: prototype on `exp/h3tiny-gbuffer`. Nothing is trained, no DCA.** Verdict at the end: the 8-block drop is a real 15–16% and is not free; 40 blocks is no worse than 42 here; 35 blocks breaks.

**H3-Tiny** is by **Haoshuai Fu and Bin Chen** ([code](https://github.com/fuyi25/MiniMax-H3-Tiny), [project page](https://fuyi25.github.io/H3-Tiny/), [weights](https://huggingface.co/H3-Tiny)). Code: Apache-2.0. Weights: MiniMax H3 Community License (Model Derivatives of MiniMax H3; the licence excludes the EU, the UK, South Korea and the USA). This probe is ours and is not affiliated with the authors.

## What H3-Tiny is, and what is used here

| Their stage | What it does | Here |
|---|---|---|
| AdaLN low-rank | Rank-8 fit of the per-block AdaLN projections, no data: 33.1B → about 20B | Already in our int8 ConvRot checkpoint (pruned AdaLN) |
| **Block removal** | Removes 8 of 50 blocks (**6–11, 13, 15**), chosen by block influence with the audio-critical blocks protected: 17.1B, **no training** | **`--plans h3tiny`: the same blocks skipped on our weights** |
| FFN narrowing + OKD | 14336 → 9792 channels, recovered by distillation from the 33B: 14.0B | Not used |
| Few-step distillation | `H3-Tiny-17B-FewStep-Ref2VA-v0.1`: 6 steps by default, 4 or more | Not used: our few-step paths are pi-H3 (2/4 steps) and DiR (4 iterations) |

**Their published claims.** Parameter counts (33.1B → 17.1B → 14.0B), "the 17B works well straight out of pruning, with no training at all", and few-step sampling in 6 steps (4 or more). The README and the project page give **no speed, memory or quality numbers**: the evidence is side-by-side galleries. Recovery and few-step training code are not released yet.

**Their checkpoints were not downloaded.** Each is 34.2 GB (bf16, diffusers format), and /media/2TB has about 60 GB free. They are also not needed: the depth plan is 8 block indices, and the pi-H3 adapter and the DiR LoRA are trained on our int8 weights, not on theirs.

**What already existed here** (reused, not re-derived):
- `LOT_H3_DROP_BLOCKS` and `blocks_dropped` from `exp/h3-tiny-drop` (ported into `lot/h3_musubi.py`).
- The RTX 5090 times with the drop from `exp/pih3-cascade` and `exp/h3-tiny-drop` (quoted below as "measured").
- The per-block sensitivity scan of `pih3_cascade.py dropscan` (pi-H3 4-step, 2 prompts), whose order gives the two deeper plans.
- `distill_tiny_h3.py`: 7 hours of self-distilling DiR onto the 42-block model **failed its pass check** (706 steps; DWD LPIPS went from 0.11 to 0.50).

## Plans

| Plan | Blocks kept | Blocks skipped |
|---|---|---|
| `full` | 50 | none |
| `h3tiny` | 42 | 6, 7, 8, 9, 10, 11, 13, 15 |
| `least10` | 40 | 4, 6, 8, 11, 13, 15, 16, 17, 19, 23 |
| `least15` | 35 | `least10` + 10, 20, 26, 40, 47 |

`least10` and `least15` are the 10 and 15 least sensitive blocks of the existing scan. The scan is from pi-H3 4-step on 2 prompts, so both plans are guesses for DiR.

## Files and commands

| File | What |
|---|---|
| [`lot/h3_musubi.py`](../lot/h3_musubi.py) | `H3_TINY_DROP`, `blocks_dropped`, `LOT_H3_DROP_BLOCKS` (ported) |
| [`lot/tiny_h3.py`](../lot/tiny_h3.py) | `plan`, `quality` (pi-H3 or DiR, any plan), `resident` (per-block times) |
| [`lot/test_tiny_h3.py`](../lot/test_tiny_h3.py) | 5 CPU tests |
| [`docs/h3tiny_results/`](h3tiny_results/) | Raw JSON and one contact sheet |

```bash
CUDA_VISIBLE_DEVICES="" python3 lot/test_tiny_h3.py                         # last line "ok 5"
cd lot && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0        # the RTX PRO 4000
python3 -u tiny_h3.py resident --sizes 512x288x12,768x512x12 --swap 16 --out OUT/resident.json
python3 -u tiny_h3.py quality --generator pih3 --steps 2 --size 768x512 --prompts 3 --swap 24 --vae --out OUT
python3 -u tiny_h3.py quality --generator dir --size 768x512 --prompts 3 --swap 36 --out OUT
```

`LOT_H3_DROP_BLOCKS` stays unset by default, so every other script runs all 50 blocks.

## CPU tests

| Test | Result |
|---|---|
| Plans | `h3tiny` is 6–11, 13, 15 (42 kept); `least10` keeps 40, `least15` 35; `least8` equals the cascade branch's set; bad specs raise ✅ |
| Nothing dropped | Bit-identical to the plain forward ✅ |
| Dropped blocks | Bit-identical to the stack with those blocks removed; all 8 indices stay in place; the state is restored after the context, also when nested ✅ |
| A dropped block does no work | Its attention is never called ✅ |
| `LOT_H3_DROP_BLOCKS` | Unset = none; `h3tiny`; index lists ✅ |

## Quality against the 50-block model (RTX PRO 4000)

768×512, 39 frames, 3 held-out prompts, same noise for every plan. **Floor** is the 50-block model again from the same noise rounded to bf16: how far a change that should do nothing moves the clip. Colour shift is the mean absolute difference of the per-frame mean colour, as a fraction of the range. Detail is high-frequency energy relative to the 50-block clip.

**pi-H3 2-step, its own DWD pixels:**

| Plan | Kept | LPIPS vs 50 blocks (per prompt) | Colour shift | Detail | DWD vs VAE of the same latent, LPIPS |
|---|---|---|---|---|---|
| full | 50 | | | 1.00 | 0.382 |
| floor | 50 | 0.210 (0.335, 0.065, 0.230) | 0.009 | 1.02 | 0.386 |
| h3tiny | 42 | 0.413 (0.498, 0.197, 0.543) | 0.050 | 1.02 | 0.391 |
| least10 | 40 | 0.394 (0.473, 0.180, 0.529) | 0.043 | 0.94 | 0.386 |
| least15 | 35 | 0.485 (0.571, 0.298, 0.588) | 0.081 | 0.91 | 0.452 |

**DiR, 4 iterations, VAE decode:**

| Plan | Kept | LPIPS vs 50 blocks (per prompt) | Colour shift | Detail |
|---|---|---|---|---|
| floor | 50 | 0.384 (0.494, 0.134, 0.524) | 0.007 | 0.99 |
| h3tiny | 42 | 0.502 (0.608, 0.239, 0.660) | 0.040 | 1.07 |
| least10 | 40 | 0.498 (0.589, 0.266, 0.640) | 0.033 | 0.92 |
| least15 | 35 | 0.604 (0.683, 0.387, 0.743) | 0.060 | 1.31 |

- **The drop gives another clip, not the same clip.** LPIPS against the 50-block output is about 2× the floor for pi-H3 and 1.3× for DiR (whose floor is already high).
- **The colour shift is real and is 5× the floor**: 0.04–0.05 on average and up to 0.10 on one prompt, against 0.007–0.009. This matches the 0.06–0.10 the cascade branch measured on the 5090.
- **By eye** ([sheet](h3tiny_results/pih3_s2_p2_strip.jpg)): `h3tiny` and `least10` are clean clips with a different layout, a brighter palette and changed object colours (a green sofa turns brown). `least15` breaks: smeared figures, soft detail.
- **40 blocks is no worse than 42 here** on LPIPS and colour, with slightly less detail (0.92–0.94). Three prompts cannot rank them.
- **The DWD head still works on the dropped stack:** DWD against the VAE decode of the same latent is 0.39 at 42 and 40 blocks, as at 50. At 35 blocks it degrades (0.45).

## Timings

**Per-block times without streaming** (RTX PRO 4000, 30 blocks that stay on the card, run on the real stack inputs, medians of 5):

| | 1,986 tokens (512×288×12) | 4,866 tokens (768×512×12 = 39 frames) |
|---|---|---|
| Per block | 13.63 ms | 47.16 ms |
| Per block with every other block skipped | 13.68 ms | 48.78 ms |
| Cost of a skipped block | 0.05 ms | 1.6 ms (inside the 3% run-to-run spread) |

A skipped block costs nothing measurable, so DiT time is `kept blocks × per-block time`. The same blocks measured 13.65 and 45.78 ms in the DCA run earlier the same day.

**Scaled to the RTX 5090.** Assumptions:
- **A1.** DiT time is proportional to the blocks kept (embedding and final layer ignored). Checked against three 5090 measurements below: it holds to 1–2%.
- **A2.** The baselines are the measured 5090 numbers of [`PIH3.md`](PIH3.md); the PRO 4000 is 3.2× slower per block (47.16 / 14.7 ms) and is used only to show that skipping is free.
- **A3.** Decode time does not change (0.08 s DWD; 0.63 s at 124 frames; 1.78 s VAE for DiR).
- **A4 (guess).** Memory falls 0.42 GB per block **if the weights are removed**. This prototype keeps them loaded, so its peak is unchanged.
- **A5 (guess).** 1344×768×124 with dropped blocks was never run; only the 50-block time is measured.

| Case, RTX 5090 | 50 blocks | 42 (h3tiny) | 40 | 35 |
|---|---|---|---|---|
| pi-H3 2-step, 768×512×39 | 1.55 s | 1.31 s (−15%); **measured 1.31–1.32** | 1.26 s (−19%) | 1.11 s (−28%) |
| pi-H3 4-step, 768×512×39 | 3.06 s | 2.58 s (−16%); **measured 2.61** | 2.46 s (−19%) | 2.17 s (−29%) |
| pi-H3 2-step, 1344×768×124 | 30.95 s | 26.10 s (−16%) | 24.89 s (−20%) | 21.85 s (−29%) |
| DiR 4 iterations, DiT only | 3.64 s | 3.06 s; **measured 3.08** | 2.91 s | 2.55 s |
| DiR + VAE, 768×512×39 | 5.42 s | 4.84 s (−11%); **measured 4.86** | 4.69 s (−13%) | 4.33 s (−20%) |
| DiR + DWD, 768×512×39 | 4.12 s | 3.50 s (−15%); **measured 3.55** | 3.35 s (−19%) | 2.96 s (−28%) |
| Peak at 1344×768×124, weights removed (A4) | 27.0 GB | 23.6 GB | 22.8 GB | 20.7 GB |

- **Break-even:** none needed. There is no overhead, so every dropped block saves 2% of DiT time. (DCA, by contrast, needed fewer than 37 blocks to break even: `docs/DCA_H3.md` on branch `exp/dca-h3`.)
- **Memory:** 1344×768×124 fits a 32 GB card at every depth, with or without the weights removed (27.0 GB today).
- **Real time:** pi-H3 2-step at 42 blocks is 1.24× real time for a 1.625 s clip, against 1.05× at 50.

## Verdict

- **The 8-block drop is worth having as a switch, not as a default.** It is a measured 15–16% with no training and no overhead, and the DWD head survives it. It also changes the clip: about 2× the same-seed floor, a visible colour and palette shift, different objects.
- **40 blocks looks as good as 42** on this small probe (−19%). It needs more prompts, and a scan made for DiR, before it replaces H3-Tiny's plan.
- **35 blocks is too far without training.**
- **Recovery training is the open problem.** The one attempt here (7 hours, DiR self-distillation) failed, and H3-Tiny's own recovery code is not released. The published FewStep checkpoint is Ref2VA with 6 steps: 6 × 42 block passes against pi-H3's 2 × 50, so by block count alone it would be about 2.5× slower per clip (not measured). It is not a shortcut.
- **Every guess:** the 40- and 35-block plans (from a 2-prompt pi-H3 scan), the memory row (A4), the 124-frame rows (A5), and all quality conclusions (3 prompts, one seed each).
