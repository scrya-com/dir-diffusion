# DiR + DWD end to end, at any resolution

How fast a clip gets from noise to pixels on MiniMax H3 with the [DiR generator](DIR.md) and the [DWD decoder](DWD.md), and what is being trained to make it faster. Branch `feat/accel`. Code: [`lot/accel_h3.py`](../lot/accel_h3.py), [`scripts/bench_accel.sh`](../scripts/bench_accel.sh), [`scripts/weekend_accel.sh`](../scripts/weekend_accel.sh).

**Status (2026-11-20):** the code and both benchmarks run; every training phase has passed a smoke test; the full run has not been done. The numbers below are the published stack (LoRA `wB`, head `wB_head`, refiner `wG`) used as it is, at sizes it was never trained on.

## Reproduce

```bash
# setup as in the README (LOT_H3_CHECKPOINT, LOT_H3_VAE); the DWD stack is not published yet: LORA=, HEAD=, REFINER=
python3 lot/accel_h3.py check --lora LORA --head HEAD            # GPU parity -> ok=1
bash scripts/bench_accel.sh speed                                # -> runs/bench_accel/speed/BENCH.md
bash scripts/bench_accel.sh quality                              # -> runs/bench_accel/quality/QUALITY.md
SIZES="640x384 1280x704" QSIZES="640x384 960x544" bash scripts/bench_accel.sh      # any multiple of 32
```

A size is `WxH` in pixels. Each side is a multiple of 32 (one token); the half-resolution and two-track rows need multiples of 64. A clip is any length H3 decodes to (1, 5, 22, 39, 56, … frames; `--frames`). Nothing in the head, the refiner or the tools holds a canvas.

## Speed (RTX 5090, 39 frames, medians of 2 warm prompts)

| Canvas | Variant | Total s | DiT s | Decode s | × real time | LPIPS vs VAE | Peak GB |
|---|---|---|---|---|---|---|---|
| 768x512 | DiR, VAE decode | 5.49 | 3.72 | 1.78 | 0.30 | | 27.2 |
| 768x512 | **DiR + DWD** | 4.19 | 3.93 | 0.26 | 0.39 | 0.100 | 24.9 |
| 768x512 | two-track fast path, VAE | 3.08 | 1.50 | 1.58 | 0.53 | | 26.6 |
| 768x512 | two-track, DWD on the LR track, VAE on the region | 2.93 | 1.61 | 1.31 | 0.55 | 0.183 | 28.6 |
| 768x512 | **two-track, DWD on both tracks** | 1.84 | 1.61 | 0.23 | 0.88 | 0.219 | 24.9 |
| 768x512 | DiR half res, VAE decode, ×2 | 1.22 | 0.92 | 0.30 | 1.33 | | 26.4 |
| 768x512 | **DiR half res + DWD, ×2** | 1.02 | 0.96 | 0.06 | 1.59 | 0.223 | 23.6 |
| 1344x768 | DiR, VAE decode | 16.77 | 12.58 | 4.19 | 0.10 | | 29.4 |
| 1344x768 | **DiR + DWD** | 13.95 | 13.27 | 0.68 | 0.12 | 0.108 | 28.4 |
| 1344x768 | two-track fast path, VAE | 7.57 | 4.26 | 3.32 | 0.21 | | 27.6 |
| 1344x768 | two-track, DWD on the LR track, VAE on the region | 5.87 | 4.22 | 1.65 | 0.28 | 0.119 | 30.1 (ran out of memory on 1 of 3 prompts) |
| 1344x768 | **two-track, DWD on both tracks** | 5.07 | 4.57 | 0.49 | 0.32 | 0.181 | 26.6 |
| 1344x768 | DiR half res, VAE decode, ×2 | 3.62 | 2.42 | 1.20 | 0.45 | | 27.3 |
| 1344x768 | **DiR half res + DWD, ×2** | 2.88 | 2.53 | 0.35 | 0.56 | 0.169 | 25.0 |

- **Total** is DiT + decode with the pixels left on the GPU. **× real time** is the clip's 1.6 s at 24 fps over the total.
- **LPIPS vs VAE** compares the DWD pixels with what the VAE gives for the same latent: the same ×2 for the half-resolution rows, the VAE composite of the same two latents for two-track. It says how well DWD stands in for the VAE on that path. It does not compare paths: half resolution and two-track are different videos from full-resolution DiR.
- **DiR + DWD** runs the last 8 of 50 blocks twice in the final iteration (once as the generator, once with the DWD LoRA), which is the 0.2 s its DiT column has over the VAE row's.
- The two-track region here is 23–24% of the tokens, on other prompts than [`TWO_TRACK.md`](TWO_TRACK.md)'s.

What the table says:

- **With DWD the DiT is 90–97% of the time.** The decoder is no longer where a second is hiding. What is left is the number of tokens and the number of iterations.
- **Tokens are the big lever.** One iteration costs 0.93 s at 768×512 (4,608 tokens) and 0.23 s at half that resolution; 3.1 s and 0.6 s at 1344×768.
- **Half resolution with a pixel ×2 is faster than real time at 768×512** (1.0 s for a 1.6 s clip), and it does not need DWD to get there (1.2 s with the VAE: a 384×256 decode takes 0.3 s). `TWO_TRACK.md`'s "half res, upscaled" row is slow only because it upscales through the VAE.
- **Two-track loses its decode to the region's VAE crop** (1.3–3.3 s). Decoding the region's HR tokens with the same DWD head and refiner (`region_decode`) brings it to 1.8 s and 5.1 s, with no VAE on the card. Untrained for it: LPIPS 0.18–0.22 against the VAE composite over the whole frame. Most of that is the LR track on its small grid, as in the half-resolution row; scored over the region's box alone (6 held-out clips) the head gives 0.119 and the published refiner 0.079.
- **Memory:** the VAE is 4.9 GB of weights. A 32 GB card holds the DiT (20.6 GB) with the VAE or with the DWD stack (1.7 GB), so the DWD rows also run with more room.

## Quality at sizes it was not trained on

The published stack, unchanged, on 4 held-out prompts per size; LPIPS against the VAE decode of the same latent:

| Size | Head | Head + refiner | PSNR dB | Decode s |
|---|---|---|---|---|
| 768x512 (trained) | 0.292 | 0.143 | 27.9 | 0.28 |
| 512x768 (trained) | 0.292 | 0.139 | 27.3 | 0.27 |
| 512x512 (trained) | 0.304 | 0.152 | 26.9 | 0.17 |
| 1024x576 | 0.290 | 0.139 | 28.9 | 0.40 |
| 576x1024 | 0.331 | 0.165 | 27.5 | 0.40 |
| 1344x768 | 0.307 | 0.145 | 28.9 | 0.69 |
| 672x384 | 0.297 | 0.145 | 27.7 | 0.17 |
| 384x256 | 0.332 | 0.183 | 24.5 | 0.06 |

These rows fed the refiner the replayed estimate, which costs about 0.015 LPIPS (see the next section); the trained sizes score 0.114 on the published held-out set when it reads the rollout's own latent, as it does at inference.

- **DWD already carries across resolution.** H3 normalises its rotary positions by image area, so the token features do shift, but the head loses only 0.00–0.04 LPIPS and the refiner, which is convolutional and reads the latent, takes most of that back.
- **The weak spots are the smallest grid and tall clips:** 384×256 (12×8 tokens, the LR track of a 768×512 two-track clip) and 576×1024.
- **bf16 is safe.** The refiner in bf16 autocast and the head in fp16 score the same as fp32 (iso 27.07 against 27.07 dB, LPIPS equal to 4 places).

## The DiT is numerically chaotic, and what that means here

- `check` runs the same 4-iteration rollout from the same noise rounded to fp16. The final latent moves by **52–56%** (relative RMS). One iteration replayed from its fp16-stored inputs moves its estimate by about **10%**.
- Decoded by the real VAE, that replayed estimate is LPIPS **0.05–0.08** away from the original decode (27–38 dB). That is the noise floor of any score that replays an iteration.
- **Refiner latent.** The training targets are the VAE decode of the collected rollout's latent, so the refiner must read that latent, not a replayed one. Fed the replayed one, the published stack scores 25.4 dB instead of 27.1 dB on the isometric clips. `accel_h3.py` keeps the rollout's own estimate with every sample (`stored_latent`). At inference the tokens and the latent come from one forward, so the question does not arise.
- **Early exit.** A decoder that stops after iteration 3 is asked for the pixels of iteration 4, which is a chaotic function of what it has. `quality --exit E --vae-exit` measures what the skipped iterations change, with the real VAE on the exit's own estimate (4 held-out clips, 768×512):

| Stop after | VAE of that estimate, against the full rollout's | DiT time saved |
|---|---|---|
| iteration 3 of 4 | 19.95 dB, LPIPS **0.311** | 27% |
| iteration 2 of 4 | 16.00 dB, LPIPS **0.462** | 52% |

For scale ([`DWD.md`](DWD.md)'s calibration): a σ=2 blur of the VAE's frames is LPIPS 0.29, half resolution upscaled 0.12. The fourth iteration is not a polish, so an early-exit stack has to do its work in 8 LoRA blocks, a linear head and a small refiner, and part of what it adds cannot be predicted.

## What is being trained (`scripts/weekend_accel.sh`)

Ranked by expected speed-up × chance of working. "Data" is new rollouts.

| # | Idea | Speed-up | Needs | Chance | Gate |
|---|---|---|---|---|---|
| 1 | **DWD at any size** (the enabler): published stack as it is, then one refiner fine-tuned on 13 sizes | decode 1.8 → 0.26 s, 4.2 → 0.68 s; 1.3× end to end at full res | nothing; then data + refiner training (no DiT in the loop) | works today; fine-tune about 80% to improve it | replaces the published refiner only if it is better on the mean of the new sizes |
| 2 | **Half res + DWD, pixel ×2** | 5.4× at 768×512, 5.8× at 1344×768 against DiR + VAE | nothing (a benchmark row); #1's refiner is trained on the LR grids, where DWD is weakest | runs today | a different, softer video: judged by eye against full res |
| 3 | **Two-track, DWD on both tracks** | 3.1 → 1.8 s, 7.6 → 5.1 s against two-track with the VAE | nothing to run; #1's refiner also trains on two-track region pairs (`tt-pairs`, loss counted on the region) and on the LR grids | runs today; about 70% to bring the whole frame from 0.18–0.22 to 0.15 | benchmark row, before and after |
| 4 | **Early exit** after iteration 3 of 4 | DiT ×0.73: 4.2 → 3.1 s at 768×512 | data, head refit, last-8-blocks LoRA, refiner | about 15% to reach 0.16; about 50% to land at 0.20–0.24 | floor ≤ 0.40; head-level LPIPS ≤ 0.42 after 90 min of LoRA; ≤ 0.16 "accepted", ≤ 0.24 "usable trade" |
| 5 | **Early exit** after iteration 2 | DiT ×0.48 | as #4 | under 5% (floor 0.46) | off unless `E1=1` and #4 was accepted |
| 6 | **Cheaper refiner** | at most 0.2 s of 4.2 s | a distilled half-width refiner | high, but worth 5% | not scheduled. Batch size by clip size is in (`refiner_chunk`): same speed, 60% of the memory |

Why the others are not in the run:

- **A DWD LoRA on two-track mixed sequences.** The LR track's tokens in the mixed sequence decode as well as plain half-resolution tokens do (LPIPS 0.101 against 0.102 at 672×384, same prompt), so the LR track needs nothing. The region's HR tokens are what `tt-pairs` trains the refiner on; a LoRA for them would need the mixed forward inside the LoRA trainer.
- **LoRA or head refit above 4,608 tokens a clip.** Both peak at 29.7 GB at 768×512 with every block resident, and block swap crashes with a stacked LoRA. So the LoRA and the head are fitted on the sizes up to 768×512 and applied to the larger ones as they are; the refiner, which trains on crops with no DiT, sees every size.
- **A learned ×2 for the half-resolution path** (a refiner that also upscales). The honest target is real full-resolution video of the same content, which a half-resolution rollout does not have; pairs made by downscaling full-resolution clips would train a plain super-resolution net. Not built.
- **Fewer tokens at full quality** needs the generator trained on the mixed or cascaded sequence (DiR distillation), which is a different run.

Phases, each with a time budget and a gate; launching again resumes (`accel/state/*.done`, `*_last.pt`, existing clips):

| Phase | What | Budget |
|---|---|---|
| preflight | disk, weights, CPU tests, `check` | 5 min |
| baseline | quality per size, the early-exit floor, the speed table | 1 h |
| collect | 3,720 training rollouts over 13 sizes + 8 held out per size; inputs of iterations 4 and 3; VAE targets | 8.5 h |
| mr | refiner cache (3 h), two-track pairs (1.8 h), refiner from the published one (10 h), gate | 15.5 h |
| e2 | refit (1.5 h), LoRA (5 h, gated at 90 min), refit, cache (3 h), refiner (12 h), gate | 23.5 h |
| mr_more | the hours the gates left over: the mr refiner on at lr 5e-5, LPIPS weight 1, then its gate again | rest |
| final | both tables with every stack that passed, `accel/FINAL.md` | 1.3 h |

## Smoke test of the run (2026-11-20, RTX 5090)

`SMOKE=1 scripts/weekend_accel.sh` ran every phase end to end in 38 minutes on 20 training clips with 3-minute budgets, then again from its own state to check that it resumes. These are plumbing checks, not forecasts.

| Phase | What happened |
|---|---|
| preflight | CPU tests pass; `check` ok: the two-pass tokens equal the one-pass ones exactly (relative gap 0) |
| baseline | quality 0.131 on two sizes; floors 0.311 and 0.462; speed table as above |
| collect | 30 clips over 5 sizes in 2.6 min, 230 MB with three exits stored |
| e2 | head refit on 10 clips; LoRA 95 steps in 3 min at 29.8 GB peak; refit; cache of 30 pairs; refiner 533 steps: held-out LPIPS 0.44 → 0.38; `quality` agrees with the trainer's own eval (0.385 against 0.383) |
| mr | cache of 30 pairs + 4 two-track pairs merged with the published run's 1,016; step 0 reproduces the published refiner (0.1325); 3 min later 0.1324 |
| mr_more | 2 more minutes at lr 5e-5, gate again: 0.1320 |
| final | all eight rows, the early-exit row included (3.22 s at 768×512: DiT 2.96 s, decode 0.26 s), and `FINAL.md` |
| resume | second launch skipped every finished step and continued |
| largest sizes | 1344×768, 768×1344 and 1280×704 through collect (29.5 GB peak with the VAE beside the DiT), both caches, two-track pairs (26.6 GB), `quality` and a refiner run that evaluates on them (17.2 GB) |
| two-track pairs | 26 clips at 768×512 and 1024×576 in 3.5 min; the published refiner on the region's box, loss and score counted on the region: LPIPS 0.079 at step 0, 0.062 after 234 steps on 20 clips (PSNR 29.2 → 30.5 dB) |

## Files

| Path | What |
|---|---|
| [`lot/accel_h3.py`](../lot/accel_h3.py) | `DwdStack`, `exit_step`, `dir_rollout`, `region_decode`; phases `check`, `collect`, `cache`, `merge-pairs`, `quality`, `bench`, `table` |
| [`lot/test_accel_h3.py`](../lot/test_accel_h3.py) | CPU tests for everything that depends on size or clip length |
| [`scripts/bench_accel.sh`](../scripts/bench_accel.sh) | One command per benchmark |
| [`scripts/weekend_accel.sh`](../scripts/weekend_accel.sh) | The supervisor; `SMOKE=1` runs every phase with tiny budgets |
| `lot/h3_musubi.py`, `lot/two_track_h3.py` | `rerun_blocks` and the `dwd=` tail: the last K blocks again with the DWD LoRA, in the dense and the two-track forward |
| `lot/refine_dwd_h3.py` | crops for mixed sizes (one size per batch), `pairs.json` may name the refiner's latent |
