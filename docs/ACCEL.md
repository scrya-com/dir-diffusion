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
- **Two-track loses its decode to the region's VAE crop** (1.3–3.3 s). Decoding the region's HR tokens with the same DWD head and refiner (`region_decode`) brings it to 1.8 s and 5.1 s, with no VAE on the card. Untrained for it: LPIPS 0.18–0.22 against the VAE composite.
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
- **Early exit.** A decoder that stops after iteration 3 is asked for the pixels of iteration 4, which is a chaotic function of what it has. So "LPIPS against the full rollout's VAE decode" has a floor well above zero for it. `quality --vae-exit` measures that floor with the real VAE on the exit's own estimate.

## What is being trained (`scripts/weekend_accel.sh`)

Ranked by expected speed-up × chance of working. "Data" is new rollouts; all of it comes from one `collect` pass.

| # | Idea | Speed-up | Needs | Gate |
|---|---|---|---|---|
| 1 | **DWD at any size** (the enabler): published stack as it is, then a refiner fine-tuned on 13 sizes | decode 1.8 → 0.26 s, 4.2 → 0.68 s; 1.3× end to end at full res | nothing; then data + refiner training (no DiT in the loop) | every size ≤ 0.15 LPIPS; the multi-res refiner replaces the published one only if it is better on the mean |
| 2 | **Half res + DWD, pixel ×2** | 5.4× at 768×512, 5.8× at 1344×768 against DiR + VAE | nothing (a row of the benchmark); #1's refiner helps the small grids | a different, softer video: judged by eye against full res |
| 3 | **Early exit:** decode after iteration 3 (then 2) of 4 | DiT ×0.73 (then ×0.48): 4.2 → 3.1 s (→ 2.1 s) at 768×512 | data, a head refit, a last-8-blocks LoRA and a refiner per exit | floor (VAE on the exit estimate) ≤ 0.30; head-level LPIPS ≤ 0.42 after 90 min of LoRA; final ≤ 0.16 against the full rollout |
| 4 | **Two-track, DWD on both tracks** | 3.1 → 1.8 s, 7.6 → 5.1 s against two-track with the VAE | nothing (benchmark row); training on mixed sequences would need a collector for them | not scheduled: half res is faster and the region is not clearly sharper yet |
| 5 | **Cheaper refiner** | at most 0.2 s of 4.2 s | a distilled half-width refiner | not scheduled: the refiner is 5% of the time. Batch size by clip size is in (`refiner_chunk`): same speed, 40% of the memory |

Why the others are not in the run:

- **A DWD LoRA on two-track mixed sequences.** The LR track's tokens in the mixed sequence decode as well as plain half-resolution tokens do (LPIPS 0.101 against 0.102 at 672×384, same prompt), so the LR track needs nothing. Only the region's HR tokens would gain, and two-track itself still has to show a sharper region.
- **LoRA or head refit above 4,608 tokens a clip.** Both peak at 29.7 GB at 768×512 with every block resident, and block swap crashes with a stacked LoRA. So the LoRA and the head are fitted on the sizes up to 768×512 and applied to the larger ones as they are; the refiner, which trains on crops with no DiT, sees every size.

Phases, each with a time budget and a gate; launching again resumes (`accel/state/*.done`, `*_last.pt`):

| Phase | What | Budget |
|---|---|---|
| preflight | disk, weights, CPU tests, `check` | 5 min |
| baseline | quality per size, the early-exit floors, the speed table | 1 h |
| collect | 4,000 rollouts over 13 sizes; inputs of iterations 4, 3 and 2; VAE targets | 8.5 h |
| e2 | exit after iteration 3: refit, LoRA (5 h), refit, cache, refiner (10 h), quality gate | 21 h |
| mr | multi-resolution refiner for the full rollout: cache, refiner from the published one (7 h), gate | 11 h |
| e1 | exit after iteration 2, only if e2 passed and 14 h are left | rest |
| final | both tables with every stack that passed, `accel/FINAL.md` | 1.3 h |

## Files

| Path | What |
|---|---|
| [`lot/accel_h3.py`](../lot/accel_h3.py) | `DwdStack`, `exit_step`, `dir_rollout`, `region_decode`; phases `check`, `collect`, `cache`, `merge-pairs`, `quality`, `bench`, `table` |
| [`lot/test_accel_h3.py`](../lot/test_accel_h3.py) | CPU tests for everything that depends on size or clip length |
| [`scripts/bench_accel.sh`](../scripts/bench_accel.sh) | One command per benchmark |
| [`scripts/weekend_accel.sh`](../scripts/weekend_accel.sh) | The supervisor; `SMOKE=1` runs every phase with tiny budgets |
| `lot/h3_musubi.py`, `lot/two_track_h3.py` | `rerun_blocks` and the `dwd=` tail: the last K blocks again with the DWD LoRA, in the dense and the two-track forward |
| `lot/refine_dwd_h3.py` | crops for mixed sizes (one size per batch), `pairs.json` may name the refiner's latent |
