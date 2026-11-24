# Alpha on pi-H3's DWD head (Stage A)

pi-H3 and DiR + DWD never run the H3 VAE decoder, so the P1 alpha branch on the VAE decoder (`alpha/h3_alpha.py`, `main`) is never called on the fast path. Stage A hangs the same kind of one-way branch off pi-H3 2-step's DWD refiner (`--pih3-steps 4` for the 4-step one) (`alpha/pih3_alpha.py`): it reads the refiner's final 16-pixel token states, and RGB is untouched.

## Inputs, from a probe

pi-H3 2-step on ten captions of the RGBA set put the subject on flat black (5), white (2), studio grey (1), a real scene (1), or filled the frame (2). No LayerDiffuse padding. So training clips composite the moving subject over dark / light / flat / gradient / scene / padded backgrounds (`alpha/data_pih3.py`), 768x512, 39 frames, noised to pi-H3 2-step's last-step sigma (default schedule and `max_raw_t` 1.0, the camera fix). The DiT and DWD head are frozen.

## Gate (set before the run)

On 60 held-out clips (the soft-heavy eval split, backgrounds cycled, both last-step sigmas):

- `soft_mae` < 0.10, `edge_mae` < 0.08;
- no background kind with `soft_mae` above 0.15;
- `rgb_psnr` (pi-H3's DWD RGB against the clip it was shown) reported, as the check that the teacher-forced input lines up with the alpha target;
- the head adds under 0.1 s to a 768x512x39 clip.

Miss: Stage B, a LoRA on pi-H3's last step only (bypass, never merged), EA loss plus an RGB-preservation term.

The 8 decoder blocks it reads are the same frozen blocks DiR's Stage 1 decoder uses (`feat/dwd-pih3`), so one alpha head may serve both; its transfer to DiR is to be tested, not assumed.

## Runs so far (pi-H3 2-step)

- `a3` (frame loss only, 25% of steps at the real last-step sigma): soft_mae stuck at 0.32 on every background at step 750. It learned the solid mask.
- `a4` (+ EA loss on the soft + edge band): soft_mae 0.291, edge_mae 0.301 at step 500; blotchy, token-grid alpha.
- **At the real last step (sigma 0.865) pi-H3 2-step redraws a different subject** (another ticket design, size and place), so the input's matte does not match even on hard edges there. Training at that sigma teaches the head to hedge; the eval at 0.865 is unscorable and is reported, not gated. The gate is judged on the sigma 0.6 eval.
- **Overfit test** (3 fixed clips, sigma 0.6, 300 steps): soft_mae 0.290 -> 0.120, edge_mae 0.388 -> 0.069, still falling. The head and its gradients work; `a4` was data (the 0.865 share) and a small, noisy step (one clip, 8 frames).
- Fair reference for matting from colour alone: P1's VAE head **without** its latent offset, about 0.21 (0.158 used the true matte in the latent).
- `a5`: `--real-sigma-frac 0 --accum 4`. If it overfits but stalls well above the gate, the next step is Stage B (a last-step LoRA so the DiT itself carries alpha), not more head tuning.

## Stage B: a last-step LoRA in a second pass (`alpha/train_stage_b.py`)

Stage A plateaued: a5's sigma-0.6 soft_mae went 0.312 -> 0.266 (step 250) -> 0.256 (step 500), edge 0.376 -> 0.241, every background in 0.26-0.31. A head reading pi-H3's RGB tokens is matting from colour, and it tops out near 0.25 (fair reference for colour-only matting: P1 without its offset, ~0.21).

Stage B lets pi-H3 itself carry alpha, without touching its RGB:
- **RGB pass:** pi-H3 2-step's ordinary last step, LoRA off, fused DWD decode. Bit-identical to pi-H3 by construction.
- **Alpha pass:** block 42's input from that forward reruns blocks 42-49 with a rank-32 LoRA on (checkpointed), then `dwd_norm_out` -> the frozen Sol refiner (`lot/pih3_dwd.py`, eager) -> the alpha head (warm-started from a5).
- Loss: EA + band (alpha only). Trained at sigma 0.45-0.75, 4 clips per step.

**Gate (set before the run):**
- sigma-0.6 soft_mae < 0.10 and edge_mae < 0.08 (unchanged).
- RGB identical with and without the alpha pass (asserted: the B = 0 alpha pass reproduces the RGB pass's block output; the RGB pass never sees the LoRA).
- Added seconds per 768x512x39 clip on the RTX 5090, measured by `--check` (estimate before measuring: about +0.2 s on 1.55 s).
- Deployment check by eye: about 8 real pi-H3 2-step generations from noise (camera-lock schedule) on smoke / glass / hair captions, RGBA composites over a checker. The trained sigma range never includes pi-H3's real last step (0.865), so this is the only test of what a user would get.
- **Early stop:** if sigma-0.6 soft_mae at step 250 is not below 0.22 (clearly under Stage A's plateau), stop: even DiT capacity cannot matte from this input.

Consequences: one alpha LoRA per generator (DiR would need its own); ComfyUI needs a custom node that reruns the last 8 blocks with the LoRA at the last step only.
