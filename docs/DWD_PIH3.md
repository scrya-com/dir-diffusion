# pi-H3's DWD decoder on DiR

Our DWD v0.2 for DiR is a linear head plus an 8M-parameter refiner, read after rerunning DiR's last 8 blocks with a LoRA. pi-H3 (Hansheng Chen, Shengqu Cai, Jiaming Song, Gordon Wetzstein, Xinchen Yan and Leonidas Guibas; Apache-2.0) decodes with a modulated norm and 8 Sol-H3 ViT decoder blocks. This branch puts pi-H3 2-step's decoder on DiR. Not affiliated with the pi-H3 authors.

## Pieces

- `lot/pih3_dwd.py`: a trainable forward of ComfyUI-piFlow's `SolH3DWDHead.refine` (layer by layer, the ring slots mapped back to clip rows). `lot/test_pih3_dwd.py`: exact against `refine` under the eager path (rel 0.0), 51.6 dB against the fused int8 kernels (their own rounding), window coverage a permutation, backward through the frozen int8 blocks (SDPA in place of int8 attention, training only). Warm decode 0.07 s at 768x512x39.
- `lot/train_pih3_dwd.py`: Stage 1 freezes the 8 Sol blocks and trains `dwd_norm_out`, `feature_expand` and a new zero-initialised branch from DiR's final-iteration input state into the expanded tokens. `--sol-lora` (Stage 2) adds a LoRA to every Sol linear; shelved, so the frozen blocks stay shared with pi-H3 2-step and its alpha head.
- `lot/compare_dwd_s1.py`: the head-to-head below. Needs a 32 GB card (v0.2's two-pass tail needs every block resident).

## Results

12 held-out DiR clips (4 each from `collect_w1_{iso,atlas,kc}`), scored against the VAE decode of the same latent; LPIPS (vgg) over every frame. Stage 1: 2000 steps, 1.4 s/step on one RTX 5090, L1 + LPIPS on 320 px crops.

| Decoder | LPIPS | PSNR | Decode s |
|---|---|---|---|
| pi-H3 2-step decoder, untrained | 0.274 | 15.7 | |
| pi-H3 4-step decoder, untrained | 0.219 | 16.6 | |
| Our DWD v0.2 (LoRA + head + refiner) | 0.1168 | 28.33 | 0.222 |
| **Stage 1** | **0.1105** | 28.19 | **0.070** |

Stage 1 is better on 9 of 12 clips (all iso and kc), slightly worse on 3 atlas portraits. The v0.2 decode time leaves out its rerun of the last 8 DiT blocks with the LoRA, which Stage 1 does not need at all, so the real gap per clip is larger. The input-state branch is used (weight norm 0 -> 5.7), so a ComfyUI node for DiR needs it; ComfyUI-piFlow's DWD node alone does not load it.

Per clip: `docs/dwd_pih3_compare.json`.
