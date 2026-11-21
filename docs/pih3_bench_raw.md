| Canvas | Frames | Variant | Total s | DiT s | Decode s | x real time | PSNR dB | LPIPS | Peak GB |
|---|---|---|---|---|---|---|---|---|---|
| 1344x768 | 39 | DiR, VAE decode | 16.56 | 12.37 | 4.19 | 0.1 |  |  | 29.4 |
| 1344x768 | 39 | DiR + DWD | 13.73 | 13.05 | 0.68 | 0.12 | 30.54 | 0.1138 | 28.4 |
| 1344x768 | 39 | DiR half res + DWD, x2 | 2.62 | 2.44 | 0.17 | 0.62 | 31.98 | 0.1608 | 25.0 |
| 1344x768 | 39 | two-track, DWD on both tracks | 4.47 | 3.99 | 0.48 | 0.36 | 30.47 | 0.1725 | 26.4 |
| 1344x768 | 124 | DiR + DWD (out of memory) |  |  |  |  |  |  |  |
| 1344x768 | 124 | pi-H3 2-step + its DWD | 30.95 | 30.32 | 0.62 | 0.17 |  |  | 27.0 |
| 1344x768 | 124 | pi-H3 4-step + its DWD | 61.42 | 60.79 | 0.62 | 0.08 |  |  | 27.0 |
| 1344x768 | 39 | pi-H3 2-step + its DWD | 5.45 | 5.25 | 0.2 | 0.3 | 19.81 | 0.4017 | 23.4 |
| 1344x768 | 39 | pi-H3 2-step, VAE decode | 9.38 | 5.23 | 4.14 | 0.17 |  |  | 28.2 |
| 1344x768 | 39 | pi-H3 2-step half res + its DWD, x2 | 0.96 | 0.91 | 0.05 | 1.69 | 18.96 | 0.4052 | 22.2 |
| 1344x768 | 39 | pi-H3 2-step two-track, its DWD on both tracks (52% region) | 2.63 | 2.47 | 0.16 | 0.62 | 18.55 | 0.4299 | 23.3 |
| 1344x768 | 39 | pi-H3 4-step + its DWD | 10.76 | 10.56 | 0.2 | 0.15 | 32.92 | 0.0868 | 23.4 |
| 1344x768 | 39 | pi-H3 4-step, VAE decode | 14.71 | 10.54 | 4.17 | 0.11 |  |  | 28.2 |
| 1344x768 | 39 | pi-H3 4-step half res + its DWD, x2 | 1.91 | 1.86 | 0.05 | 0.85 | 34.53 | 0.111 | 22.2 |
| 1344x768 | 39 | pi-H3 4-step two-track, its DWD on both tracks (29% region) | 3.86 | 3.75 | 0.11 | 0.42 | 29.63 | 0.3114 | 22.8 |
| 768x512 | 39 | DiR, VAE decode | 5.42 | 3.64 | 1.77 | 0.3 |  |  | 27.2 |
| 768x512 | 39 | DiR + DWD | 4.12 | 3.86 | 0.26 | 0.39 | 32.16 | 0.1119 | 24.9 |
| 768x512 | 39 | DiR half res + DWD, x2 | 0.98 | 0.92 | 0.06 | 1.66 | 28.37 | 0.1996 | 23.6 |
| 768x512 | 39 | two-track, DWD on both tracks | 1.67 | 1.45 | 0.21 | 0.97 | 28.73 | 0.2055 | 24.7 |
| 768x512 | 39 | pi-H3 2-step + its DWD | 1.55 | 1.47 | 0.07 | 1.05 | 19.53 | 0.3957 | 21.5 |
| 768x512 | 39 | pi-H3 2-step, VAE decode | 3.22 | 1.46 | 1.75 | 0.5 |  |  | 26.3 |
| 768x512 | 39 | pi-H3 2-step half res + its DWD, x2 | 0.36 | 0.33 | 0.03 | 4.51 | 19.09 | 0.4192 | 21.0 |
| 768x512 | 39 | pi-H3 2-step two-track, its DWD on both tracks (46% region) | 0.8 | 0.72 | 0.07 | 2.03 | 18.98 | 0.4542 | 21.4 |
| 768x512 | 39 | pi-H3 4-step + its DWD | 3.06 | 2.98 | 0.08 | 0.53 | 36.41 | 0.0673 | 21.5 |
| 768x512 | 39 | pi-H3 4-step, VAE decode | 4.74 | 2.98 | 1.76 | 0.34 |  |  | 26.3 |
| 768x512 | 39 | pi-H3 4-step half res + its DWD, x2 | 0.73 | 0.69 | 0.03 | 2.23 | 32.09 | 0.1401 | 21.0 |
| 768x512 | 39 | pi-H3 4-step two-track, its DWD on both tracks (40% region) | 1.46 | 1.39 | 0.07 | 1.11 | 29.33 | 0.2702 | 21.3 |

Medians over prompts after the first (warm-up). Total = DiT + decode, pixels left on the GPU. x real time = clip seconds at 24 fps / total. PSNR and LPIPS: the DWD pixels against the VAE's for the same latent (the full rollout's latent for an early exit; the same x2 for half res; the VAE composite of the same two latents for two-track), not timed. GPU: NVIDIA GeForce RTX 5090.
