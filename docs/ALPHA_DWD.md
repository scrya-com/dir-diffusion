# Alpha on pi-H3's DWD head (Stage A)

pi-H3 and DiR + DWD never run the H3 VAE decoder, so the P1 alpha branch on the VAE decoder (`alpha/h3_alpha.py`, `main`) is never called on the fast path. Stage A hangs the same kind of one-way branch off pi-H3 4-step's DWD refiner (`alpha/pih3_alpha.py`): it reads the refiner's final 16-pixel token states, and RGB is untouched.

## Inputs, from a probe

pi-H3 4-step on ten captions of the RGBA set put the subject on flat black (5), white (2), studio grey (1), a real scene (1), or filled the frame (2). No LayerDiffuse padding. So training clips composite the moving subject over dark / light / flat / gradient / scene / padded backgrounds (`alpha/data_pih3.py`), 768x512, 39 frames, noised to pi-H3 4-step's last-step sigma (default schedule and `max_raw_t` 1.0, the camera fix). The DiT and DWD head are frozen.

## Gate (set before the run)

On 60 held-out clips (the soft-heavy eval split, backgrounds cycled, both last-step sigmas):

- `soft_mae` < 0.10, `edge_mae` < 0.08;
- no background kind with `soft_mae` above 0.15;
- `rgb_psnr` (pi-H3's DWD RGB against the clip it was shown) reported, as the check that the teacher-forced input lines up with the alpha target;
- the head adds under 0.1 s to a 768x512x39 clip.

Miss: Stage B, a LoRA on pi-H3's last step only (bypass, never merged), EA loss plus an RGB-preservation term.
