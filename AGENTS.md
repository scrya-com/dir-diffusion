# Agent guidelines for dir-diffusion

## Name every GPU process

GPU jobs here run for minutes to hours, often several at once on two cards. nvtop, `ps` and `htop` cut long command lines short, so a job launched from a long path is unreadable. Every script that runs on a GPU, including throwaway scripts in a scratch folder, sets a short process title as its first statement after the imports:

```python
from setproctitle import setproctitle   # in the miniconda env; pip install setproctitle elsewhere

setproctitle("dwd:sgd lr=3e-5 run_v1")   # <project>:<phase> <the arguments that tell runs apart>
```

- **Format:** `project:phase` first (`dir:train`, `dwd:collect`, `nf:stage2`), then the few arguments that tell this run apart from the others. Keep it under about 60 characters.
- **Scripts with phases or subcommands:** set the title after parsing the arguments, so it names the phase.
- **Launching:** `cd` into the script's folder and run it by its bare name (`python3 -u ft_lastk.py …`), not by a long absolute path.
- **One GPU job at a time per card:** follow `lot/gpu_guard.py` and add new GPU scripts to its `GPU_SCRIPTS`.

## The reference to beat: pi-H3

**pi-H3** is the top-tier few-step MiniMax H3: the official 2/4-step model with DWD, by **Hansheng Chen, Shengqu Cai, Jiaming Song, Gordon Wetzstein, Xinchen Yan and Leonidas Guibas** (Stanford University and Luma AI; five of the six also wrote the DWD paper; [blog](https://hanshengchen.com/blog/pi-h3/), [weights](https://huggingface.co/Lakonik/pi-H3)). It beats our stack on both fidelity and speed. New work must beat it or build on it.
- **DiR + our DWD** (`johndpope/dir-diffusion-h3`, `-dwd` v0.2) is our earlier port; compare against pi-H3, not against DiR alone.
- How to run pi-H3 in this pipeline, and the parity with the authors' ComfyUI path: [`docs/PIH3.md`](docs/PIH3.md). Select it with `--generator pih3` (`lot/generators.py`).

Measured here (ours; RTX 5090, dense SDPA, int8 ConvRot, no text encoder or audio decode in the times; 3 held-out prompts, medians of the warm ones; [`docs/PIH3.md`](docs/PIH3.md)):

| 39 frames | DiR + our DWD v0.2 | pi-H3 2-step + its DWD | pi-H3 4-step + its DWD |
|---|---|---|---|
| 768×512 | 4.12 s (DWD vs VAE LPIPS 0.112) | **1.55 s** | 3.06 s (0.067) |
| 1344×768 | 13.73 s (0.114) | **5.45 s** | 10.76 s (0.087) |
| 1344×768, 124 frames (5.2 s) | out of memory | 30.95 s | 61.42 s |

- **Parity.** Our port matches the authors' ComfyUI path at the DiT's own chaos floor. The ComfyUI path itself is 1.37 s / 2.64 s (2/4-step, 768×512), 13–16% faster than our musubi kernels.
- **Two-track on pi-H3:** 1.9–2.8× faster, but a different video (LPIPS 0.45–0.74 against full resolution, same seed), and at 2 steps the box ghosts.

The authors' numbers (theirs, pi-H3 0.9, Sol attention + FP8):
- B200, 1344×768, 5 s with audio, text encoding to decoded video and audio: 2-step **3.81 s**, 4-step **7.30 s** (Sol-H3 4-step 9.48 s).
- H100, 768p, 5 s, sums of per-stage medians: 2-step DWD 6.19 s against PDMD 2-step + H3 VAE 9.87 s; 4-step 12.02 s against Sol-H3 12.61 s.
- They report better visual and audio quality than the other few-step H3 models they compared (visual galleries).
