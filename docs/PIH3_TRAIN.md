# Training pi-H3 + DiR + MPEG: a sharp 2-step fast path

**Status (2026-11-22): in progress, no results yet.** The recipe and the commands are below; the gate result will be
added here when the first run finishes. Branch `exp/pih3-cascade`.

## The goal

pi-H3 2-step at full resolution is sharp: 1.55 s per 39-frame clip at 768×512 on an RTX 5090, or 1.31 s with H3-Tiny's
8 blocks dropped. The MPEG-style fast paths, which start at half resolution, are about twice as fast but soft. They
keep 75–84% of the fine detail, and the scene sometimes changes ([`PIH3.md`](PIH3.md), the handoff and the cascade).

The reason is that pi-H3 was never trained for a half-resolution start. MPEG-Diffusion fine-tunes its model for
mixed resolution; our runs so far did not. This recipe adds that training, in the DiR style: a small LoRA learns to
make the fast path land on the 4-step model's own video.

## The recipe

```
teacher   pi-H3 4-step, full resolution                       noise z
student   pi-H3 2-step + bypass LoRA (rank 32):
            step 1 at half resolution                         noise 2·avgpool2x2(z)   (the same noise, shrunk)
            learned latent ×2 (latent_up_h3, frozen)
            re-noise with z, step 2 at full resolution
loss      MSE(final latent, teacher's)  +  0.2 · LPIPS(DWD pixels of one native block, teacher's)
```

- **Paired noise.** The student's half-resolution noise is the teacher's full-resolution noise, average-pooled 2×2
  and multiplied by 2 to keep unit variance. The full-resolution re-noise uses `z` itself. The target is then a fixed
  function of what the student sees, which is what lets it fix the scene drift as well as the softness.
- **The LoRA is bypass, not merged.** It sits on top of pi-H3's merged int8 ConvRot adapter, so it loads the same way
  in our ComfyUI pack.
- **The DWD head is differentiable in training** (`TrainHead`: dequantized refiner weights and an out-of-place copy
  of ComfyUI-piFlow's refine loop). LPIPS is taken on one native block: 5 latent frames, 17 pixel frames.
- **Optional:** `--drop h3tiny` trains the same path with H3-Tiny's 8 blocks skipped.

## Prerequisites

The pi-H3 setup in [`PIH3.md`](PIH3.md#setup): the int8 ConvRot checkpoint (`LOT_H3_CHECKPOINT`), the adapters
(`LOT_PIH3_ADAPTERS`), the ComfyUI ≥ 0.45 tree with ComfyUI-piFlow (`LOT_PIH3_COMFY`), and the encoded prompt caches
(`LOT_H3_STILLS_CACHE`, `LOT_H3_CLIPS_CACHE`, see [`lot/lot_paths.py`](../lot/lot_paths.py)). A 32 GB card with no
block swap. `pip install lpips setproctitle`. Each script refuses to start if the card is already busy.

## Commands

```bash
cd lot
D=/media/2TB/lora-data/h3_lot/pih3/distill

# 0. The learned latent x2 (once; about 10 minutes). Interpolated latents are what ghost.
python3 latent_up_h3.py --out $D/latent_up --width 160 --blocks 8 --steps 24000

# 1. Teacher clips: pi-H3 4-step, full resolution. 500 training clips plus 8 held-out prompts (about 30 minutes)
python3 pih3_distill.py teacher --out $D/data --size 768x512 --frames 39 --train 500 --holdout 8

# 2. A quick check (4 steps, prints s_per_step and peak_gb), then the run
python3 pih3_distill.py train --data $D/data --out $D/run_a --up $D/latent_up/latent_up.pt --check
python3 pih3_distill.py train --data $D/data --out $D/run_a --up $D/latent_up/latent_up.pt --steps 3000
python3 pih3_distill.py train --data $D/data --out $D/run_b --up $D/latent_up/latent_up.pt --steps 3000 --drop h3tiny

# 3. Held-out gate: time, LPIPS against the teacher, detail as a fraction of the teacher's, contact sheets
python3 pih3_distill.py eval --data $D/data --out $D/eval_a --up $D/latent_up/latent_up.pt \
  --lora $D/run_a/lora_1000.pt $D/run_a/lora_2000.pt $D/run_a/lora_3000.pt
```

Checkpoints go to `run_*/lora_<step>.pt` every 250 steps. The log is `run_*/log.jsonl`. Resume with
`--resume run_a/lora_<step>.pt --start-step <step>`. `eval` always includes two baselines with no LoRA: `cascade`
(the untrained fast path) and `full2drop` (2-step full resolution, H3-Tiny's blocks dropped).

## The gate

On the held-out prompts, at 768×512, 39 frames:

| | Needed |
|---|---|
| Time end to end | ≤ 1.0 s |
| Detail (`hf`) against the 4-step teacher | ≥ 0.95 |
| LPIPS against the teacher | clearly below the untrained `cascade` |
| Scene | the same as the teacher's, by eye on the contact sheets |

If the student does not beat `full2drop` (1.31 s, sharp), that stays the recommended fast path. A pass is then
checked at 1344×768.

## Results

Not yet. They will go here, with the eval table and the contact sheets.

## Credits

pi-H3: Hansheng Chen, Shengqu Cai, Jiaming Song, Gordon Wetzstein, Xinchen Yan, Leonidas Guibas
([blog](https://hanshengchen.com/blog/pi-h3/)). DiR: Shanchuan Lin, Yansong Peng, Fu-Yun Wang, Haoqi Fan
([arXiv 2610.12102](https://arxiv.org/abs/2610.12102)). MPEG-Diffusion: Dennis Menn, Tianpei Gu, Guande He, Aryan V S,
Yanke Song, Nan Liu, Desai Xie, Xun Huang, Chenfeng Xu, Diana Marculescu
([arXiv 2611.12069](https://arxiv.org/abs/2611.12069)). The cascade, the upsampler and this distillation are ours and
not affiliated with any of the authors.
