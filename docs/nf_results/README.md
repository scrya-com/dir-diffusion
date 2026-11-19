# Neighbor Forcing: text-prefix fix and held-out character test

DiR-Turbo (`dir_turbo_clip_iter`, tag `_100`, EMA) streamed at 960x544, NoiseBank seed 0. Likeness = insightface
buffalo_l face-embedding cosine against the character's own original clip (originals self-score 0.85).

## Held-out characters (16 characters never trained on, 18 scorable clips; Kenny McCormick has no detectable face)

4 chunks (30 latent frames, ConvKV memory) per clip. `heldout_scores.csv` has every clip.

| Variant | Mean likeness | Ace |
|---|---|---|
| Dense teacher (32 frames, no mask) | 0.789 | 0.80 |
| Stream, untrained, text cut off (old) | 0.589 | 0.52 |
| Stream, untrained, `text_reads_first` | 0.751 | 0.67 |
| Stage 2 (`nf_kc_s2`) step 50 | 0.738 | 0.71 |
| Stage 2 step 100 | 0.763 | 0.73 |
| Stage 2 step 200 | 0.767 | 0.73 |
| Stage 2 step 300 | 0.742 | 0.69 |

Paired: fix vs cut +0.16 (18/18 clips); step 200 vs untrained fix +0.015 (14/18); step 300 -0.009 (6/18).
Stage 2 drifts to tighter close-ups with harsher contrast; held-out motion 0.34 -> 0.27 over 300 steps.

- `compare_heldout.jpg`: original | dense | old stream | fixed stream | Stage 2 step 200.
- `heldout_howard_wolowitz_fixed_stream.mp4`: an unseen character, streamed with the fix, no training.
- `compare_textfix.jpg`, `ace_textfix_scores.txt`: the first test, on the Ace prompt with 3 seeds.

## Reproduce (from `lot/`)

1. `nf_kc_cache.py OUT`: encode the known-character prompts (`prompts.jsonl`, good + onthefence) into a cache of
   text encodings + zero latents (Stage 2 with `--ref none` uses the latent for its shape only); holds out Ace + 15
   random good characters.
2. `LOT_H3_TRAIN=1 train_nf_h3.py --stage 2 --dir-init <dir_turbo_clip_iter> --dir-tag _100 --dir-ema
   --text-reads-first --cache OUT/train --source '' --max-latent-hw 32 32 --holdout 2 --swap 8 --steps 300
   --save-every 50 --out runs/nf_kc_s2`
3. `nf_heldout_h3.py RENDER_OUT runs/nf_kc_s2 dense zeroshot_cut zeroshot_fix s2_100 s2_200 ...`
4. `nf_heldout_score.py RENDER_OUT` (face scoring; needs `onnxruntime-gpu` to run on the GPU, see below).

The run and renders used here (`nf_kc_s2`, `nf_kc_heldout`, `ace_textfix`) are not in git; on the training machine
they are in `/media/2TB/lora-data/h3_lot/runs/nf_worktree/`.

Face scoring on the GPU: conda's `onnxruntime` is CPU-only and saturates the machine. Use a venv with
`--system-site-packages`, `pip install --ignore-installed --no-deps onnxruntime-gpu`, and `LD_LIBRARY_PATH` set to
torch's `site-packages/nvidia/*/lib` plus an unversioned `libcudnn.so` symlink.
