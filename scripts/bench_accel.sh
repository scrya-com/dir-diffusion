#!/usr/bin/env bash
# DiR + DWD end to end: one command per benchmark, one markdown table each.
#   bash scripts/bench_accel.sh speed       # DiT + decode, split, per size and variant -> $OUT/speed/BENCH.md
#   bash scripts/bench_accel.sh quality     # LPIPS / PSNR / flicker / seam / tde vs the VAE -> $OUT/quality/QUALITY.md
#   bash scripts/bench_accel.sh             # both
# Any size works: SIZES="640x384 1280x704" bash scripts/bench_accel.sh speed
# Needs $LOT_H3_CHECKPOINT and $LOT_H3_VAE (lot/lot_paths.py), prompts ($ACCEL_CACHES, colon-separated H3 caches,
# else the default stills and clips caches), and a DWD stack: LORA, HEAD, REFINER (not published yet).
# A 32 GB card with every DiT block resident (no block swap: the DWD tail and the stacked LoRA need that).
set -euo pipefail
cd "$(dirname "$0")/../lot"
PY=${PY:-python3}
DWD=${DWD:-/media/2TB/lora-data/h3_lot/dwd}
RUN=${RUN:-johndpope/dir-diffusion-h3}            # Hugging Face repo or a local train_dir_h3 run dir
LORA=${LORA:-$DWD/wB/dwd_ft_lora.pt}
HEAD=${HEAD:-$DWD/wB_head/dwd_head_ridge.pt}
REFINER=${REFINER:-$DWD/wG/refiner_best.pt}
MODE=${MODE:-two_pass}                            # single: a refiner trained on the one-pass latent
FRAMES=${FRAMES:-39}
OUT=${OUT:-../runs/bench_accel}
SIZES=${SIZES:-"768x512 1344x768"}                # speed: canvases (multiples of 64 for the lr_ and tt_ rows)
QSIZES=${QSIZES:-"768x512 512x768 512x512 1024x576 1344x768 672x384 384x256"}   # quality: any multiples of 32
PROMPTS=${PROMPTS:-3}                             # speed: prompt 0 is warm-up and is left out of the table
HELD=${HELD:-8}                                   # quality: held-out clips per size
VARIANTS=${VARIANTS:-dir_vae,dir_dwd,dir_dwd_early,lr_vae,lr_dwd,tt_vae,tt_dwd}
SCORE=${SCORE:-1}                                 # speed: also PSNR / LPIPS of each DWD row against the VAE (not timed)
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
STACK=(--run "$RUN" --lora "$LORA" --head "$HEAD" --refiner "$REFINER" --mode "$MODE" --frames "$FRAMES")
EARLY=()                                          # an early-exit stack, when one has been trained
if [ -n "${EARLY_LORA:-}" ]; then
  EARLY=(--early-lora "$EARLY_LORA" --early-head "$EARLY_HEAD" --early-refiner "$EARLY_REFINER"
         --early-exit "${EARLY_EXIT:-2}" --early-mode "${EARLY_MODE:-two_pass}")
fi
what=${1:-all}
if [ "$what" = speed ] || [ "$what" = all ]; then
  $PY -u accel_h3.py bench "${STACK[@]}" "${EARLY[@]}" --sizes $SIZES --prompts "$PROMPTS" --variants "$VARIANTS" \
    --out "$OUT/speed" --save $([ "$SCORE" = 1 ] && echo --score)
fi
if [ "$what" = quality ] || [ "$what" = all ]; then
  $PY -u accel_h3.py quality "${STACK[@]}" --sizes $QSIZES --held "$HELD" --out "$OUT/quality"
  if [ -n "${EARLY_LORA:-}" ]; then
    $PY -u accel_h3.py quality --run "$RUN" --frames "$FRAMES" --lora "$EARLY_LORA" --head "$EARLY_HEAD" \
      --refiner "$EARLY_REFINER" --exit "${EARLY_EXIT:-2}" --mode "${EARLY_MODE:-two_pass}" --vae-exit \
      --sizes $QSIZES --held "$HELD" --out "$OUT/quality_early"
  fi
fi
