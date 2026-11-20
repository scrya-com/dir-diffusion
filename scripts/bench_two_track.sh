#!/usr/bin/env bash
# DiR + MPEG two-track benchmark: one command, one table.
#   bash scripts/bench_two_track.sh                 # all rows, published checkpoint, RTX 5090-class card
#   RUN=lot/runs/my_run TAG=_100 bash scripts/bench_two_track.sh
# Needs: $LOT_H3_CHECKPOINT, $LOT_H3_VAE, and prompts ($LOT_H3_STILLS_CACHE, or TE="a_te.safetensors b_te...").
# Writes runs/bench_two_track/<canvas>_sw<switch>/{meta.json,grid.jpg,*.mp4} and runs/bench_two_track/BENCH.md.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-johndpope/dir-diffusion-h3}          # Hugging Face repo or a local train_dir_h3 run dir
TAG=${TAG:-}
FRAMES=${FRAMES:-39}
PROMPTS=${PROMPTS:-3}                            # prompt 0 is warm-up and is left out of the table
SWAP=${SWAP:-0}                                  # 0 on a 32 GB card; try 8-16 on 24 GB
OUT=${OUT:-runs/bench_two_track}
CANVASES=${CANVASES:-"1344x768 768x512"}
SWITCHES=${SWITCHES:-"2 3"}
TE_ARGS=(); [ -n "${TE:-}" ] && TE_ARGS=(--te $TE)
for canvas in $CANVASES; do
  W=${canvas%x*}; H=${canvas#*x}
  for sw in $SWITCHES; do
    variants=two_track; [ "$sw" = "$(echo $SWITCHES | cut -d' ' -f1)" ] && variants=dir,lr,two_track   # baselines once per canvas
    python3 lot/dir_two_track_h3.py --run "$RUN" ${TAG:+--tag $TAG} --out "$OUT/${canvas}_sw$sw" \
      --variants $variants --switch $sw --size $W $H --frames $FRAMES --prompts $PROMPTS --swap $SWAP --outline "${TE_ARGS[@]}"
  done
done
# Fast paths (no training): latent-space upscale at the switch, and decode only the region's box at full res
FAST=${FAST:-1}
if [ "$FAST" = 1 ]; then
  for canvas in $CANVASES; do
    W=${canvas%x*}; H=${canvas#*x}
    python3 lot/dir_two_track_h3.py --run "$RUN" ${TAG:+--tag $TAG} --out "$OUT/${canvas}_sw2_fast" \
      --variants two_track --switch 2 --switch-upscale latent --decode crop --size $W $H --frames $FRAMES \
      --prompts $PROMPTS --swap $SWAP --outline "${TE_ARGS[@]}"
  done
fi
python3 lot/bench_table_two_track.py "$OUT"
