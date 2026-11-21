#!/usr/bin/env bash
# pi-H3 (Hansheng Chen, Shengqu Cai, Jiaming Song, Gordon Wetzstein, Xinchen Yan, Leonidas Guibas) against DiR + our
# DWD v0.2, one table: $OUT/BENCH.md. 39 frames at 768x512 and 1344x768, plus 124 frames (5 s) at 1344x768.
#   bash scripts/bench_pih3.sh
# Needs the pi-H3 int8 ConvRot DWD adapters ($LOT_PIH3_ADAPTERS), a ComfyUI >= 0.45 tree with ComfyUI-piFlow
# ($LOT_PIH3_COMFY, for the DWD head), $LOT_H3_CHECKPOINT, $LOT_H3_VAE; see docs/PIH3.md.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-../runs/bench_pih3}
SIZES=${SIZES:-"768x512 1344x768"}
LONG=${LONG:-1344x768}                            # the 5 s row (124 frames, H3's 17n+5 grid nearest 121); empty: skip
PROMPTS=${PROMPTS:-3}
export OUT PROMPTS
for steps in 2 4; do                              # pi-H3 rows; two-track opens the region halfway
  GENERATOR=pih3 STEPS=$steps SWITCH=$((steps / 2)) SIZES="$SIZES" bash bench_accel.sh speed
done
SIZES="$SIZES" VARIANTS=${DIR_VARIANTS:-dir_vae,dir_dwd,lr_dwd,tt_dwd2} bash bench_accel.sh speed
if [ -n "$LONG" ]; then
  for steps in 2 4; do
    GENERATOR=pih3 STEPS=$steps FRAMES=124 SCORE=0 PIH3_VARIANTS=pih3_dwd SIZES="$LONG" bash bench_accel.sh speed
  done
  FRAMES=124 SCORE=0 VARIANTS=dir_dwd SIZES="$LONG" bash bench_accel.sh speed
fi
${PY:-python3} ../lot/accel_h3.py table "$OUT/speed"
