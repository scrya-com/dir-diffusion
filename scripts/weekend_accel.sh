#!/usr/bin/env bash
# Weekend run: DiR + DWD on H3 as fast as possible end to end, at any resolution (RTX 5090 only).
#
#   launch   cd /media/2TB/lora-data/h3_lot/dwd && nohup setsid ./weekend_accel.sh >> weekend_accel.log 2>&1 &
#   smoke    SMOKE=1 ./weekend_accel.sh          # every phase with tiny budgets, in accel_smoke/w (about 40 min)
#   watch    grep kind=accel_weekend weekend_accel.log | tail -20        # one line per phase event
#            tail -f accel/logs/<phase>.log                              # the phase's own LOT_H3 lines
#   stop     kill $(cat accel/state/pid); then the python child by pid:  ps -eo pid,args | awk '$2 ~ /^(accel|dwd):/'
#   resume   launch again: finished phases are skipped (accel/state/*.done), trainers continue from *_last.pt,
#            collect and cache skip what exists, and the deadline stays the first launch's.
#
# Phases (minutes are budgets, cut to what the deadline leaves; a gate that fails skips only its own branch):
#   preflight   disk, weights, CPU tests, GPU parity of the DWD tail
#   baseline    published stack: quality per size, the "skip iterations and VAE-decode" floors, speed table
#   collect     multi-resolution rollouts, inputs of the last iteration (r3) and of earlier exits (r2, r1), VAE targets
#   e2          early exit after iteration 3: head refit -> last-8-blocks LoRA -> refit -> cache -> refiner -> quality gate
#   mr          multi-resolution refiner for the full rollout: cache -> refiner from the published one -> quality gate
#   e1          early exit after iteration 2, only when e2 passed, its floor allows it and 14 h are left
#   final       speed and quality tables with every stack that passed -> accel/FINAL.md
set -u
D=${D:-/media/2TB/lora-data/h3_lot/dwd}
REPO=${REPO:-/home/johndpope/Documents/GitHub/dir-diffusion-accel}
LOT=$REPO/lot
PY=${PY:-/home/johndpope/miniconda3/bin/python3}
RUN=${RUN:-/home/johndpope/Documents/GitHub/dir-diffusion/lot/runs/dir_turbo_clip}
GPU_UUID=${GPU_UUID:-GPU-022a724e-9bc8-f221-8451-ea58c21a8235}        # the RTX 5090; the RTX PRO 4000 is never touched
export CUDA_VISIBLE_DEVICES=$GPU_UUID                                 # so cuda:0 is the 5090 and there is no cuda:1
export LOT_H3_TRAIN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCEL_CACHES=${ACCEL_CACHES:-/media/2TB/lora-data/h3_lot/cache_iso3d:/media/2TB/lora-data/h3_lot/cache_atlas_clips:/media/2TB/lora-data/h3_lot/cache_known_chars/train}
BASE_LORA=${BASE_LORA:-$D/wB/dwd_ft_lora.pt}
BASE_HEAD=${BASE_HEAD:-$D/wB_head/dwd_head_ridge.pt}
BASE_REFINER=${BASE_REFINER:-$D/wG/refiner_best.pt}
BASE_PAIRS=${BASE_PAIRS:-$D/wC/cache/pairs.json}                      # the published refiner's own training pairs
SMOKE=${SMOKE:-0}

if [ "$SMOKE" = 1 ]; then
  W=${W:-$D/accel_smoke/w}; HOURS=${HOURS:-2}; TRIES=2; MIN_FREE_GB=20; MARGIN=6
  TRAIN_SIZES="768x512:5 512x512:4 672x384:4 384x256:4 1024x576:3"
  SMALL="768x512 512x512 672x384 384x256"; HELD=2; CACHE_HELD=2; PER_FOLDER=8
  QSIZES="768x512 1024x576"; UNSEEN="640x384"; FLOOR_SIZES="768x512"; BENCH_SIZES="768x512"; PROMPTS=2
  B_COLLECT=4; B_REFIT=20; REFIT_MAX=10; STAT=6; B_FT=3; FT_GATE_MIN=0; B_CACHE=10; B_REF_E=3; B_REF_MR=3; EVAL_MIN=1
  B_FINAL=15; E1_NEEDS=${E1_NEEDS:-9999}; MIN_CLIPS=10; ACCEPT_LPIPS=9; FT_GATE_MAX=9; FLOOR_MAX=9; MR_MARGIN=-9
else
  W=${W:-$D/accel}; HOURS=${HOURS:-58}; TRIES=4; MIN_FREE_GB=120; MARGIN=45
  # training clips per size (8.4 h at the measured rates); every size also gets $HELD held-out clips
  TRAIN_SIZES="768x512:500 512x768:500 512x512:300 672x384:400 384x672:200 384x256:300 832x480:300 640x640:150 1024x576:350 576x1024:200 1280x704:200 1344x768:220 768x1344:100"
  # sizes of at most 4608 tokens a 12-latent-frame clip: the LoRA step and the head refit peak at 29.7 GB there
  SMALL="768x512 512x768 512x512 672x384 384x672 384x256"; HELD=8; CACHE_HELD=3; PER_FOLDER=300
  QSIZES="768x512 512x768 512x512 1024x576 576x1024 1344x768 672x384 384x256"
  UNSEEN="960x544 544x960 896x512 448x448"                            # never trained on: does it hold between sizes?
  FLOOR_SIZES="768x512 1024x576 672x384"; BENCH_SIZES="768x512 1344x768"; PROMPTS=3
  B_COLLECT=510; B_REFIT=100; REFIT_MAX=1200; STAT=256; B_FT=300; FT_GATE_MIN=90; B_CACHE=180; B_REF_E=600; B_REF_MR=420
  EVAL_MIN=30; B_FINAL=80; E1_NEEDS=840; MIN_CLIPS=1200
  ACCEPT_LPIPS=${ACCEPT_LPIPS:-0.16}      # an early exit ships when head + refiner reach this against the full rollout's VAE
  FT_GATE_MAX=${FT_GATE_MAX:-0.42}        # head-level LPIPS the early-exit LoRA must be under after 90 min (r3's LoRA: 0.27)
  FLOOR_MAX=${FLOOR_MAX:-0.30}            # the VAE on the exit's own estimate must be this close to the full rollout
  MR_MARGIN=${MR_MARGIN:-0.002}           # the multi-res refiner must beat the published one by this (mean LPIPS)
fi
DATA=$W/data
STACK="--run $RUN --lora $BASE_LORA --head $BASE_HEAD --refiner $BASE_REFINER"
mkdir -p "$W/state" "$W/logs"
cd "$LOT" || exit 1

say() { echo "$(date '+%F %T') LOT_H3 kind=accel_weekend $*" | tee -a "$W/events.log"; }
exec 9> "$W/state/lock"
flock -n 9 || { say event=already_running dir=$W; exit 1; }
echo $$ > "$W/state/pid"
[ -f "$W/state/deadline" ] || echo $(( $(date +%s) + HOURS * 3600 )) > "$W/state/deadline"
left() { echo $(( ( $(cat "$W/state/deadline") - $(date +%s) ) / 60 )); }      # minutes to the deadline
fit() { local want=$1 room=$(( $(left) - B_FINAL - ${2:-0} )); [ "$room" -lt "$want" ] && want=$room; echo "$want"; }
num() { $PY -c "import json,sys; d=json.load(open(sys.argv[1]));
for k in sys.argv[2].split('.'): d=d[k]
print(f'{d:.4f}')" "$1" "$2" 2>/dev/null; }
le() { $PY -c "import sys; a,b=map(float,sys.argv[1:3]); sys.exit(0 if a<=b else 1)" "$1" "$2" 2>/dev/null; }
folders() { local e=$1 out="" s; shift; for s in "$@"; do [ -d "$DATA/r$e/${s%%:*}" ] && out="$out $DATA/r$e/${s%%:*}"; done; echo $out; }
wait_gpu() {                                                          # another job on the 5090: wait up to 15 min
  local n=0 used
  while [ $n -lt 90 ]; do
    used=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v u="$GPU_UUID" '$1==u{print $2}')
    [ "${used:-99999}" -lt 3000 ] && return 0
    n=$((n + 1)); sleep 10
  done
  say event=gpu_busy used_mib=$used
}

# step NAME MINUTES CMD...: skip when done; retry a crash (the phases resume); never retry a gate (3), a cache that
# ran out of time (4, its pairs are usable) or a timeout (124). Output goes to logs/NAME.log.
step() {
  local name=$1 minutes=$2 try=0 code=1; shift 2
  [ -f "$W/state/$name.done" ] && { say phase=$name event=skip; return 0; }
  if [ "$minutes" -lt 1 ]; then say phase=$name event=no_time left_min=$(left); return 5; fi
  while [ $try -lt $TRIES ]; do
    try=$((try + 1)); wait_gpu
    say phase=$name event=start try=$try budget_min=$minutes left_min=$(left)
    timeout --signal=TERM --kill-after=120 $(( (minutes + MARGIN) * 60 )) "$@" >> "$W/logs/$name.log" 2>&1
    code=$?
    say phase=$name event=exit code=$code try=$try $(grep -a "LOT_H3" "$W/logs/$name.log" | tail -1 | cut -d' ' -f2-8)
    case $code in
      0) touch "$W/state/$name.done"; return 0 ;;
      3|4|124) return $code ;;
    esac
    sleep $([ "$SMOKE" = 1 ] && echo 5 || echo 60)
  done
  say phase=$name event=gave_up tries=$try code=$code
  return $code
}

# ----------------------------------------------------------------------------------------------- phases

preflight() {
  local free=$(( $(df --output=avail -k "$D" | tail -1) / 1048576 ))
  say phase=preflight event=disk free_gb=$free need_gb=$MIN_FREE_GB
  [ "$free" -ge "$MIN_FREE_GB" ] || { say phase=preflight event=failed why=disk; exit 1; }
  local f; for f in "$BASE_LORA" "$BASE_HEAD" "$BASE_REFINER" "$BASE_PAIRS" "$RUN/config.json"; do
    [ -e "$f" ] || { say phase=preflight event=failed why=missing file=$f; exit 1; }
  done
  step tests 10 sh -c "$PY test_accel_h3.py && $PY test_dwd_h3.py && $PY test_refine_dwd_h3.py" \
    || { say phase=preflight event=failed why=tests; exit 1; }
  step check 10 $PY -u accel_h3.py check $STACK || { say phase=preflight event=failed why=check; exit 1; }
}

baseline() {
  step base_quality 40 $PY -u accel_h3.py quality $STACK --sizes $QSIZES --held $HELD --out "$W/baseline/quality"
  say phase=baseline event=quality lpips=$(num "$W/baseline/quality/quality.json" overall.lpips) \
      worst=$(num "$W/baseline/quality/quality.json" worst_lpips)
  local e
  for e in 2 1; do      # what skipping iterations costs with the real VAE: the floor of an early exit
    step base_floor_e$e 20 $PY -u accel_h3.py quality $STACK --exit $e --vae-exit --sizes $FLOOR_SIZES --held 4 --videos 0 \
      --out "$W/baseline/floor_e$e"
    $PY - "$W/baseline/floor_e$e/quality.json" > "$W/state/floor_e$e" 2>/dev/null <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))["results"]
print(f"{sum(v['VAE of the exit estimate']['lpips'] for v in r.values()) / len(r):.4f}")
EOF
    say phase=baseline event=floor exit=$e vae_on_exit_estimate_lpips=$(cat "$W/state/floor_e$e")
  done
  step base_speed 30 env PY=$PY RUN=$RUN LORA=$BASE_LORA HEAD=$BASE_HEAD REFINER=$BASE_REFINER SIZES="$BENCH_SIZES" \
    PROMPTS=$PROMPTS OUT="$W/baseline" VARIANTS=dir_vae,dir_dwd,lr_vae,lr_dwd,tt_vae,tt_dwd bash ../scripts/bench_accel.sh speed
}

collect() {
  step collect $(fit $B_COLLECT) $PY -u accel_h3.py collect --run $RUN --sizes $TRAIN_SIZES --exits 3 2 1 --held $HELD \
    --minutes $(fit $B_COLLECT) --min-free-gb $((MIN_FREE_GB / 2)) --out "$DATA"
  CLIPS=$(ls "$DATA"/r3/*/train_*.pt 2>/dev/null | wc -l)
  say phase=collect event=count train_clips=$CLIPS need=$MIN_CLIPS disk_gb=$(du -s --block-size=1G "$DATA" 2>/dev/null | cut -f1)
  [ "$CLIPS" -ge "$MIN_CLIPS" ]
}

# refit NAME EXIT LORA OUT: ridge head on the exit iteration's features with LORA on (small sizes only)
refit() {
  local name=$1 e=$2 lora=$3 out=$4
  [ -f "$W/state/$name.done" ] && return 0
  local args="--run $RUN --data $(folders $e $SMALL) --lora $lora --out $out --halo 4 --lambdas 1e-6 1e-4 \
    --stat-samples $STAT --max-samples $REFIT_MAX --swap 0"
  if [ ! -f "$out/refit_moments.pt" ]; then
    step ${name}_moments $(fit $B_REFIT) $PY -u train_dwd_h3.py refit $args || return 1
  fi
  step ${name}_solve 40 $PY -u train_dwd_h3.py refit $args --solve-only || return 1
  rm -f "$out/refit_moments.pt"                                       # 5 GB; the head is what is kept
  touch "$W/state/$name.done"
}

# early E INIT_REFINER: the stack that decodes after iteration E + 1 (0-based exit E)
early() {
  local e=$1 init=$2 X=$W/e$1 floor
  floor=$(cat "$W/state/floor_e$e" 2>/dev/null)
  if ! le "${floor:-9}" $FLOOR_MAX; then say phase=e$e event=gate_failed gate=floor lpips=${floor:-none} max=$FLOOR_MAX; return 1; fi
  say phase=e$e event=begin floor_lpips=$floor left_min=$(left)
  refit e${e}_refit0 $e "$BASE_LORA" "$X/head0" || { say phase=e$e event=failed at=refit0; return 1; }
  step e${e}_ft $(fit $B_FT) $PY -u train_dwd_h3.py ft --run $RUN --data $(folders $e $SMALL) --out "$X/lora" \
    --init "$X/head0/dwd_head_ridge.pt" --resume "$BASE_LORA" --k 8 --rank 64 --head-lr 0 --lr 5e-5 \
    --lpips-weight 0.05 --lpips-frames 1 --eval-lpips --select lpips --minutes $(fit $B_FT) --eval-minutes $EVAL_MIN \
    --checkpoint-minutes 20 --auto-resume --best-videos 1 --gate-minutes $FT_GATE_MIN --gate-key lpips --gate-max $FT_GATE_MAX
  case $? in
    0) ;;
    3) say phase=e$e event=gate_failed gate=ft_lpips max=$FT_GATE_MAX; return 1 ;;
    *) [ -f "$X/lora/dwd_ft_lora.pt" ] || { say phase=e$e event=failed at=ft; return 1; } ;;
  esac
  local lora=$X/lora/dwd_ft_lora.pt head=$X/head0/dwd_head_ridge.pt
  if [ -f "$lora" ]; then                                             # a better LoRA than the start: its own head
    refit e${e}_refit1 $e "$lora" "$X/head1" && head=$X/head1/dwd_head_ridge.pt
  else
    lora=$BASE_LORA; say phase=e$e event=note lora=never_beat_its_start using=$BASE_LORA
  fi
  say phase=e$e event=head_level $(grep -a "kind=dwd_refit_done" "$W/logs/e${e}_refit1_solve.log" 2>/dev/null | tail -1 | cut -d' ' -f3-4)
  step e${e}_cache $(fit $B_CACHE) $PY -u accel_h3.py cache --run $RUN --data "$DATA"/r$e/* --lora "$lora" --head "$head" \
    --exit $e --out "$X/ref" --train-per-folder $PER_FOLDER --held-per-folder $CACHE_HELD --minutes $(fit $B_CACHE)
  [ -f "$X/ref/cache/pairs.json" ] || { say phase=e$e event=failed at=cache; return 1; }
  step e${e}_refiner $(fit $B_REF_E) $PY -u refine_dwd_h3.py train --out "$X/ref" --init "$init" --latent-context \
    --latent-grid 256 --ema 0.999 --crop 384 --minutes $(fit $B_REF_E) --eval-minutes $EVAL_MIN --device cuda:0 --workers 6
  [ -f "$X/ref/refiner_best.pt" ] || { say phase=e$e event=failed at=refiner; return 1; }
  step e${e}_quality 40 $PY -u accel_h3.py quality --run $RUN --lora "$lora" --head "$head" --refiner "$X/ref/refiner_best.pt" \
    --exit $e --vae-exit --data "$DATA"/r$e/* --held $HELD --out "$X/quality" || return 1
  local got=$(num "$X/quality/quality.json" overall.lpips)
  printf 'LORA=%s\nHEAD=%s\nREFINER=%s\nEXIT=%s\nLPIPS=%s\n' "$lora" "$head" "$X/ref/refiner_best.pt" "$e" "$got" > "$X/stack.env"
  if le "${got:-9}" $ACCEPT_LPIPS; then
    cp "$X/stack.env" "$W/state/e$e.accepted"; say phase=e$e event=accepted lpips=$got max=$ACCEPT_LPIPS
  else
    say phase=e$e event=gate_failed gate=final_lpips lpips=${got:-none} max=$ACCEPT_LPIPS; return 1
  fi
}

mr() {                                                                # the full rollout's refiner on every size
  local X=$W/mr
  step mr_cache $(fit $B_CACHE) $PY -u accel_h3.py cache $STACK --data "$DATA"/r3/* --out "$X/new" \
    --train-per-folder $PER_FOLDER --held-per-folder $CACHE_HELD --minutes $(fit $B_CACHE)
  [ -f "$X/new/cache/pairs.json" ] || { say phase=mr event=failed at=cache; return 1; }
  step mr_merge 5 $PY -u accel_h3.py merge-pairs --pairs "$X/new/cache/pairs.json" "$BASE_PAIRS" --out "$X/ref" || return 1
  step mr_refiner $(fit $B_REF_MR) $PY -u refine_dwd_h3.py train --out "$X/ref" --init "$BASE_REFINER" --latent-context \
    --latent-grid 256 --ema 0.999 --crop 384 --minutes $(fit $B_REF_MR) --eval-minutes $EVAL_MIN --device cuda:0 --workers 6
  [ -f "$X/ref/refiner_best.pt" ] || { say phase=mr event=failed at=refiner; return 1; }
  step mr_quality_base 40 $PY -u accel_h3.py quality $STACK --data "$DATA"/r3/* --held $HELD --videos 0 --out "$X/quality_base"
  step mr_quality 40 $PY -u accel_h3.py quality --run $RUN --lora "$BASE_LORA" --head "$BASE_HEAD" \
    --refiner "$X/ref/refiner_best.pt" --data "$DATA"/r3/* --held $HELD --out "$X/quality" || return 1
  local got=$(num "$X/quality/quality.json" overall.lpips) was=$(num "$X/quality_base/quality.json" overall.lpips)
  if $PY -c "import sys; a,b,m=map(float,sys.argv[1:4]); sys.exit(0 if a<=b-m else 1)" "${got:-9}" "${was:-0}" $MR_MARGIN; then
    echo "$X/ref/refiner_best.pt" > "$W/state/mr.accepted"; say phase=mr event=accepted lpips=$got published=$was
  else
    say phase=mr event=gate_failed lpips=${got:-none} published=${was:-none} margin=$MR_MARGIN keeping=published
  fi
}

final() {
  local refiner=$BASE_REFINER early="" e
  [ -f "$W/state/mr.accepted" ] && refiner=$(cat "$W/state/mr.accepted")
  for e in 2 1; do [ -f "$W/state/e$e.accepted" ] && early=$W/state/e$e.accepted; done    # the earliest exit that passed
  local env="PY=$PY RUN=$RUN LORA=$BASE_LORA HEAD=$BASE_HEAD REFINER=$refiner PROMPTS=$PROMPTS OUT=$W/final"
  if [ -n "$early" ]; then
    . "$early"; env="$env EARLY_LORA=$LORA EARLY_HEAD=$HEAD EARLY_REFINER=$REFINER EARLY_EXIT=$EXIT"
  fi
  say phase=final event=stacks refiner=$refiner early=${early:-none}
  step final_speed $B_FINAL env $env SIZES="$BENCH_SIZES" bash ../scripts/bench_accel.sh speed
  step final_quality $B_FINAL env $env QSIZES="$QSIZES $UNSEEN" HELD=$HELD bash ../scripts/bench_accel.sh quality
  {
    echo "# DiR + DWD acceleration run, $(date '+%F %T')"
    echo; echo "Stacks: refiner \`$refiner\`; early exit \`${early:-none}\`."
    echo; echo "## Events"; echo; echo '```'; grep -a "event=\(quality\|floor\|count\|head_level\|accepted\|gate_failed\|failed\|gave_up\|no_time\|skipped\)" "$W/events.log"; echo '```'
    local t
    for t in baseline/speed/BENCH.md final/speed/BENCH.md baseline/quality/QUALITY.md final/quality/QUALITY.md \
             final/quality_early/QUALITY.md e2/quality/QUALITY.md e1/quality/QUALITY.md mr/quality/QUALITY.md; do
      [ -f "$W/$t" ] && { echo; echo "## $t"; echo; cat "$W/$t"; }
    done
  } > "$W/FINAL.md"
  say phase=final event=report file=$W/FINAL.md
}

# ----------------------------------------------------------------------------------------------- run

say event=launch dir=$W smoke=$SMOKE hours=$HOURS left_min=$(left) repo=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)
preflight
baseline
if collect; then
  early 2 "$BASE_REFINER"
  mr
  if [ -f "$W/state/e2.accepted" ] && [ "$(left)" -ge "$E1_NEEDS" ]; then
    early 1 "$W/e2/ref/refiner_best.pt"
  else
    say phase=e1 event=skipped e2_accepted=$([ -f "$W/state/e2.accepted" ] && echo 1 || echo 0) left_min=$(left) needs_min=$E1_NEEDS
  fi
else
  say phase=collect event=gate_failed why=too_few_clips
fi
final
say event=finished left_min=$(left)
