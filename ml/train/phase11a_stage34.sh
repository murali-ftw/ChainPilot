#!/bin/zsh
# Phase 11A Stage 3 (five MODEL seeds) and Stage 4 (deviation 59 reproduction).
#
# Stage 3: seeds 7/17/27 already exist from Phase 1, so only 37 and 47 are trained here.
#   These are MODEL seeds. v8 also ships five DATASET seeds (1001-1005); that is a different
#   axis and is NOT varied here -- dataset seed 1001 is fixed throughout.
# Stage 4: one arrival cell on v6 and one on v7 at the shipped config, seed 7, now that the
#   unconditional line-feature attachment is fixed. Written to a SEPARATE bundle root so the
#   recorded Phase 8 bundles are not touched or overwritten.
set -u
cd "$(dirname "$0")/../.."
LOG=${LOG:-/tmp/p11a_s34}; mkdir -p "$LOG"
PY=venv/bin/python; CFG=ml/configs/shipped.json

cell() {  # tag world task seed extra...
  local tag=$1 world=$2 task=$3 seed=$4; shift 4
  local f="$LOG/${tag}_${task}_${world}_s${seed}.log"
  [[ -f "$f.done" ]] && { echo "[skip] $tag $task $world s$seed"; return; }
  echo "=== START $tag $task $world seed $seed  $(date +%H:%M:%S)"
  $PY -u ml/train/loop.py train --config $CFG --task "$task" --world "$world" --seed $seed "$@" \
      > "$f" 2>&1 && touch "$f.done"
  echo "=== END   $tag $task $world seed $seed  $(date +%H:%M:%S)  $(tail -2 "$f" | head -1)"
}

# ---- Stage 4 first: cheapest, and it gates open item 1 -------------------
cell s4 v6 arrival_week 7 --arch lite --depth 4 --lr 2.5e-4 --bundle-root /tmp/p11a_dev59_bundles
cell s4 v7 arrival_week 7 --arch lite --depth 4 --lr 2.5e-4 --bundle-root /tmp/p11a_dev59_bundles

# ---- Stage 3: seeds 37 and 47 at the shipped configuration ---------------
for s in 37 47; do
  cell s3 v8 capacity_strain $s --arch mp   --depth 4 --lr 2.5e-4
  cell s3 v8 arrival_week    $s --arch lite --depth 4 --lr 2.5e-4
  cell s3 v8 fill_rate       $s --arch none --depth 0 --lr 1.25e-4
  cell s3 v8 shortage_qty    $s --arch mp   --depth 1 --lr 1.25e-4
done
echo "=== STAGE34 COMPLETE $(date +%H:%M:%S)"
