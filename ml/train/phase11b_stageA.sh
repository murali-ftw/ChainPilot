#!/bin/zsh
# Phase 11B Stage A -- the DATASET-seed control.
#
# 11A varied MODEL seeds with dataset seed 1001 fixed and got verdict (i). v8's own G4 varied
# DATASET seeds 1001-1005 and got the opposite. A dataset-seed effect could not have appeared in
# 11A at all. This runs the three arms across the axis G4 varied, model seed fixed at 7.
#
# Dataset seed 1001 ("v8") already has all three arms at model seed 7 from 11A, so only the four
# other dataset seeds are trained here. Capacity first (cheapest, clearest metric), then arrival.
#
# A.3 separately widens the SHUFFLED arms on dataset seed 1001 to five MODEL seeds, since 11A
# quoted edge shares against a 3-seed shuffled band while the real arms were at five.
set -u
cd "$(dirname "$0")/../.."
LOG=${LOG:-/tmp/p11b_A}; mkdir -p "$LOG"
PY=venv/bin/python; CFG=ml/configs/shipped.json

cell() {  # tag world task arch depth lr seed [extra...]
  local tag=$1 world=$2 task=$3 arch=$4 depth=$5 lr=$6 seed=$7; shift 7
  local f="$LOG/${tag}_${task}_${world}_s${seed}.log"
  [[ -f "$f.done" ]] && { echo "[skip] $tag $task $world s$seed"; return; }
  echo "=== START $tag $task $world seed $seed  $(date +%H:%M:%S)"
  $PY -u ml/train/loop.py train --config $CFG --task "$task" --world "$world" --seed $seed \
      --arch $arch --depth $depth --lr $lr "$@" > "$f" 2>&1 && touch "$f.done"
  echo "=== END   $tag $task $world seed $seed  $(date +%H:%M:%S)  $(tail -2 "$f" | head -1)"
}

# ---- A.1 capacity: three arms x the four remaining dataset seeds, model seed 7 ----
for w in v8s1002 v8s1003 v8s1004 v8s1005; do
  cell A1 $w capacity_strain mp   4 2.5e-4 7
  cell A1 $w capacity_strain mp   4 2.5e-4 7 --graph-shuffle 7
  cell A1 $w capacity_strain none 0 2.5e-4 7
done
echo "=== A1 CAPACITY COMPLETE $(date +%H:%M:%S)"

# ---- A.3 widen dataset seed 1001's SHUFFLED capacity arm to five model seeds ----
for s in 37 47; do
  cell A3 v8 capacity_strain mp 4 2.5e-4 $s --graph-shuffle $s
done
echo "=== A3 CAPACITY COMPLETE $(date +%H:%M:%S)"

# ---- A.1 arrival: three arms x four dataset seeds (budget permitting) ----
for w in v8s1002 v8s1003 v8s1004 v8s1005; do
  cell A1 $w arrival_week lite 4 2.5e-4 7
  cell A1 $w arrival_week lite 4 2.5e-4 7 --graph-shuffle 7
  cell A1 $w arrival_week none 0 2.5e-4 7
done
echo "=== A1 ARRIVAL COMPLETE $(date +%H:%M:%S)"

# ---- A.3 arrival shuffled to five model seeds on dataset seed 1001 ----
for s in 37 47; do
  cell A3 v8 arrival_week lite 4 2.5e-4 $s --graph-shuffle $s
done
echo "=== STAGEA COMPLETE $(date +%H:%M:%S)"
