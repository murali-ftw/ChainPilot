#!/bin/zsh
# Phase 1 on v8 -- the shipped configuration per task, fixed split, 3 seeds, shipped depth + h0.
#
# shipped.json is NOT modified: the world is passed on the command line and every other
# hyper-parameter comes from the config as it stands. No learning rate is retuned.
#
# fill_rate: shipped.json names b5flat22 (LightGBM) and the serving path has no LightGBM
# loader (deviation 46). Per the brief, fill's NEURAL h0 is trained instead -- which is
# shipped.json's own `_superseded` entry, arch none / depth 0 / lr 1.25e-4 -- and the config
# is left unrepaired.
#
# Ordered so the cells that carry the headline numbers land first.
set -u
cd "$(dirname "$0")/../.."
LOG=${LOG:-/tmp/phase1_v8}
mkdir -p "$LOG"
PY=venv/bin/python
CFG=ml/configs/shipped.json

run() {  # task depth-args tag
  local task=$1; shift
  local tag=$1; shift
  for s in 7 17 27; do
    local f="$LOG/${task}_${tag}_s${s}.log"
    if [[ -f "$f.done" ]]; then echo "[skip] $task $tag s$s"; continue; fi
    echo "=== START $task $tag seed $s  $(date +%H:%M:%S)"
    $PY -u ml/train/loop.py train --config $CFG --task "$task" --world v8 --seed $s "$@" \
        > "$f" 2>&1 && touch "$f.done"
    echo "=== END   $task $tag seed $s  $(date +%H:%M:%S)  $(tail -2 "$f" | head -1)"
  done
}

# 1. arrival: shipped h4 SHARE-lite, then the h0 reference that is actually SERVED
run arrival_week    h4  --arch lite --depth 4 --lr 2.5e-4
run arrival_week    h0  --arch none --depth 0 --lr 2.5e-4
# 2. capacity: shipped h4 message-passing, then h0
run capacity_strain h4  --arch mp   --depth 4 --lr 2.5e-4
run capacity_strain h0  --arch none --depth 0 --lr 2.5e-4
# 3. fill: neural h0 (deviation 46 -- see header)
run fill_rate       h0  --arch none --depth 0 --lr 1.25e-4
# 4. shortage: shipped h1, diagnostic only
run shortage_qty    h1  --arch mp   --depth 1 --lr 1.25e-4
run shortage_qty    h0  --arch none --depth 0 --lr 1.25e-4
echo "=== QUEUE COMPLETE $(date +%H:%M:%S)"
