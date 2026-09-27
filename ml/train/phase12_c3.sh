#!/bin/zsh
# Phase 12 C3 -- the part-relation ablation (Test 1.2 as respecified in reports/part2/phase-12.md A3.2).
# Two queues, one per task, run in parallel:  QUEUE=arrival | QUEUE=capacity
#   ablated arm: h4 with the part relation removed, depth 4, same lr, five model seeds
#   h0 widened 3 -> 5 seeds (s37, s47) so the gap-share denominator is at the five-seed floor too
# The log name carries arch, depth and the drop suffix (deviation 73: collisions suppress arms silently).
set -u
cd "$(dirname "$0")/../.."
LOG=${LOG:-ml/artifacts/phase12_c3_logs}; mkdir -p "$LOG"
PY=venv/bin/python; CFG=ml/configs/shipped.json

cell() {  # task arch depth lr seed [extra...]
  local task=$1 arch=$2 depth=$3 lr=$4 seed=$5; shift 5
  local extra="$*"; local sfx=""
  [[ "$extra" == *"--drop-relation part"* ]] && sfx="_droppart"
  local f="$LOG/${task}_v8_${arch}h${depth}${sfx}_s${seed}.log"
  [[ -f "$f.done" ]] && { echo "[skip] $f"; return; }
  echo "=== START $task ${arch}h${depth}${sfx} s$seed  $(date +%H:%M:%S)"
  $PY -u ml/train/loop.py train --config $CFG --task "$task" --world v8 --seed $seed \
      --arch $arch --depth $depth --lr $lr "$@" > "$f" 2>&1 && touch "$f.done"
  echo "=== END   $task ${arch}h${depth}${sfx} s$seed  $(date +%H:%M:%S)  $(tail -2 "$f" | head -1)"
}

if [[ "$QUEUE" == arrival ]]; then
  for s in 7 17 27 37 47; do cell arrival_week lite 4 2.5e-4 $s --drop-relation part; done
  for s in 37 47; do cell arrival_week none 0 2.5e-4 $s; done
else
  for s in 7 17 27 37 47; do cell capacity_strain mp 4 2.5e-4 $s --drop-relation part; done
  for s in 37 47; do cell capacity_strain none 0 2.5e-4 $s; done
fi
echo "=== QUEUE $QUEUE COMPLETE $(date +%H:%M:%S)"
