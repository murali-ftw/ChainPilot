#!/bin/zsh
# Phase 11A Stage 1.3 -- the shuffled-graph control arm.
# The real-graph and h0 arms already exist from Phase 1 (reports/part2/phase-0-1-v8.md S3.0),
# so only the shuffled arm is trained here. Same architecture, depth, lr and parameter count;
# only the neighbourhood is permuted. shipped.json is NOT edited.
set -u
cd "$(dirname "$0")/../.."
LOG=${LOG:-/tmp/p11a_s1}; mkdir -p "$LOG"
PY=venv/bin/python; CFG=ml/configs/shipped.json
run() {
  local task=$1 arch=$2 depth=$3 lr=$4
  for s in 7 17 27; do
    local f="$LOG/${task}_shuf_s${s}.log"
    [[ -f "$f.done" ]] && { echo "[skip] $task shuf s$s"; continue; }
    echo "=== START $task shuffled seed $s  $(date +%H:%M:%S)"
    # the shuffle seed is tied to the model seed: permuted once per seed, fixed for that run
    $PY -u ml/train/loop.py train --config $CFG --task "$task" --world v8 --seed $s \
        --arch $arch --depth $depth --lr $lr --graph-shuffle $s > "$f" 2>&1 && touch "$f.done"
    echo "=== END   $task shuffled seed $s  $(date +%H:%M:%S)  $(tail -2 "$f" | head -1)"
  done
}
run capacity_strain mp   4 2.5e-4      # cheapest, clearest metric -- first
run arrival_week    lite 4 2.5e-4      # if the budget holds
echo "=== STAGE1 COMPLETE $(date +%H:%M:%S)"
