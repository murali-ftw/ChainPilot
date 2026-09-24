#!/bin/zsh
# Phase 11C Stage B -- re-band the tightest at-risk capacity and fill cells at five model seeds.
#
# SCOPE, and why it is smaller than the brief's ten cells. A capacity h4-vs-h0 cell needs BOTH
# arms at five seeds (4 runs); a h4-vs-B5 cell needs only h4 (B5 already has 5 deterministic
# fits); a fill cell needs only the head (its LightGBM baselines likewise have 5 fits). The ten
# tightest cells therefore cost ~26 runs, roughly 7 h, against a remaining budget near 3.5 h.
#
# So the h4 arm is widened to five seeds and h0/B5 stay at three/five. That asymmetry is REPORTED,
# not hidden: where a comparison is decided against a 3-seed h0 band it is labelled as such.
# Arrival is excluded -- Stage A showed its count measures head-vs-constant, so re-banding would
# sharpen a mislabelled comparison.
set -u
cd "$(dirname "$0")/../.."
LOG=${LOG:-/tmp/p11c_B}; mkdir -p "$LOG"
PY=venv/bin/python; CFG=ml/configs/shipped.json

cell() {  # task world origin arch depth lr seed
  local task=$1 world=$2 origin=$3 arch=$4 depth=$5 lr=$6 seed=$7
  local f="$LOG/${task}_${world}_o${origin}_${arch}h${depth}_s${seed}.log"
  [[ -f "$f.done" ]] && { echo "[skip] $task $world o$origin s$seed"; return; }
  echo "=== START $task $world o$origin ${arch}h${depth} s$seed  $(date +%H:%M:%S)"
  $PY -u ml/train/loop.py train --config $CFG --task "$task" --world "$world" --origin $origin \
      --seed $seed --arch $arch --depth $depth --lr $lr > "$f" 2>&1 && touch "$f.done"
  echo "=== END   $task $world o$origin ${arch}h${depth} s$seed  $(date +%H:%M:%S)  $(tail -2 "$f" | head -1)"
}

for s in 37 47; do
  # fill: the two tightest cells overall (m/s 0.01 and 0.28); head only
  cell fill_rate       v6 8 none 0 1.25e-4 $s
  cell fill_rate       v6 7 none 0 1.25e-4 $s
  # capacity: the tightest h4 cells (m/s 0.15, 0.27, 0.41, 0.57); h4 only
  cell capacity_strain v6 7 mp 4 2.5e-4 $s
  cell capacity_strain v7 3 mp 4 2.5e-4 $s
  cell capacity_strain v6 3 mp 4 2.5e-4 $s
  cell capacity_strain v6 4 mp 4 2.5e-4 $s
done
echo "=== STAGEB COMPLETE $(date +%H:%M:%S)"
