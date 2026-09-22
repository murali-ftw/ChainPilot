#!/bin/bash
# Run the frozen validator over every v8 seed, then publish the seed-variance band.
# Levels run in order inside validator_v8.py and stop at the first failed ladder gate.
# Two at a time: level 4 holds the channel-week store and the label frame in memory.
set -u
cd "$(dirname "$0")/../.."
PY=./venv/bin/python
V=db/gen_v8/validator_v8.py
mkdir -p docs/v8/runs
n=0
for s in 1001 1002 1003 1004 1005; do
  d=db/gen_v8/seed_$s
  [ -d "$d" ] || continue
  ( $PY $V "$d" --out docs/v8/runs/validator_seed_$s.json > docs/v8/runs/validator_seed_$s.txt 2>&1
    echo "  seed $s exit $? -> docs/v8/runs/validator_seed_$s.txt" ) &
  n=$((n+1)); if [ $((n % 2)) -eq 0 ]; then wait; fi
done
wait
$PY $V --variance db/gen_v8/seed_100{1,2,3,4,5} --out docs/v8/seed_variance.json \
  > docs/v8/seed_variance.txt 2>&1
echo "variance -> docs/v8/seed_variance.txt"
