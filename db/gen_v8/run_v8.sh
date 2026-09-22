#!/bin/bash
# V8 deliverable run. FIVE seeds, ONE code state (brief §5.3 / §5.6).
# The generator's sha1 is recorded in each seed's manifest.json; the validator's --variance
# mode refuses to call the band valid unless every seed carries the same one.
# Batches of two: the channel-week stores peak near 10 GB per process. `wait -n` is bash 4+
# and this ships on bash 3.2, so the batching is explicit.
set -u
cd "$(dirname "$0")/../.."
PY=./venv/bin/python
OUT=db/gen_v8
LOG=${LOG:-/tmp}
SEEDS=${SEEDS:-"1001 1002 1003 1004 1005"}
n=0
for s in $SEEDS; do
  $PY $OUT/generator_v8.py --seed $s --out $OUT > "$LOG/gen_v8_$s.log" 2>&1 &
  n=$((n+1))
  if [ $((n % 2)) -eq 0 ]; then wait; fi
done
wait
echo "generation done for: $SEEDS"
