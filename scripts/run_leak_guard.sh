#!/usr/bin/env bash
# Phase 23AC T3 -- the standing leak guard (docs/standards/leak_guard.md).
# Exit 0: every FAIL is the known raw-panel leak; 1: a new FAIL or an unregistered feature (a FINDING); 2: guard invalid.
# Refuses to run on uncommitted ml/ changes (phase12_common.require_clean).
cd "$(dirname "$0")/.." || exit 3
echo "wiring hint: add scripts/run_leak_guard.sh to CI or a pre-push hook (this script edits neither)"
HADES_DEVICE=cpu ./venv/bin/python ml/eval/leak_guard.py --worlds v8,v8w1002 "$@"
code=$?
echo "leak guard exit code: $code"
exit $code
