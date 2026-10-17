#!/bin/sh
# Record one ROUTE_TRACE per prompt: one cold process each, CPU only, no pins,
# no prefetch, greedy. Usage: record.sh <converted-model> <refs-dir> <out-dir>
# (refs from make_prompts.py). Traces do not depend on the cache size or the
# thread count: a rerun at OMP_NUM_THREADS=2 was byte-identical.
set -e
SNAP=$1; REFS=$2; OUT=$3
BIN=$(cd "$(dirname "$0")/../../../c" && pwd)/olmoe
mkdir -p "$OUT"
for ref in "$REFS"/*.ref.json; do
  name=$(basename "$ref" .ref.json)
  env -i PATH="$PATH" HOME="$HOME" OMP_NUM_THREADS=4 TEMP=0 PILOT=0 HOT=0 \
    ROUTE_TRACE="$OUT/$name.trace" SNAP="$SNAP" "$BIN" 64 8 "$ref" > "$OUT/$name.log" 2>&1
done
