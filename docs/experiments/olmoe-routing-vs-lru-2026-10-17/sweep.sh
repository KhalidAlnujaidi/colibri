#!/bin/sh
# Held-out residency sweeps over recorded traces. Usage: sweep.sh <traces-dir> <out-dir>
#  within-category: train on replicate 1 of every category, evaluate on 2-4.
#  cross-<cat>:     train on every prompt of the other three categories,
#                   evaluate on all four of <cat>.
# Budgets are whole slots per layer: cap * 16 layers * 6,307,840 B (+1 kB so
# the uniform split does not round a slot down).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SIM="$HERE/../../../c/tools/residency_sim.py"
T=$1; OUT=$2
mkdir -p "$OUT"
CATS="chat code multilingual longctx"
BUDGETS=""
for cap in 4 8 12 16 24 32 48; do
  BUDGETS="$BUDGETS $(python3 -c "print(($cap*16*6307840+1000)/1e9)")"
done
COMMON="--engine olmoe --manifest $HERE/manifest-olmoe-residency.json --read-gbps 3.0 --felt-fraction 1.0 --sensitivity-layer 0 --sensitivity-layer 7 --sensitivity-layer 15"

train=""; evals=""
for c in $CATS; do
  train="$train $T/${c}_1.trace"
  evals="$evals $T/${c}_2.trace $T/${c}_3.trace $T/${c}_4.trace"
done
python3 "$SIM" sweep $COMMON --train $train --eval $evals \
  --expert-budgets $BUDGETS --json-out "$OUT/within-category.json" > "$OUT/within-category.txt" &

for held in $CATS; do
  train=""; evals=""
  for c in $CATS; do
    if [ $c = $held ]; then evals="$evals $T/${c}_1.trace $T/${c}_2.trace $T/${c}_3.trace $T/${c}_4.trace"
    else train="$train $T/${c}_1.trace $T/${c}_2.trace $T/${c}_3.trace $T/${c}_4.trace"; fi
  done
  python3 "$SIM" sweep $COMMON --train $train --eval $evals \
    --expert-budgets $BUDGETS --json-out "$OUT/cross-$held.json" > "$OUT/cross-$held.txt" &
done
wait
