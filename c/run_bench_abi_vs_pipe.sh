#!/usr/bin/env bash
# run_bench_abi_vs_pipe.sh — transport A/B: pipe (SUBMIT/DATA/DONE) vs ABI (in-process).
#
# METHOD: the two arms are run in ALTERNATING BLOCKS, not one after the other.
# Sequential A/B on a noisy machine charges any thermal/background drift entirely
# to whichever arm runs second — which is how a previous revision produced an
# impossible "1.46 s/turn of IPC overhead". Alternating cancels the drift, and
# the per-arm result is the MEDIAN OF BLOCK MEDIANS, not one long run.
#
# Both arms do identical work per turn: full prefill of a UNIQUE prompt (no KV
# prefix can match) + 1 greedy decode. Verified by the prefill audit below.
#
#   bash /Users/khalid/Desktop/projects/colibri/c/run_bench_abi_vs_pipe.sh
#
# Env: REPS_A/BLOCK_A (tiny fixture), REPS_B/BLOCK_B (real OLMoE), OLMOE_MODEL.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

REPS_A="${REPS_A:-10}";  BLOCK_A="${BLOCK_A:-200}"
REPS_B="${REPS_B:-2}";   BLOCK_B="${BLOCK_B:-3}"
OLMOE_MODEL="${OLMOE_MODEL:-$HOME/Models/olmoe_i8}"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

median_of()  { awk '{for(i=1;i<=NF;i++) if($i=="median"){print $(i+1); exit}}'; }
# median of a column of numbers on stdin (one per line, no blanks; signed)
median_num() { grep -E '^-?[0-9.]+$' | sort -n | awk '{a[NR]=$1} END{ if(NR==0){print "nan"; exit}
                                print (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2) }'; }
# min..max of a column, for showing the spread
spread_num() { grep -E '^-?[0-9.]+$' | sort -n | awk 'NR==1{min=$1} {max=$1} END{printf "%.0f..%.0f", min, max}'; }
collect() { for r in $(seq 1 "$1"); do median_of <"$2$r"; done; }

# Run one block silently; stdout (the report line) goes to $1.
run_block() { local out="$1"; shift; "$@" >"$out" 2>/dev/null; }

echo "== build =="
if [ ! -f glm_tiny/model.safetensors ]; then
    python3 tools/make_glm_tiny_fixture.py --output glm_tiny >/dev/null
    python3 tools/make_glm53_tokenizer.py --output glm_tiny/tokenizer.json >/dev/null
fi
make -s colibri olmoe bench_glm bench_olmoe 2>/dev/null
echo "  ok"

# ---------------------------------------------------------------- audit
echo
echo "== prefill audit (must show the full prompt prefilled every turn) =="
BENCH_CHILD_LOG="$TMP/child.log" ./bench_glm pipe ./colibri ./glm_tiny 3 1 64 16 16 >/dev/null 2>&1 || true
grep "API. KV slot" "$TMP/child.log" 2>/dev/null | head -3 | sed 's/^/    /' || true
echo "    (abi arm: full prefill by construction — kv_rollback(0) + prefill)"

# ---------------------------------------------------------------- arm A
echo
echo "== A. tiny GLM fixture: $REPS_A alternating blocks x $BLOCK_A iters, 1 token/req =="
for r in $(seq 1 "$REPS_A"); do
    run_block "$TMP/pt$r" ./bench_glm pipe ./colibri ./glm_tiny "$BLOCK_A" 1 64 16 16
    run_block "$TMP/at$r" env COLI_ENGINE_ID=glm ./bench_glm abi ./glm_tiny "$BLOCK_A" 1
    printf "  rep %d:  pipe %9s us   abi %9s us\n" "$r" \
           "$(median_of <"$TMP/pt$r")" "$(median_of <"$TMP/at$r")"
done
# paired analysis: for each rep, pipe_median - abi_median.
# This is a PAIRED design (same machine state, alternating), so the paired
# difference is the right statistic; the ratio of medians is not.
paired_report() {
    local reps="$1" pfx_p="$2" pfx_a="$3"
    local wins=0 n=0
    : >"$TMP/diffs"
    for r in $(seq 1 "$reps"); do
        local p a
        p=$(median_of <"$pfx_p$r"); a=$(median_of <"$pfx_a$r")
        [ -z "$p" ] || [ -z "$a" ] && continue
        awk -v p="$p" -v a="$a" 'BEGIN{printf "%.0f\n", p-a}' >>"$TMP/diffs"
        awk -v p="$p" -v a="$a" 'BEGIN{exit !(p>a)}' && wins=$((wins+1))
        n=$((n+1))
    done
    local med
    med=$(median_num <"$TMP/diffs")
    printf "  paired: n=%d reps, pipe slower in %d/%d, median paired delta %+.0f us\n" \
           "$n" "$wins" "$n" "$med"
    if [ "$n" -ge 5 ]; then
        # two-sided sign test, exact, small n
        awk -v w="$wins" -v n="$n" 'BEGIN{
            k = (w > n-w) ? w : n-w;
            # P(X>=k) for X~Bin(n,0.5), doubled
            c=1; for(i=0;i<k;i++) c = c*(n-i)/(i+1);   # C(n,k)
            s=0; for(i=k;i<=n;i++){ s+=c; c = c*(n-i)/(i+1); }
            s = s / (2^n); if (s>1) s=1;
            printf "  sign test: p ~ %.2f  -> %s\n", 2*s,
                   (2*s < 0.05 ? "SIGNIFICANT" : "not significant (noise dominates)");
        }'
    fi
}

PA=$(collect "$REPS_A" "$TMP/pt")
AA=$(collect "$REPS_A" "$TMP/at")
echo
echo "  per-rep spread (min..max of the block medians — this is the noise floor):"
printf "    pipe %s us   abi %s us\n" "$(printf '%s\n' "$PA" | spread_num)" \
                                      "$(printf '%s\n' "$AA" | spread_num)"
echo
paired_report "$REPS_A" "$TMP/pt" "$TMP/at"

# ---------------------------------------------------------------- arm B
if [ -f "$OLMOE_MODEL/tokenizer.json" ]; then
    echo
    echo "== B. real OLMoE: $REPS_B alternating blocks x $BLOCK_B iters, 1 token/req =="
    echo "   ~8 s/turn on this container (full prefill streams experts) — this is slow"
    for r in $(seq 1 "$REPS_B"); do
        run_block "$TMP/pr$r" ./bench_olmoe pipe ./olmoe "$OLMOE_MODEL" "$BLOCK_B" 1 16 8
        run_block "$TMP/ar$r" ./bench_olmoe abi "$OLMOE_MODEL" "$BLOCK_B" 1
        printf "  rep %d:  pipe %12s us   abi %12s us\n" "$r" \
               "$(median_of <"$TMP/pr$r")" "$(median_of <"$TMP/ar$r")"
    done
    PB=$(collect "$REPS_B" "$TMP/pr")
    AB=$(collect "$REPS_B" "$TMP/ar")
    echo
    echo "  per-rep spread (min..max of the block medians):"
    printf "    pipe %s us   abi %s us\n" "$(printf '%s\n' "$PB" | spread_num)" \
                                          "$(printf '%s\n' "$AB" | spread_num)"
    echo
    paired_report "$REPS_B" "$TMP/pr" "$TMP/ar"
else
    echo
    echo "(skipping arm B: no OLMoE container at $OLMOE_MODEL)"
fi

echo
echo "done."
