#!/usr/bin/env bash
# run_bench.sh — temp helper to build & run the colibri orchestration benchmark.
# Safe to delete after. One command:
#   bash /Users/khalid/Desktop/projects/colibri/c/bench_orchestration/run_bench.sh
#
# Token counts are scaled up (640 / 6400 / 64000) to expose how the C/C++
# advantage grows with frames per request. To keep wall time bounded, iters
# scale DOWN as tokens/req grows (total emitted frames stays ~constant):
#   tokens/req   iters   total token-frames
#   1            5000    5,000
#   64           5000    320,000
#   640          500     320,000
#   6400         50      320,000
#   64000        5       320,000
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CC="${CC:-clang}"
CXX="${CXX:-clang++}"
PY="${PY:-python3}"
BASE_ITERS="${BASE_ITERS:-5000}"

echo "== 1/5 Build stub_engine =="
"$CC" -O2 -Wall -Wextra -o stub_engine stub_engine.c

echo "== 2/5 Build orch_c (C) =="
"$CC" -O2 -Wall -Wextra -o orch_c orch_c.c

echo "== 3/5 Build orch_cpp (C++) =="
"$CXX" -O2 -std=c++17 -Wall -Wextra -o orch_cpp orch_cpp.cpp

echo "== 4/5 Sanity check (1 request, 2 tokens) =="
./orch_c ./stub_engine 10 2 >/dev/null
./orch_cpp ./stub_engine 10 2 >/dev/null
"$PY" orch_py.py ./stub_engine 10 2 >/dev/null
echo "sanity OK"

# tokens/req -> iters (scaled so total emitted frames stays ~constant)
run_case () {
  local t="$1"; local iters="$2"
  echo; echo "############ tokens/req = $t (iters=$iters) ############"
  echo "--- C     ---"; ./orch_c    ./stub_engine "$iters" "$t"
  echo "--- C++   ---"; ./orch_cpp  ./stub_engine "$iters" "$t"
  echo "--- Python ---"; "$PY" orch_py.py ./stub_engine "$iters" "$t"
}

echo "== 5/5 Run benchmark =="
run_case 1     "$BASE_ITERS"
run_case 64    "$BASE_ITERS"
run_case 640   $(( BASE_ITERS / 10 ))
run_case 6400  $(( BASE_ITERS / 100 ))
run_case 64000 $(( BASE_ITERS / 1000 ))

echo
echo "benchmark complete — see the tables above (all against the identical stub_engine)"
