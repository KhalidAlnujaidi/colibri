#!/usr/bin/env bash
# engine_demo.sh — build & smoke-test the Unified C Library (engine.h) reference
# backend + shared harness. Self-contained; run from anywhere:
#   bash /Users/khalid/Desktop/projects/colibri/c/engine_demo.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CC="${CC:-clang}"
CFLAGS="-O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function"

echo "== 1/4 Compile the existing colibri binary (must still build) =="
# Just a syntax/compile check of the unmodified default path, without OpenMP
# wiring (the Makefile handles that); this only proves colibri.c still compiles.
"$CC" $CFLAGS -c colibri.c -o /tmp/colibri_check.o
echo "  colibri.c compiles (default path) OK"

echo "== 2/4 Compile the engine ABI library (COLI_ENGINE_ADAPTER, no main) =="
"$CC" $CFLAGS -DCOLI_ENGINE_ADAPTER -DCOLIBRI_NO_MAIN -c colibri.c -o libcolibri_engine.o
echo "  libcolibri_engine.o OK"

echo "== 3/4 Compile the registry + harness =="
"$CC" $CFLAGS -c engine_registry.c -o engine_registry.o
"$CC" $CFLAGS agent_main.c engine_registry.o libcolibri_engine.o -lm -o agent_main
echo "  agent_main OK"

echo "== 4/4 Smoke test (no model: expect graceful failure to find shards) =="
echo "  usage check:"
./agent_main 2>&1 | head -1 || true
echo
echo "  To run against a real model directory:"
echo "    ./agent_main glm <model_dir> tool_demo"
echo
echo "build + harness smoke test complete."
