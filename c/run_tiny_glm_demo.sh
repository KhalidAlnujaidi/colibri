#!/usr/bin/env bash
# run_tiny_glm_demo.sh — PATH 3: run the unified engine ABI harness against a
# locally generated tiny GLM fixture. Zero model download, zero dependencies.
#
#   bash /Users/khalid/Desktop/projects/colibri/c/run_tiny_glm_demo.sh
#
# Steps:
#   0. informational dependency probe (never fatal)
#   1. generate the tiny GLM fixture (glm_tiny/: F32 safetensors + config.json)
#   2. add a real tokenizer.json (pure python)
#   3. rebuild the engine lib + harness, then run agent_main against it
#
# NOTE: the fixture has RANDOM weights — it is not a language model. The text it
# produces is gibberish by construction. What this validates is the ABI
# mechanics: engine_open -> tokenize -> prefill -> decode_step -> kv_get_len ->
# kv_rollback -> prefill(inject) -> decode.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CC="${CC:-clang}"
CFLAGS="-O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function"

# macOS/conda: conda's libomp and PyTorch's bundled libomp both load, and the
# OpenMP runtime aborts with "Error #15". Harmless to set; the fixture generator
# below needs no torch at all.
export KMP_DUPLICATE_LIB_OK=TRUE

echo "== 0/4 dependency probe (informational) =="
python3 - <<'PY' || true
import importlib
for m in ("torch", "transformers", "numpy"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m}: MISSING ({type(e).__name__})")
PY
echo "  (the fixture generator below uses only the python standard library)"

echo "== 1/4 generate tiny GLM fixture (glm_tiny/) =="
python3 tools/make_glm_tiny_fixture.py --output glm_tiny
ls -la glm_tiny/

echo "== 2/4 add tokenizer.json (pure python) =="
python3 tools/make_glm53_tokenizer.py --output glm_tiny/tokenizer.json
python3 - <<'PY'
import json
t = json.load(open("glm_tiny/tokenizer.json"))
n = len(t.get("model", {}).get("vocab", {}))
print(f"  tokenizer.json: {n} base tokens + {len(t.get('added_tokens', []))} added")
PY

echo "== 3/4 rebuild engine lib + harness =="
"$CC" $CFLAGS -DCOLI_ENGINE_ADAPTER -DCOLIBRI_NO_MAIN -c colibri.c -o libcolibri_engine.o
"$CC" $CFLAGS -c engine_registry.c -o engine_registry.o
"$CC" $CFLAGS agent_main.c engine_registry.o libcolibri_engine.o -lm -o agent_main

echo "== 4/4 run the ABI harness against the fixture =="
echo
echo "--- plain prefill+decode (gibberish: random weights) ---"
./agent_main glm ./glm_tiny plain
echo
echo "--- tool-call KV rollback demo ---"
./agent_main glm ./glm_tiny tool_demo
echo
echo "path 3 complete — the engine ABI ran end-to-end with no model download."
