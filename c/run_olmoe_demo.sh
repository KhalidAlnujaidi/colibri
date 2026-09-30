#!/usr/bin/env bash
# run_olmoe_demo.sh — PATH 1: build the OLMoE engine ABI backend + harness and
# (optionally) run it against a real OLMoE container.
#
#   bash /Users/khalid/Desktop/projects/colibri/c/run_olmoe_demo.sh            # build only
#   bash /Users/khalid/Desktop/projects/colibri/c/run_olmoe_demo.sh ~/Models/olmoe_i8
#
# The model container (7 GB int8) is produced by:
#   cd c && KMP_DUPLICATE_LIB_OK=TRUE python3 tools/convert_olmoe_merged.py \
#       --repo allenai/OLMoE-1B-7B-0125-Instruct --out ~/Models/olmoe_i8
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# macOS/conda: see run_tiny_glm_demo.sh — conda libomp + torch libomp collide.
export KMP_DUPLICATE_LIB_OK=TRUE

echo "== build OLMoE engine ABI backend + harness =="
make -s agent_main_olmoe
echo "  agent_main_olmoe OK"

MODEL_DIR="${1:-}"
if [ -z "$MODEL_DIR" ]; then
    echo
    echo "no model dir given — build-only run."
    echo "to convert + download the container (7 GB):"
    echo "  cd $HERE && KMP_DUPLICATE_LIB_OK=TRUE python3 tools/convert_olmoe_merged.py \\"
    echo "      --repo allenai/OLMoE-1B-7B-0125-Instruct --out ~/Models/olmoe_i8"
    echo "then:"
    echo "  bash $0 ~/Models/olmoe_i8"
    exit 0
fi

if [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
    echo "ERROR: $MODEL_DIR has no tokenizer.json — is it a converted OLMoE container?" >&2
    exit 2
fi

echo
echo "== run the ABI harness against $MODEL_DIR =="
echo
echo "--- plain prefill+decode ---"
./agent_main_olmoe olmoe "$MODEL_DIR" plain
echo
echo "--- tool-call KV rollback demo ---"
./agent_main_olmoe olmoe "$MODEL_DIR" tool_demo
echo
echo "path 1 complete — OLMoE running through the unified engine ABI."
