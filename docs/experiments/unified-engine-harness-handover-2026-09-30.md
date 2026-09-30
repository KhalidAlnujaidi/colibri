# Unified Engine Harness (`engine.h`) — Handover

**Date:** 2026-09-30 · **Workstream:** Unified C Library / Shared Harness
**Status:** reference implementation complete and verified on two model families; measurement done and interpreted.

This doc is a map, not a duplicate. Source of truth for behaviour is the code; for
numbers, the commands below reproduce them.

---

## 1. Objective

Refactor Colibrì's engine so model compute and weights are decoupled from `main()`
and the pipe protocol loop, exposing a clean C ABI that an external harness links
against — so `qwen36.c`, `glm53.c`, `colibri.c`, `olmoe.c` implement **one** engine
interface instead of each duplicating its own agent/serve loop.

Approach chosen (agreed with the user): **reference backend first** — define the ABI,
implement it for one engine, prove it end-to-end, leave the others untouched.

## 2. What was built

| Path | Role |
|---|---|
| `c/engine.h` | The public ABI: `ColiEngineAdapter` vtable (`engine_open`, `engine_destroy`, `tokenize`, `detokenize`, `prefill`, `decode_step`, `sample`, `kv_get_len`, `kv_rollback`), capabilities, `ColiSampleConfig`, registry. Mirrors the repo's existing `ColiEdgeAdapter` / `ColiSegmentAdapter` conventions (`struct_size` + `abi_version`, explicit registration, engine-id lookup). |
| `c/engine_registry.c` | Engine-neutral registry + caller-side wrappers. Knows nothing about any model family. |
| `c/colibri.c` | **GLM backend**, inside `#ifdef COLI_ENGINE_ADAPTER` at EOF (+ `#include "engine.h"`). Wraps the full model behind the ABI; reuses `step()`, `tok.h`, `sample.h`. |
| `c/olmoe.c` | **OLMoE backend**, same pattern, `#ifdef COLI_ENGINE_ADAPTER`. Proves the ABI is family-agnostic. |
| `c/agent_main.c` | The shared harness: model-agnostic, links either backend, demonstrates prefill/decode + the KV-rollback tool-call branch. |
| `c/bench_abi_vs_pipe.c` | A/B benchmark: same engine/model/workload driven over pipes vs in-process. |
| `c/tools/make_glm_tiny_fixture.py` | Stdlib-only tiny GLM fixture generator (writes safetensors by hand — no torch/transformers/numpy). |
| `c/engine_demo.sh`, `c/run_tiny_glm_demo.sh`, `c/run_olmoe_demo.sh`, `c/run_bench_abi_vs_pipe.sh` | Build + run entry points. |
| `c/Makefile` | Targets: `libcolibri_engine.o`, `libolmoe_engine.o`, `engine_registry.o`, `agent_main`, `agent_main_olmoe`, `bench_glm`, `bench_olmoe`, `engine-clean`. |
| `orchestration-benchmark.html` (repo root) | Earlier results page: C/C++/Python orchestration vs a stub engine. |
| `c/bench_orchestration/` | Earlier harness: same 3 stages in C / C++ / Python against a stub engine. |

## 3. Verified results

**Path 3 — tiny GLM fixture, zero download** (`bash c/run_tiny_glm_demo.sh`):
load → tokenize → prefill → decode → detokenize, plus a real KV surgery:
`kv rewound 93 -> 45 (discarded 48 draft tokens)`, `prefix 45 reused + 49 tool tokens + 48 decoded`.

**Path 1 — real OLMoE weights** (`bash c/run_olmoe_demo.sh ~/Models/olmoe_i8`):
7 GB int8 container converted from `allenai/OLMoE-1B-7B-0125-Instruct`; real language
output; `kv rewound 62 -> 14`. Same harness binary shape, only the backend differs.

**Transport A/B** (`bash c/run_bench_abi_vs_pipe.sh`, `REPS_A=30`):
paired alternating blocks, 1 token/req, full prefill of a unique prompt each turn
(audited: `[API] KV slot 0 prefix 1/31, prefill 30`).

- **22/30 reps favour the ABI, sign test p ≈ 0.02 → significant.**
- **Median paired delta +370 µs/turn** (pipe slower). Corrected from a script bug
  that dropped negative deltas and reported +488.
- Real OLMoE (n=2): 1/2 — no signal; consistent with <1% of a multi-second turn.
- Conclusion: the pipe costs a real ~0.37 ms/turn, **not milliseconds** as the
  original proposal claimed. Framing alone was 10–80 µs (stub benchmark); the rest
  is serve-path per-turn work (re-tokenize, KV prefix match, per-token `fflush`,
  `DONE`/`PROF` bookkeeping, two process wakeups).

## 4. Bugs found and fixed (do not re-hit these)

1. **`kv_alloc` missing in the ABI.** `model_init_range` does *not* allocate KV rows
   (`Lc`/`Rc`/`Ic`); every engine entry point calls `kv_alloc()` first. Without it the
   first attention read NULL-derefs. Now called in both `*_engine_open`.
2. **`max_context_tokens` read before `kv_alloc`** (returned 0). Now reports the
   allocated context (`COLI_ENGINE_CTX`, default 4096).
3. **Stop-token double-free.** The engine freed the caller's logits and the harness
   freed again. Contract is now: caller owns `logits` always; a stop is
   `*token_id = -1` with success (documented in `engine.h`).
4. **Weak-symbol backend registration fails on Mach-O** — neither `weak` nor
   `weak_import` produced a weak undefined reference. Removed; backends self-register
   via `__attribute__((constructor))`.
5. **`coli_omp_tune_threads()` missing in the library path.** It lives in `main()`,
   which `COLI_NO_MAIN`/`OLMOE_NO_MAIN` remove, so the ABI ran at the OpenMP default
   (18 logical CPUs) instead of the tuned 6 physical cores — worth ~1.8x on Apple
   Silicon SMT. Now called in both `*_engine_open`.
6. **Benchmark: KV prefix reuse made the arms incomparable.** The serve path reuses
   the prefix and prefilled **0** tokens after turn 1 while the ABI re-prefilled all.
   Fixed by using a **unique prompt per turn** (varying marker first) in both arms.
7. **Serve path writes the whole KV to disk every turn** (`kv_disk_append` in both
   `mux_submit` and `mux_done`) — charged to the pipe arm only. Fixed with `KVSAVE=0`
   in the child.
8. **`median_num` silently dropped negative deltas** (`grep -E '^[0-9.]+$'`), biasing
   the paired median. Fixed to `^-?[0-9.]+$`.
9. **Pipe arm never consumed `DONE`**, desyncing the next iteration. Now read, and
   counted inside the timed region (a pipe turn isn't complete until `DONE`).

## 5. Environment gotchas

- **macOS/conda:** `KMP_DUPLICATE_LIB_OK=TRUE` is required for anything importing
  torch (conda libomp + torch libomp collide; the runtime aborts with OMP Error #15).
- **`convert_olmoe_merged.py` segfaults with default threading.** Run it with
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (it is resumable; `_inflight/` is reused).
- **This agent environment has no shell.** All commands were run by the user from
  pasted one-liners/scripts — see the `run-on-machine` pattern.
- The `g_uring` (colibri.c) and `g_fused3` (olmoe.c) "unused variable" warnings are
  expected: `*_NO_MAIN` removes `main()`, which was their only user.

## 6. Open items / next steps

1. **Tier policy is not exposed through the ABI.** `DEMAND_POLICY`, `GROUP_EVICT`,
   `EXPERT_DIRECT`, `PILOT*`, `MARKOV`, `RECENT_*` are **env-only**, read inside
   `model_init_range` (`c/olmoe.c` ~887, 958, 972, 986). `ColiEngineOptions` only has
   `model_dir` / `memory_limit_bytes` / `backend_mask`. Add a `ColiTierOptions`
   sub-struct and a `coli_engine_stats()` (hits / misses / pins / `disk_ns`) so a
   harness can configure and observe the cache instead of setting process env vars.
   See `docs/experiments/olmoe-tier-caching-2026-09-29.md` — that work's wins
   (+45.7% tok/s, −54% P99) came from **policy control**, the same class of win the
   ABI should own.
2. **The A/B benchmark is prefill-dominated** (full prefill + 1 decode). Re-run it
   **decode-heavy** (fixed prompt, ~200 decode tokens) — the regime the tier work
   optimizes — to check the +370 µs figure there.
3. **Add exactness checking** (assert token-identical output across arms) and record
   machine load. The tier report does both; this benchmark does neither, and the
   per-turn spread (2.3–6.5x) is driven by page-cache/routing variance that load
   gating would surface.
4. **Chat template.** The harness sends raw tokenized text; OLMoE output carries
   artifacts (`|||IP_ADDRESS|||`) because no chat template is applied. Add a
   per-family template hook to the harness.
5. **Other backends not wired:** `qwen36.c`, `glm53.c`, `qwen38.c`, `deepseek_v4.c`,
   `kimi_k3.c`, `inkling.c` still have no `COLI_ENGINE_ADAPTER` block. The pattern is
   established (two working examples); each needs its own `step`/tokenizer/sampler
   mapped to the vtable.

## 7. How to reproduce

```bash
# build everything (engines + ABI libs + harnesses + benches)
cd c && make -s colibri olmoe bench_glm bench_olmoe agent_main agent_main_olmoe

# path 3 — tiny GLM fixture, no download
bash c/run_tiny_glm_demo.sh

# path 1 — real OLMoE (convert once, ~7 GB, resumable)
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 c/tools/convert_olmoe_merged.py \
    --repo allenai/OLMoE-1B-7B-0125-Instruct --out ~/Models/olmoe_i8
bash c/run_olmoe_demo.sh ~/Models/olmoe_i8

# transport A/B (30 paired reps for a conclusive sign test)
REPS_A=30 bash c/run_bench_abi_vs_pipe.sh
```

## 8. Suggested skills for the next session

- **`run-on-machine`** — the established pattern here: the agent cannot execute shell,
  so it writes a self-contained script and the user runs one `bash <file>` line.
- **`codebase-memory`** — for engine internals (`step`, `kv_alloc`, `model_init_range`,
  the tier policy functions) before touching them.
- **`handoff`** — when this workstream is picked up again.
