# OLMoE: does routing history place experts better than LRU on held-out prompts?

The README's first open question asks whether routing history can place experts
better than plain LRU on held-out, cross-session workloads. This is the Phase-0
answer for OLMoE-1B-7B, offline, from real `ROUTE_TRACE` streams replayed through
`c/tools/residency_sim.py --engine olmoe`. It is a simulation, not a runtime A/B:
no tok/s is claimed.

**Result, in three parts:**

1. **Routing history, as the simulator's history policies use it, loses to LRU.**
   `frequency`, `lfu` and `half-pinned` seed the training traces' counts or pins.
   On held-out prompts none passes the gate at any budget from 8 to 48 slots per
   layer, and on chat and multilingual they cost up to 2.2x LRU's felt wait.
2. **What beats LRU is the replacement rule, learned within the session.**
   `frequency-cold` (frequency admission that never sees a training trace) passes
   the gate, and the per-layer cost sensitivity, at every budget from 8 to 48
   slots: 12.7% to 22.4% less felt wait, category mean, with no category worse
   than LRU.
3. **History adds only a warm start, and the warm start roughly pays for its own
   reads.** `frequency-warm` (history picks the starting residents, then the
   session's counts take over) adds 1 to 11 points over `frequency-cold` at 8 to
   32 slots. The misses it saves are 0.9x to 1.1x the experts it seeds, and the
   simulator does not charge for reading the seeds at startup. In this data the
   warm start moves reads ahead of the first token more than it removes them.

So for OLMoE the hypothesis does not hold in its strong form. What is learned
from other prompts is worth a starting point, about as much as the reads that
load it; seeded as counts or pins it loses to LRU. The gain over LRU comes from
a frequency-aware admission rule learned from the session itself.

## Setup

- **Model:** `allenai/OLMoE-1B-7B-0924` (base), converted with
  `c/tools/convert_olmoe_merged.py`: int8 experts, 16 layers, 64 experts, top-8.
  One expert is 6,307,840 bytes (6,291,456 int8 + 16,384 of scales), both
  resident and read: [manifest-olmoe-residency.json](olmoe-routing-vs-lru-2026-10-17/manifest-olmoe-residency.json).
- **Engine:** `make -C c olmoe` at `9e12a616` (olmoe.c identical to `dev` at
  `12ddfd34`); gcc 13.3, 4 vCPU Xeon @ 2.1 GHz, 15 GB RAM, Linux 6.18.
- **Traces:** one cold process per prompt, CPU only:
  `env -i PATH HOME OMP_NUM_THREADS=4 TEMP=0 PILOT=0 HOT=0 ROUTE_TRACE=<name>.trace SNAP=<model> ./olmoe 64 8 <name>.ref.json`
  ([record.sh](olmoe-routing-vs-lru-2026-10-17/record.sh)). Greedy, 128 new
  tokens each. No pins, no prefetch, no `COLI_USAGE`. A rerun of `chat_1` at
  `OMP_NUM_THREADS=2` produced a byte-identical trace.
- **Prompts:** 16, four per category ([prompts.json](olmoe-routing-vs-lru-2026-10-17/prompts.json),
  the exact token ids in [refs/](olmoe-routing-vs-lru-2026-10-17/refs)):
  chat (22-29 tokens), code in Python, C, SQL and Rust (37-51), multilingual in
  Spanish, French, German and Arabic (36-59), and long context: the first 1,500
  tokens of four colibri docs and a question about them (1,523-1,534).
  Trace checksums: [traces.sha256](olmoe-routing-vs-lru-2026-10-17/traces.sha256).
- **Splits** ([sweep.sh](olmoe-routing-vs-lru-2026-10-17/sweep.sh)):
  *within-category*: train on prompt 1 of each category, evaluate on prompts 2-4.
  *Cross-category*: train on all prompts of three categories, evaluate on the
  fourth; one run per held-out category.
- **Budgets:** 4, 8, 12, 16, 24, 32 and 48 slots per layer (0.40 to 4.84 GB),
  uniform and trace-trained per-layer allocations. Felt wait is every miss at
  the same cost (`--read-gbps 3 --felt-fraction 1`), so percentages are miss
  reductions; sensitivity rescales layers 0, 7 and 15 by 0.5x and 1.5x.
- **Gate:** the simulator's Phase-0 gate, at least 10% less felt wait than
  uniform LRU averaged over categories, no category more than 3% worse.

Four policies are new in `residency_sim.py` for this record, as controls:
`lru-warm` (LRU starting with the training top-k), `lfu-cold` and
`frequency-cold` (`lfu` and `frequency` with the training counts dropped), and
`frequency-warm` (training top-k as starting residents, each counted once).

## Within-category held-out

Felt-wait gain over uniform LRU, uniform allocation, category mean (worst
category). PASS means the gate and the sensitivity both pass. Every row, the
per-category columns and the dynamic allocations are in
[summary.txt](olmoe-routing-vs-lru-2026-10-17/summary.txt).

| Slots/layer | LRU hit | `frequency` | `lfu` | `half-pinned` | `lru-warm` | `frequency-cold` | `frequency-warm` |
|---|---|---|---|---|---|---|---|
| 8  | 26.3% | -2.7% (-21.3%) | -4.9% (-23.0%) | -10.4% (-28.6%) | +0.1% (+0.1%) | **+17.1% (+10.4%) PASS** | **+18.2% (+10.8%) PASS** |
| 12 | 40.9% | -15.8% (-44.8%) | -17.0% (-46.2%) | -15.9% (-38.7%) | +0.3% (+0.2%) | **+12.7% (+1.2%) PASS** | **+14.5% (+1.9%) PASS** |
| 16 | 49.0% | -21.2% (-60.4%) | -22.1% (-61.6%) | -6.2% (-24.6%) | +0.6% (+0.3%) | **+13.5% (+0.6%) PASS** | **+16.4% (+1.6%) PASS** |
| 24 | 59.9% | -29.0% (-72.9%) | -27.4% (-71.4%) | -2.9% (-21.9%) | +1.6% (+0.6%) | **+18.6% (+7.4%) PASS** | **+24.0% (+9.6%) PASS** |
| 32 | 68.2% | -36.1% (-88.6%) | -32.8% (-83.0%) | -1.4% (-26.1%) | +4.8% (+1.3%) | **+20.4% (+10.7%) PASS** | **+31.6% (+22.7%) PASS** |
| 48 | 79.6% | -24.6% (-106.4%) | -14.2% (-86.6%) | +14.2% (-13.0%) | **+27.7% (+4.4%) PASS** | **+22.4% (+0.1%) PASS** | **+56.8% (+44.4%) PASS** |

At 4 slots, fewer than the 8 experts a token routes, LRU thrashes (3.0% hits)
and every policy but `lru-warm` beats it; that budget says nothing about history.

The seeded history policies fail on chat and multilingual and win on long
context. Long-context prefill dominates the training counts (a 1,500-token
prompt routes about ten times as many rows as a chat prompt), so the seeded
frequencies outweigh a short session's own counts for its whole replay and
`frequency` refuses to admit what the session actually uses. With the training
counts dropped, the same rule wins in every category.

## Cross-category held-out

Train on three categories, evaluate on the fourth. Gain over uniform LRU on the
held-out category:

| Held out | Slots | `frequency` | `half-pinned` | `lru-warm` | `frequency-cold` | `frequency-warm` |
|---|---|---|---|---|---|---|
| chat | 8 / 16 / 32 | -17.4 / -55.4 / -82.3% | -21.2 / -22.1 / -8.9% | +0.1 / +0.4 / +3.2% | +11.2 / +3.4 / +17.7% | +11.6 / +4.4 / +23.6% |
| code | 8 / 16 / 32 | +16.3 / -6.2 / -28.1% | +3.6 / +7.7 / +8.5% | +0.2 / +0.8 / +8.2% | +21.3 / +11.6 / +15.7% | +23.1 / +16.9 / +34.0% |
| multilingual | 8 / 16 / 32 | -21.7 / -52.2 / -119.4% | -28.6 / -16.7 / -31.3% | +0.1 / +0.5 / +5.2% | +18.9 / +20.4 / +12.9% | +19.9 / +22.5 / +23.5% |
| long context | 8 / 16 / 32 | +8.4 / +5.6 / +27.4% | +0.8 / +5.1 / +18.5% | +0.1 / +0.3 / +1.3% | +15.5 / +17.7 / +42.5% | +16.3 / +19.3 / +45.7% |

The pattern holds across categories: seeded history fails wherever the training
mix differs from the held-out category, and the session-learned rule does not.
Chat is the hardest held-out category for the session-learned rule
(`frequency-cold` +3.4% at 16 slots, short of the 10% gate).

## What the warm start is worth

Misses per held-out trace, within-category, and the experts `frequency-warm`
seeds per trace:

| Slots/layer | LRU | `frequency-cold` | `frequency-warm` | seeded | saved by the warm start |
|---|---|---|---|---|---|
| 8  | 14,472 | 12,067 | 11,922 | 128 | 144 |
| 16 | 10,010 | 8,591  | 8,324  | 256 | 267 |
| 32 | 6,241  | 4,353  | 3,873  | 512 | 480 |
| 48 | 4,009  | 1,985  | 1,309  | 768 | 676 |

The warm start saves about one miss per expert it seeds. The simulator counts
seeded experts as free, so if they are read at startup the warm start mostly
moves those reads before the first token. It pays where the seeds load off the
critical path or a long-running server amortizes them; in a cold, short process
it is close to neutral. `lru-warm` saves far less (15 misses for 128 seeds at 8
slots): LRU evicts most seeds before the session reuses them.

## Why recency loses here (inferred from the traversal, not measured)

`olmoe.c` fetches a prompt's experts in blocks of 128 rows, each block fetching
every expert it routed once in ascending id. A long prompt routes most of the 64
experts in every block, so prefill is a repeated scan, which LRU at fewer than
64 slots serves worst: at 16 slots LRU hits 35.7% of long-context fetches,
against 51.6% to 62.9% in the other categories. In decode each token routes 8 of 64 per layer, so 16 slots hold two
tokens of recency. A rule that admits an expert only once it is used more than
the resident it would evict keeps the session's common experts through both.

## Limits

- **Offline only.** Felt wait is misses times a fixed cost. No page cache: with
  the default `EXPERT_DROP=0` and RAM to spare, a miss in `olmoe.c` is a copy
  from the page cache, not a disk read. The result applies when RAM is short.
- **Not implementable as is.** Frequency admission runs a rejected miss without
  caching it, which needs a scratch slot; `expert_get()` always admits today.
- **Small corpus.** 16 prompts, 128 greedy tokens each, a base model; long
  context is colibri documentation only. Default decay and hysteresis, not tuned.
- **One model.** 64 experts, top-8, 16 layers. GLM and Kimi route differently.

## Next step

An opt-in admission rule in `olmoe.c`'s `expert_get()` (decayed per-layer
counts, admit on 10% hysteresis, serve a rejected miss from one scratch slot),
A/B against LRU at 16 slots with `EXPERT_DROP=1` so misses are real reads, three
runs per arm, chat and long-context prompts not seen here.

## Reproduce

```sh
python3 -m pip install torch safetensors huggingface_hub tokenizers
python3 c/tools/convert_olmoe_merged.py --repo allenai/OLMoE-1B-7B-0924 --out olmoe
make -C c olmoe
E=docs/experiments/olmoe-routing-vs-lru-2026-10-17
$E/record.sh olmoe $E/refs traces            # about 12 minutes on 4 cores
(cd traces && sha256sum -c ../$E/traces.sha256)
$E/sweep.sh traces results                   # about 6 minutes
$E/summarize.py results/within-category.json results/cross-*.json
```

`make_prompts.py` rebuilds `refs/` from the prompt texts, but the long-context
excerpts come from docs that change; `refs/` holds the exact ids used.
