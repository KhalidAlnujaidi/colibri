#!/usr/bin/env python3
"""Build an order-1 Markov transition table from ROUTE_TRACE captures.

The table answers the question the cross-layer coupling analysis raised: given
the experts a layer just chose for one position, which experts will the SAME
position choose at the next layer? That is the decision the prefetcher has to
make one layer ahead, and today it is made from marginal heat (`.coli_usage`
counts) or from the `.coli_pairs` co-occurrence sum.

Per (layer, expert), keep the top-N successors at the next layer by transition
count. Sparse and fixed-width: the same shape as `.coli_pairs` (CP_M entries per
conditioning key), so the runtime reader can be a table lookup either way.

  COLIMARKOV 1 <n_lines>
  <L> <e> f1:c1 f2:c2 ...        (up to N, count-descending)

Differences from .coli_pairs worth naming:
  * `.coli_pairs` scores over the whole routed SET (sums counts for every expert
    the layer chose) — a mixture approximation. This table is per-expert and can
    be scored the same way, but it is also directly a next-expert distribution.
  * order 1 only. A second-order table is E*N wider for a recall gain this
    corpus cannot justify; see --report, which prints the measured order-0 vs
    order-1 recall gap so the decision is made from data rather than taste.
  * `--decay` applies exponential weighting by capture order, so a later trace in
    the campaign counts more. Off (1.0) means every trace weighs the same.

Held-out discipline: the caller must pass train traces only to build, and
separate held-out traces to score. This tool never mixes them; --score refuses
to read a file that was also used to build.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

EXPERT_FIELD = re.compile(r"^(\d+):([^:]+)$")


def read_trace(path: Path) -> dict[int, dict[int, list[int]]]:
    """-> {position: {layer: [expert ids in selection order]}}.

    A position is one (forward, row): the units a transition is defined over, and
    the unit the engine prefetches for. Getting this key right is the whole
    parser, and the obvious choice is wrong: `call` advances once per LAYER
    (see the PR that made olmoe emit this stream), so keying on (call, row)
    yields 3344 single-layer positions and *no* transitions at all. The forward
    boundary is instead where the layer index wraps back to a lower value —
    the same regrouping tools/route_pairs.py does for the same reason.

    Rows are numbered within each forward, so positions never collide.
    """
    positions: dict[int, dict[int, list[int]]] = defaultdict(dict)
    fwd = 0
    last_layer = -1
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        f = raw.split()
        if not f:
            continue
        if len(f) < 4:
            raise ValueError(f"{path}:{lineno}: expected call row layer expert:gate ...")
        try:
            _call, row, layer = map(int, f[:3])
        except ValueError as e:
            raise ValueError(f"{path}:{lineno}: invalid call/row/layer") from e
        if layer < last_layer:          # layer index wrapped: a new forward began
            fwd += 1
        last_layer = layer
        ids = []
        for v in f[3:]:
            m = EXPERT_FIELD.fullmatch(v)
            if not m:
                raise ValueError(f"{path}:{lineno}: invalid expert field {v!r}")
            ids.append(int(m.group(1)))
        positions[(fwd, row)][layer] = ids
    return positions


def build(paths: list[Path], n: int, decay: float) -> tuple[dict, dict]:
    """-> (transition counts, marginal counts). Applies `decay` per trace."""
    trans: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    marg: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    n_traces = len(paths)
    for ti, path in enumerate(paths):
        w = decay ** (n_traces - 1 - ti)          # later traces weigh more
        for pos, by_layer in read_trace(path).items():
            for layer, ids in by_layer.items():
                for e in ids:
                    marg[layer][e] += w
                nxt = by_layer.get(layer + 1)
                if not nxt:
                    continue
                for e in ids:
                    d = trans[layer, e]
                    for f in nxt:
                        d[f] += w
    return trans, marg


def entries(trans, n):
    out = []
    for (layer, e), d in trans.items():
        top = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
        out.append((layer, e, top))
    out.sort()
    return out


def score(paths, table, topn):
    """Recall of the true next-layer set under a fixed candidate budget.

    Two predictors, same conditioning information, same budget:

      marginal : top-N experts by marginal heat at layer L+1 — the .coli_usage view
      markov   : top-N successors of EACH expert routed at layer L, unioned

    The union is the point and it is not a detail. Truncating the union back to N
    (the obvious implementation, and this function's first version) throws away
    the per-expert structure and scores 30.6% where the union scores 82.7% on the
    same held-out data. The budget being compared is "N candidates prefetched per
    conditioning expert", which is what the engine's prefetch queue actually
    bounds, not "N entries in a merged list".
    """
    hits = {"marginal": 0, "markov": 0}
    candidates = {"marginal": 0, "markov": 0}
    tot = 0
    marg = table["marg"]
    trans = table["trans"]
    for path in paths:
        for _pos, by_layer in read_trace(path).items():
            for layer, ids in by_layer.items():
                nxt = by_layer.get(layer + 1)
                if not nxt:
                    continue
                true = set(nxt)
                heat = sorted(marg[layer + 1],
                              key=lambda e: -marg[layer + 1][e])[:topn]
                pred = set()
                for e in ids:
                    for f, _c in trans.get((layer, e), [])[:topn]:
                        pred.add(f)
                hits["marginal"] += len(true & set(heat))
                hits["markov"] += len(true & pred)
                candidates["marginal"] += len(heat)
                candidates["markov"] += len(pred)
                tot += len(true)
    recall = {k: (v / tot if tot else 0.0) for k, v in hits.items()}
    return recall, tot, {k: (v / tot if tot else 0.0) for k, v in candidates.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output table path")
    ap.add_argument("--train", nargs="+", required=True, type=Path)
    ap.add_argument("--score", nargs="+", default=[], type=Path,
                    help="held-out traces to score on (must not be in --train)")
    ap.add_argument("-n", type=int, default=8, help="successors kept per expert")
    ap.add_argument("--decay", type=float, default=1.0,
                    help="per-trace EMA weight, applied by capture order (1.0=off)")
    ap.add_argument("--topn", type=int, default=8, help="prefetch budget when scoring")
    ap.add_argument("--report", action="store_true",
                    help="print the order-0 vs order-1 recall comparison")
    a = ap.parse_args()

    train = [p.resolve() for p in a.train]
    held = [p.resolve() for p in a.score]
    overlap = set(train) & set(held)
    if overlap:
        raise SystemExit(f"refusing: {sorted(map(str, overlap))} in both --train and --score")
    for p in train + held:
        if not p.exists():
            raise SystemExit(f"missing trace {p}")

    trans, marg = build(train, a.n, a.decay)
    tbl = entries(trans, a.n)
    with open(a.out, "w") as f:
        f.write(f"COLIMARKOV 1 {len(tbl)}\n")
        for layer, e, top in tbl:
            f.write(f"{layer} {e} " + " ".join(f"{f}:{c:.0f}" for f, c in top) + "\n")

    n_bytes = Path(a.out).stat().st_size
    print(f"{a.out}: {len(tbl)} conditioning entries, {n_bytes} bytes "
          f"({n_bytes/1024:.1f} KB), n={a.n}, decay={a.decay}", file=sys.stderr)

    if a.report and held:
        table = {"trans": {(l, e): t for l, e, t in tbl}, "marg": marg}
        r, tot, cand = score(held, table, a.topn)
        print(f"held-out {len(held)} traces, {tot} next-layer selections, "
              f"budget {a.topn}/layer", file=sys.stderr)
        for k, v in sorted(r.items()):
            print(f"  {k:9} recall {v:6.2%}  ({cand[k]:.1f} candidates/selection)",
                  file=sys.stderr)
        print(f"  order-1 gain over marginal heat: {(r['markov']-r['marginal'])*100:+.1f}pp",
              file=sys.stderr)
    if a.score:
        json.dump({"entries": len(tbl), "bytes": n_bytes, "n": a.n},
                  open(a.out + ".meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
