#!/usr/bin/env python3
"""One table per sweep JSON: felt-wait gain vs uniform LRU per budget/candidate."""
import json, sys

SLOT = 16 * 6307840
for path in sys.argv[1:]:
    d = json.load(open(path))
    print(f"\n## {path}")
    for sw in d["sweeps"]:
        cap = round(sw["expert_budget_gb"] * 1e9 / SLOT)
        sens = {(c["allocation"], c["policy"]): c["pass_all_scales"]
                for c in sw["sensitivity"]["candidates"]}
        base = sw["result"]["results"]["uniform"]["lru"]["aggregate"]
        hr = base.get("hit_rate")
        print(f"\ncap {cap}/layer ({sw['expert_budget_gb']:.3f} GB), uniform LRU hit {hr*100 if hr is not None else float('nan'):.1f}%")
        rows = sorted(sw["gate"]["candidates"], key=lambda c: -c["category_mean_gain"])
        cats = sorted(rows[0]["categories"]) if rows else []
        print("  candidate                      mean   worst  " + " ".join(f"{c[:6]:>7}" for c in cats) + "  gate  sens")
        for c in rows:
            per = " ".join(f"{c['categories'][k]['gain']*100:+6.1f}%" for k in cats)
            print(f"  {c['allocation'] + '/' + c['policy']:30s} {c['category_mean_gain']*100:+5.1f}% {c['worst_category_gain']*100:+5.1f}%  {per}  "
                  f"{'PASS' if c['pass'] else 'fail'}  {'PASS' if sens.get((c['allocation'], c['policy'])) else 'fail'}")
