#!/usr/bin/env python3
"""Build the Stage-A trace campaign for OLMoE: real agentic text -> engine refs,
grouped into named workload categories, with a strict train / held-out split by
SOURCE SESSION. No session contributes to both sides.

Why split by session and not by line: routing is highly autocorrelated inside a
run (expert_atlas/README.md), so a line-level split leaks — two lines from the
same session route nearly identically, and a table trained on one would score as
if it had generalised. Each ref here is also run in its own cold process, so a
ref is one held-out sample.

Emits, under --out:
  refs/<name>.json    engine ref.json fixtures (+ source_session, category, split)
  split.json          {category: {train: [...], heldout: [...]}}
"""
import argparse, glob, json, os, random, sys
from tokenizers import Tokenizer

sys.path.insert(0, os.path.expanduser("~/dev/mr-meeseeks/bench/prune-spike"))
from extract_corpus import iter_records, text_of, clean  # noqa: E402


BOILERPLATE = (
    "<system-reminder>", "Current runtime context.", "[MNEMON]",
    "Instructions from:", "You are ", "## Ruflo", "Tool definitions",
    "The following workspace instructions", "<hindsight_knowledge>",
    "<workspace_instructions>", "<available_tools>", "<tool_definitions>",
)


def is_boilerplate(s: str) -> bool:
    """Reject harness preamble.

    This is not tidiness: the first few records of every DSH session are the same
    injected context ("Current runtime context. This snapshot supersedes...",
    "<system-reminder> The following workspace instructions..."), so an extractor
    that takes the first long enough record from N sessions extracts the SAME
    TEXT N times. That silently collapses a 4-category campaign into one
    workload and makes every held-out split leak. Caught here by hashing the
    refs: 5 groups of byte-identical ids across 32 refs.
    """
    return any(s.startswith(b) or b in s[:400] for b in BOILERPLATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/olmoe_merged"))
    ap.add_argument("--sessions", default=os.path.expanduser(
        "~/.dsh/sessions/**/session.jsonl.zstd"))
    ap.add_argument("--out", default="stage-a/traces")
    ap.add_argument("--per-category", type=int, default=6,
                    help="refs per category, drawn from DISTINCT sessions")
    ap.add_argument("--seq-tokens", type=int, default=210)
    ap.add_argument("--min-source-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    os.makedirs(os.path.join(a.out, "refs"), exist_ok=True)

    root = os.path.expanduser("~/.dsh/sessions")
    by_cat = {}
    for f in sorted(glob.glob(a.sessions, recursive=True)):
        parts = os.path.relpath(f, root).split(os.sep)
        by_cat.setdefault(parts[0].strip("-"), []).append(f)

    # Categories must have enough DISTINCT sessions to split honestly:
    # >=3 sessions, so >=1 for train and >=2 for held-out.
    eligible = {c: sorted(fs) for c, fs in by_cat.items()
                if len(fs) >= 3 and len({os.path.dirname(f) for f in fs}) >= 3}
    cats = sorted(eligible, key=lambda c: -len(eligible[c]))[:4]
    rng = random.Random(a.seed)

    split = {}
    total = 0
    for cat in cats:
        sessions = sorted({os.path.dirname(f) for f in eligible[cat]})
        rng.shuffle(sessions)
        n_train = max(1, len(sessions) // 3)
        train_sessions = set(sessions[:n_train])
        # Allocate the quota per split explicitly. Iterating in one order and
        # stopping at the quota exhausts it on whichever side comes first — the
        # first version produced 0 held-out refs, the second 0 train refs. Both
        # are silent, so the split is asserted at the end of this loop instead.
        want = {"train": max(1, a.per_category // 4),
                "heldout": a.per_category - max(1, a.per_category // 4)}

        picked = []          # (session, ids)
        seen_text = set()
        for split_name, sess_pool in (("heldout", sessions[n_train:]),
                                      ("train", sessions[:n_train])):
            taken = 0
            for sess in sess_pool:
                if taken >= want[split_name]:
                    break
                for f in [x for x in eligible[cat] if os.path.dirname(x) == sess]:
                    got = False
                    for rec in iter_records(f):
                        s = clean(text_of(rec))
                        if len(s) < 200 or s in seen_text or is_boilerplate(s):
                            continue
                        ids = tok.encode(s, add_special_tokens=False).ids
                        if len(ids) < a.min_source_tokens:
                            continue
                        seen_text.add(s)
                        picked.append((sess, ids[:a.seq_tokens]))
                        got = True
                        taken += 1
                        break        # one ref per session: keeps sessions distinct
                    if got:
                        break
        if len(picked) < a.per_category:
            print(f"  {cat}: only {len(picked)}/{a.per_category} refs, skipped",
                  file=sys.stderr)
            continue

        names_t, names_h = [], []
        short = cat.replace("Users-khalid-", "").replace("--", "-")[:20]
        for i, (sess, ids) in enumerate(picked):
            is_train = sess in train_sessions
            name = f"{short}_{i:02d}"
            with open(os.path.join(a.out, "refs", name + ".json"), "w") as fh:
                json.dump({"prompt": tok.decode(ids[:8]), "prompt_ids": ids[:8],
                           "full_ids": ids, "text": tok.decode(ids),
                           "source_session": os.path.basename(sess),
                           "category": short,
                           "split": "train" if is_train else "heldout"}, fh)
            (names_t if is_train else names_h).append(name)
            total += 1
        split[short] = {"train": names_t, "heldout": names_h,
                        "sessions": len(sessions)}
        print(f"  {short}: {len(names_t)} train / {len(names_h)} held-out "
              f"(from {len(sessions)} sessions, {len(train_sessions)} train sessions)")

    with open(os.path.join(a.out, "split.json"), "w") as fh:
        json.dump({"categories": list(split), "split": split}, fh, indent=1)
    print(f"\n{total} refs -> {a.out}/refs")
    print("train:", sum(len(v['train']) for v in split.values()),
          "| held-out:", sum(len(v['heldout']) for v in split.values()))
    assert all(split[c]["sessions"] >= 3 for c in split), "category under-split"

    # No two refs may share content. If they do, the campaign is really one
    # workload and every held-out number is a re-run of a training sample.
    import hashlib
    digests = {}
    for f in sorted(glob.glob(os.path.join(a.out, "refs", "*.json"))):
        ids = json.load(open(f))["full_ids"]
        digests.setdefault(hashlib.sha256(bytes(str(ids), "utf8")).hexdigest()[:12],
                           []).append(os.path.basename(f))
    dupes = {h: n for h, n in digests.items() if len(n) > 1}
    if dupes:
        for h, n in dupes.items():
            print(f"  DUPLICATE CONTENT {h}: {n}", file=sys.stderr)
        raise SystemExit(f"{len(dupes)} duplicate ref group(s) — corpus is not distinct")

    # And the splits must not share a source session.
    for cat, sp in split.items():
        sess = {json.load(open(os.path.join(a.out, "refs", n + ".json")))["source_session"]
                for n in sp["train"] + sp["heldout"]}
        tr = {json.load(open(os.path.join(a.out, "refs", n + ".json")))["source_session"]
              for n in sp["train"]}
        ho = {json.load(open(os.path.join(a.out, "refs", n + ".json")))["source_session"]
              for n in sp["heldout"]}
        assert not (tr & ho), f"{cat}: session shared between train and held-out"
        assert len(sess) == len(sp["train"]) + len(sp["heldout"]), f"{cat}: session reused"
    print("checked: no duplicate content, no session shared across train/held-out")


if __name__ == "__main__":
    main()
