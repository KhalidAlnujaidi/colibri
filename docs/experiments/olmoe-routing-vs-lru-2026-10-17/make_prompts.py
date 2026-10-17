#!/usr/bin/env python3
"""Build the prompt set and one olmoe ref.json per prompt.

Four categories x four prompts. Long-context prompts are ~1500-token excerpts
of colibri docs followed by a question. full_ids is prompt + NGEN filler: the
ref harness generates len(full)-len(prompt) tokens greedily; the filler is
never compared to anything we use.
"""
import json, sys
from pathlib import Path
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[3]
SNAP = Path(sys.argv[1])
OUT = Path(sys.argv[2])
NGEN = int(sys.argv[3]) if len(sys.argv) > 3 else 128
OUT.mkdir(parents=True, exist_ok=True)

P = {
 "chat": [
  "User: I just moved to a new city and I don't know anyone. How can I make friends as an adult?\nAssistant:",
  "User: My sourdough starter smells like nail polish remover. Is it dead, and what should I do?\nAssistant:",
  "User: Can you help me plan a three-day trip to Lisbon on a modest budget? I like food and history.\nAssistant:",
  "User: What are some good ways to stay focused when working from home with kids around?\nAssistant:",
 ],
 "code": [
  "# Python 3\n# Write a function that returns the longest palindromic substring of s.\ndef longest_palindrome(s: str) -> str:\n",
  "// C\n// Implement a thread-safe bounded ring buffer with push and pop using pthread mutex and condition variables.\n#include <pthread.h>\n#include <stdlib.h>\n\ntypedef struct {\n",
  "-- SQL\n-- Tables: orders(id, customer_id, total, created_at), customers(id, name, country).\n-- Return the top 5 countries by total revenue in 2024, with the number of distinct customers.\nSELECT",
  "// Rust\n// Parse a CSV line respecting double-quoted fields with escaped quotes, returning Vec<String>.\nfn parse_csv_line(line: &str) -> Vec<String> {\n",
 ],
 "multilingual": [
  "Pregunta: ¿Cuáles son las principales causas del cambio climático y qué pueden hacer los ciudadanos para reducir su impacto?\nRespuesta:",
  "Question : Pourquoi la Révolution française a-t-elle commencé en 1789 ? Expliquez les causes principales.\nRéponse :",
  "Frage: Wie funktioniert eine Wärmepumpe, und lohnt sie sich in einem älteren Haus?\nAntwort:",
  "سؤال: ما هي فوائد قراءة الكتب يوميًا، وكيف يمكن بناء عادة القراءة؟\nالجواب:",
 ],
 "longctx": [
  ("docs/cuda.md", "Question: Summarize, in a few sentences, what the document above says about how experts are placed on the GPU and what limits it.\nAnswer:"),
  ("docs/windows.md", "Question: According to the document above, what are the main steps and pitfalls for running colibri on Windows?\nAnswer:"),
  ("docs/api.md", "Question: Based on the document above, describe the HTTP endpoints the server exposes and how a client should call them.\nAnswer:"),
  ("docs/FORMATS.md", "Question: Using the document above, explain the difference between the expert storage formats it describes.\nAnswer:"),
 ],
}

tok = Tokenizer.from_file(str(SNAP / "tokenizer.json"))
LONG_TOKENS = 1500
manifest = []
for cat, items in P.items():
    for i, item in enumerate(items, 1):
        if cat == "longctx":
            src, q = item
            body_ids = tok.encode((REPO / src).read_text(), add_special_tokens=False).ids[:LONG_TOKENS]
            text = tok.decode(body_ids) + "\n\n" + q
            meta = {"source": src, "excerpt_tokens": LONG_TOKENS, "question": q}
        else:
            text = item
            meta = {"text": text}
        ids = tok.encode(text, add_special_tokens=False).ids
        name = f"{cat}_{i}"
        (OUT / f"{name}.ref.json").write_text(json.dumps({"prompt_ids": ids, "full_ids": ids + [0] * NGEN}))
        manifest.append({"name": name, "category": cat, "prompt_tokens": len(ids), "ngen": NGEN, **meta})
(OUT / "prompts.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
for m in manifest:
    print(m["name"], m["prompt_tokens"])
