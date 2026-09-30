#!/usr/bin/env python3
"""Generate a tiny GLM (glm_moe_dsa) fixture WITHOUT torch / transformers / numpy.

This is the dependency-free fallback for `tools/make_glm_oracle.py` (which needs
torch + transformers>=5.11 + GlmMoeDsaForCausalLM). It writes the same shape of
fixture the C engine loads:

    <out>/model.safetensors   F32, no .qs sidecars (raw tensors)
    <out>/config.json         the schema load_cfg() reads

The engine loads raw F32 tensors natively when run at full precision (bits=16,
the same `16 16` the repo's own tiny oracle uses): qt_from_disk takes the
no-.qs branch -> qt_alloc(bits>=16) -> fmt=0 -> st_read_f32.

Tensor names and shapes mirror model_init_range() in colibri.c:
  - global: model.embed_tokens.weight, lm_head.weight, model.norm.weight
  - per layer: input/post_attention layernorms, q_a/q_b, kv_a/kv_b, o_proj
  - dense layers (i < first_k_dense_replace): mlp.gate/up/down_proj
  - sparse layers: mlp.gate.weight, mlp.gate.e_score_correction_bias,
    mlp.shared_experts.{gate,up,down}_proj, mlp.experts.{e}.{gate,up,down}_proj

The DSA indexer tensors are deliberately omitted: has_dsa auto-detects to 0 when
they are absent, and the indexer is a no-op for sequences shorter than
index_topk anyway (see the engine header comment).

WEIGHTS ARE RANDOM. This is not a language model; it exists to exercise the
engine ABI (load / prefill / decode / kv_rollback) with no model download.

Usage (from c/):
    python3 tools/make_glm_tiny_fixture.py --output glm_tiny
"""
from __future__ import annotations

import argparse
import json
import random
import struct
from array import array
from pathlib import Path

# ---- tiny architecture (mirrors make_glm_oracle.py's default f32 fixture) ----
VOCAB = 288          # 256 byte tokens + 25 GLM specials (0..280) + headroom
HIDDEN = 128
DENSE_INTER = 64
MOE_INTER = 32
LAYERS = 5           # 3 dense + 2 sparse
FIRST_DENSE = 3
HEADS = 4
EXPERTS = 8
TOPK = 2
SHARED = 1
Q_LORA = 64
KV_LORA = 32
QK_NOPE = 24
QK_ROPE = 8
V_HEAD = 32
QK_HEAD = QK_NOPE + QK_ROPE
INDEX_TOPK = 4096
INDEX_NH = 2
INDEX_HD = 16

SEED = 1234


def write_safetensors(path: Path, tensors: dict[str, tuple[tuple[int, ...], array]]) -> int:
    """Write an F32 safetensors file (spec: u64 LE header length + JSON + data)."""
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, (shape, data) in tensors.items():
        blob = data.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
        blobs.append(blob)

    hjson = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    pad = (-len(hjson)) % 8          # spec: header padded to 8-byte alignment
    hjson += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for blob in blobs:
            f.write(blob)
    return offset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="glm_tiny")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    def rnd(n: int) -> array:
        """N(0, 0.02) weights — non-trivial, small enough not to blow up."""
        return array("f", [rng.gauss(0.0, 0.02) for _ in range(n)])

    def ones(n: int) -> array:
        return array("f", [1.0] * n)

    t: dict[str, tuple[tuple[int, ...], array]] = {}

    # ---- global ----
    t["model.embed_tokens.weight"] = ((VOCAB, HIDDEN), rnd(VOCAB * HIDDEN))
    t["lm_head.weight"] = ((VOCAB, HIDDEN), rnd(VOCAB * HIDDEN))
    t["model.norm.weight"] = ((HIDDEN,), ones(HIDDEN))

    for i in range(LAYERS):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = ((HIDDEN,), ones(HIDDEN))
        t[p + "post_attention_layernorm.weight"] = ((HIDDEN,), ones(HIDDEN))
        # MLA attention (q/kv LoRA)
        t[p + "self_attn.q_a_proj.weight"] = ((Q_LORA, HIDDEN), rnd(Q_LORA * HIDDEN))
        t[p + "self_attn.q_a_layernorm.weight"] = ((Q_LORA,), ones(Q_LORA))
        t[p + "self_attn.q_b_proj.weight"] = ((HEADS * QK_HEAD, Q_LORA), rnd(HEADS * QK_HEAD * Q_LORA))
        t[p + "self_attn.kv_a_proj_with_mqa.weight"] = (
            (KV_LORA + QK_ROPE, HIDDEN), rnd((KV_LORA + QK_ROPE) * HIDDEN))
        t[p + "self_attn.kv_a_layernorm.weight"] = ((KV_LORA,), ones(KV_LORA))
        t[p + "self_attn.kv_b_proj.weight"] = (
            (HEADS * (QK_NOPE + V_HEAD), KV_LORA), rnd(HEADS * (QK_NOPE + V_HEAD) * KV_LORA))
        t[p + "self_attn.o_proj.weight"] = ((HIDDEN, HEADS * V_HEAD), rnd(HIDDEN * HEADS * V_HEAD))

        if i < FIRST_DENSE:
            t[p + "mlp.gate_proj.weight"] = ((DENSE_INTER, HIDDEN), rnd(DENSE_INTER * HIDDEN))
            t[p + "mlp.up_proj.weight"] = ((DENSE_INTER, HIDDEN), rnd(DENSE_INTER * HIDDEN))
            t[p + "mlp.down_proj.weight"] = ((HIDDEN, DENSE_INTER), rnd(HIDDEN * DENSE_INTER))
        else:
            t[p + "mlp.gate.weight"] = ((EXPERTS, HIDDEN), rnd(EXPERTS * HIDDEN))
            t[p + "mlp.gate.e_score_correction_bias"] = (
                (EXPERTS,), array("f", [-0.1 + 0.2 * e / max(EXPERTS - 1, 1) for e in range(EXPERTS)]))
            si = SHARED * MOE_INTER
            t[p + "mlp.shared_experts.gate_proj.weight"] = ((si, HIDDEN), rnd(si * HIDDEN))
            t[p + "mlp.shared_experts.up_proj.weight"] = ((si, HIDDEN), rnd(si * HIDDEN))
            t[p + "mlp.shared_experts.down_proj.weight"] = ((HIDDEN, si), rnd(HIDDEN * si))
            for e in range(EXPERTS):
                ep = p + f"mlp.experts.{e}."
                t[ep + "gate_proj.weight"] = ((MOE_INTER, HIDDEN), rnd(MOE_INTER * HIDDEN))
                t[ep + "up_proj.weight"] = ((MOE_INTER, HIDDEN), rnd(MOE_INTER * HIDDEN))
                t[ep + "down_proj.weight"] = ((HIDDEN, MOE_INTER), rnd(HIDDEN * MOE_INTER))

    n_bytes = write_safetensors(out / "model.safetensors", t)

    config = {
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOPK,
        "moe_intermediate_size": MOE_INTER,
        "intermediate_size": DENSE_INTER,
        "first_k_dense_replace": FIRST_DENSE,
        "q_lora_rank": Q_LORA,
        "kv_lora_rank": KV_LORA,
        "qk_nope_head_dim": QK_NOPE,
        "qk_rope_head_dim": QK_ROPE,
        "v_head_dim": V_HEAD,
        "n_shared_experts": SHARED,
        "vocab_size": VOCAB,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "rms_norm_eps": 1e-5,
        "routed_scaling_factor": 2.5,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "max_position_embeddings": 4096,
        "index_topk": INDEX_TOPK,
        "index_n_heads": INDEX_NH,
        "index_head_dim": INDEX_HD,
        "eos_token_id": 256,          # <|endoftext|> in make_glm53_tokenizer.py
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    total = 0
    for shape, data in t.values():
        n = 1
        for dim in shape:
            n *= dim
        total += n

    print(f"wrote {out}/model.safetensors  ({len(t)} tensors, {total} params, "
          f"{n_bytes / 1e6:.1f} MB f32)")
    print(f"wrote {out}/config.json        (vocab={VOCAB} hidden={HIDDEN} layers={LAYERS} "
          f"experts={EXPERTS} dense={FIRST_DENSE})")
    print("next: python3 tools/make_glm53_tokenizer.py --output "
          f"{out}/tokenizer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
