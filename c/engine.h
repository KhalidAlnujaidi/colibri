#ifndef COLIBRI_ENGINE_H
#define COLIBRI_ENGINE_H

/*
 * Public, engine-neutral full-token inference ABI for Colibri.
 *
 * This is the "Unified C Library (Shared Harness)" seam: it decouples a model
 * family's compute and weights from main() and the pipe protocol loop, so an
 * external harness (agent_main.c) can drive ANY model architecture through one
 * interface instead of each family re-implementing its own serve loop.
 *
 * Scope and placement:
 *   - This ABI mirrors the existing ColiEdgeAdapter / ColiSegmentAdapter
 *     registries: a struct-of-function-pointers vtable, explicit registration,
 *     engine-id lookup, and caller-allocated structs with struct_size/abi_version
 *     for forward/backward compatibility.
 *   - Unlike ColiEdgeAdapter (tokenizer + embed + final head only) and
 *     ColiSegmentAdapter (layer ranges / opaque sessions), this contract is the
 *     complete token-level engine: load, tokenize, prefill, autoregressive
 *     decode, sample, and KV-cache surgery (kv_get_len / kv_rollback). It is the
 *     surface an agent loop needs to do tool-call branching without a pipe.
 *
 * KV position model:
 *   The engine is position-addressed (rows are written at absolute positions
 *   0,1,2,...). The harness keeps a position cursor; engine_prefill/decode_step
 *   advance it, and engine_kv_rollback rewinds it to an earlier checkpoint
 *   (O(1) pointer/cursor adjustment; the next prefill at that position
 *   overwrites the stale rows). This is the in-memory, zero-copy analog of the
 *   pipe path's "rewind + recompute common prefix".
 *
 * Sampling is caller policy: engine_sample applies temperature/top-p/stop-set
 * and returns one token id. RNG lives in the engine (shared sample.h), seeded
 * once at engine_open.
 *
 * Registration is explicit; ordinary Colibri executables do not use this API.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_ENGINE_ABI_VERSION 1u
#define COLI_ENGINE_ENGINE_ID_CAP 64u
#define COLI_ENGINE_STATE_SCHEMA_CAP 128u
#define COLI_ENGINE_NUMERIC_CLASS_CAP 96u
#define COLI_ENGINE_TOKENIZER_CLASS_CAP 64u

typedef struct ColiEngine ColiEngine;

/* Backend / capability flags (bit-coded, like ColiEdgeAdapter). */
enum {
    COLI_ENGINE_CAP_PREFILL  = UINT64_C(1) << 0,
    COLI_ENGINE_CAP_DECODE   = UINT64_C(1) << 1,
    COLI_ENGINE_CAP_KV_SURGERY = UINT64_C(1) << 2,
    COLI_ENGINE_CAP_GRAMMAR  = UINT64_C(1) << 3,
    COLI_ENGINE_CAP_CPU      = UINT64_C(1) << 8,
    COLI_ENGINE_CAP_CUDA     = UINT64_C(1) << 9,
    COLI_ENGINE_CAP_HIP      = UINT64_C(1) << 10,
    COLI_ENGINE_CAP_METAL    = UINT64_C(1) << 11,
    COLI_ENGINE_CAP_VULKAN   = UINT64_C(1) << 12,
};

#define COLI_ENGINE_CAP_BACKEND_MASK                                    \
    (COLI_ENGINE_CAP_CPU | COLI_ENGINE_CAP_CUDA | COLI_ENGINE_CAP_HIP | \
     COLI_ENGINE_CAP_METAL | COLI_ENGINE_CAP_VULKAN)

/* Immutable facts a harness needs before driving the engine. */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t flags;
    char engine_id[COLI_ENGINE_ENGINE_ID_CAP];
    char state_schema[COLI_ENGINE_STATE_SCHEMA_CAP];
    char numeric_class[COLI_ENGINE_NUMERIC_CLASS_CAP];
    char tokenizer_class[COLI_ENGINE_TOKENIZER_CLASS_CAP];
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t max_context_tokens;
    int32_t bos_token_id;
    int32_t eos_token_id;
    uint32_t reserved_u32[4];
    uint64_t resident_bytes;
    uint64_t reserved_u64[3];
} ColiEngineCapabilities;

/* Options passed to engine_open. */
typedef struct {
    uint32_t struct_size;
    const char *model_dir;          /* directory with the shard/tokenizer files */
    uint64_t memory_limit_bytes;    /* 0 = engine default (cap) */
    uint64_t backend_mask;          /* desired backend flags (0 = any) */
    uint64_t reserved_u64[3];
} ColiEngineOptions;

/* Sampling policy. stop_tokens may be NULL when n_stop_tokens == 0. */
typedef struct {
    float temperature;              /* <= 0 -> greedy (argmax) */
    float top_p;                    /* 0..1 nucleus; 0 -> no truncation */
    const int32_t *stop_tokens;
    size_t n_stop_tokens;
    int32_t ban;                    /* >= 0: exclude this token id (spec reject) */
    uint64_t reserved_u64[2];
} ColiSampleConfig;

/* The engine vtable. engine_id must be a stable, unique string ("glm",
 * "qwen36", ...). The reserved_fn array keeps the struct ABI-extensible. */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    const char *engine_id;

    int (*engine_open)(void **engine_impl, ColiEngineCapabilities *capabilities,
                       const ColiEngineOptions *options,
                       char *error, size_t error_size);
    void (*engine_destroy)(void *engine_impl);

    int (*tokenize)(void *engine_impl, const char *text, size_t text_bytes,
                    int32_t *token_ids, size_t token_capacity,
                    size_t *token_count, char *error, size_t error_size);
    int (*detokenize)(void *engine_impl, const int32_t *token_ids,
                      size_t token_count, char *text, size_t text_capacity,
                      size_t *text_bytes, char *error, size_t error_size);

    /* Prefill: ingest n_tokens starting at the current KV position; returns
     * logits [vocab] for the final token (caller frees). Advances the cursor. */
    int (*prefill)(void *engine_impl, const int32_t *tokens, size_t n_tokens,
                   float **logits, char *error, size_t error_size);

    /* Autoregressive decode: run one forward step for token_id at the current
     * position; returns logits [vocab] for the NEXT token (caller frees).
     * Advances the cursor by one. */
    int (*decode_step)(void *engine_impl, int32_t token_id, float **logits,
                       char *error, size_t error_size);

    /* Sample one token id from logits per cfg (temperature/top-p/stop-set).
     * Ownership: the caller owns logits and frees it in every case; the engine
     * never frees it. A stop token (cfg stop set or engine eos) is reported as
     * *token_id = -1 with success, not as an error. */
    int (*sample)(void *engine_impl, const float *logits,
                  const ColiSampleConfig *cfg, int32_t *token_id,
                  char *error, size_t error_size);

    /* KV-cache surgery. */
    int (*kv_get_len)(void *engine_impl, int *len, char *error, size_t error_size);
    int (*kv_rollback)(void *engine_impl, int target_len,
                       char *error, size_t error_size);

    void (*reserved_fn[6])(void);
} ColiEngineAdapter;

/* ---- registry ---------------------------------------------------------- */
int coli_engine_adapter_register(const ColiEngineAdapter *adapter);
const ColiEngineAdapter *coli_engine_adapter_lookup(const char *engine_id);
int coli_engine_adapter_count(void);

/* ---- convenience wrappers over a resolved adapter ---------------------- */
int coli_engine_open(const char *engine_id, const ColiEngineOptions *options,
                     ColiEngine **engine, char *error, size_t error_size);
void coli_engine_close(ColiEngine *engine);
int coli_engine_capabilities(const ColiEngine *engine,
                             ColiEngineCapabilities *capabilities,
                             char *error, size_t error_size);

int coli_engine_tokenize(ColiEngine *engine, const char *text, size_t text_bytes,
                         int32_t *token_ids, size_t token_capacity,
                         size_t *token_count, char *error, size_t error_size);
int coli_engine_detokenize(ColiEngine *engine, const int32_t *token_ids,
                           size_t token_count, char *text, size_t text_capacity,
                           size_t *text_bytes, char *error, size_t error_size);
int coli_engine_prefill(ColiEngine *engine, const int32_t *tokens, size_t n_tokens,
                        float **logits, char *error, size_t error_size);
int coli_engine_decode_step(ColiEngine *engine, int32_t token_id, float **logits,
                            char *error, size_t error_size);
int coli_engine_sample(ColiEngine *engine, const float *logits,
                       const ColiSampleConfig *cfg, int32_t *token_id,
                       char *error, size_t error_size);
int coli_engine_kv_get_len(ColiEngine *engine, int *len,
                           char *error, size_t error_size);
int coli_engine_kv_rollback(ColiEngine *engine, int target_len,
                            char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_ENGINE_H */
