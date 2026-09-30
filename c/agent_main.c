/*
 * agent_main.c — the Unified C/C++ Harness (orchestration executive) for the
 * colibri engine ABI.
 *
 * This links a model backend (a translation unit compiled with
 * COLI_ENGINE_ADAPTER that calls coli_glm_engine_adapter_register()) and drives
 * it exclusively through engine.h — no pipes, no SUBMIT/DATA/DONE, no
 * snprintf/fscanf serialization. It demonstrates the two concrete wins the
 * shared-harness refactor was designed for:
 *
 *   1. In-memory prefill + autoregressive decode (zero IPC latency).
 *   2. O(1) KV-cache surgery for tool-call branching: snapshot kv_get_len
 *      BEFORE the speculative branch, decode it, then roll back and inject the
 *      tool-result tokens — without recomputing the shared prefix and without a
 *      new pipe payload.
 *
 * Usage:
 *   agent_main <engine_id> <model_dir> [tool_demo|plain]
 *
 *   engine_id : which registered adapter to use (e.g. "glm").
 *   model_dir : directory containing the shard files + tokenizer.json.
 *   tool_demo : run the KV-rollback tool-call branching demo (default).
 *   plain     : prefill + greedy decode, no rollback.
 *
 * Build (from c/):
 *   cc -O2 -DCOLI_ENGINE_ADAPTER -DCOLIBRI_NO_MAIN -c colibri.c -o libcolibri_engine.o
 *   cc -O2 -c engine_registry.c -o engine_registry.o
 *   cc -O2 agent_main.c engine_registry.o libcolibri_engine.o -lm -o agent_main
 */
#include "engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CTX 4096

static ColiSampleConfig greedy_config(void) {
    ColiSampleConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.temperature = 0.0f;   /* greedy, deterministic */
    cfg.top_p = 0.0f;
    cfg.n_stop_tokens = 0;
    cfg.ban = -1;
    return cfg;
}

/* Sample + decode loop from an already-computed logits vector. The caller's
 * logits are consumed (freed) by this function in every path. */
static int decode_loop(ColiEngine *eng, float *logits, int max_len,
                       const ColiSampleConfig *cfg,
                       int32_t *out, int *n_out, char *error, size_t error_size) {
    int n = 0;
    int32_t next = 0;
    for (int i = 0; i < max_len; i++) {
        if (coli_engine_sample(eng, logits, cfg, &next, error, error_size)) {
            free(logits);
            return -1;
        }
        free(logits); logits = NULL;
        if (next < 0) break;               /* stop token sampled */
        out[n++] = next;
        if (coli_engine_decode_step(eng, next, &logits, error, error_size)) {
            free(logits);
            return -1;
        }
    }
    free(logits);
    *n_out = n;
    return 0;
}

/* Model-agnostic one-shot: tokenize -> prefill -> decode. */
static int generate(ColiEngine *eng, const char *text, int max_len,
                    int32_t *out, int *n_out, char *error, size_t error_size) {
    int32_t prompt[512];
    size_t np = 0;
    if (coli_engine_tokenize(eng, text, strlen(text), prompt,
                             sizeof(prompt) / sizeof(prompt[0]), &np,
                             error, error_size))
        return -1;
    float *logits = NULL;
    if (coli_engine_prefill(eng, prompt, np, &logits, error, error_size))
        return -1;
    ColiSampleConfig cfg = greedy_config();
    return decode_loop(eng, logits, max_len, &cfg, out, n_out, error, error_size);
}

/* Plain prefill + greedy decode: the simplest end-to-end ABI use. */
static int demo_plain(ColiEngine *eng, const char *user_text,
                      char *error, size_t error_size) {
    int32_t out[64];
    int n = 0;
    if (generate(eng, user_text, 24, out, &n, error, error_size))
        return -1;

    int kv = 0;
    coli_engine_kv_get_len(eng, &kv, error, error_size);
    printf("  decoded tokens  : %d  (kv now %d)\n", n, kv);

    char text[1024];
    size_t tb = 0;
    if (coli_engine_detokenize(eng, out, (size_t)n, text, sizeof(text), &tb,
                               error, error_size))
        return -1;
    text[tb < sizeof(text) ? tb : sizeof(text) - 1] = 0;
    printf("  detokenized     : %.*s\n", (int)tb, text);
    return 0;
}

/* Tool-call branching demo.
 *
 *   1. prefill the user prompt            (kv = P)
 *   2. SNAPSHOT kv_get_len                (checkpoint = P)
 *   3. decode a speculative tool call     (kv = P + D)
 *   4. branch rejected -> kv_rollback(P)  (O(1): cursor rewind, no recompute)
 *   5. inject the tool result as a prefill(kv = P + R)
 *   6. continue decoding
 *
 * The point: step 4 costs nothing, and step 5 reuses the P-token prefix
 * instead of re-prefilling it. */
static int demo_tool_call(ColiEngine *eng, const char *user_text,
                          const char *tool_result,
                          char *error, size_t error_size) {
    int32_t prompt[512];
    size_t np = 0;
    if (coli_engine_tokenize(eng, user_text, strlen(user_text), prompt,
                             sizeof(prompt) / sizeof(prompt[0]), &np,
                             error, error_size))
        return -1;

    /* 1. Prefill the user prompt. */
    float *logits = NULL;
    if (coli_engine_prefill(eng, prompt, np, &logits, error, error_size))
        return -1;

    /* 2. Snapshot BEFORE the speculative branch. */
    int checkpoint = 0;
    if (coli_engine_kv_get_len(eng, &checkpoint, error, error_size))
        return -1;

    /* 3. Decode a candidate tool invocation. */
    ColiSampleConfig cfg = greedy_config();
    int32_t draft[64];
    int nd = 0;
    if (decode_loop(eng, logits, 48, &cfg, draft, &nd, error, error_size))
        return -1;
    logits = NULL;

    char draft_text[1024];
    size_t dt = 0;
    if (coli_engine_detokenize(eng, draft, (size_t)nd, draft_text,
                               sizeof(draft_text), &dt, error, error_size))
        return -1;
    draft_text[dt < sizeof(draft_text) ? dt : sizeof(draft_text) - 1] = 0;
    printf("  prompt tokens   : %d\n", (int)np);
    printf("  draft tool call : %.*s\n", (int)dt, draft_text);
    printf("  kv at snapshot  : %d  (after %d draft tokens)\n",
           checkpoint, nd);

    /* 4. Branch rejected -> O(1) rollback to before the draft. */
    if (coli_engine_kv_rollback(eng, checkpoint, error, error_size))
        return -1;

    /* 5. Inject the tool result at the rolled-back position. */
    int32_t result_tokens[128];
    size_t nrt = 0;
    if (coli_engine_tokenize(eng, tool_result, strlen(tool_result),
                             result_tokens, sizeof(result_tokens) / sizeof(result_tokens[0]),
                             &nrt, error, error_size))
        return -1;
    if (coli_engine_prefill(eng, result_tokens, nrt, &logits,
                            error, error_size))
        return -1;

    /* 6. Continue decoding from the injected tool result. */
    int32_t final[64];
    int nf = 0;
    if (decode_loop(eng, logits, 48, &cfg, final, &nf, error, error_size))
        return -1;

    int kv_now = 0;
    coli_engine_kv_get_len(eng, &kv_now, error, error_size);
    printf("  rollback        : kv rewound %d -> %d (discarded %d draft tokens)\n",
           checkpoint + nd, checkpoint, nd);
    printf("  after inject    : kv = %d  (prefix %d reused + %d tool tokens + %d decoded)\n",
           kv_now, checkpoint, (int)nrt, nf);

    char final_text[1024];
    size_t ft = 0;
    if (coli_engine_detokenize(eng, final, (size_t)nf, final_text,
                               sizeof(final_text), &ft, error, error_size))
        return -1;
    final_text[ft < sizeof(final_text) ? ft : sizeof(final_text) - 1] = 0;
    printf("  final continuation: %.*s\n", (int)ft, final_text);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <engine_id> <model_dir> [tool_demo|plain]\n",
                argv[0]);
        return 2;
    }
    const char *engine_id = argv[1];
    const char *model_dir = argv[2];
    int tool_demo = (argc < 4) || (strcmp(argv[3], "tool_demo") == 0);

    char error[256] = {0};

    /* Backends self-register at load through a constructor in the backend TU
     * (glm_engine_adapter_ctor / olmoe_engine_adapter_ctor). A build links
     * exactly one backend, so there is no cross-backend symbol reference here:
     * the harness stays backend-agnostic and links against either. */
    if (coli_engine_adapter_count() == 0) {
        fprintf(stderr, "no engine backend registered in this build\n");
        return 1;
    }

    ColiEngineOptions options;
    memset(&options, 0, sizeof(options));
    options.struct_size = sizeof(options);
    options.model_dir = model_dir;

    ColiEngine *eng = NULL;
    if (coli_engine_open(engine_id, &options, &eng, error, sizeof(error))) {
        fprintf(stderr, "engine open failed: %s\n", error);
        return 1;
    }

    ColiEngineCapabilities caps;
    memset(&caps, 0, sizeof(caps));
    caps.struct_size = sizeof(caps);
    coli_engine_capabilities(eng, &caps, error, sizeof(error));
    printf("engine %s: vocab=%u hidden=%u layers=%u ctx=%u kv_surgery=%s\n",
           caps.engine_id, caps.vocab_size, caps.hidden_size, caps.num_layers,
           caps.max_context_tokens,
           (caps.flags & COLI_ENGINE_CAP_KV_SURGERY) ? "yes" : "no");

    int rc = tool_demo
        ? demo_tool_call(eng,
                         "The user asks: what is 2+2? Use a calculator.",
                         "\n<tool_result>4</tool_result>\nSo the answer is 4.",
                         error, sizeof(error))
        : demo_plain(eng, "The user asks: what is 2+2?", error, sizeof(error));

    coli_engine_close(eng);
    return rc;
}
