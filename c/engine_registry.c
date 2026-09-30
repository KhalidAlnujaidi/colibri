/* engine_registry.c — registry + convenience wrappers for the unified engine
 * ABI (engine.h). Mirrors edge_runtime.c: a fixed-size table of registered
 * ColiEngineAdapter vtables, engine-id lookup, and thin caller-side wrappers
 * that own a ColiEngine handle (adapter + impl + capabilities).
 *
 * This file is engine-neutral: it knows nothing about a specific model family.
 * Model backends register themselves via coli_engine_adapter_register().
 */
#include "engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COLI_ENGINE_MAX_ADAPTERS 16

struct ColiEngine {
    const ColiEngineAdapter *adapter;
    void *impl;
    ColiEngineCapabilities capabilities;
};

static const ColiEngineAdapter *g_adapters[COLI_ENGINE_MAX_ADAPTERS];
static int g_adapter_count;

static int set_error(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return -1;
}

static int id_valid(const char *id) {
    if (!id || !*id) return 0;
    size_t length = strlen(id);
    if (length >= COLI_ENGINE_ENGINE_ID_CAP) return 0;
    for (size_t i = 0; i < length; i++) {
        unsigned char c = (unsigned char)id[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
              c == '_' || c == '-')) return 0;
    }
    return 1;
}

static int fixed_string_valid(const char *value, size_t capacity) {
    return value && value[0] && memchr(value, '\0', capacity) != NULL;
}

static int capabilities_valid(const ColiEngineCapabilities *capabilities,
                              const ColiEngineAdapter *adapter) {
    return capabilities->struct_size == sizeof(*capabilities) &&
           capabilities->abi_version == COLI_ENGINE_ABI_VERSION &&
           fixed_string_valid(capabilities->engine_id,
                              sizeof(capabilities->engine_id)) &&
           strcmp(capabilities->engine_id, adapter->engine_id) == 0 &&
           fixed_string_valid(capabilities->state_schema,
                              sizeof(capabilities->state_schema)) &&
           fixed_string_valid(capabilities->numeric_class,
                              sizeof(capabilities->numeric_class)) &&
           capabilities->vocab_size && capabilities->hidden_size &&
           capabilities->num_layers &&
           !!(capabilities->flags & COLI_ENGINE_CAP_PREFILL) ==
               !!adapter->prefill &&
           !!(capabilities->flags & COLI_ENGINE_CAP_DECODE) ==
               !!adapter->decode_step &&
           !!(capabilities->flags & COLI_ENGINE_CAP_KV_SURGERY) ==
               (!!adapter->kv_get_len && !!adapter->kv_rollback);
}

int coli_engine_adapter_register(const ColiEngineAdapter *adapter) {
    if (!adapter || adapter->struct_size < sizeof(*adapter) ||
        adapter->abi_version != COLI_ENGINE_ABI_VERSION ||
        !id_valid(adapter->engine_id) || !adapter->engine_open ||
        !adapter->engine_destroy || !adapter->tokenize || !adapter->detokenize ||
        !adapter->prefill || !adapter->decode_step || !adapter->sample ||
        (!!adapter->kv_get_len != !!adapter->kv_rollback))
        return -1;
    for (int i = 0; i < g_adapter_count; i++)
        if (strcmp(g_adapters[i]->engine_id, adapter->engine_id) == 0)
            return -1;
    if (g_adapter_count >= COLI_ENGINE_MAX_ADAPTERS) return -1;
    g_adapters[g_adapter_count++] = adapter;
    return 0;
}

const ColiEngineAdapter *coli_engine_adapter_lookup(const char *engine_id) {
    if (!engine_id) return NULL;
    for (int i = 0; i < g_adapter_count; i++)
        if (strcmp(g_adapters[i]->engine_id, engine_id) == 0)
            return g_adapters[i];
    return NULL;
}

int coli_engine_adapter_count(void) { return g_adapter_count; }

int coli_engine_open(const char *engine_id, const ColiEngineOptions *options,
                     ColiEngine **engine, char *error, size_t error_size) {
    if (!engine) return set_error(error, error_size, "missing engine output");
    *engine = NULL;
    if (!options || options->struct_size < sizeof(*options) ||
        !options->model_dir || !*options->model_dir)
        return set_error(error, error_size, "invalid engine options");
    const ColiEngineAdapter *adapter = coli_engine_adapter_lookup(engine_id);
    if (!adapter)
        return set_error(error, error_size, "engine is not registered");

    ColiEngine *created = calloc(1, sizeof(*created));
    if (!created)
        return set_error(error, error_size, "out of memory opening engine");
    created->adapter = adapter;
    created->capabilities.struct_size = sizeof(created->capabilities);
    created->capabilities.abi_version = COLI_ENGINE_ABI_VERSION;
    if (adapter->engine_open(&created->impl, &created->capabilities, options,
                             error, error_size) || !created->impl) {
        if (created->impl) adapter->engine_destroy(created->impl);
        free(created);
        return -1;
    }
    if (!capabilities_valid(&created->capabilities, adapter)) {
        adapter->engine_destroy(created->impl);
        free(created);
        return set_error(error, error_size,
                         "engine adapter returned incompatible capabilities");
    }
    *engine = created;
    return 0;
}

int coli_engine_capabilities(const ColiEngine *engine,
                             ColiEngineCapabilities *capabilities,
                             char *error, size_t error_size) {
    if (!engine || !capabilities ||
        capabilities->struct_size < sizeof(*capabilities))
        return set_error(error, error_size, "invalid capabilities buffer");
    size_t caller_size = capabilities->struct_size;
    memset(capabilities, 0, caller_size);
    memcpy(capabilities, &engine->capabilities, sizeof(*capabilities));
    return 0;
}

void coli_engine_close(ColiEngine *engine) {
    if (!engine) return;
    engine->adapter->engine_destroy(engine->impl);
    free(engine);
}

int coli_engine_tokenize(ColiEngine *engine, const char *text, size_t text_bytes,
                         int32_t *token_ids, size_t token_capacity,
                         size_t *token_count, char *error, size_t error_size) {
    if (!engine || !text || !token_count ||
        (!!token_ids != !!token_capacity))
        return set_error(error, error_size, "invalid tokenize request");
    return engine->adapter->tokenize(engine->impl, text, text_bytes,
                                     token_ids, token_capacity, token_count,
                                     error, error_size);
}

int coli_engine_detokenize(ColiEngine *engine, const int32_t *token_ids,
                           size_t token_count, char *text, size_t text_capacity,
                           size_t *text_bytes, char *error, size_t error_size) {
    if (!engine || !token_ids || !token_count || !text_bytes ||
        (!!text != !!text_capacity))
        return set_error(error, error_size, "invalid detokenize request");
    return engine->adapter->detokenize(engine->impl, token_ids, token_count,
                                       text, text_capacity, text_bytes,
                                       error, error_size);
}

int coli_engine_prefill(ColiEngine *engine, const int32_t *tokens, size_t n_tokens,
                        float **logits, char *error, size_t error_size) {
    if (!engine || !tokens || !n_tokens || !logits)
        return set_error(error, error_size, "invalid prefill request");
    return engine->adapter->prefill(engine->impl, tokens, n_tokens, logits,
                                    error, error_size);
}

int coli_engine_decode_step(ColiEngine *engine, int32_t token_id, float **logits,
                            char *error, size_t error_size) {
    if (!engine || !logits)
        return set_error(error, error_size, "invalid decode request");
    return engine->adapter->decode_step(engine->impl, token_id, logits,
                                        error, error_size);
}

int coli_engine_sample(ColiEngine *engine, const float *logits,
                       const ColiSampleConfig *cfg, int32_t *token_id,
                       char *error, size_t error_size) {
    if (!engine || !logits || !cfg || !token_id)
        return set_error(error, error_size, "invalid sample request");
    return engine->adapter->sample(engine->impl, logits, cfg, token_id,
                                   error, error_size);
}

int coli_engine_kv_get_len(ColiEngine *engine, int *len,
                           char *error, size_t error_size) {
    if (!engine || !len)
        return set_error(error, error_size, "invalid kv_get_len request");
    return engine->adapter->kv_get_len(engine->impl, len, error, error_size);
}

int coli_engine_kv_rollback(ColiEngine *engine, int target_len,
                            char *error, size_t error_size) {
    if (!engine || target_len < 0)
        return set_error(error, error_size, "invalid kv_rollback request");
    return engine->adapter->kv_rollback(engine->impl, target_len,
                                        error, error_size);
}
