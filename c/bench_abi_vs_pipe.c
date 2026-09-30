/*
 * bench_abi_vs_pipe.c — the measurement that motivated the unified engine ABI.
 *
 * Same engine, same model, same workload, two transports:
 *
 *   pipe  — spawn the engine binary in serve mode and drive it with the
 *           SUBMIT / DATA / DONE frame protocol over OS pipes (what the Python
 *           gateway does today).
 *   abi   — link the engine as a library and call
 *           engine_prefill / engine_decode_step / engine_sample in-process.
 *
 * Both arms do EXACTLY the same work per request: re-prefill the whole prompt
 * from position 0, then greedily decode N tokens. Neither arm reuses a KV
 * prefix, so the only difference left is the transport itself: fork/exec pipes,
 * frame serialization, snprintf/fscanf, and line-buffer flushes.
 *
 * Usage:
 *   bench_abi_vs_pipe abi  <model_dir> <iters> <tokens>
 *   bench_abi_vs_pipe pipe <engine_bin> <model_dir> <iters> <tokens> [engine argv...]
 *
 * Build (from c/):
 *   cc -O2 bench_abi_vs_pipe.c engine_registry.o libolmoe_engine.o -o bench_olmoe -lm
 *   cc -O2 bench_abi_vs_pipe.c engine_registry.o libcolibri_engine.o -o bench_glm -lm
 */
#define _POSIX_C_SOURCE 200809L
#include "engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>

#define BENCH_PROMPT "The user asks: what is 2+2?"
#define MAX_FRAME_LINE 4096

/* Engine diagnostics ([prefill] layer N/M, [API] KV slot ..., [OMP], [stop] ...)
 * all go to stderr and would drown the results. Quiet them once, and keep a
 * saved fd so this program's own messages still get through. */
static int g_msgfd = 2;

static void quiet_engine(void) {
    int saved = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
        dup2(devnull, STDERR_FILENO);
        close(devnull);
    }
    if (saved >= 0) g_msgfd = saved;
}

static void msg(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > 0) {
        ssize_t w = write(g_msgfd, buf, (size_t)n);
        (void)w;
    }
}

/* One-line loading bar, redrawn in place. */
static void progress(const char *arm, int done, int total) {
    int pct = total > 0 ? (100 * done / total) : 100;
    int filled = pct / 5;
    char bar[21];
    for (int i = 0; i < 20; i++) bar[i] = (i < filled) ? '#' : '.';
    bar[20] = 0;
    msg("\r  %-5s [%s] %3d%%  %d/%d   ", arm, bar, pct, done, total);
    if (done >= total) msg("\n");
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void report(const char *arm, int iters, int toks, double *samples) {
    qsort(samples, (size_t)iters, sizeof(double), cmp_double);
    double sum = 0;
    for (int i = 0; i < iters; i++) sum += samples[i];
    double mean = sum / iters;
    double med = samples[iters / 2];
    printf("%-5s  iters=%-5d tokens/req=%-3d  mean %9.1f us  median %9.1f us  "
           "min %9.1f  max %9.1f  req/s %9.1f\n",
           arm, iters, toks, mean, med, samples[0], samples[iters - 1], 1e6 / med);
}

/* ------------------------------------------------------------------ */
/* pipe arm: drive the engine binary over SUBMIT / DATA / DONE         */
/* ------------------------------------------------------------------ */
static pid_t spawn_engine(const char *engine_bin, const char *model_dir,
                          char *const extra[], int n_extra,
                          int *wfd_out, FILE **rf_out) {
    int to_eng[2], from_eng[2];
    if (pipe(to_eng) || pipe(from_eng)) return -1;

    pid_t pid = fork();
    if (pid == 0) {
        dup2(to_eng[0], STDIN_FILENO);
        dup2(from_eng[1], STDOUT_FILENO);
        close(to_eng[0]); close(to_eng[1]);
        close(from_eng[0]); close(from_eng[1]);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        /* BENCH_CHILD_LOG=<path>: keep the engine's own diagnostics instead of
         * discarding them, so a run can be audited (e.g. the "[API] KV slot ...
         * prefill N" lines that say how much the serve path actually prefilled). */
        const char *childlog = getenv("BENCH_CHILD_LOG");
        if (childlog && *childlog) {
            int lf = open(childlog, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (lf >= 0) { dup2(lf, STDERR_FILENO); close(lf); }
        }
        setenv("SERVE", "1", 1);
        setenv("SERVE_BATCH", "1", 1);   /* GLM: select the mux (SUBMIT/DATA/DONE) loop */
        setenv("SNAP", model_dir, 1);
        setenv("CTX", "4096", 1);
        /* The serve path persists the KV cache to disk every turn (kv_disk_append
         * in mux_submit + mux_done). That is real work, but it is serve-path
         * bookkeeping, not transport, and the ABI arm does none of it — leaving
         * it on would charge the pipe arm for something unrelated and add huge
         * variance (measured max 420 ms on the tiny fixture). */
        setenv("KVSAVE", "0", 1);
        char *argv[64];
        int n = 0;
        argv[n++] = (char *)engine_bin;
        for (int i = 0; i < n_extra && n < 62; i++) argv[n++] = extra[i];
        argv[n] = NULL;
        execv(engine_bin, argv);
        _exit(127);
    }
    close(to_eng[0]);
    close(from_eng[1]);
    *wfd_out = to_eng[1];
    *rf_out = fdopen(from_eng[0], "r");
    return pid;
}

/* Read one DATA frame body; returns 0 on DONE, 1 on DATA, -1 on error. */
static int read_frame(FILE *rf, unsigned long long want_id) {
    char line[MAX_FRAME_LINE];
    for (;;) {
        if (!fgets(line, sizeof line, rf)) return -1;
        if (!strncmp(line, "DATA ", 5)) {
            unsigned long long id; int n;
            if (sscanf(line, "DATA %llu %d", &id, &n) == 2 && n >= 0) {
                if (n > 0) {
                    char buf[65536];
                    if (n > (int)sizeof buf) n = (int)sizeof buf;
                    if (fread(buf, 1, (size_t)n, rf) != (size_t)n) return -1;
                }
                int c = getc(rf);           /* trailing '\n' */
                (void)c;
                if (id == want_id) return 1;
                continue;
            }
        }
        if (!strncmp(line, "DONE ", 5)) return 0;
        if (!strncmp(line, "ERROR", 5)) return -1;
        /* anything else (READY/STAT/PROF/HITS/EMAP/TIERS/HWINFO) is ignored */
    }
}

static int pipe_arm(const char *engine_bin, const char *model_dir,
                    int iters, int toks, char *const extra[], int n_extra) {
    int wfd = -1; FILE *rf = NULL;
    pid_t pid = spawn_engine(engine_bin, model_dir, extra, n_extra, &wfd, &rf);
    if (pid < 0 || !rf) { msg( "cannot spawn engine\n"); return 1; }

    /* READY handshake: drain lines until READY appears. */
    char line[MAX_FRAME_LINE];
    for (;;) {
        if (!fgets(line, sizeof line, rf)) { msg( "engine never reached READY\n"); return 1; }
        if (strstr(line, "READY")) break;
    }

    double *samples = malloc((size_t)iters * sizeof(double));

    for (int i = 0; i < iters; i++) {
        unsigned long long id = (unsigned long long)(i + 1);
        /* Unique prompt per turn (varying marker FIRST): no KV prefix can ever
         * match, so the engine must do a genuine full prefill every turn —
         * exactly the work the ABI arm does. Without this the serve path
         * reuses the prefix and prefills 0 tokens after turn 1, which makes the
         * two arms incomparable. */
        char prompt[256];
        int pn = snprintf(prompt, sizeof prompt, "[%d] %s", i, BENCH_PROMPT);
        char header[160];
        int hn = snprintf(header, sizeof header, "SUBMIT %llu 0 %d %d 0 1\n",
                          id, pn, toks);
        double t0 = now_us();
        if (write(wfd, header, (size_t)hn) < 0) return 1;
        if (write(wfd, prompt, (size_t)pn) < 0) return 1;
        if (write(wfd, "\n", 1) < 0) return 1;
        for (int k = 0; k < toks; k++)
            if (read_frame(rf, id) != 1) { msg( "frame error at %d/%d\n", k, toks); return 1; }
        /* The turn is not complete until DONE arrives: count that wait too. */
        if (read_frame(rf, id) != 0) { msg( "missing DONE\n"); return 1; }
        samples[i] = now_us() - t0;
        if ((i + 1) % 20 == 0 || i + 1 == iters) progress("pipe", i + 1, iters);
    }

    report("pipe", iters, toks, samples);

    close(wfd);
    fclose(rf);
    waitpid(pid, NULL, 0);
    free(samples);
    return 0;
}

/* ------------------------------------------------------------------ */
/* ABI arm: same workload, in-process function calls                   */
/* ------------------------------------------------------------------ */
static int abi_arm(const char *model_dir, int iters, int toks, const char *engine_id) {
    char error[256] = {0};
    if (coli_engine_adapter_count() == 0) {
        msg( "no engine backend registered in this build\n");
        return 1;
    }
    ColiEngineOptions options;
    memset(&options, 0, sizeof(options));
    options.struct_size = sizeof(options);
    options.model_dir = model_dir;

    ColiEngine *eng = NULL;
    if (coli_engine_open(engine_id, &options, &eng, error, sizeof(error))) {
        msg( "engine open failed: %s\n", error);
        return 1;
    }

    ColiSampleConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.temperature = 0.0f;   /* greedy, same as the pipe arm's temp=0 */
    cfg.top_p = 0.0f;
    cfg.ban = -1;

    /* Pre-tokenize every turn's prompt OUTSIDE the timed region: the pipe arm
     * sends text and the engine tokenizes it, so charging tokenization to the
     * ABI arm would be an unfair penalty. Same unique prompts as the pipe arm. */
    int32_t **prompts = malloc((size_t)iters * sizeof(int32_t *));
    size_t *lens = malloc((size_t)iters * sizeof(size_t));
    for (int i = 0; i < iters; i++) {
        char buf[256];
        int bn = snprintf(buf, sizeof buf, "[%d] %s", i, BENCH_PROMPT);
        prompts[i] = malloc(512 * sizeof(int32_t));
        size_t np = 0;
        if (coli_engine_tokenize(eng, buf, (size_t)bn, prompts[i], 512, &np,
                                 error, sizeof(error))) {
            msg( "tokenize failed: %s\n", error);
            return 1;
        }
        lens[i] = np;
    }

    double *samples = malloc((size_t)iters * sizeof(double));

    for (int i = 0; i < iters; i++) {
        float *logits = NULL;
        double t0 = now_us();
        if (coli_engine_kv_rollback(eng, 0, error, sizeof(error))) return 1;
        if (coli_engine_prefill(eng, prompts[i], lens[i], &logits, error, sizeof(error))) return 1;
        for (int k = 0; k < toks; k++) {
            int32_t tok = 0;
            if (coli_engine_sample(eng, logits, &cfg, &tok, error, sizeof(error))) return 1;
            free(logits); logits = NULL;
            if (k + 1 < toks) {
                if (coli_engine_decode_step(eng, tok, &logits, error, sizeof(error))) return 1;
            }
        }
        free(logits);
        samples[i] = now_us() - t0;
        if ((i + 1) % 20 == 0 || i + 1 == iters) progress("abi", i + 1, iters);
    }

    report("abi", iters, toks, samples);
    for (int i = 0; i < iters; i++) free(prompts[i]);
    free(prompts); free(lens);
    coli_engine_close(eng);
    free(samples);
    return 0;
}

int main(int argc, char **argv) {
    quiet_engine();   /* engine diagnostics -> /dev/null; this program's -> saved fd */
    if (argc < 5) {
        msg(
                "usage: %s abi  <model_dir> <iters> <tokens>\n"
                "       %s pipe <engine_bin> <model_dir> <iters> <tokens> [engine argv...]\n",
                argv[0], argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], "abi")) {
        const char *engine_id = getenv("COLI_ENGINE_ID");
        return abi_arm(argv[2], atoi(argv[3]), atoi(argv[4]),
                       engine_id ? engine_id : "olmoe");
    }
    if (!strcmp(argv[1], "pipe")) {
        if (argc < 6) { msg( "pipe needs <engine_bin>\n"); return 2; }
        return pipe_arm(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]),
                        &argv[6], argc - 6);
    }
    msg( "unknown arm: %s\n", argv[1]);
    return 2;
}
