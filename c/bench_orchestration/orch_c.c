/*
 * orch_c.c — the colibri ORCHESTRATION layer (stages 1–3) written in C,
 * plus a benchmark driver.
 *
 *   Stage 1  spawn the engine (fork/exec + pipes)
 *   Stage 2  render the chat prompt + write a SUBMIT frame
 *   Stage 3  read N DATA frames + DONE, stream the bytes out
 *
 * This measures ONLY the orchestration overhead: the engine side is the
 * identical stub_engine for every language, so inference cost is excluded.
 *
 * Usage: orch_c <stub_engine> <iters> <tokens_per_req>
 *
 * Build: clang -O2 -o orch_c orch_c.c
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* timing helper                                                       */
/* ------------------------------------------------------------------ */
static double now_us(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

/* ------------------------------------------------------------------ */
/* Stage 1 — spawn the engine process (fork/exec + pipes)              */
/* ------------------------------------------------------------------ */
typedef struct {
    pid_t   pid;
    int     wfd;    /* write to engine (stdin)  */
    FILE   *rf;     /* read  from engine (stdout) */
} Engine;

static Engine engine_spawn(const char *stub_path){
    int to_eng[2], from_eng[2];
    pipe(to_eng);
    pipe(from_eng);

    pid_t pid = fork();
    if (pid == 0){
        /* child: wire the pipes to stdin/stdout and exec the engine */
        dup2(to_eng[0], STDIN_FILENO);
        dup2(from_eng[1], STDOUT_FILENO);
        close(to_eng[0]);  close(to_eng[1]);
        close(from_eng[0]); close(from_eng[1]);
        execl(stub_path, stub_path, (char*)NULL);
        _exit(127); /* only if exec fails */
    }

    close(to_eng[0]);
    close(from_eng[1]);

    Engine e;
    e.pid = pid;
    e.wfd = to_eng[1];
    e.rf  = fdopen(from_eng[0], "r");
    if (!e.rf){ close(to_eng[1]); close(from_eng[0]); }
    return e;
}

static void engine_wait_ready(Engine *e){
    char line[256];
    /* skip any pre-READY preamble lines */
    do {
        if (!fgets(line, sizeof line, e->rf)) return;
    } while (strncmp(line, "READY", 5) != 0);
}

static void engine_stop(Engine *e){
    close(e->wfd);
    fclose(e->rf);
    waitpid(e->pid, NULL, 0);
}

/* ------------------------------------------------------------------ */
/* Stage 2 — render the prompt and write a SUBMIT frame                */
/* ------------------------------------------------------------------ */
static void render_prompt(const char *user_text, char *out, size_t cap){
    snprintf(out, cap, "<|user|>\n%s\n<|end|>\n<|assistant|>", user_text);
}

static void write_submit(Engine *e, unsigned long long id, const char *prompt,
                         int max_tokens, float temp, float top_p){
    char header[128];
    int plen = (int)strlen(prompt);
    int hn = snprintf(header, sizeof header,
                      "SUBMIT %llu 0 %d %d %g %g\n",
                      id, plen, max_tokens, temp, top_p);
    (void)write(e->wfd, header, (size_t)hn);
    (void)write(e->wfd, prompt, (size_t)plen);
    (void)write(e->wfd, "\n", 1);
}

/* ------------------------------------------------------------------ */
/* Stage 3 — read DATA frames + DONE, stream bytes to stdout           */
/* ------------------------------------------------------------------ */
/* Sink for streamed token bytes during the benchmark. In production the
 * gateway would emit these as SSE; here we consume them into /dev/null so the
 * 5000-iteration flood doesn't drown the results table. Opening the sink once
 * and reusing it keeps the timed work identical to the Python run. */
static FILE *g_sink = NULL;

static void read_turn(Engine *e, unsigned long long id, int max_tokens){
    (void)id; /* request id is carried in-band in the frames; not needed here */
    char kind[8];
    unsigned long long rid;
    int n;

    if (!g_sink) g_sink = fopen("/dev/null", "wb");

    for (int i = 0; i < max_tokens; i++){
        if (fscanf(e->rf, "%7s %llu %d\n", kind, &rid, &n) != 3) return;
        if (strcmp(kind, "DATA") != 0) return; /* DONE/ERROR/stop */
        char buf[65536];
        if (n > (int)sizeof buf) n = (int)sizeof buf;
        fread(buf, 1, (size_t)n, e->rf);
        getc(e->rf); /* trailing '\n' */
        /* stream the token out (like SSE emission in the real gateway) */
        fwrite(buf, 1, (size_t)n, g_sink);
    }
    /* consume the DONE line */
    char done[256];
    fgets(done, sizeof done, e->rf);
}

/* ------------------------------------------------------------------ */
/* one full orchestrated request — the thing we time                   */
/* ------------------------------------------------------------------ */
static double do_request(Engine *e, unsigned long long id, const char *user_text,
                         int max_tokens){
    double t0 = now_us();

    char prompt[128];
    render_prompt(user_text, prompt, sizeof prompt);
    write_submit(e, id, prompt, max_tokens, 0.7f, 1.0f);
    read_turn(e, id, max_tokens);

    return now_us() - t0;
}

/* ------------------------------------------------------------------ */
/* benchmark driver                                                    */
/* ------------------------------------------------------------------ */
static int cmp_ll(const void *a, const void *b){
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv){
    if (argc < 4){
        fprintf(stderr, "usage: %s <stub_engine> <iters> <tokens_per_req>\n", argv[0]);
        return 2;
    }
    const char *stub  = argv[1];
    int         iters = atoi(argv[2]);
    int         toks  = atoi(argv[3]);
    if (iters < 10) iters = 10;
    if (toks  < 1)  toks  = 1;

    /* one engine process, reused across all iterations (like a persistent
     * server) — this is exactly the "service" regime of `coli serve`. */
    Engine eng = engine_spawn(stub);
    engine_wait_ready(&eng);

    double *samples = (double*)malloc((size_t)iters * sizeof(double));

    /* warm-up (let scheduler/caches settle) */
    do_request(&eng, 0, "hi", toks);

    for (int i = 0; i < iters; i++){
        samples[i] = do_request(&eng, (unsigned long long)(i + 1), "hi", toks);
    }

    engine_stop(&eng);

    /* stats */
    qsort(samples, (size_t)iters, sizeof(double), cmp_ll);
    double sum = 0, min = samples[0], max = samples[iters - 1];
    for (int i = 0; i < iters; i++) sum += samples[i];
    double mean = sum / iters;
    double med  = samples[iters / 2];

    printf("C orchestration   iters=%d tokens/req=%d\n", iters, toks);
    printf("  mean  %8.1f us/req\n", mean);
    printf("  median %8.1f us/req\n", med);
    printf("  min   %8.1f us/req\n", min);
    printf("  max   %8.1f us/req\n", max);
    printf("  req/s %8.1f\n", 1e6 / med);

    free(samples);
    return 0;
}
