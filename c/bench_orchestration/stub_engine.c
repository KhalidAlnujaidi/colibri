/*
 * stub_engine.c — a protocol stub that behaves like a colibri engine
 * for the purpose of benchmarking the ORCHESTRATION layer only.
 *
 * It implements the same wire protocol as docs/serve_protocol.md:
 *   - prints "READY" on startup
 *   - reads "SUBMIT <id> <slot> <bytes> <max_tokens> <temp> <top_p>"
 *   - replies with <max_tokens> "DATA <id> <n>\n<n bytes>\n" frames
 *   - then a "DONE <id> STAT ..." line
 *
 * Because this stub is identical C for every orchestrator we benchmark,
 * any measured difference comes from the orchestration code (framing,
 * pipe I/O, parsing), NOT from inference — which is exactly the point.
 *
 * Build: clang -O2 -o stub_engine stub_engine.c
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv){
    /* Max tokens per request is configurable via argv[1] (default 16). */
    int max_tokens_default = 16;
    if (argc > 1) max_tokens_default = atoi(argv[1]);
    if (max_tokens_default < 1) max_tokens_default = 1;

    /* Unbuffered so every frame reaches the pipe immediately. */
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("READY\n");
    fflush(stdout);

    static const char tok[] = "tok"; /* pretend token bytes, 3 bytes */

    char line[1024];
    while (fgets(line, sizeof line, stdin)){
        char kind[16];
        unsigned long long id;
        int slot, bytes, max_tokens;
        float temp, top_p;
        int n = sscanf(line, "%15s %llu %d %d %d %f %f",
                       kind, &id, &slot, &bytes, &max_tokens, &temp, &top_p);
        if (n < 5 || strcmp(kind, "SUBMIT") != 0) continue;

        /* Consume the byte-counted payload plus its trailing newline. */
        char *payload = (char*)malloc((size_t)bytes + 1);
        if (!payload) continue;
        size_t got = fread(payload, 1, (size_t)bytes, stdin);
        payload[got] = 0;
        getc(stdin); /* trailing '\n' */

        if (max_tokens < 1) max_tokens = 1;

        /* Emit one DATA frame per generated token, then DONE. */
        for (int i = 0; i < max_tokens; i++){
            printf("DATA %llu 3\n%s\n", id, tok);
        }
        printf("DONE %llu STAT %d 0.0 0.0 0.0 %d 0\n", id, max_tokens, max_tokens);
        fflush(stdout);

        free(payload);
    }
    return 0;
}
