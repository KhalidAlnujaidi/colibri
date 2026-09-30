/*
 * orch_cpp.cpp — the colibri ORCHESTRATION layer (stages 1–3) written in
 * C++, plus a benchmark driver. Same protocol and same workload as orch_c,
 * so the two differ only in implementation language.
 *
 *   Stage 1  spawn the engine (fork/exec + pipes, wrapped in a class)
 *   Stage 2  render the chat prompt + write a SUBMIT frame (std::string)
 *   Stage 3  read N DATA frames + DONE, stream the bytes out
 *
 * Usage: orch_cpp <stub_engine> <iters> <tokens_per_req>
 *
 * Build: clang++ -O2 -std=c++17 -o orch_cpp orch_cpp.cpp
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

/* ------------------------------------------------------------------ */
/* timing helper                                                       */
/* ------------------------------------------------------------------ */
static double now_us(){
    auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::micro>(t).count();
}

/* ------------------------------------------------------------------ */
/* Stage 1 — Engine class: spawn + manage the child process            */
/* ------------------------------------------------------------------ */
class Engine {
public:
    Engine(const char *stub_path){
        int to_eng[2], from_eng[2];
        pipe(to_eng); pipe(from_eng);

        pid_ = fork();
        if (pid_ == 0){
            dup2(to_eng[0], STDIN_FILENO);
            dup2(from_eng[1], STDOUT_FILENO);
            close(to_eng[0]); close(to_eng[1]);
            close(from_eng[0]); close(from_eng[1]);
            execl(stub_path, stub_path, (char*)nullptr);
            _exit(127);
        }

        close(to_eng[0]); close(from_eng[1]);
        wfd_ = to_eng[1];
        rf_  = fdopen(from_eng[0], "r");
    }

    ~Engine(){ shutdown(); }

    void wait_ready(){
        char line[256];
        do {
            if (!fgets(line, sizeof line, rf_)) return;
        } while (strncmp(line, "READY", 5) != 0);
    }

    /* Stage 2 — render + write the SUBMIT frame */
    void submit(unsigned long long id, const std::string &prompt,
                int max_tokens, float temp, float top_p){
        std::string header = "SUBMIT " + std::to_string(id) + " 0 " +
                             std::to_string(prompt.size()) + " " +
                             std::to_string(max_tokens) + " " +
                             std::to_string(temp) + " " +
                             std::to_string(top_p) + "\n";
        write(wfd_, header.data(), header.size());
        write(wfd_, prompt.data(), prompt.size());
        write(wfd_, "\n", 1);
    }

    /* Stage 3 — read DATA frames + DONE */
    void read_turn(unsigned long long id, int max_tokens){
        (void)id; /* request id is carried in-band in the frames; not needed here */
        char kind[8];
        unsigned long long rid;
        int n;
        static FILE *sink = nullptr;
        if (!sink) sink = fopen("/dev/null", "wb");
        for (int i = 0; i < max_tokens; i++){
            if (fscanf(rf_, "%7s %llu %d\n", kind, &rid, &n) != 3) return;
            if (std::strcmp(kind, "DATA") != 0) return;
            char buf[65536];
            if (n > (int)sizeof buf) n = (int)sizeof buf;
            fread(buf, 1, (size_t)n, rf_);
            getc(rf_);
            fwrite(buf, 1, (size_t)n, sink); /* stream out to /dev/null */
        }
        char done[256];
        fgets(done, sizeof done, rf_);
    }

    void shutdown(){
        if (wfd_ >= 0){ close(wfd_); wfd_ = -1; }
        if (rf_){ fclose(rf_); rf_ = nullptr; }
        if (pid_ > 0){ waitpid(pid_, nullptr, 0); pid_ = -1; }
    }

private:
    pid_t pid_ = -1;
    int   wfd_ = -1;
    FILE *rf_  = nullptr;
};

/* ------------------------------------------------------------------ */
/* one full orchestrated request — the thing we time                   */
/* ------------------------------------------------------------------ */
static double do_request(Engine &eng, unsigned long long id,
                         const std::string &user_text, int max_tokens){
    double t0 = now_us();

    /* Stage 2: render */
    std::string prompt = "<|user|>\n" + user_text + "\n<|end|>\n<|assistant|>";
    eng.submit(id, prompt, max_tokens, 0.7f, 1.0f);
    eng.read_turn(id, max_tokens);

    return now_us() - t0;
}

/* ------------------------------------------------------------------ */
/* driver                                                              */
/* ------------------------------------------------------------------ */
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

    Engine eng(stub);
    eng.wait_ready();

    std::vector<double> samples;
    samples.reserve((size_t)iters);

    /* warm-up */
    do_request(eng, 0, "hi", toks);

    for (int i = 0; i < iters; i++)
        samples.push_back(do_request(eng, (unsigned long long)(i + 1), "hi", toks));

    std::sort(samples.begin(), samples.end());

    double sum = 0;
    for (double s : samples) sum += s;
    double mean = sum / samples.size();
    double med  = samples[samples.size() / 2];
    double min  = samples.front();
    double max  = samples.back();

    printf("C++ orchestration  iters=%d tokens/req=%d\n", iters, toks);
    printf("  mean   %8.1f us/req\n", mean);
    printf("  median %8.1f us/req\n", med);
    printf("  min    %8.1f us/req\n", min);
    printf("  max    %8.1f us/req\n", max);
    printf("  req/s  %8.1f\n", 1e6 / med);

    return 0;
}
