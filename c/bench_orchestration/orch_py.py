#!/usr/bin/env python3
"""
orch_py.py — the colibri ORCHESTRATION layer (stages 1-3) as it currently
exists in Python, plus a benchmark driver.

   Stage 1  subprocess.Popen the engine (like c/coli)
   Stage 2  render_chat + write a SUBMIT frame (like c/openai_server.py)
   Stage 3  read N DATA frames + DONE, stream bytes (like read_engine_turn)

This measures ONLY the orchestration overhead: the engine side is the same
stub_engine binary for every language, so inference cost is excluded.

Usage: python3 orch_py.py <stub_engine> <iters> <tokens_per_req>
"""
import subprocess
import sys
import time
import statistics

# Sink for streamed token bytes during the benchmark. In production the
# gateway would emit these as SSE; here we consume them (like the C/C++
# drivers writing to /dev/null) so the 5000-iteration flood doesn't drown
# the results table.
_SINK = open("/dev/null", "wb")


def _stream(body: bytes):
    _SINK.write(body)
    _SINK.flush()


def render_chat(user_text: str) -> str:
    """Stage 2 (prompt rendering) — mirrors render_chat_* in openai_server.py."""
    return f"<|user|>\n{user_text}\n<|end|>\n<|assistant|>"


class Engine:
    """Stage 1 + frame I/O — mirrors the Engine class in openai_server.py."""

    def __init__(self, stub_path: str):
        self.p = subprocess.Popen(
            [stub_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        # wait for the READY handshake
        line = self.p.stdout.readline()
        if not line.startswith(b"READY"):
            raise RuntimeError("engine did not reach READY")

    def generate(self, req_id: int, user_text: str, max_tokens: int,
                 temperature: float = 0.7, top_p: float = 1.0):
        """Stage 2 + Stage 3 — one full orchestrated request."""
        # Stage 2: render + write the SUBMIT frame (openai_server.py:3719,3743)
        prompt = render_chat(user_text)
        payload = prompt.encode("utf-8")
        header = (f"SUBMIT {req_id} 0 {len(payload)} {max_tokens} "
                  f"{temperature} {top_p}\n").encode("utf-8")
        self.p.stdin.write(header)
        self.p.stdin.write(payload)
        self.p.stdin.write(b"\n")
        self.p.stdin.flush()

        # Stage 3: read DATA frames + DONE (openai_server.py read_engine_turn)
        for _ in range(max_tokens):
            line = self.p.stdout.readline()          # "DATA <id> <n>\n"
            fields = line.decode("utf-8", "replace").strip().split()
            if not fields or fields[0] != "DATA":
                break
            size = int(fields[2])
            body = self.p.stdout.read(size)          # exactly n bytes
            self.p.stdout.read(1)                    # trailing '\n'
            _stream(body)                            # stream out to sink
        self.p.stdout.readline()                     # DONE line

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.kill()


def bench(stub: str, iters: int, toks: int):
    eng = Engine(stub)

    def request(req_id: int) -> float:
        t0 = time.perf_counter()
        eng.generate(req_id, "hi", toks)
        return (time.perf_counter() - t0) * 1e6  # microseconds

    # warm-up
    request(0)

    samples = [request(i + 1) for i in range(iters)]
    eng.close()

    mean = statistics.mean(samples)
    med = statistics.median(samples)
    lo = min(samples)
    hi = max(samples)

    print(f"Python orchestration iters={iters} tokens/req={toks}")
    print(f"  mean   {mean:8.1f} us/req")
    print(f"  median {med:8.1f} us/req")
    print(f"  min    {lo:8.1f} us/req")
    print(f"  max    {hi:8.1f} us/req")
    print(f"  req/s  {1e6 / med:8.1f}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <stub_engine> <iters> <tokens_per_req>")
        sys.exit(2)
    bench(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
