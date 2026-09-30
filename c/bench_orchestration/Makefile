# Makefile — build & benchmark the orchestration layers (C / C++ / Python)
# against the identical stub_engine protocol stub.
#
#   make build   -> compile stub_engine, orch_c, orch_cpp
#   make bench   -> build, then run all three at scaled iters for tokens =
#                   1, 64, 640, 6400, 64000
#   make clean   -> remove build artifacts
#
# iters scale DOWN as tokens/req grows so total emitted frames stays ~constant
# (BASE_ITERS=5000 -> 5000/5000/500/50/5 iters for 1/64/640/6400/64000).
#
# Requires: clang / clang++ (or gcc / g++) and python3.

CC   ?= clang
CXX  ?= clang++
PY   ?= python3

CFLAGS   = -O2 -Wall -Wextra
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

BASE_ITERS ?= 5000

BINS  = stub_engine orch_c orch_cpp
TOKS  = 1 64 640 6400 64000

.PHONY: all build bench clean

all: build

build: $(BINS)

stub_engine: stub_engine.c
	$(CC) $(CFLAGS) -o $@ $<

orch_c: orch_c.c
	$(CC) $(CFLAGS) -o $@ $<

orch_cpp: orch_cpp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# ---- benchmark: run all three for each token count --------------------
bench: build
	@echo "================================================================"
	@echo " colibri orchestration-layer benchmark (scaled)"
	@echo " engine = stub_engine (identical C for every orchestrator)"
	@echo " base iters = $(BASE_ITERS)"
	@echo " tokens = $(TOKS)"
	@echo "================================================================"
	@run_case() { \
		t="$$1"; it="$$2"; \
		echo; echo "############ tokens/req = $$t (iters=$$it) ############"; \
		echo "--- C    ---"; ./orch_c    ./stub_engine "$$it" "$$t"; \
		echo "--- C++  ---"; ./orch_cpp  ./stub_engine "$$it" "$$t"; \
		echo "--- Python ---"; $(PY) orch_py.py ./stub_engine "$$it" "$$t"; \
	}; \
	run_case 1     $(BASE_ITERS); \
	run_case 64    $(BASE_ITERS); \
	run_case 640   $$(( BASE_ITERS / 10 )); \
	run_case 6400  $$(( BASE_ITERS / 100 )); \
	run_case 64000 $$(( BASE_ITERS / 1000 ))

clean:
	rm -f $(BINS)
