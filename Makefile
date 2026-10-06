# bend-ml: setup and verification shortcuts. Run `make setup` once, then `make check`.
export BEND_NO_TELEMETRY := 1
export PATH := $(HOME)/.bend/bin:$(HOME)/.elan/bin:$(PATH)
PY := reference/.venv/bin/python

.PHONY: setup setup-lite check check-full laws bench media clean

setup:            ## Bend, Lean, Python venv, MNIST and GPT-2 data
	scripts/setup.sh

setup-lite:       ## same, without the 550 MB GPT-2 download
	scripts/setup.sh --no-gpt2

check:            ## proofs, kernel, shape errors, PyTorch/tiktoken comparisons (about 30 s)
	$(PY) reference/check_all.py

check-full:       ## everything above plus GPT-2 and MNIST (about 2 min)
	$(PY) reference/check_all.py --full

laws:             ## print every law statement of every package
	$(PY) reference/list_laws.py

bench:            ## matrix · vector and compile-time benchmarks, saved in bench/results/ (about 30 min; use an idle machine)
	$(PY) bench/mv_bands.py > bench/results/mv_bands-$$(date +%F).txt
	$(PY) scripts/compile_times.py > bench/results/compile-times-$$(date +%F).md

media:            ## regenerate the README images and the presentation video (needs Brave, node, ffmpeg)
	scripts/make_media.sh

clean:
	rm -rf reference/__pycache__ .scratch
