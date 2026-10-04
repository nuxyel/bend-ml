#!/usr/bin/env bash
# Runs the real commands shown in the README and the video and stores their output as text in
# docs/media/outputs/. Requires `make setup`. Usage: scripts/capture_outputs.sh [--skip-slow]
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.elan/bin:$HOME/.bend/bin:$PATH" BEND_NO_TELEMETRY=1
O=docs/media/outputs; mkdir -p "$O"
PY=reference/.venv/bin/python
# strip the line numbers and ANSI codes that are not part of the message
clean() { sed 's/\x1b\[[0-9;]*m//g'; }

bend tensor/tests/bad_matmul.bend 2>&1 | clean | head -8 > "$O/bad_matmul.txt"
bend tensor/tests/bad_reshape.bend 2>&1 | clean | head -8 > "$O/bad_reshape.txt"
bend bpe/main.bend --verdict 2>&1 | clean > "$O/verdict.txt"
$PY reference/list_laws.py 2>&1 | grep -A6 "^law roundtrip$" > "$O/law_roundtrip.txt"
$PY reference/list_laws.py 2>&1 | tail -1 > "$O/law_count.txt"

[ "${1:-}" = "--skip-slow" ] && exit 0
bend demos/gpt2/fast.bend -o /tmp/bendml_gpt2 >/dev/null 2>&1
/tmp/bendml_gpt2 "The capital of France is" 8 2>&1 | clean > "$O/gpt2.txt"
bend demos/mnist/fast.bend -o /tmp/bendml_mnist >/dev/null 2>&1
/tmp/bendml_mnist 1 0 0.1 2>&1 | clean | grep "epoch" > "$O/mnist.txt"
make check-full 2>&1 | clean | tail -40 > "$O/check_full.txt"
