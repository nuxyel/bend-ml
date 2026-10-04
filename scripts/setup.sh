#!/usr/bin/env bash
# One-shot, idempotent setup for the bend-ml checks. No sudo.
#
#   scripts/setup.sh              # Bend 2.0.35, Lean 4.34.0, Python venv, MNIST and GPT-2 data
#   scripts/setup.sh --no-gpt2    # skip the 550 MB GPT-2 download and its preparation
#
# Pins: Bend 2.0.35 (checked by SHA256) and Lean v4.34.0 (the kernel behind `bend --verdict`).
set -euo pipefail

BEND_VERSION=2.0.35
LEAN_VERSION=4.34.0
GPT2=1
[ "${1:-}" = "--no-gpt2" ] && GPT2=0

cd "$(dirname "$0")/.."
ROOT=$PWD
export BEND_NO_TELEMETRY=1
export PATH="$HOME/.bend/bin:$HOME/.elan/bin:$PATH"

say() { printf '\n== %s\n' "$*"; }
have_bend() { command -v bend >/dev/null 2>&1 && [ "$(bend version 2>/dev/null | awk '{print $NF}')" = "$BEND_VERSION" ]; }
have_lean() { command -v lean >/dev/null 2>&1 && lean --version 2>/dev/null | grep -q "version $LEAN_VERSION"; }

# --- 1. Bend, pinned and verified -------------------------------------------------------
say "Bend $BEND_VERSION"
if have_bend; then
  echo "already installed"
else
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  OS=linux; ARCH=x64;   SHA=63039d1a119f716767ac5a7d8fe0717cfacf219c6c253c35192148e0dade722f ;;
    Linux-aarch64) OS=linux; ARCH=arm64; SHA=09b813073241628f590f2c2fe420299ec25e4dddd6cf3fdc49c9486339989564 ;;
    *) echo "unsupported platform $(uname -s)-$(uname -m): Bend runs on Linux (use WSL on Windows)"; exit 1 ;;
  esac
  HOME_BEND=${BEND_HOME:-$HOME/.bend}
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  NAME="bend-$BEND_VERSION-$OS-$ARCH.tar.gz"
  curl --proto '=https' --tlsv1.2 -fsSL -o "$TMP/$NAME" "https://github.com/bendlang/bend/releases/download/v$BEND_VERSION/$NAME"
  echo "$SHA  $TMP/$NAME" | sha256sum -c - >/dev/null || { echo "SHA256 mismatch for $NAME"; exit 1; }
  tar -xzf "$TMP/$NAME" -C "$TMP" 2>/dev/null
  mkdir -p "$HOME_BEND/bin"
  rm -rf "$HOME_BEND/bend2" "$HOME_BEND/guide"
  mv "$TMP/bend/bend2" "$TMP/bend/guide" "$HOME_BEND/"
  mv -f "$TMP/bend/bin/bend" "$HOME_BEND/bin/bend"
fi
have_bend || { echo "ERROR: bend $BEND_VERSION not found after install"; exit 1; }
bend version

# --- 2. Lean (kernel for --verdict) -----------------------------------------------------
say "Lean $LEAN_VERSION"
if have_lean; then
  echo "already installed"
else
  curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --no-modify-path --default-toolchain "leanprover/lean4:v$LEAN_VERSION"
fi
have_lean || { echo "ERROR: lean $LEAN_VERSION not found after install"; exit 1; }
lean --version

# --- 3. Python environment --------------------------------------------------------------
say "Python environment"
[ -d reference/.venv ] || python3 -m venv reference/.venv
reference/.venv/bin/pip install -q --upgrade pip
reference/.venv/bin/pip install -q -r reference/requirements.txt
reference/.venv/bin/python -c "import torch, numpy, tiktoken, safetensors, regex; print('torch', torch.__version__, '| numpy', numpy.__version__, '| tiktoken', tiktoken.__version__)"

# --- 4. MNIST ---------------------------------------------------------------------------
say "MNIST data"
mkdir -p demos/mnist/data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  [ -f "demos/mnist/data/$f" ] || curl -sSfL "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz" | gunzip > "demos/mnist/data/$f"
done
[ -f demos/mnist/data/init/w1.txt ] || reference/.venv/bin/python reference/mnist_torch.py --make-init --epochs 1 --max-batches 1 >/dev/null
echo ok

# --- 5. GPT-2 ---------------------------------------------------------------------------
if [ "$GPT2" = 1 ]; then
  say "GPT-2 data (about 550 MB)"
  mkdir -p demos/gpt2/data
  for f in model.safetensors vocab.json merges.txt; do
    [ -f "demos/gpt2/data/$f" ] || curl -sSfL -o "demos/gpt2/data/$f" "https://huggingface.co/openai-community/gpt2/resolve/main/$f"
  done
  [ -f demos/gpt2/data/merges_num.txt ] || reference/.venv/bin/python reference/gpt2_prep.py
  [ -d demos/gpt2/data/w ] || reference/.venv/bin/python reference/gpt2_prep.py --split
  echo ok
fi

say "setup finished: run 'make check' (or 'make check-full')"
