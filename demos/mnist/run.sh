#!/usr/bin/env bash
# Treina o MLP em Bend e em PyTorch com os mesmos dados, pesos iniciais e hiperparâmetros.
# Uso (da raiz do repositório): demos/mnist/run.sh [épocas] [máx. lotes; 0 = todos] [lr]
set -euo pipefail
EP=${1:-1}; MB=${2:-0}; LR=${3:-0.1}
export PATH="$HOME/.bend/bin:$PATH" BEND_NO_TELEMETRY=1
PY=reference/.venv/bin/python
D=demos/mnist/data
mkdir -p "$D"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  [ -f "$D/$f" ] || { curl -sSfL "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz" | gunzip > "$D/$f"; }
done
[ -f "$D/init/w1.txt" ] || $PY reference/mnist_torch.py --make-init --epochs 1 --max-batches 1 >/dev/null
echo "== Bend"; bend demos/mnist/train.bend -o /tmp/mnist_train >/dev/null; /tmp/mnist_train "$EP" "$MB" "$LR"
echo "== PyTorch"; $PY reference/mnist_torch.py --epochs "$EP" --max-batches "$MB" --lr "$LR"
