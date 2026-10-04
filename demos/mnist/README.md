# MNIST in Bend: a 784-128-10 MLP

A two-layer network trained with plain SGD. **Every matrix product, every gradient and every weight update has its shape checked by the type**: if `dW` had the wrong dimension, the program would not compile.

There are two implementations, with identical results:

- `fast.bend` (v2): written with [`bend-ml-tensor-array`](../../tensor-array) (flat `Array`, parallel products). **6.6 s per epoch.**
- `train.bend` (v1): matrices as lists, kept as the baseline. 544 s per epoch.

## Run (from the repository root)

```bash
# 1. data and initial weights (once)
reference/.venv/bin/python reference/mnist_torch.py --make-init --epochs 1 --max-batches 1
# (download the 4 MNIST files into demos/mnist/data/; see run.sh)

# 2. training in Bend:  <epochs> <max batches (0 = all)> <lr>
bend demos/mnist/fast.bend -o /tmp/mnist_fast
/tmp/mnist_fast 1 0 0.1

# 3. the PyTorch twin (same initial weights, same batches, same lr)
reference/.venv/bin/python reference/mnist_torch.py --epochs 1 --lr 0.1
```

There is a shortcut that does it all: `demos/mnist/run.sh`.

## Correctness check

With the same initial weights, the same batch order and the same `lr`, the two implementations must produce the same losses and hits. After 50 steps:

| | mean training loss | test hits |
|---|---|---|
| Bend | 1.6616004 | 7829 / 10000 |
| PyTorch | 1.661600 | 7829 / 10000 |

The full results for one and three epochs, with timings, are in `BENCHMARK.md`.
