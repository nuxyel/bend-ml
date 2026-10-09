# MNIST: an honest benchmark, Bend vs PyTorch

Versions follow [`docs/VERSIONING.md`](../../docs/VERSIONING.md) (renumbered on 2026-10-09: v0.2 was called v1, v0.3 v2, v1.0 v2.1, v1.1 v3.0, v1.2 v3.1).

**Summary (v0.3):** the results are **identical** (same loss, same accuracy after 1 and after 3 epochs). In v0.2 Bend took **544 s per epoch** (~1800x PyTorch); in v0.3, with matrices in a flat `Array` and parallel products, it takes **6.6 s** (~33x PyTorch with 16 threads, ~22x with 1). The difference between v0.2 and v0.3 is the data structure, not the language; what is left is scalar code against BLAS/SIMD. Re-measured on 2026-10-04 on an idle machine: 7.0, 7.0 and 7.2 s per epoch for Bend (the 6.6 s of the tables below are the original runs) and 0.18 to 0.20 s for PyTorch, i.e. ~35x.

## v0.3: flat `Array` (`demos/mnist/fast.bend`, package `bend-ml-tensor-array@0.1.1.0`)

| | 1 epoch | loss / hits after 1, 2, 3 epochs |
|---|---|---|
| **Bend v0.3, 16 threads** | **6.6 s** | 0.5204771 / 9129 · 0.27043572 / 9298 · 0.2156194 / 9418 |
| Bend v0.3, 1 thread | 18.1 s | |
| PyTorch, 16 threads | 0.19 to 0.21 s | 0.520477 / 9129 · 0.270433 / 9298 · 0.215613 / 9418 |
| PyTorch, 1 thread | 0.30 s | |

Every product, every gradient and every update has its shape checked by the type (a `dW` requested with swapped dimensions does not compile: `docs/shape-error-array-bad_grad.txt`). How we got here, step by step (every number is in `NOTES.md`, experiments 1 to 9):

| step | epoch |
|---|---|
| v0.2: matrices as linked lists | 544.3 s |
| flat `Array`, `Array.get/set` by index, 1 thread | 11.5 s |
| + large products in 2^3 parallel blocks (16 threads) | 6.3 s |
| the package's typed API (list↔Array conversions in the wrappers) | 6.6 s |

---

## v0.2: lists (`demos/mnist/train.bend`), kept as the baseline

## Setup

| | |
|---|---|
| Model | MLP 784 → 128 → 10, ReLU, plain SGD, `lr = 0.1`, batches of 100, 600 batches per epoch, no shuffling |
| Data | MNIST (60,000 train, 10,000 test), pixels `/255`, no other normalization |
| Initial weights | the same text files in both implementations (`demos/mnist/data/init/`), `U(-1/√n, 1/√n)` |
| CPU | Intel Core Ultra 7 155H (16 cores, 22 threads), 32 GB of RAM |
| OS | Linux 7.2.5 (Omarchy) |
| Bend | 2.0.35, native binary (`bend ... -o`), clang 22.1.8; 1 thread (the list-based `matmul` only reached ~1.8x with more threads) |
| PyTorch | 2.14.1 (CPU), Python 3.14.7, 16 threads or 1 thread |
| GPU | tried: faster on compute-bound loops but slower on these memory-bound kernels (NOTES.md, experiment 10), so not used |

## Result of one full epoch

| | mean training loss | test hits | epoch time |
|---|---|---|---|
| **Bend** | 0.52047706 | 9129 / 10000 | **544.3 s** |
| PyTorch (16 threads) | 0.520477 | 9129 / 10000 | 0.21 s |
| PyTorch (1 thread) | 0.520477 | 9128 / 10000 | 0.30 s |

- The loss matches to 6 places; the 1-hit difference in the 1-thread PyTorch comes from the summation order in `F32`.
- They also matched after 50 steps (`1.6616005` and `7829` hits, in both).
- **Where Bend loses:** ~0.9 s per batch against ~0.5 ms. Each step does ~20 million multiply-adds; Bend did ~24 million per second effectively on one thread (PyTorch uses vectorized BLAS). No optimization was attempted beyond pre-transposing operands and not computing `dx` in the first layer.

## What Bend offers that the comparison does not show

- Every matrix product and every gradient has its dimension checked by the type (`bad_matmul.bend` does not compile).
- The tokenizer and the `reshape`/autodiff laws have proofs verified by the kernel.

## Reproduce

```bash
demos/mnist/run.sh 1 0 0.1     # 1 full epoch in both (Bend takes ~7 s)
demos/mnist/run.sh 1 50 0.1    # only 50 batches (fast; checks the equivalence)
```
