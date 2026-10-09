# GPT-2 small: Bend vs PyTorch

Versions follow [`docs/VERSIONING.md`](../../docs/VERSIONING.md) (renumbered on 2026-10-09: v0.2 was called v1, v0.3 v2, v1.0 v2.1, v1.1 v3.0, v1.2 v3.1).

**Summary (v0.3):** GPT-2 small with 124 M parameters runs in Bend and **generates exactly the same tokens as PyTorch**, with logits equal to within 6e-4 over 11 prompts (2e-4 on the first 3). In v0.2 it took ~3 s per token (~150x PyTorch); in v0.3 (matrix · vector products over `Array`) it takes **~0.1 s per token**: ~5x PyTorch with 16 threads (21 ms per forward pass of 11 tokens) and ~2x PyTorch with 1 thread (55 ms). What still weighs is loading the weights (9 s against ~1 s), because the 124 M numbers become trees of nodes (~1.5 GB of resident memory at the peak).

## v1.2.1: one band computes like `Mat.matmul_nt` (package `bend-ml-tensor-array@0.1.6.0`)

Each band of a matrix · vector product now runs the `gemm` of `Mat.matmul_nt`, writing its rows into an `Array`
and reading them back, instead of building the result list row by row (`lcols`). The numbers are the same bit
for bit; the ids and logits match PyTorch on the 11 prompts. I found the cause by swapping one v1.0 piece at a
time into the v1.2 demo (`NOTES.md`, exp. 14): the loader, the copy of `x` and the row lookup changed nothing;
the product kernel was the whole gap.

All four versions measured in one session on an idle machine, "The capital of France is" + 32 tokens (36
forward passes), median of 5 alternated runs; one thread pinned to a performance core
(`bench/results/gpt2-versions-2026-10-09.txt`):

| | 36 passes, 1 thread | 36 passes, 16 threads | load, 16 threads | peak resident |
|---|---|---|---|---|
| v1.0 | **5.17 s** | 5.20 s | 6.8 s | 1462 MB |
| v1.1 | 5.78 s (+12%) | 3.99 s | 5.6 s | 965 MB |
| v1.2 | 5.51 s (+7%) | 3.82 s | 4.3 s | 932 MB |
| **v1.2.1** | 5.28 s (+2%) | **3.69 s** | **4.3 s** | **932 MB** |

On one thread v1.2.1 is still ~2% behind v1.0; I have not found where that last part goes. Per product on 16
threads with 2^4 bands (`bench/results/mv_bands-2026-10-09.txt`), the list API is 19-32% faster than in
0.1.5.0; the logits product goes from 5.6 ms to 4.0 ms.

## v1.2: faster loading, no one-thread cost (package `bend-ml-tensor-array@0.1.5.0`)

- The weight files are decoded straight into the arrays (no list of `F32` in between), and each file's size
  is checked before it is read.
- With 2 threads or fewer the matrices stay in one band, which runs exactly v1.0's kernel; with more, 2^4
  bands (2^5 for the embedding), and the products use `Bands.matvec_l` (no `Mat` conversions).

"The capital of France is" + 32 tokens (36 forward passes), alternated 3 times each, background load
(`bench/results/gpt2-v31-2026-10-06.txt`):

| | load | 36 forward passes, 1 thread | 36 forward passes, 16 threads |
|---|---|---|---|
| v1.0 | 6-7 s | 4.7-5.0 s | 4.7 s |
| v1.1 | 5.1-6.1 s | 5.2 s | 3.8-4.2 s |
| **v1.2** | **3.9-5.1 s** | **4.8 s** | **3.6-3.8 s** |

Re-measured on an idle machine on 2026-10-08 (nothing else running, on AC power), alternated 5 times
(`bench/results/gpt2-idle-2026-10-08.txt`):

| | load | 36 forward passes, 1 thread | 36 forward passes, 16 threads |
|---|---|---|---|
| v1.0 | 5-7 s | 4.9-5.3 s | 4.8-5.2 s |
| **v1.2** | **3.8-5.4 s** | **5.2-5.5 s** | **3.7-4.3 s** |

On one thread v1.2 is 3-6% slower than v1.0 in this run (the product kernel of the bands, fixed in v1.2.1). Per product, with the matrix already built (`bench/results/mv_bands-2026-10-08.txt`,
16 threads, list API): 2304 × 768 from 0.78 ms to 0.29 ms (2^5 bands); 50257 × 768 from 31.8 ms to 5.5 ms. These
match the numbers taken under load on 2026-10-06 within a few percent.

Over 36 positions the attention over the key/value cache (still lists) takes a growing share of each
token, so the end-to-end gain is smaller than the gain on the matrix · vector products. The same ids and
logits as v1.0 on the 11 prompts.

## v1.1: weights in `Bands`, products in parallel (package `bend-ml-tensor-array@0.1.4.0`)

The five weight matrices of each product (qkv, projection, the two MLP matrices, and the 50257 × 768
embedding used for the logits) are kept in 2^4 row bands; each matrix · vector product runs its bands in
parallel and copies no weights (`NOTES.md`, exp. 11). The ids and the logits are the same as v0.3's, digit for
digit, on the 11 prompts.

The machine had background load during these runs, so v1.0 and v1.1 were run alternately, 5 times each
(`bench/results/gpt2-v21-vs-v3-2026-10-05.txt`); "The capital of France is" + 8 tokens, 13 forward passes:

| | v1.0 (sequential) | v1.1, 16 threads | v1.1, 1 thread |
|---|---|---|---|
| 13 forward passes (median) | 1.7 s | 1.4 s | 1.9 s |
| per token | ~0.13 s | ~0.11 s | ~0.15 s |

In a quieter moment the same comparison gave 1.2 s against 0.7 s (~0.054 s per token, ~2.6x the 21 ms of
16-thread PyTorch). On one thread v1.1 is ~12% slower than v1.0 (the typed result is converted to a `Mat` and
back to a list, and the bands' lists are appended at every node). Re-measured on an idle machine in the v1.2 section.

Per product, with the matrix already built (`bench/mv_bands.py`, 16 threads): 2304 × 768 from 0.80 ms to
0.32 ms; 50257 × 768 (the logits) from 25.1 ms to 6.1 ms.

## v0.3: `demos/gpt2/fast.bend` (package `bend-ml-tensor-array@0.1.1.0`)

| GPT-2 small, "The capital of France is", 8 tokens | per token | total |
|---|---|---|
| Bend v0.2 (lists) | ~3 s | 49.8 s |
| Bend v0.3 with `par = 3` (a copy of the matrix per task) | ~1.1 s | 22 s |
| **Bend v0.3, sequential (`par = 0`)** | **~0.1 s** | **11.1 s** (9 s of loading + 1.2 s for the 8 tokens) |
| PyTorch (CPU, 16 threads, no KV cache): forward pass of 11 tokens | **21 ms** | 1.3 s in `gpt2_ref.py` (includes ~1 s to load the weights) |
| PyTorch (CPU, 1 thread, no KV cache): forward pass of 11 tokens | 55 ms | |

The same verification as v0.2 (`reference/test_gpt2.py`): 3 prompts, 22 tokens, identical ids, |Δlogit| ≤ 2e-4.

In matrix · vector products, `par = 3` is ~10x **worse** than sequential: the cost of cloning the weight matrix per task exceeds the cost of computing (each weight is read once). Isolated measurement in `NOTES.md`, experiment 9.

---

## v0.2: lists (`demos/gpt2/gpt2.bend`), kept as the baseline

## Verification (`reference/test_gpt2.py`)

Greedy generation from three prompts, comparing ids, text and the logit of the chosen token at each step (22 tokens):

| prompt | ids | text | max \|Δ logit\| |
|---|---|---|---|
| `The capital of France is` (8 tokens) | identical | identical | 8e-5 |
| `Machine learning is` (8 tokens) | identical | identical | 2e-4 |
| `1, 2, 3, 4,` (6 tokens) | identical | identical | 1.1e-4 |

Example: `The capital of France is the capital of the French Republic, and` (in both implementations).

The tokenizer in Bend (pre-tokenizer + 50,000 rules, package `bend-ml-bpe-tokenizer`) matches `tiktoken` on 16 out of 16 texts (`reference/test_gpt2_tok.py`).

## Performance

| | time |
|---|---|
| Bend: loading 124 M weights | ~10 s |
| Bend: each generated token | ~3 s (one position with a KV cache; 12 layers + 50,257 logits) |
| Bend: 8 tokens, total | 49.8 s |
| PyTorch (CPU, no KV cache, 16 threads): 8 tokens, total including loading the weights | 1.3 s |

Setup: Intel Core Ultra 7 155H (22 threads), 32 GB; Bend 2.0.35 (1 effective thread), clang 22.1.8; PyTorch 2.14.1 (CPU). The GPU was tried and is slower than the CPU for these memory-bound kernels (NOTES.md, experiment 10), so it is not used.

## Memory

~2 GB for the weights in lists of `F32` (16 bytes per number).
