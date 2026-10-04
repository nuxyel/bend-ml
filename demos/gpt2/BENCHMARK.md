# GPT-2 small: Bend vs PyTorch

**Summary (v2):** GPT-2 small with 124 M parameters runs in Bend and **generates exactly the same tokens as PyTorch**, with logits equal to within 2e-4. In v1 it took ~3 s per token (~150x PyTorch); in v2 (matrix · vector products over `Array`) it takes **~0.1 s per token**: ~5x PyTorch with 16 threads (21 ms per forward pass of 11 tokens) and ~2x PyTorch with 1 thread (55 ms). What still weighs is loading the weights (9 s against ~1 s), because the 124 M numbers become trees of nodes (~1.5 GB of resident memory at the peak).

## v2: `demos/gpt2/fast.bend` (package `bend-ml-tensor-array@0.1.1.0`)

| GPT-2 small, "The capital of France is", 8 tokens | per token | total |
|---|---|---|
| Bend v1 (lists) | ~3 s | 49.8 s |
| Bend v2 with `par = 3` (a copy of the matrix per task) | ~1.1 s | 22 s |
| **Bend v2, sequential (`par = 0`)** | **~0.1 s** | **11.1 s** (9 s of loading + 1.2 s for the 8 tokens) |
| PyTorch (CPU, 16 threads, no KV cache): forward pass of 11 tokens | **21 ms** | 1.3 s in `gpt2_ref.py` (includes ~1 s to load the weights) |
| PyTorch (CPU, 1 thread, no KV cache): forward pass of 11 tokens | 55 ms | |

The same verification as v1 (`reference/test_gpt2.py`): 3 prompts, 22 tokens, identical ids, |Δlogit| ≤ 2e-4.

In matrix · vector products, `par = 3` is ~10x **worse** than sequential: the cost of cloning the weight matrix per task exceeds the cost of computing (each weight is read once). Isolated measurement in `NOTES.md`, experiment 9.

---

## v1: lists (`demos/gpt2/gpt2.bend`), kept as the baseline

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
