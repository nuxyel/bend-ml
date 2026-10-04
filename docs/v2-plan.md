# v2 plan: measure the performance ceiling and deepen what only Bend has

## Why

v1.0 showed Bend ~1800x slower than PyTorch on MNIST. But the comparison was unfair to the language:
matrices as linked lists, one effective thread, no GPU. v2 has two axes, in this order:

1. **Performance:** find the real ceiling of each technique and the final distance to PyTorch (without promising to reach it).
2. **Guarantees:** use whatever energy is left to deepen what PyTorch does not have: invariants in the type, more proved laws.

Renan's decisions: both axes, performance first; try CUDA 12 **without sudo** first; no fixed numeric goal (measure and reduce).

## Axis 1: performance (each step records its number in `NOTES.md`)

| # | Experiment | What it measures | Exit criterion |
|---|---|---|---|
| 1 | Reproducible baseline: a single micro-benchmark of a 100×784 · 784×128 `matmul` and of one MNIST step | time and threads | `bench/` scripts with saved results |
| 2 | `Array<F32>` instead of lists (index `i*c + j`, `Array.get/set`) | the gain from indexable memory | number compared with lists |
| 3 | Why parallelism stalls at ~1.8x: reproduce with a minimal case, vary the granularity and the shape of the split, read `bend guide shaders` and `paper/BendRT.pdf` | bottleneck (allocator, reference counting, unbalanced split) | cause identified, or a minimal reproducible report for Bend's GitHub |
| 4 | `matmul` tiling and an allocation-free inner loop | locality gain | number |
| 5 | GPU: CUDA 12 without sudo (toolkit in `~/.local/cuda12`, a `/usr/local/cuda` link only if possible); test `pow2!` and then `matmul` with `!` | whether the RTX 4050 runs; GPU gain | it ran, or the limit is documented |
| 6 | Rebuild the MNIST step and GPT-2 with the best technique | final time vs PyTorch | before/after table in `BENCHMARK.md` |

Rule: a discarded experiment also goes into the notes (with its number). No optimization that breaks the equivalence with PyTorch (the `check_all.py` tests remain valid).

## Axis 2: guarantees (only after axis 1)

- **Invariant in the type:** `Mat<r,c>` with the list of rows tied to `r` and `c` (for example an indexed type `Rows(r, c)`), removing the "library" invariant from the README.
- **More laws:** `transpose(transpose(m)) = m`, `matmul` associative over the structure, a correct `Mat.of` (only returns `Some` if the sizes match), `softmax` with a structural sum.
- **Tokenizer:** a law that `train` produces well-formed tables (`wf = True`), closing the gap in the `bpe` README.
- Every new law: a clean `--verdict`, published in a new package version (four-number versions).

## Deliverables

- `NOTES.md` with every measurement and discarded hypothesis.
- `demos/*/BENCHMARK.md` updated (before/after, hardware, versions).
- New packages as `0.2.x.0`; tag `v2.0.0` and a release.
- If the cause of the parallelism stall or another limitation belongs to Bend: a minimal report for `github.com/bendlang/bend/issues` (as ready text; the agent does not open issues on your behalf).

## Stop criterion for v2

1. Every axis-1 technique measured and recorded, with the final distance to PyTorch.
2. At least one new axis-2 guarantee proved (`--verdict`) and published.
3. `check_all.py --full` green; benchmarks and README updated; `v2.0.0` published.

## Result (2026-10-04)

Axis 1 (performance), every experiment measured and recorded in `NOTES.md` (experiments 1 to 9):

| Experiment | Result |
|---|---|
| 1-2. lists vs `Array` | ~49x on one thread (44 M vs 2,200 M multiply-adds/s) |
| 3. why parallelism stalls at ~1.8x | the bottleneck was the data structure and the `Array` size (2^20 vs 2^17 costs ~60%); clones do **not** cause contention; the cache/tiling hypothesis was refuted |
| 4. tiling | no effect (1, 8, 16, 32 blocks of columns: same time) |
| 5. GPU | **not done**: Bend asks for CUDA 12 and the machine has Arch's CUDA 13. The Bend binary honors `CUDA_HOME`, so a user-local CUDA 12 toolkit (headers and `libnvrtc`) could be tried without sudo; it is left as a pending item |
| 6. full MNIST | 544 s → 6.6 s per epoch, same loss and hits |
| 7. GPT-2 | ~3 s → ~0.1 s per token, same ids and logits |
| Finding | parallelizing matrix · vector by copying the matrix costs 10x more than computing; future solution: split the `Array` tree without copying |

Axis 2 (guarantees): `train_wf` and `roundtrip_trained` proved and published in `bend-ml-bpe-tokenizer@0.1.1.0`; the whole MNIST training step is type-checked (`tensor-array`).

Not done: the `capacity >= r*c` invariant in the `Mat` type (it stays documented as a limit).
