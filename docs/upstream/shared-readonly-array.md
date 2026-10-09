# Draft issue for bendlang/bend: a safe read-only borrow of an `Array` across a parallel let

Status: posted as https://github.com/bendlang/bend/issues/1376 (2026-10-06), still open on 2026-10-08. Checked for duplicates on 2026-10-06: #885 asked for a
bang to borrow an `Array` and was answered with `Array.fork` / `Array.join`, which are `@unsafe` (still on
main 0ad47fc).

---

**Title:** A read-only borrow of an `Array` for a parallel let, without `@unsafe`, so libraries with laws can share weights across tasks

## Summary

In matrix · vector (the inner loop of LLM inference), every weight is read once. To run it on
several cores, each task needs to read part of the weight matrix. Today there are three ways, and a
library that proves laws can use none of them well:

| way | cost | problem |
|---|---|---|
| `Array.clone` the matrix per task | O(n) per task per call | costs more than the products: 2304 × 768 with 8 tasks takes 10.4 ms, the sequential product 0.79 ms |
| `Array.fork` / `Array.join` (#885) | O(1) | `@unsafe`: `--check-only` and `--verdict` then report `SOME PROOFS FAIL: 1 def relies on unsafe or foreign code`, so a package that publishes laws cannot contain it |
| split the matrix into its own arrays, one per task, once (what we do) | one copy at load time | works and is proved, but the split is fixed at load time and the code has to carry a tree of arrays |

## What we measured

[bend-ml](https://github.com/nuxyel/bend-ml), Bend 2.0.35, Intel Core Ultra 7 155H (6 P + 8 E + 2 LP-E
cores), `bend -o`, time per product with the matrix already built (`bench/mv_bands.py`):

| W (rows x columns) | sequential | clone per task (8 tasks) | 2^4 bands | 2^5 bands |
|---|---|---|---|---|
| 2304 x 768 | 0.80 ms | 10.4 ms | 0.32 ms | 0.33 ms |
| 768 x 768 | 0.26 ms | 4.76 ms | 0.16 ms | 0.15 ms |
| 3072 x 768 | 1.07 ms | 23.5 ms | 0.42 ms | 0.40 ms |
| 768 x 3072 | 1.26 ms | 22.7 ms | 0.45 ms | 0.42 ms |
| 50257 x 768 | 25.1 ms | 337 ms | 6.1 ms | 4.5 ms |

All with `--threads 16`; the full table, with 1, 4 and 8 threads, is `bench/results/mv_bands-2026-10-05.txt`.

`Bands<r, c>` in `bend-ml-tensor-array` keeps the matrix as a tree of row bands, each band its own
`Array`; a `match` on the tree hands each task its band in O(1), and only the input vector (c numbers)
is cloned. It is safe and proved (`half_cover`), and it is the workaround we use. But the bands are
decided when the matrix is loaded, and every operation that needs the whole matrix (an embedding lookup,
a transposed product for training) has to walk the tree.

The guide's Ownership section (`guide/SHADERS.md`) describes the borrow the compiler already does: "The
compiler borrows a boxed parameter (not an `Array`) that the def only matches or passes to a borrower".
Arrays are the one kind of data that a library cannot share read-only between tasks without either copying
or `@unsafe`.

## Request

A borrow that the type system can check, for data that many tasks only read. For example
`Array.borrow(a: Array<T>, f: ReadOnly<T> -> R) -> Array<T> & R`, where `ReadOnly<T>` is `Data` (so it can
go down both sides of a parallel let with `+`) and only offers `get` and `size`. No write can reach the
shared block, so there is nothing to race and nothing for the termination or ownership checks to reject.

This matches two items on the compiler roadmap (2026-09-29): "way better support for shared atomic
arrays" and "more parallelism options (not just fork/join)".

## Repro

- `bench/mv.bend` (`Mat.matmul_nt` with `par = 3`, cloning per task) against `bench/mv_bands.bend` (bands):
  `reference/.venv/bin/python bench/mv_bands.py`.
- Numbers and discussion: `NOTES.md`, experiments 9 and 11.
