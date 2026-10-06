# v3 plan: answer the launch feedback and make matrix · vector parallel without copies

## Why

v2.1 was announced on X on 2026-10-05. The replies gave a concrete list:

- Victor Taelin: "compile-time sized vectors are one of the applications of dependent types that I miss the
  most in TypeScript". The reaction was to the types, not to the speed.
- "Rust dfdx already does compile-time type checking, no?"
- "How bad do compile times get on deeper nets?"
- kazzzz520 ([Bend-Conv](https://github.com/Kaz9487/Bend-Conv), a tensor and convolution library in Bend):
  "The math itself was never the slow part. Copying data, computing addresses, and waking up threads is."
- Taelin's roadmap for the compiler lists "way better support for shared atomic arrays" and "more
  parallelism options (not just fork/join)".

v3 has two parts. A llama-architecture model and a training framework come after v3.

## Part A: `Bands<r, c>` and parallel matrix · vector

In v2, `Mat.matmul_nt` with n = 1 could only run in parallel by cloning the whole weight matrix for each task;
each weight is read once in matrix · vector, so the copies cost more than the products (`NOTES.md`, exp. 9:
`par = 3` was 10x slower than sequential).

`Array<T>` in Base is a plain data type (`ALeaf` / `ANode`), so a `match` hands out its subtrees in O(1) with no
`@unsafe`. `Bands<r, c>` keeps a matrix as a tree of row bands, each band its own `Array`, so each task takes
its band and only the input vector is copied.

- The split is part of the type: a node with r rows has bands of `half(r)` and `r - half(r)` rows, and the law
  `half_cover` proves that they add up to r. The row count at run time is tied to the type by an erased proof.
- `Bands.matvec` uses the same dot-product kernel as `Mat.matmul_nt`, so its results are bit-identical.
- `Bands.fill_at` (streaming load), `Bands.read_row` (embedding lookup), `Bands.to_list`.
- Measured with `bench/mv_bands.bend` against `bench/mv.bend`; applied to `demos/gpt2/fast.bend`.

## Part B: answers to the feedback

1. A README section on what this adds over const-generic shape checking (dfdx), with sources.
2. A measurement of check and build times as the number of distinct typed layers grows.
3. A draft report for `bendlang/bend` on sharing read-only arrays between tasks (`docs/upstream/`).

## Stop criterion

1. `Bands` proved (`--verdict`), tested against NumPy and bit for bit against `matmul_nt`, published.
2. The mat·vec result measured and recorded, including a negative result if it does not pay off.
3. GPT-2 still matches PyTorch token for token.
4. `make check-full` green locally and in CI; tag `v3.0.0` and a release.
