# bend-ml-tensor-array

Tensors in Bend 2 over a flat `Array<F32>`, **with the shape in the type**: the same guarantee as [`bend-ml-tensor`](../tensor) (a product with wrong dimensions does not compile), but **~50x faster** than the list-based `bend-ml-tensor`, because it reads and writes by index and runs products in parallel blocks of rows.

- Bend: **2.0.39** · License: MIT.
- `bend tensor-array/main.bend` and `--verdict` → `ALL PROOFS CHECK` (no `@unsafe`, no `?TODO`).
- Tests against PyTorch and NumPy: `reference/test_tensor_array.py` (280 checks, maximum error ≈ 1.4e-6; `Bands.matvec` equal to `Mat.matmul_nt` bit for bit).

```python
import Base
import bend-ml-tensor-array@0.1.17.0/main.bend as TA

def prod(a: TA.Mat<100n, 784n>, w: TA.Mat<784n, 128n>) -> TA.MMul<100n, 784n, 128n>:
  TA.Mat.matmul(100n, 784n, 128n, 3n, a, w)   # 3n = 2^3 = 8 parallel blocks
```

## How it works

- `Mat<r, c>` stores `r*c` numbers in an `Array<F32>`, row by row (index `i*c + j`). The dimensions are **erased** type parameters.
- An `Array` is affine (a single owner), so **every operation also returns what it read**: `Mat.matmul(a, b)` returns `MMul{a, b, c}`. Use `Mat.clone` when you need two copies.
- `par` (in the products) is the log2 of the most blocks computed in parallel; `0` is sequential. It splits the rows of `C`; a product with one row (matrix · vector) runs sequentially at every `par` since 0.1.12.0 (for matrix · vector in parallel, use `Bands`). Since 0.1.11.0 it is an upper bound: a product splits only when it does at least 2^23 multiply-adds and each cell of `C` sums at least 64 of them (`split_depth`), and otherwise runs sequentially; the bits are the same at every depth.

## Operations

| Operation | Type |
|---|---|
| `Mat.zeros`, `Mat.fill`, `Mat.of_list`, `Mat.from_list`, `Mat.to_list`, `Mat.clone` | `Mat<r,c>` (`of_list` only returns `Some` if the list has `r*c` numbers; `from_list` does not check) |
| `Mat.matmul` | `Mat<n,k> · Mat<k,m> = Mat<n,m>` |
| `Mat.matmul_nt` | `Mat<n,k> · Mat<m,k>ᵀ = Mat<n,m>` (weights with one row per output) |
| `Mat.matmul_tn` | `Mat<k,n>ᵀ · Mat<k,m> = Mat<n,m>` (weight gradient: `Xᵀ · dY`) |
| `Mat.matmul_acc`, `Mat.matmul_nt_acc`, `Mat.matmul_tn_acc` | the same products added into a `C: Mat<n,m>` it is given, `C + A·B`, without allocating the product (a gradient that accumulates: `dW += Xᵀ · dY`); the bits of `Mat.add(C, A·B)` |
| `Mat.add_row` | adds a bias `Mat<1,m>` to each row of `Mat<n,m>` |
| `Mat.relu`, `Mat.relu_bwd` | the activation and the gradient that passes through it |
| `Mat.sgd`, `Mat.add` | `w - lr·dw` and `a + b` |
| `Mat.col_sums`, `Mat.col_sums_acc` | `Mat<n,m> -> Mat<1,m>`; the second adds the sums into a `Mat<1,m>` it is given |
| `Mat.softmax_ce` | mean cross-entropy and its gradient `(softmax − one-hot)/n` |
| `Mat.count_correct` | argmax hits per row |
| `Mat.read_row`, `Mat.write_row` | a row as a list |

## Bands: matrix · vector in parallel without copying the weights

In matrix · vector (one token through a layer of a language model) every weight is read once, so
`Mat.matmul_nt` with `par > 0`, which clones the matrix for each block, is slower than the sequential
product (10.4 ms against 0.79 ms for 2304 × 768). `Bands<r, c>` keeps the matrix as a tree of row bands,
each band its own `Array`. Each task takes its band with a `match` (O(1), no `@unsafe`) and only the input
vector is copied.

```python
w : TA.Bands<2304n, 768n> = TA.Bands.zeros(4n, 2304n, 768n)   # 2^4 bands, filled with Bands.fill_at
TA.Bands.matvec(2304n, 768n, w, x)   # x : Mat<1, 768>  ->  BV{w, y : Mat<1, 2304>}
```

| Operation | Type |
|---|---|
| `Bands.zeros(d, r, c)` | `Bands<r,c>` in `2^d` bands |
| `Bands.fill_at(r, c, i, n, xs, b)` | writes `n` numbers at flat index `i` (row-major); numbers past `r*c` are dropped |
| `Bands.matvec` | `Bands<r,c> · Mat<1,c> = Mat<1,r>`, the bands in parallel; the same numbers as `Mat.matmul_nt`, bit for bit |
| `Bands.matvec_l` | the same product with `x` and `y` as lists (no `Mat` conversions): the fast path for a caller that already holds lists |
| `Bands.read_row` | row `i` as a list (an embedding lookup) |
| `Bands.to_list` | all `r*c` numbers in row order |

A node with `r` rows has bands of `half(r)` and `r - half(r)` rows; the split is in the type, so the row
count of every band follows from `r` without reading the tree. Time per product, 16 threads, Intel Core
Ultra 7 155H (`bench/mv_bands.py`, `bench/results/`):

| W | sequential | `par = 3` (clones W) | 2^4 bands |
|---|---|---|---|
| 2304 × 768 | 0.80 ms | 10.4 ms | 0.32 ms |
| 3072 × 768 | 1.07 ms | 23.5 ms | 0.42 ms |
| 50257 × 768 | 25.1 ms | 337 ms | 6.1 ms |

## Laws

| Law | In plain language |
|---|---|
| `cap_ok` | an `Array` with `2^cap_depth(n)` slots always has room for `n` numbers, so the constructors never allocate too little |
| `half_cover` | the two bands of a node, `half(r)` rows and `r - half(r)` rows, add up to exactly `r` rows: no row is lost and none is counted twice |
| `leaf_len` | reading a band's results back from its `Array` (`read_l`, then `bv_g2`) gives exactly one number per row read |
| `band_len` | the product of one band with `rows` rows gives exactly `rows` numbers |

Not proved yet: that the whole tree (`Bands.matvec_l`) gives exactly `r` numbers. The proof would apply the
join lemma (`join_len`, proved) to the two recursive results and to the induction hypotheses about the same
calls, which uses each band's `Array` twice; Bend's affine rules refuse that. `reference/test_tensor_array.py`
checks the length (and every number) on 20 shapes and all depths.

Inside `Bands`, every function that walks the tree carries an erased proof that the row count it computes
with at run time equals the one in the type.

## A shape error is a type error

`(2×3) · (4×5)` does not compile (`tensor-array/tests/bad_matmul.bend`):

```
Error:
- expected : TA.Mat<3n, 5n>
- observed : TA.Mat<4n, 5n>
```

`Bands.matvec` with a vector of the wrong length does not compile either (`tests/bad_bands.bend`: `expected TA.Mat<1n, 768n>, observed TA.Mat<1n, 2304n>`).

And a layer's gradient does not type-check with swapped dimensions (`tests/bad_grad.bend`): `matmul_tn(x, dy)` with `x: 100×784` and `dy: 100×128` is `784×128`; asking for `128×784` gives `expected MMulTN<128n,100n,784n>, observed MMulTN<784n,100n,128n>`.

## Performance (1 thread, 100 M multiply-adds)

| | time | multiply-adds/s |
|---|---|---|
| `bend-ml-tensor` (lists) | 2.27 s | 44 M |
| `bend-ml-tensor-array` (`Array.get`/`set` by index) | **0.046 s** | **2,200 M** |
| PyTorch (1 thread, BLAS) | 0.0016 s | 62,000 M |

With parallel blocks, the large MNIST product (100×784·784×128) gains ~2x on 8 threads. PyTorch is still ~27x ahead on one thread: Bend 2.0.35 generates scalar code, with no BLAS or SIMD. See `demos/mnist/BENCHMARK.md` for the full training run.

## What is NOT proved

- The capacity arithmetic is proved (`cap_ok`: `2^cap_depth(n) >= n`), but that `Array.new(d)` really allocates `2^d` slots is a property of the runtime and stays trusted. The invariant "the `Array` capacity is `>= r*c`" is not carried in the type of `Mat`; the constructors (`zeros`, `fill`, `of_list`) establish it.
- `softmax_ce` and `count_correct` expect `labels` with `n` entries (one per row); the type does not check it. Use `softmax_ce_checked` and `count_correct_checked`, which return `None` when `length(labels) != n`.
- `F32` numerics are validated by tests against PyTorch, not by proof.

## Versions

- `0.1.17.0`: `Mat.softmax_ce` walks its rows over the array in place instead of as lists, with the list version's
  bits (checked word for word on six shapes, with -0, inf, NaN and subnormal logits). 64 rows of 49,152 take 3.0x fewer
  instructions and cycles; the MNIST demo's rows of 10 gain nothing measurable. `tensor-fork` and `tensor-gpu` 0.1.4.0
  still import 0.1.16.0, so a program that uses one of them stays on 0.1.16.0.
- `0.1.16.0`: 0.1.14.0's code again, with its bits; use it (or 0.1.14.0), not 0.1.15.0. In a probe of SmolLM2's three
  products 0.1.15.0's loop took 23% fewer cycles, but inside BendTorch's whole SmolLM2-135M step clang vectorized its
  two-cell pass with shuffles, and the step took 24.6 s instead of 14.2 s on one thread (13.0 instead of 10.8 s on 16).
  With one cell a pass and 0.1.15.0's eight sums, the step was as fast as with 0.1.14.0 (14.1 s; 16 threads 2% faster),
  so the new order gave nothing there and moved every product's bits. NOTES.md, 2026-10-25.
- `0.1.15.0` (not recommended, see 0.1.16.0): the dot product inside every product keeps eight partial sums instead of four (product p into sum p mod 8,
  lane p mod 8 of a cell of 8), and a pass over k computes two cells of C, so each cell of A is read once for both. On
  one thread, over SmolLM2-135M's three linear-layer products, 23% fewer cycles (idle machine, 2026-10-25); against a
  float64 dot the error is about 25% lower at k = 256 to 1536 (a simulation of both orders). Every product's last bits
  move: they are no longer those of 0.1.14.0. The scalar loop (one-row products, `Bands.matvec`) keeps the same order,
  so it still equals the cells of 8 bit for bit (`tests/t8.bend`).
- `0.1.14.0`: `Mat.matmul_precise`, opt-in: `Mat.matmul` with k summed in blocks of 1024 (each block's dot with the
  usual four partial sums, the block totals added in order), for long dots. SmolLM2's head product (k = 49,152) as four
  running sums of 12,288 terms lost the small ones: BendTorch's gradients were 1.8x further from float64 than PyTorch's
  float32, 0.03x with this product. It copies each block's slices of A and B and runs one product a block; at the head's
  shape (256 x 49,152 x 576, 16 threads) that was faster than `Mat.matmul` (0.38 against 0.52 s with tensor-fork's
  products), the blocks fitting the cache. Up to 1024 steps it is `Mat.matmul`'s bits; every other function unchanged
  (`tests/blocks.bend`: the cells against dotr's blocks, built independently).
- `0.1.13.0`: `cap_depth` compares with Base's `Nat.is_le`, one native comparison in the compiled
  program, instead of `leb`, which walked its arguments one step at a time: n steps in all for a capacity of n cells.
  On the host that was little (under 0.4 ms a call); inside a bang it runs on one GPU lane, where `cap_depth(786432)`
  took ~128 ms against ~4.5 ms now (a bang that only computes it, 20 calls, RTX 4050 Laptop). A lemma proves
  `Nat.is_le` equal to `leb`, so the law `cap_ok` is the same statement; the same capacities (`tests/cap.bend`).
- `0.1.12.0`: a product with one row runs the scalar loop at every `par`. Up to 0.1.11.0 it split by columns and
  cloned both operands at every node: one row against a 50,257 x 768 matrix at `par` 4 peaked at 9.5 GB (now 565 MB)
  or ran out of the default heap, and 8192 x 2048 took 124 ms on 16 threads (now 9.9). The same bits
  (`tests/onerow.bend`). `Bands.matvec` is unchanged.
- `0.1.11.0`: the products choose their split depth from the work (`split_depth`): `par` or 0. On 16
  threads a weight gradient over 8 tokens of SmolLM2-135M's sizes (1536 x 8 by 8 x 576) goes from 9.3 to 2.8 ms at `par`
  3, one over 32 tokens from 10.2 to 5.9 ms, and MNIST's small last layer from 0.083 to 0.034 ms; products large enough
  to gain keep their split and their time. The same bits (`tests/split.bend`).
- `0.1.10.0`: accumulating products, `Mat.matmul_acc`, `Mat.matmul_nt_acc`, `Mat.matmul_tn_acc`
  and `Mat.col_sums_acc`, which add the product into a matrix they are given instead of returning a new one. Each cell
  of the product is computed as before and added once to the cell of C, so the result has the bits of `Mat.add(C,
  A·B)` (`tests/acc.bend`, on every path and layout), without allocating the product and walking it a second time. The
  other functions are unchanged and give the same bits as 0.1.9.0. The capacity of a new array (`cap_depth`, the
  smallest `d` with `n <= 2^d`, behind the law `cap_ok`) is now found by doubling, in log2(n) steps of a tail loop,
  instead of n steps each with a comparison of up to n steps: the same capacity (`tests/cap.bend`), 2% off an MNIST
  training epoch in the native build, and the difference between 247 s and 4.3 s for 50 batches in the JavaScript build.
- `0.1.9.0`: the parallel products in cells of 8 no longer copy both operands at every node of
  the fork tree. The larger operand is packed straight into one piece per block (a block owns its piece), and the
  smaller one is copied once per node; the inner loop carries its four sums as arguments instead of a record per step.
  The same bits as 0.1.8.0 (`tests/t8.bend`). An MLP training epoch on MNIST: 4.9 to 3.9 s on one thread, 5.0 to 4.0 s
  on 16. At these sizes (a product is about 1.3 ms of work) more threads add little: the cost of splitting and joining
  weighs as much as the work.
- `0.1.8.0`: products with two or more rows (`Mat.matmul`, `Mat.matmul_nt`, `Mat.matmul_tn`) pack
  A's rows and B's columns once into cells of 8 F32, so the inner loop reads 8 numbers per array access. The additions
  keep `dotr`'s order (product p to partial sum p mod 4, combined as `dotr` combines them after k products), so the
  results are the same bits as 0.1.7.0's, apart from which NaN payload comes out; `tensor-array/tests/t8.bend` checks it
  on every layout and every k from 0 to 19. Products with one row (`Bands.matvec*`, `gemm` called directly) keep the
  scalar loop. An MLP training epoch on MNIST (784-128-10, batches of 100) goes from 8.7 to 4.8 s on one thread.
- `0.1.7.0`: the dot product inside every product (`gemm`, so `Mat.matmul*`, `Bands.matvec*`) keeps four partial sums instead of one running sum. GPT-2 is 4.6% faster on one thread (no change on 16), per product 4-12% on one thread and up to 26% on 16; the logits move closer to PyTorch (largest difference lower on 10 of 11 prompts) and are no longer bit for bit those of 0.1.6.0. `Bands.matvec` still equals `Mat.matmul_nt` bit for bit. Checked on Bend 2.0.36.
- `0.1.6.0`: each band of `Bands.matvec` / `Bands.matvec_l` runs the `gemm` of `Mat.matmul_nt` and reads its rows back, instead of building the list row by row: the same numbers bit for bit, 3-4% faster on GPT-2 with 1 and with 16 threads, 19-32% per product on 16 threads. `leaf_len` and `band_len` are proved again over the new code.
- `0.1.5.0`: `Bands.matvec_l` (lists in and out) and the laws `leaf_len` and `band_len`.
- `0.1.4.0`: `Bands<r, c>` (row bands, a parallel matrix · vector that copies no weights), with the law `half_cover`.

- `0.1.3.0`: `cap_depth` is now defined by `Nat` recursion and proved (`law cap_ok`); adds `Mat.softmax_ce_checked`, `Mat.count_correct_checked` and `Mat.fill_at` (load a matrix block by block).
- `0.1.2.0`: the same API, with English comments and README.
- `0.1.1.0`: column split for `n = 1` (matrix · vector) and `Mat.from_list`.
- `0.1.0.0`: first publication.
