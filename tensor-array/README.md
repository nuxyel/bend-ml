# bend-ml-tensor-array

Tensors in Bend 2 over a flat `Array<F32>`, **with the shape in the type**: the same guarantee as [`bend-ml-tensor`](../tensor) (a product with wrong dimensions does not compile), but **~50x faster** than the list-based `bend-ml-tensor`, because it reads and writes by index and runs products in parallel blocks of rows.

- Bend: **2.0.35** · License: MIT.
- `bend tensor-array/main.bend` and `--verdict` → `ALL PROOFS CHECK` (no `@unsafe`, no `?TODO`).
- Tests against PyTorch and NumPy: `reference/test_tensor_array.py` (280 checks, maximum error ≈ 1.4e-6; `Bands.matvec` equal to `Mat.matmul_nt` bit for bit).

```python
import Base
import bend-ml-tensor-array@0.1.5.0/main.bend as TA

def prod(a: TA.Mat<100n, 784n>, w: TA.Mat<784n, 128n>) -> TA.MMul<100n, 784n, 128n>:
  TA.Mat.matmul(100n, 784n, 128n, 3n, a, w)   # 3n = 2^3 = 8 parallel blocks
```

## How it works

- `Mat<r, c>` stores `r*c` numbers in an `Array<F32>`, row by row (index `i*c + j`). The dimensions are **erased** type parameters.
- An `Array` is affine (a single owner), so **every operation also returns what it read**: `Mat.matmul(a, b)` returns `MMul{a, b, c}`. Use `Mat.clone` when you need two copies.
- `par` (in the products) is the log2 of the number of blocks computed in parallel; `0` is sequential. It splits the rows of `C`, or the columns when `n = 1` (matrix · vector). Each block works on a copy (`Array.clone`, cheap) of `A` and `B`.

## Operations

| Operation | Type |
|---|---|
| `Mat.zeros`, `Mat.fill`, `Mat.of_list`, `Mat.from_list`, `Mat.to_list`, `Mat.clone` | `Mat<r,c>` (`of_list` only returns `Some` if the list has `r*c` numbers; `from_list` does not check) |
| `Mat.matmul` | `Mat<n,k> · Mat<k,m> = Mat<n,m>` |
| `Mat.matmul_nt` | `Mat<n,k> · Mat<m,k>ᵀ = Mat<n,m>` (weights with one row per output) |
| `Mat.matmul_tn` | `Mat<k,n>ᵀ · Mat<k,m> = Mat<n,m>` (weight gradient: `Xᵀ · dY`) |
| `Mat.add_row` | adds a bias `Mat<1,m>` to each row of `Mat<n,m>` |
| `Mat.relu`, `Mat.relu_bwd` | the activation and the gradient that passes through it |
| `Mat.sgd`, `Mat.add` | `w - lr·dw` and `a + b` |
| `Mat.col_sums` | `Mat<n,m> -> Mat<1,m>` |
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
| `leaf_len` | the kernel of one band (`lcols`, then `bv_leaf`) puts exactly one number per row it computes on the output list |
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

- `0.1.5.0`: `Bands.matvec_l` (lists in and out) and the laws `leaf_len` and `band_len`.
- `0.1.4.0`: `Bands<r, c>` (row bands, a parallel matrix · vector that copies no weights), with the law `half_cover`.

- `0.1.3.0`: `cap_depth` is now defined by `Nat` recursion and proved (`law cap_ok`); adds `Mat.softmax_ce_checked`, `Mat.count_correct_checked` and `Mat.fill_at` (load a matrix block by block).
- `0.1.2.0`: the same API, with English comments and README.
- `0.1.1.0`: column split for `n = 1` (matrix · vector) and `Mat.from_list`.
- `0.1.0.0`: first publication.
