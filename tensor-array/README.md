# bend-ml-tensor-array

Tensors in Bend 2 over a flat `Array<F32>`, **with the shape in the type**: the same guarantee as [`bend-ml-tensor`](../tensor) (a product with wrong dimensions does not compile), but **~50x faster** than the list-based `bend-ml-tensor`, because it reads and writes by index and runs products in parallel blocks of rows.

- Bend: **2.0.35** · License: MIT.
- `bend tensor-array/main.bend` and `--verdict` → `ALL PROOFS CHECK` (no `@unsafe`, no `?TODO`).
- Tests against PyTorch: `reference/test_tensor_array.py` (52 checks, maximum error ≈ 1.4e-6).

```python
import Base
import bend-ml-tensor-array@0.1.3.0/main.bend as TA

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

## A shape error is a type error

`(2×3) · (4×5)` does not compile (`tensor-array/tests/bad_matmul.bend`):

```
Error:
- expected : TA.Mat<3n, 5n>
- observed : TA.Mat<4n, 5n>
```

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

- `0.1.3.0`: `cap_depth` is now defined by `Nat` recursion and proved (`law cap_ok`); adds `Mat.softmax_ce_checked`, `Mat.count_correct_checked` and `Mat.fill_at` (load a matrix block by block).
- `0.1.2.0`: the same API, with English comments and README.
- `0.1.1.0`: column split for `n = 1` (matrix · vector) and `Mat.from_list`.
- `0.1.0.0`: first publication.
