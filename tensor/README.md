# bend-ml-tensor

Tensors in Bend 2 with the **shape in the type**: a shape error is a type error, not a run-time failure.

- Bend: **2.0.35** · License: MIT · Depends on `bend-ml-nat-lemmas@0.1.0.0`.
- `bend tensor/main.bend` and `bend tensor/main.bend --verdict` → `ALL PROOFS CHECK` (no `@unsafe`, no `?TODO`).

```python
import Base
import bend-ml-tensor@0.1.2.0/main.bend as T

def prod() -> T.Mat<2n, 4n>:
  T.Mat.matmul(2n, 3n, 4n, T.Mat.fill(2n, 3n, 2.0), T.Mat.fill(3n, 4n, 3.0))
```

This package stores matrices as lists of rows. For a ~50x faster flat-`Array` version with the same shape guarantees, see [`bend-ml-tensor-array`](../tensor-array).

## Types

- `Vec<n>`: a vector of `n` `F32` numbers.
- `Mat<r, c>`: an `r × c` matrix (a list of `r` rows with `c` numbers each).
- The dimensions are **erased** type parameters: they cost nothing at run time, but the checker verifies them in every operation.
- `Mat.of(r, c, rows)` and `Vec.of(n, list)` only return a value (`Some`) if the real sizes match.

## Operations

`Mat.add/sub/mul/scale`, `Mat.matmul`, `Mat.matmul_t` (the second operand already transposed, for large weights: `Mat<n,k> · Mat<m,k>ᵀ = Mat<n,m>`), `Mat.transpose`, `Mat.relu`, `Mat.gelu` (tanh, like GPT-2), `Mat.softmax` (per row, stable), `Mat.layernorm`, `Mat.add_row` (adds a `Vec<m>` bias to each row), `Mat.col_sums`, `Mat.matvec`, `Mat.reshape`, `Mat.flatten`, `Mat.reshape_swap`; and `Vec.add/sub/mul/scale/dot/sum/relu/softmax`.

## A shape error is a type error

`(2×3) · (4×5)` does not compile (`tensor/tests/bad_matmul.bend`):

```
Error:
- expected : T.Mat<3n, 5n>
- observed : T.Mat<4n, 5n>
```

A `reshape` from 2×6 (12 elements) to 5×3 (15) is refused, because there is no proof of `12 == 15` (`tensor/tests/bad_reshape.bend`):

```
Error:
- expected : 12n
- observed : 15n
```

A `reshape` from 2×6 to 3×4 compiles because the proof is just computing (`{==}`).

## Proved LAWS (in plain language)

| Law | What it states |
|---|---|
| `reshape_swap` | `r × c` and `c × r` have the same number of elements: `r*c = c*r` (uses `mul_comm` from nat-lemmas). So `Mat.reshape_swap` always exists. |
| `reshape_flat` | `r × c` has the same number of elements as `1 × (r*c)`: `r*c = 1*(r*c)` (uses `mul_one_l`). So `Mat.flatten` always exists. |

Besides the laws, **the type of `Mat.matmul` is itself a guarantee**: `Mat<n,k> → Mat<k,m> → Mat<n,m>`; and `Mat.reshape` only accepts an argument that is a proof of `r1*c1 = r2*c2`.

## What is NOT proved

- The numbers: `F32` is not a real number (it rounds), so numerical correctness is validated by tests against PyTorch, not by proof (`reference/test_tensor.py`): `matmul`, `transpose`, `relu`, `softmax` (even with huge values), `gelu` and `layernorm` match with a maximum error on the order of 1e-6.
- The internal invariant "each row has exactly `c` numbers" holds because the operations preserve it and the constructors check it (`Mat.of`); it is not a proof in the type.

## Performance

The data are lists (copyable, with no logarithmic-cost indexing): about 44 million multiply-adds per second on one thread. See `demos/mnist/BENCHMARK.md`.

## Versions

- `0.1.2.0`: the same API, with English comments and README.
- `0.1.1.0`: adds `Mat.matmul_t`.
- `0.1.0.0`: first publication.
