# bend-ml-autograd

Automatic differentiation in Bend 2, with a **proved law**: the reverse mode (what neural-network training uses) computes the same derivative as the forward mode.

- Bend: **2.0.35** · License: MIT · Depends on `bend-ml-nat-lemmas@0.1.0.0` and `bend-ml-tensor@0.1.0.0`.
- `bend autograd/main.bend` and `--verdict` → `ALL PROOFS CHECK` (no `@unsafe`, no `?TODO`).

## What it has

1. **A proved model (over `Nat`)**: expressions `NE` with one variable `X`, constants, sum and product; `nval`, `nfwd` (forward-mode derivative, with dual numbers) and `nbwd` (reverse mode, passing the gradient from top to bottom).
2. **Scalar autograd in `F32`** (micrograd style): `G` with `GCst`, `GVar`, `GAdd`, `GMul`, `GRelu`, `GTanh`, `GExp`; `gval` (value) and `ggrad` (gradient with respect to each variable).
3. **Layers with tensors**, with the shape of each gradient guaranteed by the type: `linear` / `linear_bwd`, `relu_bwd`, `ce_loss` / `ce_grad` (cross-entropy), `sgd`, `sgd_vec`.

## The proved LAW (in plain language)

| Law | What it states |
|---|---|
| `reverse_eq_forward` | For any expression made of constants, `X`, sums and products, and for any `x`: the reverse-mode gradient, starting with gradient 1 at the output, equals the forward-mode derivative. |

The idea of the proof (the comments in `main.bend` give the details): we prove something stronger, `reverse(e, g) = g × forward(e)`, by induction on the expression. For a sum, distributivity joins the two halves; for a product, `(g·vb)·fa + (g·va)·fb = g·(fa·vb + va·fb)` follows from associativity, commutativity and distributivity, all coming from `nat-lemmas`. The case `g = 1` gives the law.

## What is NOT proved (and how it is verified)

- The law holds for `Nat` (a commutative semiring). `F32` has the same structure but rounds, so **`F32` numerical correctness is verified by gradient checking against PyTorch**, not by proof.
- Non-polynomial operations (`relu`, `tanh`, `exp`) are not part of the law.
- Tensor shapes: `linear_bwd` returns `dx: Mat<n,i>`, `dW: Mat<i,o>`, `db: Vec<o>`; a product with swapped dimensions does not compile. The inner lists remain a library invariant (see `bend-ml-tensor`).

## Gradient checking (against PyTorch)

`reference/test_autograd.py` compares Bend with PyTorch's autograd: 5 scalar expressions at 3 points each (with `relu` on both sides of zero), `linear_bwd` (3 shapes), `relu_bwd` and cross-entropy (loss and gradient on the logits). 21 checks, maximum error ≈ 5e-7, 0 failures:

```bash
reference/.venv/bin/python reference/test_autograd.py
```

## Versions

- `0.1.1.0`: the same API, with English comments and README.
- `0.1.0.0`: first publication.
