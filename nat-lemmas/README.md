# bend-ml-nat-lemmas

Proved lemmas about `Nat` and `List` that Bend 2's Base does not have yet. They are the foundation for the rest of bend-ml (tensor shapes, the tokenizer).

- Bend: **2.0.35**
- License: MIT
- No `@unsafe`, no `?TODO`: `bend nat-lemmas/main.bend` prints `ALL PROOFS CHECK`, and `bend nat-lemmas/main.bend --verdict` re-checks it with the Lean-proved kernel.

## Usage

```python
import Base
import bend-ml-nat-lemmas@0.1.1.0/main.bend as NL

def comm(a: Nat, +b: Nat) -> {Nat.add(a, b) == Nat.add(b, a) : Nat}:
  NL.add_comm(a, b)
```

A parameter carries `+` when it may be used more than once (Bend's affine-variable rule).

## Proved LAWS (in plain language)

| Law | What it states |
|---|---|
| `add_zero` | `a + 0 = a` |
| `add_succ` | `1 + (a + b) = a + (1 + b)` |
| `add_comm` | `a + b = b + a` (addition is commutative) |
| `add_assoc` | `a + (b + c) = (a + b) + c` (addition is associative) |
| `mul_zero` | `a * 0 = 0` |
| `mul_succ` | `a * (1 + b) = a + a * b` |
| `mul_comm` | `a * b = b * a` (multiplication is commutative) |
| `mul_dist` | `a*c + b*c = (a + b)*c` (distributivity) |
| `mul_assoc` | `a * (b * c) = (a * b) * c` (multiplication is associative) |
| `mul_one_l`, `mul_one_r` | `1 * a = a` and `a * 1 = a` |
| `append_nil` | `xs ++ [] = xs` |
| `append_assoc` | `(xs ++ ys) ++ zs = xs ++ (ys ++ zs)` |
| `length_append` | `length(xs ++ ys) = length(xs) + length(ys)` |
| `product_append` | `product(xs ++ ys) = product(xs) * product(ys)` |

`product(xs)` multiplies the elements of a list of `Nat` (the empty list is 1). It is the number of elements of a tensor whose shape is `xs`, which is why `product_append` is the basis of `reshape` with a proof.

## How to read a proof

In Bend, a proof is a function whose type is the statement. `match` does case analysis; the recursive call is the induction hypothesis; `%e : P` rewrites the goal using the equality `e`; `{==}` closes the goal when both sides are already the same term. The comments in `main.bend` explain each step.

## Credits

The proofs of `add_*` and `mul_*` follow Bend's official `proof_numerics` demo, adapted to Base's `Nat.add` and `Nat.mul`.

## Known limits

- Lemmas about `reverse`, `take`/`drop` and ordering (`<=`) are left for future versions.

## Versions

- `0.1.1.0`: the same laws, with English comments and README.
- `0.1.0.0`: first publication.
