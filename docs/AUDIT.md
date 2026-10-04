# Audit guide

This project claims that some properties are **proved** and others only **tested**. Tests cannot replace a
review of the proved statements: a proof is only as meaningful as the law it proves. This guide tells a
reviewer what to read, what is trusted, and how to re-check everything.

## 1. Read the specification, not the proofs

```bash
reference/.venv/bin/python reference/list_laws.py                  # every law: name, quantifiers, claim
reference/.venv/bin/python reference/list_laws.py --check-readme   # every law is documented in its README
```

The output is the whole specification (the proofs are the `def`s right below each `law` in
`<package>/main.bend`; the kernel checks them, you do not need to read them). Questions to ask of each law:

- Is the claim what the README says it is (read the plain-language table next to it)?
- Is it **vacuous**? A law with a hypothesis that is never satisfiable proves nothing.
  Example: `roundtrip` assumes `wf(table) = True`; `train_wf` proves that the tables produced by `train` satisfy
  it, and `roundtrip_trained` combines both with no hypothesis left. `bpe/tests`-style runs
  (`reference/test_fuzz_bpe.py`) also show the hypothesis holds on real tables.
- Do the **definitions** the law mentions (`encode`, `decode`, `nbwd`, `nfwd`, `wf`, ...) mean what their names say?
  The laws are only as good as those definitions.
- What do the quantity marks mean? `+x` means "may be used more than once", `-x` means "erased at run time";
  they restrict how a proof may use a variable, they do not weaken the statement.

## 2. What is proved

| Package | Laws | What they guarantee |
|---|---|---|
| `bend-ml-nat-lemmas` | 15 (`add_comm`, `mul_assoc`, `append_assoc`, `length_append`, `product_append`, ...) | Ordinary arithmetic and list facts the shape proofs need |
| `bend-ml-bpe-tokenizer` | `roundtrip`, `vocab_bound`, `dec_append`, `train_wf`, `roundtrip_trained` | `decode(encode(s)) = s` for every well-formed table, in particular every trained one |
| `bend-ml-tensor` | `reshape_swap`, `reshape_flat` | The element counts that make a reshape possible |
| `bend-ml-tensor-array` | see `list_laws.py` | Capacity bound of the flat array (when present) |
| `bend-ml-autograd` | `reverse_eq_forward` | Reverse-mode autodiff equals forward mode, over `Nat` expressions with sums and products |

Besides laws, **types** carry guarantees that no law restates: `Mat.matmul : Mat<n,k> -> Mat<k,m> -> Mat<n,m>`,
`Mat.matmul_tn` for `dW = Xᵀ·dY`, and `Mat.reshape` taking a proof of `r1*c1 = r2*c2`. The negative tests
(`tensor/tests/bad_*.bend`, `tensor-array/tests/bad_*.bend`) check that wrong shapes are refused.

## 3. What is trusted (the trust base)

1. **Bend's type checker and the proof kernel.** `bend X.bend --verdict` re-checks a file with
   `bendtt.lean`, a small kernel with a Lean proof that no accepted definition has type `Empty` and that live
   code halts (`~/.bend/bend2/bendtt.lean`, `paper/BendTT.pdf`). The compiler and runtime, which run the code,
   are **not** part of the proof guarantee; they are tested by the comparisons below.
2. **Primitives declared as bodiless `law`s in Base**: all `F32` operations (`F32.add`, `F32.exp`, ...), `U32`
   conversions and the IO effects. They are opaque to the prover, so nothing here proves facts about floating
   point; that is why numerics are tested, not proved.
3. **`Array` primitives** (`Array.new`, `get`, `set`, `clone`): a size-`2^d` array is trusted to behave like a
   function from indexes to values.
4. **The Python references** (PyTorch, tiktoken, NumPy) used as oracles.

## 4. What is tested but not proved

- Numerical agreement of `F32` code with PyTorch: `reference/test_tensor*.py`, `test_autograd*.py`,
  `test_gpt2.py` (logits within 2e-4), MNIST losses and hits.
- The tokenizer against `tiktoken`: `test_gpt2_tok.py`, plus the seeded fuzz test `test_fuzz_bpe.py`.
- The packages **as published**: `test_published.py` imports each one from BendHub at the version in the README.
- Everything above runs in CI on every push (`.github/workflows/ci.yml`).

## 5. Known gaps (also listed in the READMEs and `NOTES.md`)

- The choice of merges made by `train` (which pair is the most frequent) is not proved. The roundtrip does not
  depend on it, but the quality of the tokenizer does.
- `Mat<r,c>` does not carry "the array has capacity >= r*c" in its type; the constructors allocate it.
- GPT-2's pre-tokenizer follows the GPT-2 regex on ASCII exactly; non-ASCII symbols are approximated by ranges.
- No Unicode tables for non-ASCII decimal digits.
- Tests were written by the author of the code. An independent review of the laws in section 1 is the missing piece.

## 6. Reviewer checklist

- [ ] `make setup && make check-full` passes on a clean checkout (or see the green CI run).
- [ ] `grep -rn "@unsafe\|?TODO" */main.bend` prints nothing.
- [ ] For every package: `bend <package>/main.bend --verdict` prints `ALL PROOFS CHECK`.
- [ ] `list_laws.py` statements match the README tables; no law is vacuous (section 1).
- [ ] The definitions used by the laws (`encode`, `decode`, `wf`, `nfwd`, `nbwd`) read as intended.
- [ ] Section 3's trust base is acceptable for your use.
- [ ] Anything surprising goes to a GitHub issue; a counterexample to a law would be a kernel bug.
