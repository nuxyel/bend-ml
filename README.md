# bend-ml

**Machine learning in [Bend 2](https://bend-lang.com), where a shape error is a type error.**

Five packages published on BendHub, an MNIST MLP and GPT-2 small (124 M), all written in Bend, with proofs verified by Bend's kernel and numbers checked against PyTorch and `tiktoken`. Bend version: **2.0.35**.

```python
# (2x3) · (4x5): does not compile
def bad() -> TA.MMul<2n, 3n, 5n>:
  TA.Mat.matmul(2n, 3n, 5n, 0n, TA.Mat.zeros(2n, 3n), TA.Mat.zeros(4n, 5n))
```
```
Error:
- expected : TA.Mat<3n, 5n>
- observed : TA.Mat<4n, 5n>
```

## Packages on BendHub

| Package | Version | What it is | Proved LAWS |
|---|---|---|---|
| [`bend-ml-nat-lemmas`](nat-lemmas) | 0.1.1.0 | `Nat` and `List` lemmas that Base does not have | `add_comm`, `add_assoc`, `mul_comm`, `mul_assoc`, `mul_dist`, `append_assoc`, `length_append`, `product_append`... (15) |
| [`bend-ml-bpe-tokenizer`](bpe) | 0.1.2.0 | Byte-level BPE tokenizer (GPT-2 style) | **roundtrip** `decode(encode(s)) = s`, `vocab_bound`, `dec_append`, **`train_wf`** (`train` always produces a well-formed table) and **`roundtrip_trained`** (roundtrip for any trained table, no hypothesis) |
| [`bend-ml-tensor`](tensor) | 0.1.2.0 | `Vec<n>` and `Mat<r,c>` with the shape in the type, over lists | `reshape_swap`, `reshape_flat`; `reshape` only compiles with a proof that the number of elements does not change |
| [`bend-ml-tensor-array`](tensor-array) | 0.1.2.0 | **New in v2.** `Mat<r,c>` over a flat `Array<F32>`, the same shape guarantee, **~50x faster** | the products `matmul`, `matmul_nt`, `matmul_tn` (the shape of `dW = Xᵀ·dY` is checked by the type), parallel blocks |
| [`bend-ml-autograd`](autograd) | 0.1.1.0 | Automatic differentiation and layers with typed backward | **`reverse_eq_forward`**: the reverse mode of autodiff gives the same result as the forward mode |

```python
import bend-ml-tensor-array@0.1.2.0/main.bend as TA
import bend-ml-bpe-tokenizer@0.1.2.0/main.bend as BPE
```

## Demos (v2)

| Demo | Result |
|---|---|
| [MNIST](demos/mnist) (784-128-10 MLP) | **6.6 s per epoch** (v1: 544 s). Loss and hits **identical** to PyTorch with the same weights and batches: 0.5204771 / 9129, then 0.27043572 / 9298, then 0.2156194 / 9418. PyTorch takes 0.2 to 0.3 s per epoch ([honest benchmark](demos/mnist/BENCHMARK.md)). |
| [GPT-2 small](demos/gpt2) (124 M) | **~0.1 s per token** (v1: 3 s). It generates the **same tokens** as PyTorch (22 tokens, 3 prompts), logits within 2e-4. PyTorch does the forward pass in 21 ms (16 threads) or 55 ms (1 thread); loading the weights takes 9 s in Bend ([details](demos/gpt2/BENCHMARK.md)). Tokenizer identical to `tiktoken` on 16/16 texts. |

```
$ ./gpt2_fast "The capital of France is" 8
  id 262  logit -100.24986 ...
text: The capital of France is the capital of the French Republic, and
```

### What changed from v1 to v2

v1 used linked lists for the matrices and measured ~1800x PyTorch on MNIST. v2 measured each hypothesis (`NOTES.md`, experiments 1 to 9): swapping lists for a flat `Array` gave **~49x on one thread**, products in parallel blocks gave ~1.8x more, and in GPT-2 I found that **parallelizing matrix · vector by copying the matrix costs 10x more than computing**, so it runs sequentially. Distance to PyTorch now: MNIST ~22 to 33x, GPT-2 ~2 to 5x per token.

## What is proved and what is tested

- **Proved by the kernel** (`bend X.bend --verdict`): the laws above, in five packages. No `@unsafe`, no `?TODO`, in any of them.
- **By the type**: the shapes of `matmul`, of each layer gradient (`dW: Mat<i,o>`) and `reshape` with a proof. The whole MNIST training step is checked this way.
- **Tested, not proved**: all `F32` numerics (it is not a real number; it rounds). Gradient checking and comparison with PyTorch live in `reference/`. The tokenizer is exact on ASCII; bytes ≥ 128 count as letters in the GPT-2 pre-tokenizer.

## Verify everything

```bash
export PATH="$HOME/.bend/bin:$PATH"
reference/.venv/bin/python reference/check_all.py          # 27 checks, ~30 s
reference/.venv/bin/python reference/check_all.py --full   # + GPT-2 and MNIST (32 checks, ~1.5 min)
```

Data and weights preparation: [`reference/`](reference) (`gpt2_prep.py`, `mnist_torch.py`) and the README of each demo.

## Limits

- **Performance:** Bend 2.0.35 generates scalar code, with no BLAS or SIMD: on the same matrix product PyTorch does 62 G multiply-adds/s on 1 thread and Bend with `Array` ~2.3 G/s (~27x). Parallelism scales ~2 to 4x on this hybrid CPU (P+E cores). GPU not used: Bend asks for CUDA 12 and Arch ships 13.
- **Memory:** the GPT-2 weights become trees of nodes (~10 GB while loading, 9 s).
- Only `Nat`, `U32` and `F32`; no `F64`.
- `Mat<r,c>` does not carry the invariant "the `Array` has capacity ≥ r*c" in its type: the constructors guarantee it, but it is not a fact of the type.
- The GPT-2 pre-tokenizer treats every byte ≥ 128 as a letter (exact for accented letters and other alphabets).

## Structure

```
nat-lemmas/  bpe/  tensor/  tensor-array/  autograd/   # packages (main.bend + README + LICENSE)
demos/mnist  demos/gpt2                 # demos with README and BENCHMARK
reference/                              # PyTorch, tiktoken and check_all.py
poc/  bench/                            # proofs of concept (v0.2) and v2 micro-benchmarks
docs/                                   # shape-error messages and launch material
NOTES.md                                # decisions, findings and debts
```

## License

[MIT](LICENSE).
