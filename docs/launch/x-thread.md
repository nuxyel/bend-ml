# Rascunho da thread no X (em inglês)

Revise o tom antes de postar. Todos os números abaixo estão em `NOTES.md` e nos BENCHMARK.md.

---

**1/**
@VictorTaelin you said "build something in Bend". I built ML in it, with the one thing nobody else has: shapes in the type.

A matmul of (2×3)·(4×5) doesn't compile:

`expected Mat<3n,5n> / observed Mat<4n,5n>`

Repo: github.com/nuxyel/bend-ml
[video]

**2/**
4 packages on BendHub (MIT):
• bend-ml-nat-lemmas: the Nat/List lemmas Base lacks
• bend-ml-bpe-tokenizer: BPE with a proven roundtrip
• bend-ml-tensor: Vec<n>, Mat<r,c>, reshape needs a proof
• bend-ml-autograd

All `--verdict` clean, no @unsafe.

**3/**
Two laws I'm proud of:
• decode(encode(s)) == s for any well-formed merge table
• reverse-mode autodiff == forward-mode (over Nat, using the proven mul/add lemmas)

Floats aren't reals, so F32 numerics are checked against PyTorch instead. (21 gradient checks, max err ~5e-7.)

**4/**
GPT-2 small (124M) runs in Bend: my tokenizer matches tiktoken (16/16 texts), and generation matches PyTorch token for token, logits within 2e-4.

"The capital of France is the capital of the French Republic, and"

**5/**
Honest part: it's slow. MNIST epoch: 544 s in Bend vs 0.3 s PyTorch (same loss, same 9129/10000). GPT-2 ~3 s/token. No BLAS, lists as matrices, CUDA 12 not set up here. My row-parallel matmul only scaled ~1.8x and I couldn't tell why.

Benchmarks with hardware/versions in the repo.

**6/**
Things I hit that might be useful: imports aren't re-exported (a library needs law+proof in one file); match only on parameters; `--` for negative CLI args; Base has almost no lemmas.

Happy to help turn this into a proper `llama.bend` starting point. 🙂
