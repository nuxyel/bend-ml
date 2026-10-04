# X thread draft

Review the tone before posting. Every number below is in `NOTES.md` and the BENCHMARK.md files.

---

**1/**
@VictorTaelin you said "build something in Bend". I built ML in it, with the one thing nobody else has: shapes in the type.

A matmul of (2×3)·(4×5) doesn't compile:

`expected Mat<3n,5n> / observed Mat<4n,5n>`

Repo: github.com/nuxyel/bend-ml
[video]

**2/**
5 packages on BendHub (MIT):
• bend-ml-nat-lemmas: the Nat/List lemmas Base lacks
• bend-ml-bpe-tokenizer: BPE with a proven roundtrip
• bend-ml-tensor and bend-ml-tensor-array: shapes in the type, on lists and on flat Arrays
• bend-ml-autograd

All `--verdict` clean, no @unsafe.

**3/**
Laws I'm proud of:
• decode(encode(s)) == s, for ANY table you get from `train`, no hypotheses (train_wf + roundtrip)
• reverse-mode autodiff == forward-mode (over Nat, using the proven mul/add lemmas)
• reshape only compiles with a proof that the element count is preserved

Floats aren't reals, so F32 numerics are checked against PyTorch instead.

**4/**
GPT-2 small (124M) runs in Bend. Token for token the same as PyTorch, logits within 2e-4. My tokenizer matches tiktoken on 16/16 texts.

"The capital of France is the capital of the French Republic, and"

~0.1 s/token (PyTorch: 21-55 ms for the whole forward).

**5/**
Speed, honestly. v1 was 1800x slower than PyTorch on MNIST: matrices as linked lists, 544 s/epoch. I measured every hypothesis (all in NOTES.md):
• flat Array instead of lists: 49x
• parallel blocks: ~2x more
• parallelizing matvec by copying the matrix: 10x WORSE
Now: 6.6 s/epoch vs 0.2-0.3 s. Same loss, same accuracy (9129 -> 9298 -> 9418).

**6/**
Things I hit that might be useful: imports aren't re-exported (a library needs law+proof in one file); match only on parameters; Array reads return the array (affine); `--` for negative CLI args; Base has almost no lemmas. Remaining gap: scalar code vs AVX/BLAS, and Array.clone is O(n).

Happy to help turn this into a proper `llama.bend` starting point. 🙂
