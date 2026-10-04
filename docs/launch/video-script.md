# Video script (~60 s)

It shows three things: the **shape error caught by the compiler**, the **verified proof** and **GPT-2 generating text in Bend**. Terminal screen, large font, no heavy editing.

## Preparation (once)

```bash
cd ~/Repos/bend2-ml
export PATH="$HOME/.bend/bin:$HOME/.elan/bin:$PATH"
bend demos/gpt2/fast.bend -o /tmp/gpt2_fast     # compile before recording
clear
```

To record the screen: Omarchy's capture shortcut, or `wf-recorder -f bend-ml.mp4`.

## Scenes

| Time | On screen | Voice / caption |
|---|---|---|
| 0–5 s | `cat tensor/tests/bad_matmul.bend` | "Multiplying (2×3) by (4×5)." |
| 5–15 s | `bend tensor/tests/bad_matmul.bend` → `expected Mat<3n,5n> / observed Mat<4n,5n>` | "In Bend, that is a **type** error: it does not compile." |
| 15–22 s | `bend tensor/tests/bad_reshape.bend` → `expected 12n / observed 15n` | "Reshape only compiles with the proof that the number of elements does not change." |
| 22–32 s | `bend bpe/main.bend --verdict` → `ALL PROOFS CHECK`; show `law roundtrip` | "The tokenizer has a proved roundtrip: decode(encode(s)) = s, verified by the kernel." |
| 32–50 s | `/tmp/gpt2_fast "The capital of France is" 8` (each token takes ~0.1 s; speed up only the 9 s of loading) → `The capital of France is the capital of the French Republic, and` | "GPT-2 small, 124 million parameters, entirely in Bend, the same tokens as PyTorch, ~0.1 s per token." |
| 50–60 s | `reference/.venv/bin/python reference/check_all.py` → `TOTAL: 27/27 checks ok` | "Five packages on BendHub, MIT. Link in the thread." |

## Tips

- GPT-2 takes ~9 s to load and ~0.1 s per token: speed up only the loading and say so in the caption ("sped up").
- To show MNIST: `demos/mnist/run.sh 1 0 0.1` trains one full epoch in ~7 s in Bend and ~0.3 s in PyTorch (be honest in the video!); both print `9129/10000`.
