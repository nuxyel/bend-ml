# GPT-2 small (124M) in Bend

Inference of GPT-2 small written in Bend: pre-tokenizer, BPE tokenizer with the 50,000 official rules (the `bend-ml-bpe-tokenizer` package, roundtrip proved), 12 transformer layers with a key/value cache, and greedy generation. The matrices use `bend-ml-tensor-array`, with the shape in the type.

Two implementations, with identical output:

- `fast.bend` (v2): matrix · vector products over a flat `Array`. **~0.1 s per token.**
- `gpt2.bend` (v1): matrices as lists, kept as the baseline. ~3 s per token.

## Run (from the repository root)

```bash
reference/.venv/bin/python reference/gpt2_prep.py          # downloads nothing; converts demos/gpt2/data/model.safetensors
reference/.venv/bin/python reference/gpt2_prep.py --split  # per-tensor weights (Conv1D already transposed)

bend demos/gpt2/fast.bend -o /tmp/gpt2_fast
/tmp/gpt2_fast "The capital of France is" 8
```

The weights (`model.safetensors`), `vocab.json` and `merges.txt` come from <https://huggingface.co/openai-community/gpt2>; they go in `demos/gpt2/data/` (outside git).

## Verification

- **Tokenizer**: `reference/test_gpt2_tok.py` compares with `tiktoken` on 16 texts (ASCII, contractions, numbers, spaces and newlines, accents, Cyrillic, CJK): 16 out of 16 identical.
- **Model**: `reference/test_gpt2.py` runs the same model in PyTorch with the same weights; the generated ids and the logits are compared with Bend's (see `BENCHMARK.md`).

## Known limits

- The pre-tokenizer treats every byte >= 128 as a letter. That matches GPT-2's regex for accented letters and other alphabets, but does not split Unicode symbols that are not letters.
- Greedy generation only (no temperature or sampling).
- Memory: the 124 M weights live in trees of nodes (~10 GB during loading with `fast.bend`).
