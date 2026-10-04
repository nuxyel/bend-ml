# bend-ml-bpe-tokenizer

**Byte-level BPE** tokenizer (GPT-2 style) in Bend 2, with the **roundtrip proved** by the kernel: `decode(encode(s)) == s`.

- Bend: **2.0.35** · License: MIT · Depends on `bend-ml-nat-lemmas@0.1.0.0`.
- `bend bpe/main.bend` and `bend bpe/main.bend --verdict` → `ALL PROOFS CHECK` (no `@unsafe`, no `?TODO`).

## Model

- **Token**: `B{n}` (raw byte `n`) or `M{k}` (created by the rule with id `k`). GPT-2's numeric ids are `n` for bytes and `256 + k` for merges.
- **Rule**: `Rule{id, a, b}` merges the token pair `(a, b)` into `M{id}`.
- **Table**: a list of rules from the **oldest to the newest**, like GPT-2's `merges.txt`.
- `encode(table, tokens)` applies the rules in order (each one merges, left to right, the non-overlapping occurrences of the pair).
- `decode(table, tokens)` expands each token back into bytes, looking only at the rules older than it.
- `train(n, 0n, tokens)` learns up to `n` merges (tie-break: the first pair that appeared).

```python
import Base
import bend-ml-bpe-tokenizer@0.1.2.0/main.bend as BPE

# BPE.encode(table, BPE.lift(bytes))   BPE.decode(table, ids)   BPE.train(30n, 0n, BPE.lift(bytes))
```

## Proved LAWS (in plain language)

| Law | What it states |
|---|---|
| `roundtrip` | If the table is **well formed** (no rule id repeats: `wf(table) = True`), then `decode(encode(bytes)) = bytes`, for **any** byte sequence. |
| `vocab_bound` | Every token that `encode` emits is a raw byte or was created by a rule of the table; an unknown id never appears. |
| `dec_append` | Decoding two token lists together is decoding each one and joining the bytes. |
| `train_wf` | The table that `train` returns is **always well formed** (the rule ids are `next, next+1, ...`, none repeats), for any corpus and any number of merges. |
| `roundtrip_trained` | A consequence of the first and fourth laws: train a table on **any corpus** and `decode(encode(s)) = s` for any `s`, **with no hypothesis at all**. |

The idea of the roundtrip proof (the comments in `main.bend` give the details): when rule `k` is applied, each pair `(a, b)` becomes `M{k}`, and `M{k}` expands to `expansion(a) ++ expansion(b)`; so decoding does not change. This only holds if `k` did not already appear in the list, which is exactly what a well-formed table guarantees (distinct ids). The auxiliary lemmas prove that adding the new rule on top of the table does not change the expansion of the tokens that already existed.

## Tests (against Python)

`reference/test_bpe.py` runs the Bend CLI (`bpe/cli.bend`) against `reference/bpe_ref.py` (same algorithm and tie-break), with English text, accented text (`coração`, `ação`) and UTF-8 emoji/CJK: `train`, `encode`, `decode` and the roundtrip match in every case. To run it:

```bash
reference/.venv/bin/python reference/test_bpe.py
```

## Known limits

- The *choice* of merges by `train` (which pair is the most frequent) is not proved, and does not need to be: the roundtrip's correctness does not depend on it (`train_wf` + `roundtrip`). Tables read from a `merges.txt` with sequential ids also satisfy `wf`, but that reading (the CLI's `rules.go`) is not proved.
- The lookup of the lowest-rank pair is done by applying the rules in order (equivalent to GPT-2's algorithm for trained tables), at a cost of `rules × size`. For GPT-2's full vocabulary (50 thousand rules) it is used per word, after the pre-tokenizer.
- The package does not include GPT-2's regex pre-tokenizer; the GPT-2 demo implements it (`demos/gpt2/tok.bend`).

## Versions

- `0.1.2.0`: the same laws, with English comments and README.
- `0.1.1.0`: adds `train_wf` and `roundtrip_trained`.
- `0.1.0.0`: first publication (`roundtrip`, `vocab_bound`, `dec_append`).
