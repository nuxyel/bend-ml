# Draft issue for bendlang/bend: an erased last parameter turns a parallel let into sequential calls

Status: draft, not posted. Renan posts it. Checked for duplicates on 2026-10-05 (searched "erased",
"parallel let", "fork join"): none.

---

**Title:** C backend: a def whose last parameter is erased compiles its parallel let as two sequential calls

## Summary

Adding one erased parameter at the end of a recursive def's parameter list, and passing it unchanged,
makes `bend -o` emit the parallel let `a b = f(..) g(..)` as two sequential calls: `--threads 16` takes
as long as `--threads 1`. The same parameter placed anywhere before a runtime parameter keeps the fork.

## Repro

[`erased_last_ok.bend`](erased_last_ok.bend) and [`erased_last_bad.bend`](erased_last_bad.bend) differ
only in `-z: Nat`, the last parameter of `tree`:

```python
def tree(+d: Nat, +i: U32, -z: Nat) -> U32:
  match d:
    case 0n:
      work(30000000n, i)
    case 1n+q:
      a b = tree(q, (i * 2 : U32), z) tree(q, (i * 2 + 1 : U32), z)
      (a + b : U32)
```

```
bend erased_last_ok.bend -o ok && bend erased_last_bad.bend -o bad
for t in 1 16; do time ./ok --threads $t; time ./bad --threads $t; done
```

| | `--threads 1` | `--threads 16` |
|---|---|---|
| `erased_last_ok.bend` | 0.054 s | 0.011 s (4.9x) |
| `erased_last_bad.bend` | 0.053 s | 0.055 s |

Both print 3847108984.

In the emitted C (`bend X.bend -o x.c`), the good version has a join function for the parallel let
(`FID_TREE_J2`, two child tasks under one join); the bad one has only the continuation frames
`FID_TREE_K2` and `FID_TREE_K3`, so the second call runs after the first returns.

## Where it bit us

[bend-ml](https://github.com/nuxyel/bend-ml)'s `Bands.matvec` carries an erased proof that the run-time row
count equals the one in the type (`-e: {rr == r : Nat}`). With `-e` last, the bands ran one after another;
with a runtime parameter after it, the 50257 × 768 product went from 26.7 ms to 3.3 ms on 16 threads.
Details: `NOTES.md`, experiment 11.

Bend 2.0.35, Linux x86_64, clang 22.1.8, Intel Core Ultra 7 155H.
