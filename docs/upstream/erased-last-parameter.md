# Draft issue for bendlang/bend: an erased last parameter turns a parallel let into sequential calls

Status: posted as https://github.com/bendlang/bend/issues/1374 (2026-10-06). Fix proposed in
https://github.com/bendlang/bend/pull/1377 (2026-10-07): `anf` cut the call prefix before an erased argument too, so
`tree(q, i)` was split off as a sequential let; now only a live argument cuts it (NOTES.md, "Upstream PR for #1374").
The sibling reports are https://github.com/bendlang/bend/issues/1375 (`io-fork-parallel.md`) and
https://github.com/bendlang/bend/issues/1376 (`shared-readonly-array.md`). Checked for duplicates on 2026-10-06 (searched "erased",
"parallel let", "fork join"): none. Reproduced on 2.0.35 and on main 0ad47fc.

---

**Title:** C backend: a def whose last parameter is erased compiles its parallel let as two sequential calls

## Summary

Adding one erased parameter at the end of a recursive def's parameter list, passed unchanged, makes
`bend -o` emit the parallel let `a b = f(..) g(..)` as two sequential calls: `--threads 16` takes as long
as `--threads 1`. The same parameter placed before a runtime parameter keeps the fork.

## Repro

Three files that differ only in `-z: Nat`:

- `erased_last_ok.bend`: `def tree(+d: Nat, +i: U32)` (no erased parameter)
- `erased_last_bad.bend`: `def tree(+d: Nat, +i: U32, -z: Nat)` (erased parameter last)
- `erased_middle_ok.bend`: `def tree(+d: Nat, -z: Nat, +i: U32)` (erased parameter before `i`)

```python
def tree(+d: Nat, +i: U32, -z: Nat) -> U32:
  match d:
    case 0n:
      work(600000000n, i)
    case 1n+q:
      a b = tree(q, (i * 2 : U32), z) tree(q, (i * 2 + 1 : U32), z)
      (a + b : U32)
```

`work` is a flat loop seeded per leaf; `main` is `tree(4n, 1, 0n)` (16 leaves).

| | `--threads 1` | `--threads 16` |
|---|---|---|
| `erased_last_ok.bend` | 1.01 s | 0.17 s |
| `erased_last_bad.bend` | 1.01 s | 1.02 s |
| `erased_middle_ok.bend` | 1.01 s | 0.17 s |

All three print the same number. The same results with Bend 2.0.35 and with main at 0ad47fc
(`bun bend2/main.ts`).

In the emitted C (`bend X.bend -o x.c`), the good versions have a join function for the parallel let
(`FID_TREE_J2`); the bad one has only the continuation frames `FID_TREE_K2` and `FID_TREE_K3`, so the second
call runs after the first returns.

## Where it bit us

[bend-ml](https://github.com/nuxyel/bend-ml)'s `Bands.matvec` carries an erased proof that the run-time row
count equals the one in the type (`-e: {rr == r : Nat}`). With `-e` last, the bands of a matrix · vector
product ran one after another; with a runtime parameter after it, GPT-2's 50257 × 768 logits product went from
26.7 ms to 3.3 ms on 16 threads. We now check the join in CI (the C must contain the join of `bmv`).

Linux x86_64, clang 22.1.8, Intel Core Ultra 7 155H.
