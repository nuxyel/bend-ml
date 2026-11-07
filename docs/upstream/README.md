# Repros for open Bend issues

The programs behind the `bendlang/bend` issues I opened that are still open, so anyone can run them. The issue texts
themselves live in the issues; the programs of closed issues are at the commits their issues link.

| File | Issue |
|---|---|
| `read_at_bytes.bend`, `read_at_bytes.c` | [#1573](https://github.com/bendlang/bend/issues/1573): a file read into an `Array<U32>` |
| `matvec_schedule.bend`, `matvec_schedule.sh` | [#1648](https://github.com/bendlang/bend/issues/1648): a memory-bound loop's speed depends on unrelated code |
| `gpu_array_new_fill.bend` | [#1778](https://github.com/bendlang/bend/issues/1778): `Array.new` inside a bang fills on one thread |
| `array_set_laws.bend` | [#26](https://github.com/nuxyel/bend-ml/issues/26), [bendlang/bend#1934](https://github.com/bendlang/bend/issues/1934): `Array.set` keeps `Array.size`, get after set, and a write leaves the other cells alone, proved outside Base (the same proofs are in `tensor-array/main.bend` as `size_set`, `get_after_set` and `get_other_set`) |
| `bpe_encode.bend`, `bpe_encode.sh` | [#2081](https://github.com/bendlang/bend/issues/2081): on the host, Bend 2.0.40 makes every take of a shared node a call (from [#1978](https://github.com/bendlang/bend/pull/1978)); [#2082](https://github.com/bendlang/bend/issues/2082): [#2036](https://github.com/bendlang/bend/pull/2036) gives this program 5.6x the L1 load misses |

When an issue closes, its files leave this folder.
