# Repros for open Bend issues

The programs behind the `bendlang/bend` issues I opened that are still open, so anyone can run them. The issue texts
themselves live in the issues; the programs of closed issues are at the commits their issues link.

| File | Issue |
|---|---|
| `read_at_bytes.bend`, `read_at_bytes.c` | [#1573](https://github.com/bendlang/bend/issues/1573): a file read into an `Array<U32>` |
| `matvec_schedule.bend`, `matvec_schedule.sh` | [#1648](https://github.com/bendlang/bend/issues/1648): a memory-bound loop's speed depends on unrelated code |
| `gpu_array_new_fill.bend` | [#1778](https://github.com/bendlang/bend/issues/1778): `Array.new` inside a bang fills on one thread |

When an issue closes, its files leave this folder.
