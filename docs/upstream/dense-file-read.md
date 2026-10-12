# Draft issue for bendlang/bend: reading a binary file without one list cell per byte

Status: draft, not posted (2026-10-12). Checked for duplicates on 2026-10-12: searches for `read_at`, `read_bytes`,
"File read Array", "binary file" and "bytes U32 list" found only #578 (user-friendly file IO, closed), which is about
text. Renan posts it himself. Numbers from llama.bend (a larger model, slower to load) stay out until that repository
is public.

---

**Title:** A file read into an `Array<U32>` (or 4 bytes per `U32`), so loading model weights is not one list cell per byte

## Summary

`File.read_at` and `File.read_bytes` return a `List<U32>` with one cell per byte. Loading a neural network's weights
means reading hundreds of megabytes of little-endian `F32`, and in [bend-ml](https://github.com/nuxyel/bend-ml)
almost all of the loading time goes to building and walking those lists, not to decoding the numbers or writing them
into arrays. A read that returns denser data would remove that cost: either straight into an `Array<U32>`, or a list
with four bytes per `U32`.

## What I measured

bend-ml's GPT-2 small demo (`demos/gpt2/fast.bend`, 124 M parameters, about 500 MB of `F32`), Bend 2.0.35, Intel
Core Ultra 7 155H, load time printed by the demo, alternated runs (bend-ml `NOTES.md`, "Exp. 13"):

| loader | load time |
|---|---|
| the current loader: read 1 MB blocks with `File.read_at`, decode 4 bytes into an `F32`, `Array.set` | 4.1-5.1 s |
| the same reads, walking each block 4 bytes at a time **without** building an `F32` or writing the array | 4.2-5.0 s |
| the same reads, dropping each block's list unread | 11.3-13.8 s |
| reading every block first, then decoding in parallel lets (two-phase) | 4.7-5.9 s, twice the memory |

Building the numbers and writing the arrays cost next to nothing: the time is producing the per-byte list (one cons
per byte, in `io_list` of the C effect) and walking it. Erasing a large unread list costs more than walking it.
Parallel decoding does not help because the walk it can split is a small part of the time (and `IO.fork` does not
spread pure work over cores, #1375 / #1415). For comparison, in a later run on the same machine (Bend 2.0.36, 16
threads), loading GPT-2 and generating one token took 4.45 s in bend-ml, 1.16 s in PyTorch and 0.16 s in llama.cpp
(F32 GGUF).

## What would help

Any of these, in order of how much they would change:

1. `File.read_at_u32(file, offset, count) -> Array<U32>`: `count` little-endian words read straight into an array
   (or into an existing one, at an index). Decoding `F32` is then one `bitcast` per element, and a loader that
   proves its sizes can keep its laws.
2. `File.read_at` returning `List<U32>` with four bytes packed per element: a quarter of the cells, the same
   interface shape.
3. A way to map a file read-only into an `Array<U32>` (like `mmap`), for programs that only read their weights.

## Repro

The loader is `fill_bytes` / `stream_exact` in `demos/gpt2/fast.bend` of bend-ml: `File.read_at` in 1 MB blocks,
four bytes decoded into an `F32` and written with `Array.set`. The variants in the table are small edits of
`fill_bytes`: walking the block four bytes at a time without decoding or writing, and dropping the block unread.
The demo prints its load time.
