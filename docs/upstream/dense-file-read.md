# Draft issue for bendlang/bend: reading a binary file without one list cell per byte

Status: draft, not posted. Written 2026-10-12; updated 2026-10-14 with a standalone repro on Bend 2.0.37 and
llama.bend's numbers (public since 2026-10-13). Checked for duplicates on 2026-10-12 and again on 2026-10-14: searches
for `read_at`, `read_bytes`, "File read Array", "binary file" and "bytes U32 list" found only #578 (user-friendly file
IO, closed), which is about text. On main `014ccd03` (2026-10-14), `File.read_at` and `File.read_bytes` still build
their result with `io_list` (`bend2/effs/file_read.c`). Renan posts it himself.

---

**Title:** A file read into an `Array<U32>` (or 4 bytes per `U32`), so loading model weights is not one list cell per byte

## Summary

`File.read_at` and `File.read_bytes` return a `List<U32>` with one cell per byte. Loading a neural network's weights
means reading hundreds of megabytes of little-endian `F32`, and in [bend-ml](https://github.com/nuxyel/bend-ml)
almost all of the loading time goes to building and walking those lists, not to decoding the numbers or writing them
into arrays. A read that returns denser data would remove that cost: either straight into an `Array<U32>`, or a list
with four bytes per `U32`.

The cost shows up without any model: reading a 384 MiB file with `File.read_at` and adding up its bytes takes 1.5 s
in Bend, against 57 ms for the same loop in C (repro below).

## What I measured

### A standalone repro (Bend 2.0.37)

`read_at_bytes.bend` reads a 384 MiB file of random bytes in 1 MiB blocks with `File.read_at` and adds up the bytes;
`read_at_bytes.c` does the same with `pread`. Intel Core Ultra 7 155H, Linux, one thread pinned to a P-core
(`taskset -c 3`), five alternated runs each, load average 1.0-1.7 (not fully idle). Both print the same sum.

| program | time |
|---|---|
| Bend 2.0.37, `File.read_at` into `List<U32>`, walked once | 1.52-1.57 s |
| C, `pread` into a buffer, walked once (`clang -O2`) | 56-58 ms |

About 27x. The work per byte is one addition in both; the difference is the list, 384 M cells built by the host and
walked by the program.

### Model loading

[llama.bend](https://github.com/nuxyel/llama.bend) runs SmolLM2 135M, 360M and 1.7B with the same loader pattern.
Wall time of a run that loads the model and generates one token, 16 threads, Bend 2.0.37 (llama.bend
`BENCHMARK.md`):

| system | 135M | 360M | 1.7B (6.8 GB of `F32`) |
|---|---|---|---|
| llama.bend | 4.4 s | 11.5 s | 66.4 s |
| transformers (PyTorch) | 2.7 s | 2.7 s | 4.1 s |
| llama.cpp (F32 GGUF) | 0.2 s | 0.2 s | 0.5 s |

Generation itself is 77 ms per token for the 135M and about 430 ms for the 1.7B, so for the 1.7B, loading takes as
long as about 150 tokens.

### Where the time goes (bend-ml, Bend 2.0.35)

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

`read_at_bytes.bend` and `read_at_bytes.c`:

```
head -c 402653184 /dev/urandom > w.bin
bend read_at_bytes.bend -o read_at_bytes && taskset -c 3 ./read_at_bytes
clang -O2 read_at_bytes.c -o read_at_bytes_c && taskset -c 3 ./read_at_bytes_c
```

```python
import Base

# Reads w.bin (384 MiB; make it with `head -c 402653184 /dev/urandom > w.bin`) in 1 MiB blocks with
# File.read_at and adds up its bytes. Each block arrives as a List<U32> of 2^20 cells, one per byte.
# The same loop in C is read_at_bytes.c; both print the same sum.

def walk(bs: List<&2, U32>, acc: U32) -> U32:
  match bs:
    case Nil{}:
      acc
    case Con{b, t}:
      walk(t, (acc + b : U32))

def loop(k: Nat, +off: U32, m: File & Result<&1, &1, U32 & String, List<&2, U32>>, acc: U32) -> IO(U32):
  match k m:
    case 0n Tuple{f, r}:
      do IO<U32>:
        File.close(f)
        return acc
    case 1n+p Tuple{f, r}:
      do IO<U32>:
        bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
        m2 : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, (off + 1048576 : U32), 1048576)
        loop(p, (off + 1048576 : U32), m2, walk(bs, acc))

def main() -> IO(Unit):
  do IO<Unit>:
    t0 : Nat <- IO.now()
    f : File <- IO.try(File, File.open("w.bin", "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, 0, 1048576)
    s : U32 <- loop(384n, 0, m, 0)
    t1 : Nat <- IO.now()
    IO.print("sum " ++ U32.show(s) ++ "  ms " ++ Nat.show(Nat.sub(t1, t0)))
```

```c
// The same work as read_at_bytes.bend in C: pread 1 MiB blocks of w.bin and add up the bytes.
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
int main(void) {
  struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
  int fd = open("w.bin", O_RDONLY); unsigned char *buf = malloc(1 << 20); uint32_t s = 0;
  for (off_t off = 0; off < 384L << 20; off += 1 << 20) {
    ssize_t n = pread(fd, buf, 1 << 20, off);
    for (ssize_t i = 0; i < n; i++) s += buf[i];
  }
  close(fd); clock_gettime(CLOCK_MONOTONIC, &b);
  printf("sum %u  ms %ld\n", s, (b.tv_sec - a.tv_sec) * 1000 + (b.tv_nsec - a.tv_nsec) / 1000000);
}
```

The GPT-2 variants in the table above are small edits of `fill_bytes` / `stream_exact` in bend-ml's
`demos/gpt2/fast.bend`: walking each block four bytes at a time without decoding or writing, and dropping the block
unread. The demo prints its load time.
