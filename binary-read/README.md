# bend-ml-binary-read

Reads little-endian `F32` and `BF16` numbers from a file straight into an `Array<F32>`, for loading model weights, at any byte offset of files past 4 GiB. On BendHub as `bend-ml-binary-read@0.2.0.0`; 0.1.0.0 had `read_f32` only.

- Bend: **2.0.39** · License: MIT.
- **Foreign code.** Its six defs are library effects written in C (`read_f32.c`, `read_at.c`) and JavaScript (`read_f32.js`, `read_at.js`). `bend binary-read/main.bend` says so (`6 defs rely on unsafe or foreign code`), and `--verdict` fails for any file that imports it, used or not. Import it only where weights are loaded, never in a file that holds laws; none of the other bend-ml packages imports it, and `make check` checks that.
- Tests, on the C and the JavaScript lanes:
  - `binary-read/tests/read_f32.bend`: a whole read, an offset, the array's capacity, `-0.0`, `+inf`, a subnormal, a read cut short by the end of the file.
  - `binary-read/tests/test_read_at.py` over `read_at.bend`: reads just below and above 2, 4 and 8 GiB of a sparse 8 GiB file (short reads, a hole, past the end), reads into a cell offset with the cells around them left alone, reads clipped by the array's capacity, `size64`, and 10,019 BF16 numbers written by PyTorch, equal to `bf16.to(torch.float32)` bit for bit.

```python
import Base
import bend-ml-binary-read@0.2.0.0/main.bend as R

# n numbers at byte `offset` of the file, into an array of 2^d cells (d with 2^d >= n)
def load(f: File, +offset: U32, +n: U32, +d: Nat) -> IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  R.read_f32(f, offset, n, Array.new(F32, d, 0.0))
```

## Why it exists

`File.read_at` answers a `List<U32>` with one cell per byte, and building and walking that list is what loading weights costs: about 500 million list cells for GPT-2 small's 124 million numbers. `read_f32` writes the file's bytes into the array's own cells instead. On GPT-2 small it takes the load from about 4 s to about 0.6 s, with the same bits in every weight (`NOTES.md`).

The idea and its shape come from bendlang/bend#1430 by @totaland, which proposed a read into an array for Base. Base keeps thin wrappers over system calls, so it lives here as a library effect. I asked for a dense read in Base in bendlang/bend#1573.

## Behavior

`read_f32(file, offset, n, a)` reads `4 * n` bytes at byte `offset` into cells `0 .. n-1` of `a`, stopping at the array's capacity and at the end of the file. It answers the file, the array and the number of bytes read, or the error. The file's position does not move. A short read leaves the cells past it as they were, and a last partial number writes only its bytes over the cell's bits. The C lane refuses to compile on a big-endian host. Its offset is a `U32`, so it reads a file up to its first 4 GiB.

## Past 4 GiB, BF16, and into an existing array (0.2.0.0)

Bend has no 64-bit integer: Base's numbers are `U32`, `F32` and `Nat`, and `File.read_at` takes a `U32` offset. So a 64-bit offset or size crosses as two `U32`s, `hi` and `lo`, meaning `hi * 2^32 + lo`. Offsets are bytes, because a tensor in a safetensors file starts at 8 + the header's length + its own offset, and the header can have any length.

| def | what it does |
|---|---|
| `size64(file)` | the file's size as `(hi, lo)`; Base's `File.size` fails past 4 GiB |
| `read_f32_at(file, hi, lo, n, a)` | as `read_f32`, at a 64-bit offset; answers the number of whole numbers read, so up to 2^32 - 1 numbers (16 GiB) |
| `read_bf16_at(file, hi, lo, n, a)` | `n` BF16 numbers (`2 * n` bytes), each widened exactly to the F32 whose top 16 bits are the BF16's (what `bf16.to(torch.float32)` gives); a last odd byte is dropped |
| `read_f32_into(file, hi, lo, n, a, at)` | as `read_f32_at`, writing cells `at .. at + n - 1` of `a`: one tensor into its place in a larger array, with no copy |
| `read_bf16_into(file, hi, lo, n, a, at)` | the same for BF16 |

Each one stops at the array's capacity and at the end of the file, leaves the cells past what it read as they were, and does not move the file's position.

## When it goes away

When Bend can do the same in plain Bend at the same speed: a dense read in Base (bendlang/bend#1573), with the numbers decoded by `bitcast` as `demos/gpt2/load.bend` already does, or a compiler that removes the byte list. Then each read becomes a def in Bend with the same signature (and the `hi`/`lo` pairs go once Base has 64-bit offsets), its callers do not change, and this test and the GPT-2 bit oracle check the swap.
