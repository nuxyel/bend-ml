# bend-ml-binary-read

Reads little-endian `F32` numbers from a file straight into an `Array<F32>`, for loading model weights. On BendHub as `bend-ml-binary-read@0.1.0.0`.

- Bend: **2.0.38** · License: MIT.
- **Foreign code.** `read_f32` is a library effect written in C (`read_f32.c`) and JavaScript (`read_f32.js`). `bend binary-read/main.bend` says so (`1 def relies on unsafe or foreign code: read_f32`), and `--verdict` fails for any file that imports it, used or not. Import it only where weights are loaded, never in a file that holds laws; none of the other bend-ml packages imports it, and `make check` checks that.
- Test: `binary-read/tests/read_f32.bend`, the same lines on the C and the JavaScript lanes (a whole read, an offset, the array's capacity, `-0.0`, `+inf`, a subnormal, a read cut short by the end of the file).

```python
import Base
import bend-ml-binary-read@0.1.0.0/main.bend as R

# n numbers at byte `offset` of the file, into an array of 2^d cells (d with 2^d >= n)
def load(f: File, +offset: U32, +n: U32, +d: Nat) -> IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  R.read_f32(f, offset, n, Array.new(F32, d, 0.0))
```

## Why it exists

`File.read_at` answers a `List<U32>` with one cell per byte, and building and walking that list is what loading weights costs: about 500 million list cells for GPT-2 small's 124 million numbers. `read_f32` writes the file's bytes into the array's own cells instead. On GPT-2 small it takes the load from about 4 s to about 0.6 s, with the same bits in every weight (`NOTES.md`).

The idea and its shape come from bendlang/bend#1430 by @totaland, which proposed a read into an array for Base. Base keeps thin wrappers over system calls, so it lives here as a library effect. I asked for a dense read in Base in bendlang/bend#1573.

## Behavior

`read_f32(file, offset, n, a)` reads `4 * n` bytes at byte `offset` into cells `0 .. n-1` of `a`, stopping at the array's capacity and at the end of the file. It answers the file, the array and the number of bytes read, or the error. The file's position does not move. A short read leaves the cells past it as they were, and a last partial number writes only its bytes over the cell's bits. The C lane refuses to compile on a big-endian host. The offset is a `U32`, so a file is read up to its first 4 GB.

## When it goes away

When Bend can do the same in plain Bend at the same speed: a dense read in Base (bendlang/bend#1573), with the numbers decoded by `bitcast` as `demos/gpt2/load.bend` already does, or a compiler that removes the byte list. Then `read_f32` becomes a def in Bend with the same signature, its callers do not change, and this test and the GPT-2 bit oracle check the swap.
