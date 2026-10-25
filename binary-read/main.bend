# bend-ml-binary-read: little-endian F32 numbers from a file, read straight into an Array<F32>.
#
#   import bend-ml-binary-read@0.2.0.0/main.bend as R
#
# Since 0.2.0.0 it also reads at 64-bit byte offsets, reads BF16 files (each number widened exactly to F32), reads into
# a cell offset of an existing array, and gives a file's size past 4 GiB. Bend has no 64-bit integer (Base's numbers
# are U32, F32 and Nat, and File.read_at takes a U32 offset), so a 64-bit offset or size crosses as two U32s, hi and
# lo, meaning hi * 2^32 + lo. The offsets are bytes, not cells: a tensor in a safetensors file starts at 8 + the
# header's length + its own offset, and the header has any length.
#
# File.read_at answers a List<U32> with one cell per byte, and building and walking that list is
# what loading model weights costs (NOTES.md, exp. 13; bendlang/bend#1573). This effect reads the
# bytes into the array's own cells instead, with no list in between. The idea and its shape come
# from bendlang/bend#1430 by @totaland, which proposed it for Base; Base keeps thin syscall
# wrappers, so it lives here as a library effect.
#
# It is foreign code: --verdict fails for any file that imports this one, used or not. Keep it out
# of files that hold laws, and import it only where weights are loaded.
import Base

# Reads n numbers (4 * n bytes) at byte `offset` of `file` into cells 0 .. n-1 of `a`, stopping at
# a's capacity and at the end of the file. Answers the file, the array and the number of bytes
# read, or the error; the file's position does not move. A short read leaves the cells past it as
# they were (a last partial number writes only its bytes).
def read_f32(file: File, offset: U32, n: U32, a: Array<F32>) ->
  IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  import "./read_f32.c"
  import "./read_f32.js"

# The size of `file` in bytes as (hi, lo): hi * 2^32 + lo. Base's File.size fails past 4 GiB.
def size64(file: File) ->
  IO(File & Result<&1, &1, U32 & String, U32 & U32>):
  import "./read_at.c"
  import "./read_at.js"

# As read_f32, at the byte offset hi * 2^32 + lo, and it answers the number of whole numbers read (not bytes), so a
# read of up to 2^32 - 1 numbers (16 GiB) answers its count in a U32. Reads n numbers into cells 0 .. n-1 of `a`,
# stopping at a's capacity and at the end of the file; the file's position does not move. A short read leaves the
# cells past it as they were, except that a last partial number writes only its bytes, as in read_f32.
def read_f32_at(file: File, hi: U32, lo: U32, n: U32, a: Array<F32>) ->
  IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  import "./read_at.c"
  import "./read_at.js"

# n little-endian BF16 numbers (2 * n bytes) at the byte offset hi * 2^32 + lo, each widened exactly into a cell of
# `a`: the F32 whose top 16 bits are the BF16's and whose low 16 bits are zero (what bf16.to(torch.float32) gives).
# Stops at a's capacity and at the end of the file, answers the number of whole numbers read, and leaves the cells
# past them as they were (a last odd byte is dropped). The file's position does not move.
def read_bf16_at(file: File, hi: U32, lo: U32, n: U32, a: Array<F32>) ->
  IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  import "./read_at.c"
  import "./read_at.js"

# As read_f32_at, writing cells at .. at + n - 1 of `a` (stopping at a's capacity): one tensor of a file into its place
# in a larger array, with no copy.
def read_f32_into(file: File, hi: U32, lo: U32, n: U32, a: Array<F32>, at: U32) ->
  IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  import "./read_at.c"
  import "./read_at.js"

# As read_bf16_at, writing cells at .. at + n - 1 of `a` (stopping at a's capacity).
def read_bf16_into(file: File, hi: U32, lo: U32, n: U32, a: Array<F32>, at: U32) ->
  IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  import "./read_at.c"
  import "./read_at.js"
