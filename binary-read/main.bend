# bend-ml-binary-read: little-endian F32 numbers from a file, read straight into an Array<F32>.
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
