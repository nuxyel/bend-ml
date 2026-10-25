# Draft comment for bendlang/bend#1843: a second library that took route (a)

Status: posted 2026-10-24 (https://github.com/bendlang/bend/issues/1843#issuecomment-6390291082). #1843 (by ScottMoore0, opened 2026-10-24) proposes `"r+"`,
`File.write_at` and 64-bit offsets and sizes. For the last, its option (a) is two `U32` halves (`File.read_at64`,
`File.size64`), and it cites the last comment on #1027 (Renan's: model weights past 4 GiB). This comment adds one data
point for (a) and nothing else. It does not ask for the dense read again (#1573 has that).

---

One more data point for (a), from the model-weights side that #1027's last comment mentioned.

`bend-ml-binary-read` (a library effect, the route #1573 ended on; 0.2.0.0 on BendHub) needed the same thing to load SmolLM2-1.7B: the
BF16 safetensors file is 3.4 GB, so its tensors start past 2^31, and the F32 copy is 6.8 GB. I added it the way (a)
describes. The offset is two `U32`s, `hi * 2^32 + lo`, and the size answers `(hi, lo)`: `size64(file)`,
`read_f32_at(file, hi, lo, n, a)` and `read_bf16_at(file, hi, lo, n, a)`, which read into an `Array<F32>`. On the C
lane the halves become `(u64)hi << 32 | lo` for `pread`. On the JavaScript lane `hi * 2^32 + lo` is an exact Number
below 2^53, passed to `pread` as a `BigInt`. Neither lane needed a new value type.

Tested on both lanes against Python: 28 reads just below and above 2, 4 and 8 GiB of a sparse 8 GiB file, including
short reads, a hole, reads past the end, and `size64`. Every answer matches the file's bytes.

So two libraries, written separately, ended up with the same boundary and the same `size64` name. If Base adds (a), the
two `U32` halves are what I would keep calling. My reads stay a library because they write into an array, and Base
answers a list.
