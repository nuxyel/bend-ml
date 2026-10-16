# Draft comment for bendlang/bend#1573: the library route, measured

Status: posted on #1573 (2026-10-16, https://github.com/bendlang/bend/issues/1573#issuecomment-6225576863), with Renan's go-ahead. It follows up Renan's comment
of 2026-10-15 on #1573, which asked which shape a dense read would take in Base, and nicolas-abril's note closing #1430
(2026-10-09): not in Base, "a library can do exactly the same thing", and a Base-shaped fix would be "a general byte
primitive rather than an effect that fills one array type", discussed in an issue first. Nothing here asks Base to take
or return an `Array`.

---

A data point while the shape is open: I took the library route #1430's review suggested. `bend-ml-binary-read` (on
BendHub, `0.1.0.0`) is one effect with a `.c` and a `.js` side, `read_f32(file, offset, n, a)`, which reads F32 words
straight into an `Array<F32>`, kept out of every package with laws.

What it changed, on an idle Core Ultra 7 155H, alternated runs, the same outputs bit for bit:

| | `File.read_at`, 1 MB blocks | the library effect |
|---|---|---|
| GPT-2 small weights loaded, 1 thread | 4.7 s | 0.8 s |
| SmolLM2-1.7B, load and one token, 16 threads | 72.2 s | 9.1 s |

So the list is most of the cost of loading a model, as the issue says. The library works, with the trade-off the review
named: its C side uses `blk_loc` and `blk_cls`, which have no ABI promise, so it has to follow the runtime. That is the
part a Base primitive would remove for everyone who loads binary data.

The question from my last comment is still the one I'd like to settle before writing anything: within the types Base
effects use, would a read that packs four little-endian bytes per `U32` (the same `List`, a quarter of the cells; `F32`
decoded with one `bitcast` per cell) fit, or would you rather have a dedicated byte value? I can prepare either, with
tests on both lanes.
