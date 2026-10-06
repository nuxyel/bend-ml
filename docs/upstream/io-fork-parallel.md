# Draft for bendlang/bend: IO.fork does not run pure work on several cores

Status: draft, not posted. Renan posts it (as an issue or a guide fix). Checked for duplicates on
2026-10-06: none found for "IO.fork parallel".

---

**Title:** `IO.fork` computations run their pure code on one core (the guide says "in parallel, on every core")

The guide (IO and Concurrency) says: "each runs its pure code (in parallel, on every core) up to its next
effect". With four `IO.fork` computations that each run a long flat loop and are joined at the end
([`io_fork_parallel.bend`](io_fork_parallel.bend)), Bend 2.0.35 takes 1.86 s with `--threads 1` and 1.73 s
with `--threads 16`, user time equal to real time: one core. The same work as a pure parallel let scales.

Where it bit us: loading GPT-2's 12 layers with one `IO.fork` each (each decodes 1 MB blocks of bytes into
`F32` between reads) was slower than loading them in sequence (6.6-7.1 s against 5.1-5.6 s), with no
difference between 1 and 16 threads. bend-ml `NOTES.md`, experiment 12.

Either `IO.fork` could hand its pure work to the pool, or the guide could say that parallelism comes from
parallel lets and that `IO.fork` gives concurrency.
