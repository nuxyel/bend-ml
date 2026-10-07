# Draft for bendlang/bend: IO.fork does not run pure work on several cores

Status: posted as https://github.com/bendlang/bend/issues/1375 (2026-10-06). Checked for duplicates on 2026-10-06: none found.
Reproduced on 2.0.35 and on main 0ad47fc.

---

**Title:** Question: should computations started with IO.fork run their pure work on different cores?

## What we see

Four `IO.fork` computations, each reading the clock inside the fork and then running a flat loop seeded
with it (so the work can only happen inside the forked computation), joined at the end:

| | `--threads 1` | `--threads 16` |
|---|---|---|
| `io_fork_parallel.bend`: 4 `IO.fork`, then 4 `IO.join` | 1.68 s | 1.68 s |
| `io_fork_sequential.bend`: the same 4 jobs one after another | 1.68 s | 1.68 s |
| `io_fork_parallel_control.bend`: the same loops as two parallel lets of two calls | 1.69 s | 0.85 s |

User time equals real time in the first two rows: the forked computations run on one core, one after
another. The same on Bend 2.0.35 and on main at 0ad47fc.

## Why we ask

The guide (IO and Concurrency) says: "A Bend program is a set of computations interleaved by one event
loop, as in Node.js: each runs its pure code (in parallel, on every core) up to its next effect". We read it
as "forked computations run their pure parts at the same time on different cores", and planned around it:
loading GPT-2's 12 layers with one `IO.fork` each was slower than loading them in sequence (6.6-7.1 s against
5.1-5.6 s), the same with 1 and 16 threads.

If the intended meaning is that each computation's own pure code can use parallel lets, while computations
take turns on the event loop, a sentence in the guide saying so would have saved us the experiment. If
forked computations are meant to overlap, the repro above shows they do not.
