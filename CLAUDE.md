# bend-ml

Machine learning in Bend 2 where a shape error is a type error and the structural guarantees are
laws checked by Bend's kernel. Five packages on BendHub (`bend-ml-*`), an MNIST and a GPT-2 small demo,
both matched against PyTorch. Started from Victor Taelin's wishlist "AI framework / PyTorch /
llama.bend" (https://x.com/VictorTaelin/status/2104949395770712276). Two goals: packages other people
use on BendHub, and a portfolio piece for HOC.

## Where things are

- `NOTES.md`: every finding, measurement and discarded hypothesis, by date and experiment number. Read
  the relevant experiment before changing performance-sensitive code.
- `docs/v3.1-plan.md`: the current plan and its stop criterion. Earlier plans: `docs/v3-plan.md`,
  `docs/v2-plan.md`.
- `docs/AUDIT.md`: the trust base and what is proved, tested or trusted.
- `docs/upstream/`: drafts of reports for `bendlang/bend`. Renan posts them.
- Plans after v3 (llama.bend, a training framework) live in Renan's private notes, outside this repository.
- `make check` / `make check-full` (`reference/check_all.py`) is the gate before a commit; CI runs the full
  one on every push to `devel` and `main` and on pull requests.

## Branches

Work on `devel`: every commit and push goes there, and CI checks it. `main` only receives stable releases,
through a pull request `devel` → `main` merged after its CI passes; the version tag (`vX.Y.Z`) and the GitHub
release are made on `main` after that merge. Version numbers follow `docs/VERSIONING.md` (major only for a change in
what bend-ml is; renumbered on 2026-10-09, old tags kept).

## About Renan

Works in AI engineering (agents, NLP) and writes Python and TypeScript; studies ML formally. No background
in type theory: explain each proof in plain language, in a comment and in chat. Machine: Core Ultra 7
155H (22 threads), 32 GB RAM, RTX 4050 6 GB (for our memory-bound kernels the GPU is slower than the CPU,
NOTES exp. 10).

## Rules

1. Bend is pinned to **2.0.39** (and Lean 4.34.0 for `--verdict`). Bend 2 changes fast and differs from
   Bend 1: before writing code, read `bend guide`, `bend base <name>` and the Base source
   (`~/.bend/bend2/base.bend`). A version bump goes in `NOTES.md`.
2. Run the checker on every change; it takes a fraction of a second.
3. Every proof is real: no `@unsafe`, no `?TODO`, no axioms. A law that cannot be proved yet is
   recorded in `NOTES.md` as debt and left out of the README. Two packages stand outside the proofs, by Renan's
   decision, and no package with laws imports them: `binary-read` (foreign code, six reads) and `tensor-fork` (Base's
   `@unsafe` `Array.fork`, all Bend; decided 2026-10-16). `make check` pins the list of defs each one has outside the
   proofs, so it cannot grow unnoticed. A third, approved by Renan on 2026-10-18: `tensor-gpu`
   (tensor-array's products under `!`, Base's `@unsafe` `Array.fork`/`Array.join`, all Bend), on the same terms: no
   package with laws imports it, `make check` pins its defs outside the proofs, and it gives tensor-array's bits. A new exception is Renan's call.
4. Prove structure (shapes, sizes, composition); test numbers. `F32` is not a real number, so numerics
   are validated against PyTorch / NumPy / tiktoken by the scripts in `reference/`.
5. Every proved law appears in its package README in plain language, and in `reference/list_laws.py`.
6. Publishing to BendHub is public and permanent: four-number versions (`0.1.4.0`), names of 12+
   characters, a `LICENSE` next to the entry file. After publishing, point the demos at the new version
   and add it to `reference/test_published.py`.
7. Benchmarks are honest: hardware and versions written down, losses shown, numbers re-measured on an
   idle machine before they go into the README.
8. Everything in the repository is in English. Commits are small, with clear messages and no
   attribution lines (no `Co-Authored-By`, no "Generated with").
9. Public text (READMEs, `NOTES.md`, `docs/`, release notes) is written in Renan's first person singular:
   "I measured", "my PR", never "we"/"our". Where a name is needed: "Renan Vinícius (@nuxyel)". This file
   stays in the third person: it tells agents whose decisions are whose. Drafts already posted upstream and
   dated entries are left as written.

## Bend 2.0.39 pitfalls we paid for

- A compiled program's runtime options go between `+RTS` and `-RTS` (since 2.0.38): `./x +RTS --threads 16 -RTS`.
  A bare `./x --threads 16` passes `--threads 16` to the program and runs on the default thread count with no error,
  so a thread-count test silently tests nothing. `--gpu off` goes in the same bracket.
  Three guards (2026-10-23): `scripts/check_rts.py` (in the checks, with `rts-allow.txt`) fails on one written outside
  the bracket; test programs that take no arguments refuse any; `scripts/rts_probe.py` checks, locally, that 1 and 16
  threads use 1 and several cores.
- `match` only inspects a parameter or a pattern variable; a computed value (including a destructuring
  `let` of a call) needs its own def or a parameter computed by the caller.
- No mutual recursion. Termination reads arguments left to right: each must be passed unchanged until one
  shrinks. Erased (`-x`) parameters are skipped, so put a changing size after the shrinking argument or
  make it erased.
- Variables are affine; `+x` reuses a `Data` value. A parameter used in a type counts as a use.
- `%e : P` rewrites: `_` marks where the right-hand side of `e` sits in the current goal, and the new goal
  has the left-hand side there; use `Equal.sym` to go the other way.
- Parallel lets (`a b = f(..) g(..)`) suffer from non-tail recursive helpers in the fork tree (use tail
  loops with an accumulator). Check scaling with `+RTS --threads 1` against `+RTS --threads 16`, and read the emitted
  C (`bend X.bend -o x.c`) when it does not scale. (Up to 2.0.36 a parallel let silently became sequential
  when the def's last parameter was erased, NOTES exp. 11; fixed in 2.0.37 by bendlang/bend#1377.)
- On the GPU a lane has a 2,048-word stack: a non-tail recursion inside a `!` faults past it ("memory fault (machine
  stack overflow?)", e.g. `List.append` over 1,023 elements) while `+RTS --gpu off` runs; use tail loops in bangs (bendlang/bend
  #1393). Every device loop also polls for errors, which costs 4-20x on loops with little work a step (a draft issue
  in bend-ml's `docs/upstream/gpu-error-poll.md`); loops that read memory pay 1.2-1.9x.
- On CUDA a bang deals a balanced tree's leaves to threads in a run-time order: each block of 128 threads gets one
  aligned range of 128 leaves, but inside it the order changes from run to run, so "leaf k on lane k" layouts read
  uncoalesced (16 sectors a warp instead of 4). Check layouts with `ncu` (sectors per request); a
  renumbering of the low 7 bits helps a 2^14 tree (issue draft `docs/upstream/gpu-leaf-order.md`).
- `IO.fork` gives concurrency, not parallelism: forked computations take turns on one event loop
  (NOTES exp. 12). This is intended; the guide says so since bendlang/bend#1415. Use parallel lets in pure
  code for several cores.
- Proofs follow the affine rules too: a lemma about values that hold an `Array` cannot mention them twice at
  run-time positions; pass the expression in an erased argument (`-s`) and the arrays once.
- `Array` is a binary tree in Base (`ALeaf`/`ANode`) but one flat block in the compiled program: `get`/`set` are
  an index mask and a load or store (no descent), while matching an array copies each half into its own block
  (BendRT, "Arrays"), so never walk an array by `match`; `Array.clone` is O(n); `Array.fork` is `@unsafe`.
  Indexes wrap around, so never write past a capacity. Corrected 2026-10-13 (NOTES).
- An `import ... as M` does not re-export. A BendHub package can hold several files (users import each one as
  `pkg@ver/file.bend`; NOTES 2026-10-10, lead 5), but then every caller must import every file it uses.
  `--publish` sends the entry file and everything it imports (store packages stay out), so a file nothing imports
  needs an entry that imports it; the entry may have any name. Relative imports inside a published package
  (`./x.bend`, `../d/x.bend`) resolve; a hub file may import only hub files. Paths: segments `[A-Za-z_][A-Za-z0-9_-]*`
  (no dots); names `[a-z][a-z0-9-]{0,63}@a.b.c.d` (checked in Bend 2.0.37's source and against a published
  multi-file package, 2026-10-19).
- A template (`~l`) is substituted at each use, so a record of functions passed as one is built again wherever it is
  used, and its plain-value fields are computed then. A composite whose value field reads its children
  (`count: count(a) + count(b)`) builds their whole records at every use and drops them: +5% instructions in a
  two-layer model (2026-10-17). Make such fields closures (`count: Unit -> U32`).

## Agent skills

### Issue tracker

Issues live in GitHub Issues (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
