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
- Plans after v3 (llama.bend, a training framework) live in the private repo `nuxyel/bend2-notes`.
- `make check` / `make check-full` (`reference/check_all.py`) is the gate before a commit; CI runs the full
  one on every push to `devel` and `main` and on pull requests.

## Branches

Work on `devel`: every commit and push goes there, and CI checks it. `main` only receives stable releases,
through a pull request `devel` → `main` merged after its CI passes; the version tag (`vX.Y.Z`) and the GitHub
release are made on `main` after that merge.

## About Renan

Works in AI engineering (agents, NLP) and writes Python and TypeScript; studies ML formally. No background
in type theory: explain each proof in plain language, in a comment and in chat. Machine: Core Ultra 7
155H (22 threads), 32 GB RAM, RTX 4050 6 GB (for our memory-bound kernels the GPU is slower than the CPU,
NOTES exp. 10).

## Rules

1. Bend is pinned to **2.0.35** (and Lean 4.34.0 for `--verdict`). Bend 2 changes fast and differs from
   Bend 1: before writing code, read `bend guide`, `bend base <name>` and the Base source
   (`~/.bend/bend2/base.bend`). A version bump goes in `NOTES.md`.
2. Run the checker on every change; it takes a fraction of a second.
3. Every proof is real: no `@unsafe`, no `?TODO`, no axioms. A law that cannot be proved yet is
   recorded in `NOTES.md` as debt and left out of the README.
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

## Bend 2.0.35 pitfalls we paid for

- `match` only inspects a parameter or a pattern variable; a computed value (including a destructuring
  `let` of a call) needs its own def or a parameter computed by the caller.
- No mutual recursion. Termination reads arguments left to right: each must be passed unchanged until one
  shrinks. Erased (`-x`) parameters are skipped, so put a changing size after the shrinking argument or
  make it erased.
- Variables are affine; `+x` reuses a `Data` value. A parameter used in a type counts as a use.
- `%e : P` rewrites: `_` marks where the right-hand side of `e` sits in the current goal, and the new goal
  has the left-hand side there; use `Equal.sym` to go the other way.
- Parallel lets (`a b = f(..) g(..)`) silently become sequential when the def's **last parameter is
  erased** (put a runtime parameter after it; fixed upstream by bendlang/bend#1377, not in a release
  yet), and suffer from non-tail recursive helpers in the fork tree (use tail loops with an accumulator). Check scaling with `--threads 1` against `--threads 16`, and
  read the emitted C (`bend X.bend -o x.c`) when it does not scale. NOTES exp. 11.
- `IO.fork` gives concurrency, not parallelism: forked computations take turns on one event loop
  (NOTES exp. 12). This is intended; the guide says so since bendlang/bend#1415. Use parallel lets in pure
  code for several cores.
- Proofs follow the affine rules too: a lemma about values that hold an `Array` cannot mention them twice at
  run-time positions; pass the expression in an erased argument (`-s`) and the arrays once.
- `Array` is a plain ADT (`ALeaf`/`ANode`): matching an owned array hands out its halves in O(1) and is
  safe; `Array.clone` is O(n); `Array.fork` is `@unsafe`. Indexes wrap around, so never write past a
  capacity.
- An `import ... as M` does not re-export, so a publishable package is one file.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
