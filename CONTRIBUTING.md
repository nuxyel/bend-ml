# Contributing to bend-ml

Thanks for looking at bend-ml. I'm Renan Vinícius (@nuxyel) and I maintain it alone, so I read every issue and pull
request myself. Bug reports, numbers from other machines, fixes, new laws and new operations are all welcome.

## Branches

- **Open pull requests against `devel`.** All work happens there; `main` only receives releases, through a pull
  request from `devel`, and the version tags live on `main`. GitHub proposes `main` by default, so change the base
  branch when you open the PR. If you forget, I'll retarget it.
- Branch from `devel` in your fork, not from `main`.

## Before you start

- For anything beyond a small fix (a change to a package's types or laws, a new package, performance work, a new
  dependency), open an issue first so we can agree on the approach before you spend time on it.
- Read [`NOTES.md`](NOTES.md) before touching performance-sensitive code: it records every experiment, including the
  ones that did not pay off. [`docs/AUDIT.md`](docs/AUDIT.md) says what is proved, what is tested and what is
  trusted.
- Bend 2 is young and differs a lot from Bend 1. The "pitfalls" section of [`CLAUDE.md`](CLAUDE.md) lists the traps
  I hit in practice; read it before writing Bend.

## Setup and checks

```bash
make setup        # the pinned Bend 2.0.38, Lean 4.34.0, the Python venv and the data (make setup-lite skips GPT-2)
make check        # the proofs, the kernel (--verdict), the rejected programs, the comparisons with PyTorch and tiktoken
make check-full   # the same plus GPT-2 and MNIST
make laws         # every law statement of every package
```

A pull request should pass `make check`; CI runs `make check-full` on it. If it changes the numbers a package
computes, say why in the PR and show the comparison with PyTorch / NumPy / tiktoken still passes.

## Rules for code

- **Bend stays pinned to 2.0.37.** A version bump is its own change, with an entry in `NOTES.md`.
- **Every proof is real.** No `@unsafe`, no `?TODO`, no axioms in the packages. A law that cannot be proved yet goes
  into an issue, not into the package.
- **Prove structure, test numbers.** Shapes, sizes and composition belong in the types and laws; `F32` results are
  checked against PyTorch / NumPy / tiktoken by the scripts in `reference/`.
- **Every proved law is explained in plain language** in its package's README and listed in
  `reference/list_laws.py`.
- Each package (`tensor/`, `tensor-array/`, `autograd/`, `bpe/`, `nat-lemmas/`) is published to BendHub as one file,
  so it imports only Base and other published packages, at pinned versions. Published versions are permanent: a
  change to a package becomes a new version, which I publish.

## Benchmarks

Performance claims need numbers that someone else can reproduce: the hardware, the Bend and compiler versions, an
idle machine, runs alternated between the old and the new version, and one-thread timings pinned to a single core.
Report the losses and the noise too, not only the best run. A speedup that changes the results is a bug.

## Commits and pull requests

- Small commits, one logical change each, with clear messages in English.
- Keep a pull request to one topic. Several small PRs are easier to review than one large one.
- Don't commit datasets, model weights or large logs.
- Using AI tools is fine. You are still the author: say in the PR what you ran yourself and what you did not, and
  keep the description short enough for a person to read.

## License

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE) of this repository.
