# Versioning

bend-ml releases use `MAJOR.MINOR.PATCH`, tagged `vX.Y.Z` on `main` (see "Branches" in `CLAUDE.md`).

| Part | When it changes | Example |
|---|---|---|
| MAJOR | what bend-ml is or can do changes: a new class of capability, or a break in how the packages are used | 1.0: the first stable, reproducible release that matches PyTorch; 2.0 could be llama.bend or a training framework |
| MINOR | a new capability inside the same scope | 1.1: `Bands`, parallel matrix · vector without copies |
| PATCH | fixes, speed-ups without new API, a Bend version bump, documentation, re-measured numbers | 1.2.1: the one-thread fix and the bump to the Bend release that carries bendlang/bend#1377 |

The BendHub packages (`bend-ml-*`) have their own four-number versions (`0.1.6.0`), required by BendHub and
independent of the project's version. A release lists the package versions it uses.

## Renumbering of 2026-10-09

Until 2026-10-06 every large step got a new major, so the project reached "v3.1" in four days without a change
that big. On 2026-10-09 the releases were renumbered with the rules above. The new tags point to the same
commits; the GitHub releases moved to the new tags and say "formerly …". The old tags stay, so links that use
them keep working (the old `v1.0.0` is the only one that moved, because its name is now taken).

| Old | New | What it was |
|---|---|---|
| v0.1.0 | v0.1.0 | `bend-ml-nat-lemmas` on BendHub |
| v1.0.0 | v0.2.0 | shapes in the type, 4 packages, MNIST and GPT-2 small over lists |
| v2.0.0 | v0.3.0 | tensors over a flat `Array` (~50x faster) |
| v2.1.0 | **v1.0.0** | the first stable release: reproducible setup, CI, broader tests, proved capacity bound; the launch |
| v3.0.0 | v1.1.0 | `Bands`: parallel matrix · vector without copying weights |
| v3.1.0 | v1.2.0 | faster loading, `matvec_l`, laws `leaf_len` and `band_len`, `devel`/`main` workflow |

Notes:

- Tools that sort tags as versions will see the old `v3.1.0` as newer than `v1.2.0`; the GitHub "Latest" release
  is the one to follow.
- The old `v2.1.0` tag keeps a pre-release holding only `bend-ml.mp4`, so links posted before the renumbering
  still play the video. The video itself shows the old names (v1, v2).
- Dated entries in `NOTES.md`, the `docs/*-plan.md` files and `bench/results/` keep the names they were written
  with; read them with the table above.
