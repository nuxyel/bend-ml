# NOTES — Phase 0 (reconnaissance)

Date: 2026-10-03. Everything below was verified on the machine or in a primary source (the installed guide, `base.bend`, the official repository, the site). What was **not** verified is marked *(not verified)*.

## Pinned version

- **Bend 2.0.37** (`bend version`), since 2026-10-13 (2.0.36 from 2026-10-10, 2.0.35 before). Pin this version until further notice.
- Official repository: `github.com/bendlang/bend` (not `HigherOrderCO/Bend`, which is Bend 1).
- Installed with `curl -fsSL https://bend-lang.com/install.sh | sh` (script read before running: it downloads the GitHub release, checks the SHA256, installs only into `~/.bend`, no sudo).
- Update: `bend update` (runs the installer again). Do not update without recording it here, since Bend 2 changes fast.
- Telemetry: one query per day to bend-lang.com (version, OS, CPU). Turn it off with `export BEND_NO_TELEMETRY=1`.
- PATH: `export PATH="$HOME/.bend/bin:$PATH"`.

## Where things are

| What | Where |
|---|---|
| Language guide (677 lines) | `bend guide` or `~/.bend/guide/GUIDE.md` |
| Effects (custom C/JS) | `bend guide effects` |
| Shaders / parallel code | `bend guide shaders` |
| Base (3009 lines) | `~/.bend/bend2/base.bend` or `bend base [name]` |
| Lean proof kernel | `~/.bend/bend2/bendtt.lean` |
| Examples and demos | only in the GitHub repo: `demos/` (not shipped by the installer) |
| Papers | `paper/BendTT.pdf`, `paper/BendRT.pdf` in the repo |

Demos relevant to us: `proof_numerics` (proof of `add_comm`, `add_assoc`, `mul_comm`, `mul_dist` and `divmod` over `Nat`; it is the model for `nat-lemmas`), `proof_typed_eval`, `proof_insertion_sort`, `pure_par_sum`, `pure_par_sort`.

## What a proof project looks like

- `LAWS.bend`: the human writes the laws (`law name: for x: T  {a == b : T}`).
- `PROOF.bend`: imports `LAWS.bend` and proves each law with a `def` of the same name.
- `bend PROOF.bend` prints `ALL PROOFS CHECK` or `SOME PROOFS FAIL`. It is the gate before committing.
- There are no tactics: a proposition is a type, a proof is a `def`. `{==}` is reflexivity, `%e : P` rewrites, a recursive call is the induction hypothesis.
- `?name` prints the goal; `?TODO` leaves the proof open.
- Checker speed: the `add_zero` test over `Nat` took **0.17 s** in total.

## What Base offers

- Types: `Nat` (unary: `Zero`/`Succ`), `U32`, `F32`, `Char`, `String` (a list of `Char`), `List`, `Array` (binary tree, in-place mutation), `Map` (`String` keys), `Maybe`, `Result`, `Sigma`, `Either`, `Equal`.
- **Existing lemmas (very few):**
  - Equality: `Equal.cong`, `Equal.sym`, `Equal.trans`.
  - Nat: `Nat.ge_refl`, `Nat.max_ge_l`, `Nat.max_ge_r`.
  - Word/U32: `Word.add_comm`, `U32.add_comm`.
- **Not in Base:** `Nat.add_comm`, `add_assoc`, `mul_comm`, `mul_assoc`, distributivity, no `List` lemma (`append_assoc`, `length_append`, `reverse_reverse`...), nothing about `product`. **Conclusion: the `nat-lemmas` package is necessary**, and probably `List` lemmas too.
- The `proof_numerics` demo already proves the Nat lemmas above, so it can be used as a starting point (read it before rewriting).
- `List` operations: `map`, `length`, `append`, `reverse`, `take`, `drop`, `zip`, `foldl`, `foldr`, `filter`, `sort`, `range`, `replicate`.

## Numbers

- Only **`Nat`, `U32` and `F32`** exist. There is no `U64`, `I32`, `I64` or `F64` (a Metal limitation, according to the README).
- **There is no "byte" type.** Bytes will be `U32` (value 0–255) in a `List<U32>`; the bound is our own proof/invariant.
- `F32` operations: `add sub mul div mod pow neg abs sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil trunc round min max clamp lerp`, comparisons, `show`, `read`, `from_nat`, `to_nat`. Tested: `F32.exp(1.0)` prints `2.7182817`.
- **The `F32` operations are primitive `law`s, with no proof**: in Base they appear as `law F32.add: for a: F32 for b: F32  F32`. That is, they are opaque to the prover. This matches rule 5 of `CLAUDE.md` (do not prove anything numerical about floats).
- Consequence for Phase 3: accumulating loss and large sums in `F32` loses precision; without `F64`, gradient checking needs larger tolerances.
- A `Nat` literal above `256n` becomes `U32.to_nat`, capped at `4294967295n`. Beware of unary `Nat` in large dimensions (e.g. 784, 50257); a performance test is needed before deciding the shape representation.

## Backends

- It compiles to **C** (needs clang 14+; 19+ for `!`), **JS**, **Metal** (macOS) and **CUDA** (Linux). Lua, Luau and Python are planned.
- The GPU is enabled per call with `!` (e.g. `f!(x)`); in a native binary, `./prog --gpu off` forces the CPU and `--gpu 4GB` limits the GPU heap.
- **On this machine:**
  - clang 22.1.8 installed (ok).
  - `nvidia-smi` shows an RTX 4050 Laptop, 6141 MiB, driver 610.57.04.
  - **The CUDA Toolkit is not installed** (`nvcc` missing; `/usr/local/cuda` and `/opt/cuda` do not exist). The guide says that on Linux it needs **CUDA 12 at `/usr/local/cuda`**.
  - That is: the GPU does **not** run yet. CUDA 12 must be installed and tested with `pow2!`. *(not verified whether Arch/Omarchy's CUDA 12 works with driver 610 and the RTX 4050; test before planning Phase 3 on the GPU.)*
  - Without a GPU, `!` runs in parallel on the CPU, so the code still works.
- Without unified memory (a discrete card), moving data CPU↔GPU has a cost; the Apple Silicon gain described in the guide does not apply here.

## IO

- `File.open`, `File.read`, **`File.read_bytes`**, `File.read_at`, `File.size`, `File.write`, `File.write_bytes` (this one takes `List<U32>`), `File.close`. Enough to load MNIST and weights.
- There is also `IO.args`, `IO.now` (for benchmarks), `IO.random_u32`, `IO.get_env`, `IO.fork/join`, `IO.thread_count`, TCP/UDP, `Process.run`.
- Custom effects in C/JS are possible (`bend guide effects`), but the C API has no ABI promise: rebuild on every update.
- State of loading large files (GPT-2 has ~500 MB of weights) *(not verified)*: a `U32` list uses a lot of memory; evaluate `Array` and block reading.

## Proofs and trust

- `bend X.bend --verdict` re-checks with the Lean-proved kernel. **It requires `lean` v4.34.0** (via elan) or `$BENDTT` pointing to the compiled kernel. **It is not installed here**; the command failed with `Executable not found in $PATH: "lean"`. Install elan before publishing, to be able to say "verified by the audited kernel".
- `@unsafe def` skips the termination check and the checker shows `SOME PROOFS FAIL`. **Forbidden without a warning** (rule 4). Without `@unsafe`, there is no way to "work around" a proof, except leaving `?TODO`, which the checker flags.
- Mutual recursion is forbidden; the shrinking parameter must come first (the termination check reads from left to right).
- There is no `if`; use `match` on `True{}`/`False{}`.
- `match` only inspects a parameter or a pattern variable, never a computed value (`match f(x):` is refused). You have to go through a helper function.
- Variables are **affine** (used at most once); `+x` allows reuse if the type is `Data`. This will affect autograd a lot (a value used in the forward and the backward pass needs `+`). Warn Renan already: it is the strangest part for someone coming from Python.

## BendHub

- Site: `hub.bend-lang.com` ("mini Hacker News": Packages and Posts tabs, search, `/post/new`, `/auctions`).
- Login: `bend login`, **GitHub only**. Publish: `bend file.bend --publish name@version` (publishes the file and everything it imports). Without a name: `--publish` returns the hash and the line `import 0x<hash>/main.bend as P`.
- Import: `import name@version/main.bend as P`, or by hash.
- Publication is **public and permanent**; dependents reference by hash. Put a `LICENSE` next to the entry file (first line `SPDX-License-Identifier: MIT`). Without a `LICENSE`, it becomes MIT-0. Adding a `LICENSE` later changes the hash and requires a new version.
- **Names:** names with **12+ characters** are free (first come); short names are sold at auction. **`bpe`, `tensor` and `nat-lemmas` are too short or close to it** (`nat-lemmas` has 10). Use long names, for example `bend-ml-bpe-tokenizer` and `bend-ml-nat-lemmas`. Names can be withdrawn within 14 days if no version has been linked; they cannot be transferred.
- **Does something similar already exist?** *Not verified.* The hub page loads the list with JavaScript and the fetch only showed the tab titles. The hub still has to be looked at in a browser (search for `tokenizer`, `bpe`, `tensor`, `autograd`, `matrix`, `nat`) before deciding between contributing and differentiating. Action for Renan.

## Recorded debts and risks

- No proof debt yet (no code has been written).
- Risk: `--verdict` unavailable until Lean is installed.
- Risk: GPU unavailable until CUDA 12 is installed.
- Risk: unary `Nat` can be slow for large dimensions.
- Risk: the affinity (single use) of variables complicates autograd and tensors.

## Recommendation of adjustments to the plan

1. **Before Phase 1, an action for Renan (≈15 min):** open `hub.bend-lang.com` and search for `bpe`, `tokenizer`, `tensor`, `autograd`, `nat`. If a tokenizer already exists, decide between contributing and differentiating.
2. **Package names:** replace `bpe`/`tensor`/`nat-lemmas` with names of 12+ characters (see above). Choose and reserve early.
3. **Reverse the priority of `nat-lemmas`:** it is a real prerequisite of Phase 2 and also helps Phase 1 (proofs about `List`). Suggestion: build a mini `nat-lemmas` + `list-lemmas` right at the start of Phase 1, starting from the `proof_numerics` demo. It is the smallest publishable deliverable and gives the first BendHub package early (traction).
4. **Phase 1 (BPE):** bytes are `U32` in a `List<U32>`; there is no byte type. The roundtrip law needs a definition of "well-formed table" that the proof uses. Start with the simplest possible law (empty table → identity) and grow from there.
5. **Phase 2 (tensor):** represent the shape as `List<Nat>` in the type, but validate the cost of unary `Nat` and of affine variables in a small proof of concept (a vector with its size in the type + `dot`) before committing. If it is too heavy, use `Nat` only in the type (erased at run time with `-`) and `U32` in the data.
6. **Phase 3:** install CUDA 12 and test `pow2!` before promising a GPU. Plan B: parallel CPU (`./prog --threads N`), which is already the default. MNIST on the CPU should be enough for the demo. GPT-2 in `F32` with 6 GB of VRAM is feasible in size (124M × 4 B ≈ 500 MB), but depends on `File.read_bytes` handling the file.
7. **Install elan + Lean 4.34.0** to use `--verdict`. It enters the "ready to send to Taelin" criterion.

## Suggested next steps

1. Renan: check BendHub in the browser (item 1).
2. Renan: install CUDA 12 (`/usr/local/cuda`) and/or elan (`--verdict`) if desired; I can help with the commands.
3. `git init` and a GitHub remote (today the directory is not a repository).
4. Proof of concept: a vector with its size in the type, to decide the shape representation.
5. Start `nat-lemmas`.

## v0.1 — nat-lemmas (2026-10-03)

- 15 laws proved in `nat-lemmas/main.bend` (Nat: addition/multiplication; List: append/length; `product_append`). `bend nat-lemmas/main.bend` → `ALL PROOFS CHECK`. No `@unsafe` or `?TODO`. No proof debt.
- **Discovery:** an `import ... as NL` only exposes the defs of the file itself, it does not re-export. That is why the publishable package is **a single file**, with `law` and proof side by side (the `LAWS.bend`/`PROOF.bend` split is for application projects, not for libraries).
- **Discovery:** in `%e : P`, `_` marks where the right-hand side `b` of `e : {a == b}` is; the goal becomes `P` with `a`. When the goal has `a` and not `b`, use `Equal.sym` first.
- **Discovery:** list lemmas where elements are used more than once need `for +xs` and `Con{+h, +t}`.
- `--verdict` not run yet (Lean 4.34.0 missing).
- **Published on BendHub:** `bend-ml-nat-lemmas@0.1.0.0`, hash `0xa7aa06c09e97c6747c12cc64bb5203d9` (2026-10-03). BendHub versions have **four numbers** (`0.1.0.0`). Import: `import bend-ml-nat-lemmas@0.1.0.0/main.bend as NL`. Verified in a clean folder.
- The usage comment in `nat-lemmas/main.bend` was fixed to `0.1.0.0` after publication; only a comment changed, but the published hash is that of the previous file.

## Environment and v0.2 — shape PoC (2026-10-03)

- `reference/.venv`: torch 2.14.1+cpu, tiktoken 0.14.0, numpy 2.5.3, safetensors. Lean via elan did not install directly (a DNS failure in elan), but `bend --verdict` built its own kernel and **`nat-lemmas/main.bend --verdict` gives ALL PROOFS CHECK** (first run ~1 min, then 0.2 s).
- CUDA: Arch's package is **CUDA 13.3**; Bend asks for **CUDA 12 at `/usr/local/cuda`**. I did not install it (it needs sudo and the version is wrong). Everything stays on parallel CPU for now.
- **Vec(n) by type recursion** (`def Vec(n) -> Data: match n`, like `Word(n)` in Base): it works and a size error is a type error, but it **overflows the stack at n ≈ 50 thousand** (8192 is fine). It serves small dimensions, not data of 38M floats.
- **Representation decision:** `type Mat<-r, -c> is Type: Mat{data: Array<F32>}`. Dimensions are **erased** type parameters; data in an `Array<F32>`, row by row (index `i*c + j`). An incompatible dimension is a type error (`poc/mat_shape_error.bend`, message in `docs/shape-error-matrix.txt`). Row×column product of 784 elements: 0.002 s native.
- Honest limit: the `Array` size (a power of 2) is **not** tied to the type; the invariant `|data| >= r*c` holds by construction through the package's constructors, not by proof. The shape proof is about the algebra (`reshape` requires equal `product`, with `nat-lemmas`).
- Bend rules learned: `match` only opens **parameters**, so a pair returned by `Array.get` needs its own function or to be passed as an argument (the `Array.map.go` pattern); no mutual recursion; pairs match as `Tuple{a, b}` in `match`; functions must be defined above where they are used; types with `Array` are `Type` (affine), not `Data`.

## v0.3 — bend-ml-bpe-tokenizer (2026-10-03)

- Published: `bend-ml-bpe-tokenizer@0.1.0.0`, hash `0x3333bd5274f4ce66fdf8fe0c26c6b532`. Laws `roundtrip`, `vocab_bound`, `dec_append` proved and accepted by the kernel (`--verdict` ALL PROOFS CHECK). Import verified in a clean folder, including using the published law.
- **The design that made the proof feasible:** tokens `B{n}`/`M{k}` (no index arithmetic), each rule carries its id, the table goes from the oldest rule to the newest and the `done` accumulator of `encode.go` evolves exactly like Base's `List.reverse.go`. The expansion (`exp`) is recursive over the list of rules (it only looks at older rules), with no vocabulary table and no fuel. The "well-formed table" condition is just "distinct ids" (`wf`).
- ~20 auxiliary lemmas (`ext_dec`, `mp`, `kp`, `enc_dec`...). Techniques learned: `match` only on parameters ⇒ pass a `hit: Bool` computed by the caller and prove `hit == peek(...)`; only structural recursion, with the shrinking argument FIRST; `%e : P` needs `_` on the right-hand side `b` of `e`, so use `Equal.sym` to rewrite in the other direction; refute `True == False` with the motive `BD(b, t, f)`.
- Tests: `reference/test_bpe.py` compares `train/encode/decode` in Bend with `reference/bpe_ref.py` on 4 corpora × 4 samples (English, accents, mixed UTF-8 with emoji/CJK, empty): 0 failures.
- The published `bpe/LICENSE` has no `SPDX-License-Identifier` line (the installer showed "License: see LICENSE"); `nat-lemmas` has it. Identical MIT text.
- Pending for GPT-2: a regex pre-tokenizer and loading the official `merges.txt` (a table of 50 thousand rules, base ids in `bytes_to_unicode` order).

## v0.4 — bend-ml-tensor (2026-10-03)

- Published: `bend-ml-tensor@0.1.0.0`, hash `0xf9837737d2c3f58ae1d0c5df42255584`. `ALL PROOFS CHECK` and `--verdict` OK; import and shape error verified from BendHub in a clean folder.
- **Final representation:** `Vec<-n>` and `Mat<-r,-c>` with erased dimensions, data in `List<&2, F32>` / a list of rows (copyable and parallelizable). Measured: 10 M multiply-adds in 0.23 s on one thread (~44 M/s). Discarded `Array` (log n index with `Array.size` on every access) and `Vec(n)` by type recursion (the stack overflows near 50 thousand).
- A shape error = a type error: `docs/shape-error-bad_matmul.txt` (`expected Mat<3,5>, got Mat<4,5>`) and `docs/shape-error-bad_reshape.txt` (`expected 12, got 15`). That is the content of the video.
- Laws: `reshape_swap` (r·c = c·r) and `reshape_flat` (r·c = 1·(r·c)), proved with `mul_comm` and `mul_one_l` from the published nat-lemmas.
- Numerical tests: `reference/test_tensor.py` compares with PyTorch: matmul (4 shapes), transpose, relu, softmax (including values ±1000), gelu (tanh), layernorm. Maximum error ≈ 1e-6, 0 failures.
- New rules: a parameter also used in the type counts as a use (it needs `+` if used again); defs must come before their use (no free ordering); `match` in binder order.

## v0.5 — bend-ml-autograd (2026-10-03)

- Published: `bend-ml-autograd@0.1.0.0`, hash `0x174ef0d27bc1c621ddb9961c7f224de1`. `ALL PROOFS CHECK` and `--verdict` OK.
- **Proved law `reverse_eq_forward`**: autodiff reverse mode == forward mode, over `Nat` expressions (constants, X, sum, product). Proof by induction of the stronger result `reverse(e,g) = g × forward(e)`, using `mul_comm`, `mul_assoc`, `mul_dist`, `mul_zero`, `mul_one_*` from the published nat-lemmas and a derived `dist_l` (left distributivity).
- Scalar autograd in `F32` (GCst, GVar, GAdd, GMul, GRelu, GTanh, GExp) and layers with typed backward (`linear_bwd` returns `Mat<n,i> & (Mat<i,o> & Vec<o>)`; swapping a dimension does not compile).
- Gradient checking: `reference/test_autograd.py` compares with PyTorch's autograd, 21 checks (5 expressions × 3 points, linear_bwd ×3, relu_bwd, cross-entropy ×2), maximum error ≈ 5e-7, 0 failures.
- Bend: negative command-line arguments need `--` (`bend cli.bend -- scalar 1 -1.3 ...`).

## v0.6 — MNIST (in progress)

- `demos/mnist/train.bend` (784-128-10 MLP, SGD, batches of 100) and the twin `reference/mnist_torch.py` with the same initial weights (`demos/mnist/data/init/*.txt`) and the same batch order.
- **Correctness validation:** after 50 steps, Bend: `train_loss=1.6616005`, `7829/10000`; PyTorch: `train_loss=1.661600`, `test_acc=0.7829`. Identical.
- Performance (honest): ~0.84 s per batch of 100 on one thread (≈ 24 M effective multiply-adds/s). The parallelism of `x y = f g` scales well in `pow2` (8x with 16 threads), but the row-wise list `matmul` only reached ~1.8x (with or without data sharing); cause unresolved. `bend -o` + `--threads N` is the mechanism.
- `tensor@0.1.1.0` adds `Mat.matmul_t` (an already-transposed operand) for GPT-2.

## v0.6 and v1.0 — MNIST, GPT-2 and delivery (2026-10-03)

- **MNIST (1 full epoch, 600 batches of 100, lr 0.1):** Bend loss `0.52047706`, 9129/10000, 544.3 s; PyTorch loss `0.520477`, 9129/10000 (9128 with 1 thread), 0.21 s (16 threads) / 0.30 s (1 thread). Identical correctness; Bend is ~1800x slower for lack of BLAS and because it uses lists. Details and hardware in `demos/mnist/BENCHMARK.md`.
- **GPT-2 small (124 M) in Bend:** weights loaded in ~10 s (bytes -> F32 by the bitcast `U32{w}` -> `F32{w}`, in 1 MB blocks); ~3 s per token with a KV cache; ids identical to PyTorch's over 22 tokens (3 prompts), logits within 2e-4. `reference/test_gpt2.py`.
- **GPT-2 tokenizer in Bend:** the regex pre-tokenizer over bytes + the 50,000 rules via `bend-ml-bpe-tokenizer`; 16/16 texts identical to `tiktoken`. The equivalence "applying the rules in order == GPT-2's rank-based algorithm" was confirmed against the real tiktoken (`reference/gpt2_prep.py`). Limit (lifted later, see the Unicode pre-tokenizer in the stability work): bytes >= 128 counted as letters.
- **Published packages:** `bend-ml-nat-lemmas@0.1.0.0` (`0xa7aa06c0...`), `bend-ml-bpe-tokenizer@0.1.0.0` (`0x3333bd52...`), `bend-ml-tensor@0.1.0.0` (`0xf9837737...`) and `0.1.1.0` (`0xdbe1000e...`, adds `matmul_t`), `bend-ml-autograd@0.1.0.0` (`0x174ef0d2...`). All import from BendHub in a clean folder and the laws are reusable (`BPE.roundtrip`, `AG.reverse_eq_forward`).
- `reference/check_all.py --full`: 25/25 checks ok (proofs + kernel, expected shape errors, Python references, tokenizer, GPT-2, MNIST).

### Debts and known limits

- **No proof debt**: no `@unsafe`, no `?TODO` published.
- **Parallelism unresolved:** the guide's `pow2` scales ~8x with 16 threads, but the row-wise `matmul` over lists only reached ~1.8x (with or without shared data, at any depth). Discarded hypotheses: laziness (strict sums at the leaves), sharing (private data per task). Not investigated: shared allocator, pointer-based memory access. Worth reporting on Bend's GitHub.
- **GPU not used:** Bend asks for CUDA 12 at `/usr/local/cuda`; Arch's `cuda` package is 13.3 and installing it needs sudo. The RTX 4050 (6 GB) stayed out of the benchmarks.
- The invariant "rows with `c` numbers" of `Mat<r,c>` is not in the type.
- The published `bpe/LICENSE` has no SPDX line (only the MIT text).
- For Renan: (1) open hub.bend-lang.com and look for third-party tokenizer/tensor/autograd packages (the list is loaded by JS and I could not read it); (2) record the video and post the thread (`docs/launch/`); (3) if you want the GPU, install CUDA 12 at `/usr/local/cuda`; (4) the `bend login` token is at `~/.bend/bender.json` (permission 600); the `bend-ml` key was revoked.

## v2 — performance experiments (2026-10-04)

Plan in `docs/v2-plan.md`; micro-benchmarks in `bench/` (`timeit.py` measures the median of 3 runs per number of threads).

### Exp. 1-2: lists vs `Array<F32>` (100 M multiply-adds, 1 thread)

| | time | multiply-adds/s |
|---|---|---|
| lists (`bench/mm_list.bend`, like `bend-ml-tensor` 0.1.x) | 2.274 s | 44 M |
| `Array<F32>` with `Array.get` (`bench/mm_array_check.bend`, distinct data) | **0.046 s** | **2,200 M** |

- **A ~49x gain just by changing the data structure.** Result checked against NumPy (`1024010000` for 128001 products of 8000.79; the small deviation is `F32` rounding).
- Back-of-the-envelope for MNIST: ~18 G multiply-adds per epoch (3 products per batch × 600 batches). At 2.2 G/s that is ~8 s per epoch, against 544 s with lists and 0.3 s for 1-thread PyTorch. **The distance to PyTorch would drop from ~1800x to ~30x**, before any parallelism. This is an estimate, not a measurement of the full training.
- Cost of `Array`: it is affine (`Type`), so each read returns the array together with the element, and `Array.get` recomputes `Array.size` (log n). It still gained a lot.
- The v1 conclusion ("Bend is ~1800x slower") was an effect of the data structure, not of the language.

### Exp. 3-5: parallelism, cache, clone and distance to PyTorch (2026-10-04)

All numbers: 1 thread unless stated, median of 3 runs, Intel Core Ultra 7 155H (6 P + 8 E + 2 LP-E cores, 22 threads), `bend -o`. Code in `bench/`.

- **Indexing beats walking the tree:** `Array.get` by index: 2,200 M multiply-adds/s; walking both trees together and rebuilding them (`bench/tree_dot.bend`): 31 M/s (70x worse). In Bend, the fast path of `Array` is the indexed one.
- **`Array.clone` is cheap** (~17 µs per array of 131,072 elements, with the result's read forced). **The hypothesis "clone shares nodes and reads become contended atomics" was refuted:** 16 tasks reading clones of one pair scale like 16 tasks with private arrays (`bench/share_test.bend`: ~4.1x with 16 threads in both cases).
- **Parallelism, order of magnitude:** with independent work and small arrays (784 elements), 16 threads give ~4 to 6x (3.2 G multiply-adds: 0.88 s on 1 thread, 0.13 s on 22). On the 100×784×128 product in blocks of rows (`bench/mm_par.bend`), ~2.1x with 8 threads, the same for 2^d = 8 or 16 blocks. It does not depend on tiling (`bench/mm_tile.bend`: 1, 8, 16 and 32 blocks of columns, the same time) nor on the clone cost (22% of the time). The guide's `pow2` reaches 8x, so the limit is the type of workload. Hybrid CPU: the practical ceiling is below 22x.
- **The size of the `Array` costs:** the same product with arrays of 2^20 slots instead of 2^17 was ~60% slower (1.9 s -> 3.1 s for 4 G multiply-adds): the tree depth enters every `get`. Allocate the smallest possible array.
- **Distance to PyTorch on the 100×784×128 product:** PyTorch 1 thread = 0.162 ms (62 G multiply-adds/s); 16 threads = 0.047 ms (212 G/s). Bend: lists 44 M/s (**~1400x behind**), `Array` 1 thread 2.3 G/s (**~27x behind**), `Array` with parallel blocks ~4.5 G/s (**~47x behind parallel PyTorch**). What is left: scalar code against AVX/FMA with BLAS.

### Exp. 6: full MNIST over `Array` (`demos/mnist/fast.bend`, 2026-10-04)

Flat matrices in `Array<F32>`, `gemm` with strides (covers `X·W`, `H·Wᵀ` and `Xᵀ·dZ` without transposing), bias/relu/mask/SGD in place by index, softmax and cross-entropy per row with lists of 10, batches read from the file at each step (`File.read_at`). 1 thread.

| 50 batches | loss | hits | time |
|---|---|---|---|
| lists (`train.bend`) | 1.6616005 | 7829 | ~42 s |
| **`Array` (`fast.bend`)** | **1.6616004** | **7829** | **0.9 s** |
| PyTorch | 1.661600 | 7829 | 0.02 s |

| 1 epoch (600 batches) | loss | hits | time |
|---|---|---|---|
| lists | 0.52047706 | 9129 | 544.3 s |
| **`Array`** | **0.5204771** | **9129** | **11.5 s** (47x faster) |
| PyTorch 1 thread | 0.520477 | 9128 | 0.30 s |
| PyTorch 16 threads | 0.520477 | 9129 | 0.21 s |

Distance to PyTorch (1 thread): from ~1800x to **~38x**. Correctness preserved (same loss and same hits).
The ~19 ms per batch split into ~9 ms of `gemm` (20 M multiply-adds at 2.3 G/s) and ~10 ms of everything else (builds `X` with 78,400 `set`s, reads 78 KB, bias/relu/mask/SGD over ~100 thousand elements with get+set).

### Exp. 7: parallel `gemm` in MNIST (`fast.bend` now uses 2^3 blocks in the two large products)

Blocks of rows of C, each with a copy of A and B (`Array.clone`), results in lists written into C at the end. Same correctness (50 batches: 1.6616004 and 7829; epoch: 0.5204771 and 9129).

| 1 epoch | time |
|---|---|
| `Array`, sequential | 11.5 s |
| `Array`, parallel gemm, 2^2 blocks, 8 threads | 7.2 s |
| `Array`, parallel gemm, 2^3 blocks, 16 threads | **6.3 s** |
| `Array`, parallel gemm, 2^4 blocks, 16 threads | 7.5 s |

Distance to PyTorch: 6.3 s against 0.21 s (16 threads, ~30x) and 0.30 s (1 thread, ~21x). What is left is the sequential part (data loading ~2 s per epoch; bias/relu/mask/SGD over ~100 thousand elements): Amdahl's law.

### Exp. 8: the `bend-ml-tensor-array` package and MNIST with the typed API (2026-10-04)

- Published: `bend-ml-tensor-array@0.1.0.0` (MIT, `--verdict` ok, no `@unsafe`). `Mat<r,c>` over a flat `Array<F32>`; `matmul`, `matmul_nt`, `matmul_tn` with parallel blocks; bias, relu, `relu_bwd`, SGD, column sums, `softmax_ce`, `count_correct`. 39 checks against PyTorch (`reference/test_tensor_array.py`, maximum error ≈ 1.4e-6). Imports from BendHub in a clean folder.
- **A shape error in the gradient too:** `tensor-array/tests/bad_grad.bend` (`dW` requested as 128×784 instead of 784×128) does not compile (`docs/shape-error-array-*.txt`).
- `demos/mnist/fast.bend` now uses only the typed API (importing from BendHub): 50 batches 1.6616004 / 7829; **1 epoch 0.5204771 / 9129 in 7.6 s** (the raw-kernels version took 6.3 s: the extra cost is the list <-> Array conversions in the `of_list`/`scale255` wrappers, not yet optimized).
- Bend rules learned: `Mat` is `Type` (affine), so `Maybe<&1, Mat<...>>` and not `&2`; one record `MMul{a, b, c}` per operation solves "return what it read"; a nested `&`/`Tuple` pattern in `match` fails ("annotated term (cannot infer)"), use records with their own `type`.

### Exp. 9: GPT-2 over `Array` and why parallelizing with copies does not pay off in matrix · vector (2026-10-04)

`demos/gpt2/fast.bend`: the matrix · vector products (`matmul_nt` with `n = 1`: qkv, projection, MLP and logits; 124 M multiply-adds per token) over `bend-ml-tensor-array`; attention, LayerNorm and GELU stay in lists. Ids and logits identical to PyTorch's (`reference/test_gpt2.py`, 22 tokens, |Δlogit| ≤ 2e-4).

| GPT-2 small, "The capital of France is", 8 tokens | per token | total |
|---|---|---|
| lists (`gpt2.bend`) | ~3 s | 49.8 s (10 s of loading + 8 tokens) |
| `Array`, products in 2^3 blocks of columns (`par = 3`) | ~1.1 s | 22 s |
| `Array`, sequential (`par = 0`) | **~0.1 s** | **11.1 s** (9 s of loading + 1.2 s for the 8 tokens) |
| PyTorch (CPU, no KV cache): forward pass of 11 tokens, 16 threads / 1 thread | 21 ms / 55 ms | 1.3 s in `gpt2_ref.py` (includes ~1 s of weight loading) |

- **Sequential beats parallel by 10x.** Isolated with `bench/mv.bend` (768×2304, 100 repetitions): sequential 0.76 ms per product (2.3 G multiply-adds/s); `par = 3` 10 ms per product, **the same with 1, 8 or 16 threads**. The cause is cost, not contention: each node of the task tree does an `Array.clone` of the **whole** weight matrix (28 clones per call); cloning costs ~0.1 ns per element and, in matrix · vector, each weight is read only once (~0.4 ns), so copying P times costs more than computing. In matrix · matrix products (MNIST: each weight is read 100 times) the copy is negligible and parallelism wins.
- Future solution (not done): split the weight matrix **without copying**, destructuring `ANode{xs, ys}` into subtrees and giving one to each task. It requires rows aligned to a power of 2 (zero padding, ~1.3 to 1.8x memory).
- Distance to PyTorch: from ~150x per token (v1: 3 s against 21 ms) to **~5x** (v2: ~0.1 s against 21 ms, 16 threads) and ~2x against 1-thread PyTorch (55 ms). **Correction:** an earlier version of this note estimated PyTorch at ~0.15 s per token; that was the total time divided by 8 (it includes loading weights), not the *forward pass*. Bend's total is still dominated by weight loading (9 s against ~1 s).
- Memory: the process reserves ~10 GB of address space while loading (trees of nodes, not contiguous vectors), but the resident peak was 2.3 GB, and 1.5 GB after streaming the blocks into the arrays (WP6; measured with `scripts/peak_rss.sh`). The "~10 GB" in earlier text was the virtual size.

### Final MNIST with the typed API (3 runs, 16 threads; `demos/mnist/fast.bend`, `bend-ml-tensor-array@0.1.1.0`)

| | 1 epoch | 3 epochs (loss / hits) |
|---|---|---|
| v1, lists | 544.3 s | |
| v2, typed `Array`, 16 threads | **6.6 s** (6.6 / 6.7 / 6.6) | 0.5204771 / 9129; 0.27043572 / 9298; 0.2156194 / 9418 |
| v2, typed `Array`, 1 thread | 18.1 s | |
| PyTorch (16 threads) | 0.19 to 0.21 s | 0.520477 / 9129; 0.270433 / 9298; 0.215613 / 9418 |

A ~82x gain over v1; distance to PyTorch (16 threads) ~33x, (1 thread, 0.30 s) ~22x. The same losses and hits across the three epochs.

### Axis 2 (guarantees) and closing v2 (2026-10-04)

- **`bend-ml-bpe-tokenizer@0.1.1.0`** (published): `train_wf` (the table from `train` is always well formed, by induction over `train.go` with the invariant "all ids already used are smaller than the next one", using 4 `Nat` lemmas: `lt_succ`, `lt_ne`, `lt_up`, and list lemmas: `below_has`, `below_up`, `below_cons`, `below_fresh`) and `roundtrip_trained` (a direct consequence: roundtrip for any trained table, no hypothesis). `--verdict` ok. Imported from BendHub in a clean folder, with the law applied.
- The `bpe` README used to say "train has no proof"; now it does.
- Packages in v2: nat-lemmas 0.1.0.0, bpe 0.1.1.0, tensor 0.1.1.0, **tensor-array 0.1.1.0 (new)**, autograd 0.1.0.0. `reference/check_all.py`: 27/27 (quick) and 32/32 (`--full`).
- Pending for Renan: (1) review the tone of the thread and record the video (`docs/launch/`); (2) if you want the GPU, install CUDA 12 at `/usr/local/cuda` and I will measure `!`; (3) the parallel gain in matrix · vector requires partitioning the `Array` without copying (idea in exp. 9) — worth a report to the Bend team: `Array.clone` is O(n) and `Array.fork` is `@unsafe`.
- **Finding for the GPU attempt:** the Bend binary honors the `CUDA_HOME` environment variable (default `/usr/local/cuda`) and only needs `include/nvrtc.h` (plus `cuda.h`) and `lib64/libnvrtc` from that directory, then links with `-lcuda -lnvrtc`. A user-local CUDA 12 toolkit could therefore be tried without sudo. Not done.

### English republication (2026-10-04)

The whole repository was translated to English (documentation, comments, CLI and demo messages, commit messages), so each package was republished with English source comments and README. The proofs and the API are unchanged; every package passes `ALL PROOFS CHECK` and `--verdict`.

| Package | New version | Hash |
|---|---|---|
| `bend-ml-nat-lemmas` | 0.1.1.0 | `0x2bbe4207657ea8b1158567d930651411` |
| `bend-ml-bpe-tokenizer` | 0.1.2.0 | `0xa071a92aabcffe6fc04dc3a60f0096ba` |
| `bend-ml-tensor` | 0.1.2.0 | `0x48e80e20946abebea50e06561806a40d` |
| `bend-ml-tensor-array` | 0.1.2.0 | `0x83d7a8813f8d06d870224fde14ffb05a` |
| `bend-ml-autograd` | 0.1.1.0 | `0x0ed882696c6c048c64c66242a999a426` |

The older versions stay on BendHub (publication is permanent). The demos and benchmarks now import the new versions; `reference/check_all.py --full` is 32/32.

The test inputs that contain accented Portuguese words (`reference/test_bpe.py`, `test_gpt2_tok.py`, `gpt2_prep.py`) were kept on purpose: they exercise multi-byte UTF-8.

Commit history: rewritten on 2026-10-04 to remove the `Co-Authored-By` trailers, translate the messages to English and split the work into small commits (the content of each commit is unchanged; the tags `v0.1.0` and `v1.0.0` point to the equivalent commits).

### Exp. 10: GPU with a user-local CUDA 12 (2026-10-04)

- **Setup works without sudo:** the Bend binary honors `CUDA_HOME`; the NVIDIA redistributable archives `cuda_cudart` 12.9.79 and `cuda_nvrtc` 12.9.86 (headers and `libnvrtc` only) unpacked into `~/.local/cuda12` were enough. The driver 610.57.04 supplies `libcuda`. Steps in `docs/gpu-setup.md`. The guide's `pow2!(26n)` builds (`prog.gpu` is written next to the binary), runs on the RTX 4050 and returns the right value.
- **But the GPU is slower than the CPU for every kernel I tried** (22-thread CPU vs `!` on the GPU, same program, `--gpu off` for the CPU):

| workload (each task builds its own data) | GPU | CPU | GPU / CPU |
|---|---|---|---|
| dot products on `Array`, 32 tasks | 0.40 s | 0.022 s | 18x slower |
| dot products on `Array`, 1024 tasks, k = 20,000 | 3.13 s | 1.00 s | 3.1x slower |
| dot products on `Array`, 1024 tasks, k = 200,000 | 29.6 s | 9.3 s | 3.2x slower |
| dot products on lists, 1024 tasks, k = 3,000 | 18.1 s | 2.5 s | 7.3x slower |
| dot products on lists, 1024 tasks, k = 30,000 | 178.6 s | 24.3 s | 7.3x slower |

  There is no crossover as the work grows: the ratio is flat. The shared-`Array` matrix product (`bench/mm_par.bend` with `par!`) aborts on the GPU with `memory fault`, while it runs on the CPU. The guide itself says GPU work shines on uniform numeric kernels (mandelbrot, n-body) and that data is "managed" memory that faults across PCIe on first touch; our kernels are long sequential loops over heap structures, which is the opposite.
- **First conclusion was too hasty.** Those runs used 32 to 1,024 tasks, but the guide says a `!` call should fan out to ~16,384 leaves (one per lane) and end in *flat* loops (scalars only, tail calls). With a flat numeric kernel and 2^14 leaves (`/tmp/flat.bend`: `spin` over `F32`), the GPU **does win**, and the gap grows with the work per lane (the ~0.2 s fixed cost is NVRTC compilation and start-up):

| 16,384 leaves, flat `F32` loop | GPU | CPU (22 threads) |
|---|---|---|
| 20,000 iterations each (0.33 G total) | 0.217 s | 0.054 s |
| 200,000 iterations each (3.3 G total) | 0.254 s | 0.300 s |
| 2,000,000 iterations each (33 G total) | **0.479 s** | 1.808 s (GPU **3.8x faster**) |

- **What loses is memory-bound work.** 16,384 dot products of 784 elements over lists built before the `!`: GPU 0.79 s vs CPU 0.23 s (whole program, including building the data). Repeating the `!` over the same data (so it is already on the device) costs ~0.05 s per repetition on the GPU against ~0.034 s on the CPU, i.e. still no gain: each list or `Array` element is a dependent pointer load, and Bend's heap is "managed" memory (pages migrate across PCIe on first touch). Our tensor kernels are exactly that shape, so the arithmetic advantage of the GPU never shows.
- **Conclusion:** the GPU path is not broken (compute-bound flat loops run 3.8x faster than the 22-thread CPU), but with matrices stored as lists or `Array` trees, the matrix products are memory-bound and the CPU is the faster target in Bend 2.0.35. The benchmarks in this repository therefore use the CPU. A GPU-friendly matrix product would need data laid out so each lane streams scalars (as in the guide's shader demo), which is a different design from the `Array`-backed `Mat`.

## Stability work for v2.1 (2026-10-04)

Goal: a stranger can clone the repository, reproduce every claim, and see it verified on every push.

- **Reproducibility:** `make setup` installs Bend 2.0.35 (tarball SHA256-checked), Lean 4.34.0 (elan), a Python venv and the MNIST and GPT-2 data without sudo; `check_all.py` fails early if `bend` is not 2.0.35 or `lean` is not 4.34.0. A fresh clone with an empty `HOME` passed `make setup && make check`.
- **CI** (`.github/workflows/ci.yml`, ubuntu-latest, Python 3.13, runner with 4 vCPU and ~15 GB RAM): push, pull request, manual and weekly. Run of 2026-10-04 (`Link the audit guide`): 9 min 21 s in total = toolchain and data cache 33 s, setup 33 s, quick checks 2 min 19 s, full checks 5 min 43 s. All checks pass, GPT-2 included.
- **Checks:** `make check` runs 31 checks (~1.6 min locally); `make check-full` runs 36 (~4.7 min locally, adds GPT-2 and MNIST). They include: the five packages with `--verdict`, no `@unsafe`/`?TODO`, four shape errors that must not compile, tensor and tensor-array against PyTorch (random shapes, all `par` levels), autograd on 120 random expression trees against PyTorch, 80 random BPE corpora against a Python reference, the GPT-2 tokenizer against `tiktoken` on 79 texts (49 fixed with symbols, emoji, CJK, NBSP, plus 30 random Unicode strings), 11 GPT-2 prompts against PyTorch, and the import of each package from BendHub (`test_published.py`).
- **Tested dependency versions** (`reference/requirements.txt` has ranges): Python 3.14.7 locally and 3.13 in CI, numpy 2.5.3, torch 2.14.1 (CPU), tiktoken 0.14.0, safetensors 0.8.0, regex 2026.9.29, requests 2.34.2, clang 22.1.8 locally, Bend 2.0.35, Lean 4.34.0.
- **Proved now:** `cap_ok` (`2^cap_depth(n) >= n`) for the `Array` capacity of `Mat`. What stays trusted: that `Array.new(d)` allocates `2^d` slots. 24 laws in 5 packages (`reference/list_laws.py`).
- **Checked at run time:** `Mat.softmax_ce_checked` and `Mat.count_correct_checked` return `None` when `length(labels) != n`; the BPE CLI and the GPT-2 demos refuse a merges table with an odd number of ids or a reference to an id that does not exist yet (`wf` holds by construction because the CLI numbers the rules).
- **Pre-tokenizer:** it decodes UTF-8 and classifies each code point with a table generated from the `regex` module (`reference/gen_unicode_table.py`, 939 ranges, `demos/gpt2/unicode.bend`); whitespace backs off a whole character. It matched `tiktoken` on every text, including 180 random Unicode strings. Remaining limits: code points above U+1FFFF and invalid UTF-8 count as letters.
- **Memory:** the "~10 GB while loading" figure was the **virtual** size. Resident peak of `fast.bend` was 2.3 GB, and 1.5 GB after streaming 1 MB blocks straight into the arrays (`Mat.fill_at`); loading went from 15 s to 10 s. Output (ids and logits) is unchanged.
- **MNIST timing:** an epoch of the local `tensor-array` and of the published 0.1.2.0 version both took 16.0 s on a loaded machine (load average ~2.5, the 6.6 s figure was taken on an idle machine), so the proved `cap_depth` costs nothing measurable.
- **Republished:** `bend-ml-tensor-array@0.1.3.0`, hash `0xa78e1f1609ed090d0092c5fff6f9bac7`. The usage comment in `tensor-array/main.bend` was updated to 0.1.3.0 after publication (comment only).
- **History:** the repository history was rewritten on 2026-10-04 (English messages, smaller commits, no attribution lines); old clones must be cloned again.

## README and video (2026-10-04)

- The README images are generated, not drawn: `scripts/capture_outputs.sh` stores the real command outputs in `docs/media/outputs/`, `docs/media/shoot.mjs` renders them as terminal screenshots, `docs/media/charts.mjs` builds the charts, the package diagram and the banner from `docs/media/numbers.json`, and `docs/media/render_video.mjs` renders `docs/media/video/scene.html` into a 60 s MP4 with headless Brave and ffmpeg (`make media`).
- Numbers re-measured for them on an idle machine: MNIST epoch 7.0, 7.0, 7.2 s (6.6 s in the original runs; 12.5 s once under load), PyTorch 0.18 to 0.20 s (0.55 s once), so PyTorch is ~35x ahead rather than the ~22 to 33x quoted before; GPT-2 loads in 7 s and generates the prompt plus 8 tokens in 1.2 s.
- The stale label "3 prompts" of the GPT-2 check was fixed: `test_gpt2.py` has always run all 11 prompts.
- The MP4 (~10 MB) is attached to the v2.1.0 release instead of the repository, so it does not stay in the history forever.
- Visual identity (second version, after the first looked like a generic terminal theme): the colours are bend-lang.com's own CSS tokens, the only typeface is a subset of iA Writer Mono S (SIL OFL; renamed "Bend ML Mono" because a modified font may not keep the reserved names "iA Writer" and "Plex"; license in `docs/media/fonts/LICENSE.md`), and the SVG figures embed it so GitHub renders them as designed. Charts use Bend's own style: vertical bars on a linear scale, Bend in violet, the rest in grey, values off the chart hatched.

## v3: answering the launch feedback (2026-10-05)

Plan: `docs/v3-plan.md`. The launch post (2026-10-05) asked: "dfdx already does compile-time shapes?",
"compile times on deeper nets?", and kazzzz520 (Bend-Conv) pointed at copying as the real cost of
parallel work. Plans after v3 (llama.bend, a training framework) are in my private notes.

### Exp. 11: `Bands<r, c>`, matrix · vector in parallel without copying the weights

- **Idea** (Bend-Conv's band storage, typed): keep the weight matrix as a tree of row bands, each band its
  own `Array`. `Array` is a plain ADT in Base (`ALeaf`/`ANode`), so a `match` hands each task its band in
  O(1), with no `@unsafe`; only the input vector (c numbers) is cloned per split.
- **Types:** `BNode{x: Bands<half(r), c>, y: Bands<r - half(r), c>}`, an indexed type that Bend 2.0.35
  accepts (a wrong row count does not compile). Law `half_cover`: `n == half(n) + (n - half(n))`, via
  `half_le` and `add_sub`. Every function that walks the tree carries an erased proof `rr == r` tying the
  row count used at run time to the type (`eq_half`, `eq_rest`).
- **Two traps that kept it sequential** (found by reading the emitted C, `bend X.bend -o x.c`):
  1. **An erased parameter at the end of the parameter list turns the parallel let into two sequential
     calls.** `bmv(+c, -r, b, xp, +rr, -e)` had no join task in the C; `bmv(+c, -r, b, +rr, -e, xp)` has one.
     Minimal repro: `docs/upstream/erased_last_{ok,bad}.bend` (16 leaves of a flat loop: ok 0.054 s → 0.011 s
     with 16 threads; bad 0.053 s → 0.055 s). Draft report: `docs/upstream/erased-last-parameter.md`, posted as
     [bendlang/bend#1374](https://github.com/bendlang/bend/issues/1374); cause and fix in
     [bendlang/bend#1377](https://github.com/bendlang/bend/pull/1377) (merged 2026-10-07; see "Upstream PR for #1374").
  2. **A non-tail recursive helper in the fork tree halves the gain.** `half(n) = 1 + half(n - 2)` called at
     every node: 50257 × 768, 40 products, 16 threads, 0.69 s; the same with a tail loop (or `Nat.div`)
     0.36 s. `half` is now `half.go(n, acc)`, a flat loop.
- **Result** (`bench/mv_bands.py`, per product with the matrix already built; full table with 1, 4, 8 and 16
  threads in `bench/results/mv_bands-2026-10-05.txt`, measured with the list-in/list-out API; the typed
  `Mat<1, c>` API adds a `from_list`/`to_list` per call: 0.39 ms against 0.32 ms in one run on 2304 × 768, 16 threads, 2^4 bands):

| W | `matmul_nt` sequential | `matmul_nt`, par = 3 (clones W) | Bands 2^4, 16 threads | Bands 2^5, 16 threads |
|---|---|---|---|---|
| 2304 × 768 | 0.80 ms | 10.4 ms | 0.32 ms | 0.33 ms |
| 768 × 768 | 0.26 ms | 4.76 ms | 0.16 ms | 0.15 ms |
| 3072 × 768 | 1.07 ms | 23.5 ms | 0.42 ms | 0.40 ms |
| 768 × 3072 | 1.26 ms | 22.7 ms | 0.45 ms | 0.42 ms |
| 50257 × 768 | 25.1 ms | 337 ms | 6.1 ms | 4.5 ms |

  2 bands never helped (the same time as 1); 4 bands give ~1.6x on 4 threads; 16 to 32 bands give 2.1x to
  5.6x. Small products are dominated by waking the threads.
- **Bit-identical:** each band runs the same dot-product kernel as `matmul_nt`, so `Bands.matvec` equals
  `Mat.matmul_nt(1n, ..)` bit for bit (`reference/test_tensor_array.py` checks it on 15 shapes and all depths),
  and GPT-2's 11 prompts give the same ids and the same logits as v2.1, digit for digit.
- **Loading:** streaming 1 MB blocks through `Bands.fill_at` cost +1.2 s on the 48 layer matrices (a block
  that crosses a band boundary is walked by both bands; ~720 crossings), while the logits matrix (15
  crossings) was unaffected. The demo now reads each band from its own range of the file (`load_tree`,
  `get_band`), so no block crosses a boundary.
- **Published:** `bend-ml-tensor-array@0.1.4.0`, hash `0x1e52188e4a40cfe41a87f1688743dff4`.
- **GPT-2** (`demos/gpt2/fast.bend`, weights in 2^4 bands): "The capital of France is" + 8 tokens, 13 forward
  passes. The machine had background load (an Android emulator and a Gradle daemon), so v2.1 and v3 were run
  alternately, 5 times each (`bench/results/gpt2-v21-vs-v3-2026-10-05.txt`):

| | v2.1 (`Mat`, sequential) | v3 (`Bands`), 16 threads | v3 (`Bands`), 1 thread |
|---|---|---|---|
| 13 forward passes, median | 1.7 s | 1.4 s | 1.9 s |
| per token | ~0.13 s | ~0.11 s | ~0.15 s |
| loading (noisy) | 10-12 s | 8-12 s | 8-12 s |

  In a quieter moment earlier the same comparison gave 1.2 s against 0.7 s (~0.09 s against ~0.054 s per
  token). On one thread v3 is ~12% slower: the typed API converts y to a `Mat` and the demo converts it back
  to a list, and the bands' lists are appended at every node. The README keeps the v2 numbers until an idle
  re-measure.

### Shape checking with run-time sizes (answer to "dfdx already does this?")

dfdx (Rust, const generics; last release v0.13.0, July 2023) checks constant shapes at compile time and
falls back to `assert_eq!` at run time once a size is a `usize` (matmul, `reshape_like`). New examples:
`examples/symbolic_reshape.bend` (n from the command line, `Mat<n, 6>` → `Mat<n·2, 3>` with
`mul_assoc` as the proof, for every n), `symbolic_reshape_bad.bend` (`Mat<n, 5>`: no proof, does not
compile) and `square_transpose_bad.bend` (a 768 × 768 weight used as `X·W` instead of `X·Wᵀ`: compiles with
constant sizes, does not compile when the layer is written for any `d_in`, `d_out`). All in `check_all.py`.
README section 01 has the comparison with links to dfdx's source.

### Compile times (answer to "how bad do compile times get on deeper nets?")

`scripts/compile_times.py`, table in `bench/results/compile-times-2026-10-05.md` (the machine was not idle):
checking takes 0.22 s for 8 distinct typed layers and 0.27 s for 128; the build (mostly clang) 1.1 s and
4.7 s. GPT-2 checks in 0.38 s and builds in 5.5 s.

### Corrections

- The README said GPT-2's logits match PyTorch "within 2e-4" on 11 prompts. That bound came from v1's
  3 prompts; over the 11 prompts the largest difference is 5.8e-4 (`import numpy as np\n\n`), in v2.1 and v3
  alike (logits are of order 100; the test's tolerance is 5e-2). Fixed in the README, `docs/AUDIT.md` and
  `demos/gpt2/BENCHMARK.md`.

### Upstream drafts (Renan posts them)

- `docs/upstream/erased-last-parameter.md`: the compiler bug above, with the repro. Posted as
  [bendlang/bend#1374](https://github.com/bendlang/bend/issues/1374); fix in [#1377](https://github.com/bendlang/bend/pull/1377).
- `docs/upstream/shared-readonly-array.md` ([bendlang/bend#1376](https://github.com/bendlang/bend/issues/1376)): a read-only `Array` borrow that does not need `@unsafe`
  (`Array.fork`, the answer to #885, is `@unsafe`, and `--check-only` then reports
  `SOME PROOFS FAIL: 1 def relies on unsafe or foreign code`).

## v3.1 (2026-10-06)

Plan: `docs/v3.1-plan.md`. From now on work happens on the `devel` branch; `main` only receives releases
through a pull request, and tags are made on `main` (rule in `CLAUDE.md`).

### Exp. 12: loading GPT-2's weights

v3.0 profile: `wte` (38.6 M numbers) 1.9 s, the 12 layers (85 M) 4.8 s, ~50 ns per number, spent turning
1 MB blocks into a list of bytes, then a list of `F32`, then writing the array.

- **Decoding straight into the array** (`fill_bytes` in `demos/gpt2/fast.bend`: 4 bytes → `F32` → `Array.set`,
  no intermediate `F32` list): load 5.1-5.6 s → 3.8-4.3 s, alternated runs, the same machine state (-25%).
- **`IO.fork` does not parallelize pure work (negative result).** Loading the 12 layers with one `IO.fork`
  each, joined in order, took 6.6-7.1 s instead of 5.1-5.6 s, the same with 1 and 16 threads. A minimal test
  (`docs/upstream/io_fork_parallel.bend`: 4 forks of a flat loop of 4 G steps seeded at run time): 1.86 s with 1 thread,
  1.73 s with 16, user time equal to real time. In Bend 2.0.35 `IO.fork` interleaves computations on one core;
  the guide's "each runs its pure code (in parallel, on every core)" does not hold across forks. Parallelism
  across cores comes from parallel lets in pure code. Reverted; reported as
  [bendlang/bend#1375](https://github.com/bendlang/bend/issues/1375) (`docs/upstream/io-fork-parallel.md`).
- **Size check at the boundary:** every weight file must hold exactly `r*c*4` bytes (`check_size` with
  `File.size`), otherwise the program stops: `h3.fw.bin: expected 9437184 bytes, found 1000000; run
  reference/gpt2_prep.py --split again` (tested with a truncated copy in a temporary tree of links).

### The one-thread regression of v3.0, fixed

On one thread v3.0 was slower than v2.1 (36 forward passes: 5.2 s against 4.7 s). Two causes: the typed API
converts `y` to a `Mat` and the demo back to a list (`Bands.matvec_l` removes that: 5.0 s), and the bands
themselves (their results are appended at every node and `x` is cloned per node). With 2 threads or fewer
the demo now uses one band (`depth_for(IO.thread_count())`), which runs exactly v2.1's kernel: 4.8 s against
4.7-5.0 s for v2.1. With 16 threads: 3.6-3.8 s (v3.0: 3.8-4.2 s; v2.1: 4.7 s). Over 36 forward passes the
attention over the key/value cache, still in lists, takes a growing share, so the end-to-end gain is smaller
than the per-product gain. `bench/results/gpt2-v31-2026-10-06.txt`. The 11 prompts give the same ids and
logits as v3.0 and v2.1, digit for digit.

### Laws: one band yes, the whole tree not yet

- Proved (`bend-ml-tensor-array@0.1.5.0`): `leaf_len` (the kernel of a band puts one number per row on its
  list), `band_len` (a band of `rows` rows gives `rows` numbers), and the helper `join_len` (joining two halves
  adds their lengths). 27 laws in 5 packages.
- **Open:** "`Bands.matvec_l` gives exactly r numbers". The induction applies `join_len` to the two recursive
  results and needs the induction hypotheses about the same calls; that mentions each band's `Array` twice at
  run-time positions, and the affine checker refuses it (`x2 (consumed more than once)`, even in the rewrite
  motive). Erased copies do not help: a rewrite needs a non-erased proof, and an erased scrutinee may only be
  matched "in a dead region". Covered by tests (20 shapes, all depths, every number).
- A trick that worked: pass the expression that mentions the arrays only in an erased argument (the length
  `-s` of `len_step`, `succ_out`, `zero_out`) and the arrays themselves once.

### Robustness

- `reference/check_all.py` compiles `tensor-array/tests/par_bands.bend` to C and requires the join task of
  `bmv` (`FID_..._BMV_J..`). Checked against a mutant with the erased parameter last: no join, the check fails.
- `reference/test_tensor_array.py`: 280 checks, now with empty bands (r = 3, d = 5), one row, one column,
  blocks of one number, a block larger than the matrix, and `matvec_l` equal to `matvec` bit for bit.
- `make bench` writes `bench/results/mv_bands-<date>.txt` and `compile-times-<date>.md`.
- CI pinned to `ubuntu-24.04`, actions on Node 24 (`checkout@v7`, `setup-python@v7`, `cache@v6`,
  `upload-artifact@v7`), and it runs on `devel` too.
- Published `bend-ml-tensor-array@0.1.5.0`, hash `0x8c5737e1fd0b5ceef704ad97770f38ee`.

### Not done

- An idle-machine measurement: Renan's Android emulator and Gradle daemon ran the whole session (load average
  1-4), and they are his to close. The README chart keeps v2's numbers; `make bench` on an idle machine, then
  `docs/media/numbers.json` and `make media`, is the remaining step.
- Cheaper concatenation (a tree of lists flattened once): not needed once one thread uses one band.
- Loading in parallel: needs a pure parallel decode (for example splitting a band's `Array` by its `ANode`
  halves and decoding each half in a parallel let); not tried in v3.1, see exp. 13.

### For Renan to post upstream (all three posted on 2026-10-06; see "Upstream issues opened" below)

- `docs/upstream/erased-last-parameter.md` (the parallel let made sequential by an erased last parameter):
  [bendlang/bend#1374](https://github.com/bendlang/bend/issues/1374), fix in [#1377](https://github.com/bendlang/bend/pull/1377).
- `docs/upstream/shared-readonly-array.md` (a read-only `Array` borrow without `@unsafe`):
  [bendlang/bend#1376](https://github.com/bendlang/bend/issues/1376).
- `IO.fork` does not run pure work on several cores (exp. 12), although the guide says it does. Draft and
  repro: `docs/upstream/io-fork-parallel.md`, `io_fork_parallel.bend`: [bendlang/bend#1375](https://github.com/bendlang/bend/issues/1375).

### v3.1 measurements added at the end (2026-10-06)

- `bench/results/mv_bands-2026-10-06.txt` (not idle, load 2-4): the list API (`matvec_l`) is a little faster
  than the typed one with bands (50257 × 768, 2^5 bands, 16 threads: 5.1 ms against 6.7 ms). With **one band**,
  on the 50257-row logits matrix, `matvec_l` takes 29.6 ms against 23.5 ms for `Mat.matmul_nt` (~25% slower; ~1%
  on the smaller matrices). It is the residual one-thread cost (36 forward passes: 4.8 s against 4.7 s). Not
  investigated; probable cause: the 50k-number result built as a list and reversed, against array writes.
- Peak resident memory (`scripts/peak_rss.sh`, "Hi" + 1 token, 16 threads): v2.1 1462 MB, v3.0 895 MB,
  v3.1 903 MB. The peak *virtual* size of v3.1 is 41 GB (v2.1 and v3.0: 10.6 GB): address space reserved, not
  memory used; cause not investigated.
- `IO.fork` control: the same loops as parallel lets scale (1.69 s → 0.85 s); `docs/upstream/io_fork_parallel_control.bend`.

### Upstream issues opened (2026-10-06, by Renan)

- [bendlang/bend#1374](https://github.com/bendlang/bend/issues/1374): an erased last parameter makes a parallel let sequential (`docs/upstream/erased-last-parameter.md`).
- [bendlang/bend#1375](https://github.com/bendlang/bend/issues/1375): `IO.fork` computations run their pure work on one core (`docs/upstream/io-fork-parallel.md`; not a duplicate of #831, which was about parallel lets after closure calls and is fixed in 2.0.13).
- [bendlang/bend#1376](https://github.com/bendlang/bend/issues/1376): a read-only `Array` borrow without `@unsafe` (`docs/upstream/shared-readonly-array.md`).

### Upstream PR for #1374 (2026-10-07, by Renan)

- [bendlang/bend#1377](https://github.com/bendlang/bend/pull/1377) (from `nuxyel/bend`, branch
  `fix/1374-erased-last-fork`), announced on [#1374](https://github.com/bendlang/bend/issues/1374#issuecomment-6030940652).
  #1375 and #1376 stay with the Bend team.
- **Merged** 2026-10-07 12:29 UTC as `f76c251a`, approved by a maintainer with no requested changes; the diff
  is the one submitted, and #1374 is closed. Not in a release yet: 2.0.36 was cut before the merge, so
  bend-ml stays on 2.0.35 with its workaround until a release carries the fix.
- **Cause** (found with temporary logs in `anf`, `bend2/comp.ts`): `anf`'s `spine` cut the call prefix before
  every argument, live or erased. In `tree(q, i, z)` with `z` erased, the prefix `tree(q, i)` already carries
  every live argument, so `term_spine` reads it as a complete call; `anf` cut it into a sequential
  `h = tree(q, i)`, and the parallel let's values became plain variables (continuations, no join). With the
  erased parameter in the middle, the prefix before it is not a complete call, so nothing was cut.
- **Fix** (5 lines, +9 ttok, `comp.ts` 63,858 of its 64,000 cap): only a live argument over-applies a call, so
  the prefix is cut only before one. Side effect: a tail call into a non-flat def with an erased last
  parameter is a tail call again (`Word.add_comm.go` → `Word.add_comm.arm` lost 8 continuations).
- **Measured** (this machine, main 0ad47fc): the #1374 repro 1.013 s → 0.169 s on 16 threads; bend-ml's `bmv`
  with `-e` last gets its join (`BMV_J27`). No regression found: C/JS/.mjs/binaries for all 1,659 files under
  Bend's `tests/`, `bench/runtime/` and `demos/` byte-identical except `proof/word_add_comm` and one renamed
  local; the three lanes print the same on all 1,593 tests; the 17 runtime benches build identical binaries
  (SEQ 0.999, PAR 1.004 geomean); compile time 1.007; checker benches 0.97-1.00. Cluster gates not run.
- **Test** `tests/compile/fork_erased_last.bend`: its C gains the join, but stdout can't see a fork, so it does
  not catch a revert (said in its header and in the PR, following Bend's review practice).
- Once a Bend release carries the fix, bend-ml's workaround (a runtime parameter after `-e` in `bmv`) is no
  longer needed, but the CI check on the join stays; a version bump goes in this file (CLAUDE.md rule 1).

## Upstream outcome (2026-10-08)

- [bendlang/bend#1375](https://github.com/bendlang/bend/issues/1375) closed by
  [#1415](https://github.com/bendlang/bend/pull/1415), a guide change, not a scheduler change: IO computations
  take turns on one event loop, `IO.fork` returns a result channel and is not a CPU-parallel job, and pure work
  goes on several cores only through parallel lets. The maintainers re-ran our repro: 4 forks 1.79 s with 1 and
  16 threads, the same jobs one after another 1.80 s, two parallel lets 0.90 s with 16. So parallel loading
  has to be a pure parallel decode (exp. 13).
- [bendlang/bend#1374](https://github.com/bendlang/bend/issues/1374): fixed by
  [#1377](https://github.com/bendlang/bend/pull/1377), merged 2026-10-07; 2.0.36 (2026-10-07 04:53 UTC) was cut
  before the merge, so it waits for the next release.
- [bendlang/bend#1376](https://github.com/bendlang/bend/issues/1376) (read-only `Array` borrow): open.

### Exp. 13: a pure parallel decode for loading (negative, 2026-10-08)

With `IO.fork` ruled out (#1415), the remaining route was to read the bytes sequentially and decode them in
parallel lets. Measured on this machine (load average 1.5-2.2, background load), "Hi" + 1 token, the load
time printed by `demos/gpt2/fast.bend`, alternated runs; prototypes in `.scratch/e13/` (not committed).

- **Where the time goes.** A variant that walks each 1 MB block 4 bytes at a time without building any `F32`
  or writing the array loads in 4.2-5.0 s, the same as the real loader (4.1-5.1 s). Building the numbers and
  `Array.set` cost next to nothing; the cost is producing and walking the byte lists (`File.read_at` builds a
  `List<U32>` with one cons per byte in C, `io_list`). A variant that drops each list unread is much slower
  (11.3-13.8 s): erasing a 1 M-cell list costs more than walking it.
- **Two-phase loading** (`load_tree` reads the byte blocks of every band of a subtree into a tree shaped like
  the bands, then `decode` fills the bands with a parallel let per node; a subtree is split while its bytes
  exceed a budget). The C has the join (`FID_DECODE_J404`). Load times: budget 4 MB 4.8-5.2 s with 1 thread,
  5.7-5.9 s with 16; 16 MB 4.7-5.2 s / 5.2-5.4 s; 64 MB 5.0-5.2 s / 4.9-5.5 s. Current loader: 5.0-5.2 s /
  4.2-4.7 s. Peak resident memory with 64 MB: 1725 MB (current: 894 MB).
- **Conclusion:** no gain and twice the memory, so the stop criterion of the plan applies; the demo keeps the
  streaming loader. The walk that a parallel decode can split is a small part; the per-byte list built by the
  host and held across the IO steps is the cost. The lever left is upstream: a read that gives denser data
  (for example `File.read_at` into an `Array<U32>`, or 4 bytes per `U32`). An idea, not a draft yet.

### Bend main checked ahead of the release (2026-10-08)

- `make check-full` against Bend main `60fa05d` (2026-10-08, carries #1377; run with `bun bend2/main.ts`
  through a copy of `check_all.py` whose `BEND` points at it and whose version pin is skipped): **43/43 ok**,
  including the C join check of `bmv`, GPT-2 against PyTorch on 11 prompts and `--verdict` on every package.
  Nothing to fix before the bump.

### Checklist for the bump to the release that carries #1377

1. Read the release notes and `bend guide` for changes; `bend update`, then `bend version`.
2. `scripts/setup.sh`: `BEND_VERSION` and the two SHA256 sums (from the release's checksums).
3. `reference/check_all.py` (`check_toolchain`), `.github/workflows/ci.yml` (step name), the README badge and
   setup line, the `Bend:` line of each package README, CLAUDE.md rule 1 and the pitfalls title, this file's
   "Pinned version".
4. CLAUDE.md pitfall and README section 05: the erased-last-parameter bug becomes history (fixed in the new
   version); the C join check stays.
5. `make check-full`, `make bench` and the GPT-2 timings on an idle machine; a short entry here.
6. Packages: republish only if their source changes (the `Bend:` line alone is not a reason); then the
   release PR `devel` → `main`, tag `v1.2.1` on `main` (numbering: `docs/VERSIONING.md`).

### Idle measurement (2026-10-08, pending since v3.1)

Nothing else running (I closed Brave first), on AC power, `performance` profile; the load average of
1.2-1.5 comes from the runs themselves.

- GPT-2, 36 forward passes, alternated 5 times (`bench/results/gpt2-idle-2026-10-08.txt`): v2.1 load 5-7 s,
  1 thread 4.9-5.3 s, 16 threads 4.8-5.2 s; v3.1 load 3.8-5.4 s, 1 thread 5.2-5.5 s, 16 threads 3.7-4.3 s. On one
  thread v3.1 is 3-6% slower than v2.1 here (under load on 2026-10-06 it looked equal: 4.8 against 4.7 s).
- `make bench` (`mv_bands-2026-10-08.txt`): within a few percent of the 2026-10-06 numbers taken under load;
  logits 50257 × 768 sequential 31.8 ms, list API with 2^5 bands on 16 threads 5.5 ms. Compile times
  (`compile-times-2026-10-08.md`) are lower than under load: 128 layers check in 0.14 s and build in 3.3 s.
- Peak resident memory of the current demo: 894 MB (`scripts/peak_rss.sh`).
- README, `demos/gpt2/BENCHMARK.md` and `docs/media/numbers.json` updated (the figure now says 27 laws).

## Renumbering of the releases (2026-10-09)

My call: a major version must mean a large change in what bend-ml is, not every big step. The rule and
the old → new table are in `docs/VERSIONING.md` (v2.1 → v1.0, v3.0 → v1.1, v3.1 → v1.2; the next is v1.2.1).
New tags on the same commits; the GitHub releases moved to them with a "formerly" line; the old tags stay, and
the old `v1.0.0` is the only tag that moved (to the old v2.1.0 commit). The old `v2.1.0` tag keeps a pre-release
with only `bend-ml.mp4`, so posted links to the video still work (both URLs checked: 200). Dated entries in this
file keep the names they were written with.

From here on this file is written in the first person (I measured, my call), like the README; the entries
above keep "Renan" as they were written.

### Exp. 14: the one-thread gap against v1.0 (2026-10-09)

v1.2 (old v3.1) was 3-6% slower than v1.0 on one thread, although with ≤ 2 threads it uses one band, which was
meant to be v1.0's kernel. My earlier guess (`matvec_l` on the logits matrix) did not hold: the idle per-product
bench shows the list API with one band within a few percent of `Mat.matmul_nt`, worth ~1 ms of the ~7 ms per pass.

- **Harness** (`.scratch/e14/`, not committed): local copies of each demo printing times in ms, alternated runs,
  one thread pinned to a performance core with `taskset -c 3`. Unpinned, two copies of the same binary differed
  by 1.7% and runs spread 10% (the 155H mixes P- and E-cores); pinned, the spread is ~1-2%.
- **Bisect**, each variant puts one v1.0 piece back into the v1.2 demo (36 passes, 1 thread, median of 5):
  v1.0 4856 ms, v1.2 5134; loader of v1.0 (`get_mat`, `Mat.fill_at`) 5165; no `Array.clone` of `x` on one band
  5154; `Mat.read_row` for the embedding 5165; **products through `Mat.matmul_nt` 4961**. Only the products move it.
  Products through `matmul_nt` for the logits only: 4956; for the layers only: 4924. The two effects do not add
  up (both: 4913), and putting a single 1 × 1 `matmul_nt` in the program off the hot path gives 5161, so most of
  the gain comes from the kernel itself (writing into an `Array`, then `read_l`), not from code layout.
- **Fix** in `bend-ml-tensor-array@0.1.6.0` (hash `0x793d6e22f8bd33acb2636b33544424bc`): `bv_mat` runs `gemm`
  with n = 1 into a fresh `Array` and reads it back (`bv_gemm`, `bv_g`, `bv_g2`); `bv_rows`/`bv_leaf` are gone.
  The laws were proved again over the new code: `leaf_len` (reading `m + 1` numbers back gives `m + 1`, by
  induction on `m`), `band_len` (a band of `rows` rows gives `rows` numbers; `g_len` opens the single-constructor
  `G` so the gemm's result need not be computed). 27 laws, `--verdict` ok. A first try with
  `Nat.sub(1 + p, 1)` did not reduce to `p` in the checker; passing `p` directly fixed it.
- **Results** (one session, idle, `bench/results/gpt2-versions-2026-10-09.txt`): 1 thread v1.0 5168, v1.1 5777,
  v1.2 5509, v1.2.1 5280 ms; 16 threads v1.0 5198, v1.1 3986, v1.2 3815, v1.2.1 3689 ms; load and resident peak
  unchanged (932 MB). Per product (`mv_bands-2026-10-09.txt`): 16 threads, 2^4 bands, list API 19-32% faster;
  one band within noise. Same ids and logits; 280 tensor-array checks bit for bit.
- **Open:** ~2% on one thread against v1.0. Skipping the clone of `x` gave 0.6%, within noise.
- A correction: I first read a drop of the peak *virtual* size (41 GB → 10.6 GB) as an effect of the fix; it came
  from a shorter prompt. With the same prompt v1.1, v1.2 and v1.2.1 all reserve 41 GB.

## Next: llama.bend (2026-10-09)

I started step 1 of my plans after v1.2 in its own private repository, `nuxyel/llama.bend`: SmolLM2-135M in
Bend, using `bend-ml-tensor-array@0.1.6.0` and `bend-ml-bpe-tokenizer@0.1.2.0` from BendHub. Its findings live in
its own `NOTES.md`; bend-ml's README will mention it when it goes public.

## Leads from Stelliferous v0.0.2 (2026-10-10)

kazzzz520 renamed Bend-Conv to Stelliferous and published v0.0.2 on BendHub (`stelliferous@0.0.2.0`;
[post](https://x.com/kazzzz520/status/2108962063846645902), repository read at `ad83301`). YOLOv5n forward in
110 ms on one thread and 43 ms on four (PyTorch 49 and 15 ms; EPYC 9B45, Bend 2.0.36, Clang 19). I quoted it
([post](https://x.com/nuxyel/status/2108968007506513990)). Nothing below is measured on bend-ml yet; each item is
a lead to test, with where it comes from.

1. **Leaf work as a fork-free task** (their `docs/toolchain_rules.md`, rule H4). In the grow turn a worker that
   reaches a leaf inlined into the forking recursion runs that whole leaf there, so four leaves take the time of
   two. Their remedy: the parallel let calls a def that neither forks nor calls a closure (bit 2 of its row in
   the `FID_T` table; their `scripts/toolchain_rules/fork_flags.py` prints it), with the hot loop in a flat def
   one level below (rule F6: a def called directly by a parallel let becomes a task, so its loop is a worklist
   self-jump, not a C loop). They measured 171 → 121 ms at four threads. This may explain exp. 11's "2 bands
   never helped" and the 1.6x of 4 bands on 4 threads. Test: read `bmv`'s emitted C for which tasks are fork-free
   and where the band loop sits, move the leaf behind a fork-free def, rerun `bench/mv_bands.py` at 2, 4 and 16
   threads with 2^1 to 2^5 bands; bit-identity must hold since the kernel does not change.
2. **Clang vectorizes 8-wide records** (their `lib/tensor_lanes.bend`, `Tile8`). There is no SIMD type in Bend;
   they read 8 values into a flat record, write the 8 operations out, and write them back, compiled with
   `-O3 -march=native -ffp-contract=off`. Their GEMM is 4 output channels by 8 positions with local
   accumulators. Test on `gemm`'s inner dot: 8 partial sums instead of one. It changes the summation order, so
   `Bands.matvec` stays equal to `Mat.matmul_nt` only if both change together, and the PyTorch comparisons must
   stay within their tolerances. First check which C flags `bend` passes and whether they can be set.
3. **Inlining of small `spin_N` chains** (rule F7) and **shared constructors** (G2, G3): a helper inlined in a
   small program can become a real call per element once more operations are linked, and a value used twice can
   make every constructor of its type shared. Every build prints `N of M constructors shared`; record it for the
   GPT-2 demo and the bench.
4. **Own `exp`/`tanh`/`log` in F32** (their `lib/math_fp32.bend`): Cephes/Chebyshev polynomials with F32
   coefficients, special cases by bit selection, checked against MPFR over every F32 input (max 0.8 ULP for
   `exp`, 1.06 for `tanh`; their GELU is not accurate yet). For bend-ml: softmax and GELU that do not depend on
   the C library, and the same exhaustive check (2^32 inputs) for any scalar function I write.
5. **Multi-file packages on BendHub**: users import one file of a package, `stelliferous@0.0.2.0/tensor_f32.bend`.
   My rule "a publishable package is one file" came from the missing re-export; importing each file directly
   avoids it. Worth trying for llama.bend before its files grow.
6. **`.npy` loading** (their `lib/tensor_file.bend`): blocks of 16384 values through the official `File` effects,
   bytes gathered into little-endian words. Exp. 13 found the per-byte list to be the cost of my loader, and
   this reads the same lists, so I expect the same limit; time it on GPT-2's weights before adopting anything.
7. **Bend 2.0.36** (released 2026-10-07 04:53 UTC) does **not** carry #1377 (merged 12:29 UTC the same day), so
   the bump checklist above still waits for the next release. 2.0.36 does change the scheduler ("the work pass
   after a grow runs each lane's own ring", bendlang/bend#1132), which touches lead 1; measure on it before
   drawing conclusions from 2.0.35 alone.

## Testing the Stelliferous leads on Bend 2.0.35 and 2.0.36 (2026-10-10)

Setup: Bend 2.0.36 installed next to 2.0.35 (release tarball, SHA256 checked against the release; run through a
wrapper that sets `HOME` to `~/.cache/bend-2.0.36/home`, since the `bend` binary reads `~/.bend` and not
`BEND_HOME`). The scripts now take `BEND_BIN` (every `reference/` and `bench/` script) and `BEND_PIN`
(`check_all.py`). Brave was open during the runs (load average 0.5-3, mostly from the builds); every comparison
alternates its variants in the same session.

- **2.0.36:** `BEND_BIN=… BEND_PIN=2.0.36 make check-full` passes 43/43, `--verdict` included. It still compiles
  with `clang -std=c11 -O3` (no `-march`), and per-product times match 2.0.35 within noise (tables below).
  Base gains `Poll` and the `try_` channel and socket operations, and `Chan.send` now answers a `Result`; none of
  it touches bend-ml.

### Exp. 15: the leaf as a fork-free task (lead 1, negative)

- **Diagnosis:** in the C of `tensor-array/tests/par_bands.bend` the only task of the product is `BMV`, marked
  "may fork", with the leaf's `gemm` inlined in it: the shape rule H4 warns about. Fork flags read with
  Stelliferous's `scripts/toolchain_rules/fork_flags.py`.
- **Change tried:** a node whose two bands are leaves calls `bmv_leaf` (a def that neither forks nor calls a
  closure) in its parallel let; the C then shows `BMV_LEAF` as fork-free on both versions, and the leaf pair no
  longer clones `x`. Proofs unchanged.
- **Result** (`bench/results/exp15-leaf-task-2026-10-10.txt`, list API, 1/2/4/8/16 threads, 2^1 to 2^5 bands,
  three shapes, both versions; then the logits product with 60 products and 7 runs): old and new agree within
  ~2% everywhere. The old kernel already scaled: 768 × 768 with 2 bands goes 0.288 → 0.178 ms on 2 threads, and
  the logits product with 4 bands on 4 threads takes ~7 ms against ~26 ms on one. The effect H4 describes does
  not show in this kernel; I did not find why. Change reverted, so the package does not change.

### Exp. 16: shared constructors and inlining (lead 3, nothing to fix)

C of `demos/gpt2/fast.bend` (both versions): 128 `spin` functions, all `INLINE`, none `FAR`. 33 of them call
`ctr_take` (a match on a shared value); the ones with F32 work are list folds (a sum, a sum of squares) over
vectors of 768 or 3072 numbers. The inner loop of `dotg` has no `ctr_take`. Bend 2.0.35 does not print a
"constructors shared" line; that line comes from Stelliferous's own launcher.

### Exp. 17: the dot product with rotating accumulators (lead 2)

- **Change tried:** `dotr` keeps 4 (or 8) partial sums and rotates them at every step, so each one gets every
  4th product; the end sums them pairwise. Still one tail-recursive def; `dotg` calls it, so `gemm`,
  `Mat.matmul_nt` and the bands all use it and stay equal to each other.
- **Per product** (`bench/results/exp17-dot-accumulators-2026-10-10.txt`, 5 runs; "native" is the emitted C
  compiled by hand with `-march=native`, a ceiling that users of `bend` do not get):

| W | bands, threads | current | 4 accumulators | 8 accumulators |
|---|---|---|---|---|
| 768 × 768 | 1, 1 | 0.283 ms | 0.249 | 0.308 |
| 2304 × 768 | 1, 1 | 0.796 | 0.701 | 0.849 |
| 2304 × 768 | 2^4, 16 | 0.233 | 0.172 | 0.209 |
| 50257 × 768 | 1, 1 | 24.4 | 23.4 | 25.3 |
| 50257 × 768 | 2^4, 16 | 4.22 | 3.61 | 3.78 |

  `-march=native` did not help any variant: the loop is bound by the array access of every element (the
  `blk_loc` check, the index mask, the read), not by the vector width.
- **GPT-2** (36 passes, alternated 5 times; one thread pinned with `taskset -c 3`): 1 thread 4914 → 4686 ms
  (median, -4.6%); 16 threads 3798 → 3812 ms (no change).
- **Numbers:** the 11 prompts give the same ids and text, and the logits move closer to PyTorch: the largest
  difference drops on 9 of the 11 prompts (for example 5.8e-4 → 2.8e-4, 2.0e-4 → 8e-5). The summation order
  changes, so the logits are no longer bit for bit those of v1.2.1.
- **Adopted** (my call, 2026-10-10) with four accumulators, in `bend-ml-tensor-array@0.1.7.0`, together with the bump to
  Bend 2.0.36. Before that call: Short term: -5% on one thread, nothing on 16 for GPT-2, logits closer to
  PyTorch. Medium and long term: llama.bend and a training framework would inherit the kernel, and any future
  proof about the dot product's value would be about this order.

### Lead 4: our own exp and tanh (closed for speed)

Replacing GELU by the identity in a copy of the demo changes 36 passes on one thread from 4832 to 4803 ms
(medians of 5, within noise), and softmax runs over at most 36 positions. Neither is worth replacing for
speed; an own `exp`/`tanh` would only matter for results that do not depend on the C library.

### Lead 5: BendHub packages with several files (confirmed)

`import stelliferous@0.0.2.0/math_fp32.bend as M` works on 2.0.35: BendHub stores the whole package directory
(`~/.bend/lib/<hash>/` holds all of its `.bend` files) and the file's own `./` imports resolve inside it. A
package can have several files; what Bend lacks is re-export, so a user imports each file they call.

### Lead 6: Stelliferous's `.npy` loader (no gain)

`load_npy` of a 50257 × 768 F32 array (154 MB) takes 4.7-5.3 s, about 30 MB/s. My GPT-2 loader reads the
498 MB of weights in ~4.2 s, about 120 MB/s, so there is nothing to take for loading.

### Idle measurement for v1.2.2 (2026-10-11)

Nothing else running (one terminal session), AC power, `performance` profile. The 0.1.7.0 candidate is the
`tensor-array` source of `c2c12d9`, imported locally (it is not published yet); the bench and the demo point to it
through copies in `.scratch/`.

- **GPT-2** (`bench/results/gpt2-v122-2026-10-11.txt`), v1.2.1 kernel and candidate, each built with Bend 2.0.35
  and 2.0.36, alternated: 16 threads (7 runs) 3678 / 3671 ms for v1.2.1 and 3564 / 3516 ms for the candidate;
  1 thread pinned (5 runs) 4962 / 4920 and 4677 / 4678 ms. So 2.0.36 times like 2.0.35, and the candidate is
  4-5% faster on both thread counts. Peak resident 932 MB (v1.2.1: 936 MB). A first batch of 5 runs right after
  the builds gave 4161 against 4058 ms on 16 threads, with a wide spread (min 3626); I left it out and reran with
  more runs.
- **Per product** (`bench/results/mv_bands-2026-10-11.txt`, against `mv_bands-2026-10-09.txt`, different
  sessions): `Mat.matmul_nt` on one thread 5-12% faster (logits 31.6 → 27.9 ms); list API with 2^4 bands on 16
  threads 10-21% faster (logits 4.0 → 3.2 ms).
- **Compile times** (`bench/results/compile-times-2026-10-11.md`, Bend 2.0.36): 128 layers check in 0.13 s and
  build in 3.3 s, as on 2.0.35.
- README, `demos/gpt2/BENCHMARK.md` and `docs/media/numbers.json` updated (the charts do not show the Bend version,
  so the media were not rendered again).
- **Published:** `bend-ml-tensor-array@0.1.7.0`, hash `0x373e5f0e1b2ccdc5e9e1fe2004c1644c` (2026-10-11). The GPT-2 demo,
  `bench/mv_bands*.bend` and the README table now use it; `test_published.py` reads the version from the README.

## Evidence run of every release (2026-10-11)

I measured every release from v0.2.0 to v1.2.2 in one session, next to PyTorch, NumPy, llama.cpp, llm.c, candle and dfdx,
with one protocol. Harness, raw data and report are kept privately (material for a paper). Every release gives the same GPT-2 tokens as PyTorch; the worst logit
difference is 5.82e-4 from v0.3 to v1.2.1 and 3.20e-4 in v1.2.2; the law count goes 21, 23, 24, 25, 27, 27, 27, and every
package of every release passes `--verdict`.

- **Debt: `demos/mnist/train.bend` has not type-checked since v0.3.** It imports `bend-ml-tensor@0.1.2.0`, while
  `bend-ml-autograd@0.1.1.0` was published against `bend-ml-tensor@0.1.0.0`, and Bend treats the two `Mat` types as
  different types (`expected 0xf983…/main.Mat<784n, 128n>, observed T.Mat<784n, 128n>` in `s5`). No check builds it, so
  nothing caught it; `demos/mnist/README.md` still calls it the kept baseline. Fix: import the tensor version autograd was
  built on (or publish autograd against 0.1.2.0), and add a build of `train.bend` to `check_all.py`. Only v0.2's list
  baseline could be measured: 885 ms per batch, or 531 s for 600 batches, close to the 544 s per epoch on record.
- **Fixed (2026-10-11).** `demos/mnist/train.bend` imports `bend-ml-tensor@0.1.0.0`, the version `bend-ml-autograd@0.1.1.0`
  was published against; the translation to 0.1.2.0 changed no API, only the comments, so nothing else moves. It gives the
  same loss and hits as PyTorch after 10 batches (2.2162626 and 4934/10000). `check_all.py` now builds both list baselines
  (`gpt2.bend`, `train.bend`) in the quick check, and runs those 10 batches in the full one: 40 and 48 checks, all passing.

### README and media rewritten around the evidence (2026-10-11)

The README still had the structure of the first proof of concept. It now leads with the evidence of every release: frames
in the launch video's identity rendered by `docs/media/deck.mjs` from `numbers.json` ("evidence", copied from the run by a
script; the source tables are kept privately with the raw data). The two PoC charts (`benchmarks.svg`, `findings.svg`)
are retired; the launch video stays linked as the v1.0 video and is rendered only with `make media` `--video`.
- **Correction (2026-10-11).** Exp. 17 above and the v1.2.2 notes said the largest logit difference drops on 9 of the 11
  prompts. Counted prompt by prompt (the same values), it drops on 10: only "The capital of France is" moves the other way
  (8.0e-5 → 9.0e-5). The worst case over all prompts goes from 5.82e-4 to 3.20e-4.

## Next steps (2026-10-11)

- **Bend 2.0.37**: done on 2026-10-13 (below), except `bmv`'s comment, which waits for the next change to
  tensor-array's source.
- **The per-element cost of `Array`.** On every product the time goes to the array access of each element (locating the
  block, the index mask, the load; corrected on 2026-10-13, below), not to the multiply-add: four partial sums gave 4-5%, eight were slower,
  and `-march=native` changed nothing (exp. 17). Two directions: storage the C compiler can vectorize (a contiguous leaf
  of several numbers per access, or a primitive for a dense block), and a read-only borrow of an `Array` across a
  parallel let, which would remove the per-split clone of the input vector (bendlang/bend#1376, open). Measure each
  against `bench/mv_bands.py` and the GPT-2 timings before changing a published package.
- **llama.bend** depends on both: its next step (a key/value cache in flat Arrays) is in its own NOTES.

## Bend 2.0.37 (2026-10-13)

Bend 2.0.37 carries my #1377 (an erased last parameter no longer makes a parallel let sequential) and #1479 (a slow
grow's forks go to idle threads). The checklist above, step by step:

- **Pin.** `scripts/setup.sh` (version and both SHA256 sums), `check_all.py`, the CI step, the README badge and setup
  line, the `Bend:` line of each package README, CLAUDE.md and "Pinned version". The Python venv had to be rebuilt: it
  still pointed at the repository's old path.
- **Pitfall.** CLAUDE.md and the README's upstream section say the bug is fixed in 2.0.37; the C join check stays.
- **`bmv`'s comment stays.** It sits in tensor-array's source, which is published as 0.1.7.0: changing a comment would
  make the repository differ from the package, and a comment alone is not worth a new version. It goes with the next
  real change to tensor-array. The parameter order it explains is still valid.
- **`make check-full`: 48/48**, with "Bands.matvec compiles to a parallel join".
- **GPT-2 timings, the same commit (`124a0aa`) built with 2.0.36 and 2.0.37**, 7 alternated rounds in one session on
  mains power (the evidence harness, `gpt2` block; medians and interquartile ranges, ms per token): one thread 129.8
  (121.8-145.9) on 2.0.36 against 130.1 (122.9-134.0) on 2.0.37; 16 threads 111.0 (102.2-118.2) against 117.1
  (106.3-122.2). No difference beyond the spread. For #1377 that is expected (`bmv` already had the workaround); I did
  not look further into why #1479 changes nothing here. The 2.0.37 binary is less than half the size (0.41 MB
  against 0.93 MB).
- **Packages.** No source changed, so none is republished.

## Correction: an `Array` access is not a tree descent (2026-10-13)

These notes (exp. 17, "Next steps") said that each element of a product costs "a depth-bounded descent, an index mask, a
load". Wrong about the descent. In Base, `Array` is defined as a binary tree (`ALeaf`/`ANode`), but the compiler
recognizes the array operations and emits a flat block: `Array.get` becomes `blk_loc` (the block's address, following
one redirect when the array is shared), `blk_at` (the index masked to the block's size) and `blk_read` (a packed 32-bit
load for `F32`). Bend v2.0.37, `bend2/comp.ts`: `arr_op` and `arr_cells` (lines 1730-1780), `blk_at` (4045), `blk_read`
(4030). What exp. 17 measured stands: the loop is bound by the access, not by the floating-point unit, and
`-march=native` changed nothing for one element per access. What changes is the direction: a contiguous buffer already
exists, so the lead is to read several numbers per access. A record of 8 `F32` per array cell (`Tile8`), with the same
four rotating sums (the same order of additions), made an isolated 49152x576 product about 3x faster on one thread, and
clang emitted packed SSE for it (a Codex research run, reproduced here: 313-326 ms against 88-116 ms for 20 products,
indicative timings). Not integrated or measured in a demo yet.

## Upstream: a dense file read (2026-10-14)

Exp. 13's lever, taken upstream: [bendlang/bend#1573](https://github.com/bendlang/bend/issues/1573) asks for a file read
into an `Array<U32>` (four little-endian bytes per word) instead of a `List<U32>` with one cell per byte. Draft and repro
in `docs/upstream/` (`dense-file-read.md`, `read_at_bytes.bend`, `read_at_bytes.c`). The standalone repro, Bend 2.0.37,
one thread pinned to CPU 3, five alternated runs: reading and summing 384 MiB in 1 MiB blocks takes 1.52-1.57 s with
`File.read_at` and 56-58 ms with `pread` in C, the same sum. The work per byte is one addition in both, so the gap is
the list. The gain would be in loading only (llama.bend's 1.7B: 66 s); the cost per token does not change.

## Products in cells of 8 (tensor-array 0.1.8.0; 2026-10-15)

Products with two or more rows pack A's rows and B's columns once into cells of 8 F32 (`T8`), so the inner loop reads 8
numbers per array access instead of one (the lead of 2026-10-13, checked first with sizes read at run time: 3.0-3.3x on
an isolated product). The additions keep `dotr`'s order, so every result is the same bits as 0.1.7.0's apart from NaN
payloads (`tensor-array/tests/t8.bend`); one-row products (`Bands.matvec*`, llama.bend's direct `gemm` calls) keep the
scalar loop. Result, on an MLP training epoch on MNIST (784-128-10, SGD, batches of 100, 5 alternated rounds): 8.7 to
4.8 s on one thread, 6.0 to 4.7 s on 16; the exact outputs of that training program and of llama.bend unchanged. A first version that built each
cell through a list was 2.3x slower than 0.1.7.0: packing has to read the 8 numbers directly. The parallel path still
copies its operands per block (next: B2).

- Published: `bend-ml-tensor-array@0.1.8.0`, hash `0x3283e28131a46d2d98db11347f49016c` (`--verdict` ok, no `@unsafe`). From a
  clean folder, `tests/t8.bend` against the published package gives `0 0 0 0 0 0 0 0`; `reference/test_published.py` ok.
  `bench/mv_bands*.bend` point at it; `demos/gpt2/fast.bend` waits (it has uncommitted local changes).

## The parallel products without per-block copies (tensor-array 0.1.9.0; 2026-10-15)

The parallel path in cells of 8 copied both packed operands at every node of the fork tree. Now the larger operand
(A's rows or B's columns, whichever is longer) is packed straight into one piece per block, and the smaller one is
copied once per node (`2^d - 1` copies); the split is by rows or by columns of C to match. The inner loop also passes
its four sums as arguments instead of building a record per step. Same bits (`tests/t8.bend`, and the training program's
exact outputs). The MNIST training epoch, 5 alternated rounds: 4.9 to 3.9 s on one thread, 5.0 to 4.0 s on 16. More threads still add
little at this size: a 100 x 784 by 784 x 128 product is about 1.3 ms of work, and splitting it in 8 measured 1.5x
faster at best in isolation (a version where every leaf owns all its data reached 2.8x), so the fork, the packing and
the writes of C weigh as much as the work. A coarser grain (a whole step, or several products at once) is the lead.

- Published: `bend-ml-tensor-array@0.1.9.0`, hash `0x4b971e2f6428dc3f80dcd5a04dedc5f2` (`--verdict` ok, no `@unsafe`). From a
  clean folder, `tests/t8.bend` against the published package gives `0 0 0 0 0 0 0 0`; `reference/test_published.py` ok.

## v1.2.5 and v1.2.6 (2026-10-16)

v1.2.5 released the faster products. It also carried, by mistake, a work-in-progress `demos/gpt2/fast.bend` whose loader
imports a file that is not in the repository, so the GPT-2 demo did not compile. CI did not catch it: the full-checks
step piped `make check-full` through `tee`, which hid the exit status, and the job passed with 48 of 50 checks. v1.2.6
puts the demo back to v1.2.4's loader and runs that step with `pipefail`; its CI ran 50/50, including the GPT-2 demo
against PyTorch. v1.2.5's release notes point to v1.2.6.

## Accumulating products (tensor-array 0.1.10.0; 2026-10-16, not published yet)

A training loop accumulates its weight gradient, `dW += Xᵀ · dY`, and with the products it had to do that in two steps:
`Mat.matmul_tn` into a new matrix, then `Mat.add` into the gradient, a second pass over every weight plus an allocation.
`Mat.matmul_acc`, `Mat.matmul_nt_acc`, `Mat.matmul_tn_acc` and `Mat.col_sums_acc` take the destination and add each cell
of the product into it. The cell is computed exactly as before and added once, `c + x`, so the result has the bits of
`Mat.add(c, A·B)`: `tests/acc.bend` checks it word for word on every path (one row, scalar and parallel by columns;
cells of 8, sequential, parallel by rows and by columns), every layout and every k from 0 to 19, with special values in
C; `reference/test_tensor_array.py` compares the four functions with NumPy. The code is one flag through the writers of
C; `gemm` and `gemm_t8` keep their signatures and the other functions their bits.

`dW += dYᵀ·X` repeated, idle machine, 9 alternated rounds, medians, one thread:

| n x o x i | two steps | accumulating | ratio |
|---|---|---|---|
| 32 x 1536 x 576 (an MLP projection of SmolLM2-135M, 32 tokens) | 260 ms | 184 ms | 0.71 |
| 8 x 1536 x 576 | 359 ms | 159 ms | 0.44 |
| 128 x 1536 x 576 | 176 ms | 152 ms | 0.86 |
| 100 x 128 x 784 (MNIST's first layer) | 175 ms | 154 ms | 0.88 |

The pass it removes is proportional to the weights and the product to the weights times the rows, so the gain grows as
the batch shrinks. On an MLP training epoch on MNIST the whole run is 3% faster (4.25 to 4.12 s, 15 rounds). At these
sizes the parallel product (8 blocks, 16 threads) was slower than one thread for both versions (455 against 260 ms at
32 x 1536 x 576): with few rows, splitting and joining costs more than it saves.

The same version also changes how a new array's capacity is computed. `cap_depth(n)`, the smallest `d` with
`n <= 2^d`, came from `cap_pair`, a recursion on `n` (not a tail call) where each step compares two unary numbers, up to
`n` steps: quadratic in `n` when it runs at run time. In the JavaScript build it overflowed the stack on any array of more
than ~40,000 numbers, and with a larger stack it was 98% of the time of an MNIST run (247 s for 50 batches against 4.3 s
with the fix). `cap_dbl` doubles a power of two until it holds `n`, log2(n) steps in a tail loop. Its fuel is `n`,
always enough; if it ran out the answer would be `cap_pair`'s, so the law `cap_ok` (`n <= 2^cap_depth(n)`) is proved by
the same lemmas and no arithmetic about the number of doublings (`--verdict` ok). `tests/cap.bend` checks that both give
the same depth for every `n` up to 2000 and that the depth is the smallest at four large sizes. Native builds gain
too, a little: an MNIST training epoch 4.26 to 4.19 s on one thread.

