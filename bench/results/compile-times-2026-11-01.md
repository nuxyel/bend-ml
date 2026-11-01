`python3 scripts/compile_times.py`, median of 3 runs, Bend 2.0.39, clang 23.1.3, Intel Core Ultra 7 155H, nothing else running (2026-11-01, load1 1.35 decaying from the matrix · vector run before it).

| program | size | check | build (incl. clang) |
|---|---|---|---|
| 8 dense layers, all sizes distinct | 89 lines | 0.12 s | 0.71 s |
| 32 dense layers, all sizes distinct | 305 lines | 0.13 s | 1.07 s |
| 128 dense layers, all sizes distinct | 1169 lines | 0.14 s | 3.09 s |
| MNIST training (demos/mnist/fast.bend) | 342 lines | 0.16 s | 1.64 s |
| GPT-2 small inference (demos/gpt2/fast.bend) | 559 lines | 0.23 s | 3.56 s |
