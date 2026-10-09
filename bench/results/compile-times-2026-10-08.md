`python3 scripts/compile_times.py`, median of 3 runs, Bend 2.0.35, clang 22.1.8, Intel Core Ultra 7 155H, nothing else running (2026-10-08).

| program | size | check | build (incl. clang) |
|---|---|---|---|
| 8 dense layers, all sizes distinct | 89 lines | 0.12 s | 0.70 s |
| 32 dense layers, all sizes distinct | 305 lines | 0.13 s | 1.10 s |
| 128 dense layers, all sizes distinct | 1169 lines | 0.14 s | 3.31 s |
| MNIST training (demos/mnist/fast.bend) | 342 lines | 0.13 s | 1.89 s |
| GPT-2 small inference (demos/gpt2/fast.bend) | 588 lines | 0.21 s | 4.10 s |
