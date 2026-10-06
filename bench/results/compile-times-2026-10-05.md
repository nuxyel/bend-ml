# Check and build times (2026-10-05)

`python3 scripts/compile_times.py`, median of 3 runs, Bend 2.0.35, clang 22.1.8, Intel Core Ultra 7 155H. The machine was not idle (an Android emulator was running), so treat the build column as an upper bound.

| program | size | check | build (incl. clang) |
|---|---|---|---|
| 8 dense layers, all sizes distinct | 89 lines | 0.22 s | 1.11 s |
| 32 dense layers, all sizes distinct | 305 lines | 0.22 s | 1.57 s |
| 128 dense layers, all sizes distinct | 1169 lines | 0.27 s | 4.71 s |
| MNIST training (demos/mnist/fast.bend) | 342 lines | 0.24 s | 2.70 s |
| GPT-2 small inference (demos/gpt2/fast.bend) | 541 lines | 0.38 s | 5.46 s |
