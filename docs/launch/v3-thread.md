# v3 follow-up post (draft; Renan posts it, as a reply to the launch post)

Replies to the questions on the first post, with numbers. Re-check the numbers against `NOTES.md` exp. 11
before posting.

---

1/ bend-ml v3: answers to your questions.

"dfdx already does compile-time shapes?" for constant sizes, yes. once a size is a runtime usize it falls
back to assert_eq! at run time. in Bend a runtime size stays a variable in the type: reshape Mat<n,6> ->
Mat<n*2,3> compiles once, for every n, with a proof that n*(2*3) = (n*2)*3.

2/ the bug that fits by accident: a 768x768 weight used as X·W instead of X·Wᵀ. with constant sizes nothing
can catch it. write the layer for any d_in, d_out and it stops compiling, even if you only ever call it with
768 and 768. examples/square_transpose_bad.bend

3/ "compile times on deeper nets?" 128 dense layers, every size different: type-check 0.27 s (8 layers:
0.22 s). the build is mostly clang: 4.7 s.

4/ speed: @kazzzz520 was right, copying was the cost. weights now live in row bands, each its own array, and
each thread takes its band without a copy. the split is in the type, with a proof the bands cover every row.
GPT-2's 50257x768 logits product: 25 ms -> 6 ms on 16 threads, bit-identical.

5/ two Bend 2.0.35 traps I hit (reports going upstream): an erased parameter at the end of a function's
parameter list makes its parallel let run sequentially; a non-tail recursive helper inside the fork tree
halves the gain.

github.com/nuxyel/bend-ml · bend-ml-tensor-array@0.1.4.0 on BendHub
