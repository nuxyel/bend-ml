# Examples

## The batch size comes from the input file

`runtime_batch.bend` runs one dense layer, `H = relu(X·W + b)`, on a batch whose size the program only learns when it reads its input. The file's first number is the batch size `n`, followed by `n` rows of 4 numbers:

```
3
0.5 1.0 -0.5 2.0
1.5 -2.0 0.25 0.0
-1.0 0.5 3.0 1.0
```

```
$ bend examples/runtime_batch.bend -- examples/data/batch3.txt
batch size read from the file: 3
X : Mat<3, 4>
H = relu(X·W + b) : Mat<3, 3>
  1.6750001 0 0
  0 0 1.675
  0 3.5 2.85
```

The shapes are still checked at compile time. The layer is written once for every batch size:

```python
def layer(+n: Nat, x: TA.Mat<n, 4n>) -> TA.Mat<n, 3n>:
  TA.Mat.relu(n, 3n, take_b(n, TA.Mat.add_row(n, 3n, take_c(n, TA.Mat.matmul(n, 4n, 3n, 0n, x, weights())), bias())))
```

The checker never needs the value of `n`, only that the 4 columns of `X` meet the 4 rows of `W`. The one thing that can go wrong at run time is the file holding the wrong amount of numbers, and it is checked once, where the data enters: `Mat.of_list(n, 4n, xs)` returns `Some(Mat<n, 4>)` or `None`.

```
$ bend examples/runtime_batch.bend -- examples/data/wrong_count.txt
error: the file says the batch has 4 rows of 4 numbers (16 in total), but it holds 12
```

`runtime_batch_bad.bend` is the same layer with `W` of the wrong shape. It does not compile, even though `n` is unknown; note `n : Nat` in the context, with no value:

```
$ bend examples/runtime_batch_bad.bend
SOME PROOFS FAIL
Error:
- expected : TA.Mat<4n, 3n>
- observed : TA.Mat<5n, 3n>
Context:
- n : Nat
- x : TA.Mat<n, 4n>
Location: layer
```

`reference/test_runtime_batch.py` writes files with batch sizes from 1 to 200, compares the output with NumPy, and checks that files whose count disagrees with their header are rejected. Both run in `make check`.
