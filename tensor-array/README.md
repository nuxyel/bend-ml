# bend-ml-tensor-array

Tensores em Bend 2 sobre `Array<F32>` plano, **com a shape no tipo**: a mesma garantia do [`bend-ml-tensor`](../tensor) (produto com dimensões erradas não compila), mas **~50× mais rápido** que o `bend-ml-tensor` baseado em listas, porque lê e escreve por índice e roda produtos em blocos de linhas em paralelo.

- Bend: **2.0.35** · Licença: MIT.
- `bend tensor-array/main.bend` e `--verdict` → `ALL PROOFS CHECK` (sem `@unsafe`, sem `?TODO`).
- Testes contra o PyTorch: `reference/test_tensor_array.py` (39 verificações, erro máximo ≈ 1,4e-6).

```python
import Base
import bend-ml-tensor-array@0.1.0.0/main.bend as TA

def prod(a: TA.Mat<100n, 784n>, w: TA.Mat<784n, 128n>) -> TA.MMul<100n, 784n, 128n>:
  TA.Mat.matmul(100n, 784n, 128n, 3n, a, w)   # 3n = 2^3 = 8 blocos paralelos
```

## Como funciona

- `Mat<r, c>` guarda `r*c` números num `Array<F32>` linha a linha (índice `i*c + j`). As dimensões são parâmetros de tipo **apagados**.
- Um `Array` é afim (um só dono), então **toda operação devolve também o que leu**: `Mat.matmul(a, b)` devolve `MMul{a, b, c}`. Use `Mat.clone` quando precisar de duas cópias.
- `par` (nos produtos) é o log2 do número de blocos de linhas calculados em paralelo; `0` é sequencial. Cada bloco trabalha numa cópia (`Array.clone`, barato) de `A` e `B`.

## Operações

| Operação | Tipo |
|---|---|
| `Mat.zeros`, `Mat.fill`, `Mat.of_list`, `Mat.to_list`, `Mat.clone` | `Mat<r,c>` (`of_list` só devolve `Some` se a lista tem `r*c` números) |
| `Mat.matmul` | `Mat<n,k> · Mat<k,m> = Mat<n,m>` |
| `Mat.matmul_nt` | `Mat<n,k> · Mat<m,k>ᵀ = Mat<n,m>` (pesos com uma linha por saída) |
| `Mat.matmul_tn` | `Mat<k,n>ᵀ · Mat<k,m> = Mat<n,m>` (gradiente de pesos: `Xᵀ · dY`) |
| `Mat.add_row` | soma um bias `Mat<1,m>` a cada linha de `Mat<n,m>` |
| `Mat.relu`, `Mat.relu_bwd` | ativação e o gradiente que a atravessa |
| `Mat.sgd`, `Mat.add` | `w - lr·dw` e `a + b` |
| `Mat.col_sums` | `Mat<n,m> -> Mat<1,m>` |
| `Mat.softmax_ce` | entropia cruzada média e seu gradiente `(softmax − one-hot)/n` |
| `Mat.count_correct` | acertos de argmax por linha |
| `Mat.read_row`, `Mat.write_row` | uma linha como lista |

## Erro de shape = erro de tipo

`(2×3) · (4×5)` não compila (`tensor-array/tests/bad_matmul.bend`):

```
Error:
- expected : TA.Mat<3n, 5n>
- observed : TA.Mat<4n, 5n>
```

E o gradiente de uma camada não fecha com as dimensões trocadas (`tests/bad_grad.bend`): `matmul_tn(x, dy)` com `x: 100×784` e `dy: 100×128` é `784×128`; pedir `128×784` dá `expected MMulTN<128n,100n,784n>, observed MMulTN<784n,100n,128n>`.

## Desempenho (1 thread, 100 M multiplicações-e-somas)

| | tempo | mult-soma/s |
|---|---|---|
| `bend-ml-tensor` (listas) | 2,27 s | 44 M |
| `bend-ml-tensor-array` (`Array.get`/`set` por índice) | **0,046 s** | **2 200 M** |
| PyTorch (1 thread, BLAS) | 0,0016 s | 62 000 M |

Com blocos paralelos o produto grande de MNIST (100×784·784×128) ganha ~2× em 8 threads. O PyTorch continua ~27× à frente num thread: o Bend 2.0.35 gera código escalar, sem BLAS nem SIMD. Veja `demos/mnist/BENCHMARK.md` para o treino completo.

## O que NÃO é provado

- A invariante "a capacidade do `Array` é `>= r*c`" vale porque os construtores (`zeros`, `fill`, `of_list`) alocam o tamanho certo; ela não é um fato no tipo.
- `softmax_ce` e `count_correct` esperam `labels` com `n` entradas (uma por linha); não é checado pelo tipo.
- A numérica em `F32` é validada por testes contra o PyTorch, não por prova.
