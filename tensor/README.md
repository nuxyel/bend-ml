# bend-ml-tensor

Tensores em Bend 2 com a **shape no tipo**: um erro de shape é um erro de tipo, não uma falha em tempo de execução.

- Bend: **2.0.35** · Licença: MIT · Depende de `bend-ml-nat-lemmas@0.1.0.0`.
- `bend tensor/main.bend` e `bend tensor/main.bend --verdict` → `ALL PROOFS CHECK` (sem `@unsafe`, sem `?TODO`).

```python
import Base
import bend-ml-tensor@0.1.0.0/main.bend as T

def prod() -> T.Mat<2n, 4n>:
  T.Mat.matmul(2n, 3n, 4n, T.Mat.fill(2n, 3n, 2.0), T.Mat.fill(3n, 4n, 3.0))
```

## Tipos

- `Vec<n>`: vetor com `n` números `F32`.
- `Mat<r, c>`: matriz `r × c` (lista de `r` linhas com `c` números).
- As dimensões são parâmetros de tipo **apagados**: não custam nada em tempo de execução, mas o checker as confere em toda operação.
- `Mat.of(r, c, linhas)` e `Vec.of(n, lista)` só devolvem um valor (`Some`) se os tamanhos reais batem.

## Operações

`Mat.add/sub/mul/scale`, `Mat.matmul`, `Mat.transpose`, `Mat.relu`, `Mat.gelu` (tanh, como o GPT-2), `Mat.softmax` (por linha, estável), `Mat.layernorm`, `Mat.add_row` (soma um bias `Vec<m>` a cada linha), `Mat.col_sums`, `Mat.matvec`, `Mat.reshape`, `Mat.flatten`, `Mat.reshape_swap`; e `Vec.add/sub/mul/scale/dot/sum/relu/softmax`.

## Erro de shape = erro de tipo

`(2×3) · (4×5)` não compila (`tensor/tests/bad_matmul.bend`):

```
Error:
- expected : T.Mat<3n, 5n>
- observed : T.Mat<4n, 5n>
```

`reshape` de 2×6 (12 elementos) para 5×3 (15) é recusado, porque não existe a prova `12 == 15` (`tensor/tests/bad_reshape.bend`):

```
Error:
- expected : 12n
- observed : 15n
```

`reshape` de 2×6 para 3×4 compila porque a prova é só calcular (`{==}`).

## LAWS provadas (em português)

| Lei | O que afirma |
|---|---|
| `reshape_swap` | `r × c` e `c × r` têm o mesmo número de elementos: `r*c = c*r` (usa `mul_comm` do nat-lemmas). Por isso `Mat.reshape_swap` sempre existe. |
| `reshape_flat` | `r × c` tem o mesmo número de elementos que `1 × (r*c)`: `r*c = 1*(r*c)` (usa `mul_one_l`). Por isso `Mat.flatten` sempre existe. |

Além das leis, **o próprio tipo de `Mat.matmul` é uma garantia**: `Mat<n,k> → Mat<k,m> → Mat<n,m>`; e `Mat.reshape` só aceita um argumento que é uma prova de `r1*c1 = r2*c2`.

## O que NÃO é provado

- Os números: `F32` não é um número real (arredondamento), então a correção numérica é validada por testes contra o PyTorch, não por prova (`reference/test_tensor.py`): `matmul`, `transpose`, `relu`, `softmax` (inclusive com valores enormes), `gelu` e `layernorm` batem com erro máximo da ordem de 1e-6.
- A invariante interna "cada linha tem exatamente `c` números" vale porque as operações a preservam e os construtores a verificam (`Mat.of`); ela não é uma prova no tipo.

## Desempenho

Os dados são listas (copiáveis e sem índices com custo logarítmico): ≈ 44 milhões de multiplicações-e-somas por segundo numa thread. Veja `demos/mnist/BENCHMARK.md` quando existir.
