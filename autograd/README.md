# bend-ml-autograd

Diferenciação automática em Bend 2, com uma **lei provada**: o modo reverso (o que o treino de redes usa) calcula a mesma derivada que o modo direto.

- Bend: **2.0.35** · Licença: MIT · Depende de `bend-ml-nat-lemmas@0.1.0.0` e `bend-ml-tensor@0.1.0.0`.
- `bend autograd/main.bend` e `--verdict` → `ALL PROOFS CHECK` (sem `@unsafe`, sem `?TODO`).

## O que tem

1. **Modelo provado (sobre `Nat`)**: expressões `NE` com uma variável `X`, constantes, soma e produto; `nval`, `nfwd` (derivada em modo direto, com números duais) e `nbwd` (modo reverso, passando o gradiente de cima para baixo).
2. **Autograd de escalares em `F32`** (estilo micrograd): `G` com `GCst`, `GVar`, `GAdd`, `GMul`, `GRelu`, `GTanh`, `GExp`; `gval` (valor) e `ggrad` (gradiente em relação a cada variável).
3. **Camadas com tensores**, com o formato de cada gradiente garantido pelo tipo: `linear` / `linear_bwd`, `relu_bwd`, `ce_loss` / `ce_grad` (entropia cruzada), `sgd`, `sgd_vec`.

## LAW provada (em português)

| Lei | O que afirma |
|---|---|
| `reverse_eq_forward` | Para qualquer expressão feita de constantes, `X`, somas e produtos, e para qualquer `x`: o gradiente em modo reverso, começando com gradiente 1 na saída, é igual à derivada em modo direto. |

Ideia da prova (os comentários em `main.bend` detalham): provamos algo mais forte, `reverso(e, g) = g × direto(e)`, por indução na expressão. Na soma, a distributividade junta as duas metades; no produto, `(g·vb)·fa + (g·va)·fb = g·(fa·vb + va·fb)` segue de associatividade, comutatividade e distributividade, todas vindas do `nat-lemmas`. O caso `g = 1` dá a lei.

## O que NÃO é provado (e como é verificado)

- A lei vale para `Nat` (um semianel comutativo). O `F32` tem a mesma estrutura, mas arredonda, então **a correção numérica do `F32` é verificada por gradient checking contra o PyTorch**, não por prova.
- Operações não polinomiais (`relu`, `tanh`, `exp`) não entram na lei.
- Formato dos tensores: `linear_bwd` retorna `dx: Mat<n,i>`, `dW: Mat<i,o>`, `db: Vec<o>`; um produto com dimensões trocadas não compila. As listas internas continuam sendo uma invariante de biblioteca (ver `bend-ml-tensor`).

## Gradient checking (contra o PyTorch)

`reference/test_autograd.py` compara o Bend com o autograd do PyTorch: 5 expressões escalares em 3 pontos cada (com `relu` dos dois lados do zero), `linear_bwd` (3 formas), `relu_bwd` e entropia cruzada (perda e gradiente nos logits). 21 verificações, erro máximo ≈ 5e-7, 0 falhas:

```bash
reference/.venv/bin/python reference/test_autograd.py
```
