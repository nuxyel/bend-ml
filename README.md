# bend-ml

**Machine learning em [Bend 2](https://bend-lang.com), onde um erro de shape é um erro de tipo.**

Quatro pacotes publicados no BendHub, uma MLP no MNIST e o GPT-2 small (124 M) inteiros em Bend, com as provas verificadas pelo kernel do Bend e os números conferidos contra PyTorch e `tiktoken`. Versão do Bend: **2.0.35**.

```python
# (2x3) · (4x5): não compila
def bad() -> T.Mat<2n, 5n>:
  T.Mat.matmul(2n, 3n, 5n, T.Mat.fill(2n, 3n, 1.0), T.Mat.fill(4n, 5n, 1.0))
```
```
Error:
- expected : T.Mat<3n, 5n>
- observed : T.Mat<4n, 5n>
```

## Pacotes no BendHub

| Pacote | Versão | O que é | LAWS provadas |
|---|---|---|---|
| [`bend-ml-nat-lemmas`](nat-lemmas) | 0.1.0.0 | Lemas de `Nat` e `List` que a Base não tem | `add_comm`, `add_assoc`, `mul_comm`, `mul_assoc`, `mul_dist`, `append_assoc`, `length_append`, `product_append`... (15) |
| [`bend-ml-bpe-tokenizer`](bpe) | 0.1.0.0 | Tokenizer BPE byte-level (estilo GPT-2) | **roundtrip** `decode(encode(s)) = s`, `vocab_bound`, `dec_append` |
| [`bend-ml-tensor`](tensor) | 0.1.1.0 | `Vec<n>` e `Mat<r,c>` com shape no tipo | `reshape_swap`, `reshape_flat`; `reshape` só compila com a prova de que o nº de elementos não muda |
| [`bend-ml-autograd`](autograd) | 0.1.0.0 | Diferenciação automática e camadas com backward tipado | **`reverse_eq_forward`**: o modo reverso do autodiff dá o mesmo que o modo direto |

```python
import bend-ml-tensor@0.1.1.0/main.bend as T
import bend-ml-bpe-tokenizer@0.1.0.0/main.bend as BPE
```

## Demos

| Demo | Resultado |
|---|---|
| [MNIST](demos/mnist) (MLP 784-128-10) | Em Bend: perda `0,52047706`, **9129/10000**. PyTorch com os mesmos pesos e lotes: `0,520477`, **9129/10000**. Bend leva 544 s por época contra 0,3 s do PyTorch (ver o [benchmark honesto](demos/mnist/BENCHMARK.md)). |
| [GPT-2 small](demos/gpt2) (124 M) | Gera os **mesmos tokens** que o PyTorch (22 tokens, 3 prompts), logits a menos de 2e-4. ~3 s por token, 10 s para carregar os pesos ([detalhes](demos/gpt2/BENCHMARK.md)). Tokenizer idêntico ao `tiktoken` em 16/16 textos. |

```
$ ./gpt2 "The capital of France is" 8
  id 262  logit -100.24986 ...
texto: The capital of France is the capital of the French Republic, and
```

## O que é provado e o que é testado

- **Provado pelo kernel** (`bend X.bend --verdict`): as leis acima. Sem `@unsafe`, sem `?TODO`, em nenhum pacote.
- **Pelo tipo**: shapes de `matmul`, de cada gradiente de camada (`dW: Mat<i,o>`) e `reshape` com prova.
- **Testado, não provado**: toda a numérica em `F32` (não é um número real; arredonda). Gradient checking e comparação com PyTorch em `reference/`. O tokenizer em ASCII é exato; bytes ≥ 128 contam como letra no pré-tokenizador do GPT-2.

## Verificar tudo

```bash
export PATH="$HOME/.bend/bin:$PATH"
reference/.venv/bin/python reference/check_all.py          # ~20 verificações, ~20 s
reference/.venv/bin/python reference/check_all.py --full   # + GPT-2 e MNIST (~5 min)
```

Preparação dos dados e pesos: [`reference/`](reference) (`gpt2_prep.py`, `mnist_torch.py`) e os README de cada demo.

## Limites

- **Desempenho**: o Bend 2.0.35 não tem BLAS; as matrizes são listas encadeadas. O MNIST é ~1800× mais lento que o PyTorch numa thread, o GPT-2 ~40×. O paralelismo por linhas escalou só ~1,8× (causa não resolvida; ver `NOTES.md`). GPU não usada: o Bend pede CUDA 12 e o Arch traz o 13.
- Só `Nat`, `U32` e `F32`; sem `F64`.
- A invariante "cada linha de uma `Mat<r,c>` tem `c` números" é mantida pela biblioteca e pelos construtores verificados, não pelo tipo.

## Estrutura

```
nat-lemmas/  bpe/  tensor/  autograd/   # pacotes (main.bend + README + LICENSE)
demos/mnist  demos/gpt2                 # demos com README e BENCHMARK
reference/                              # PyTorch, tiktoken e o check_all.py
poc/                                    # provas de conceito da v0.2
docs/                                   # mensagens de erro de shape e material de lançamento
NOTES.md                                # decisões, descobertas e dívidas
```

## Licença

[MIT](LICENSE).
