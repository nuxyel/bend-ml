# bend-ml

**Machine learning em [Bend 2](https://bend-lang.com), onde um erro de shape é um erro de tipo.**

Cinco pacotes publicados no BendHub, uma MLP no MNIST e o GPT-2 small (124 M) inteiros em Bend, com as provas verificadas pelo kernel do Bend e os números conferidos contra PyTorch e `tiktoken`. Versão do Bend: **2.0.35**.

```python
# (2x3) · (4x5): não compila
def bad() -> TA.MMul<2n, 3n, 5n>:
  TA.Mat.matmul(2n, 3n, 5n, 0n, TA.Mat.zeros(2n, 3n), TA.Mat.zeros(4n, 5n))
```
```
Error:
- expected : TA.Mat<3n, 5n>
- observed : TA.Mat<4n, 5n>
```

## Pacotes no BendHub

| Pacote | Versão | O que é | LAWS provadas |
|---|---|---|---|
| [`bend-ml-nat-lemmas`](nat-lemmas) | 0.1.0.0 | Lemas de `Nat` e `List` que a Base não tem | `add_comm`, `add_assoc`, `mul_comm`, `mul_assoc`, `mul_dist`, `append_assoc`, `length_append`, `product_append`... (15) |
| [`bend-ml-bpe-tokenizer`](bpe) | 0.1.1.0 | Tokenizer BPE byte-level (estilo GPT-2) | **roundtrip** `decode(encode(s)) = s`, `vocab_bound`, `dec_append`, **`train_wf`** (o `train` sempre gera tabela bem formada) e **`roundtrip_trained`** (roundtrip para qualquer tabela treinada, sem hipótese) |
| [`bend-ml-tensor`](tensor) | 0.1.1.0 | `Vec<n>` e `Mat<r,c>` com shape no tipo, sobre listas | `reshape_swap`, `reshape_flat`; `reshape` só compila com a prova de que o nº de elementos não muda |
| [`bend-ml-tensor-array`](tensor-array) | 0.1.1.0 | **Novo na v2.** `Mat<r,c>` sobre `Array<F32>` plano, mesma garantia de shape, **~50× mais rápido** | produtos `matmul`, `matmul_nt`, `matmul_tn` (a forma de `dW = Xᵀ·dY` é conferida pelo tipo), blocos paralelos |
| [`bend-ml-autograd`](autograd) | 0.1.0.0 | Diferenciação automática e camadas com backward tipado | **`reverse_eq_forward`**: o modo reverso do autodiff dá o mesmo que o modo direto |

```python
import bend-ml-tensor-array@0.1.1.0/main.bend as TA
import bend-ml-bpe-tokenizer@0.1.1.0/main.bend as BPE
```

## Demos (v2)

| Demo | Resultado |
|---|---|
| [MNIST](demos/mnist) (MLP 784-128-10) | **6,6 s por época** (v1: 544 s). Perda e acertos **idênticos** ao PyTorch com os mesmos pesos e lotes: 0,5204771 / 9129, depois 0,27043572 / 9298, depois 0,2156194 / 9418. O PyTorch leva 0,2 a 0,3 s por época ([benchmark honesto](demos/mnist/BENCHMARK.md)). |
| [GPT-2 small](demos/gpt2) (124 M) | **~0,1 s por token** (v1: 3 s). Gera os **mesmos tokens** que o PyTorch (22 tokens, 3 prompts), logits a menos de 2e-4. O PyTorch faz o forward em 21 ms (16 threads) ou 55 ms (1 thread); carregar os pesos leva 9 s no Bend ([detalhes](demos/gpt2/BENCHMARK.md)). Tokenizer idêntico ao `tiktoken` em 16/16 textos. |

```
$ ./gpt2_fast "The capital of France is" 8
  id 262  logit -100.24986 ...
texto: The capital of France is the capital of the French Republic, and
```

### O que mudou da v1 para a v2

A v1 usava listas encadeadas para as matrizes e media ~1800× o PyTorch no MNIST. A v2 mediu cada hipótese (`NOTES.md`, exp. 1 a 9): trocar listas por `Array` plano deu **~49× num thread**, produtos em blocos paralelos mais ~1,8×, e no GPT-2 achei que **paralelizar matriz·vetor copiando a matriz custa 10× mais que calcular**, então ele roda sequencial. Distância ao PyTorch agora: MNIST ~22 a 33×, GPT-2 ~2 a 5× por token.

## O que é provado e o que é testado

- **Provado pelo kernel** (`bend X.bend --verdict`): as leis acima, em cinco pacotes. Sem `@unsafe`, sem `?TODO`, em nenhum.
- **Pelo tipo**: shapes de `matmul`, de cada gradiente de camada (`dW: Mat<i,o>`) e `reshape` com prova. O passo de treino do MNIST inteiro é checado assim.
- **Testado, não provado**: toda a numérica em `F32` (não é um número real; arredonda). Gradient checking e comparação com PyTorch em `reference/`. O tokenizer em ASCII é exato; bytes ≥ 128 contam como letra no pré-tokenizador do GPT-2.

## Verificar tudo

```bash
export PATH="$HOME/.bend/bin:$PATH"
reference/.venv/bin/python reference/check_all.py          # 27 verificações, ~30 s
reference/.venv/bin/python reference/check_all.py --full   # + GPT-2 e MNIST (32 verificações, ~1,5 min)
```

Preparação dos dados e pesos: [`reference/`](reference) (`gpt2_prep.py`, `mnist_torch.py`) e os README de cada demo.

## Limites

- **Desempenho:** o Bend 2.0.35 gera código escalar, sem BLAS nem SIMD: no mesmo produto de matrizes o PyTorch faz 62 G mult-soma/s em 1 thread e o Bend com `Array` ~2,3 G/s (~27×). O paralelismo escala ~2 a 4× nesta CPU híbrida (P+E cores). GPU não usada: o Bend pede CUDA 12 e o Arch traz o 13.
- **Memória:** os pesos do GPT-2 viram árvores de nós (~10 GB durante a carga, 9 s).
- Só `Nat`, `U32` e `F32`; sem `F64`.
- `Mat<r,c>` não carrega no tipo a invariante "o `Array` tem capacidade ≥ r*c": os construtores garantem, mas não é um fato do tipo.
- O pré-tokenizador do GPT-2 trata todo byte ≥ 128 como letra (exato para letras acentuadas e de outros alfabetos).

## Estrutura

```
nat-lemmas/  bpe/  tensor/  tensor-array/  autograd/   # pacotes (main.bend + README + LICENSE)
demos/mnist  demos/gpt2                 # demos com README e BENCHMARK
reference/                              # PyTorch, tiktoken e o check_all.py
poc/  bench/                            # provas de conceito (v0.2) e micro-benchmarks da v2
docs/                                   # mensagens de erro de shape e material de lançamento
NOTES.md                                # decisões, descobertas e dívidas
```

## Licença

[MIT](LICENSE).
