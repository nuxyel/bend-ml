# MNIST: benchmark honesto, Bend × PyTorch

**Resumo (v2):** os resultados são **idênticos** (mesma perda, mesma acurácia em 1 e em 3 épocas). Na v1 o Bend levava **544 s por época** (~1800× o PyTorch); na v2, com matrizes em `Array` plano e produtos paralelos, leva **6,6 s** (~33× o PyTorch com 16 threads, ~22× com 1). A diferença entre v1 e v2 é a estrutura de dados, não a linguagem; o que resta é código escalar contra BLAS/SIMD.

## v2: `Array` plano (`demos/mnist/fast.bend`, pacote `bend-ml-tensor-array@0.1.1.0`)

| | 1 época | perda / acertos após 1, 2, 3 épocas |
|---|---|---|
| **Bend v2, 16 threads** | **6,6 s** | 0,5204771 / 9129 · 0,27043572 / 9298 · 0,2156194 / 9418 |
| Bend v2, 1 thread | 18,1 s | |
| PyTorch, 16 threads | 0,19 a 0,21 s | 0,520477 / 9129 · 0,270433 / 9298 · 0,215613 / 9418 |
| PyTorch, 1 thread | 0,30 s | |

Cada produto, cada gradiente e cada atualização tem o formato conferido pelo tipo (`dW` pedido com as dimensões trocadas não compila: `docs/shape-error-array-bad_grad.txt`). Como chegamos aqui, passo a passo (cada número em `NOTES.md`, exp. 1 a 9):

| passo | época |
|---|---|
| v1: matrizes em listas encadeadas | 544,3 s |
| `Array` plano, `Array.get/set` por índice, 1 thread | 11,5 s |
| + produtos grandes em 2^3 blocos paralelos (16 threads) | 6,3 s |
| API tipada do pacote (conversões lista↔Array nos wrappers) | 6,6 s |

---

## v1: listas (`demos/mnist/train.bend`), mantido como linha de base

## Configuração

| | |
|---|---|
| Modelo | MLP 784 → 128 → 10, ReLU, SGD puro, `lr = 0,1`, lotes de 100, 600 lotes por época, sem embaralhar |
| Dados | MNIST (60 000 treino, 10 000 teste), pixels `/255`, sem outra normalização |
| Pesos iniciais | os mesmos arquivos de texto nas duas implementações (`demos/mnist/data/init/`), `U(-1/√n, 1/√n)` |
| CPU | Intel Core Ultra 7 155H (16 núcleos, 22 threads), 32 GB de RAM |
| SO | Linux 7.2.5 (Omarchy) |
| Bend | 2.0.35, binário nativo (`bend ... -o`), clang 22.1.8; 1 thread (o `matmul` por listas só chegou a ~1,8× com mais threads) |
| PyTorch | 2.14.1 (CPU), Python 3.14.7, 16 threads ou 1 thread |
| GPU | não usada (o Bend pede CUDA 12; o Arch tem CUDA 13; RTX 4050 de 6 GB fica fora) |

## Resultado de 1 época completa

| | perda média de treino | acertos no teste | tempo da época |
|---|---|---|---|
| **Bend** | 0,52047706 | 9129 / 10000 | **544,3 s** |
| PyTorch (16 threads) | 0,520477 | 9129 / 10000 | 0,21 s |
| PyTorch (1 thread) | 0,520477 | 9128 / 10000 | 0,30 s |

- A perda coincide em 6 casas; a diferença de 1 acerto no PyTorch de 1 thread vem da ordem de soma em `F32`.
- Também coincidiram depois de 50 passos (`1,6616005` e `7829` acertos, nos dois).
- **Onde o Bend perde:** ~0,9 s por lote contra ~0,5 ms. Cada passo faz ~20 milhões de multiplicações-e-somas; o Bend fez ~24 milhões por segundo efetivos numa thread (o PyTorch usa BLAS vetorizado). Não houve tentativa de otimização além de pré-transpor operandos e de não calcular `dx` na primeira camada.

## O que o Bend oferece que a comparação não mostra

- Cada produto de matrizes e cada gradiente tem a dimensão conferida pelo tipo (`bad_matmul.bend` não compila).
- O tokenizer e as leis de `reshape`/autodiff têm provas verificadas pelo kernel.

## Reproduzir

```bash
demos/mnist/run.sh 1 0 0.1     # 1 época completa nos dois (o Bend leva ~9 min)
demos/mnist/run.sh 1 50 0.1    # só 50 lotes (rápido; confere a equivalência)
```
