# MNIST: benchmark honesto, Bend × PyTorch

**Resumo:** os resultados são **idênticos** (mesma perda, mesma acurácia), mas o Bend é cerca de **1800 vezes mais lento** que o PyTorch numa thread. Não tento esconder: o Bend 2.0.35 não tem BLAS nem kernels vetorizados, e as matrizes aqui são listas encadeadas.

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
