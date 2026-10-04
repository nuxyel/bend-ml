# MNIST em Bend: MLP 784-128-10

Uma rede de duas camadas treinada com SGD puro, escrita com `bend-ml-tensor` e `bend-ml-autograd`. **Cada multiplicação de matrizes, cada gradiente e cada atualização de pesos tem o formato conferido pelo tipo**: se `dW` tivesse a dimensão errada, o programa não compilaria.

## Rodar (da raiz do repositório)

```bash
# 1. dados e pesos iniciais (uma vez)
reference/.venv/bin/python reference/mnist_torch.py --make-init --epochs 1 --max-batches 1
# (baixe os 4 arquivos do MNIST em demos/mnist/data/; veja BENCHMARK.md)

# 2. treino em Bend:  <épocas> <máx. lotes (0 = todos)> <lr>
bend demos/mnist/train.bend -o /tmp/mnist_train
/tmp/mnist_train 1 0 0.1

# 3. o gêmeo em PyTorch (mesmos pesos iniciais, mesmos lotes, mesmo lr)
reference/.venv/bin/python reference/mnist_torch.py --epochs 1 --lr 0.1
```

Há um atalho que faz tudo: `demos/mnist/run.sh`.

## Verificação de correção

Com os mesmos pesos iniciais, a mesma ordem de lotes e o mesmo `lr`, as duas implementações têm de produzir as mesmas perdas e acertos. Depois de 50 passos:

| | perda média de treino | acertos no teste |
|---|---|---|
| Bend | 1,6616005 | 7829 / 10000 |
| PyTorch | 1,661600 | 7829 / 10000 |

Os resultados completos de uma época e o tempo estão em `BENCHMARK.md`.
