# Plano v2: medir o teto de desempenho e aprofundar o que só o Bend tem

## Por que

A v1.0 mostrou um Bend ~1800× mais lento que o PyTorch no MNIST. Mas a comparação foi injusta com a linguagem:
matrizes em listas encadeadas, uma thread efetiva, sem GPU. A v2 tem dois eixos, nesta ordem:

1. **Desempenho:** descobrir o teto real de cada técnica e a distância final para o PyTorch (sem prometer alcançá-lo).
2. **Garantias:** usar o que sobrar de fôlego para aprofundar o que o PyTorch não tem: invariantes no tipo, mais leis provadas.

Decisões do Renan: os dois eixos, desempenho primeiro; CUDA 12 tentando **sem sudo** primeiro; sem meta numérica fixa (medir e reduzir).

## Eixo 1: desempenho (cada passo registra o número em `NOTES.md`)

| # | Experimento | O que mede | Critério de saída |
|---|---|---|---|
| 1 | Linha de base reprodutível: um micro-benchmark único de `matmul` 100×784 · 784×128 e de um passo do MNIST | tempo e threads | script `bench/run.sh` com resultado salvo |
| 2 | `Array<F32>` no lugar de listas (índice `i*c + j`, `Array.get/set`) | ganho de usar memória indexável | número comparado com listas |
| 3 | Por que o paralelismo trava em ~1,8×: reproduzir com um caso mínimo, variar granularidade e a forma da divisão, ler `bend guide shaders` e `paper/BendRT.pdf` | gargalo (alocador, contagem de referências, divisão desbalanceada) | causa identificada ou relato mínimo reproduzível para o GitHub do Bend |
| 4 | Blocagem do `matmul` (tiles) e laço interno sem alocação | ganho de localidade | número |
| 5 | GPU: CUDA 12 sem sudo (toolkit em `~/.local/cuda12`, link `/usr/local/cuda` só se possível); testar `pow2!` e depois `matmul` com `!` | se a RTX 4050 roda; ganho de GPU | rodou ou limite documentado |
| 6 | Reconstruir o passo do MNIST e o GPT-2 com a melhor técnica | tempo final vs PyTorch | tabela antes/depois em `BENCHMARK.md` |

Regra: um experimento descartado também entra nas notas (com o número). Nada de otimização que quebre a equivalência com o PyTorch (os testes de `check_all.py` continuam valendo).

## Eixo 2: garantias (só depois do eixo 1)

- **Invariante no tipo:** `Mat<r,c>` com a lista de linhas amarrada a `r` e `c` (por exemplo um tipo indexado `Rows(r, c)`), eliminando a invariante "de biblioteca" do README.
- **Mais leis:** `transpose(transpose(m)) = m`, `matmul` associativo sobre a estrutura, `Mat.of` correto (só devolve `Some` se os tamanhos batem), `softmax` com soma estrutural.
- **Tokenizer:** lei de que `train` produz tabelas bem formadas (`wf = True`), fechando a lacuna do README do `bpe`.
- Cada lei nova: `--verdict` limpo, publicada em nova versão do pacote (versões de 4 números).

## Entregas

- `NOTES.md` com cada medição e hipótese descartada.
- `demos/*/BENCHMARK.md` atualizados (antes/depois, hardware, versões).
- Pacotes novos como `0.2.x.0`; tag `v2.0.0` e release.
- Se a causa do paralelismo ou outra limitação for do Bend: um relato mínimo para `github.com/bendlang/bend/issues` (em texto pronto; eu não abro issue em seu nome).

## Critério de parada da v2

1. Cada técnica do eixo 1 medida e registrada, com a distância final para o PyTorch.
2. Pelo menos uma garantia nova do eixo 2 provada (`--verdict`) e publicada.
3. `check_all.py --full` verde; benchmarks e README atualizados; `v2.0.0` publicada.
