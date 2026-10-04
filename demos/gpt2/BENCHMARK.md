# GPT-2 small: Bend × PyTorch

**Resumo (v2):** o GPT-2 small de 124 M de parâmetros roda em Bend e **gera exatamente os mesmos tokens do PyTorch**, com logits iguais em até 2e-4. Na v1 levava ~3 s por token (~150× o PyTorch); na v2 (produtos matriz·vetor sobre `Array`) leva **~0,1 s por token**: ~5× o PyTorch com 16 threads (21 ms por forward de 11 tokens) e ~2× o PyTorch com 1 thread (55 ms). O que ainda pesa é carregar os pesos (9 s contra ~1 s), porque os 124 M de números viram árvores de nós (~10 GB de memória durante a carga).

## v2: `demos/gpt2/fast.bend` (pacote `bend-ml-tensor-array@0.1.1.0`)

| GPT-2 small, "The capital of France is", 8 tokens | por token | total |
|---|---|---|
| Bend v1 (listas) | ~3 s | 49,8 s |
| Bend v2 com `par = 3` (cópias da matriz por tarefa) | ~1,1 s | 22 s |
| **Bend v2, sequencial (`par = 0`)** | **~0,1 s** | **11,1 s** (9 s de carga + 1,2 s para os 8 tokens) |
| PyTorch (CPU, 16 threads, sem KV cache): forward de 11 tokens | **21 ms** | 1,3 s no `gpt2_ref.py` (inclui ~1 s para carregar os pesos) |
| PyTorch (CPU, 1 thread, sem KV cache): forward de 11 tokens | 55 ms | |

Mesma verificação da v1 (`reference/test_gpt2.py`): 3 prompts, 22 tokens, ids idênticos, |Δlogit| ≤ 2e-4.

Em matriz·vetor, `par = 3` é ~10× **pior** que sequencial: o custo de clonar a matriz de pesos por tarefa supera o de calcular (cada peso é lido uma vez). Medição isolada em `NOTES.md`, exp. 9.

---

## v1: listas (`demos/gpt2/gpt2.bend`), mantido como linha de base

## Verificação (`reference/test_gpt2.py`)

Geração gulosa a partir de três prompts, comparando ids, texto e o logit do token escolhido em cada passo (22 tokens):

| prompt | ids | texto | max \|Δ logit\| |
|---|---|---|---|
| `The capital of France is` (8 tokens) | idênticos | idêntico | 8e-5 |
| `Machine learning is` (8 tokens) | idênticos | idêntico | 2e-4 |
| `1, 2, 3, 4,` (6 tokens) | idênticos | idêntico | 1,1e-4 |

Exemplo: `The capital of France is the capital of the French Republic, and` (nas duas implementações).

O tokenizer em Bend (pré-tokenizador + 50 000 regras, pacote `bend-ml-bpe-tokenizer`) bate com o `tiktoken` em 16 de 16 textos (`reference/test_gpt2_tok.py`).

## Desempenho

| | tempo |
|---|---|
| Bend: carregar 124 M de pesos | ~10 s |
| Bend: cada token gerado | ~3 s (uma posição com KV cache; 12 camadas + 50 257 logits) |
| Bend: 8 tokens, total | 49,8 s |
| PyTorch (CPU, sem KV cache, 16 threads): 8 tokens, total incluindo carregar pesos | 1,3 s |

Configuração: Intel Core Ultra 7 155H (22 threads), 32 GB; Bend 2.0.35 (1 thread efetiva), clang 22.1.8; PyTorch 2.14.1 (CPU). Sem GPU: a RTX 4050 (6 GB) não é usada (o Bend pede CUDA 12 em `/usr/local/cuda` e o Arch traz o 13).

## Memória

~2 GB para os pesos em listas de `F32` (16 bytes por número).
