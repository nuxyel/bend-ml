# GPT-2 small: Bend × PyTorch

**Resumo:** o GPT-2 small de 124 M de parâmetros roda em Bend e **gera exatamente os mesmos tokens do PyTorch**, com logits iguais em até 2e-4. É ~40× mais lento no total e ~3 s por token contra milissegundos.

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
