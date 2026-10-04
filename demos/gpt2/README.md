# GPT-2 small (124M) em Bend

Inferência do GPT-2 small escrita em Bend: pré-tokenizador, tokenizer BPE com as 50 000 regras oficiais (pacote `bend-ml-bpe-tokenizer`, roundtrip provado), 12 camadas de transformador com cache de chaves/valores, e geração gulosa. As matrizes usam `bend-ml-tensor`, com a shape no tipo.

## Rodar (da raiz do repositório)

```bash
reference/.venv/bin/python reference/gpt2_prep.py          # baixa nada; converte demos/gpt2/data/model.safetensors
reference/.venv/bin/python reference/gpt2_prep.py --split  # pesos por tensor (Conv1D já transposto)

bend demos/gpt2/gpt2.bend -o /tmp/gpt2
/tmp/gpt2 "The capital of France is" 8
```

Os pesos (`model.safetensors`), `vocab.json` e `merges.txt` vêm de <https://huggingface.co/openai-community/gpt2>; ficam em `demos/gpt2/data/` (fora do git).

## Verificação

- **Tokenizer**: `reference/test_gpt2_tok.py` compara com o `tiktoken` em 16 textos (ASCII, contrações, números, espaços e quebras de linha, acentos, cirílico, CJK): 16 de 16 idênticos.
- **Modelo**: `reference/gpt2_ref.py` roda o mesmo modelo em PyTorch com os mesmos pesos; os ids gerados e os logits são comparados com os do Bend (ver `BENCHMARK.md`).

## Limites conhecidos

- O pré-tokenizador trata todo byte >= 128 como letra. Isso casa com a regex do GPT-2 para letras acentuadas e de outros alfabetos, mas não separa símbolos Unicode que não são letras.
- Só geração gulosa (sem temperatura nem amostragem).
- Memória: os 124 M de pesos ficam em listas de `F32` (~2 GB).
