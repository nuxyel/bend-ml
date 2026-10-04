# Roteiro do vídeo (~60 s)

Mostra três coisas: o **erro de shape pego pelo compilador**, a **prova verificada** e o **GPT-2 gerando texto em Bend**. Tela do terminal, fonte grande, sem edição pesada.

## Preparação (uma vez)

```bash
cd ~/Repos/bend2-ml
export PATH="$HOME/.bend/bin:$HOME/.elan/bin:$PATH"
bend demos/gpt2/fast.bend -o /tmp/gpt2_fast     # compila antes de gravar
clear
```

Gravar a tela: o atalho de captura do Omarchy, ou `wf-recorder -f bend-ml.mp4`.

## Cenas

| Tempo | Na tela | Fala / legenda |
|---|---|---|
| 0–5 s | `cat tensor/tests/bad_matmul.bend` | "Multiplicar (2×3) por (4×5)." |
| 5–15 s | `bend tensor/tests/bad_matmul.bend` → `expected Mat<3n,5n> / observed Mat<4n,5n>` | "Em Bend, isso é erro de **tipo**: não compila." |
| 15–22 s | `bend tensor/tests/bad_reshape.bend` → `expected 12n / observed 15n` | "Reshape só compila com a prova de que o número de elementos não muda." |
| 22–32 s | `bend bpe/main.bend --verdict` → `ALL PROOFS CHECK`; mostrar `law roundtrip` | "O tokenizer tem o roundtrip provado: decode(encode(s)) = s, verificado pelo kernel." |
| 32–50 s | `/tmp/gpt2_fast "The capital of France is" 8` (cada token leva ~0,1 s; acelerar só os 9 s de carga) → `The capital of France is the capital of the French Republic, and` | "GPT-2 small de 124 milhões de parâmetros, inteiro em Bend, mesmos tokens do PyTorch, ~0,1 s por token." |
| 50–60 s | `reference/.venv/bin/python reference/check_all.py` → `TOTAL: 27/27` | "Cinco pacotes no BendHub, MIT. Link no tópico." |

## Dicas

- O GPT-2 leva ~9 s para carregar e ~0,1 s por token: acelere só a carga e diga isso na legenda ("acelerado").
- Para mostrar o MNIST: `demos/mnist/run.sh 1 0 0.1` treina uma época completa em ~7 s no Bend e ~0,3 s no PyTorch (honesto no vídeo!), os dois imprimem `9129/10000`.
