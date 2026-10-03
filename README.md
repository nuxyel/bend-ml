# bend-ml

Machine learning em [Bend 2](https://bend-lang.com), com **tipos dependentes**: a ideia é um "PyTorch onde erro de shape não compila", com IA escrevendo o código e LAWS (provas verificadas pelo kernel) garantindo a correção.

> Status: início. Versão do Bend fixada: **2.0.35**.

## Pacotes

| Pacote | Estado | O que é |
|---|---|---|
| `bend-ml-nat-lemmas` | em andamento (v0.1) | Lemas provados de `Nat` e `List` que a Base do Bend ainda não tem |
| `bend-ml-bpe-tokenizer` | planejado (v0.3) | Tokenizer BPE byte-level com roundtrip provado |
| `bend-ml-tensor` | planejado (v0.4) | Tensores com shape no tipo + autograd |

## Roadmap

| Versão | Entrega |
|---|---|
| v0.1 | repositório, licença e `bend-ml-nat-lemmas` publicado |
| v0.2 | prova de conceito: vetor com tamanho no tipo |
| v0.3 | tokenizer BPE |
| v0.4 | tensores com shape no tipo |
| v0.5 | autograd (escalar, depois tensorial) |
| v0.6 | demo MNIST + benchmark honesto contra PyTorch |
| v1.0 | pacote de entrega (vídeo, thread, stretch GPT-2 small) |

## Como verificar

```bash
export PATH="$HOME/.bend/bin:$PATH"
bend nat-lemmas/PROOF.bend   # ALL PROOFS CHECK
```

## Licença

[MIT](LICENSE).
