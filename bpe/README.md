# bend-ml-bpe-tokenizer

Tokenizer **BPE byte-level** (estilo GPT-2) em Bend 2, com o **roundtrip provado** pelo kernel: `decode(encode(s)) == s`.

- Bend: **2.0.35** · Licença: MIT · Depende de `bend-ml-nat-lemmas@0.1.0.0`.
- `bend bpe/main.bend` e `bend bpe/main.bend --verdict` → `ALL PROOFS CHECK` (sem `@unsafe`, sem `?TODO`).

## Modelo

- **Token**: `B{n}` (byte cru `n`) ou `M{k}` (criado pela regra de id `k`). Os ids numéricos do GPT-2 são `n` para bytes e `256 + k` para as fusões.
- **Regra**: `Rule{id, a, b}` funde o par de tokens `(a, b)` em `M{id}`.
- **Tabela**: lista de regras da **mais antiga para a mais nova**, como o `merges.txt` do GPT-2.
- `encode(tabela, tokens)` aplica as regras em ordem (cada uma funde, da esquerda para a direita, as ocorrências não sobrepostas do par).
- `decode(tabela, tokens)` expande cada token de volta para bytes olhando só as regras mais antigas que ele.
- `train(n, 0n, tokens)` aprende até `n` fusões (desempate: o primeiro par que apareceu).

```python
import Base
import bend-ml-bpe-tokenizer@0.1.1.0/main.bend as BPE

# BPE.encode(table, BPE.lift(bytes))   BPE.decode(table, ids)   BPE.train(30n, 0n, BPE.lift(bytes))
```

## LAWS provadas (em português)

| Lei | O que afirma |
|---|---|
| `roundtrip` | Se a tabela é **bem formada** (nenhum id de regra se repete: `wf(tabela) = True`), então `decode(encode(bytes)) = bytes`, para **qualquer** sequência de bytes. |
| `vocab_bound` | Todo token que `encode` emite é um byte cru ou foi criado por uma regra da tabela; nunca aparece um id desconhecido. |
| `dec_append` | Decodificar duas listas de tokens juntas é decodificar cada uma e juntar os bytes. |
| `train_wf` | A tabela que `train` devolve é **sempre bem formada** (os ids das regras são `next, next+1, ...`, nenhum repete), para qualquer corpus e quantas fusões forem pedidas. |
| `roundtrip_trained` | Consequência das duas primeiras: treine uma tabela em **qualquer corpus** e `decode(encode(s)) = s` para qualquer `s`, **sem nenhuma hipótese**. |

A prova do roundtrip tem a seguinte ideia (os comentários em `main.bend` detalham): ao aplicar a regra `k`, cada par `(a, b)` vira `M{k}`, e `M{k}` expande para `expansão(a) ++ expansão(b)`; logo o decode não muda. Isso só vale se `k` ainda não aparecia na lista, e é exatamente isso que a tabela bem formada garante (ids distintos). Os lemas auxiliares provam que acrescentar a regra nova no topo da tabela não muda a expansão dos tokens que já existiam.

## Testes (contra Python)

`reference/test_bpe.py` roda o CLI em Bend (`bpe/cli.bend`) contra `reference/bpe_ref.py` (mesmo algoritmo e desempate), com textos em inglês, com acentos (`coração`, `ação`) e com emoji/CJK em UTF-8: `train`, `encode`, `decode` e roundtrip batem em todos os casos. Rodar:

```bash
reference/.venv/bin/python reference/test_bpe.py
```

## Limites conhecidos

- A *escolha* das fusões pelo `train` (qual par é o mais frequente) não é provada, e não precisa ser: a correção do roundtrip não depende dela (`train_wf` + `roundtrip`). Tabelas lidas de um `merges.txt` com ids sequenciais também cumprem `wf`, mas essa leitura (`rules.go` do CLI) não é provada.
- A busca do par de menor rank é feita aplicando as regras em ordem (equivalente ao algoritmo do GPT-2 para tabelas treinadas), com custo `regras × tamanho`. Para o vocabulário completo do GPT-2 (50 mil regras) o uso é por palavra, depois do pré-tokenizador.
- Ainda não há o pré-tokenizador por regex do GPT-2; vem em versão futura.

## Versões

- `0.1.1.0`: acrescenta `train_wf` e `roundtrip_trained`.
- `0.1.0.0`: primeira publicação (`roundtrip`, `vocab_bound`, `dec_append`).
