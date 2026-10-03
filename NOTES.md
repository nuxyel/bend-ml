# NOTES — Fase 0 (reconhecimento)

Data: 2026-10-03. Tudo abaixo foi verificado na máquina ou em fonte primária (guia instalado, `base.bend`, repositório oficial, site). O que **não** foi verificado está marcado como *(não verificado)*.

## Versão fixada

- **Bend 2.0.35** (`bend version`). Fixar esta versão até segunda ordem.
- Repositório oficial: `github.com/bendlang/bend` (não `HigherOrderCO/Bend`, que é o Bend 1).
- Instalado com `curl -fsSL https://bend-lang.com/install.sh | sh` (script lido antes de rodar: baixa o release do GitHub, confere SHA256, instala só em `~/.bend`, sem sudo).
- Atualizar: `bend update` (roda o instalador de novo). Não atualizar sem registrar aqui, pois o Bend 2 muda rápido.
- Telemetria: uma consulta por dia ao bend-lang.com (versão, OS, CPU). Desligar com `export BEND_NO_TELEMETRY=1`.
- PATH: `export PATH="$HOME/.bend/bin:$PATH"`.

## Onde está o quê

| O quê | Onde |
|---|---|
| Guia da linguagem (677 linhas) | `bend guide` ou `~/.bend/guide/GUIDE.md` |
| Efeitos (C/JS customizados) | `bend guide effects` |
| Shaders / código paralelo | `bend guide shaders` |
| Base (3009 linhas) | `~/.bend/bend2/base.bend` ou `bend base [nome]` |
| Kernel de provas em Lean | `~/.bend/bend2/bendtt.lean` |
| Exemplos e demos | só no repo GitHub: `demos/` (não vêm no instalador) |
| Papers | `paper/BendTT.pdf`, `paper/BendRT.pdf` no repo |

Demos relevantes para nós: `proof_numerics` (prova de `add_comm`, `add_assoc`, `mul_comm`, `mul_dist` e `divmod` sobre `Nat`; é o modelo para o `nat-lemmas`), `proof_typed_eval`, `proof_insertion_sort`, `pure_par_sum`, `pure_par_sort`.

## Como é um projeto de provas

- `LAWS.bend`: o humano escreve as leis (`law nome: for x: T  {a == b : T}`).
- `PROOF.bend`: importa o `LAWS.bend` e prova cada lei com um `def` de mesmo nome.
- `bend PROOF.bend` imprime `ALL PROOFS CHECK` ou `SOME PROOFS FAIL`. É o portão antes de commitar.
- Não há táticas: proposição é tipo, prova é `def`. `{==}` é reflexividade, `%e : P` reescreve, chamada recursiva é hipótese de indução.
- `?nome` imprime o objetivo; `?TODO` deixa a prova aberta.
- Velocidade do checker: o teste de `add_zero` com `Nat` levou **0,17 s** no total.

## O que a Base oferece

- Tipos: `Nat` (unário: `Zero`/`Succ`), `U32`, `F32`, `Char`, `String` (lista de `Char`), `List`, `Array` (árvore binária, mutação in-place), `Map` (chaves `String`), `Maybe`, `Result`, `Sigma`, `Either`, `Equal`.
- **Lemas existentes (muito poucos):**
  - Igualdade: `Equal.cong`, `Equal.sym`, `Equal.trans`.
  - Nat: `Nat.ge_refl`, `Nat.max_ge_l`, `Nat.max_ge_r`.
  - Word/U32: `Word.add_comm`, `U32.add_comm`.
- **Não existe na Base:** `Nat.add_comm`, `add_assoc`, `mul_comm`, `mul_assoc`, distributividade, nenhum lema de `List` (`append_assoc`, `length_append`, `reverse_reverse`...), nada sobre `product`. **Conclusão: o pacote `nat-lemmas` é necessário**, e provavelmente também lemas de `List`.
- O demo `proof_numerics` já prova os lemas de Nat acima, então dá para usar como ponto de partida (ler antes de reescrever).
- Operações de `List`: `map`, `length`, `append`, `reverse`, `take`, `drop`, `zip`, `foldl`, `foldr`, `filter`, `sort`, `range`, `replicate`.

## Números

- Só existem **`Nat`, `U32` e `F32`**. Não há `U64`, `I32`, `I64` nem `F64` (limitação do Metal, segundo o README).
- **Não existe tipo "byte".** Bytes serão `U32` (valor 0–255) numa `List<U32>`; o limite fica numa prova/invariante nossa.
- Operações de `F32`: `add sub mul div mod pow neg abs sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil trunc round min max clamp lerp`, comparações, `show`, `read`, `from_nat`, `to_nat`. Testado: `F32.exp(1.0)` imprime `2.7182817`.
- **As operações de `F32` são `law` primitivas, sem prova**: na Base aparecem como `law F32.add: for a: F32 for b: F32  F32`. Ou seja, são opacas para o provador. Isso bate com a regra 5 do `CLAUDE.md` (não provar nada numérico sobre float).
- Consequência para a Fase 3: acumular loss e somas grandes em `F32` perde precisão; sem `F64`, o gradient checking precisa de tolerâncias maiores.
- `Nat` literal acima de `256n` vira `U32.to_nat`, com teto em `4294967295n`. Cuidado com `Nat` unário em dimensões grandes (ex.: 784, 50257); precisa de teste de desempenho antes de decidir a representação das shapes.

## Backends

- Compila para **C** (precisa de clang 14+; 19+ para `!`), **JS**, **Metal** (macOS) e **CUDA** (Linux). Lua, Luau e Python estão planejados.
- A GPU é ativada por chamada com `!` (ex.: `f!(x)`); em binário nativo, `./prog --gpu off` força CPU e `--gpu 4GB` limita o heap da GPU.
- **Nesta máquina:**
  - clang 22.1.8 instalado (ok).
  - `nvidia-smi` mostra RTX 4050 Laptop, 6141 MiB, driver 610.57.04.
  - **CUDA Toolkit não está instalado** (`nvcc` ausente; `/usr/local/cuda` e `/opt/cuda` não existem). O guia diz que no Linux precisa de **CUDA 12 em `/usr/local/cuda`**.
  - Ou seja: a GPU **não** roda ainda. Falta instalar o CUDA 12 e testar com `pow2!`. *(não verificado se o CUDA 12 do Arch/Omarchy funciona com o driver 610 e com a RTX 4050; testar antes de planejar a Fase 3 em GPU.)*
  - Sem GPU, `!` roda em paralelo na CPU, então o código continua funcionando.
- Sem memória unificada (placa dedicada), mover dados CPU↔GPU tem custo; o ganho do Apple Silicon descrito no guia não vale aqui.

## IO

- `File.open`, `File.read`, **`File.read_bytes`**, `File.read_at`, `File.size`, `File.write`, `File.write_bytes` (esta recebe `List<U32>`), `File.close`. Suficiente para carregar MNIST e pesos.
- Também há `IO.args`, `IO.now` (para benchmark), `IO.random_u32`, `IO.get_env`, `IO.fork/join`, `IO.thread_count`, TCP/UDP, `Process.run`.
- Efeitos próprios em C/JS são possíveis (`bend guide effects`), mas a API C não tem promessa de ABI: reconstruir a cada atualização.
- Estado do carregamento de arquivos grandes (GPT-2 tem ~500 MB de pesos) *(não verificado)*: lista de `U32` usa muita memória; avaliar `Array` e leitura em blocos.

## Provas e confiança

- `bend X.bend --verdict` reverifica com o kernel provado em Lean. **Requer `lean` v4.34.0** (via elan) ou `$BENDTT` apontando para o kernel compilado. **Não está instalado aqui**; o comando falhou com `Executable not found in $PATH: "lean"`. Instalar o elan antes de publicar, para poder dizer "verificado pelo kernel auditado".
- `@unsafe def` pula a checagem de terminação e o checker mostra `SOME PROOFS FAIL`. **Proibido sem aviso** (regra 4). Sem `@unsafe`, não há como "contornar" uma prova, exceto deixar `?TODO`, que o checker acusa.
- Recursão mútua é proibida; o parâmetro que encolhe deve vir primeiro (a checagem de terminação lê da esquerda para a direita).
- Não existe `if`; usar `match` em `True{}`/`False{}`.
- `match` só inspeciona parâmetro ou variável de padrão, nunca valor calculado (`match f(x):` é recusado). É preciso passar por uma função auxiliar.
- Variáveis são **afins** (usadas no máximo uma vez); `+x` permite reuso se o tipo for `Data`. Isso vai afetar muito o autograd (um valor usado no forward e no backward precisa de `+`). Já avisar o Renan: é a parte mais estranha para quem vem de Python.

## BendHub

- Site: `hub.bend-lang.com` ("mini Hacker News": abas Packages e Posts, busca, `/post/new`, `/auctions`).
- Login: `bend login`, **só via GitHub**. Publicar: `bend arquivo.bend --publish nome@versao` (publica o arquivo e tudo que ele importa). Sem nome: `--publish` devolve o hash e a linha `import 0x<hash>/main.bend as P`.
- Importar: `import nome@versao/main.bend as P`, ou por hash.
- Publicação é **pública e permanente**; dependentes referenciam por hash. Colocar `LICENSE` ao lado do arquivo de entrada (primeira linha `SPDX-License-Identifier: MIT`). Sem `LICENSE`, vira MIT-0. Adicionar `LICENSE` depois muda o hash, e exige nova versão.
- **Nomes:** nomes com **12+ caracteres** são gratuitos (primeiro a chegar); nomes curtos são vendidos em leilão. **`bpe`, `tensor` e `nat-lemmas` são curtos demais ou próximos disso** (`nat-lemmas` tem 10). Usar nomes longos, por exemplo `bend-ml-bpe-tokenizer` e `bend-ml-nat-lemmas`. Nomes podem ser retirados em 14 dias se nenhuma versão foi vinculada; não podem ser transferidos.
- **Já existe algo parecido?** *Não verificado.* A página do hub carrega a lista por JavaScript e o fetch mostrou só os títulos das abas. Falta olhar o hub no navegador (busca por `tokenizer`, `bpe`, `tensor`, `autograd`, `matrix`, `nat`) antes de decidir entre contribuir e diferenciar. Ação para o Renan.

## Dívidas e riscos registrados

- Nenhuma dívida de prova ainda (nenhum código foi escrito).
- Risco: `--verdict` indisponível até instalar Lean.
- Risco: GPU indisponível até instalar CUDA 12.
- Risco: `Nat` unário pode ser lento para dimensões grandes.
- Risco: afinidade (uso único) de variáveis complica autograd e tensores.

## Recomendação de ajustes ao plano

1. **Antes da Fase 1, ação do Renan (≈15 min):** abrir `hub.bend-lang.com` e procurar `bpe`, `tokenizer`, `tensor`, `autograd`, `nat`. Se já houver um tokenizer, decidir entre contribuir e diferenciar.
2. **Nomes dos pacotes:** trocar `bpe`/`tensor`/`nat-lemmas` por nomes de 12+ caracteres (ver acima). Escolher e reservar cedo.
3. **Inverter a prioridade do `nat-lemmas`:** ele é pré-requisito real da Fase 2 e também ajuda a Fase 1 (provas sobre `List`). Sugestão: fazer um mini `nat-lemmas` + `list-lemmas` logo no começo da Fase 1, partindo do demo `proof_numerics`. É a menor entrega publicável e dá o primeiro pacote no BendHub cedo (tração).
4. **Fase 1 (BPE):** bytes são `U32` numa `List<U32>`, não há tipo byte. A lei do roundtrip precisa de uma definição de "tabela bem formada" que a prova use. Começar com a lei mais simples possível (tabela vazia → identidade) e subir.
5. **Fase 2 (tensor):** representar shape como `List<Nat>` no tipo, mas validar o custo do `Nat` unário e das variáveis afins numa prova de conceito pequena (vetor com tamanho no tipo + `dot`) antes de se comprometer. Se for pesado demais, usar `Nat` só no tipo (apagado em tempo de execução com `-`) e `U32` nos dados.
6. **Fase 3:** instalar CUDA 12 e testar `pow2!` antes de prometer GPU. Plano B: CPU paralela (`./prog --threads N`), que já é o padrão. MNIST em CPU deve bastar para a demo. GPT-2 em `F32` com 6 GB de VRAM é viável em tamanho (124M × 4 B ≈ 500 MB), mas depende de `File.read_bytes` aguentar o arquivo.
7. **Instalar elan + Lean 4.34.0** para usar `--verdict`. Entra no critério "pronto para mandar ao Taelin".

## Próximos passos sugeridos

1. Renan: conferir o BendHub no navegador (item 1).
2. Renan: instalar CUDA 12 (`/usr/local/cuda`) e/ou elan (`--verdict`) se quiser; posso ajudar com os comandos.
3. `git init` e remote no GitHub (hoje o diretório não é um repositório).
4. Prova de conceito: vetor com tamanho no tipo, para decidir a representação de shapes.
5. Começar o `nat-lemmas`.

## v0.1 — nat-lemmas (2026-10-03)

- 15 leis provadas em `nat-lemmas/main.bend` (Nat: soma/produto; List: append/length; `product_append`). `bend nat-lemmas/main.bend` → `ALL PROOFS CHECK`. Nenhum `@unsafe` nem `?TODO`. Nenhuma dívida de prova.
- **Descoberta:** um `import ... as NL` só expõe os defs do próprio arquivo, não os reexporta. Por isso o pacote publicável é **um arquivo só**, com `law` e prova lado a lado (a separação `LAWS.bend`/`PROOF.bend` fica para projetos de aplicação, não para bibliotecas).
- **Descoberta:** em `%e : P`, `_` marca onde está o lado direito `b` de `e : {a == b}`; o objetivo vira `P` com `a`. Quando o objetivo tem `a` e não `b`, usar `Equal.sym` antes.
- **Descoberta:** lemas sobre lista com elementos usados mais de uma vez precisam de `for +xs` e `Con{+h, +t}`.
- `--verdict` ainda não rodado (falta Lean 4.34.0).
- **Publicado no BendHub:** `bend-ml-nat-lemmas@0.1.0.0`, hash `0xa7aa06c09e97c6747c12cc64bb5203d9` (2026-10-03). Versões do BendHub têm **quatro números** (`0.1.0.0`). Import: `import bend-ml-nat-lemmas@0.1.0.0/main.bend as NL`. Verificado em pasta limpa.
- O comentário de uso em `nat-lemmas/main.bend` foi corrigido para `0.1.0.0` depois da publicação; só mudou comentário, mas o hash publicado é o do arquivo anterior.
