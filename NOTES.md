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

## Ambiente e v0.2 — PoC de shapes (2026-10-03)

- `reference/.venv`: torch 2.14.1+cpu, tiktoken 0.14.0, numpy 2.5.3, safetensors. Lean via elan não instalou direto (falha de DNS no elan), mas `bend --verdict` compilou o próprio kernel e **`nat-lemmas/main.bend --verdict` dá ALL PROOFS CHECK** (primeiro run ~1 min, depois 0,2 s).
- CUDA: o pacote do Arch é o **CUDA 13.3**; o Bend pede **CUDA 12 em `/usr/local/cuda`**. Não instalei (exige sudo e versão errada). Tudo segue em CPU paralela por enquanto.
- **Vec(n) por recursão de tipo** (`def Vec(n) -> Data: match n`, como `Word(n)` na Base): funciona e o erro de tamanho é de tipo, mas **estoura a pilha em n ≈ 50 mil** (8192 ok). Serve para dimensões pequenas, não para dados de 38M de floats.
- **Decisão de representação:** `type Mat<-r, -c> is Type: Mat{data: Array<F32>}`. Dimensões são parâmetros de tipo **apagados**; dados em `Array<F32>` linha a linha (índice `i*c + j`). Dimensão incompatível é erro de tipo (`poc/mat_shape_error.bend`, mensagem em `docs/shape-error-matriz.txt`). Produto linha×coluna de 784 elementos: 0,002 s nativo.
- Limite honesto: o tamanho do `Array` (potência de 2) **não** é amarrado ao tipo; a invariante `|data| >= r*c` vale por construção pelos construtores do pacote, não por prova. A prova de shapes é sobre a álgebra (`reshape` exige `product` igual, com `nat-lemmas`).
- Regras do Bend aprendidas: `match` só abre **parâmetros**, então um par devolvido por `Array.get` precisa de função própria ou de ser passado como argumento (padrão de `Array.map.go`); sem recursão mútua; pares casam como `Tuple{a, b}` no `match`; funções precisam estar definidas acima de onde são usadas; tipos com `Array` são `Type` (afins), não `Data`.

## v0.3 — bend-ml-bpe-tokenizer (2026-10-03)

- Publicado: `bend-ml-bpe-tokenizer@0.1.0.0`, hash `0x3333bd5274f4ce66fdf8fe0c26c6b532`. Leis `roundtrip`, `vocab_bound`, `dec_append` provadas e aceitas pelo kernel (`--verdict` ALL PROOFS CHECK). Import verificado em pasta limpa, inclusive usando a lei publicada.
- **Desenho que tornou a prova viável:** tokens `B{n}`/`M{k}` (sem aritmética de índices), cada regra carrega seu id, a tabela vai da regra mais antiga para a mais nova e o acumulador `done` do `encode.go` evolui exatamente como `List.reverse.go` da Base. A expansão (`exp`) é recursiva sobre a lista de regras (só olha regras mais antigas), sem tabela de vocabulário nem combustível. A condição "tabela bem formada" é só "ids distintos" (`wf`).
- ~20 lemas auxiliares (`ext_dec`, `mp`, `kp`, `enc_dec`...). Técnicas aprendidas: `match` só em parâmetros ⇒ passar `hit: Bool` calculado pelo chamador e provar `hit == peek(...)`; recursão só estrutural com o argumento que encolhe PRIMEIRO; `%e : P` precisa de `_` no lado direito `b` de `e`, então use `Equal.sym` para reescrever no outro sentido; refutar `True == False` com o motivo `BD(b, t, f)`.
- Testes: `reference/test_bpe.py` compara `train/encode/decode` em Bend com `reference/bpe_ref.py` em 4 corpora × 4 amostras (inglês, acentos, UTF-8 misto com emoji/CJK, vazio): 0 falhas.
- O `bpe/LICENSE` publicado não tem a linha `SPDX-License-Identifier` (o instalador mostrou "License: see LICENSE"); `nat-lemmas` tem. Texto MIT idêntico.
- Pendente do GPT-2: pré-tokenizador por regex e carga do `merges.txt` oficial (tabela de 50 mil regras, ids base em ordem `bytes_to_unicode`).

## v0.4 — bend-ml-tensor (2026-10-03)

- Publicado: `bend-ml-tensor@0.1.0.0`, hash `0xf9837737d2c3f58ae1d0c5df42255584`. `ALL PROOFS CHECK` e `--verdict` OK; import e erro de shape verificados a partir do BendHub em pasta limpa.
- **Representação final:** `Vec<-n>` e `Mat<-r,-c>` com dimensões apagadas, dados em `List<&2, F32>` / lista de linhas (copiáveis e paralelizáveis). Medido: 10 M de multiplicações-e-somas em 0,23 s numa thread (~44 M/s). Descartado `Array` (índice log n com `Array.size` a cada acesso) e `Vec(n)` por recursão de tipo (pilha estoura perto de 50 mil).
- Erro de shape = erro de tipo: `docs/shape-error-bad_matmul.txt` (`esperado Mat<3,5>, recebido Mat<4,5>`) e `docs/shape-error-bad_reshape.txt` (`esperado 12, recebido 15`). Esse é o conteúdo do vídeo.
- Leis: `reshape_swap` (r·c = c·r) e `reshape_flat` (r·c = 1·(r·c)), provadas com `mul_comm` e `mul_one_l` do nat-lemmas publicado.
- Testes numéricos: `reference/test_tensor.py` compara com PyTorch: matmul (4 formas), transpose, relu, softmax (inclusive valores ±1000), gelu (tanh), layernorm. Máximo erro ≈ 1e-6, 0 falhas.
- Regras novas: parâmetro usado também no tipo conta como uso (precisa `+` se for usado de novo); defs precisam vir antes do uso (nada de ordem livre); `match` na ordem dos binders.

## v0.5 — bend-ml-autograd (2026-10-03)

- Publicado: `bend-ml-autograd@0.1.0.0`, hash `0x174ef0d27bc1c621ddb9961c7f224de1`. `ALL PROOFS CHECK` e `--verdict` OK.
- **Lei provada `reverse_eq_forward`**: modo reverso == modo direto do autodiff, sobre expressões de `Nat` (constantes, X, soma, produto). Prova por indução do resultado mais forte `reverso(e,g) = g × direto(e)`, usando `mul_comm`, `mul_assoc`, `mul_dist`, `mul_zero`, `mul_one_*` do nat-lemmas publicado e uma `dist_l` derivada (distributividade à esquerda).
- Autograd de escalares em `F32` (GCst, GVar, GAdd, GMul, GRelu, GTanh, GExp) e camadas com backward tipado (`linear_bwd` devolve `Mat<n,i> & (Mat<i,o> & Vec<o>)`; trocar uma dimensão não compila).
- Gradient checking: `reference/test_autograd.py` compara com o autograd do PyTorch, 21 verificações (5 expressões × 3 pontos, linear_bwd ×3, relu_bwd, cross-entropy ×2), erro máximo ≈ 5e-7, 0 falhas.
- Bend: argumentos negativos na linha de comando precisam de `--` (`bend cli.bend -- scalar 1 -1.3 ...`).

## v0.6 — MNIST (em andamento)

- `demos/mnist/train.bend` (MLP 784-128-10, SGD, lotes de 100) e o gêmeo `reference/mnist_torch.py` com os mesmos pesos iniciais (`demos/mnist/data/init/*.txt`) e mesma ordem de lotes.
- **Validação de corretude:** após 50 passos, Bend: `loss_treino=1.6616005`, `7829/10000`; PyTorch: `loss_treino=1.661600`, `acc_teste=0.7829`. Idênticos.
- Desempenho (honesto): ~0,84 s por lote de 100 numa thread (≈ 24 M mult-soma/s efetivos). O paralelismo de `x y = f g` escala bem no `pow2` (8× com 16 threads), mas o `matmul` por linhas em listas só chegou a ~1,8× (com ou sem compartilhamento de dados); causa não resolvida. O `bend -o` + `--threads N` é o mecanismo.
- `tensor@0.1.1.0` acrescenta `Mat.matmul_t` (operando já transposto) para o GPT-2.

## v0.6 e v1.0 — MNIST, GPT-2 e entrega (2026-10-03)

- **MNIST (1 época completa, 600 lotes de 100, lr 0,1):** Bend perda `0,52047706`, 9129/10000, 544,3 s; PyTorch perda `0,520477`, 9129/10000 (9128 com 1 thread), 0,21 s (16 threads) / 0,30 s (1 thread). Correção idêntica; o Bend é ~1800× mais lento por falta de BLAS e por usar listas. Detalhes e hardware em `demos/mnist/BENCHMARK.md`.
- **GPT-2 small (124 M) em Bend:** pesos carregados em ~10 s (bytes -> F32 por bitcast `U32{w}` -> `F32{w}`, em blocos de 1 MB); ~3 s por token com KV cache; ids idênticos aos do PyTorch em 22 tokens (3 prompts), logits a menos de 2e-4. `reference/test_gpt2.py`.
- **Tokenizer do GPT-2 em Bend:** pré-tokenizador da regex sobre bytes + as 50 000 regras via `bend-ml-bpe-tokenizer`; 16/16 textos idênticos ao `tiktoken`. A equivalência "aplicar as regras em ordem == algoritmo por rank do GPT-2" foi confirmada contra o tiktoken real (`reference/gpt2_prep.py`). Limite: bytes >= 128 contam como letra.
- **Pacotes publicados:** `bend-ml-nat-lemmas@0.1.0.0` (`0xa7aa06c0...`), `bend-ml-bpe-tokenizer@0.1.0.0` (`0x3333bd52...`), `bend-ml-tensor@0.1.0.0` (`0xf9837737...`) e `0.1.1.0` (`0xdbe1000e...`, acrescenta `matmul_t`), `bend-ml-autograd@0.1.0.0` (`0x174ef0d2...`). Todos importam do BendHub numa pasta limpa e as leis são reutilizáveis (`BPE.roundtrip`, `AG.reverse_eq_forward`).
- `reference/check_all.py --full`: 25/25 verificações ok (provas + kernel, erros de shape esperados, referências Python, tokenizer, GPT-2, MNIST).

### Dívidas e limites conhecidos

- **Nenhuma dívida de prova**: nenhum `@unsafe`, nenhum `?TODO` publicado.
- **Paralelismo não resolvido:** o `pow2` do guia escala ~8× com 16 threads, mas o `matmul` por linhas sobre listas só chegou a ~1,8× (com ou sem dados compartilhados, em qualquer profundidade). Hipóteses descartadas: laziness (somas estritas nas folhas), sharing (dados privados por tarefa). Não investigado: alocador compartilhado, acesso de memória por ponteiros. Vale reportar no GitHub do Bend.
- **GPU não usada:** o Bend pede CUDA 12 em `/usr/local/cuda`; o pacote `cuda` do Arch é o 13.3 e a instalação exige sudo. A RTX 4050 (6 GB) ficou fora dos benchmarks.
- A invariante "linhas com `c` números" de `Mat<r,c>` não está no tipo.
- O `bpe/LICENSE` publicado não tem a linha SPDX (só o texto MIT).
- Para o Renan: (1) abrir hub.bend-lang.com e procurar tokenizer/tensor/autograd de terceiros (a lista é carregada por JS e não consegui ler); (2) gravar o vídeo e postar a thread (`docs/launch/`); (3) se quiser GPU, instalar CUDA 12 em `/usr/local/cuda`; (4) o token do `bend login` está em `~/.bend/bender.json` (permissão 600); a chave `bend-ml` foi revogada.

## v2 — experimentos de desempenho (2026-10-04)

Plano em `docs/v2-plan.md`; micro-benchmarks em `bench/` (`timeit.py` mede a mediana de 3 execuções por número de threads).

### Exp. 1-2: listas × `Array<F32>` (100 M multiplicações-e-somas, 1 thread)

| | tempo | multiplicações-e-somas/s |
|---|---|---|
| listas (`bench/mm_list.bend`, como o `bend-ml-tensor` 0.1.x) | 2,274 s | 44 M |
| `Array<F32>` com `Array.get` (`bench/mm_array_check.bend`, dados distintos) | **0,046 s** | **2 200 M** |

- **Ganho de ~49× só trocando a estrutura de dados.** Resultado conferido com NumPy (`1024010000` para 128001 produtos de 8000,79; o desvio pequeno é arredondamento de `F32`).
- Conta de bolso para o MNIST: ~18 G de mult-soma por época (3 produtos por lote × 600 lotes). A 2,2 G/s dá ~8 s por época, contra 544 s com listas e 0,3 s do PyTorch com 1 thread. **A distância ao PyTorch cairia de ~1800× para ~30×**, antes de qualquer paralelismo. Isto é uma estimativa, não uma medição do treino completo.
- Custo do `Array`: é afim (`Type`), então cada leitura devolve o array junto com o elemento, e `Array.get` recalcula `Array.size` (log n). Mesmo assim ganhou muito.
- A conclusão da v1 ("o Bend é ~1800× mais lento") era um efeito da estrutura de dados, não da linguagem.

### Exp. 3-5: paralelismo, cache, clone e distância ao PyTorch (2026-10-04)

Todos os números: 1 thread salvo indicação, mediana de 3 execuções, Intel Core Ultra 7 155H (6 P + 8 E + 2 LP-E cores, 22 threads), `bend -o`. Código em `bench/`.

- **Indexar vence percorrer a árvore:** `Array.get` com índice: 2 200 M mult-soma/s; percorrer as duas árvores juntas e reconstruí-las (`bench/tree_dot.bend`): 31 M/s (70× pior). Em Bend, o caminho rápido do `Array` é o indexado.
- **`Array.clone` é barato** (~17 µs por array de 131 072 elementos, com leitura do resultado forçada). **A hipótese "o clone compartilha nós e as leituras viram atômicas disputadas" foi refutada:** 16 tarefas lendo clones de um mesmo par escalam como 16 tarefas com arrays privados (`bench/share_test.bend`: ~4,1× com 16 threads nos dois casos).
- **Paralelismo, ordem de grandeza:** com trabalho independente e arrays pequenos (784 elementos), 16 threads dão ~4 a 6× (3,2 G mult-soma: 0,88 s em 1 thread, 0,13 s em 22). No produto 100×784×128 em blocos de linhas (`bench/mm_par.bend`), ~2,1× com 8 threads, igual para 2^d = 8 ou 16 blocos. Não depende de blocagem (`bench/mm_tile.bend`: 1, 8, 16 e 32 blocos de colunas, mesmo tempo) nem do custo dos clones (22% do tempo). O `pow2` do guia chega a 8×, então o limite é do tipo de carga. CPU híbrida: o teto prático é menor que 22×.
- **O tamanho do `Array` custa:** o mesmo produto com arrays de 2^20 casas em vez de 2^17 ficou ~60% mais lento (1,9 s -> 3,1 s para 4 G mult-soma): a profundidade da árvore entra em cada `get`. Alocar o menor array possível.
- **Distância ao PyTorch no produto 100×784×128:** PyTorch 1 thread = 0,162 ms (62 G mult-soma/s); 16 threads = 0,047 ms (212 G/s). Bend: listas 44 M/s (**~1400× atrás**), `Array` 1 thread 2,3 G/s (**~27× atrás**), `Array` com blocos paralelos ~4,5 G/s (**~47× atrás do PyTorch paralelo**). O que sobra: código escalar contra AVX/FMA com BLAS.

### Exp. 6: MNIST completo sobre `Array` (`demos/mnist/fast.bend`, 2026-10-04)

Matrizes planas em `Array<F32>`, `gemm` com strides (cobre `X·W`, `H·Wᵀ` e `Xᵀ·dZ` sem transpor), bias/relu/máscara/SGD em lugar por índice, softmax e entropia cruzada por linha com listas de 10, lotes lidos do arquivo a cada passo (`File.read_at`). 1 thread.

| 50 lotes | perda | acertos | tempo |
|---|---|---|---|
| listas (`train.bend`) | 1,6616005 | 7829 | ~42 s |
| **`Array` (`fast.bend`)** | **1,6616004** | **7829** | **0,9 s** |
| PyTorch | 1,661600 | 7829 | 0,02 s |

| 1 época (600 lotes) | perda | acertos | tempo |
|---|---|---|---|
| listas | 0,52047706 | 9129 | 544,3 s |
| **`Array`** | **0,5204771** | **9129** | **11,5 s** (47× mais rápido) |
| PyTorch 1 thread | 0,520477 | 9128 | 0,30 s |
| PyTorch 16 threads | 0,520477 | 9129 | 0,21 s |

Distância ao PyTorch (1 thread): de ~1800× para **~38×**. Correção preservada (mesma perda e mesmos acertos).
Os ~19 ms por lote se dividem em ~9 ms de `gemm` (20 M mult-soma a 2,3 G/s) e ~10 ms de todo o resto (monta `X` com 78 400 `set`, lê 78 KB, bias/relu/máscara/SGD sobre ~100 mil elementos com get+set).

### Exp. 7: `gemm` paralelo no MNIST (`fast.bend` agora usa 2^3 blocos nos dois produtos grandes)

Blocos de linhas de C, cada um com cópia de A e B (`Array.clone`), resultados em listas escritas em C no fim. Mesma correção (50 lotes: 1,6616004 e 7829; época: 0,5204771 e 9129).

| 1 época | tempo |
|---|---|
| `Array`, sequencial | 11,5 s |
| `Array`, gemm paralelo, 2^2 blocos, 8 threads | 7,2 s |
| `Array`, gemm paralelo, 2^3 blocos, 16 threads | **6,3 s** |
| `Array`, gemm paralelo, 2^4 blocos, 16 threads | 7,5 s |

Distância ao PyTorch: 6,3 s contra 0,21 s (16 threads, ~30×) e 0,30 s (1 thread, ~21×). O que sobra é a parte sequencial (carga de dados ~2 s por época; bias/relu/máscara/SGD sobre ~100 mil elementos): lei de Amdahl.

### Exp. 8: pacote `bend-ml-tensor-array` e MNIST com a API tipada (2026-10-04)

- Publicado: `bend-ml-tensor-array@0.1.0.0` (MIT, `--verdict` ok, sem `@unsafe`). `Mat<r,c>` sobre `Array<F32>` plano; `matmul`, `matmul_nt`, `matmul_tn` com blocos paralelos; bias, relu, `relu_bwd`, SGD, soma de colunas, `softmax_ce`, `count_correct`. 39 verificações contra o PyTorch (`reference/test_tensor_array.py`, erro máximo ≈ 1,4e-6). Importa do BendHub numa pasta limpa.
- **Erro de shape também no gradiente:** `tensor-array/tests/bad_grad.bend` (`dW` pedido como 128×784 em vez de 784×128) não compila (`docs/shape-error-array-*.txt`).
- `demos/mnist/fast.bend` agora usa só a API tipada (importando do BendHub): 50 lotes 1,6616004 / 7829; **1 época 0,5204771 / 9129 em 7,6 s** (a versão com kernels crus levava 6,3 s: o custo extra são as conversões lista <-> Array dos wrappers `of_list`/`scale255`, ainda não otimizadas).
- Regras do Bend aprendidas: `Mat` é `Type` (afim), então `Maybe<&1, Mat<...>>` e não `&2`; um registro `MMul{a, b, c}` por operação resolve o "devolver o que leu"; um padrão aninhado de `&`/`Tuple` em `match` falha ("annotated term (cannot infer)"), use registros com um `type` próprio.
