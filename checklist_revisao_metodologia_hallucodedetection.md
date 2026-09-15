# Plano de Revisão da Metodologia — HalluCodeDetection

## Objetivo deste documento

Este documento orienta a revisão da seção `Methodology` do artigo. O agente deve **editar a metodologia existente**, preservando o que já está correto e corrigindo apenas o que estiver incompleto, inconsistente ou pouco reproduzível.

Princípios:
- não inventar detalhes experimentais;
- quando uma informação ainda não estiver confirmada, inspecionar o código/repositório antes de escrever;
- se algo não puder ser confirmado no código, reportar explicitamente a dúvida em vez de assumir um comportamento.

---

# 1. Corrigir a descrição do treinamento especializado

## Problema atual

A metodologia atual dá a entender que:
- `Syntax` foi removido em todo o treinamento;
- todo o treinamento utilizou dados balanceados;
- as buscas Optuna foram executadas com uma quantidade fixa/completa de trials.

Isso não representa corretamente o experimento.

## Estrutura correta

### Busca 1 — Accuracy-oriented training
- distribuição original do dataset;
- sem balanceamento;
- `Syntax` permanece no dataset;
- loss padrão;
- alvo: validation Accuracy;
- busca de hiperparâmetros com Optuna.

### Busca 2 — Macro-F1-oriented / label-weighted training
- mesma distribuição original;
- sem remoção de `Syntax`;
- sem balanceamento explícito dos exemplos;
- uso de `label_weighted`;
- alvo: Macro-F1.

### Busca 3 — Balanced training / Accuracy
- balanceamento apenas no treino;
- redução de exemplos `Correct`;
- remoção de `Syntax`;
- validação e teste intactos;
- alvo de seleção: Accuracy.

### Busca 4 — Balanced training / Macro-F1
- mesma preparação balanceada;
- alvo de seleção: Macro-F1.

## Optuna
Não informar um número fixo de trials como se todas as buscas tivessem sido concluídas até esse limite. Informar apenas:
- Optuna foi utilizado;
- TPE foi utilizado, se confirmado no código/configuração;
- o espaço de busca;
- o critério de seleção do melhor modelo.

Se desejar citar quantidade, usar apenas trials efetivamente concluídos e verificados.

## Checklist
- [ ] Search 1 = original distribution + Accuracy.
- [ ] Search 2 = original distribution + label weighting + Macro-F1.
- [ ] Syntax não aparece como removida nas buscas 1 e 2.
- [ ] Balanceamento limitado às buscas 3 e 4.
- [ ] Validation/test permanecem intactos.
- [ ] Nenhum número arbitrário/fixo de Optuna trials.
- [ ] Critério de seleção explícito para cada busca.

---

# 2. Explicar corretamente o ground truth multilabel

O texto deve explicar que labels são acumuladas ao longo dos test cases.

Exemplo:
- alguns testes produzem saída incorreta → `Functional`;
- outros lançam exceção/timeout → `Runtime`;
- o mesmo código pode receber ambas.

`Correct` deve ser atribuído somente quando todos os testes passam e nenhuma outra classe de erro está presente.

O agente deve verificar no código:
- se `Correct` é formalmente mutuamente exclusivo;
- se `Syntax` encerra a avaliação;
- se um exemplo com `Syntax` pode receber outras labels.

## Checklist
- [ ] Acúmulo de labels explicado.
- [ ] `Functional + Runtime` simultâneo explicado.
- [ ] Exclusividade de `Correct` confirmada no código.
- [ ] Tratamento de `Syntax` confirmado.
- [ ] Nada escrito como se as classes fossem mutuamente exclusivas.

---

# 3. Criar pseudocódigo visual do ground truth

Adicionar um pseudocódigo simples, adaptado ao comportamento real do código.

Estrutura conceitual:

```text
Input: generated code, EvalPlus tests

labels = {}

if code does not compile:
    labels += Syntax
    return labels

for each test case:
    execute code

    if execution raises exception or timeout:
        labels += Runtime
        continue

    if output differs from expected result:
        labels += Functional

if labels is empty:
    labels += Correct

return labels
```

O agente deve conferir a implementação antes de usar esse fluxo literalmente.

## Checklist
- [ ] Pseudocódigo corresponde ao código real.
- [ ] Mostra Runtime e Functional acumuláveis.
- [ ] Mostra quando `Correct` é atribuído.
- [ ] Inclui timeout, se aplicável.
- [ ] Não introduz pacote LaTeX desnecessário.

---

# 4. Detalhar o protocolo de code generation

## Informações confirmadas
- uma geração por tarefa/modelo;
- temperatura = 1.0;
- `top_p` e `top_k` mantidos nos defaults do modelo/backend;
- sem seed fixa;
- backend não precisa ser detalhado agora, salvo necessidade de reprodução.

## O agente deve verificar no código

### Extração de código
Investigar:
- como a resposta bruta vira código;
- se há parser/extractor;
- se Markdown fences são removidos;
- se blocos ```python``` são identificados;
- se texto antes/depois do código é descartado;
- diferenças entre HumanEval e MBPP;
- como o cabeçalho da função é tratado;
- em que situações o cabeçalho pode ser perdido.

### Markdown/texto adicional
Investigar:
- se o conteúdo é limpo automaticamente;
- se resposta com texto extra vira `Syntax`;
- se falha de extração gera código incompleto;
- se existe fallback quando nenhum bloco é encontrado.

A metodologia deve separar:
1. generation;
2. response extraction/normalization;
3. ground-truth execution.

## Checklist
- [ ] Uma geração por tarefa/modelo.
- [ ] Temperature = 1.0.
- [ ] top_p/top_k descritos como defaults.
- [ ] Sem seed fixa.
- [ ] Extração descrita a partir do código real.
- [ ] Tratamento de Markdown/texto extra confirmado.
- [ ] Relação entre falha de extração e Syntax documentada.

---

# 5. Explicar corretamente a knowledge distillation com três judges

## Confirmado
Cada exemplo é enviado aos três judges:
- Muse Glimmer 30B;
- Qwen 3.8 27B;
- Ornith 1.5 35B.

Cada judge recebe a tarefa, o código e os labels de ground truth e gera sua própria explicação.

Portanto:
- três explicações por exemplo;
- sem votação;
- sem seleção de vencedor;
- judges não definem a classe;
- as explicações entram no dataset de treinamento;
- na avaliação quantitativa, apenas as classes previstas contam para as métricas.

## O agente deve verificar
- como as três explicações são armazenadas;
- se o exemplo é triplicado ou se as explicações são concatenadas;
- como `judge -> explanation` é representado;
- como respostas inválidas/ausentes são tratadas.

## Checklist
- [ ] Todos os três judges processam cada exemplo.
- [ ] Judges não classificam o ground truth.
- [ ] Sem votação.
- [ ] Três rationales por exemplo.
- [ ] Forma real de inserção no dataset confirmada.
- [ ] Métricas usam labels, não qualidade textual da explicação.

---

# 6. Justificar o uso de explicações/rationales no fine-tuning

A metodologia deve explicar por que o target contém `label + explanation` em vez de apenas label.

## Argumentação recomendada
Supervisão apenas por label informa qual saída é esperada, mas oferece pouca informação explícita sobre a relação entre:
- requisito da tarefa;
- comportamento do código;
- categoria de erro.

Em um dataset desbalanceado, isso pode favorecer correlações superficiais e shortcuts associados à classe dominante.

As explicações funcionam como **supervisão auxiliar mais rica**, com o objetivo de ensinar:
- qual classe emitir;
- quais propriedades do código justificam essa decisão.

## Literatura
Usar:
- Hsieh et al. — *Distilling Step-by-Step* (ACL 2023);
- Ludan et al. — *Explanation-based Finetuning Makes Models More Robust to Spurious Cues* (ACL 2023).

Usar Hsieh para sustentar que rationales podem transferir informação intermediária durante distillation.

Usar Ludan para sustentar que explanation-based fine-tuning pode reduzir dependência de spurious cues.

## Cautela
Não escrever:
- “explanations eliminate bias”;
- “rationales guarantee robustness”.

Preferir:
- “provide richer auxiliary supervision”;
- “reduce reliance on label-only shortcuts”;
- “encourage the model to associate predictions with evidence in the code”.

## Desenho do estudo
Enfatizar:
- o judge não cria o label;
- o label vem do ground truth por execução;
- o judge apenas gera uma explicação condicionada ao label correto.

As referências já estão em `refs.bib`. O agente deve localizar os bibkeys reais e não inventar chaves.

## Checklist
- [ ] Motivação das rationales adicionada.
- [ ] Hsieh et al. citado.
- [ ] Ludan et al. citado.
- [ ] Bibkeys reais usados.
- [ ] Nenhuma promessa de eliminação de bias.
- [ ] Ground truth continua objetivo e derivado da execução.
- [ ] Teacher explica o label, não o escolhe.

---

# 7. Explicitar o protocolo SLM vs Large LLM

Adicionar que os Large LLMs da RQ4 usam exatamente o mesmo protocolo de detecção dos SLMs:
- mesmo conjunto de exemplos;
- mesma task description;
- mesmo código de entrada;
- mesmo system prompt;
- mesmas definições de classe;
- mesmo formato de saída;
- mesmas métricas;
- mesma temperatura na detecção;
- sem acesso a testes ou ground truth.

## Checklist
- [ ] Protocolo idêntico explicitado.
- [ ] Mesmo input.
- [ ] Mesmas métricas.
- [ ] Mesmas labels.
- [ ] Nenhuma informação adicional para os Large LLMs.

---

# 8. Corrigir a descrição da quantização

## Correto
### Code generation inicial
SLMs locais: versões Q4_K_M para permitir avaliar modelos maiores dentro das limitações computacionais.

### Gemma
Também há avaliações sem redução de precisão, quando aplicável.

### Fine-tuning
Modelos base de fine-tuning e modelos fine-tuned não devem ser descritos como Q4_K_M.

Distinguir:
- Q4_K_M para inferência;
- QLoRA/NF4 durante o treinamento.

### Large LLMs
DeepSeek V4.1 Flash e MiMo V2.5 via API não devem ser descritos como Q4_K_M sem evidência explícita.

## Checklist
- [ ] Q4_K_M limitado à fase correta.
- [ ] Motivação computacional preservada.
- [ ] Gemma full precision separado.
- [ ] Q4_K_M não confundido com QLoRA/NF4.
- [ ] Fine-tuned models não descritos como Q4_K_M.
- [ ] Large LLMs por API sem quantização assumida.

---

# 9. Tornar o baseline Pyright reproduzível

## O agente deve investigar no código
Localizar:
- comando/configuração do Pyright;
- versão/config file, se houver;
- `typeCheckingMode`;
- diagnostic rules;
- severidades consideradas;
- filtro aplicado ao output;
- quais diagnósticos são mapeados para `Runtime`;
- quais são ignorados;
- tratamento de falha da ferramenta.

Documentar o mapping real, por exemplo:

```text
reportUndefinedVariable -> Runtime
reportCallIssue -> Runtime
...
```

Somente se confirmado pelo código.

### py_compile
Confirmar:
- falha de compilação -> `Syntax`;
- sucesso -> ausência de sinal de Syntax;
- não utilizado para Runtime ou Functional.

## Checklist
- [ ] Mapping Pyright→Runtime identificado no código.
- [ ] Diagnostic rules/severidades documentadas.
- [ ] Configuração do Pyright documentada.
- [ ] Falhas da ferramenta tratadas.
- [ ] py_compile→Syntax explicitamente definido.
- [ ] Nenhum mapping inventado.

---

# 10. Completar os ciclos interativos com ferramentas

Existem dois usos:
1. tool-assisted code generation;
2. tool-assisted code verification.

---

## 10.1 Tool-assisted code generation

## Confirmado
- máximo de 5 rounds;
- temperatura = 0;
- mesmos modelos das fases anteriores;
- modelo gera/revisa;
- ferramentas produzem feedback;
- modelo decide `Submit` ou revisa;
- ciclo para quando o modelo submete ou atinge 5 rounds;
- EvalPlus roda apenas no candidato final e nunca entra no prompt.

## O agente deve verificar

### Histórico
Confirmar se o modelo recebe:
- histórico completo das rodadas;
ou
- apenas task + candidato atual + feedback atual.

### Ordem das ferramentas
Confirmar:
- py_compile roda primeiro?
- Pyright só roda se compile passa?
- existe short-circuit?

### Nenhum diagnóstico
Confirmar:
- o modelo ainda faz self-review?
- recebe mensagem “no issue found”?
- pode revisar mesmo sem diagnostics?

### Atualização do candidato
Confirmar:
- revisão substitui completamente o candidato;
- existe validação antes da substituição;
- qualquer revisão pode virar o novo candidato.

## Regressões
Se o resultado final ficar pior que o inicial, isso deve ser contabilizado como regressão do processo interativo.

Na análise, considerar:
- improved;
- unchanged;
- regressed.

## Checklist
- [ ] 5 rounds.
- [ ] Temperature = 0.
- [ ] Critério de parada correto.
- [ ] Histórico verificado.
- [ ] Ordem py_compile/Pyright verificada.
- [ ] Comportamento sem diagnostics verificado.
- [ ] Atualização do candidato documentada.
- [ ] EvalPlus apenas no final.
- [ ] Regressões mantidas e mensuradas.

---

## 10.2 Tool-assisted code verification

Descrever a comparação controlada:

```text
Model without static feedback
vs.
Same model + same sample + static feedback
```

Manter:
- mesma task;
- mesmo código;
- mesmo prompt base;
- mesma temperatura;
- mesmas métricas.

O feedback é evidência, não ground truth.

O prompt deve deixar claro:
- ferramentas podem ser incompletas;
- ausência de diagnóstico não significa código correto;
- ferramentas não detectam Functional Error de forma geral.

## O agente deve verificar
- formato real do feedback no prompt;
- se mensagens “no error found” aparecem;
- se todos os modelos recebem o mesmo formato;
- se diagnostics são resumidos/transformados.

## Checklist
- [ ] Baseline sem ferramentas claramente definido.
- [ ] Tratamento difere apenas pelo feedback.
- [ ] Mesmos exemplos.
- [ ] Mesma temperatura.
- [ ] Feedback não tratado como label.
- [ ] Limitações das ferramentas explicitadas.
- [ ] Formato real do feedback documentado.

---

# Checklist global

## Treinamento
- [ ] Buscas 1–4 corretamente diferenciadas.
- [ ] Syntax removida apenas nas buscas balanceadas.
- [ ] Sem número arbitrário de Optuna trials.
- [ ] Quantização descrita apenas onde usada.

## Ground truth
- [ ] Multilabel explicado.
- [ ] Pseudocódigo incluído.
- [ ] Correct exclusivity confirmada.
- [ ] Runtime + Functional simultâneo explicado.
- [ ] Syntax treatment confirmado.

## Geração
- [ ] Uma geração por tarefa/modelo.
- [ ] Temperature = 1.0.
- [ ] top_p/top_k defaults.
- [ ] Sem seed fixa.
- [ ] Extração do código confirmada.
- [ ] Markdown/texto extra documentado.

## Distillation
- [ ] Três judges por exemplo.
- [ ] Três rationales.
- [ ] Sem votação.
- [ ] Judges não definem labels.
- [ ] Hsieh e Ludan citados corretamente.
- [ ] Métricas usam labels, não explicações.

## RQ4
- [ ] Mesmo protocolo para SLMs e Large LLMs.

## Static tools
- [ ] py_compile -> Syntax.
- [ ] Pyright mapping verificado no código.
- [ ] Diagnostics/severidades documentados.

## Interactive
- [ ] Máximo de 5 rounds.
- [ ] Temperature = 0.
- [ ] Histórico verificado.
- [ ] Ordem das ferramentas verificada.
- [ ] Comportamento sem diagnostics verificado.
- [ ] EvalPlus só no final.
- [ ] Regressões mensuradas.

## Regra final

Antes de editar qualquer trecho que dependa da implementação:

> **inspecione o código primeiro.**

Se o comportamento não puder ser confirmado:

> **não invente; reporte como não confirmado.**
