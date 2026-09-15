Claro. Com a nova divisão das RQs, a RQ3 fica mais focada: **não deve comparar com LLMs grandes** — isso agora pertence à RQ4. Ela deve responder apenas se **especializar um SLM para detecção melhora a tarefa e como diferentes estratégias de treinamento afetam o resultado**.

## RQ3 — Specialized SLMs for Error Detection

**Research Question:**  
*To what extent can specialized SLMs improve the detection of errors in generated code?*

### Objetivo da seção

Mostrar:

- se o fine-tuning melhora a capacidade de detecção em relação ao modelo base;
- como o desbalanceamento do dataset influencia o treinamento;
- como diferentes objetivos de otimização mudam o comportamento do detector;
- se balancear o conjunto de treino melhora ou piora o desempenho;
- quais classes mais se beneficiam ou sofrem regressão com cada estratégia.

---

## Estrutura sugerida da RQ3

### Parágrafo 1 — Dataset imbalance and motivation

Apresentar a distribuição original:

- Correct: 72.4%
- Functional: 21.3%
- Runtime: 6.6%
- Syntax: 1.5%

Explicar que:

- o dataset é fortemente desbalanceado;
- Accuracy favorece previsões da classe dominante;
- isso pode levar o modelo a “jogar seguro” prevendo `Correct`;
- por isso foram testadas estratégias diferentes de treinamento.

Também mencionar:

- split por `task_id`;
- balanceamento aplicado apenas ao treino;
- validation e test permaneceram intactos.

---

### Parágrafo 2 — First search: optimizing Accuracy

Descrever a primeira busca Optuna:

- loss `standard`;
- distribuição original;
- objetivo: validation Accuracy;
- busca sobre:
  - modelo base;
  - LoRA rank;
  - alpha;
  - dropout;
  - learning rate;
  - epochs;
  - bias.

Não listar todos os trials.

Reportar apenas:

- melhor configuração;
- melhor validation Accuracy;
- modelo base escolhido.

Melhor trial:

- Gemma 4 E4B;
- `r=16`;
- `alpha=16`;
- dropout `0.1`;
- lr `1e-4`;
- 3 epochs;
- bias `none`;
- val_acc = 66.6%.

---

### Parágrafo 3 — Behavior of the Accuracy-optimized model

Comparar **Gemma 4 E4B base** vs **Best Acc**.

Destacar:

- Correct Recall:
  - base: 38.0%
  - Best Acc: 85.3%
- Accuracy:
  - base: 45.9%
  - Best Acc: 77.8%

Usar a matriz de confusão para mostrar:

- base confunde muitos `Correct` como `Functional`;
- Best Acc corrige fortemente essa tendência.

Mas também mostrar o custo:

- Syntax cai drasticamente;
- Runtime melhora pouco;
- Functional pode passar a ser confundido com Correct.

Mensagem principal:

> Optimizing Accuracy substantially improves recognition of the dominant Correct class, but produces a detector less balanced across minority classes.

---

### Parágrafo 4 — Second search: optimizing Macro-F1

Explicar a motivação:

- Accuracy melhorou;
- porém o desempenho por classe continuou desequilibrado.

Segunda busca:

- mesma distribuição de dados;
- sem balanceamento do dataset;
- estratégia `label_weighted`;
- peso de classe = 3.0;
- objetivo: Macro-F1.

Melhor configuração:

- Gemma 4 E4B;
- `r=8`;
- `alpha=8`;
- dropout `0.1`;
- lr `1e-4`;
- 3 epochs;
- bias `none`;
- val_MF1 = 52.4%;
- val_MR = 54.9%.

---

### Parágrafo 5 — Accuracy vs Macro-F1 optimization

Comparar diretamente:

#### Best Acc
- Accuracy: 77.8%
- MF1: 50.9%
- CR: 85.3%
- FR: 69.1%
- RR: 27.8%
- SR: 11.8%

#### Weighted MF1
- Accuracy: 73.5%
- MF1: 54.2%
- CR: 75.9%
- FR: 77.9%
- RR: 32.9%
- SR: 64.7%

Discussão:

- Best Acc maximiza desempenho global;
- Weighted MF1 sacrifica alguns pontos de Accuracy;
- em troca, melhora:
  - Functional;
  - Runtime;
  - principalmente Syntax;
- o objetivo de otimização muda claramente o perfil do detector.

Esse deve ser um dos principais findings da RQ3.

---

### Parágrafo 6 — Balanced training experiment

Explicar a terceira estratégia:

- alterar a própria distribuição do treino;
- reduzir exemplos `Correct`;
- remover `Syntax`;
- testar proporções como 1:1:1 ou 1:3:3.

Objetivo:

> verificar se aumentar a representação relativa das classes minoritárias melhora a detecção.

Apresentar:

#### Balanced Acc
- Accuracy: 67.8%
- MF1: 38.1%
- CR: 74.4%
- FR: 65.8%
- RR: 15.2%
- SR: 0%

#### Balanced MF1
- Accuracy: 63.7%
- MF1: 36.0%
- CR: 68.6%
- FR: 65.8%
- RR: 13.9%
- SR: 0%

---

### Parágrafo 7 — Why balancing did not outperform weighting

Interpretar:

- balancear explicitamente o treino não superou as estratégias anteriores;
- Accuracy e MF1 caíram;
- Runtime também piorou;
- Syntax ficou em 0% porque a classe foi removida do treino.

Discussão:

- simples rebalanceamento de exemplos não garante melhor aprendizagem;
- remover exemplos Correct pode reduzir informação importante sobre a classe dominante;
- alterar a função objetivo/loss foi mais eficaz que alterar a distribuição do dataset.

Finding:

> Objective-oriented training was more effective than explicit dataset balancing in this setting.

---

### Parágrafo 8 — Syntax as a special case

Retomar os achados anteriores:

- Syntax é extremamente rara;
- muitos casos classificados como Syntax resultam de formato/estrutura de saída incompatível;
- não necessariamente de Python sintaticamente inválido.

Relacionar isso ao treinamento:

- classe pequena;
- heterogênea;
- difícil de aprender;
- muito sensível ao objetivo de treinamento.

Exemplos:

- Best Acc: SR 11.8%
- Weighted MF1: SR 64.7%
- balanced models: SR 0%

Não reclassificar os exemplos; apenas explicar o comportamento.

---

### Parágrafo 9 — Base vs specialized models

Fazer uma síntese direta do efeito do fine-tuning.

Comparar:

#### Base Gemma 4 E4B
- Acc: 45.9%
- MF1: 37.0%

#### Best Acc
- Acc: 77.8%
- MF1: 50.9%

#### Weighted MF1
- Acc: 73.5%
- MF1: 54.2%

Discutir:

- fine-tuning produz ganho substancial;
- a especialização muda fortemente a capacidade de reconhecer Correct;
- Weighted MF1 produz o detector mais equilibrado;
- Best Acc produz o detector mais forte em Accuracy.

---

### Parágrafo 10 — Changes in the error profile

Usar as matrizes de confusão para mostrar que o ganho não é uniforme.

Base:
- 49.2% de Correct → Functional.

Best Acc:
- Correct sobe para 85.5%;
- mas Functional → Correct cresce;
- Syntax cai.

Weighted MF1:
- Correct 76.2%;
- Functional 81.0%;
- Runtime 32.9%;
- Syntax 68.8%.

Mensagem:

> Fine-tuning changes the type of mistakes made by the detector, not only its aggregate performance.

---

## O que deve aparecer em tabela

### Tabela A — Dataset distribution

Manter:

- split;
- registros;
- Correct;
- Functional;
- Runtime;
- Syntax.

Nota:

- multilabel;
- percentuais podem somar >100%;
- split por tarefa;
- balanceamento apenas no treino.

### Tabela B — Grande tabela geral da RQ2

Pode ser referenciada aqui para comparar:

- base;
- Best Acc;
- Weighted MF1;
- Balanced Acc;
- Balanced MF1.

Não repetir tudo.

### Tabela C — Opcional: fine-tuning deltas

Se quiser resumir o efeito:

| Model | ΔAcc | ΔMF1 | ΔCR | ΔFR | ΔRR | ΔSR |
|---|---:|---:|---:|---:|---:|---:|

Tudo em relação ao Gemma 4 E4B base.

Essa tabela é opcional.

---

## O que fica fora da RQ3

Não discutir aqui:

- comparação com LLMs grandes;
- comparação de escala SLM vs LLM;
- ferramentas estáticas.

Esses pontos agora pertencem a RQ4 e RQ5.

---

## Answer to RQ3

A conclusão da RQ deve responder:

- sim, fine-tuning melhora significativamente a detecção;
- Accuracy-oriented training favorece Correct;
- Macro-F1-oriented training produz comportamento mais equilibrado;
- balanceamento explícito não superou a estratégia ponderada;
- o objetivo de treinamento influencia fortemente quais classes o modelo aprende melhor.

Uma síntese conceitual seria:

> Specialized training substantially improves SLM-based error detection, but the resulting detector is highly sensitive to the optimization objective and class-imbalance strategy. Accuracy-oriented fine-tuning favors the dominant Correct class, whereas Macro-F1-oriented training yields more balanced performance across error categories. Explicit dataset balancing did not outperform objective-based weighting in the evaluated setting.