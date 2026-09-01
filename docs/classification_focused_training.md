# Treinamento orientado à classificação

## Objetivo

O projeto gera uma resposta JSON com duas partes: os rótulos em `levels` e uma
`explanation` para cada rótulo. A classificação é o resultado principal; a
explicação é evidência legível para a decisão.

O treinamento padrão de SFT trata todos os tokens da resposta-alvo de forma
igual. Assim, uma explicação longa pode contribuir muito mais para a loss do
que os poucos tokens que representam `correct`, `functional`, `runtime` ou
`syntax`.

## Estratégias de loss

`training.classification_loss_strategy` permite executar experiências
comparáveis:

```yaml
# Baseline: loss SFT normal em todos os tokens.
classification_loss_strategy: standard

# Variante orientada à classificação.
classification_loss_strategy: label_weighted
classification_loss_weight: 3.0
```

No modo `label_weighted`, a cross-entropy dos tokens que codificam os quatro
nomes de classe recebe o multiplicador configurado. Os demais tokens, inclusive
as explicações, continuam supervisionados com peso 1. Isso preserva a capacidade
de justificar, mas aumenta o sinal de treinamento associado à decisão de classe.

O peso `3.0` é um ponto de partida. Ele não significa que a classe recebe 75%
da loss: a contribuição final depende do número de tokens da explicação. Para
uma comparação justa, mantenha dataset, seed, espaço de hiperparâmetros e número
de trials iguais; altere apenas a estratégia e use um nome de estudo distinto.

## Métricas multilabel

Para cada geração, os conjuntos de rótulos esperados e previstos são comparados
entre `correct`, `functional`, `runtime` e `syntax`.

- **Accuracy**: correspondência exata entre os dois conjuntos.
- **Recall da classe**: `TP / (TP + FN)` para cada classe.
- **Macro Recall**: média simples dos quatro recalls.
- **F1 da classe**: `2TP / (2TP + FP + FN)`.
- **Macro F1**: média simples dos quatro F1s.

As métricas são multilabel: uma geração pode ter mais de uma classe. Respostas
com JSON inválido ou sem rótulos válidos são previstas como conjunto vazio e,
portanto, geram falsos negativos para os rótulos esperados. Uma exceção de
inferência é registrada como exemplo pulado; um trial de busca deve ser tratado
com cautela se não completar toda a validação.

O parser aceita agora itens com `level` mesmo quando `explanation` está ausente.
Isso evita descartar uma classificação correta por falta da justificativa; a
explicação ainda é solicitada e treinada.

## Optuna e retenção de modelos

Escolha a métrica primária no YAML:

```yaml
optuna_target: macro_f1      # accuracy | macro_f1 | macro_recall
```

O Optuna TPE maximiza a métrica escolhida. Ao reter modelos no SSD, os empates
são ordenados por: métrica alvo, Macro F1, Macro Recall e Accuracy. Portanto,
com `optuna_target: macro_f1`, Macro Recall é o desempate principal após F1.

Cada trial registra Accuracy, Macro F1 e Macro Recall no JSONL e no storage do
Optuna. Trials antigos, feitos antes dessas métricas existirem, não podem ser
importados em um estudo cujo alvo seja `macro_f1` ou `macro_recall`. Use nomes
de estudo diferentes, por exemplo:

```yaml
optuna_study_name: hallucination_detection_macro_f1_standard
classification_loss_strategy: standard
```

e:

```yaml
optuna_study_name: hallucination_detection_macro_f1_weighted
classification_loss_strategy: label_weighted
classification_loss_weight: 3.0
```

Use também diretórios de saída distintos para cada experimento. A configuração
atual usa `models/macro_f1_weighted` e `checkpoints/macro_f1_weighted` no SSD.
A política de retenção considera somente modelos do mesmo diretório de saída,
portanto não remove os modelos de uma variante ao executar a outra.
