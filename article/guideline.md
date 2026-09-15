# Guia de Escrita do Artigo — HalluCodeDetection

## 1. Objetivo central

Investigar até que ponto **Small Language Models (SLMs)** conseguem:

1. gerar código correto;
2. reconhecer falhas em código gerado;
3. beneficiar-se de treinamento específico para a tarefa de detecção;
4. competir ou complementar ferramentas tradicionais de análise estática.

A contribuição principal **não deve ser apresentada como uma nova taxonomia de alucinações**. A categorização dos erros deve ser tratada como uma adaptação de classificações já utilizadas na literatura.

---

## 2. Tarefa de classificação

Na etapa de detecção, o modelo recebe:

- a descrição da tarefa de programação;
- o código gerado para essa tarefa.

A partir dessas informações, deve identificar uma ou mais classes entre:

- **Correct**
- **Functional Error**
- **Runtime Error**
- **Syntax Error**

A tarefa deve ser descrita como **multilabel**, já que um mesmo código pode apresentar mais de um tipo de erro simultaneamente.

A classificação deve ser fundamentada em trabalhos prévios, como Dou et al. e outros estudos relacionados. O artigo não deve incluir a dimensão de qualidade de código utilizada por alguns trabalhos, pois ela não faz parte do experimento.

---

## 3. Research Questions

### RQ1 — Code Generation Capability

**To what extent can different SLMs generate correct code?**

Objetivo: avaliar quantitativamente o desempenho dos diferentes modelos na geração de código.

A análise deve considerar, quando aplicável:

- proporção de soluções corretas;
- frequência das diferentes categorias de erro;
- diferenças de desempenho entre os modelos avaliados.

Não realizar análise por família ou tamanho de modelo. Cada modelo deve ser tratado individualmente.

---

### RQ2 — Error Detection Capability

**To what extent can SLMs correctly identify errors in generated code?**

Objetivo: avaliar se SLMs generalistas conseguem reconhecer corretamente os erros presentes no código gerado.

A análise pode incluir:

- desempenho geral;
- precision;
- recall;
- F1;
- desempenho por classe;
- matriz de confusão ou equivalente apropriado para multilabel;
- diferenças entre os modelos;
- identificação das classes mais fáceis e mais difíceis de detectar.

---

### RQ3 — Specialized Error Detection

**To what extent can a model specifically trained for error detection improve the identification of errors in generated code?**

Objetivo: avaliar se o treinamento/fine-tuning de um modelo especificamente para a tarefa de detecção melhora o desempenho em relação aos modelos generalistas.

Devem ser apresentados:

- configuração do treinamento;
- dataset utilizado;
- divisão entre treino, validação e teste;
- métricas obtidas;
- comparação com os modelos não especializados;
- ganhos e perdas por classe.

A contribuição dessa RQ deve ser apresentada como a avaliação da viabilidade de especializar um SLM para detecção de erros em código gerado.

---

### RQ4 — Comparison with Static Analysis Tools

**How does model-based error detection compare with traditional static analysis tools?**

Objetivo: comparar a abordagem baseada em modelos com ferramentas tradicionais de análise estática.

A comparação deve considerar não apenas quem obtém melhor desempenho global, mas também quais tipos de falha cada abordagem consegue ou não identificar.

Se os resultados permitirem, discutir:

- cobertura;
- precision/recall;
- complementaridade;
- tipos de erro detectados apenas pelos modelos;
- tipos de erro detectados apenas pelas ferramentas estáticas;
- limitações de cada abordagem.

---

## 4. Estrutura sugerida do artigo

## 4.1 Introduction

Deve apresentar:

- crescimento do uso de modelos generativos para programação;
- problema de código incorreto/alucinado;
- dificuldade de confiar automaticamente em código gerado;
- necessidade de mecanismos de detecção;
- oportunidade de utilizar SLMs locais ou especializados;
- lacuna entre geração, detecção baseada em modelos e análise estática;
- objetivo do trabalho;
- RQs;
- principais contribuições.

---

## 4.2 Background and Related Work

Pode ser dividida em blocos como:

### 4.2.1 LLMs/SLMs for Code Generation

Trabalhos sobre geração automática de código e avaliação de corretude.

### 4.2.2 Errors and Hallucinations in Code Generation

Definições e classificações de erros/alucinações em código.

Aqui devem ser introduzidas as categorias utilizadas no artigo e os trabalhos que fundamentam sua adoção.

### 4.2.3 Detection of Incorrect Generated Code

Trabalhos que utilizam LLMs, classificadores ou outros mecanismos para verificar código gerado.

### 4.2.4 Static Analysis for Generated Code

Ferramentas e abordagens tradicionais relevantes para a comparação realizada na RQ4.

O Related Work deve deixar explícita a diferença entre o trabalho proposto e estudos que:

- apenas medem geração de código;
- apenas estudam alucinações;
- apenas utilizam ferramentas estáticas.

---

## 4.3 Methodology / Study Design

A metodologia deve permitir a reprodução do experimento.

### 4.3.1 Study Overview

Visão geral de todo o pipeline experimental.

### 4.3.2 Dataset and Programming Tasks

Descrever:

- origem dos problemas;
- quantidade de exemplos;
- critérios de seleção;
- preparação dos dados.

### 4.3.3 Evaluated Models

Apresentar:

- modelos utilizados;
- versões;
- configurações relevantes;
- forma de execução.

Não estruturar a análise em famílias ou tamanhos.

### 4.3.4 Code Generation Procedure

Descrever:

- como as soluções foram geradas;
- prompts;
- parâmetros;
- infraestrutura;
- condições experimentais.

### 4.3.5 Ground Truth and Error Classification

Explicar:

- como é determinado se o código é correto;
- como as quatro categorias são atribuídas;
- como os casos com múltiplos erros são tratados;
- natureza multilabel da tarefa.

### 4.3.6 Error Detection Experiment

Descrever como os SLMs recebem:

- descrição da tarefa;
- código gerado;

e realizam a classificação.

### 4.3.7 Specialized Model Training

Descrever:

- modelo base;
- procedimento de fine-tuning;
- dataset;
- divisão treino/validação/teste;
- hiperparâmetros;
- protocolo de avaliação.

### 4.3.8 Static Analysis Baselines

Descrever:

- ferramentas utilizadas;
- versões;
- configurações;
- saídas consideradas;
- como as saídas foram mapeadas para uma comparação válida com os modelos.

### 4.3.9 Evaluation Metrics

Apresentar:

- métricas utilizadas;
- cálculo;
- justificativa;
- métricas específicas para cenário multilabel, quando aplicável.

---

## 4.4 Results and Discussion

Organizar diretamente pelas RQs.

### 4.4.1 RQ1 — Code Generation Capability

Apresentar:

- resultados quantitativos gerais;
- diferenças entre os modelos;
- distribuição das categorias de erro;
- síntese dos principais achados.

Finalizar com uma resposta direta à RQ.

Exemplo de estrutura:

> **Answer to RQ1.** Síntese objetiva do principal resultado.

### 4.4.2 RQ2 — Error Detection Capability

Apresentar:

- desempenho dos diferentes modelos;
- métricas globais;
- métricas por classe;
- dificuldades de classificação;
- comparação entre modelos.

Finalizar com uma resposta direta à RQ.

### 4.4.3 RQ3 — Specialized Error Detection

Apresentar:

- desempenho do modelo especializado;
- comparação com modelos generalistas;
- ganhos e perdas;
- desempenho por classe;
- impacto do treinamento específico.

Finalizar com uma resposta direta à RQ.

### 4.4.4 RQ4 — Static Analysis Comparison

Apresentar:

- desempenho das ferramentas estáticas;
- comparação com abordagens baseadas em modelos;
- diferenças por classe;
- cobertura;
- possíveis complementaridades.

Finalizar com uma resposta direta à RQ.

---

## 5. Foco inicial da análise

O foco inicial deve ser **quantitativo**.

Neste momento, não é necessário criar uma análise qualitativa extensa dos exemplos.

Análises qualitativas de:

- padrões de alucinação;
- exemplos específicos;
- causas de falha;
- comportamentos inesperados;

podem ser adicionadas posteriormente apenas se os resultados quantitativos indicarem a necessidade de aprofundamento.

Não criar uma RQ específica de análise qualitativa por enquanto.

---

## 6. Threats to Validity

Criar uma seção específica cobrindo pelo menos:

### Construct Validity

Discutir se:

- as quatro classes representam adequadamente os tipos de erro estudados;
- as métricas escolhidas representam adequadamente a tarefa;
- o mapeamento das ferramentas estáticas para as classes é válido.

### Internal Validity

Discutir potenciais problemas relacionados a:

- processo de classificação;
- prompts;
- execução dos códigos;
- ground truth;
- erros de implementação;
- configuração dos modelos;
- configuração das ferramentas.

### External Validity

Discutir a generalização para:

- outros datasets;
- outras linguagens de programação;
- outros modelos;
- outros tipos de tarefa;
- outros cenários de geração de código.

### Conclusion Validity

Discutir:

- tamanho da amostra;
- estabilidade das métricas;
- desbalanceamento das classes;
- comparações estatísticas;
- interpretação dos resultados.

---

## 7. Conclusion

A conclusão deve sintetizar os resultados, e não apenas repetir a descrição do trabalho.

Ela deve responder, em sequência:

1. quão bem os SLMs geram código correto;
2. quão bem conseguem detectar seus erros;
3. se um modelo especializado melhora essa capacidade;
4. como essa abordagem se compara à análise estática;
5. quais implicações isso traz para o uso confiável de modelos de código.

Também deve apresentar:

- principais limitações;
- implicações práticas;
- possíveis trabalhos futuros.

---

## 8. Contribuições esperadas

O artigo deve posicionar suas principais contribuições aproximadamente como:

1. Uma avaliação empírica da capacidade de diferentes SLMs de gerar código correto.

2. Uma avaliação da capacidade de SLMs de identificar múltiplas categorias de erros em código gerado.

3. O treinamento e a avaliação de um modelo especializado para detecção desses erros.

4. Uma comparação sistemática entre detecção baseada em modelos e ferramentas tradicionais de análise estática.

5. Evidências sobre os limites e o potencial de SLMs como mecanismos de verificação de código gerado.

---

## 9. Narrativa central do artigo

A narrativa geral deve seguir esta sequência:

**gerar código → verificar se os próprios modelos reconhecem os erros → especializar um modelo para essa tarefa → comparar essa solução com ferramentas clássicas de Engenharia de Software.**

Essa sequência deve orientar:

- as RQs;
- a metodologia;
- a apresentação dos resultados;
- a discussão;
- a conclusão.

---

## 10. Ordem sugerida de escrita

A escrita pode seguir a seguinte ordem de trabalho:

1. **Results and Discussion**
2. **Conclusion**
3. **Methodology**
4. **Related Work**
5. **Introduction**
6. **Abstract**

A ideia é começar pelos resultados já consolidados e fazer com que as demais seções sejam construídas a partir das evidências efetivamente obtidas.
