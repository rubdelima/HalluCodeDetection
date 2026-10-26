# Plano executável de revisão para adequação ao SIGSOFT

## Objetivo e regra de integridade metodológica

Este plano orienta um agente a revisar e atualizar o artigo e o seu pacote de replicação sem fabricar resultados, metadados, hipóteses ou evidência estatística. O estudo deve permanecer explicitamente **guiado por RQs e exploratório** se as hipóteses confirmatórias não foram definidas antes de observar os resultados. Não introduza H0/H1 retrospectivas para transformar resultados já observados em testes confirmatórios.

O objetivo não é aplicar um teste de hipótese a toda RQ por simetria. Para cada RQ, o agente deve primeiro definir o estimando e decidir se há uma comparação inferencial válida. RQs descritivas devem reportar estimativas, distribuição e intervalos de confiança; RQs comparativas podem incluir testes two-sided e tamanhos de efeito, quando as unidades de análise e os pressupostos permitirem.

## Política para hipóteses e testes inferenciais

1. Preserve as cinco RQs como questões pré-definidas do estudo.
2. Inclua na metodologia uma declaração como: *“This study is research-question driven and exploratory; it does not formulate a priori confirmatory hypotheses. When a predefined RQ involves a comparison, we use two-sided inferential tests to quantify evidence for the observed difference.”*
3. Não crie H0/H1 novas a partir dos resultados. Testes inferenciais exploratórios são aceitáveis, desde que sejam identificados como tais e reportem estimativa de efeito, IC e correção de múltiplas comparações.
4. Não remova automaticamente o teste existente de RQ1. Mantenha-o se a auditoria confirmar que as mesmas tasks foram avaliadas nas condições comparadas e que os pressupostos são atendidos. Corrija sua apresentação, justificativa e relato de efeito/IC.
5. Não aplique p-values a uma RQ somente porque outra RQ tem p-value. Use inferência para toda RQ **comparativa** que tenha dados adequados; use apenas estatística descritiva e IC para RQs cuja finalidade seja caracterizar, medir prevalência, mapear erros ou relatar um caso de uso sem comparação causal/comparativa definida.

## Matriz de decisão estatística por RQ

O agente deve preencher esta matriz com os nomes e dados reais das RQs antes de alterar o manuscrito.

| RQ | Pergunta/estimando real | Tipo: descritiva ou comparativa | Unidade de análise | Pareamento disponível? | Análise a usar | Teste possível? | Decisão e justificativa no artigo |
|---|---|---|---|---|---|---|---|
| RQ1 | Preencher a partir do artigo | Preencher | Task | Sim/não | Preencher | Preencher | Preencher |
| RQ2 | Preencher a partir do artigo | Preencher | Preencher | Preencher | Preencher | Preencher | Preencher |
| RQ3 | Preencher a partir do artigo | Preencher | Preencher | Preencher | Preencher | Preencher | Preencher |
| RQ4 | Preencher a partir do artigo | Preencher | Preencher | Preencher | Preencher | Preencher | Preencher |
| RQ5 | Preencher a partir do artigo | Preencher | Preencher | Preencher | Preencher | Preencher | Preencher |

### Regras de escolha

- **Corretude binária de vários modelos nas mesmas tasks:** use Cochran's Q para a comparação global; se significativo, use McNemar two-sided em pares pré-especificados ou em todos os pares, com Holm. Reporte também diferença de proporções pareada em pontos percentuais e IC 95% por bootstrap pareado sobre tasks.
- **Corretude binária de duas condições nas mesmas tasks** (por exemplo, com/sem ferramenta): use McNemar two-sided. Reporte matriz de discordâncias, diferença de proporções pareada e IC 95% por bootstrap pareado.
- **Métricas de detector multilabel** (macro-F1, recall, precision, subset accuracy): reporte a métrica por condição e IC 95% por bootstrap que reamostre a unidade independente correta. Se as várias amostras pertencem à mesma task, faça bootstrap por task, não por linha, para preservar dependência. Para comparar detectores nas mesmas tarefas, use teste de permutação pareado two-sided sobre a diferença da métrica; use-o somente se a implementação e o desenho satisfizerem a troca sob a hipótese nula.
- **Frequências de tipos de hallucination, distribuição de classes e cobertura:** reporte contagem, proporção e IC 95% de Wilson (ou bootstrap por task, se houver agrupamento). Não use teste de hipótese sem uma comparação definida.
- **Número de iterações, custo, latência ou métricas contínuas pareadas:** reporte mediana, IQR e diferença pareada com IC 95% por bootstrap. Só use Wilcoxon signed-rank se o pareamento for válido e a interpretação de diferença for apropriada; caso contrário prefira teste de permutação pareado two-sided.
- **Experimento sem replicação, sem pareamento válido ou com observações dependentes não modeladas:** não execute teste de hipótese. Reporte a limitação e use apenas descritivos/IC compatíveis com a unidade de análise.
- **Inferência além de EvalPlus:** não trate as 563 tasks como amostra aleatória de todos os programas. As conclusões inferenciais devem ser condicionadas ao benchmark; a generalização externa deve ser tratada em Threats to Validity.

## Lista de execução do agente

O agente deve marcar cada item com `[x]` somente depois de executar a ação, atualizar os artefatos indicados e registrar o caminho/commit ou comando usado abaixo do item. Itens bloqueados devem permanecer `[ ]` e receber uma nota `BLOCKED:` com a causa e o impacto no texto.

### A. Inventário, rastreabilidade e congelamento dos dados

- [x] Localizar a versão-fonte do manuscrito, o pacote de replicação, tabelas, figuras e todos os arquivos brutos usados em cada RQ.
  Evidência: `artifacts/manifest.md`.
- [x] Criar `artifacts/manifest.md` com: arquivo de dados, RQ, número de tasks/amostras, modelos/condições, unidade de análise, colunas, transformações e tabela/figura do artigo que o consome.
  Evidência: `artifacts/manifest.md`.
- [ ] Verificar que os números publicados podem ser reproduzidos dos dados brutos. Não sobrescrever nem editar os dados brutos.
  BLOCKED: os JSONL locais das fases 6 e 7 estão no SSD não montado; os demais valores foram regenerados e divergências antigas de RQ1/RQ2 foram corrigidas.
- [x] Registrar o identificador/commit do código, o hash ou versão dos arquivos de dados e a data de execução do pipeline.
  Evidência: `artifacts/statistics/analysis_metadata.json`.
- [x] Verificar se houve uma única geração por task/model e se há seeds, amostras repetidas, temperaturas e demais configurações salvas.
  Evidência: `artifacts/manifest.md` e Methodology.
- [x] Se reexecutar gerações for viável, criar um plano de replicação com múltiplas gerações por task/condição, seeds explícitas e orçamento definido; executar somente se o acesso aos modelos, custos e versões forem autorizados.
  Evidência: `artifacts/repeated_generation_plan.md`; a execução não foi autorizada e exigiria 36.595 chamadas.
- [x] Se reexecutar gerações não for viável, manter os resultados existentes, declarar geração única/estocástica como ameaça à validade de conclusão e não alegar estabilidade que os dados não demonstram.
  Evidência: Threats to Validity / Conclusion Validity.

### B. Auditoria das RQs e dos testes já existentes

- [x] Preencher a matriz de decisão estatística acima usando as cinco RQs reais do artigo.
  Evidência: `artifacts/statistical_audit.md`.
- [x] Localizar cada p-value, teste, correção de múltiplas comparações, threshold e decisão inferencial já presente no manuscrito.
- [x] Para cada teste existente, confirmar: hipótese nula operacional, teste two-sided ou justificativa de unilateralidade, unidade de análise, independência entre units, pareamento, variável de desfecho, número de comparações e família de correção.
- [x] Verificar especificamente se Cochran's Q e McNemar da RQ1 usam exatamente as mesmas tasks em todas as condições comparadas e se tasks podem ser tratadas como independentes entre si.
  Evidência: validação automática de 13 modelos por 563 tasks alinhadas.
- [x] Manter, corrigir ou remover cada teste existente conforme a auditoria; registrar a decisão em `artifacts/statistical_audit.md`.
- [x] Inserir na metodologia a declaração de estudo RQ-driven/exploratório e de uso de testes two-sided quando aplicável.
- [x] Não adicionar H0/H1 confirmatórias retrospectivas. Se o artigo usar o termo “hypothesis”, substituir por “research question”, “comparison”, “estimand” ou “exploratory analysis”, conforme apropriado.

### C. Código estatístico reprodutível

- [x] Criar um único ponto de entrada reprodutível, por exemplo `scripts/run_statistical_analysis.py` ou equivalente, que leia apenas dados brutos/manifestados e escreva resultados em `artifacts/statistics/`.
- [x] Fixar ambiente e dependências em arquivo versionado (`requirements.txt`, `environment.yml`, `pyproject.toml` ou equivalente), incluindo versões de bibliotecas estatísticas.
  Evidência: SciPy em `pyproject.toml`/`uv.lock`.
- [x] Implementar validações automáticas: schema dos dados, identificadores únicos de task, alinhamento correto das condições pareadas, ausência de duplicatas inesperadas e contagens iguais às declaradas no artigo.
- [x] Implementar Cochran's Q e testes post-hoc de McNemar two-sided para comparações binárias de múltiplos modelos, apenas onde a matriz preenchida indicar que são válidos.
- [x] Aplicar Holm à família de comparações definida previamente no plano de análise; salvar p-values brutos, ajustados e a definição da família.
- [x] Implementar bootstrap pareado por task para diferenças de proporção e métricas de detecção. Usar número de reamostragens, seed e método de IC explicitamente registrados; usar pelo menos 10.000 reamostragens salvo limitação computacional documentada.
- [x] Para métricas multilabel, implementar reamostragem agrupada por task quando múltiplas linhas pertencem à mesma task; não reamostrar linhas como se fossem independentes.
- [x] Implementar teste de permutação pareado two-sided para comparações de métricas complexas quando a auditoria aprovar sua validade; registrar o número de permutações e seed.
- [x] Gerar automaticamente tabelas em CSV/LaTeX com estimativa, IC 95%, tamanho de efeito, estatística de teste, p-value bruto, p-value ajustado, tamanho amostral e unidade de análise.
- [x] Gerar automaticamente figuras ou tabelas de distribuição apenas a partir do mesmo pipeline; não calcular valores manualmente para o manuscrito.
- [x] Executar o pipeline completo e salvar log, versão do ambiente e resultados em `artifacts/statistics/`.
- [x] Fazer uma revisão manual de sinais, denominadores, percentuais e arredondamento comparando as saídas do código com os dados de entrada.

### D. Atualização da metodologia e dos resultados

- [x] Criar/expandir uma subseção `Statistical Analysis` na metodologia.
- [x] Declarar explicitamente as variáveis/fatores experimentais: modelo avaliado, estratégia de especialização/training e disponibilidade de feedback de ferramentas estáticas.
- [x] Declarar os controles: tasks, taxonomia, estrutura de prompt, configuração de decoding, métricas e protocolo mantidos fixos nas comparações diretas.
- [x] Declarar os fatores não totalmente controlados: backend de inferência, precisão/quantização, versão de modelos hospedados, mudanças de API e eventuais diferenças de infraestrutura.
- [x] Explicar por que a task é a unidade de análise e por que as observações são pareadas quando a mesma task é avaliada por mais de uma condição.
- [x] Documentar pressupostos e limitações de cada família de testes usada, incluindo outcome binário, pareamento, independência entre tasks e condição para testes de permutação/bootstrap.
- [x] Declarar que todos os testes inferenciais são two-sided, exceto se existir uma justificativa metodológica prévia e verificável para teste unilateral.
- [ ] Atualizar cada tabela/trecho de resultado comparativo com tamanho de efeito e IC 95%, além de p-value ajustado quando houver teste.
  BLOCKED: concluído para todos os dados disponíveis; as linhas SLM de RQ4 permanecem descritivas até os JSONL do SSD serem restaurados.
- [x] Para RQs descritivas, substituir linguagem de “significância” por contagens, proporções, ICs e interpretação limitada ao benchmark.
- [x] Para RQs comparativas sem suporte para teste, explicar no resultado ou em Threats to Validity por que não há inferência formal.
- [x] Conferir que cada afirmação de superioridade, melhoria ou diferença no resumo, introdução, resultados, discussão e conclusão aponta para uma estimativa/IC ou é reescrita como observação descritiva.

### E. Amostra, objetos experimentais e reprodutibilidade

- [x] Inserir a justificativa de amostra: foram usadas todas as 563 tasks disponíveis nas porções HumanEval+ e MBPP+ do EvalPlus, sem selecionar subconjunto aleatório.
- [x] Explicitar que cobertura completa do benchmark não equivale à representatividade de todos os domínios, linguagens ou tarefas de programação.
- [x] Justificar HumanEval+/MBPP+ como objetos experimentais e discutir object-treatment confounds: benchmark, distribuição de dificuldade, linguagem, modelos, amostragem e ferramentas.
- [ ] Inserir uma tabela de ambiente com GPU, CPU, RAM, sistema operacional, Python, framework/backends, versões e configuração de quantização quando aplicável.
  DEFERRED: por decisão do autor, a subseção e a tabela de ambiente não fazem parte da versão atual do manuscrito e poderão ser adicionadas posteriormente.
- [x] Registrar identificadores completos dos modelos, endpoints/model versions hospedados, datas/período de consulta, parâmetros de decoding, seeds, versões/commit do EvalPlus e da ferramenta estática (por exemplo, Pyright).
- [x] Informar método e seed de quaisquer splits train/validation/test e da busca de hiperparâmetros (por exemplo, Optuna).
- [x] Atualizar o pacote de replicação com comando de ponta a ponta e instruções para reproduzir tabelas, figuras e análises estatísticas.

### F. Threats to Validity, ética e comunicação responsável

- [x] Adicionar a subseção `Conclusion Validity` antes de `Reliability`.
- [x] Em `Conclusion Validity`, discutir explicitamente: geração única por task/model, estocasticidade/temperature, ausência ou disponibilidade de seeds, número de comparações, desbalanceamento de classes, baixa frequência de Syntax, tamanho efetivo de subgrupos, incerteza em métricas e limites de apenas 563 tasks.
- [x] Atualizar as subseções de validade de construto, interna e externa para remover duplicação e deixar cada ameaça ligada a uma mitigação concreta.
- [x] Adicionar uma subseção curta `Ethical Considerations` ou `Risks and Responsible Use`.
- [x] Declarar que não foram usados participantes ou dados pessoais, que benchmarks públicos foram usados e que a execução de código ocorreu em ambiente isolado, se e somente se isso for factual e verificável.
- [x] Declarar que o detector não é prova de correção: false negatives podem induzir confiança indevida e a abordagem deve complementar compilação, testes, análise estática e revisão humana.
- [x] Rever figuras para confirmar que normalização, redução a rótulo primário, denominadores e diferenças de N estão legíveis; declarar que qualquer redução para visualização não substitui a avaliação multilabel.

### G. Revisão final orientada ao SIGSOFT

- [x] Preencher novamente o checklist SIGSOFT item a item, marcando `Yes`, `Partial`, `No` ou `N/A` e incluindo uma justificativa curta e uma localização no manuscrito para todos os itens não `Yes`.
  Evidência: `artifacts/sigsoft_checklist.md`.
- [x] Marcar como `N/A`, não como falha, itens exclusivos de participantes humanos (participantes, incentivos, random assignment de pessoas, attention check), com justificativa de estudo computacional pareado.
- [x] Confirmar que o artigo não promete hipóteses a priori nem causalidade que o desenho não suporta.
- [x] Confirmar que nenhuma tabela contém somente p-values: toda comparação inferencial deve trazer efeito e IC 95%.
- [x] Compilar o manuscrito e verificar referências cruzadas, números de tabelas/figuras, citações e consistência entre texto, tabelas e artefatos gerados.
  Evidência: PDF de 24 páginas e verificação visual das páginas alteradas.
- [x] Criar `artifacts/final_revision_report.md` listando: arquivos alterados, análises executadas, decisões estatísticas por RQ, itens bloqueados, limitações remanescentes e comandos de reprodução.
- [x] Marcar todas as tarefas concluídas e deixar sem marcação apenas itens realmente bloqueados, com justificativa verificável.

## Critérios de aceite

O plano estará concluído somente quando: (1) cada RQ tiver uma decisão estatística documentada; (2) todos os resultados estatísticos forem produzidos por código versionado e executado; (3) comparações inferenciais tiverem teste justificado, p-value ajustado quando aplicável, tamanho de efeito e IC 95%; (4) RQs descritivas não receberem testes artificiais; (5) a metodologia documentar desenho, controles, dados e ambiente; (6) `Conclusion Validity` e riscos/ética estiverem presentes; e (7) o checklist SIGSOFT revisado puder apontar para evidência concreta no manuscrito ou no pacote de replicação.
