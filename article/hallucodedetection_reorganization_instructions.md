# HalluCodeDetection Paper Reorganization Instructions

## Objective

Reorganize the LaTeX paper so that **Methodology**, **Experimental Setup**, and **Results** are clearly separated.

The current paper places methodological decisions, experimental configuration, and concrete execution protocols together inside `3_methodology.tex`. The goal is to split these responsibilities without unnecessarily rewriting the paper.

The reorganization must preserve the existing scientific content, wording, citations, tables, figures, labels, and results as much as possible. Prefer **moving existing text** over rewriting it. Only rewrite short transition sentences or headings when required to make the new structure coherent.

The repository structure is currently:

```text
article/
├── main.tex
└── sections/
    ├── 1_introduction.tex
    ├── 2_related_work.tex
    ├── 3_methodology.tex
    ├── 4_1_rq1.tex
    ├── 4_2_rq2.tex
    ├── 4_3_rq3.tex
    ├── 4_4_rq4.tex
    ├── 4_5_rq5.tex
    ├── 5_threats.tex
    └── 6_conclusion.tex
```

The target structure must be:

```text
article/
├── main.tex
└── sections/
    ├── 1_introduction.tex
    ├── 2_related_work.tex
    ├── 3_methodology.tex
    ├── 4_experimental_setup.tex
    ├── 5_1_rq1.tex
    ├── 5_2_rq2.tex
    ├── 5_3_rq3.tex
    ├── 5_4_rq4.tex
    ├── 5_5_rq5.tex
    ├── 6_threats.tex
    └── 7_conclusion.tex
```

Update `article/main.tex` accordingly. Preserve the current inclusion mechanism (`\input`, `\include`, or whichever command is already used) instead of changing it unnecessarily.

---

# 1. Conceptual distinction to enforce

The new organization must follow this distinction.

## Methodology

The Methodology section explains the **scientific design of the study**.

It should answer questions such as:

- What is being studied?
- What are the research questions?
- What is the unit of analysis?
- How are hallucinations operationalized as observable errors?
- How is the execution-based ground truth defined?
- Why is multilabel classification used?
- Which benchmark is used and why?
- Which evaluation metrics are used?
- How is the statistical analysis designed?
- Which variables/factors are compared and which factors are not controlled?

Methodology should explain the study independently of the exact model names, command-line options, quantization format, prompt implementation details, or hyperparameter values whenever possible.

## Experimental Setup

The Experimental Setup section explains **what was concretely executed**.

It should answer questions such as:

- Which models were evaluated?
- Which inference configurations were used?
- How was code generation executed?
- What prompts and temperatures were used?
- How was the error-detection experiment executed?
- How was self-evaluation operationalized?
- Which teacher models were used?
- How was the specialized detector trained?
- Which QLoRA and hyperparameter-search settings were used?
- Which static-analysis tools were used and how were their outputs mapped?
- How did the tool-assisted detection experiment work?
- How did the iterative code-generation loop work?

## Results

The RQ files should remain focused on **what happened after the experiments were executed**:

- observed percentages;
- Accuracy, Macro-F1, Recall, and other metrics;
- comparisons;
- statistical evidence;
- tables and plots;
- interpretation of observed results.

Do not move result values into Methodology or Experimental Setup unless a value is genuinely part of the experimental configuration.

---

# 2. Required new structure for `3_methodology.tex`

Refactor `3_methodology.tex` so that it contains only methodological content.

A recommended structure is:

```latex
\section{Methodology}
\label{sec:methodology}

\subsection{Study Design and Research Questions}

\subsection{Benchmark and Unit of Analysis}

\subsection{Ground Truth and Error Classification}

\subsection{Evaluation Metrics}

\subsection{Statistical Analysis}
```

The exact subsection names may be adjusted slightly if necessary, but preserve this conceptual organization.

## 2.1 Study Design and Research Questions

Start from the current `Study Overview` subsection.

Preserve the existing explanation that the study investigates whether SLMs can:

- generate correct code;
- identify errors in generated code;
- evaluate their own outputs;
- benefit from specialization;
- use static-analysis feedback;
- be compared with substantially larger language models under the same protocol.

Keep the existing methodological paragraph about experimental factors and controlled conditions:

- evaluated model;
- specialization/training strategy;
- availability of static-analysis feedback;
- fixed programming tasks;
- fixed taxonomy;
- fixed prompt structure for direct comparisons;
- fixed output schema;
- fixed metrics;
- fixed evaluation harness;
- confounding factors such as model family, pretraining, backend, numerical precision, quantization, and hosted-model revision.

This paragraph is methodological and should remain in `3_methodology.tex`.

Explicitly mention the five RQs at a high level, but **reuse the existing RQ wording from the paper** rather than inventing new research questions.

Do not place model-specific execution details here.

## 2.2 Benchmark and Unit of Analysis

Move the methodological part of the current `Programming Tasks and Code Generation` subsection here.

Keep:

- EvalPlus as the benchmark;
- HumanEval+ and MBPP+;
- 164 HumanEval tasks;
- 399 MBPP tasks;
- 563 tasks in total;
- the justification for using all tasks instead of sampling a subset;
- the fact that these are executable Python function-level problems;
- the limitations on representativeness;
- the programming task as the statistical/independent unit where already defined.

Do **not** keep here:

- model prompts;
- temperature;
- top-p/top-k;
- random seed;
- response normalization;
- function extraction details.

Those belong in Experimental Setup.

## 2.3 Ground Truth and Error Classification

Keep the current `Ground Truth and Error Classification` content in Methodology because it defines the operational reference used throughout the study.

Preserve the distinction between:

- `Syntax Error`;
- `Runtime Error`;
- `Functional Error`;
- `Correct`.

Use terminology consistently:

- A hallucination is the generative phenomenon.
- Syntax, Runtime, and Functional are **observable error outcomes**, not causes of hallucination.
- `Correct` is the non-error outcome.

Preserve the multilabel behavior:

- Syntax terminates evaluation and is exclusive;
- Runtime and Functional may co-occur across different tests;
- Correct is exclusive and is assigned only when no error is observed.

### Replace the current verbatim pseudocode

The current ground-truth procedure is represented as a figure containing a `verbatim` block. Replace it with a proper LaTeX algorithm.

Use:

```latex
\begin{algorithm}
...
\end{algorithm}
```

and an algorithmic environment such as `algpseudocode` if required.

The algorithm should express the same procedure already described in the paper:

1. receive generated code and EvalPlus tests;
2. compile the code;
3. if compilation fails, return Syntax Error;
4. otherwise execute the test cases;
5. record Runtime Error on exception or timeout;
6. record Functional Error when execution completes but the output differs from the expected result;
7. if no error label was added, assign Correct;
8. return the accumulated labels.

Do not change the semantics of the existing procedure.

If the required LaTeX packages are not already loaded, add only the minimal packages required to `main.tex`, preferably:

```latex
\usepackage{algorithm}
\usepackage{algpseudocode}
```

Do not introduce a new algorithm package if the document already uses an equivalent package.

Change the label from a figure-oriented label to an algorithm-oriented label if appropriate, for example:

```latex
\label{alg:ground_truth}
```

and update all references consistently.

## 2.4 Evaluation Metrics

Keep the current `Evaluation Metrics` subsection in Methodology.

Preserve:

- per-class precision;
- per-class recall;
- Macro-F1;
- Macro Recall;
- Exact-set Accuracy;
- the explanation that the task is multilabel.

Avoid unnecessary rewriting.

If a metric is used only for a specific RQ but is already correctly defined here, leave it here.

## 2.5 Statistical Analysis

Keep the current `Statistical Analysis` subsection in Methodology.

Preserve the existing RQ-specific analysis plan, including:

- exploratory rather than confirmatory framing;
- task as the independent unit;
- paired comparisons where applicable;
- Wilson confidence intervals where applicable;
- Cochran's Q;
- exact two-sided McNemar tests;
- Holm correction;
- task-level bootstrap;
- task-clustered bootstrap;
- paired permutation tests;
- the RQ-specific statistical procedures;
- the 10,000-resample setting;
- the existing analysis seed;
- the limitations associated with one generation per task/model.

This section explains how evidence is evaluated and therefore belongs in Methodology.

Do not move observed result values or interpretations into this subsection.

---

# 3. Required new file: `4_experimental_setup.tex`

Create:

```text
article/sections/4_experimental_setup.tex
```

with:

```latex
\section{Experimental Setup}
\label{sec:experimental_setup}
```

A recommended structure is:

```latex
\subsection{Models and Inference Configuration}

\subsection{Code Generation Experiment}

\subsection{Error Detection and Self-Evaluation Experiment}

\subsection{Specialized Detector Training}
\subsubsection{Knowledge Distillation}
\subsubsection{Fine-Tuning and Hyperparameter Search}

\subsection{Static Analysis Configuration}

\subsection{Tool-Assisted Experiments}
\subsubsection{Error Detection with Static Feedback}
\subsubsection{Iterative Code Generation with Static Feedback}
```

Again, small naming adjustments are acceptable, but do not collapse unrelated experiments into one large subsection.

---

# 4. Content to move into `4_experimental_setup.tex`

## 4.1 Models and Inference Configuration

Move the current `Evaluated Models` subsection here.

Preserve:

- the model table;
- SLM and larger-model grouping;
- parameter counts;
- citations;
- local quantization configuration;
- full-precision Gemma variants;
- hosted-model note;
- direct-response mode;
- disabled thinking/reasoning mode;
- temperature settings where they describe common inference configuration.

The judge-model selection and fine-tuning-model selection may remain here as a brief overview, but the detailed explanation should also appear in the relevant specialized-training subsection if needed.

Do not rewrite the model table unless required to fix compilation after moving it.

## 4.2 Code Generation Experiment

Move the experimental parts of the current `Programming Tasks and Code Generation` subsection here.

Keep:

- HumanEval receives a function signature and requires body completion;
- MBPP requires a full function implementation;
- system-prompt behavior;
- benchmark-specific user prompt behavior;
- one generated solution per model/task;
- temperature 1.0;
- backend-default top-p/top-k;
- no fixed random seed;
- response normalization;
- code-fence removal;
- entry-point detection;
- HumanEval reconstruction behavior;
- MBPP function reconstruction behavior;
- handling of prose mixed with code;
- extraction failures that may later appear as Syntax Error.

The benchmark description and justification itself should stay in Methodology.

## 4.3 Error Detection and Self-Evaluation Experiment

Move the current `Error Detection Experiment` subsection here.

Keep:

- model input consists only of the programming task and generated code;
- no reference labels;
- no test execution results;
- classification prompt;
- JSON output;
- natural-language explanation requirement;
- temperature 0.0;
- same protocol for SLMs and larger models;
- identical task, code, prompt, class definitions, output format, metrics, and temperature for direct comparisons.

The error-category definition table may remain in Methodology near the ground-truth definition if that produces a cleaner conceptual flow. Do not duplicate the table.

The Experimental Setup should reference the methodological definition of the categories instead of redefining them.

Add the self-evaluation protocol here if it is currently described only implicitly in the paper. Do not invent a new method. Reuse the existing procedure used to compute self-error detection in RQ2.

## 4.4 Specialized Detector Training

Move the current `Specialized Model Training and Knowledge Distillation` subsection here.

### Knowledge Distillation

Preserve:

- three teacher/judge models;
- task description;
- generated code;
- generator identity if currently used;
- ground-truth labels supplied to the teacher;
- teacher generates an explanation but does not assign or modify the label;
- each example is processed by all three teachers;
- no voting/winner selection;
- one rationale per teacher;
- retries for failed teacher responses;
- execution-derived labels remain authoritative;
- rationales are auxiliary supervision only.

The conceptual motivation for rationale-based supervision may be kept here because it directly explains the training setup, but avoid expanding it beyond the current text.

### Fine-Tuning and Hyperparameter Search

Preserve:

- Gemma base models;
- conversational training format;
- QLoRA;
- NF4 training quantization;
- LoRA citation;
- hyperparameter table;
- train/validation/test split;
- task-level split;
- 15\% validation and 20\% test;
- original-distribution training configurations;
- Accuracy-oriented configuration;
- Macro-F1-oriented configuration;
- label-weighted loss;
- balanced configurations;
- downsampling ratios;
- removal of Syntax from balanced training;
- Optuna;
- TPE;
- hyperparameter search space;
- model-selection rule.

Do not move any observed performance values from RQ3 into this section.

## 4.5 Static Analysis Configuration

Move the current `Static Analysis Tools` subsection here.

Preserve the scientific behavior of the tools, but improve how the configuration is documented.

### Avoid `\texttt{...}`-heavy prose

Do not use `\texttt` for ordinary tool names or prose.

For example, avoid writing a paragraph dominated by forms such as:

```latex
\texttt{reportUndefinedVariable}
\texttt{reportUnboundVariable}
...
```

Similarly, there is no need to reproduce the full Pyright command line in monospace in the main prose.

Instead, describe the configuration semantically.

For example, state that Pyright is executed:

- using Python 3.13 semantics;
- under its default type-checking mode;
- considering only error-severity diagnostics;
- using structured JSON output for parsing.

### Replace the raw Pyright-rule enumeration with semantic groups

The current text lists every accepted Pyright diagnostic rule individually. Replace this with a concise table that explains **what kinds of problems are considered Runtime signals**.

A suitable structure is:

```latex
\begin{table}[htbp]
\centering
\caption{Pyright diagnostic groups mapped to Runtime Errors.}
\label{tab:pyright_runtime_groups}
\begin{tabular}{p{0.28\columnwidth}p{0.66\columnwidth}}
\toprule
\textbf{Diagnostic group} & \textbf{Examples of detected problems} \\
\midrule
Name and binding errors & Undefined, unbound, or possibly unbound variables. \\
Call and argument errors & Invalid function calls, incompatible arguments, or invalid call signatures. \\
Type consistency errors & Incompatible assignments, return values, general type mismatches, or invalid type arguments. \\
Attribute, indexing, and operator errors & Invalid member access, unsupported indexing, or invalid operand combinations. \\
Optional-value misuse & Calling, indexing, iterating over, or accessing members of values that may be \emph{None}. \\
\bottomrule
\end{tabular}
\end{table}
```

This is an example of the desired presentation. Adapt the wording to remain faithful to the exact current mapping.

Important:

- Do **not** change which diagnostics are actually accepted by the implementation.
- This is a documentation/presentation change only.
- Parsing-related diagnostics should still map to Syntax.
- Style and non-target diagnostics should still be ignored.
- Pyright should still not be treated as a Functional Error detector.
- Tool failures should still produce no derived prediction.

If exact Pyright rule names are needed for reproducibility, keep them outside the main narrative, for example in supplementary material, an appendix, or the public implementation. Do not keep the exhaustive raw-name list in the main paragraph.

### Python compilation

Describe Python compilation in normal prose.

State that it:

- parses the candidate source;
- emits a Syntax signal if parsing fails;
- does not execute the program;
- is deterministic;
- is not used to infer Runtime or Functional Errors.

Avoid unnecessary monospace formatting.

## 4.6 Tool-Assisted Experiments

Move the current `Interactive Cycles with Tools` subsection here and split it explicitly into the two experiments.

### Error Detection with Static Feedback

This corresponds to the current `For Code Verification` subsection.

Preserve:

- held-out generated solution;
- same model and same sample with and without static feedback;
- same task;
- same code;
- same base prompt;
- same temperature;
- same metrics;
- static feedback as supporting evidence, not ground truth;
- compile status;
- runtime static-analysis status;
- diagnostic information;
- explicit warning that absence of diagnostics does not prove correctness;
- multilabel output;
- reference labels used only afterward for evaluation.

Keep the existing verification-cycle figure unless a small caption or placement adjustment is required.

### Iterative Code Generation with Static Feedback

This corresponds to the current `For Code Generation` subsection.

Preserve:

- initial solution;
- compilation first;
- runtime-oriented static analysis only if compilation succeeds;
- feedback returned to the model;
- self-review;
- submit or revise decision;
- no full history carried across iterations;
- revision replaces the current candidate;
- code-only fallback behavior;
- maximum of five rounds;
- temperature 0;
- EvalPlus applied only to the final candidate;
- test results never shown to the model;
- improved, unchanged, and regressed outcomes retained.

Keep the existing generation-cycle figure.

It is acceptable to place the verification experiment before the iterative generation experiment if that order better matches RQ4, but do not change their semantics.

---

# 5. Rename the Results and Later Section Files

Rename:

```text
4_1_rq1.tex -> 5_1_rq1.tex
4_2_rq2.tex -> 5_2_rq2.tex
4_3_rq3.tex -> 5_3_rq3.tex
4_4_rq4.tex -> 5_4_rq4.tex
4_5_rq5.tex -> 5_5_rq5.tex
5_threats.tex -> 6_threats.tex
6_conclusion.tex -> 7_conclusion.tex
```

Do not rewrite the RQ content merely because the files are renamed.

If the paper currently starts the results section inside `4_1_rq1.tex`, update the section heading so that it naturally becomes Section 5 after inserting Experimental Setup.

Preserve the existing RQ subsection labels whenever possible so that cross-references do not break.

---

# 6. Update `main.tex`

Inspect `article/main.tex` and update the section inclusion order to:

```text
1_introduction
2_related_work
3_methodology
4_experimental_setup
5_1_rq1
5_2_rq2
5_3_rq3
5_4_rq4
5_5_rq5
6_threats
7_conclusion
```

Do not change the bibliography, document class, formatting, or unrelated packages.

Only add algorithm-related packages if they are required by the new ground-truth algorithm and an equivalent package is not already loaded.

After changing file names, search the repository for references to the old names and update them.

---

# 7. Preserve text and avoid unnecessary rewriting

This is a reorganization task, not a full rewrite.

Follow these rules strictly:

1. **Move first, rewrite second.**
   If an existing paragraph already belongs in the new location, move it unchanged.

2. Rewrite only when:
   - a transition refers to the old section structure;
   - a subsection title no longer matches its content;
   - duplicated definitions appear after moving content;
   - the Pyright configuration needs to be converted from raw rule names into semantic groups;
   - the ground-truth pseudocode must be converted into an Algorithm environment.

3. Do not simplify away methodological details that affect reproducibility.

4. Do not introduce new experiments, models, metrics, or claims.

5. Do not alter result values.

6. Do not alter citations unless required to move them with their associated text.

7. Do not replace precise scientific claims with stronger claims.

8. Preserve the paper's distinction between:
   - hallucination as the generative phenomenon;
   - observable Syntax, Runtime, and Functional Errors as outcomes.

9. Avoid em dashes in newly written prose.

10. Avoid `\texttt` in prose. Use normal technical wording instead.

---

# 8. Specific mapping from the current `3_methodology.tex`

The current file contains the following subsections:

```text
Study Overview
Evaluated Models
Programming Tasks and Code Generation
Ground Truth and Error Classification
Error Detection Experiment
Specialized Model Training and Knowledge Distillation
    Knowledge Distillation
    Specialized Model Training
Evaluation Metrics
Statistical Analysis
Static Analysis Tools
Interactive Cycles with Tools
    For Code Generation
    For Code Verification
```

Move them as follows.

| Current content | Target file | Target subsection |
|---|---|---|
| Study Overview | `3_methodology.tex` | Study Design and Research Questions |
| Evaluated Models | `4_experimental_setup.tex` | Models and Inference Configuration |
| EvalPlus description and benchmark justification from Programming Tasks and Code Generation | `3_methodology.tex` | Benchmark and Unit of Analysis |
| Prompting, generation configuration, sampling, extraction, and normalization from Programming Tasks and Code Generation | `4_experimental_setup.tex` | Code Generation Experiment |
| Ground Truth and Error Classification | `3_methodology.tex` | Ground Truth and Error Classification |
| Ground-truth verbatim pseudocode | `3_methodology.tex` | Replace with Algorithm environment |
| Error-category conceptual definitions | `3_methodology.tex` | Ground Truth and Error Classification |
| Error Detection Experiment | `4_experimental_setup.tex` | Error Detection and Self-Evaluation Experiment |
| Knowledge Distillation | `4_experimental_setup.tex` | Specialized Detector Training / Knowledge Distillation |
| Specialized Model Training | `4_experimental_setup.tex` | Specialized Detector Training / Fine-Tuning and Hyperparameter Search |
| Evaluation Metrics | `3_methodology.tex` | Evaluation Metrics |
| Statistical Analysis | `3_methodology.tex` | Statistical Analysis |
| Static Analysis Tools | `4_experimental_setup.tex` | Static Analysis Configuration |
| Interactive Cycles / For Code Verification | `4_experimental_setup.tex` | Tool-Assisted Experiments / Error Detection with Static Feedback |
| Interactive Cycles / For Code Generation | `4_experimental_setup.tex` | Tool-Assisted Experiments / Iterative Code Generation with Static Feedback |

---

# 9. Avoid duplication after splitting sections

After moving content, carefully inspect the new files for repeated explanations.

In particular:

- define the benchmark once in Methodology, then refer to it from Experimental Setup;
- define the error taxonomy once in Methodology, then refer to the categories from experiment descriptions;
- define metrics once in Methodology;
- do not repeat the statistical-analysis plan in the Results;
- do not repeat model configuration in each RQ;
- keep model-specific and tool-specific execution details in Experimental Setup.

Short reminders are acceptable, but avoid full repeated paragraphs.

---

# 10. Cross-reference cleanup

After the reorganization:

- update section references if they point to obsolete labels;
- update the ground-truth pseudocode reference if it changes from Figure to Algorithm;
- preserve existing table and figure labels whenever possible;
- verify that no `\ref{...}` points to a deleted label;
- verify that no section refers to the old section number explicitly in prose;
- avoid hard-coded section numbers where a `\ref` should be used.

---

# 11. Compilation and validation

After completing the changes:

1. Compile the full paper from `article/main.tex`.
2. Fix only errors introduced by the reorganization.
3. Check for:
   - missing files;
   - broken references;
   - duplicated labels;
   - undefined citations;
   - misplaced floats;
   - missing Algorithm packages;
   - obsolete references to the old filenames;
   - duplicated section headings.
4. Confirm that all original result tables and figures still appear.
5. Confirm that the scientific content and numerical results are unchanged.
6. Confirm that the paper now has the high-level order:

```text
Introduction
Related Work
Methodology
Experimental Setup
Results
Threats to Validity
Conclusion
```

---

# 12. Final editing principle

The purpose of this task is structural clarity.

The resulting paper should make the distinction immediately visible:

- **Methodology:** what the study means, measures, and compares.
- **Experimental Setup:** what was concretely executed.
- **Results:** what was observed.
- **Threats to Validity:** what limits those observations.
- **Conclusion:** what can be concluded from them.

Do not use this task as an opportunity to rewrite unrelated parts of the article.
