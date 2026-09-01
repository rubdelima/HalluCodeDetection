# EvalPlus Integration Guide

This document describes how to integrate **EvalPlus** into the HalluCodeDetection experimental pipeline.

The goal is **not** to replace the project's current LLM generation pipeline. EvalPlus will be used primarily for:

1. collecting executable programming tasks from **HumanEval+** and **MBPP+**;
2. obtaining stronger executable test suites;
3. evaluating generated Python solutions against the original and extended tests;
4. deriving project-specific labels:
   - `syntax_error`
   - `runtime_error`
   - `functional_error`
   - `correct`
5. preserving enough metadata for reproducible experiments, generalization studies, and later qualitative analysis.

> **Dependency management:** this project uses **uv**. Use `uv add` instead of `pip install`.

---

## 1. Why EvalPlus

EvalPlus extends two widely used executable code-generation benchmarks:

- **HumanEval+**
  - based on HumanEval;
  - includes substantially more tests than the original benchmark;
  - provides the task prompt, function entry point, canonical solution, original test inputs, and additional EvalPlus test inputs.

- **MBPP+**
  - based on MBPP;
  - provides a stronger test suite than the original MBPP benchmark;
  - follows the same general EvalPlus evaluation workflow.

For this project, the most important property is that generated code can be evaluated automatically using executable test cases.

The desired classification remains:

```text
generated solution
       |
       v
can the code be parsed/compiled?
       |
   no  +--------------------------> syntax_error
       |
      yes
       |
       v
does execution raise a runtime failure?
       |
   yes +--------------------------> runtime_error
       |
      no
       |
       v
does the implementation satisfy the tests?
       |
   no  +--------------------------> functional_error
       |
      yes
       |
       v
     correct
```

EvalPlus provides the benchmark and execution oracle.  
The four-class error taxonomy above is **project-specific logic** and is not the default EvalPlus classification.

---

# 2. Environment Setup with `uv`

## 2.1 Initialize the project

If the repository does not already use `uv`:

```bash
uv init
```

If `pyproject.toml` already exists, do not run `uv init` again.

---

## 2.2 Add EvalPlus

For the latest stable PyPI release:

```bash
uv add evalplus
```

If the project needs the latest EvalPlus version directly from GitHub:

```bash
uv add "evalplus @ git+https://github.com/evalplus/evalplus"
```

Prefer the stable release for reproducible experiments unless a feature or bug fix requires the Git version.

After dependency resolution, commit both:

```text
pyproject.toml
uv.lock
```

to the repository.

This ensures that the exact dependency environment can be reproduced.

---

## 2.3 Run EvalPlus commands through `uv`

Instead of invoking the executable directly, prefer:

```bash
uv run evalplus.evaluate --help
```

Other useful commands include:

```bash
uv run evalplus.codegen --help
uv run evalplus.sanitize --help
uv run evalplus.syncheck --help
```

The project will normally **not** use `evalplus.codegen`, because generation is already handled by the project's own LLM/Ollama pipeline.

---

# 3. Recommended Project Structure

A suggested structure is:

```text
project/
├── pyproject.toml
├── uv.lock
│
├── data/
│   ├── evalplus/
│   │   ├── tasks/
│   │   │   ├── humaneval_plus.jsonl
│   │   │   └── mbpp_plus.jsonl
│   │   │
│   │   ├── generations/
│   │   │   ├── humaneval/
│   │   │   └── mbpp/
│   │   │
│   │   ├── evalplus_samples/
│   │   │   ├── humaneval/
│   │   │   └── mbpp/
│   │   │
│   │   └── evaluated/
│   │       ├── humaneval/
│   │       └── mbpp/
│
├── src/
│   ├── datasets/
│   │   └── evalplus_loader.py
│   │
│   ├── generation/
│   │   └── generate_evalplus.py
│   │
│   └── evaluation/
│       ├── export_evalplus_samples.py
│       ├── classify_failures.py
│       └── run_evalplus.py
│
└── docs/
    └── evalplus.md
```

The exact names may be changed to match the repository, but the conceptual separation is important:

1. **task collection**
2. **generation**
3. **official EvalPlus evaluation**
4. **project-specific failure classification**

---

# 4. Loading HumanEval+ and MBPP+

EvalPlus exposes both datasets through Python.

```python
from evalplus.data import get_human_eval_plus, get_mbpp_plus
```

Load HumanEval+:

```python
human_eval_plus = get_human_eval_plus()
```

Load MBPP+:

```python
mbpp_plus = get_mbpp_plus()
```

Both functions return dictionaries keyed by `task_id`.

Example:

```python
from evalplus.data import get_human_eval_plus

dataset = get_human_eval_plus()

print(len(dataset))

task_id = next(iter(dataset))
problem = dataset[task_id]

print(task_id)
print(problem.keys())
```

---

# 5. Important EvalPlus Task Fields

The main fields exposed by EvalPlus include:

| Field | Meaning | Use in this project |
|---|---|---|
| `task_id` | Unique task identifier | Store permanently |
| `entry_point` | Function that must be implemented | Evaluation metadata |
| `prompt` | Function signature and task description | Input to the generator |
| `canonical_solution` | Reference implementation | Oracle/evaluation only |
| `base_input` | Original benchmark test inputs | Evaluation |
| `plus_input` | Additional EvalPlus inputs | Stronger evaluation |

Typical inspection:

```python
problem = human_eval_plus["HumanEval/0"]

print(problem["task_id"])
print(problem["entry_point"])
print(problem["prompt"])
print(len(problem["base_input"]))
print(len(problem["plus_input"]))
```

## Critical leakage rule

Never expose the following to the code generator or hallucination detector:

```text
canonical_solution
base_input
plus_input
expected outputs
execution results
```

unless an experiment explicitly studies additional execution context.

For the normal detection task, the detector should receive only:

```text
task specification
+
generated candidate code
```

The canonical solution and tests belong exclusively to the evaluation oracle.

---

# 6. Collecting the EvalPlus Tasks

It is useful to export an immutable local representation of the task metadata used in the experiment.

Example:

```python
import json
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus


def export_tasks(dataset_name: str, output_path: str) -> None:
    if dataset_name == "humaneval":
        dataset = get_human_eval_plus()
    elif dataset_name == "mbpp":
        dataset = get_mbpp_plus()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for task_id, problem in dataset.items():
            record = {
                "benchmark": dataset_name,
                "task_id": task_id,
                "entry_point": problem["entry_point"],
                "prompt": problem["prompt"],

                # Keep only in the oracle-side dataset.
                "canonical_solution": problem.get("canonical_solution"),

                # Useful for reproducibility and execution.
                "base_input": problem.get("base_input"),
                "plus_input": problem.get("plus_input"),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

Example execution:

```bash
uv run python src/datasets/evalplus_loader.py
```

## Recommended principle

Maintain two conceptual views:

### Oracle-side task record

May contain:

```text
canonical_solution
base_input
plus_input
```

### Generation-side task record

Should contain only:

```text
benchmark
task_id
entry_point
prompt
```

This separation minimizes accidental leakage.

---

# 7. Generating Code

The project should continue using its own code-generation pipeline.

Do **not** require EvalPlus to generate the code.

The EvalPlus documentation explicitly supports custom generation workflows.

Conceptually:

```python
solution = generate_with_model(problem["prompt"])
```

The project should record more metadata than EvalPlus strictly requires.

Recommended generation record:

```json
{
  "benchmark": "humaneval",
  "task_id": "HumanEval/0",
  "entry_point": "has_close_elements",
  "generator": "qwen2.5-coder:7b",
  "generator_family": "qwen",
  "quantization": "Q4_K_M",
  "temperature": 0.7,
  "seed": 42,
  "sample_id": 0,
  "prompt": "...",
  "generated_code_raw": "...",
  "generated_code": "..."
}
```

---

# 8. Prompting the Code Generator

For HumanEval+, the EvalPlus `prompt` already normally includes the function signature and docstring.

A minimal generation prompt can therefore be:

```python
def build_generation_prompt(problem: dict) -> str:
    return f"""Complete the following Python function.

Return only valid Python code.
Do not include explanations.
Do not include Markdown code fences.
Do not create additional tests.

{problem["prompt"]}
"""
```

The same high-level policy should be used consistently across all generator models.

Record the exact prompt template used in the experiment.

---

# 9. Raw Output vs Sanitized Output

Always preserve the original model response.

Recommended fields:

```text
generated_code_raw
generated_code
sanitized
```

Example:

```json
{
  "generated_code_raw": "```python\ndef foo(...)\n```",
  "generated_code": "def foo(...)\n",
  "sanitized": true
}
```

Why preserve both?

Because malformed formatting may itself be relevant to hallucination/error analysis.

---

# 10. EvalPlus Sanitization

EvalPlus provides a sanitizer for generated output.

For JSONL samples:

```bash
uv run evalplus.sanitize --samples samples.jsonl
```

This produces a sanitized file such as:

```text
samples-sanitized.jsonl
```

EvalPlus also provides syntax checking:

```bash
uv run evalplus.syncheck \
    --samples samples.jsonl \
    --dataset humaneval
```

For MBPP:

```bash
uv run evalplus.syncheck \
    --samples samples.jsonl \
    --dataset mbpp
```

## Important methodological decision

For HalluCodeDetection, do not silently replace every raw generation with sanitized code before classification.

Otherwise, formatting or syntax failures introduced by the model may disappear.

Recommended policy:

1. retain the raw model output;
2. perform project-specific syntax classification on the code representation intended by the experiment;
3. optionally produce a sanitized copy for an additional controlled experiment;
4. record whether sanitization changed the sample.

This allows later analysis such as:

```text
raw syntax error
vs.
sanitizer-recoverable formatting issue
```

---

# 11. Multiple Generations per Task

The current expanded experiment should use multiple samples whenever computationally feasible.

For example:

```text
3 generations per model per task
```

or:

```text
5 generations per model per task
```

Example:

```python
for task_id, problem in dataset.items():
    for model in models:
        for sample_id in range(NUM_SAMPLES):
            generated = generate(
                model=model,
                prompt=problem["prompt"],
            )

            save_generation(
                task_id=task_id,
                model=model,
                sample_id=sample_id,
                generated_code=generated,
            )
```

This is useful because a single model can produce different failure types for the same task.

Example:

```text
HumanEval/42 + Qwen
├── sample 0 -> correct
├── sample 1 -> functional_error
└── sample 2 -> runtime_error
```

---

# 12. EvalPlus Sample Format

The official EvalPlus evaluator expects JSONL records containing a task identifier plus either `solution` or `completion`.

Recommended format for this project:

```json
{
  "task_id": "HumanEval/0",
  "solution": "def has_close_elements(...):\n    ..."
}
```

EvalPlus accepts:

```text
task_id
solution
```

or:

```text
task_id
completion
```

Only one is required.

If both are present, `solution` is used.

---

# 13. Exporting Project Generations to EvalPlus JSONL

The project dataset may contain many extra fields. Export only what EvalPlus needs into a dedicated evaluation file.

Example:

```python
import json


def export_evalplus_jsonl(records, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            evalplus_record = {
                "task_id": record["task_id"],
                "solution": record["generated_code"],
            }

            f.write(
                json.dumps(evalplus_record, ensure_ascii=False)
                + "\n"
            )
```

Do not discard the richer project record.

Treat the EvalPlus JSONL as an evaluation adapter.

---

# 14. Running Official EvalPlus Evaluation

HumanEval+:

```bash
uv run evalplus.evaluate \
    --dataset humaneval \
    --samples samples.jsonl
```

MBPP+:

```bash
uv run evalplus.evaluate \
    --dataset mbpp \
    --samples samples.jsonl
```

EvalPlus reports results for:

```text
Base
Base + Extra
```

Interpretation:

- **Base**
  - original HumanEval or MBPP tests;

- **Base + Extra**
  - original tests plus the additional EvalPlus tests.

This distinction should be preserved in the project dataset.

---

# 15. Re-running Evaluations

EvalPlus caches evaluation results.

A result file is typically created with a name similar to:

```text
samples_eval_results.jsonl
```

If evaluation needs to be repeated after changing the samples or execution settings, either:

1. remove the cached result; or
2. use the EvalPlus option intended to force re-execution.

Before large experiments, verify the exact CLI options available in the installed version:

```bash
uv run evalplus.evaluate --help
```

---

# 16. Full Test Details

EvalPlus can run with test details enabled.

Example:

```bash
uv run evalplus.evaluate \
    --dataset humaneval \
    --samples samples.jsonl \
    --test-details
```

This is useful for HalluCodeDetection because the project needs more information than only `pass@k`.

However, it is more expensive because EvalPlus may execute substantially more tests instead of stopping immediately after the first failure.

Recommended workflow:

```text
development / debugging
    -> --test-details

large generation sanity checks
    -> normal evaluation first

final labeling / detailed analysis
    -> --test-details where needed
```

---

# 17. HumanEval+ Mini

EvalPlus provides a reduced HumanEval+ test set through the `--mini` option.

Example:

```bash
uv run evalplus.evaluate \
    --dataset humaneval \
    --samples samples.jsonl \
    --mini
```

Use this only when faster iteration is necessary.

For the final dataset used in the paper, prefer the full HumanEval+ evaluation unless the experiment explicitly studies the Mini variant.

---

# 18. Execution Safety

Generated code must be considered untrusted.

EvalPlus strongly recommends sandboxed execution.

For the final experimental pipeline, prefer Docker or another isolated execution environment.

The official EvalPlus Docker image can execute samples inside a container.

A generic pattern is:

```bash
docker run --rm --pull=always \
    -v "$(pwd)/evalplus_results:/app" \
    ganler/evalplus:latest \
    evalplus.evaluate \
    --dataset humaneval \
    --samples /app/samples.jsonl
```

Adapt paths to the project layout.

## Why sandboxing matters

Generated programs may:

- enter infinite loops;
- allocate excessive memory;
- spawn processes;
- access files;
- behave unexpectedly.

Do not rely only on Python `try/except` for isolation.

---

# 19. EvalPlus Execution Limits

EvalPlus includes execution controls such as time and memory limits.

Its execution documentation defines timeouts relative to:

```text
minimum time limit
and
ground-truth execution time × multiplier
```

Generated solutions that time out or run out of memory are treated as failed by EvalPlus.

The project should record such failures explicitly rather than silently treating all of them as generic functional failures.

Recommended internal fields:

```text
timed_out
out_of_memory
exception_type
execution_status
```

---

# 20. Project-Specific Four-Class Labeling

EvalPlus itself primarily answers whether a solution passes the benchmark.

HalluCodeDetection needs a richer label.

Use the following operational definition.

---

## 20.1 `syntax_error`

Definition:

> The generated Python program cannot be parsed or compiled because of invalid Python syntax.

Initial check:

```python
def has_syntax_error(code: str) -> bool:
    try:
        compile(code, "<generated>", "exec")
        return False
    except SyntaxError:
        return True
```

Examples:

```text
SyntaxError
IndentationError
```

Note that `IndentationError` is a subclass of `SyntaxError`.

Recommended record:

```json
{
  "label": "syntax_error",
  "syntax_valid": false,
  "exception_type": "SyntaxError"
}
```

---

## 20.2 `runtime_error`

Definition:

> The code is syntactically valid, but evaluating the candidate causes an execution failure rather than merely producing an incorrect result.

Examples may include:

```text
NameError
TypeError
IndexError
KeyError
ZeroDivisionError
RecursionError
Timeout
Out-of-memory failure
```

The exact mapping must be implemented carefully using the execution information returned by the evaluation pipeline.

Recommended record:

```json
{
  "label": "runtime_error",
  "syntax_valid": true,
  "execution_completed": false,
  "exception_type": "TypeError"
}
```

---

## 20.3 `functional_error`

Definition:

> The generated code executes without an execution failure but does not satisfy the expected functional behavior.

Typical case:

```text
expected output != actual output
```

or an assertion/property fails even though the function completed normally.

Recommended record:

```json
{
  "label": "functional_error",
  "syntax_valid": true,
  "execution_completed": true,
  "plus_passed": false
}
```

---

## 20.4 `correct`

Definition:

> The solution successfully satisfies the full EvalPlus test suite used by the experiment.

Recommended record:

```json
{
  "label": "correct",
  "syntax_valid": true,
  "execution_completed": true,
  "base_passed": true,
  "plus_passed": true
}
```

For the main paper dataset, `correct` should normally mean passing **Base + Extra**, not only the original benchmark tests.

---

# 21. Classification Precedence

The project needs deterministic precedence rules.

Recommended order:

```text
1. syntax_error
2. runtime_error
3. functional_error
4. correct
```

Pseudo-code:

```python
if not syntax_valid:
    label = "syntax_error"

elif execution_failure:
    label = "runtime_error"

elif not plus_tests_pass:
    label = "functional_error"

else:
    label = "correct"
```

Do not derive these categories only from EvalPlus aggregate `pass@k`.

The classifier needs per-sample execution information.

---

# 22. Base vs Plus Results

One of the most valuable EvalPlus properties is the distinction between original tests and stronger tests.

For every generation, store:

```text
base_passed
plus_passed
```

Possible cases:

| Base | Plus | Interpretation |
|---|---|---|
| pass | pass | robustly correct under EvalPlus |
| pass | fail | passes weak/original tests but fails stronger tests |
| fail | fail | already fails original benchmark |
| fail | pass | should normally be investigated as inconsistent |

The most interesting subgroup is:

```text
base_passed = true
plus_passed = false
```

These samples look correct under the original benchmark but are exposed as incorrect by the additional tests.

This group is especially valuable for evaluating hallucination detectors.

---

# 23. Recommended Additional Field: `failure_stage`

Store where the failure was detected.

Example values:

```text
syntax
base
plus
timeout
memory
unknown
```

Example:

```json
{
  "base_passed": true,
  "plus_passed": false,
  "failure_stage": "plus",
  "label": "functional_error"
}
```

This allows later analyses such as:

```text
How many functional failures were invisible to the original tests?
```

---

# 24. Recommended Master Record Schema

A rich internal record should look approximately like this:

```json
{
  "benchmark": "humaneval",
  "task_id": "HumanEval/42",
  "entry_point": "example_function",

  "generator": "qwen2.5-coder:7b",
  "generator_family": "qwen",
  "quantization": "Q4_K_M",
  "temperature": 0.7,
  "seed": 42,
  "sample_id": 2,

  "task_prompt": "...",

  "generated_code_raw": "...",
  "generated_code": "...",
  "sanitized": false,

  "syntax_valid": true,

  "base_passed": true,
  "plus_passed": false,

  "execution_completed": true,
  "timed_out": false,
  "out_of_memory": false,

  "exception_type": null,
  "exception_message": null,

  "failure_stage": "plus",
  "label": "functional_error",

  "evalplus_version": "...",
  "dataset_version": "...",
  "evaluation_timestamp": "..."
}
```

---

# 25. Information to Preserve for Reproducibility

At minimum, record:

### Benchmark metadata

```text
benchmark
task_id
entry_point
dataset version
```

### Generator metadata

```text
model
model version/tag
quantization
temperature
top_p
seed, if supported
sample_id
generation prompt template version
```

### Generated output

```text
raw output
normalized code
whether sanitization was applied
```

### Evaluation metadata

```text
EvalPlus version
base result
plus result
exception type
timeout
memory failure
final project label
```

---

# 26. Dataset Versioning

EvalPlus datasets are maintained over time.

Because test cases and contracts can be corrected between releases, record:

```text
EvalPlus package version
HumanEval+ dataset version
MBPP+ dataset version
```

At minimum:

```bash
uv run python -c "import evalplus; print(evalplus.__version__)"
```

If the exact dataset version is exposed by the installed API or configuration, save it with the generated experiment metadata as well.

For publication, the replication package should specify the exact versions used.

---

# 27. Avoiding Data Leakage in Train/Test Splits

Never split only by generated sample.

The same programming task must not appear in both training and test partitions.

Wrong:

```text
HumanEval/20 + Gemma sample 0 -> train
HumanEval/20 + Qwen sample 1  -> test
```

Even though the generated code differs, the task specification is identical.

Correct:

```text
HumanEval/20
    all models
    all samples
        -> same partition
```

Use `task_id` as the split group.

Example with scikit-learn-style logic:

```python
task_ids = sorted(df["task_id"].unique())

# split task_ids first
# then assign every generated sample according to its task_id
```

This should be true for both HumanEval+ and MBPP+.

---

# 28. Benchmark-Aware Splits

The project should retain the benchmark name explicitly:

```text
humaneval
mbpp
```

This enables several experimental settings.

### In-domain

Train/test on unseen tasks from the same benchmarks.

### Cross-benchmark

Example:

```text
Train:
MBPP+

Test:
HumanEval+
```

or the reverse.

### Mixed training with held-out tasks

Example:

```text
Train:
MBPP+ tasks A
HumanEval+ tasks A

Test:
MBPP+ unseen tasks
HumanEval+ unseen tasks
```

These experiments test different forms of generalization.

---

# 29. Cross-Generator Evaluation

Because the generator identity is stored for each sample, the project can later perform leave-one-generator-out experiments.

Example:

```text
Train:
Gemma
Qwen
DeepSeek
StarCoder

Test:
GPT-OSS
```

This answers whether the detector learns general error patterns rather than generator-specific artifacts.

Keep generator metadata from the beginning so that this experiment remains possible.

---

# 30. Separating Oracle Data from Detector Inputs

For each labeled sample, maintain a complete oracle-side record but build detector inputs from a restricted projection.

Example detector input:

```json
{
  "task": "...",
  "generated_code": "..."
}
```

Target:

```json
{
  "label": "functional_error"
}
```

Do not pass:

```text
canonical solution
tests
test inputs
exception logs
base/plus status
expected output
```

to the detector unless that is a separately defined experimental condition.

---

# 31. Do Not Train on EvalPlus Evaluation Artifacts Accidentally

A frequent data-engineering risk is carrying evaluation metadata into the model input.

Before training, explicitly select columns.

Example:

```python
TRAIN_INPUT_COLUMNS = [
    "task_prompt",
    "generated_code",
]

TARGET_COLUMN = "label"
```

Avoid using a generic serialization of the complete record.

Otherwise fields such as:

```text
exception_type
plus_passed
failure_stage
```

would trivially reveal the label.

---

# 32. Suggested End-to-End Pipeline

The complete workflow should be:

```text
1. install EvalPlus with uv
        |
        v
2. load HumanEval+ / MBPP+
        |
        v
3. store task metadata
        |
        v
4. send only task prompt to generator
        |
        v
5. preserve raw generated response
        |
        v
6. normalize/extract executable code if required
        |
        v
7. run syntax validation
        |
        +---- invalid ----------> syntax_error
        |
        v
8. export EvalPlus-compatible samples
        |
        v
9. execute tests in sandbox
        |
        v
10. collect Base and Base+Extra results
        |
        v
11. inspect execution failure information
        |
        +---- runtime failure --> runtime_error
        |
        +---- test mismatch ----> functional_error
        |
        +---- all pass ---------> correct
        |
        v
12. save rich labeled record
        |
        v
13. split by task_id
        |
        v
14. train/evaluate hallucination detector
```

---

# 33. Minimal Collection Script

A minimal loader can be:

```python
from evalplus.data import get_human_eval_plus, get_mbpp_plus


DATASETS = {
    "humaneval": get_human_eval_plus,
    "mbpp": get_mbpp_plus,
}


def load_dataset(name: str):
    try:
        return DATASETS[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown EvalPlus dataset: {name}"
        ) from exc
```

Usage:

```python
dataset = load_dataset("humaneval")

for task_id, problem in dataset.items():
    print(task_id)
    print(problem["prompt"])
```

---

# 34. Minimal Generation Adapter

```python
def generate_dataset_samples(
    dataset,
    model_name,
    num_samples,
    generate_fn,
):
    records = []

    for task_id, problem in dataset.items():
        for sample_id in range(num_samples):
            code = generate_fn(
                model=model_name,
                prompt=problem["prompt"],
            )

            records.append(
                {
                    "task_id": task_id,
                    "entry_point": problem["entry_point"],
                    "model": model_name,
                    "sample_id": sample_id,
                    "prompt": problem["prompt"],
                    "generated_code_raw": code,
                }
            )

    return records
```

---

# 35. Minimal Syntax Labeling Stage

```python
def syntax_status(code: str) -> tuple[bool, str | None]:
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        return False, type(exc).__name__

    return True, None
```

Do this before execution.

---

# 36. Official Evaluation Stage

Export valid samples to:

```text
samples.jsonl
```

Then run:

```bash
uv run evalplus.evaluate \
    --dataset humaneval \
    --samples samples.jsonl \
    --test-details
```

or:

```bash
uv run evalplus.evaluate \
    --dataset mbpp \
    --samples samples.jsonl \
    --test-details
```

Use sandboxed execution for the final experiment.

---

# 37. Important Distinction: EvalPlus Failure vs Project Label

Do not assume:

```text
EvalPlus fail == functional_error
```

A failed EvalPlus sample can represent:

```text
syntax failure
runtime exception
timeout
memory failure
wrong functional output
```

Therefore:

```text
EvalPlus pass/fail
```

is an oracle signal, while:

```text
syntax_error
runtime_error
functional_error
correct
```

is the project's semantic mapping over execution behavior.

The classification adapter must preserve this distinction.

---

# 38. Handling Timeouts

A timeout means the implementation did not complete successfully.

For the current four-class taxonomy, the project must define a consistent policy.

Recommended policy:

```text
timeout -> runtime_error
```

because the implementation fails during execution rather than returning a valid but incorrect result.

However, preserve:

```text
exception_type = "Timeout"
```

or:

```text
timed_out = true
```

so timeout cases can later be analyzed separately.

Document this decision in the paper.

---

# 39. Handling Out-of-Memory Failures

Recommended mapping:

```text
OOM -> runtime_error
```

while preserving:

```json
{
  "out_of_memory": true,
  "exception_type": "OutOfMemory"
}
```

Again, this is a project-specific decision, not an EvalPlus default four-class taxonomy.

---

# 40. Handling Invalid Model Formatting

Models sometimes return:

```text
Here is the solution:

```python
...
```
```

There are two possible experimental policies.

### Policy A — strict

Evaluate the response as produced.

Formatting that makes it invalid Python may become:

```text
syntax_error
```

### Policy B — extraction

Apply a deterministic code extractor before evaluation.

Then store:

```text
raw response
extracted code
extractor version
```

For reproducibility, choose one policy before the final dataset generation and apply it consistently.

A useful additional experiment is:

```text
strict output
vs.
sanitized/extracted output
```

but it should not be mixed silently.

---

# 41. Recommended Validation Before Large-Scale Generation

Before generating thousands of samples:

## Step 1

Load 3–5 HumanEval+ tasks.

## Step 2

Generate one solution from one model.

## Step 3

Verify that all metadata is stored correctly.

## Step 4

Export to EvalPlus JSONL.

## Step 5

Run:

```bash
uv run evalplus.syncheck \
    --samples samples.jsonl \
    --dataset humaneval
```

## Step 6

Run full evaluation:

```bash
uv run evalplus.evaluate \
    --dataset humaneval \
    --samples samples.jsonl \
    --test-details
```

## Step 7

Manually inspect at least:

```text
one correct sample
one functional failure
one runtime failure
one syntax failure
```

when available.

Only after this validation should the pipeline be scaled.

---

# 42. Recommended Debugging Dataset

Maintain a tiny deterministic subset for development.

For example:

```text
HumanEval:
5 tasks

MBPP:
5 tasks

1 generator
1 sample per task
```

This allows pipeline changes to be checked in seconds instead of regenerating the full experiment.

Do not use this debug subset for final reported metrics.

---

# 43. Large-Scale Evaluation Strategy

For large experiments:

```text
Phase 1
Generate and store outputs.

Phase 2
Run syntax validation.

Phase 3
Run EvalPlus evaluation.

Phase 4
Convert execution outcomes into four-class labels.

Phase 5
Validate label distribution.

Phase 6
Freeze dataset.

Phase 7
Create task-level train/validation/test splits.
```

Do not train the detector while data generation is still changing.

Create a frozen dataset version first.

---

# 44. Dataset Freeze

Before training, produce a versioned manifest.

Example:

```json
{
  "dataset_name": "hallucode_evalplus",
  "version": "v1",
  "evalplus_version": "...",
  "benchmarks": [
    "humaneval+",
    "mbpp+"
  ],
  "generators": [
    "...",
    "..."
  ],
  "samples_per_task_model": 3,
  "classification_policy": "v1",
  "split_policy": "group_by_task_id"
}
```

This is highly useful for a journal replication package.

---

# 45. Quality Checks After Labeling

Run automated checks such as:

```text
syntax_error
    -> syntax_valid must be false

correct
    -> syntax_valid must be true
    -> plus_passed must be true

functional_error
    -> syntax_valid must be true
    -> plus_passed must be false
    -> no runtime failure should have precedence

runtime_error
    -> syntax_valid must be true
    -> execution failure metadata should exist
```

Example:

```python
assert not (
    (df["label"] == "correct")
    & (~df["plus_passed"])
).any()
```

---

# 46. Manual Validation

Even with executable labels, manually inspect a stratified sample.

Recommended sample:

```text
20 correct
20 functional_error
20 runtime_error
all syntax_error if the class remains small
```

Check whether the operational label corresponds to the behavior observed during execution.

Document the manual validation process.

---

# 47. What Should Be Reported in the Paper

For each benchmark, report:

```text
number of tasks
number of generators
samples per task/model
total generations
correct count
functional-error count
runtime-error count
syntax-error count
```

Also report:

```text
Base-pass rate
Plus-pass rate
Base-pass / Plus-fail count
```

The last group is especially important because it quantifies solutions that appear correct under weaker tests but fail stronger EvalPlus validation.

---

# 48. Strong Experiment Enabled by EvalPlus

A useful research question is:

> How well can a specialized detector identify generated programs that pass the original benchmark tests but fail the stronger EvalPlus tests?

Create a subset:

```python
hard_functional = df[
    (df["base_passed"])
    & (~df["plus_passed"])
]
```

This is a particularly valuable subset because these programs are harder to detect through shallow correctness signals.

---

# 49. Another Strong Experiment: Cross-Benchmark Generalization

Example:

```text
Train:
MBPP+

Test:
HumanEval+
```

Then reverse:

```text
Train:
HumanEval+

Test:
MBPP+
```

This tests whether the detector learns general patterns of incorrect code rather than benchmark-specific characteristics.

---

# 50. Another Strong Experiment: Unseen Generators

Example:

```text
Train:
all generators except GPT-OSS

Test:
GPT-OSS
```

Repeat for each generator.

This can be implemented later only if the generation metadata has been retained from the beginning.

---

# 51. Reproducibility Checklist

Before considering the EvalPlus dataset complete:

- [ ] `uv.lock` committed
- [ ] EvalPlus version recorded
- [ ] HumanEval+ tasks collected
- [ ] MBPP+ tasks collected
- [ ] canonical solutions isolated from generator inputs
- [ ] original prompts preserved
- [ ] raw LLM responses preserved
- [ ] extraction/sanitization policy documented
- [ ] multiple samples assigned unique `sample_id`
- [ ] generator metadata stored
- [ ] syntax validation executed
- [ ] official EvalPlus evaluation executed
- [ ] Base results stored
- [ ] Plus results stored
- [ ] runtime exceptions captured
- [ ] timeout policy documented
- [ ] OOM policy documented
- [ ] final four-class labels generated
- [ ] task-level leakage checks passed
- [ ] dataset frozen before training
- [ ] train/validation/test split grouped by `task_id`
- [ ] manual label sanity check performed

---

# 52. Commands Summary

## Add EvalPlus

```bash
uv add evalplus
```

Latest Git version:

```bash
uv add "evalplus @ git+https://github.com/evalplus/evalplus"
```

## Check installation

```bash
uv run evalplus.evaluate --help
```

## Syntax check HumanEval samples

```bash
uv run evalplus.syncheck \
    --samples samples.jsonl \
    --dataset humaneval
```

## Syntax check MBPP samples

```bash
uv run evalplus.syncheck \
    --samples samples.jsonl \
    --dataset mbpp
```

## Optional sanitization

```bash
uv run evalplus.sanitize \
    --samples samples.jsonl
```

## Evaluate HumanEval+

```bash
uv run evalplus.evaluate \
    --dataset humaneval \
    --samples samples.jsonl
```

## Evaluate MBPP+

```bash
uv run evalplus.evaluate \
    --dataset mbpp \
    --samples samples.jsonl
```

## Detailed HumanEval+ evaluation

```bash
uv run evalplus.evaluate \
    --dataset humaneval \
    --samples samples.jsonl \
    --test-details
```

## HumanEval+ Mini

```bash
uv run evalplus.evaluate \
    --dataset humaneval \
    --samples samples.jsonl \
    --mini
```

---

# 53. EvalPlus vs HalluCodeDetection Responsibilities

| Responsibility | EvalPlus | HalluCodeDetection |
|---|---:|---:|
| Provide HumanEval+/MBPP+ tasks | Yes | No |
| Provide stronger tests | Yes | No |
| Provide canonical oracle | Yes | No |
| Generate code using project models | Optional | **Yes** |
| Preserve generator metadata | No | **Yes** |
| Define four hallucination/error classes | No | **Yes** |
| Detect syntax failures | Partial tooling | **Yes, explicitly** |
| Execute benchmark tests | **Yes** | Uses EvalPlus |
| Separate Base vs Plus | **Yes** | Store results |
| Map runtime behavior to final labels | No | **Yes** |
| Build task-level dataset splits | No | **Yes** |
| Train the hallucination detector | No | **Yes** |

---

# 54. Recommended Final Design

The recommended design for the expanded experiment is:

```text
EvalPlus
    |
    +--> HumanEval+
    |
    +--> MBPP+
           |
           v
    task specification
           |
           v
 project LLM generators
           |
           v
 raw generated code
           |
           v
 project syntax check
           |
           v
 EvalPlus sandbox evaluation
       |              |
       v              v
     Base          Base + Extra
       \              /
        \            /
         v          v
      execution metadata
             |
             v
 project classification
   |       |       |       |
 syntax  runtime functional correct
             |
             v
     frozen labeled dataset
             |
             v
     split by task_id
             |
             v
 specialized detector training
```

The essential methodological rule is:

> **EvalPlus supplies executable ground truth; HalluCodeDetection supplies the error taxonomy and experimental protocol.**

This preserves comparability with the existing paper while making the dataset larger, more reliable, and more suitable for cross-benchmark and cross-generator generalization experiments.

---

# 55. Official References

Use the following official resources when implementation details must be checked against the currently installed EvalPlus version:

- EvalPlus repository: `https://github.com/evalplus/evalplus`
- EvalPlus command documentation: `https://github.com/evalplus/evalplus/blob/master/docs/cli.md`
- EvalPlus program execution documentation: `https://github.com/evalplus/evalplus/blob/master/docs/execution.md`
- EvalPlus project page / leaderboard: `https://evalplus.github.io/`

Because EvalPlus is actively maintained, always verify CLI flags with:

```bash
uv run evalplus.evaluate --help
```

before freezing the final experimental pipeline.
