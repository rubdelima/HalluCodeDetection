# Planned repeated-generation replication

This protocol is specified prospectively but was **not executed** in the current
revision because it requires renewed access to every hosted/local model and a
substantial inference budget. It must not be used to claim sampling stability
for the reported single-generation experiment.

## Design

- Re-run the 563 EvalPlus tasks for every Phase-1 model with five independent
  generations per task/model condition.
- Use explicit backend seeds `20260921` through `20260925` where the backend
  exposes a seed. Record when a hosted endpoint accepts but does not guarantee
  seed determinism.
- Preserve the original prompt and temperature 1.0; explicitly set and log
  top-p/top-k instead of inheriting provider defaults.
- Save each generation under a new immutable run ID. Do not append the rows to
  the current Phase-1 dataset or use them in the existing train/test split.
- Evaluate every response with the same pinned EvalPlus harness and store the
  model identifier, endpoint date, decoding parameters, seed, raw response,
  normalized code, and execution-derived labels.

## Analysis

- Report per-model mean Correct proportion across runs, between-run standard
  deviation, and a task-clustered interval that retains the repeated responses
  within each task.
- Estimate both task-to-task and generation-to-generation variability with a
  hierarchical bootstrap (tasks first, then repetitions within tasks).
- Repeat the paired model contrasts on run-matched generations and report how
  often the direction of each effect changes across the five runs.
- Keep this replication exploratory and separate from the current one-sample
  results unless the protocol is executed in full for all compared conditions.

## Budget gate

The design requires `13 × 563 × 5 = 36,595` generation calls plus EvalPlus
execution. API cost/rate limits and local accelerator time must be approved
before execution. Partial execution would not support the intended cross-model
stability comparison.
