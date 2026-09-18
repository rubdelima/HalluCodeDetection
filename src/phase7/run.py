from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.constants import HalluCodeDetectionConfig
from src.constants.models import ModelInfo
from src.core import ui
from src.dataset.judge_dataset import load_hallucination_dataset
from src.dataset.load import load_all_examples
from src.dataset.utils import append_jsonl, load_jsonl
from src.evaluations.metrics import add_classification_metrics, recalls_by_level
from src.evaluations.ui import (
    add_evaluation_task,
    get_progress_evaluation,
    render_summary_table,
    update_evaluation_task,
)
from src.models import get_model_handler
from src.models.base import BaseModelHandler
from src.schemas.dataset import BaseResultRow
from src.schemas.evalplus import EvalPlusExample
from src.schemas.evaluation import EvaluationResume
from src.schemas.judge import JudgeResponse
from src.schemas.phase7 import Phase7ResultRow
from src.static_analysis import StaticToolFeedback, analyze_with_compile_and_pyright


@dataclass(frozen=True)
class Phase7Sample:
    sample_id: str
    base: BaseResultRow
    example: EvalPlusExample


def _load_phase1_test_samples(config: HalluCodeDetectionConfig) -> list[Phase7Sample]:
    # Only task IDs are taken from the existing grouped split. Candidate code
    # and labels always come directly from Phase 1, never from Phase 6.
    split = load_hallucination_dataset(config, correct_size=1.0)
    test_task_ids = {str(row["task_id"]) for row in split["test"]}
    base_path = Path(config.dataset_building_config.results_dir) / "dataset_base.json"
    base_rows = load_jsonl(base_path, BaseResultRow)
    examples = load_all_examples(config.dataset_building_config.datasets_base)

    samples: list[Phase7Sample] = []
    seen: set[str] = set()
    for row in base_rows:
        task_id = row.task_id or f"{row.benchmark}/{row.benchmark_id}"
        if task_id not in test_task_ids:
            continue
        example = examples.get((row.benchmark, row.benchmark_id))
        if example is None:
            continue
        sample_id = f"{row.benchmark}:{row.benchmark_id}:{row.model}"
        if sample_id in seen:
            continue
        seen.add(sample_id)
        samples.append(Phase7Sample(sample_id, row, example))
    return samples


def _select_models(
    configured: list[ModelInfo], model_names: list[str] | None
) -> list[ModelInfo]:
    if model_names is None:
        return configured
    requested = set(model_names)
    selected = [model for model in configured if model.id in requested]
    missing = requested - {model.id for model in configured}
    if missing:
        available = ", ".join(model.id for model in configured)
        raise ValueError(
            f"Phase-7 model(s) not configured: {', '.join(sorted(missing))}. "
            f"Available: {available}"
        )
    return selected


def _evaluate_sample(
    sample: Phase7Sample,
    model: ModelInfo,
    handler: BaseModelHandler,
    temperature: float,
) -> Phase7ResultRow:
    expected = sorted({level.level_name for level in sample.base.levels})
    feedback = StaticToolFeedback(compile_ok=False)
    try:
        feedback = analyze_with_compile_and_pyright(sample.base.code, sample.example)
        response = handler.generate_judge_with_tools(
            example_prompt=sample.example.prompt,
            code=sample.base.code,
            tool_feedback=feedback.as_prompt(),
            temperature=temperature,
        )
        predicted = sorted(
            {item.level for item in response.analysis.levels}
        ) if response.analysis else []
        return Phase7ResultRow(
            model_id=model.id,
            kind=model.type,
            sample_id=sample.sample_id,
            benchmark=sample.base.benchmark,
            benchmark_id=sample.base.benchmark_id,
            task_id=sample.example.task_id,
            response_model=sample.base.model,
            expected_levels=expected,
            predicted_levels=predicted,
            correct=bool(predicted) and predicted == expected,
            tool_feedback=feedback,
            judge_response=response,
        )
    except Exception as exc:  # noqa: BLE001 - persist failures for --retry
        return Phase7ResultRow(
            model_id=model.id,
            kind=model.type,
            sample_id=sample.sample_id,
            benchmark=sample.base.benchmark,
            benchmark_id=sample.base.benchmark_id,
            task_id=sample.example.task_id,
            response_model=sample.base.model,
            expected_levels=expected,
            tool_feedback=feedback,
            judge_response=JudgeResponse(raw_response=""),
            error=f"{type(exc).__name__}: {exc}"[:2_000],
        )


def _add_to_resume(
    resume: EvaluationResume,
    row: Phase7ResultRow,
    primary_level: str,
) -> None:
    if row.error:
        resume.skipped_responses += 1
        return
    resume.total_responses += 1
    if row.judge_response.analysis:
        resume.parsed_responses += 1
    if row.correct and primary_level in resume.corrects_by_level:
        resume.corrects_by_level[primary_level] += 1  # type: ignore[literal-required]
    add_classification_metrics(resume, row.expected_levels, row.predicted_levels)
    resume.overall_accuracy = (
        sum(resume.corrects_by_level.values()) / resume.total_responses
        if resume.total_responses else 0.0
    )


def run_phase7(
    config: HalluCodeDetectionConfig,
    *,
    retry: bool = False,
    model_names: list[str] | None = None,
    limit: int | None = None,
    only_errors: bool = False,
) -> None:
    phase = config.phase7_config
    models = _select_models(phase.models, model_names)
    if phase.num_ctx is not None or phase.max_tokens is not None:
        models = [
            model.model_copy(update={
                **({"num_ctx": phase.num_ctx} if phase.num_ctx is not None else {}),
                **({"max_tokens": phase.max_tokens} if phase.max_tokens is not None else {}),
            })
            if model.type == "ollama" else model
            for model in models
        ]
    if not models:
        ui.console.print("[yellow]Skipping Phase 7: no models configured.[/]")
        return

    samples = _load_phase1_test_samples(config)
    if only_errors:
        samples = [
            sample for sample in samples
            if {level.level_name for level in sample.base.levels} != {"correct"}
        ]
    if limit is not None:
        if limit < 1:
            raise ValueError("--phase7_limit must be at least 1.")
        samples = samples[:limit]
    if not samples:
        ui.console.print("[yellow]Skipping Phase 7: no Phase-1 test samples selected.[/]")
        return

    output_dir = Path(phase.results_dir or Path(config.dataset_building_config.results_dir) / "phase7")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = (
        "classification_with_tools_smoke.jsonl"
        if limit is not None
        else "classification_with_tools.jsonl"
    )
    output_path = output_dir / output_name
    if phase.overwrite and output_path.exists():
        output_path.unlink()
    prior_rows = load_jsonl(output_path, Phase7ResultRow, quiet=True)
    existing = {(row.model_id, row.sample_id): row for row in prior_rows}
    samples_by_id = {sample.sample_id: sample for sample in samples}

    summary_rows: list[dict[str, object]] = []
    started = time.monotonic()
    for model in models:
        resume = EvaluationResume()
        for (model_id, sample_id), row in existing.items():
            sample = samples_by_id.get(sample_id)
            if (
                model_id == model.id
                and sample is not None
                and not (retry and row.error is not None)
            ):
                _add_to_resume(resume, row, sample.base.primary_level_name)

        pending = [
            sample for sample in samples
            if (model.id, sample.sample_id) not in existing
            or (retry and existing[(model.id, sample.sample_id)].error is not None)
        ]
        progress = get_progress_evaluation()
        progress_id = add_evaluation_task(
            progress,
            total=len(pending),
            provider=model.type,
            model=model.name,
            resume=resume,
        )
        if pending:
            handler = get_model_handler(model)
            try:
                def record(
                    sample: Phase7Sample,
                    row: Phase7ResultRow,
                    index: int,
                    *,
                    current_model: ModelInfo = model,
                    current_resume: EvaluationResume = resume,
                    current_progress=progress,
                    current_progress_id=progress_id,
                ) -> None:
                    append_jsonl(output_path, [row.model_dump()])
                    existing[(current_model.id, sample.sample_id)] = row
                    _add_to_resume(current_resume, row, sample.base.primary_level_name)
                    if row.error:
                        ui.console.print(
                            f"[yellow]Phase 7 skipped {sample.sample_id}: {row.error}[/]"
                        )
                    update_evaluation_task(
                        current_progress, current_progress_id, current_resume,
                        sample=f"sample {index}", advance=1,
                    )

                with progress:
                    if model.type == "opencode_go" and model.threads > 1:
                        executor = ThreadPoolExecutor(
                            max_workers=model.threads,
                            thread_name_prefix=f"phase7-{model.id}",
                        )
                        futures = {
                            executor.submit(
                                _evaluate_sample, sample, model, handler, phase.model_temperature
                            ): sample
                            for sample in pending
                        }
                        try:
                            for index, future in enumerate(as_completed(futures), start=1):
                                sample = futures[future]
                                record(sample, future.result(), index)
                        except BaseException:
                            for future in futures:
                                future.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
                        else:
                            executor.shutdown(wait=True)
                    else:
                        for index, sample in enumerate(pending, start=1):
                            record(
                                sample,
                                _evaluate_sample(
                                    sample, model, handler, phase.model_temperature
                                ),
                                index,
                            )
            finally:
                handler.close()

        summary_rows.append({
            "provider": model.type,
            "model": model.name,
            "parsed": resume.parsed_responses,
            "total": resume.total_responses,
            "skipped": resume.skipped_responses,
            "accuracy": f"{resume.overall_accuracy * 100:.1f}",
            "macro_f1": f"{resume.macro_f1 * 100:.1f}",
            **{level: f"{value * 100:.1f}" for level, value in recalls_by_level(resume).items()},
        })

    ui.console.print(render_summary_table(summary_rows))
    ui.console.print(
        f"[green]Phase 7 finished in {time.monotonic() - started:.1f}s:[/] {output_path}"
    )
