from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datasets import Dataset
import torch
from src.core import ui

from src.evaluations.ui import (
    add_evaluation_task,
    get_progress_evaluation,
    render_summary_table,
    update_evaluation_task,
)
from src.evaluations.metrics import add_classification_metrics, recalls_by_level
from src.constants import HalluCodeDetectionConfig
from src.constants.models import ModelInfo
from src.dataset.utils import load_jsonl, append_jsonl
from src.schemas.evaluation import EvaluationSummaryRow, EvaluationResume
from src.schemas.judge import JudgeResponse
from src.models import get_model_handler
from typing import Optional, List, Tuple

from src.dataset.judge_dataset import Record, load_hallucination_dataset


def _display_model_name(model_info: ModelInfo) -> str:
    """Avoid leaking long local model paths into progress and summary output."""
    return "trained_model" if Path(model_info.id).is_absolute() else model_info.name

def filter_records(
    model_info: ModelInfo,
    dataset_split: Dataset,
    summary_dict: Optional[dict[tuple[str, str, int], EvaluationSummaryRow]],
    overwrite: bool = True,
    retry: bool = False,
    )->Tuple[List[Tuple[int, Record]], EvaluationResume]:
    evaluation_resume = EvaluationResume()
    records : List[Tuple[int, Record]] = []
    
    if overwrite or not summary_dict:
        return [
            (idx, Record(**sample)) for idx, sample in enumerate(dataset_split)], evaluation_resume
    
    for idx, sample in enumerate(dataset_split):
        key = (model_info.id, model_info.type, idx)
        
        if key in summary_dict:
            row = summary_dict[key]
            if row.error:
                if retry:
                    records.append((idx, Record(**sample)))
                else:
                    evaluation_resume.skipped_responses += 1
                continue
            evaluation_resume.total_responses += 1
            if row.judge_response.analysis:
                evaluation_resume.parsed_responses += 1
            add_classification_metrics(
                evaluation_resume,
                row.expected_levels,
                row.predicted_levels,
            )
            if row.correct:
                evaluation_resume.overall_accuracy += 1
                record = Record(**sample)
                evaluation_resume.corrects_by_level[record.level] += 1
            continue

        records.append((idx, Record(**sample)))
                    
    if evaluation_resume.total_responses > 0:
        evaluation_resume.overall_accuracy = evaluation_resume.overall_accuracy / evaluation_resume.total_responses
    
    return records, evaluation_resume
    

def evaluate_model(
    dataset_split: Dataset, 
    model_info: ModelInfo,
    model_temperature : float = 0.0,
    overwrite: bool = True,
    checkpoint:Optional[int] = None,
    file_save: Optional[Path] = None,
    summary_dict: Optional[dict[tuple[str, str, int], EvaluationSummaryRow]] = None,
    retry: bool = False,
    )-> EvaluationResume:
    
    filtred, evaluation_resume = filter_records(
        model_info=model_info,
        dataset_split=dataset_split,
        summary_dict=summary_dict,
        overwrite=overwrite,
        retry=retry,
    )

    if not filtred:
        return evaluation_resume
    
    model_handler = get_model_handler(model_info)
    
    to_update: list[dict[str, object]] = []

    def evaluate_sample(sample_index: int, sample: Record) -> EvaluationSummaryRow:
        expected_levels = sorted(set(sample.levels))
        try:
            judge_response = model_handler.generate_judge(
                example_prompt=sample.problem_description,
                code=sample.generated_code,
                temperature=model_temperature,
            )
            predicted_levels = sorted(
                {item.level for item in judge_response.analysis.levels}
            ) if judge_response.analysis else []
            return EvaluationSummaryRow(
                model_id=model_info.id,
                kind=model_info.type,
                sample_index=sample_index,
                expected_levels=expected_levels,
                predicted_levels=predicted_levels,
                correct=bool(predicted_levels) and predicted_levels == expected_levels,
                judge_response=judge_response,
            )
        except Exception as exc:  # noqa: BLE001
            return EvaluationSummaryRow(
                model_id=model_info.id,
                kind=model_info.type,
                sample_index=sample_index,
                expected_levels=expected_levels,
                predicted_levels=[],
                correct=None,
                judge_response=JudgeResponse(raw_response=""),
                error=f"{type(exc).__name__}: {exc}"[:2_000],
            )

    def record_evaluation(
        progress,
        task_id,
        progress_idx: int,
        sample: Record,
        evaluation: EvaluationSummaryRow,
    ) -> None:
        nonlocal to_update
        if evaluation.error:
            evaluation_resume.skipped_responses += 1
            ui.console.print(
                f"[yellow]Skipping sample {evaluation.sample_index} for "
                f"{model_info.id}: {evaluation.error}[/]"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            evaluation_resume.total_responses += 1
            if evaluation.judge_response.analysis:
                evaluation_resume.parsed_responses += 1
            if evaluation.correct:
                evaluation_resume.corrects_by_level[sample.level] += 1
            add_classification_metrics(
                evaluation_resume,
                evaluation.expected_levels,
                evaluation.predicted_levels,
            )

        evaluation_resume.overall_accuracy = (
            sum(evaluation_resume.corrects_by_level.values()) / evaluation_resume.total_responses
            if evaluation_resume.total_responses > 0 else 0.0
        )
        evaluation_resume.evaluations.append(evaluation)
        update_evaluation_task(
            progress,
            task_id,
            evaluation_resume,
            sample=f"sample {progress_idx}",
            advance=1,
        )
        to_update.append(evaluation.model_dump())
        if checkpoint and progress_idx % checkpoint == 0 and file_save and to_update:
            append_jsonl(file_save, to_update)
            to_update = []

    try:
        progress = get_progress_evaluation()
        task_id = add_evaluation_task(
            progress,
            total=len(filtred),
            provider=model_info.type,
            model=_display_model_name(model_info),
            resume=evaluation_resume,
        )

        with progress:
            if model_info.type == "opencode_go" and model_info.threads > 1:
                executor = ThreadPoolExecutor(
                    max_workers=model_info.threads,
                    thread_name_prefix=f"opencode-evaluate-{model_info.id}",
                )
                futures = {}
                try:
                    futures = {
                        executor.submit(evaluate_sample, sample_index, sample): (sample_index, sample)
                        for sample_index, sample in filtred
                    }
                    for progress_idx, future in enumerate(as_completed(futures), start=1):
                        _, sample = futures[future]
                        record_evaluation(
                            progress,
                            task_id,
                            progress_idx,
                            sample,
                            future.result(),
                        )
                except BaseException:
                    for future in futures:
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                else:
                    executor.shutdown(wait=True)
            else:
                for progress_idx, (sample_index, sample) in enumerate(filtred, start=1):
                    record_evaluation(
                        progress,
                        task_id,
                        progress_idx,
                        sample,
                        evaluate_sample(sample_index, sample),
                    )

    finally:
        if file_save and to_update:
            append_jsonl(file_save, to_update)
        model_handler.close()
    
    evaluation_resume.overall_accuracy = sum(evaluation_resume.corrects_by_level.values()) / evaluation_resume.total_responses if evaluation_resume.total_responses > 0 else 0.0
    
    return evaluation_resume

def evaluate_models(
    config: HalluCodeDetectionConfig,
    retry: bool = False,
    model_names: list[str] | None = None,
) -> None:
    results_dir = Path(config.dataset_building_config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = results_dir / "evaluation_results.jsonl"
    
    split_raw = load_hallucination_dataset(config, correct_size=1.0)
    
    test_dataset = split_raw["test"]
    if len(test_dataset) == 0:
        ui.console.print("[yellow]Skipping evaluation: test dataset is empty.[/]")
        return
    
    summary = load_jsonl(output_path, EvaluationSummaryRow)
    summary_dict = {
        (row.model_id, row.kind, row.sample_index): row for row in summary
    }
    
    SPLIT_SIZE = len(test_dataset)
    evaluation_models = config.evaluation_config.models
    if model_names is not None:
        requested_models = set(model_names)
        evaluation_models = [
            model for model in evaluation_models if model.id in requested_models
        ]
        configured_ids = {model.id for model in config.evaluation_config.models}
        missing_models = requested_models - configured_ids
        if missing_models:
            available = ", ".join(model.id for model in config.evaluation_config.models)
            raise ValueError(
                f"Evaluation model(s) not configured: {', '.join(sorted(missing_models))}. "
                f"Available models: {available}"
            )

    if not evaluation_models:
        ui.console.print("[yellow]Skipping evaluation: no evaluation models configured.[/]")
        return

    num_ctx = config.evaluation_config.num_ctx
    think = config.evaluation_config.think
    
    for model_info in evaluation_models:
        eval_model_info = model_info
        if model_info.type == "ollama":
            eval_model_info = model_info.model_copy(update={
                "num_ctx": model_info.num_ctx if model_info.num_ctx is not None else num_ctx,
                "think": model_info.think if model_info.think is not None else think,
            })

        if (
            eval_model_info.type == "gemma"
            and eval_model_info.local_path
            and not Path(eval_model_info.local_path).exists()
        ):
            ui.console.print(
                f"[yellow]Skipping {eval_model_info.id}: local model not found at "
                f"{eval_model_info.local_path}.[/]"
            )
            continue
        
        completed = len([
            item for item in summary
            if item.model_id == eval_model_info.id and item.kind == eval_model_info.type
        ])
        if not config.evaluation_config.overwrite and not retry and completed >= SPLIT_SIZE:
            ui.console.print(f"[yellow]Skipping {eval_model_info.id}: evaluation already complete.[/]")
            continue
        
        evaluate_model(
            dataset_split=test_dataset,
            model_info=eval_model_info,
            model_temperature=config.evaluation_config.model_temperature,
            overwrite=config.evaluation_config.overwrite,
            checkpoint=config.evaluation_config.checkpoint_interval,
            file_save=output_path,
            summary_dict=summary_dict,
            retry=retry,
        )
    # Reload persisted results so the final table compares every configured
    # baseline and fine-tuned model, including models skipped on resume.
    persisted = load_jsonl(output_path, EvaluationSummaryRow)
    persisted_dict = {
        (row.model_id, row.kind, row.sample_index): row for row in persisted
    }
    summary_rows: list[dict[str, object]] = []
    for model_info in evaluation_models:
        _, resume = filter_records(
            model_info=model_info,
            dataset_split=test_dataset,
            summary_dict=persisted_dict,
            overwrite=False,
        )
        if resume.total_responses == 0:
            continue
        summary_rows.append(
            {
                "provider": model_info.type,
                "model": _display_model_name(model_info),
                "parsed": resume.parsed_responses,
                "total": resume.total_responses,
                "skipped": resume.skipped_responses,
                "macro_f1": f"{resume.macro_f1 * 100:.1f}",
                **{level: f"{value * 100:.1f}" for level, value in recalls_by_level(resume).items()},
                "accuracy": f"{resume.overall_accuracy * 100:.1f}",
            }
        )
        
    if summary_rows:
        ui.console.print(render_summary_table(summary_rows))
