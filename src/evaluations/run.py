from __future__ import annotations

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
    evaluation_resume = EvaluationResume() #type:ignore
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
    
    model_handler = get_model_handler(model_info)
    
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
            for progress_idx, (sample_index, sample) in enumerate(filtred):
                update_evaluation_task(
                    progress,
                    task_id,
                    evaluation_resume,
                    sample=f"sample {progress_idx + 1}",
                )
                expected_levels = sorted(set(sample.levels))
                try:
                    judge_response = model_handler.generate_judge(
                        example_prompt=sample.problem_description,
                        code=sample.generated_code,
                        temperature=model_temperature,
                    )
                    evaluation_resume.total_responses += 1

                    if judge_response.analysis:
                        evaluation_resume.parsed_responses += 1

                    predicted_levels = sorted(
                        {je.level for je in judge_response.analysis.levels}
                    ) if judge_response.analysis else []
                    correct = bool(predicted_levels) and predicted_levels == expected_levels

                    if correct:
                        evaluation_resume.corrects_by_level[sample.level] += 1

                    evaluation = EvaluationSummaryRow(
                        model_id=model_info.id,
                        kind=model_info.type,
                        sample_index=sample_index,
                        expected_levels=expected_levels,
                        predicted_levels=predicted_levels,
                        correct=correct,
                        judge_response=judge_response,
                    )
                    add_classification_metrics(
                        evaluation_resume,
                        expected_levels,
                        predicted_levels,
                    )
                except Exception as exc:
                    evaluation_resume.skipped_responses += 1
                    error = f"{type(exc).__name__}: {exc}"
                    ui.console.print(
                        f"[yellow]Skipping sample {sample_index} for {model_info.id}: {error}[/]"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    evaluation = EvaluationSummaryRow(
                        model_id=model_info.id,
                        kind=model_info.type,
                        sample_index=sample_index,
                        expected_levels=expected_levels,
                        predicted_levels=[],
                        correct=None,
                        judge_response={"raw_response": ""},
                        error=error[:2_000],
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
                    sample=f"sample {progress_idx + 1}",
                    advance=1,
                )
                
                to_update.append(evaluation.model_dump())
                if checkpoint and (progress_idx + 1) % checkpoint == 0:
                    if file_save and to_update:
                        append_jsonl(file_save, to_update)
                        to_update = []

    finally:
        if file_save and to_update:
            append_jsonl(file_save, to_update)
        model_handler.close()
    
    evaluation_resume.overall_accuracy = sum(evaluation_resume.corrects_by_level.values()) / evaluation_resume.total_responses if evaluation_resume.total_responses > 0 else 0.0
    
    return evaluation_resume

def evaluate_models(config: HalluCodeDetectionConfig, retry: bool = False) -> None:
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
    if not config.evaluation_config.models:
        ui.console.print("[yellow]Skipping evaluation: no evaluation models configured.[/]")
        return

    num_ctx = config.evaluation_config.num_ctx
    think = config.evaluation_config.think
    
    for model_info in config.evaluation_config.models:
        eval_model_info = model_info
        if model_info.type == "ollama":
            eval_model_info = model_info.model_copy(update={
                "num_ctx": model_info.num_ctx if model_info.num_ctx is not None else num_ctx,
                "think": model_info.think if model_info.think is not None else think,
            })
        
        completed = len([
            item for item in summary
            if item.model_id == eval_model_info.id and item.kind == eval_model_info.type
        ])
        if not config.evaluation_config.overwrite and completed >= SPLIT_SIZE:
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
    for model_info in config.evaluation_config.models:
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
