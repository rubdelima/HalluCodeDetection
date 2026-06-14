from __future__ import annotations

from pathlib import Path
from datasets import Dataset
from src.core import ui

from src.evaluations.ui import (
    add_evaluation_task,
    get_progress_evaluation,
    render_summary_table,
    update_evaluation_task,
)
from src.constants import HalluCodeDetectionConfig
from src.constants.models import ModelInfo
from src.dataset.utils import load_jsonl, append_jsonl
from src.schemas.evaluation import EvaluationSummaryRow, EvaluationResume
from src.models import get_model_handler
from typing import Optional, List, Tuple

from src.dataset.judge_dataset import Record, load_hallucination_dataset

def filter_records(
    model_info: ModelInfo,
    dataset_split: Dataset,
    sumary_dict: Optional[dict[tuple[str, str, int], EvaluationSummaryRow]],
    overwrite: bool = True
    )->Tuple[List[Record], EvaluationResume]:
    evaluation_resume = EvaluationResume() #type:ignore
    records : List[Record] = []
    
    if overwrite or not sumary_dict:
        return [
            Record(**dataset_split[idx]) for idx, sample in enumerate(dataset_split)], evaluation_resume
    
    total_judges_responses = {
        "correct": 0,
        "functional_error": 0,
        "runtime_error": 0,
        "syntax_error": 0
    }
    
    for idx, sample in enumerate(dataset_split):
        evaluation_resume.total_responses += 1
        key = (model_info.id, model_info.type, idx)
        
        if key in sumary_dict:
            total_judges_responses[sumary_dict[key].predicted_level] += 1
            
            judge_analysis = sumary_dict[key].judge_response.analysis
            if judge_analysis:
                evaluation_resume.parsed_responses += 1
                if sumary_dict[key].correct:
                    evaluation_resume.overall_accuracy += 1
                    evaluation_resume.corrects_by_level[judge_analysis.level] += 1
                    
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
    summary_dict: Optional[dict[tuple[str, str, int], EvaluationSummaryRow]] = None
    )-> EvaluationResume:
    
    filtred, evaluation_resume = filter_records(
        model_info=model_info,
        dataset_split=dataset_split,
        sumary_dict=summary_dict,
        overwrite=overwrite
    )
    
    model_handler = get_model_handler(model_info)
    
    to_update = []
    try:
        progress = get_progress_evaluation()
        task_id = add_evaluation_task(
            progress,
            total=len(filtred),
            provider=model_info.type,
            model=model_info.id,
            resume=evaluation_resume,
        )

        with progress:
            for idx, sample in enumerate(filtred):
                update_evaluation_task(
                    progress,
                    task_id,
                    evaluation_resume,
                    sample=f"sample {idx + 1}",
                )
                judge_response = model_handler.generate_judge(
                    example_prompt = sample.problem_description,
                    code = sample.generated_code,
                    temperature = model_temperature
                )
                evaluation_resume.total_responses += 1
                
                predicted_level = judge_response.analysis.level if judge_response.analysis else None
                    
                if predicted_level is not None:
                    evaluation_resume.parsed_responses += 1
                    evaluation_resume.corrects_by_level[predicted_level] += 1 if predicted_level == sample.level else 0
                
                evaluation_resume.overall_accuracy = (
                    sum(evaluation_resume.corrects_by_level.values()) / evaluation_resume.total_responses
                    if evaluation_resume.total_responses > 0 else 0.0
                )
                
                evaluation = EvaluationSummaryRow(
                    model_id=model_info.id,
                    kind=model_info.type,
                    sample_index=idx,
                    expected_level = sample.level, #type: ignore
                    correct = sample.level == predicted_level if predicted_level is not None else False,
                    predicted_level = predicted_level,
                    judge_response = judge_response,
                )
                
                evaluation_resume.evaluations.append(evaluation)
                update_evaluation_task(
                    progress,
                    task_id,
                    evaluation_resume,
                    sample=f"sample {idx + 1}",
                    advance=1,
                )
                
                if checkpoint and idx % checkpoint == 0:
                    to_update.append(evaluation.model_dump())
                    if file_save and to_update:
                        append_jsonl(file_save, to_update)
                        to_update = []

    finally:
        if file_save and to_update:
            append_jsonl(file_save, to_update)
        model_handler.close()
    
    evaluation_resume.overall_accuracy = sum(evaluation_resume.corrects_by_level.values()) / evaluation_resume.total_responses if evaluation_resume.total_responses > 0 else 0.0
    
    return evaluation_resume

def evaluate_models(config: HalluCodeDetectionConfig) -> None:
    results_dir = Path(config.dataset_building_config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = results_dir / "evaluation_results.jsonl"
    
    split_raw = load_hallucination_dataset(config)
    
    test_dataset = split_raw["test"]
    
    summary = load_jsonl(output_path, EvaluationSummaryRow)
    summary_dict = {
        (row.model_id, row.kind, row.sample_index): row for row in summary
    }
    
    SPLIT_SIZE = len(test_dataset)
    summary_rows: list[dict[str, object]] = []
    
    for model_info in config.evaluation_config.models:
        
        if len([item for item in summary if item.model_id == model_info.id]) == SPLIT_SIZE:
            ui.console.print(f"[yellow]Skipping {model_info.id}: evaluation already complete.[/]")
            continue
        
        evaluation_result = evaluate_model(
            dataset_split=test_dataset,
            model_info=model_info,
            model_temperature=config.evaluation_config.model_temperature,
            checkpoint=config.evaluation_config.checkpoint_interval,
            file_save=output_path,
            summary_dict=summary_dict
        )
        summary_rows.append(
            {
                "provider": model_info.type,
                "model": model_info.id,
                "parsed": evaluation_result.parsed_responses,
                "total": evaluation_result.total_responses,
                **evaluation_result.corrects_by_level,
                "accuracy": f"{evaluation_result.overall_accuracy * 100:.2f}%",
            }
        )
        
    if summary_rows:
        ui.console.print(render_summary_table(summary_rows))
