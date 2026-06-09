from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, cast

from src.core import ui

from src.constants.dataset import DatasetBuildingConfig

from src.dataset.load import load_mbpp_split
from src.dataset.utils import load_jsonl, append_jsonl, get_pending_tasks
from src.dataset.ui import get_augmentation_progress

from src.models import get_model_handler

from src.schemas.dataset import BaseResultRow, JudgeResultRow

def _parse_judge_response(content: str) -> dict[str, str]:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "explanation" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    return {"explanation": content.strip()}


def dataset_judge(config : DatasetBuildingConfig) -> None:
    if config.judge_model is None:
        ui.console.print("No judge model specified in the configuration. Please specify a judge model to run the dataset judge step.")
        raise ValueError("No judge model specified in the configuration.")
    
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_path = results_dir / "dataset_base.json"
    judge_path = results_dir / "dataset_judge.jsonl"

    if not base_path.exists():
        ui.console.print(f"Base results file not found in {results_dir}. Please run the dataset building step first.")
        return
    
    base_results = load_jsonl(str(base_path), BaseResultRow)
    
    if not base_results:
        ui.console.print("No base results found to judge.")
        return

    existing = load_jsonl(str(judge_path), JudgeResultRow)
    mbpp_train = load_mbpp_split("train")
    mbpp_by_id = {ex.benchmark_id: ex for ex in mbpp_train}
    
    pending: list[tuple[BaseResultRow, int]] = get_pending_tasks(mbpp_by_id, existing, base_results, config.judge_model.id)

    if not pending:
        ui.console.print("All items already judged.")
        return

    handler = get_model_handler(config.judge_model)

    try:
        pending_write: list[dict[str, object]] = []
        with get_augmentation_progress(ui.console) as progress:
            task_id = progress.add_task("judge",total=len(pending),model=config.judge_model.id,bench="-",bench_id="-")
            for index, (pending_item, bench_id) in enumerate(pending, start=1):
                example = mbpp_by_id[bench_id]
                try:
                    judge_result = handler.analyze_hallucination(example.prompt,pending_item,config.model_temperature)
                    pending_write.append(judge_result.model_dump())
                
                except Exception as exc:  # noqa: BLE001
                    ui.console.print(f"Error judging item {bench_id}: {exc}")

                if (index % config.checkpoint_interval == 0) and pending_write:
                    append_jsonl(judge_path, pending_write)
                    pending_write = []

                progress.update(
                    task_id,
                    advance=1,
                    model=config.judge_model.id,
                    bench=pending_item.benchmark,
                    bench_id=str(bench_id),
                )

        if pending_write:
            append_jsonl(judge_path, pending_write)

    except Exception as exc:
        ui.console.print(f"Error occurred while judging dataset: {exc}")
    
    finally:
        handler.close()
        
    ui.console.print("Dataset judge finished.")
