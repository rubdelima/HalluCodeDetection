from __future__ import annotations

from pathlib import Path

from src.core import ui

from src.constants.dataset import DatasetBuildingConfig

from src.dataset.load import load_all_examples
from src.dataset.utils import load_jsonl, append_jsonl, get_pending_tasks
from src.dataset.ui import get_augmentation_progress

from src.models import get_model_handler

from src.schemas.dataset import BaseResultRow, JudgeResultRow


def dataset_judge(config: DatasetBuildingConfig) -> None:
    judges = config.judge_models
    if not judges:
        ui.console.print("No judge model specified in the configuration. Please specify at least one judge model.")
        raise ValueError("No judge model specified in the configuration.")

    results_dir = Path(config.results_dir)
    judge_dir = Path(config.judge_results_dir)
    judge_dir.mkdir(parents=True, exist_ok=True)

    base_path = results_dir / "dataset_base.json"
    judge_path = judge_dir / "dataset_judge.jsonl"

    if not base_path.exists():
        ui.console.print(f"Base results file not found in {results_dir}. Please run the dataset building step first.")
        return

    base_results = load_jsonl(str(base_path), BaseResultRow)
    if not base_results:
        ui.console.print("No base results found to judge.")
        return

    examples_by_key = load_all_examples(config.datasets_base)

    for judge_model in judges:
        existing = load_jsonl(str(judge_path), JudgeResultRow)
        pending = get_pending_tasks(examples_by_key, existing, base_results, judge_model.id)

        if not pending:
            ui.console.print(f"All items already judged by {judge_model.id}.")
            continue

        handler = get_model_handler(judge_model)
        try:
            pending_write: list[dict[str, object]] = []
            with get_augmentation_progress(ui.console) as progress:
                task_id = progress.add_task(
                    "judge", total=len(pending), model=judge_model.id, bench="-", bench_id="-"
                )
                for index, (pending_item, bench_id) in enumerate(pending, start=1):
                    example = examples_by_key[(pending_item.benchmark, bench_id)]
                    try:
                        judge_result = handler.analyze_hallucination(
                            example.prompt, pending_item, config.judge_temperature
                        )
                        pending_write.append(judge_result.model_dump())
                    except Exception as exc:  # noqa: BLE001
                        ui.console.print(f"[yellow]Error judging item {bench_id} (will retry later): {exc}[/]")

                    if (index % config.checkpoint_interval == 0) and pending_write:
                        append_jsonl(judge_path, pending_write)
                        pending_write = []

                    progress.update(
                        task_id,
                        advance=1,
                        model=judge_model.id,
                        bench=pending_item.benchmark,
                        bench_id=str(bench_id),
                    )

            if pending_write:
                append_jsonl(judge_path, pending_write)
        finally:
            handler.close()

    ui.console.print("Dataset judge finished.")
