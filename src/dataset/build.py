from __future__ import annotations

import atexit
import json
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.core import ui
from src.core.log import get_logger, setup_logging
from src.dataset.load import load_evalplus, sample_examples
from src.schemas.dataset import BaseResultRow, Level
from src.schemas.evalplus import EvalPlusExample
from src.evaluations.classify import run_evalplus_tests
from src.models.ollama_handler import OllamaHandler
from src.constants.dataset import DatasetBuildingConfig

log = get_logger("build")


@dataclass(frozen=True)
class TaskKey:
    benchmark: str
    benchmark_id: int
    model: str


def ensure_results_dir(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)


def results_file_path(results_dir: Path) -> Path:
    return results_dir / "dataset_base.json"


def load_existing_results(path: Path) -> list[BaseResultRow]:
    if not path.exists():
        return []
    results: list[BaseResultRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(BaseResultRow.model_validate(json.loads(line)))
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping invalid result line: %s", exc)
    return results


def save_results_jsonl(path: Path, results: Iterable[BaseResultRow]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item.model_dump(), ensure_ascii=True) + "\n")


def build_task_keys(
    models: list[str],
    examples: Iterable[EvalPlusExample],
    existing: set[TaskKey],
) -> list[TaskKey]:
    pending: list[TaskKey] = []
    for model in models:
        for example in examples:
            key = TaskKey(example.benchmark_name, example.benchmark_id, model)
            if key not in existing:
                pending.append(key)
    return pending


def count_levels(results: Iterable[BaseResultRow]) -> dict[str, int]:
    counts = {
        "correct": 0,
        "functional": 0,
        "runtime": 0,
        "syntax": 0,
        "total": 0,
    }
    for item in results:
        counts["total"] += 1
        names = {lv.level_name for lv in item.levels}
        if names == {"correct"}:
            counts["correct"] += 1
        else:
            for name in names:
                if name in counts and name != "correct":
                    counts[name] += 1
    return counts


def count_levels_by_model(results: Iterable[BaseResultRow]) -> dict[str, dict[str, int]]:
    model_counts: dict[str, dict[str, int]] = {}
    for item in results:
        model = item.model
        if model not in model_counts:
            model_counts[model] = {
                "correct": 0,
                "functional": 0,
                "runtime": 0,
                "syntax": 0,
                "total": 0,
            }
        model_counts[model]["total"] += 1
        names = {lv.level_name for lv in item.levels}
        if names == {"correct"}:
            model_counts[model]["correct"] += 1
        else:
            for name in names:
                if name in model_counts[model] and name != "correct":
                    model_counts[model][name] += 1
    return model_counts


def build_dataset(config: DatasetBuildingConfig) -> None:
    datasets = config.datasets_base
    models = [model.id for model in config.models]

    if not datasets:
        raise ValueError("No datasets configured in config.yaml")
    if not models:
        raise ValueError("No models configured in config.yaml")

    results_dir = Path(config.results_dir)
    ensure_results_dir(results_dir)
    setup_logging(results_dir / "build.log")
    results_path = results_file_path(results_dir)

    if config.overwrite and results_path.exists():
        results_path.unlink()

    existing_results = load_existing_results(results_path)
    existing_keys = {
        TaskKey(r.benchmark, int(r.benchmark_id), r.model)
        for r in existing_results
    }

    examples: list[EvalPlusExample] = []
    for dataset_name in datasets:
        examples.extend(load_evalplus(dataset_name))
    examples_subset = sample_examples(examples, config.dataset_load, seed=42)
    examples_by_key = {
        (ex.benchmark_name, ex.benchmark_id): ex for ex in examples_subset
    }

    pending_keys = build_task_keys(models, examples_subset, existing_keys)
    total_count = len(pending_keys)
    if total_count == 0:
        ui.console.print("All tasks already completed.")
        return

    log.info(
        "Start build: datasets=%s models=%s results_dir=%s done=%d pending=%d",
        datasets, models, str(results_dir), len(existing_keys), total_count,
    )
    ui.console.print(
        f"[cyan]Resuming: {len(existing_keys)} already done, {total_count} pending.[/]"
    )

    counts = count_levels(existing_results)
    model_counts = count_levels_by_model(existing_results)
    pending_write: list[BaseResultRow] = []
    model_handler: OllamaHandler | None = None
    current_model: str | None = None
    skipped = 0

    active_handler: list[OllamaHandler | None] = [None]

    def _close_active_handler() -> None:
        handler = active_handler[0]
        active_handler[0] = None
        if handler is not None:
            try:
                handler.close()
                log.info("Model handler closed.")
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to close model handler: %s", exc)

    def _cleanup() -> None:
        _close_active_handler()

    atexit.register(_cleanup)

    def _signal_handler(signum, frame) -> None:  # noqa: ANN001
        log.warning("Received signal %s; closing model handler.", signum)
        ui.console.print(f"[yellow]Received signal {signum}, closing model...[/]")
        _close_active_handler()
        sys.exit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass

    progress, task_id = ui.build_progress(total_count, counts)

    def _advance(task_key: TaskKey) -> None:
        progress.update(
            task_id,
            advance=1,
            model=task_key.model,
            bench=task_key.benchmark,
            bench_id=str(task_key.benchmark_id),
            c=counts["correct"],
            f=counts["functional"],
            r=counts["runtime"],
            s=counts["syntax"],
        )

    start_time = time.monotonic()
    try:
        with progress:
            for index, task_key in enumerate(pending_keys, start=1):
                try:
                    example = examples_by_key.get((task_key.benchmark, task_key.benchmark_id))
                    if example is None:
                        log.warning("Example not found for %s/%s", task_key.benchmark, task_key.benchmark_id)
                        _advance(task_key)
                        continue

                    ui.console.clear()
                    if task_key.model not in model_counts:
                        model_counts[task_key.model] = {
                            "correct": 0,
                            "functional": 0,
                            "runtime": 0,
                            "syntax": 0,
                            "total": 0,
                        }
                    ui.console.print(ui.render_status_table(model_counts))
                    progress.refresh()

                    if current_model != task_key.model:
                        _close_active_handler()
                        model_handler = None
                        log.info("Loading model %s", task_key.model)
                        model_handler = OllamaHandler(task_key.model)
                        active_handler[0] = model_handler
                        current_model = task_key.model

                    if model_handler is None:
                        raise RuntimeError("Model handler not initialized")

                    code = model_handler.generate_code(example, config.model_temperature)
                    result = run_evalplus_tests(code, example, config.tests_timeout)

                    ui.console.clear()
                    names = {lv.level_name for lv in result.levels}
                    counts["total"] += 1
                    model_counts[task_key.model]["total"] += 1
                    if names == {"correct"}:
                        counts["correct"] += 1
                        model_counts[task_key.model]["correct"] += 1
                    else:
                        for name in names:
                            if name in counts and name != "correct":
                                counts[name] += 1
                                model_counts[task_key.model][name] += 1
                    result_row: BaseResultRow = BaseResultRow(
                        benchmark=task_key.benchmark,
                        benchmark_id=task_key.benchmark_id,
                        task_id=example.task_id,
                        model=task_key.model,
                        levels=[Level(lv.level_name, list(lv.evidences)) for lv in result.levels],
                        tc_ok=result.tc_ok,
                        tc_fail=result.tc_fail,
                        code=code,
                        base_passed=result.base_passed,
                        plus_passed=result.plus_passed,
                        exception_type=result.exception_type,
                        failure_stage=result.failure_stage,
                    )
                    pending_write.append(result_row)

                    log.info(
                        "done %s/%s (%s) -> %s tc_ok=%d tc_fail=%d",
                        task_key.benchmark, task_key.benchmark_id, task_key.model,
                        ", ".join(names), result.tc_ok, result.tc_fail,
                    )

                    if index % config.checkpoint_interval == 0 and pending_write:
                        save_results_jsonl(results_path, pending_write)
                        pending_write = []

                    _advance(task_key)
                except Exception as exc:  # noqa: BLE001
                    log.exception(
                        "Skipped %s/%s (%s)",
                        task_key.benchmark, task_key.benchmark_id, task_key.model,
                    )
                    ui.console.print(
                        f"[yellow]Warning: skipped {task_key.benchmark}/{task_key.benchmark_id} "
                        f"({task_key.model}): {exc}[/]"
                    )
                    skipped += 1
                    _close_active_handler()
                    model_handler = None
                    current_model = None
                    if pending_write:
                        try:
                            save_results_jsonl(results_path, pending_write)
                        except Exception as save_exc:  # noqa: BLE001
                            log.error("Failed to flush pending results: %s", save_exc)
                        pending_write = []
                    _advance(task_key)
    finally:
        if pending_write:
            try:
                save_results_jsonl(results_path, pending_write)
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to flush results: %s", exc)
                ui.console.print(f"[red]Failed to flush results: {exc}[/]")
        _close_active_handler()

    elapsed = time.monotonic() - start_time
    log.info("Build finished in %.1fs. skipped=%d", elapsed, skipped)
    ui.console.print(
        f"Dataset build finished in {elapsed:.1f}s. skipped {skipped} cases."
    )
