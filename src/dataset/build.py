from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

from src.core import ui
from src.dataset.load import MBPPExample, load_mbpp_split, sample_examples
from src.dataset.types import BaseResultRow
from src.evaluations.tests import run_tests
from src.models.ollama_handler import OllamaHandler
from src.constants.dataset import DatasetBuildingConfig


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
            results.append(cast(BaseResultRow, json.loads(line)))
    return results


def save_results_jsonl(path: Path, results: Iterable[BaseResultRow], overwrite: bool) -> None:
    if overwrite and path.exists():
        path.unlink()
    with path.open("a", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def build_task_keys(
    models: list[str],
    examples: Iterable[MBPPExample],
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
        "functional_error": 0,
        "runtime_error": 0,
        "syntax_error": 0,
    }
    for item in results:
        level = item["level"]
        if level in counts:
            counts[level] += 1
    return counts


def count_levels_by_model(results: Iterable[BaseResultRow]) -> dict[str, dict[str, int]]:
    model_counts: dict[str, dict[str, int]] = {}
    for item in results:
        model = item["model"]
        if model not in model_counts:
            model_counts[model] = {
                "correct": 0,
                "functional_error": 0,
                "runtime_error": 0,
                "syntax_error": 0,
            }
        level = item["level"]
        if level in model_counts[model]:
            model_counts[model][level] += 1
    return model_counts


def build_dataset(config: DatasetBuildingConfig) -> None:
    
    build_cfg = cast(dict[str, object], config.get("dataset_build", {}))

    def to_float(value: object, default: float) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def to_int(value: object, default: int) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    def to_str_list(value: object, default: list[str]) -> list[str]:
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        return default

    dataset_fraction = to_float(build_cfg.get("dataset_load", 1.0), 1.0)
    dataset_base = to_str_list(build_cfg.get("dataset_base", ["mbpp"]), ["mbpp"])
    models = to_str_list(build_cfg.get("models", []), [])
    spinner_length = to_int(build_cfg.get("spinner_length", 600), 600)
    timeout_seconds = to_int(build_cfg.get("tests_timeout", 30), 30)
    checkpoint_interval = to_int(build_cfg.get("checkpoint_interval", 10), 10)
    overwrite_results = bool(build_cfg.get("overwrite_results", False))
    results_dir = Path(str(build_cfg.get("results_dir", "data/")))
    model_options_value = build_cfg.get("model_config")
    if not isinstance(model_options_value, dict):
        model_options_value = {}
    model_options = cast(dict[str, object], model_options_value)

    if "mbpp" not in dataset_base:
        raise ValueError("Only mbpp is supported in this phase.")
    if not models:
        raise ValueError("No models configured in config.yaml")

    ensure_results_dir(results_dir)
    results_path = results_file_path(results_dir)
    existing_results = [] if overwrite_results else load_existing_results(results_path)

    existing_keys = {
        TaskKey(r["benchmark"], int(r["benchmark_id"]), r["model"])
        for r in existing_results
    }

    mbpp_train = load_mbpp_split("train")
    mbpp_subset = sample_examples(mbpp_train, dataset_fraction, seed=42)
    examples_by_id = {ex.benchmark_id: ex for ex in mbpp_subset}

    pending_keys = build_task_keys(models, mbpp_subset, existing_keys)
    total_count = len(pending_keys)
    if total_count == 0:
        ui.console.print("All tasks already completed.")
        return

    counts = count_levels(existing_results)
    model_counts = count_levels_by_model(existing_results)
    pending_write: list[BaseResultRow] = []
    model_handler: OllamaHandler | None = None
    current_model: str | None = None

    progress, task_id = ui.build_progress(total_count, counts)

    start_time = time.monotonic()
    with progress:
        for index, task_key in enumerate(pending_keys, start=1):
            example = examples_by_id.get(task_key.benchmark_id)
            if example is None:
                continue

            ui.console.clear()
            if task_key.model not in model_counts:
                model_counts[task_key.model] = {
                    "correct": 0,
                    "functional_error": 0,
                    "runtime_error": 0,
                    "syntax_error": 0,
                }
            ui.console.print(ui.render_status_table(model_counts))
            progress.refresh()

            try:
                if current_model != task_key.model:
                    if model_handler is not None:
                        model_handler.close()
                    model_handler = OllamaHandler(task_key.model)
                    current_model = task_key.model
                
                if model_handler is None:
                    raise RuntimeError("Model handler not initialized")
                code = model_handler.generate_code(example, model_options, spinner_length)
                
                level, error_text = run_tests(
                    code,
                    example.tests,
                    timeout_seconds,
                    example.function_signature,
                )
            
            except Exception as exc:  # noqa: BLE001
                code = ""
                level = "runtime_error"
                error_text = f"Model error: {exc}"

            ui.console.clear()

            counts[level] += 1
            model_counts[task_key.model][level] += 1
            result: BaseResultRow = {
                "benchmark": task_key.benchmark,
                "benchmark_id": task_key.benchmark_id,
                "model": task_key.model,
                "level": level,
                "code": code,
                "error": error_text,
            }
            pending_write.append(result)

            if index % checkpoint_interval == 0:
                save_results_jsonl(results_path, pending_write, overwrite=False)
                pending_write = []
            progress.update(
                task_id,
                advance=1,
                model=task_key.model,
                bench=task_key.benchmark,
                bench_id=str(task_key.benchmark_id),
                c=counts["correct"],
                f=counts["functional_error"],
                r=counts["runtime_error"],
                s=counts["syntax_error"],
            )

    if pending_write:
        save_results_jsonl(results_path, pending_write, overwrite=False)
    if model_handler is not None:
        model_handler.close()

    elapsed = time.monotonic() - start_time
    ui.console.print(f"Dataset build finished in {elapsed:.1f}s.")
