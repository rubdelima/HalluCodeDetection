from __future__ import annotations

import json
from pathlib import Path

from src.constants.training import TrainingHyperparameters
from src.dataset.utils import load_jsonl
from src.schemas.training import TrainingResult


def hyperparameters_key(hyperparameters: TrainingHyperparameters) -> str:
    base_hyperparameters = TrainingHyperparameters.model_validate(
        hyperparameters.model_dump(mode="json")
    )
    return json.dumps(base_hyperparameters.model_dump(mode="json"), sort_keys=True)


def metric_for_search(result: TrainingResult) -> float:
    return result.test_acc if result.test_acc is not None else result.val_acc


def load_training_results(path: Path) -> list[TrainingResult]:
    return load_jsonl(path, TrainingResult)


def write_training_results(path: Path, results: list[TrainingResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(result.model_dump_json() + "\n")


def latest_results_by_key(results: list[TrainingResult]) -> dict[str, TrainingResult]:
    result_by_key: dict[str, TrainingResult] = {}
    for result in results:
        result_by_key[hyperparameters_key(result)] = result
    return result_by_key


def build_training_status_rows(
    configured_hyperparameters: list[TrainingHyperparameters],
    results: list[TrainingResult],
    known_hyperparameters: list[TrainingHyperparameters] | None = None,
) -> list[dict[str, object]]:
    result_by_key = latest_results_by_key(results)
    known_keys = {
        hyperparameters_key(hyperparameters)
        for hyperparameters in (known_hyperparameters or configured_hyperparameters)
    }
    rows: list[dict[str, object]] = []

    for hyperparameters in configured_hyperparameters:
        result = result_by_key.get(hyperparameters_key(hyperparameters))
        rows.append(
            {
                "status": "done" if result else "pending",
                "hyperparameters": result or hyperparameters,
                "result": result,
            }
        )

    for result in results:
        if hyperparameters_key(result) in known_keys:
            continue
        rows.append(
            {
                "status": "json-only",
                "hyperparameters": result,
                "result": result,
            }
        )

    return sorted(rows, key=_status_sort_key)


def display_hyperparameters_for_run(
    selected_hyperparameters: list[TrainingHyperparameters],
    previous_results: list[TrainingResult],
) -> list[TrainingHyperparameters]:
    displayed: list[TrainingHyperparameters] = []
    displayed_keys: set[str] = set()

    for hyperparameters in selected_hyperparameters:
        displayed.append(hyperparameters)
        displayed_keys.add(hyperparameters_key(hyperparameters))

    for result in previous_results:
        key = hyperparameters_key(result)
        if key in displayed_keys:
            continue
        displayed.append(result)
        displayed_keys.add(key)

    return displayed


def _status_sort_key(row: dict[str, object]) -> tuple[int, float, str]:
    status = str(row.get("status", ""))
    result = row.get("result")
    score = metric_for_search(result) if isinstance(result, TrainingResult) else -1.0
    hyperparameters = row.get("hyperparameters")
    model_id = (
        hyperparameters.model_name.id
        if isinstance(hyperparameters, TrainingHyperparameters)
        else ""
    )
    status_rank = {"done": 0, "json-only": 0, "pending": 1}.get(status, 3)
    return (status_rank, -score, model_id)
