from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from rich.table import Table
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from src.core import ui
from src.dataset.types import BaseResultRow, JudgeResultRow
from src.models.base import BaseModelHandler
from src.models.gemma import GemmaHandler
from src.models.ollama_handler import OllamaHandler
from src.training.data import apply_sampling, build_dataset, load_jsonl, select_records, stratified_split


def _to_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _to_int(value: object, default: int) -> int:
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


def _to_str_list(value: object, default: list[str]) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return default


def _extract_json_payload(text: str) -> dict[str, object] | None:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_level(value: object) -> str:
    return str(value).strip().lower()


def _render_summary_table(rows: list[dict[str, object]]) -> Table:
    table = Table(title="Evaluation Summary", show_header=True, header_style="bold")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Correct", style="green")
    table.add_column("Total", style="blue")
    table.add_column("Accuracy", style="blue")
    for row in rows:
        total = 0
        total_raw = row.get("total")
        if isinstance(total_raw, int):
            total = total_raw

        correct = 0
        correct_raw = row.get("correct")
        if isinstance(correct_raw, int):
            correct = correct_raw

        accuracy = (correct / total) if total else 0.0
        table.add_row(
            str(row.get("provider", "-")),
            str(row.get("model", "-")),
            str(correct),
            str(total),
            f"{accuracy:.4f}",
        )
    return table


def _load_handler(kind: str, model: str) -> BaseModelHandler:
    if kind == "ollama":
        return OllamaHandler(model)
    if kind in {"gemma", "lora"}:
        return GemmaHandler(model)
    return GemmaHandler(model)


def _load_existing_results(path: Path) -> dict[tuple[str, str, int], dict[str, object]]:
    if not path.exists():
        return {}

    existing: dict[tuple[str, str, int], dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            model = item.get("model")
            kind = item.get("kind")
            sample_index = item.get("sample_index")
            if isinstance(model, str) and isinstance(kind, str) and isinstance(sample_index, int):
                existing[(kind, model, sample_index)] = item
    return existing


def _evaluation_tasks(config: dict[str, object]) -> list[tuple[str, str]]:
    evaluation_cfg = cast(dict[str, object], config.get("evaluation", {}))
    tasks: list[tuple[str, str]] = []
    tasks.extend(("ollama", model) for model in _to_str_list(evaluation_cfg.get("ollama_models", []), []))
    tasks.extend(("gemma", model) for model in _to_str_list(evaluation_cfg.get("gemma_models", []), []))
    tasks.extend(("lora", model) for model in _to_str_list(evaluation_cfg.get("lora_models", []), []))
    return tasks


def _task_group(kind: str) -> str:
    if kind == "ollama":
        return "ollama"
    if kind == "lora":
        return "lora"
    return "gemma"


def _append_jsonl(path: Path, items: list[dict[str, object]]) -> None:
    if not items:
        return
    with path.open("a", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def _pending_sample_indices(
    kind: str,
    model: str,
    dataset: list[dict[str, object]],
    existing_results: dict[tuple[str, str, int], dict[str, object]],
) -> list[int]:
    pending_indices: list[int] = []
    for sample_index, _sample in enumerate(dataset):
        if (kind, model, sample_index) not in existing_results:
            pending_indices.append(sample_index)
    return pending_indices


def evaluate_models(config: dict[str, object]) -> None:
    build_cfg = cast(dict[str, object], config.get("dataset_build", {}))

    results_dir = Path(str(build_cfg.get("results_dir", "data/results/")))
    results_dir.mkdir(parents=True, exist_ok=True)

    base_path = results_dir / "dataset_base.json"
    judge_path = results_dir / "dataset_judge.jsonl"
    output_path = results_dir / "evaluation_results.jsonl"

    if not base_path.exists():
        ui.console.print("No base results found. Run --build_dataset first.")
        return
    if not judge_path.exists():
        ui.console.print("No judge results found. Run --dataset_judge first.")
        return

    dataset_load = _to_float(config.get("dataset_load", 1.0), 1.0)
    correct_size = _to_float(config.get("correct_size", 1.0), 1.0)
    random_seed = _to_int(config.get("random_seed", 42), 42)
    validation_size = _to_float(config.get("validation_size", 0.1), 0.1)
    test_size = _to_float(config.get("test_size", 0.2), 0.2)
    spinner_length = _to_int(build_cfg.get("spinner_length", 600), 600)
    generation_options = cast(dict[str, object], build_cfg.get("model_config", {}))
    generation_options = {key: value for key, value in generation_options.items() if key != "temperature"}
    checkpoint_interval = _to_int(cast(dict[str, object], config.get("evaluation", {})).get("checkpoint_interval", 10), 10)
    checkpoint_interval = max(1, checkpoint_interval)

    base_results = [cast(BaseResultRow, item) for item in load_jsonl(str(base_path))]
    judge_results = [cast(JudgeResultRow, item) for item in load_jsonl(str(judge_path))]

    if not base_results:
        ui.console.print("No base results found to evaluate.")
        return
    if not judge_results:
        ui.console.print("No judge results found to evaluate.")
        return

    records = select_records(base_results, judge_results, None)
    records = apply_sampling(records, dataset_load, correct_size, random_seed)
    if not records:
        ui.console.print("No evaluation records after filtering.")
        return

    split_raw = stratified_split(build_dataset(records), validation_size, test_size, random_seed)
    test_dataset = split_raw["test"]

    tasks = _evaluation_tasks(config)
    if not tasks:
        ui.console.print("No evaluation models configured.")
        return

    existing_results = _load_existing_results(output_path)

    pending_tasks: list[tuple[str, str, list[int]]] = []
    for kind, model in tasks:
        pending_indices = _pending_sample_indices(kind, model, list(test_dataset), existing_results)
        if pending_indices:
            pending_tasks.append((kind, model, pending_indices))
        else:
            ui.console.print(f"Skipping {model} ({_task_group(kind)}): all samples already evaluated.")

    if not pending_tasks:
        ui.console.print("No pending evaluation records for any configured model.")
        return

    model_counts: dict[str, dict[str, int]] = {}
    overall_counts = {
        "correct": 0,
        "functional_error": 0,
        "runtime_error": 0,
        "syntax_error": 0,
    }
    current_handler: BaseModelHandler | None = None
    current_kind: str | None = None
    current_model: str | None = None
    summary_rows: list[dict[str, object]] = []

    total_samples = sum(len(pending_indices) for _kind, _model, pending_indices in pending_tasks)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.fields[provider]}"),
        TextColumn("{task.fields[model]}"),
        TextColumn("{task.fields[sample]}"),
        TextColumn("C:{task.fields[c]}"),
        TextColumn("F:{task.fields[f]}"),
        TextColumn("R:{task.fields[r]}"),
        TextColumn("S:{task.fields[s]}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=ui.console,
    )
    task_id = progress.add_task(
        "evaluation",
        total=total_samples,
        provider="-",
        model="-",
        sample="-",
        c=overall_counts["correct"],
        f=overall_counts["functional_error"],
        r=overall_counts["runtime_error"],
        s=overall_counts["syntax_error"],
    )

    with progress:
        for kind, model, pending_indices in pending_tasks:
            if current_handler is not None and (current_kind != kind or current_model != model):
                current_handler.close()
                current_handler = None

            if current_handler is None:
                current_handler = _load_handler(kind, model)
                current_kind = kind
                current_model = model

            model_counts.setdefault(
                model,
                {
                    "correct": 0,
                    "functional_error": 0,
                    "runtime_error": 0,
                    "syntax_error": 0,
                },
            )

            correct = 0
            total = 0
            pending_write: list[dict[str, object]] = []
            for sample_index in pending_indices:
                sample = test_dataset[sample_index]
                sample_label = f"#{sample_index + 1}/{len(test_dataset)}"
                expected_level = _normalize_level(sample["level"])
                try:
                    response = current_handler.analyze_hallucination(
                        str(sample["problem_description"]),
                        str(sample["generated_code"]),
                        generation_options,
                        spinner_length,
                    )
                    payload = _extract_json_payload(response)
                    predicted_level = _normalize_level(payload.get("level") if payload else "")
                    if predicted_level in model_counts[model]:
                        model_counts[model][predicted_level] += 1
                        overall_counts[predicted_level] += 1
                    total += 1
                    if predicted_level == expected_level:
                        correct += 1
                    record = {
                        "model": model,
                        "kind": _task_group(kind),
                        "sample_index": sample_index,
                        "expected_level": expected_level,
                        "predicted_level": predicted_level,
                        "raw_response": response,
                        "parsed_response": payload,
                        "correct": predicted_level == expected_level,
                    }
                except Exception as exc:  # noqa: BLE001
                    total += 1
                    ui.console.print(f"Evaluation failed for {model}: {exc}")
                    record = {
                        "model": model,
                        "kind": _task_group(kind),
                        "sample_index": sample_index,
                        "expected_level": expected_level,
                        "predicted_level": "error",
                        "raw_response": f"Model error: {exc}",
                        "parsed_response": None,
                        "correct": False,
                    }

                pending_write.append(record)
                if len(pending_write) >= checkpoint_interval:
                    _append_jsonl(output_path, pending_write)
                    for item in pending_write:
                        item_model = item.get("model")
                        item_kind = item.get("kind")
                        item_sample_index = item.get("sample_index")
                        if isinstance(item_model, str) and isinstance(item_kind, str) and isinstance(item_sample_index, int):
                            existing_results[(item_kind, item_model, item_sample_index)] = item
                    ui.console.print(f"Saved {len(pending_write)} evaluation rows to {output_path}")
                    pending_write = []

                progress.update(
                    task_id,
                    advance=1,
                    provider=_task_group(kind),
                    model=model,
                    sample=sample_label,
                    c=overall_counts["correct"],
                    f=overall_counts["functional_error"],
                    r=overall_counts["runtime_error"],
                    s=overall_counts["syntax_error"],
                )

            _append_jsonl(output_path, pending_write)
            for item in pending_write:
                item_model = item.get("model")
                item_kind = item.get("kind")
                item_sample_index = item.get("sample_index")
                if isinstance(item_model, str) and isinstance(item_kind, str) and isinstance(item_sample_index, int):
                    existing_results[(item_kind, item_model, item_sample_index)] = item
            if pending_write:
                ui.console.print(f"Saved {len(pending_write)} evaluation rows to {output_path}")

            accuracy = (correct / total) if total else 0.0
            summary_rows.append(
                {
                    "provider": _task_group(kind),
                    "model": model,
                    "correct": correct,
                    "total": total,
                }
            )

            ui.console.clear()
            ui.console.print(_render_summary_table(summary_rows))

    if current_handler is not None:
        current_handler.close()

    ui.console.print(_render_summary_table(summary_rows))
