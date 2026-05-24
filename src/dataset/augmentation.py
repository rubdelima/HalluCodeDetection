from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, cast

from src.core import ui
from src.dataset.load import load_mbpp_split
from src.dataset.types import BaseResultRow, JudgeResultRow
from src.models.ollama_handler import OllamaHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def _load_jsonl(path: Path) -> list[BaseResultRow]:
    if not path.exists():
        return []
    items: list[BaseResultRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(cast(BaseResultRow, json.loads(line)))
    return items


def _append_jsonl(path: Path, items: Iterable[JudgeResultRow]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def _parse_judge_response(content: str) -> dict[str, str]:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "explanation" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    return {"explanation": content.strip()}


def dataset_judge(config: dict[str, object], model_name: str | None = None) -> None:
    build_cfg = cast(dict[str, object], config.get("dataset_build", {}))
    results_dir = Path(str(build_cfg.get("results_dir", "data/results/")))
    results_dir.mkdir(parents=True, exist_ok=True)

    base_path = results_dir / "dataset_base.json"
    judge_path = results_dir / "dataset_judge.jsonl"

    judge_model_value = model_name or build_cfg.get("judge_model")
    if not isinstance(judge_model_value, str) or not judge_model_value:
        raise ValueError("No judge model configured.")
    judge_model = judge_model_value

    base_results = _load_jsonl(base_path)
    if not base_results:
        ui.console.print("No base results found to judge.")
        return

    existing: list[JudgeResultRow] = []
    if judge_path.exists():
        with judge_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                existing.append(cast(JudgeResultRow, json.loads(line)))

    def to_int(value: object | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    existing_keys: set[tuple[str, int, str, str]] = set()
    for existing_item in existing:
        bench_id = to_int(existing_item.get("benchmark_id"))
        if bench_id is None:
            continue
        existing_keys.add(
            (
                existing_item["benchmark"],
                bench_id,
                existing_item["response_model"],
                existing_item["judge_model"],
            )
        )

    mbpp_train = load_mbpp_split("train")
    mbpp_by_id = {ex.benchmak_id: ex for ex in mbpp_train}

    pending: list[tuple[BaseResultRow, int]] = []
    for base_item in base_results:
        bench_id = to_int(base_item.get("benchmark_id"))
        if bench_id is None:
            continue
        if bench_id not in mbpp_by_id:
            continue
        key = (
            base_item["benchmark"],
            bench_id,
            base_item["model"],
            judge_model,
        )
        if key not in existing_keys:
            pending.append((base_item, bench_id))

    if not pending:
        ui.console.print("All items already judged.")
        return

    model_options = build_cfg.get("model_config")
    if not isinstance(model_options, dict):
        model_options = {}
    checkpoint_interval_value = build_cfg.get("checkpoint_interval", 10)
    checkpoint_interval = checkpoint_interval_value if isinstance(checkpoint_interval_value, int) else 10

    handler = OllamaHandler(judge_model)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.fields[model]}"),
        TextColumn("{task.fields[bench]}"),
        TextColumn("{task.fields[bench_id]}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=ui.console,
    )
    task_id = progress.add_task(
        "judge",
        total=len(pending),
        model=judge_model,
        bench="-",
        bench_id="-",
    )

    pending_write: list[JudgeResultRow] = []
    with progress:
        for index, (pending_item, bench_id) in enumerate(pending, start=1):
            example = mbpp_by_id[bench_id]

            try:
                content = handler.generate_judge(
                    example.prompt,
                    str(pending_item["code"]),
                    str(pending_item["level"]),
                    str(pending_item["error"]),
                    model_options,
                    spinner_length=400,
                )
                parsed = _parse_judge_response(content)
                explanation = parsed.get("explanation", "")
            except Exception as exc:  # noqa: BLE001
                explanation = f"Model error: {exc}"

            pending_write.append(
                {
                    "benchmark": pending_item["benchmark"],
                    "benchmark_id": bench_id,
                    "response_model": pending_item["model"],
                    "judge_model": judge_model,
                    "explanation": explanation,
                }
            )

            if index % checkpoint_interval == 0:
                _append_jsonl(judge_path, pending_write)
                pending_write = []

            progress.update(
                task_id,
                advance=1,
                model=judge_model,
                bench=pending_item["benchmark"],
                bench_id=str(bench_id),
            )

    if pending_write:
        _append_jsonl(judge_path, pending_write)

    handler.close()
    ui.console.print("Dataset judge finished.")
