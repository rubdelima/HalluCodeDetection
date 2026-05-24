from __future__ import annotations

import json
from pathlib import Path

from src.core import ui


def _load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    results: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def _model_summary(results: list[dict]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for item in results:
        model = item.get("model", "unknown")
        if model not in summary:
            summary[model] = {
                "correct": 0,
                "functional_error": 0,
                "runtime_error": 0,
                "syntax_error": 0,
            }
        level = item.get("level")
        if level in summary[model]:
            summary[model][level] += 1
    return summary


def _bar(value: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "-" * width
    filled = int((value / total) * width)
    return "[green]" + ("#" * filled) + "[/]" + ("-" * (width - filled))


def view_dataset(config: dict) -> None:
    ui.console.print("Use view_textual.py for the Textual interface.")