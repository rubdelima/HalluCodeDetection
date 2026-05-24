from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegistryEntry:
    path: str
    eval_score: float
    run_config: dict[str, object]


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    eval_score: float
    status: str
    run_config: dict[str, object]


def load_registry(path: Path) -> list[RegistryEntry]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    entries: list[RegistryEntry] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            entry_path = item.get("path")
            eval_score = item.get("eval_score")
            run_config = item.get("run_config")
            if isinstance(entry_path, str) and isinstance(eval_score, (int, float)):
                entries.append(
                    RegistryEntry(
                        path=entry_path,
                        eval_score=float(eval_score),
                        run_config=run_config if isinstance(run_config, dict) else {},
                    )
                )
    return entries


def save_registry(path: Path, entries: list[RegistryEntry]) -> None:
    payload = [
        {
            "path": entry.path,
            "eval_score": entry.eval_score,
            "run_config": entry.run_config,
        }
        for entry in entries
    ]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def remove_entry(entry: RegistryEntry) -> None:
    path = Path(entry.path)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def register_model(
    registry_path: Path,
    entry: RegistryEntry,
    max_saved_models: int,
) -> bool:
    entries = load_registry(registry_path)

    if max_saved_models <= 0:
        return False

    if len(entries) < max_saved_models:
        entries.append(entry)
        save_registry(registry_path, entries)
        return True

    worst = min(entries, key=lambda item: item.eval_score)
    if entry.eval_score > worst.eval_score:
        remove_entry(worst)
        entries = [item for item in entries if item.path != worst.path]
        entries.append(entry)
        save_registry(registry_path, entries)
        return True

    return False


def compute_run_id(run_config: dict[str, object]) -> str:
    payload = json.dumps(run_config, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_history(path: Path) -> list[RunRecord]:
    if not path.exists():
        return []
    records: list[RunRecord] = []
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
            run_id = item.get("run_id")
            eval_score = item.get("eval_score")
            status = item.get("status")
            run_config = item.get("run_config")
            if (
                isinstance(run_id, str)
                and isinstance(eval_score, (int, float))
                and isinstance(status, str)
                and isinstance(run_config, dict)
            ):
                records.append(
                    RunRecord(
                        run_id=run_id,
                        eval_score=float(eval_score),
                        status=status,
                        run_config=run_config,
                    )
                )
    return records


def has_run(path: Path, run_id: str) -> bool:
    for record in load_history(path):
        if record.run_id != run_id:
            continue
        validation_total = record.run_config.get("validation_total")
        if record.status in {"saved", "rejected"}:
            if validation_total is None:
                return True
            if isinstance(validation_total, (int, float)) and validation_total > 0:
                return True
        if record.status not in {"saved", "rejected"}:
            return True
    return False


def append_history(path: Path, record: RunRecord) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "run_id": record.run_id,
                    "eval_score": record.eval_score,
                    "status": record.status,
                    "run_config": record.run_config,
                },
                ensure_ascii=True,
            )
            + "\n"
        )
