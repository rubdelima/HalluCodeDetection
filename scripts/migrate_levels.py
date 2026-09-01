#!/usr/bin/env python
"""Migra dataset_base.json do formato antigo (level único) para o novo (levels múltiplos).

Antigo:  {"level": {"level_name": "...", "evidences": [...]}}
Novo:    {"levels": [{"level_name": "...", "evidences": [...]}, ...]}

Separa as evidências em 'runtime' e 'functional' para permitir múltiplas classificações.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def evidence_kind(evidence: str) -> str:
    match = re.match(r"^\s*\[(?:base|plus)\]\s*(.*)$", evidence)
    rest = match.group(1) if match else evidence
    return "functional" if rest.startswith("input=") else "runtime"


def migrate_record(record: dict) -> dict:
    if "levels" in record:
        return record

    level = record.get("level")
    if not isinstance(level, dict):
        return record

    level_name = level.get("level_name", "correct")
    evidences = list(level.get("evidences", []))

    if level_name == "syntax":
        levels = [{"level_name": "syntax", "evidences": evidences}]
    elif level_name == "correct":
        levels = [{"level_name": "correct", "evidences": []}]
    elif level_name == "runtime":
        runtime_evidences = [e for e in evidences if evidence_kind(e) == "runtime"]
        functional_evidences = [e for e in evidences if evidence_kind(e) == "functional"]
        levels = []
        if runtime_evidences:
            levels.append({"level_name": "runtime", "evidences": runtime_evidences})
        elif record.get("exception_type"):
            levels.append({
                "level_name": "runtime",
                "evidences": [f"{record['exception_type']} raised during execution"],
            })
        if functional_evidences:
            levels.append({"level_name": "functional", "evidences": functional_evidences})
    else:  # functional
        levels = [{"level_name": "functional", "evidences": evidences}]

    record.pop("level", None)
    record["levels"] = levels
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/results/evalplus/dataset_base.json")
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"Arquivo não encontrado: {path}")
        return

    backup = path.with_suffix(".json.bak2")
    shutil.copy2(path, backup)
    print(f"Backup criado em {backup}")

    migrated: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        migrated.append(migrate_record(json.loads(line)))

    with path.open("w", encoding="utf-8") as handle:
        for record in migrated:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Migrados {len(migrated)} registros.")


if __name__ == "__main__":
    main()
