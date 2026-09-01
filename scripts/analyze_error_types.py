#!/usr/bin/env python
"""Analisa casos com múltiplos tipos de erro e acertos parciais no dataset_base.json."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from rich.console import Console

console = Console()


def load_results(results_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(results_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def level_names(record: dict) -> list[str]:
    if "levels" in record:
        return [lv["level_name"] for lv in record["levels"]]
    return [record.get("level", {}).get("level_name", "correct")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/results/evalplus/dataset_base.json")
    args = parser.parse_args()

    rows = load_results(args.results)

    # 1) Acerto parcial: alguns testes passaram e outros falharam
    partial = [r for r in rows if r["tc_ok"] > 0 and r["tc_fail"] > 0]
    partial_by_level = Counter(
        "mixed" if set(level_names(r)) >= {"runtime", "functional"}
        else level_names(r)[0]
        for r in partial
    )

    # 2) Mais de um tipo de erro: 'runtime' e 'functional' no mesmo código
    mixed = [
        r for r in rows
        if set(level_names(r)) >= {"runtime", "functional"}
    ]

    console.print(f"[bold]Total de registros:[/] {len(rows)}\n")

    console.print("[bold]1) Acerto parcial (tc_ok > 0 e tc_fail > 0):[/]")
    console.print(f"   Casos: {len(partial)}  ({len(partial)/len(rows)*100:.1f}% do total)")
    for level, count in sorted(partial_by_level.items(), key=lambda kv: -kv[1]):
        console.print(f"     {level:12s} {count}")

    console.print("\n[bold]2) Mais de um tipo de erro (runtime + functional no mesmo código):[/]")
    console.print(f"   Casos: {len(mixed)}  ({len(mixed)/len(rows)*100:.1f}% do total)")

    if mixed:
        example = mixed[0]
        console.print("\n[bold]Exemplo de caso mixed:[/]")
        console.print(f"   {example['benchmark']}/{example['benchmark_id']} ({example['model']})")
        console.print(f"   levels={level_names(example)} tc_ok={example['tc_ok']} tc_fail={example['tc_fail']}")
        for lv in example.get("levels", [example.get("level", {})]):
            console.print(f"   - {lv['level_name']}: {len(lv['evidences'])} evidências")
            for ev in lv["evidences"][:3]:
                console.print(f"       {ev[:100]}")


if __name__ == "__main__":
    main()
