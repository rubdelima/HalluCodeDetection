#!/usr/bin/env python
"""Analisa o dataset base da Fase 1 (dataset_base.json) e gera um resumo + ranking."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def load_config_models(config_path: str) -> list[str]:
    import yaml

    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return list(cfg.get("dataset_building", {}).get("models", []))


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
    parser = argparse.ArgumentParser(description="Analisa dataset_base.json")
    parser.add_argument("--results", default="data/results/evalplus/dataset_base.json")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    rows = load_results(args.results)
    models = load_config_models(args.config)

    keys = {(r["benchmark"], r["benchmark_id"], r["model"]) for r in rows}
    tasks_by_benchmark: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        tasks_by_benchmark[r["benchmark"]].add(r["benchmark_id"])

    expected = sum(len(ids) * len(models) for ids in tasks_by_benchmark.values())
    done = len(keys)
    missing = expected - done

    console.print(f"[bold]Arquivo:[/] {args.results}")
    console.print(f"[bold]Registros:[/] {len(rows)} | [bold]Únicos:[/] {done} | [bold]Esperado:[/] {expected} | [bold]Faltando:[/] {missing}")

    per_model: dict[str, Counter] = {}
    for m in models:
        per_model[m] = Counter()
    for r in rows:
        names = level_names(r)
        per_model[r["model"]]["total"] += 1
        if names == ["correct"]:
            per_model[r["model"]]["correct"] += 1
        else:
            for n in names:
                if n != "correct":
                    per_model[r["model"]][n] += 1

    table = Table(title="Ranking por modelo (Pass@1 = correct/total)")
    table.add_column("#", justify="right")
    table.add_column("Modelo")
    table.add_column("Concluídos", justify="right")
    table.add_column("Correct", justify="right", style="green")
    table.add_column("Functional", justify="right", style="yellow")
    table.add_column("Runtime", justify="right", style="orange3")
    table.add_column("Syntax", justify="right", style="red")
    table.add_column("Pass@1", justify="right", style="blue")

    ranked = sorted(
        models,
        key=lambda m: (
            per_model[m]["correct"] / per_model[m]["total"]
            if per_model[m]["total"] else 0.0
        ),
        reverse=True,
    )

    for pos, m in enumerate(ranked, start=1):
        c = per_model[m]
        total = c["total"]
        pass_rate = (c["correct"] / total * 100) if total else 0.0
        table.add_row(
            str(pos),
            m,
            str(total),
            str(c["correct"]),
            str(c["functional"]),
            str(c["runtime"]),
            str(c["syntax"]),
            f"{pass_rate:.1f}%",
        )
    console.print(table)

    console.print("[bold]Por benchmark:[/]")
    for bench, ids in sorted(tasks_by_benchmark.items()):
        bench_rows = [r for r in rows if r["benchmark"] == bench]
        corr = sum(1 for r in bench_rows if level_names(r) == ["correct"])
        console.print(
            f"  {bench}: {len(ids)} tarefas | corretas={corr}/{len(bench_rows)} "
            f"({corr/len(bench_rows)*100:.1f}%)"
        )

    base_pass_plus_fail = sum(
        1 for r in rows if r.get("base_passed") is True and r.get("plus_passed") is False
    )
    console.print(f"\n[bold]Subgrupo interessante (base passou, plus falhou):[/] {base_pass_plus_fail}")

    incomplete = [m for m in models if per_model[m]["total"] < sum(len(v) for v in tasks_by_benchmark.values())]
    if incomplete:
        console.print("\n[bold][yellow]Modelos incompletos:[/][/]")
        for m in incomplete:
            need = sum(len(v) for v in tasks_by_benchmark.values())
            console.print(f"  {m}: {per_model[m]['total']}/{need}")
    else:
        console.print("\n[bold][green]Todos os modelos concluíram todas as tarefas.[/][/]")


if __name__ == "__main__":
    main()
