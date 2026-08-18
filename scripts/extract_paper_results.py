"""Extrai métricas de avaliação para atualizar a tabela do paper."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

SELECTED_MODEL_IDS = [
    "gpt-oss:20b",
    "gemma3:4b",
    "gemma3:1b",
    "gemma4:e4b",
    "qwen2.5-coder:7b",
    "google/gemma-3-4b-it",
    "data/trained_models/f2743b4e",
]

LEVEL_COLUMNS = [
    ("correct", "Código Correto"),
    ("functional_error", "Erros Funcionais"),
    ("runtime_error", "Erros de Execução"),
    ("syntax_error", "Erros de Sintaxe"),
]

CSV_COLUMNS = [
    "Modelo",
    "Parâmetros",
    "Quantização",
    "Acurácia Total",
    "Identificação de Erros",
    "Código Correto",
    "Erros Funcionais",
    "Erros de Execução",
    "Erros de Sintaxe",
]


def parse_score(value: str) -> tuple[float, int, int]:
    percentage_text, counts_text = value.split("%", maxsplit=1)
    correct_text, total_text = counts_text.strip().removeprefix("(").removesuffix(")").split("/")
    return float(percentage_text), int(correct_text), int(total_text)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_model_metadata(config_path: Path) -> dict[str, dict[str, str]]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    metadata: dict[str, dict[str, str]] = {}
    for group in config.get("models", {}).values():
        for model in group or []:
            model_id = str(model["id"])
            name = str(model.get("name") or model_id).strip()
            size = model.get("size")
            metadata[model_id] = {
                "name": name,
                "parameters": f"{size}B" if size is not None else "",
                "quantization": infer_quantization(name),
            }
    return metadata


def infer_quantization(model_name: str) -> str:
    upper_name = model_name.upper()
    if "QLORA" in upper_name:
        return "4-bit QLoRA"
    if "FP16" in upper_name:
        return "FP16"
    if "QNT" in upper_name:
        return "QNT"
    return "Não informado"


def format_score(correct: int, total: int) -> str:
    percentage = (correct / total * 100) if total else 0.0
    return f"{percentage:.1f}% ({correct}/{total})"


def model_id(record: dict[str, Any]) -> str:
    return str(record.get("model_id") or record.get("model") or "")


def build_rows(records: list[dict[str, Any]], metadata: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    records_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        mid = model_id(record)
        if mid in SELECTED_MODEL_IDS:
            records_by_model[mid].append(record)

    rows: list[dict[str, str]] = []
    for mid in SELECTED_MODEL_IDS:
        model_records = records_by_model[mid]
        model_meta = metadata.get(mid, {})

        total = len(model_records)
        total_correct = sum(
            record.get("expected_level") == record.get("predicted_level")
            for record in model_records
        )

        error_records = [
            record
            for record in model_records
            if record.get("expected_level") != "correct"
        ]
        error_correct = sum(
            record.get("expected_level") == record.get("predicted_level")
            for record in error_records
        )

        row = {
            "Modelo": model_meta.get("name") or mid,
            "Parâmetros": model_meta.get("parameters", ""),
            "Quantização": model_meta.get("quantization", ""),
            "Acurácia Total": format_score(total_correct, total),
            "Identificação de Erros": format_score(error_correct, len(error_records)),
        }

        for level, column in LEVEL_COLUMNS:
            level_records = [
                record
                for record in model_records
                if record.get("expected_level") == level
            ]
            level_correct = sum(
                record.get("predicted_level") == level
                for record in level_records
            )
            row[column] = format_score(level_correct, len(level_records))

        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    totals = [
        (row["Modelo"], *parse_score(row["Acurácia Total"]))
        for row in rows
    ]
    max_n = max((total for _, _, _, total in totals), default=0)
    complete_totals = [
        item
        for item in totals
        if item[3] == max_n
    ]
    best_complete = max(complete_totals, key=lambda item: item[1]) if complete_totals else None
    incomplete = [
        item
        for item in totals
        if item[3] != max_n
    ]

    lines = [
        "Resumo automático para atualização do paper",
        "",
        f"Modelos com avaliação completa: n = {max_n}.",
    ]
    if best_complete:
        name, percentage, correct, total = best_complete
        lines.append(
            f"Melhor acurácia total entre avaliações completas: {name}, "
            f"{percentage:.1f}% ({correct}/{total})."
        )

    for row in rows:
        if row["Modelo"] == "Gemma 3 4B IT (FP16)":
            lines.append(
                "Modelo base Gemma 3 4B IT (FP16): "
                f"{row['Acurácia Total']} de acurácia total; "
                f"{row['Código Correto']} em código correto; "
                f"{row['Erros Funcionais']} em erros funcionais; "
                f"{row['Erros de Execução']} em erros de execução; "
                f"{row['Erros de Sintaxe']} em erros de sintaxe."
            )
        if row["Modelo"] == "Best Trained Model (QLoRA)":
            lines.append(
                "Modelo treinado Best Trained Model (QLoRA): "
                f"{row['Acurácia Total']} de acurácia total no subconjunto já avaliado."
            )

    if incomplete:
        lines.append("")
        lines.append("Atenção: há modelos com avaliação incompleta no arquivo atual:")
        for name, percentage, correct, total in incomplete:
            lines.append(f"- {name}: {percentage:.1f}% ({correct}/{total}).")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera CSV com as principais métricas para atualizar o paper."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--input", type=Path, default=Path("data/results/evaluation_results.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/results/paper_model_metrics.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/results/paper_model_summary.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    metadata = load_model_metadata(args.config)
    rows = build_rows(records, metadata)
    write_csv(args.output, rows)
    write_summary(args.summary_output, rows)
    print(f"CSV salvo em: {args.output}")
    print(f"Resumo salvo em: {args.summary_output}")


if __name__ == "__main__":
    main()
