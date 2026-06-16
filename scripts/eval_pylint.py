"""Avalia Pylint como baseline de análise estática no test split do dataset."""

import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas.dataset import BaseResultRow, JudgeResultRow
from src.dataset.judge_dataset import select_records, build_dataset, stratified_split
from src.dataset.utils import load_jsonl

OUT_PATH = Path("data/results/pylint_results.jsonl")

# Erros do Pylint que mapeiam para syntax_error
SYNTAX_ERROR_CODES = {"E0001"}

# Erros do Pylint que mapeiam para runtime_error
RUNTIME_ERROR_CODES = {
    "E0100",  # init-is-generator
    "E0101",  # return-in-init
    "E0102",  # function-redefined
    "E0103",  # not-in-loop
    "E0104",  # return-outside-function
    "E0105",  # yield-outside-function
    "E0106",  # return-arg-in-generator
    "E0107",  # nonexistent-operator
    "E0108",  # duplicate-argument-name
    "E0110",  # abstract-class-instantiated
    "E0112",  # too-many-star-expressions
    "E0401",  # import-error
    "E0402",  # relative-beyond-top-level
    "E0601",  # used-before-assignment
    "E0602",  # undefined-variable
    "E0603",  # undefined-all-variable
    "E0611",  # no-name-in-module
    "E0633",  # unpacking-non-sequence
    "E1101",  # module-has-no-member
    "E1102",  # not-callable
    "E1111",  # assignment-from-no-return
    "E1120",  # no-value-for-argument
    "E1121",  # too-many-function-args
    "E1123",  # unexpected-keyword-arg
    "E1124",  # redundant-keyword-arg
    "E1125",  # missing-kwoa
    "E1126",  # invalid-sequence-index
    "E1127",  # invalid-slice-index
    "E1128",  # assignment-from-none
    "E1129",  # not-context-manager
    "E1132",  # repeated-keyword
    "E1200",  # bad-format-character
    "E1205",  # logging-too-many-args
    "E1206",  # logging-too-few-args
    "E1300",  # bad-format-string
    "E1301",  # truncated-format-string
    "E1302",  # mixed-format-string
    "E1303",  # format-needs-mapping
    "E1304",  # missing-format-string-key
    "E1305",  # too-many-format-args
    "E1306",  # too-few-format-args
}


def strip_markdown(code: str) -> str:
    """Replica a lógica de _strip_code_fences do pipeline de avaliação:
    só remove as cercas se o código tiver AMBAS abertura e fechamento."""
    fenced = re.match(r"^```(?:python)?\s*(.*?)```\s*$", code, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return code.strip()


def run_pylint(code: str) -> list[dict]:
    clean = strip_markdown(code)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
        f.write(clean)
        path = f.name
    try:
        result = subprocess.run(
            ["pylint", "--output-format=json", "--disable=all", "--enable=E", path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return json.loads(result.stdout) if result.stdout.strip().startswith("[") else []
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    finally:
        os.unlink(path)


def map_to_level(messages: list[dict]) -> str:
    codes = {m["message-id"] for m in messages}
    if codes & SYNTAX_ERROR_CODES:
        return "syntax_error"
    if codes & RUNTIME_ERROR_CODES:
        return "runtime_error"
    return "correct"


def print_report(records: list[dict]) -> None:
    labels   = ["correct", "functional_error", "runtime_error", "syntax_error"]
    expected = [r["expected_level"] for r in records]
    predicted = [r["predicted_level"] for r in records]

    total = len(records)
    correct_total = sum(e == p for e, p in zip(expected, predicted))
    print(f"\nAcurácia total: {correct_total/total:.1%} ({correct_total}/{total})")
    print()
    print(f"{'Classe':<22} {'Acertos/Total':>15} {'Acurácia':>10}")
    print("-" * 50)
    for label in labels:
        idxs = [i for i, e in enumerate(expected) if e == label]
        if not idxs:
            continue
        hits = sum(predicted[i] == label for i in idxs)
        print(f"{label:<22} {hits:>6}/{len(idxs):<8} {hits/len(idxs):>9.1%}")

    print()
    print("Distribuição de predições do Pylint:")
    print(dict(Counter(predicted)))


def main() -> None:
    print("Reconstruindo test split...")
    base_results = load_jsonl("data/results/dataset_base.json", BaseResultRow)
    judge_results = load_jsonl("data/results/dataset_judge.jsonl", JudgeResultRow)
    records_raw = select_records(base_results, judge_results)
    dataset = build_dataset(records_raw)
    dataset = stratified_split(dataset, validation_size=0.1, test_size=0.2, seed=42)
    test = dataset["test"]
    print(f"Test split: {len(test)} amostras")

    results = []
    for i, sample in enumerate(test):
        messages = run_pylint(sample["generated_code"])
        predicted = map_to_level(messages)
        results.append({
            "sample_index": i,
            "expected_level": sample["level"],
            "predicted_level": predicted,
            "correct": predicted == sample["level"],
            "pylint_codes": [m["message-id"] for m in messages],
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(test)} processados...")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResultados salvos em: {OUT_PATH}")

    print_report(results)


if __name__ == "__main__":
    main()
