"""Reproducible statistical analysis for the SIGSOFT revision.

The script reads immutable experiment outputs, validates their pairing, and
writes all derived statistics to ``artifacts/statistics``.  It never edits the
raw JSONL files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
import transformers
from datasets import Dataset
from scipy.stats import binomtest, chi2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import HalluCodeDetectionConfig
from src.dataset.judge_dataset import grouped_split

LEVELS = ("correct", "functional", "runtime", "syntax")
SEED = 20260920
N_RESAMPLES = 10_000

RQ1_LOCAL = Path("data/results/evalplus/dataset_base.json")
RQ1_API = Path("data/results/evalplus/phase1_opencode_baselines/dataset_base.json")
EVALUATION = Path("data/results/evalplus/evaluation_results.jsonl")
PHASE5 = Path("data/results/evalplus/phase5/static_analysis_results.jsonl")
PHASE6_API = Path("data/results/evalplus/phase6/generation_with_tools.jsonl")
PHASE7_API = Path("data/results/evalplus/phase7/classification_with_tools.jsonl")
PHASE6_LOCAL = Path("/mnt/ssd/HalluCode/results/phase6_local_tools/generation_with_tools.jsonl")
PHASE7_LOCAL = Path("/mnt/ssd/HalluCode/results/phase7_local_tools/classification_with_tools.jsonl")

MODEL_NAMES = {
    "gpt-oss:20b": "GPT-OSS 20B",
    "gemma3:4b": "Gemma 3 4B",
    "qwen2.5-coder:7b": "Qwen 2.5 Coder 7B",
    "gemma4:31b": "Gemma 4 31B",
    "muse-glimmer:30b": "Muse Glimmer 30B",
    "qwen3.8:27b": "Qwen 3.8 27B",
    "devstral-small-2:24b": "Devstral Small 2 24B",
    "qwen2.5-coder:32b": "Qwen 2.5 Coder 32B",
    "qwen3-coder:30b": "Qwen 3 Coder 30B",
    "glm-4.7-flash:latest": "GLM 4.7 Flash 30B",
    "ornith-1.5:35b": "Ornith 1.5 35B",
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "mimo-v2.5": "MiMo V2.5",
    "google/gemma-4-E4B-it": "Gemma 4 E4B IT",
    "trained/gemma-4-e4b-best": "Best Acc",
    "trained/gemma-4-e4b-weighted-macro-f1-best": "Weighted MF1",
    "trained/gemma-4-e4b-balanced-macro-f1-best": "Balanced MF1",
    "trained/gemma-4-e4b-balanced-accuracy-retrained": "Balanced Acc",
}


@dataclass(frozen=True)
class TestMetadata:
    task_ids: list[str]
    sample_ids: list[set[str]]
    response_models: list[set[str]]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    b_only = int(np.sum((a == 0) & (b == 1)))
    a_only = int(np.sum((a == 1) & (b == 0)))
    discordant = a_only + b_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue
    )
    return a_only, b_only, p_value


def cochran_q(matrix: np.ndarray) -> tuple[float, int, float]:
    n_models = matrix.shape[1]
    column_sums = matrix.sum(axis=0)
    row_sums = matrix.sum(axis=1)
    total = column_sums.sum()
    denominator = n_models * total - np.square(row_sums).sum()
    if denominator == 0:
        return 0.0, n_models - 1, 1.0
    statistic = (n_models - 1) * (
        n_models * np.square(column_sums).sum() - total * total
    ) / denominator
    return float(statistic), n_models - 1, float(chi2.sf(statistic, n_models - 1))


def paired_bootstrap_binary(
    matrix: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    indices = rng.integers(0, matrix.shape[0], size=(N_RESAMPLES, matrix.shape[0]))
    return matrix[indices].mean(axis=1)


def level_set(row: dict[str, Any], field: str = "levels") -> set[str]:
    values = row.get(field, [])
    return {
        str(value.get("level_name")) if isinstance(value, dict) else str(value)
        for value in values
    }


def rq1_analysis(output: Path, rng: np.random.Generator) -> None:
    rows = load_jsonl(RQ1_LOCAL) + load_jsonl(RQ1_API)
    keys: dict[str, dict[tuple[str, int], int]] = defaultdict(dict)
    for row in rows:
        model = str(row["model"])
        key = (str(row["benchmark"]), int(row["benchmark_id"]))
        if key in keys[model]:
            raise ValueError(f"Duplicate Phase-1 model/task row: {model}/{key}")
        keys[model][key] = int(level_set(row) == {"correct"})
    if len(keys) != 13 or any(len(items) != 563 for items in keys.values()):
        raise ValueError("RQ1 requires 13 models with exactly 563 unique tasks each.")
    common = sorted(set.intersection(*(set(items) for items in keys.values())))
    if len(common) != 563:
        raise ValueError("RQ1 model conditions are not aligned on the same 563 tasks.")

    models = sorted(keys)
    matrix = np.array([[keys[model][task] for model in models] for task in common], dtype=np.int8)
    bootstrap = paired_bootstrap_binary(matrix, rng)
    estimates: list[dict[str, Any]] = []
    for index, model in enumerate(models):
        successes = int(matrix[:, index].sum())
        low, high = wilson(successes, len(common))
        estimates.append({
            "model_id": model,
            "model": MODEL_NAMES.get(model, model),
            "correct": successes,
            "n_tasks": len(common),
            "proportion": successes / len(common),
            "ci95_low_wilson": low,
            "ci95_high_wilson": high,
            "unit": "EvalPlus task",
        })
    write_csv(output / "rq1_model_estimates.csv", estimates)

    statistic, df, global_p = cochran_q(matrix)
    write_csv(output / "rq1_global_test.csv", [{
        "test": "Cochran Q",
        "statistic": statistic,
        "df": df,
        "p_value": global_p,
        "n_tasks": len(common),
        "conditions": len(models),
        "alternative": "two-sided omnibus",
        "unit": "EvalPlus task",
    }])

    pairwise: list[dict[str, Any]] = []
    for left in range(len(models)):
        for right in range(left + 1, len(models)):
            left_only, right_only, p_value = exact_mcnemar(matrix[:, left], matrix[:, right])
            deltas = bootstrap[:, left] - bootstrap[:, right]
            pairwise.append({
                "model_a": MODEL_NAMES.get(models[left], models[left]),
                "model_b": MODEL_NAMES.get(models[right], models[right]),
                "estimate_a": matrix[:, left].mean(),
                "estimate_b": matrix[:, right].mean(),
                "paired_difference_pp": (matrix[:, left].mean() - matrix[:, right].mean()) * 100,
                "ci95_low_pp": np.percentile(deltas, 2.5) * 100,
                "ci95_high_pp": np.percentile(deltas, 97.5) * 100,
                "a_correct_b_wrong": left_only,
                "a_wrong_b_correct": right_only,
                "p_value_raw": p_value,
                "n_tasks": len(common),
                "test": "exact McNemar two-sided",
                "family": "all 78 Phase-1 model pairs",
            })
    adjusted = holm_adjust([float(row["p_value_raw"]) for row in pairwise])
    for row, value in zip(pairwise, adjusted):
        row["p_value_holm"] = value
    write_csv(output / "rq1_pairwise_mcnemar.csv", pairwise)


def latest_evaluations() -> dict[str, dict[int, dict[str, Any]]]:
    latest: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(EVALUATION):
        latest[str(row["model_id"])][int(row["sample_index"])] = row
    return latest


def build_test_metadata(config_path: str) -> TestMetadata:
    config = HalluCodeDetectionConfig(config_path)
    base_rows = load_jsonl(RQ1_LOCAL)
    judge_rows = load_jsonl(
        Path(config.dataset_building_config.judge_results_dir) / "dataset_judge.jsonl"
    )
    base_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in base_rows:
        base_by_key[(str(row["benchmark"]), int(row["benchmark_id"]), str(row["model"]))] = row

    records: list[dict[str, Any]] = []
    for judge in judge_rows:
        key = (
            str(judge["benchmark"]),
            int(judge["benchmark_id"]),
            str(judge["response_model"]),
        )
        base = base_by_key.get(key)
        if base is None:
            continue
        labels = list(level_set(base))
        primary = str(base.get("levels", [{}])[0].get("level_name", "correct"))
        records.append({
            "benchmark": str(base["benchmark"]),
            "benchmark_id": int(base["benchmark_id"]),
            "task_id": str(base.get("task_id") or f"{base['benchmark']}/{base['benchmark_id']}"),
            "level": primary,
            "levels": labels,
            "response_model": str(base["model"]),
        })

    rng = random.Random(config.dataset_config.random_seed)
    correct = [row for row in records if row["level"] == "correct"]
    other = [row for row in records if row["level"] != "correct"]
    rng.shuffle(correct)
    selected = other + correct
    rng.shuffle(selected)
    split = grouped_split(
        Dataset.from_list(selected),
        validation_size=config.dataset_config.validation_size,
        test_size=config.dataset_config.test_size,
        seed=config.dataset_config.random_seed,
    )
    test = split["test"]
    task_ids: list[str] = []
    sample_ids: list[set[str]] = []
    response_models: list[set[str]] = []
    for row in test:
        task_id = str(row["task_id"])
        benchmark = str(row["benchmark"])
        benchmark_id = int(row["benchmark_id"])
        response_model = str(row["response_model"])
        task_ids.append(task_id)
        response_models.append({response_model})
        sample_ids.append({f"{benchmark}:{benchmark_id}:{response_model}"})
    return TestMetadata(task_ids, sample_ids, response_models)


def sample_stats(expected: Iterable[str], predicted: Iterable[str]) -> np.ndarray:
    expected_set, predicted_set = set(expected), set(predicted)
    values = [1.0, float(expected_set == predicted_set and bool(predicted_set))]
    for level in LEVELS:
        values.extend([
            float(level in expected_set and level in predicted_set),
            float(level not in expected_set and level in predicted_set),
            float(level in expected_set and level not in predicted_set),
        ])
    return np.array(values, dtype=float)


def metrics_from_stats(stats: np.ndarray) -> np.ndarray:
    one_dimensional = stats.ndim == 1
    values = stats.reshape(1, -1) if one_dimensional else stats
    total = values[:, 0]
    accuracy = np.divide(values[:, 1], total, out=np.zeros_like(total), where=total != 0)
    precisions, recalls, f1s = [], [], []
    for index in range(len(LEVELS)):
        tp, fp, fn = values[:, 2 + index * 3 : 5 + index * 3].T
        precisions.append(np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0))
        recalls.append(np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0))
        f1s.append(np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) != 0))
    result = np.column_stack([
        accuracy,
        np.mean(precisions, axis=0),
        np.mean(f1s, axis=0),
        np.mean(recalls, axis=0),
        *recalls,
    ])
    return result[0] if one_dimensional else result


METRIC_NAMES = (
    "accuracy", "macro_precision", "macro_f1", "macro_recall",
    "correct_recall", "functional_recall", "runtime_recall", "syntax_recall",
)


def grouped_stats(rows: dict[int, dict[str, Any]], metadata: TestMetadata) -> tuple[list[str], np.ndarray]:
    by_task: dict[str, np.ndarray] = {}
    for index, row in rows.items():
        if row.get("error") or index >= len(metadata.task_ids):
            continue
        task = metadata.task_ids[index]
        by_task.setdefault(task, np.zeros(14, dtype=float))
        by_task[task] += sample_stats(row.get("expected_levels", []), row.get("predicted_levels", []))
    tasks = sorted(by_task)
    return tasks, np.stack([by_task[task] for task in tasks])


def bootstrap_grouped(stats: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    weights = rng.multinomial(stats.shape[0], np.full(stats.shape[0], 1 / stats.shape[0]), N_RESAMPLES)
    return metrics_from_stats(weights @ stats)


def rq2_analysis(output: Path, metadata: TestMetadata, rng: np.random.Generator) -> None:
    latest = latest_evaluations()
    metric_rows: list[dict[str, Any]] = []
    for model, model_rows in sorted(latest.items()):
        tasks, stats = grouped_stats(model_rows, metadata)
        if not tasks:
            continue
        estimate = metrics_from_stats(stats.sum(axis=0))
        bootstrap = bootstrap_grouped(stats, rng)
        row: dict[str, Any] = {
            "model_id": model,
            "model": MODEL_NAMES.get(model, model),
            "n_rows": int(stats[:, 0].sum()),
            "n_tasks": len(tasks),
            "unit": "programming task (cluster bootstrap)",
        }
        for index, name in enumerate(METRIC_NAMES):
            row[name] = estimate[index]
            row[f"{name}_ci95_low"] = np.percentile(bootstrap[:, index], 2.5)
            row[f"{name}_ci95_high"] = np.percentile(bootstrap[:, index], 97.5)
        metric_rows.append(row)
    write_csv(output / "rq2_detection_metrics_ci.csv", metric_rows)

    self_rows: list[dict[str, Any]] = []
    for model, model_rows in sorted(latest.items()):
        if model not in MODEL_NAMES:
            continue
        selected = [
            row for index, row in model_rows.items()
            if not row.get("error")
            and index < len(metadata.response_models)
            and model in metadata.response_models[index]
            and set(row.get("expected_levels", [])) != {"correct"}
        ]
        if not selected:
            continue
        successes = sum(bool(row.get("correct")) for row in selected)
        low, high = wilson(successes, len(selected))
        self_rows.append({
            "model_id": model,
            "model": MODEL_NAMES.get(model, model),
            "detected": successes,
            "self_errors": len(selected),
            "proportion": successes / len(selected),
            "ci95_low_wilson": low,
            "ci95_high_wilson": high,
            **{
                field: value
                for level in ("functional", "runtime", "syntax")
                for field, value in (
                    (f"{level}_detected", sum(level in row.get("predicted_levels", []) for row in selected if level in row.get("expected_levels", []))),
                    (f"{level}_total", sum(level in row.get("expected_levels", []) for row in selected)),
                )
            },
        })
    write_csv(output / "rq2_self_detection_wilson.csv", self_rows)


def paired_metric_comparison(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    metadata: TestMetadata,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    common = sorted(
        index for index in set(left) & set(right)
        if not left[index].get("error") and not right[index].get("error")
    )
    tasks = sorted({metadata.task_ids[index] for index in common})
    task_index = {task: index for index, task in enumerate(tasks)}
    left_stats = np.zeros((len(tasks), 14))
    right_stats = np.zeros((len(tasks), 14))
    for index in common:
        position = task_index[metadata.task_ids[index]]
        left_stats[position] += sample_stats(left[index]["expected_levels"], left[index]["predicted_levels"])
        right_stats[position] += sample_stats(right[index]["expected_levels"], right[index]["predicted_levels"])
    observed_left = metrics_from_stats(left_stats.sum(axis=0))
    observed_right = metrics_from_stats(right_stats.sum(axis=0))
    weights = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), N_RESAMPLES)
    boot_delta = metrics_from_stats(weights @ right_stats) - metrics_from_stats(weights @ left_stats)
    swaps = rng.integers(0, 2, size=(N_RESAMPLES, len(tasks))).astype(float)
    perm_left = swaps @ right_stats + (1 - swaps) @ left_stats
    perm_right = swaps @ left_stats + (1 - swaps) @ right_stats
    perm_delta = metrics_from_stats(perm_right) - metrics_from_stats(perm_left)
    return observed_left, observed_right, boot_delta, perm_delta, len(common), len(tasks)


def rq3_analysis(output: Path, metadata: TestMetadata, rng: np.random.Generator) -> None:
    latest = latest_evaluations()
    base_id = "google/gemma-4-E4B-it"
    targets = [
        "trained/gemma-4-e4b-best",
        "trained/gemma-4-e4b-weighted-macro-f1-best",
        "trained/gemma-4-e4b-balanced-macro-f1-best",
        "trained/gemma-4-e4b-balanced-accuracy-retrained",
    ]
    rows: list[dict[str, Any]] = []
    for target in targets:
        left, right, boot, perm, n_rows, n_tasks = paired_metric_comparison(
            latest[base_id], latest[target], metadata, rng
        )
        for metric_index, metric in enumerate(METRIC_NAMES):
            difference = right[metric_index] - left[metric_index]
            p_value = (np.sum(np.abs(perm[:, metric_index]) >= abs(difference)) + 1) / (N_RESAMPLES + 1)
            rows.append({
                "base": MODEL_NAMES[base_id],
                "specialized": MODEL_NAMES[target],
                "metric": metric,
                "base_estimate": left[metric_index],
                "specialized_estimate": right[metric_index],
                "paired_difference": difference,
                "ci95_low": np.percentile(boot[:, metric_index], 2.5),
                "ci95_high": np.percentile(boot[:, metric_index], 97.5),
                "p_value_raw": p_value,
                "n_rows": n_rows,
                "n_tasks": n_tasks,
                "test": "task-clustered paired permutation, two-sided",
                "unit": "programming task",
            })
    primary = [index for index, row in enumerate(rows) if row["metric"] in {"accuracy", "macro_f1"}]
    adjusted = holm_adjust([float(rows[index]["p_value_raw"]) for index in primary])
    for row in rows:
        row["p_value_holm_primary_family"] = ""
        row["family"] = "descriptive metric"
    for index, value in zip(primary, adjusted):
        rows[index]["p_value_holm_primary_family"] = value
        rows[index]["family"] = "8 predefined base-vs-specialized comparisons: Accuracy and Macro-F1"
    write_csv(output / "rq3_specialization_comparisons.csv", rows)


def merge_phase6_rows() -> list[dict[str, Any]]:
    paths = [PHASE6_API] + ([PHASE6_LOCAL] if PHASE6_LOCAL.exists() else [])
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            latest[(str(row["model"]), str(row["benchmark"]), int(row["benchmark_id"]))] = row
    return list(latest.values())


def rq4_phase6_analysis(output: Path, rng: np.random.Generator) -> None:
    baseline_rows = load_jsonl(RQ1_LOCAL) + load_jsonl(RQ1_API)
    baseline = {
        (str(row["model"]), str(row["benchmark"]), int(row["benchmark_id"])): row
        for row in baseline_rows
    }
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merge_phase6_rows():
        if not row.get("error"):
            by_model[str(row["model"])].append(row)
    rows: list[dict[str, Any]] = []
    for model, current_rows in sorted(by_model.items()):
        pairs = [
            (baseline[(model, str(row["benchmark"]), int(row["benchmark_id"]))], row)
            for row in current_rows
            if (model, str(row["benchmark"]), int(row["benchmark_id"])) in baseline
        ]
        if not pairs:
            continue
        before = np.array([level_set(old) == {"correct"} for old, _ in pairs], dtype=np.int8)
        after = np.array([level_set(new) == {"correct"} for _, new in pairs], dtype=np.int8)
        matrix = np.column_stack([before, after])
        boot = paired_bootstrap_binary(matrix, rng)
        before_only, after_only, p_value = exact_mcnemar(before, after)
        cycles = np.array([len(new.get("rounds", [])) for _, new in pairs], dtype=float)
        rows.append({
            "model_id": model,
            "model": MODEL_NAMES.get(model, model),
            "phase1_correct": before.mean(),
            "phase6_correct": after.mean(),
            "paired_difference_pp": (after.mean() - before.mean()) * 100,
            "ci95_low_pp": np.percentile(boot[:, 1] - boot[:, 0], 2.5) * 100,
            "ci95_high_pp": np.percentile(boot[:, 1] - boot[:, 0], 97.5) * 100,
            "phase1_only_correct": before_only,
            "phase6_only_correct": after_only,
            "p_value_raw": p_value,
            "cycles_mean": cycles.mean(),
            "cycles_median": np.median(cycles),
            "cycles_q1": np.percentile(cycles, 25),
            "cycles_q3": np.percentile(cycles, 75),
            "n_tasks": len(pairs),
            "test": "exact McNemar two-sided",
        })
    adjusted = holm_adjust([float(row["p_value_raw"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_value_holm"] = value
        row["family"] = f"{len(rows)} available Phase-6 vs Phase-1 model comparisons"
    write_csv(output / "rq4_phase6_generation_comparisons.csv", rows)


def rq4_static_tools_analysis(output: Path) -> None:
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(PHASE5):
        if row.get("status") == "ok":
            by_tool[str(row["tool"])].append(row)

    rows: list[dict[str, Any]] = []
    for tool, tool_rows in sorted(by_tool.items()):
        for scope, scoped in (
            ("test", [row for row in tool_rows if row.get("is_test")]),
            ("global", tool_rows),
        ):
            result: dict[str, Any] = {
                "tool_id": tool,
                "tool": str(tool_rows[0].get("tool_label", tool)),
                "scope": scope,
                "n_programs": len(scoped),
                "mean_seconds": float(
                    np.mean([float(item["elapsed_seconds"]) for item in scoped])
                ),
                "unit": "generated program",
            }
            for level in ("runtime", "syntax"):
                positives = [item for item in scoped if item[f"expected_{level}"]]
                detected = sum(bool(item[f"predicted_{level}"]) for item in positives)
                low, high = wilson(detected, len(positives))
                result[f"{level}_detected"] = detected
                result[f"{level}_total"] = len(positives)
                result[f"{level}_recall"] = (
                    detected / len(positives) if positives else math.nan
                )
                result[f"{level}_ci95_low_wilson"] = low
                result[f"{level}_ci95_high_wilson"] = high
            rows.append(result)
    write_csv(output / "rq4_static_tool_estimates.csv", rows)


def rq4_phase7_analysis(
    output: Path, metadata: TestMetadata, rng: np.random.Generator
) -> None:
    latest = latest_evaluations()
    phase7_paths = [PHASE7_API] + ([PHASE7_LOCAL] if PHASE7_LOCAL.exists() else [])
    phase7: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in phase7_paths:
        for row in load_jsonl(path):
            phase7[str(row["model_id"])][str(row["sample_id"])] = row

    no_tool_by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for model, model_rows in latest.items():
        for index, row in sorted(model_rows.items()):
            if row.get("error") or index >= len(metadata.sample_ids):
                continue
            for sample_id in metadata.sample_ids[index]:
                if sample_id not in no_tool_by_sample[model]:
                    no_tool_by_sample[model][sample_id] = row

    results: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for model, tool_rows in sorted(phase7.items()):
        common_ids = sorted(
            sample_id for sample_id in set(tool_rows) & set(no_tool_by_sample[model])
            if not tool_rows[sample_id].get("error")
        )
        if not common_ids:
            continue
        task_ids = sorted({str(tool_rows[sample]["task_id"]) for sample in common_ids})
        task_position = {task: index for index, task in enumerate(task_ids)}
        left_stats = np.zeros((len(task_ids), 14))
        right_stats = np.zeros((len(task_ids), 14))
        for sample_id in common_ids:
            tool = tool_rows[sample_id]
            no_tool = no_tool_by_sample[model][sample_id]
            position = task_position[str(tool["task_id"])]
            left_stats[position] += sample_stats(no_tool["expected_levels"], no_tool["predicted_levels"])
            right_stats[position] += sample_stats(tool["expected_levels"], tool["predicted_levels"])
        left = metrics_from_stats(left_stats.sum(axis=0))
        right = metrics_from_stats(right_stats.sum(axis=0))
        weights = rng.multinomial(len(task_ids), np.full(len(task_ids), 1 / len(task_ids)), N_RESAMPLES)
        boot = metrics_from_stats(weights @ right_stats) - metrics_from_stats(weights @ left_stats)
        swaps = rng.integers(0, 2, size=(N_RESAMPLES, len(task_ids))).astype(float)
        perm = metrics_from_stats(swaps @ left_stats + (1 - swaps) @ right_stats) - metrics_from_stats(
            swaps @ right_stats + (1 - swaps) @ left_stats
        )
        for metric_index, metric in enumerate(METRIC_NAMES):
            difference = right[metric_index] - left[metric_index]
            p_value = (np.sum(np.abs(perm[:, metric_index]) >= abs(difference)) + 1) / (N_RESAMPLES + 1)
            results.append({
                "model_id": model,
                "model": MODEL_NAMES.get(model, model),
                "metric": metric,
                "without_tools": left[metric_index],
                "with_tools": right[metric_index],
                "paired_difference": difference,
                "ci95_low": np.percentile(boot[:, metric_index], 2.5),
                "ci95_high": np.percentile(boot[:, metric_index], 97.5),
                "p_value_raw": p_value,
                "n_unique_samples": len(common_ids),
                "n_tasks": len(task_ids),
                "test": "task-clustered paired permutation, two-sided",
            })
            raw_p_values.append(p_value)
    adjusted = holm_adjust(raw_p_values)
    for row, value in zip(results, adjusted):
        row["p_value_holm"] = value
        row["family"] = f"all {len(results)} available Phase-7 metric comparisons"
    write_csv(output / "rq4_phase7_detection_comparisons.csv", results)


def write_latex_summary(output: Path) -> None:
    rq1 = list(csv.DictReader((output / "rq1_global_test.csv").open(encoding="utf-8")))
    rq3 = list(csv.DictReader((output / "rq3_specialization_comparisons.csv").open(encoding="utf-8")))
    rq4 = list(csv.DictReader((output / "rq4_phase6_generation_comparisons.csv").open(encoding="utf-8")))
    rq7 = list(csv.DictReader((output / "rq4_phase7_detection_comparisons.csv").open(encoding="utf-8")))
    lines = [
        "% Automatically generated by scripts/run_statistical_analysis.py",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Exploratory inferential analyses generated from paired task-level data.}",
        "\\label{tab:statistical_summary}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Comparison & Effect (pp) & 95\\% CI & Adjusted $p$ \\\\",
        "\\midrule",
    ]
    for row in rq4:
        effect = float(row["paired_difference_pp"])
        low = float(row["ci95_low_pp"])
        high = float(row["ci95_high_pp"])
        p_value = float(row["p_value_holm"])
        lines.append(
            f"{row['model']} (Phase 6--1) & {effect:.1f} & [{low:.1f}, {high:.1f}] & {p_value:.3g} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        f"% Cochran Q: Q={float(rq1[0]['statistic']):.3f}, df={rq1[0]['df']}, p={float(rq1[0]['p_value']):.6g}",
        "",
        "% RQ3 primary family: base versus specialized Accuracy and Macro-F1.",
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Specialized model & Metric & Base & Specialized & Effect [95\\% CI] & Holm $p$ \\\\",
        "\\midrule",
    ])
    for row in rq3:
        if row["metric"] not in {"accuracy", "macro_f1"}:
            continue
        label = "Accuracy" if row["metric"] == "accuracy" else "Macro-F1"
        effect = float(row["paired_difference"]) * 100
        low = float(row["ci95_low"]) * 100
        high = float(row["ci95_high"]) * 100
        lines.append(
            f"{row['specialized']} & {label} & {float(row['base_estimate']) * 100:.1f} & "
            f"{float(row['specialized_estimate']) * 100:.1f} & {effect:+.1f} [{low:+.1f}, {high:+.1f}] & "
            f"{float(row['p_value_holm_primary_family']):.3g} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
        "% RQ4 Phase 7 family: tool-feedback versus no-feedback metrics.",
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Metric & No tools & Tools & Effect [95\\% CI] & Holm $p$ \\\\",
        "\\midrule",
    ])
    for row in rq7:
        label = row["metric"].replace("_", " ").title()
        effect = float(row["paired_difference"]) * 100
        low = float(row["ci95_low"]) * 100
        high = float(row["ci95_high"]) * 100
        lines.append(
            f"{row['model']} & {label} & {float(row['without_tools']) * 100:.1f} & "
            f"{float(row['with_tools']) * 100:.1f} & {effect:+.1f} [{low:+.1f}, {high:+.1f}] & "
            f"{float(row['p_value_holm']):.3g} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    (output / "statistical_tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_phase3_balanced.yaml")
    parser.add_argument("--output", type=Path, default=Path("artifacts/statistics"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    metadata = build_test_metadata(args.config)
    rq1_analysis(args.output, rng)
    rq2_analysis(args.output, metadata, rng)
    rq3_analysis(args.output, metadata, rng)
    rq4_static_tools_analysis(args.output)
    rq4_phase6_analysis(args.output, rng)
    rq4_phase7_analysis(args.output, metadata, rng)
    write_latex_summary(args.output)

    input_paths = [RQ1_LOCAL, RQ1_API, EVALUATION, PHASE5, PHASE6_API, PHASE7_API]
    optional_paths = [PHASE6_LOCAL, PHASE7_LOCAL]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    metadata_json = {
        "executed_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "git_commit": commit,
        "seed": SEED,
        "bootstrap_resamples": N_RESAMPLES,
        "permutation_resamples": N_RESAMPLES,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "inputs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in input_paths
        ],
        "optional_inputs": [
            {"path": str(path), "available": path.exists(), **({"sha256": sha256(path)} if path.exists() else {})}
            for path in optional_paths
        ],
    }
    (args.output / "analysis_metadata.json").write_text(
        json.dumps(metadata_json, indent=2) + "\n", encoding="utf-8"
    )
    missing = [str(path) for path in optional_paths if not path.exists()]
    report = [
        "# Statistical pipeline validation",
        "",
        "- RQ1: 13 models × 563 unique, identically paired tasks validated.",
        f"- RQ2/RQ3: {len(metadata.task_ids)} test rows clustered within {len(set(metadata.task_ids))} tasks.",
        "- Duplicate evaluation attempts are resolved by retaining the last persisted model/sample row.",
        "- Bootstrap and permutation resampling use the programming task as the independent unit.",
        f"- Seed: `{SEED}`; resamples/permutations: `{N_RESAMPLES}`.",
    ]
    if missing:
        report.append(f"- BLOCKED local tool-assisted raw inputs not mounted: `{', '.join(missing)}`.")
    (args.output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (args.output / "pipeline.log").write_text(
        "Statistical pipeline completed successfully.\n"
        f"Commit: {commit}\n"
        f"Seed: {SEED}\n"
        f"Bootstrap/permutation resamples: {N_RESAMPLES}\n"
        f"Validated RQ1: 13 models x 563 paired tasks.\n"
        f"Validated RQ2/RQ3: {len(metadata.task_ids)} rows in "
        f"{len(set(metadata.task_ids))} task clusters.\n"
        f"Unavailable optional inputs: {', '.join(missing) if missing else 'none'}\n",
        encoding="utf-8",
    )
    print(f"Statistical artifacts written to {args.output}")


if __name__ == "__main__":
    main()
