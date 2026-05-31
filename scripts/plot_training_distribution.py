"""Gráfico de barras: distribuição de classes por partição do dataset de treinamento."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, DatasetDict

from plot_shared import COLORS, LEVEL_LABELS, LEVEL_ORDER, OUT_DIR, SPLIT_LABELS, load_jsonl

BASE_PATH = Path("data/results/dataset_base.json")
JUDGE_PATH = Path("data/results/dataset_judge.jsonl")
JUDGE_MODEL = "gpt-oss:20b"


def _to_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def build_level_list(base_path: Path, judge_path: Path, judge_model: str) -> list[str]:
    base_records = load_jsonl(base_path)
    judge_records = load_jsonl(judge_path)

    judge_keys: set[tuple] = set()
    for r in judge_records:
        if r.get("judge_model") != judge_model:
            continue
        bid = _to_int(r.get("benchmark_id"))
        if bid is None:
            continue
        judge_keys.add((str(r.get("benchmark")), bid, str(r.get("response_model"))))

    levels = []
    for r in base_records:
        bid = _to_int(r.get("benchmark_id"))
        if bid is None:
            continue
        key = (str(r.get("benchmark")), bid, str(r.get("model")))
        if key in judge_keys:
            levels.append(str(r.get("level", "")))
    return levels


def stratified_split(levels: list[str], validation_size: float, test_size: float, seed: int) -> DatasetDict:
    ds = Dataset.from_list([{"level": lv} for lv in levels])
    ds = ds.map(lambda s: {"stratify_level": s["level"]}, batched=False)
    ds = ds.class_encode_column("stratify_level")

    holdout = validation_size + test_size
    first = ds.train_test_split(test_size=holdout, seed=seed, stratify_by_column="stratify_level")
    temp = first["test"]

    test_ratio = test_size / holdout
    second = temp.train_test_split(test_size=test_ratio, seed=seed, stratify_by_column="stratify_level")

    return DatasetDict({
        "train": first["train"].remove_columns("stratify_level"),
        "validation": second["train"].remove_columns("stratify_level"),
        "test": second["test"].remove_columns("stratify_level"),
    })


def plot(split_counts: dict[str, Counter], out_dir: Path) -> None:
    splits = ["train", "validation", "test"]
    x = np.arange(len(splits))
    bar_width = 0.18
    offsets = np.linspace(-(len(LEVEL_ORDER) - 1) / 2, (len(LEVEL_ORDER) - 1) / 2, len(LEVEL_ORDER)) * bar_width

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, level in enumerate(LEVEL_ORDER):
        counts = [split_counts[sp].get(level, 0) for sp in splits]
        bars = ax.bar(x + offsets[i], counts, bar_width * 0.9, color=COLORS[level], label=LEVEL_LABELS[level])
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    totals = [sum(split_counts[sp].values()) for sp in splits]
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{SPLIT_LABELS[sp]}\n(n = {totals[i]})" for i, sp in enumerate(splits)],
        fontsize=11,
    )
    ax.set_ylabel("Número de Amostras", fontsize=11)
    ax.set_title("Distribuição do Dataset de Treinamento por Partição", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(c for cnt in split_counts.values() for c in cnt.values()) * 1.2)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "training_dataset_distribution.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {out_path}")


def main() -> None:
    levels = build_level_list(BASE_PATH, JUDGE_PATH, JUDGE_MODEL)
    print(f"Total de registros correspondidos: {len(levels)}")

    dataset = stratified_split(levels, validation_size=0.1, test_size=0.2, seed=42)

    split_counts: dict[str, Counter] = {}
    for split_name, split_ds in dataset.items():
        split_counts[split_name] = Counter(split_ds["level"])
        print(f"{split_name}: {dict(split_counts[split_name])}")

    plot(split_counts, OUT_DIR)


if __name__ == "__main__":
    main()
