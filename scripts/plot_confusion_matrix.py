"""Matrizes de confusão: modelo base (google/gemma-3-4b-it) vs adaptado (gemma-hallucination-qlora)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from plot_shared import LEVEL_LABELS_AXIS, LEVEL_ORDER, OUT_DIR, load_jsonl

EVAL_PATH = Path("data/results/evaluation_results.jsonl")

MODEL_CONFIGS = [
    {
        "model": "google/gemma-3-4b-it",
        "title": "Modelo Base\n(google/gemma-3-4b-it)",
        "filename": "confusion_matrix_base_gemma.png",
    },
    {
        "model": "gemma-hallucination-qlora",
        "title": "Modelo Adaptado\n(gemma-hallucination-qlora)",
        "filename": "confusion_matrix_adapted_gemma.png",
    },
]


def plot_cm(y_true: list[str], y_pred: list[str], title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LEVEL_ORDER)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.0%})"

    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    n = len(y_true)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=LEVEL_LABELS_AXIS,
        yticklabels=LEVEL_LABELS_AXIS,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Proporção normalizada por linha"},
    )

    ax.set_xlabel("Classe Predita", fontsize=11)
    ax.set_ylabel("Classe Real", fontsize=11)
    ax.set_title(f"{title}\nAcurácia: {acc:.1%}  (n = {n})", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9, rotation=0)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {out_path}")


def main() -> None:
    records = load_jsonl(EVAL_PATH)

    for cfg in MODEL_CONFIGS:
        model_records = [
            r for r in records
            if r["model"] == cfg["model"] and r.get("predicted_level", "") != ""
        ]
        if not model_records:
            print(f"Nenhum registro encontrado para o modelo: {cfg['model']}")
            continue

        y_true = [r["expected_level"] for r in model_records]
        y_pred = [r["predicted_level"] for r in model_records]

        plot_cm(y_true, y_pred, cfg["title"], OUT_DIR / cfg["filename"])


if __name__ == "__main__":
    main()
