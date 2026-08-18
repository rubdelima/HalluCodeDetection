"""Matrizes de confusão dos modelos avaliados para detecção de alucinação."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_shared import LEVEL_LABELS_AXIS, LEVEL_ORDER, MODEL_DISPLAY, OUT_DIR, load_jsonl, model_key, slugify_model

EVAL_PATH = Path("data/results/evaluation_results.jsonl")

MODEL_CONFIGS = [
    {
        "model": "gpt-oss:20b",
        "title": "GPT-OSS 20B",
        "filename": "confusion_matrix_gpt_oss_20b.png",
    },
    {
        "model": "gemma3:4b",
        "title": "Gemma 3 4B",
        "filename": "confusion_matrix_gemma3_4b.png",
    },
    {
        "model": "gemma3:1b",
        "title": "Gemma 3 1B",
        "filename": "confusion_matrix_gemma3_1b.png",
    },
    {
        "model": "gemma4:e4b",
        "title": "Gemma 4 E4B",
        "filename": "confusion_matrix_gemma4_e4b.png",
    },
    {
        "model": "qwen2.5-coder:7b",
        "title": "Qwen 2.5-Coder 7B",
        "filename": "confusion_matrix_qwen2_5_coder_7b.png",
    },
    {
        "model": "google/gemma-3-4b-it",
        "title": "Modelo Base\n(google/gemma-3-4b-it)",
        "filename": "confusion_matrix_base_gemma.png",
    },
    {
        "model": "data/trained_models/f2743b4e",
        "title": "Modelo Adaptado\n(data/trained_models/f2743b4e)",
        "filename": "confusion_matrix_adapted_gemma.png",
    },
]


def plot_cm(y_true: list[str], y_pred: list[str], title: str, out_path: Path) -> None:
    cm = np.zeros((len(LEVEL_ORDER), len(LEVEL_ORDER)), dtype=int)
    level_index = {level: idx for idx, level in enumerate(LEVEL_ORDER)}
    for expected, predicted in zip(y_true, y_pred):
        cm[level_index[expected], level_index[predicted]] += 1

    row_totals = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)

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
            if model_key(r) == cfg["model"] and r.get("predicted_level") in LEVEL_ORDER
        ]
        if not model_records:
            print(f"Nenhum registro encontrado para o modelo: {cfg['model']}")
            continue

        y_true = [r["expected_level"] for r in model_records]
        y_pred = [r["predicted_level"] for r in model_records]

        title = cfg.get("title") or MODEL_DISPLAY.get(cfg["model"], cfg["model"])
        filename = cfg.get("filename") or f"confusion_matrix_{slugify_model(cfg['model'])}.png"
        plot_cm(y_true, y_pred, title, OUT_DIR / filename)


if __name__ == "__main__":
    main()
