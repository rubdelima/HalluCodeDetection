"""Gráficos de pizza: distribuição de classificação por modelo (dataset_base.json)."""

from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from plot_shared import COLORS, LEVEL_LABELS, LEVEL_ORDER, MODEL_DISPLAY, OUT_DIR, load_jsonl

DATA_PATH = Path("data/results/dataset_base.json")


def build_distribution(records: list[dict]) -> dict[str, dict[str, int]]:
    dist: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        dist[r["model"]][r["level"]] += 1
    return dist


def make_autopct(values: list[int]):
    def autopct(pct: float) -> str:
        total = sum(values)
        count = int(round(pct * total / 100))
        if pct < 5:
            return ""
        return f"{pct:.1f}%\n({count})"
    return autopct


def plot_model(model: str, counts: dict[str, int], out_dir: Path) -> None:
    total = sum(counts.values())

    sizes = [counts.get(lv, 0) for lv in LEVEL_ORDER]
    colors = [COLORS[lv] for lv in LEVEL_ORDER]
    present = [(s, c) for s, c, _ in zip(sizes, colors, LEVEL_ORDER) if s > 0]
    p_sizes, p_colors = zip(*present)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.pie(
        p_sizes,
        colors=p_colors,
        autopct=make_autopct(list(p_sizes)),
        pctdistance=0.75,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9},
    )

    ax.set_title(
        f"{MODEL_DISPLAY.get(model, model)}\n(n = {total})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    legend_handles = [
        mpatches.Patch(color=COLORS[lv], label=LEVEL_LABELS[lv])
        for lv in LEVEL_ORDER
        if counts.get(lv, 0) > 0
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.18),
    )

    plt.tight_layout()

    slug = model.replace(":", "_").replace(".", "_")
    out_path = out_dir / f"classification_{slug}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {out_path}")


def main() -> None:
    records = load_jsonl(DATA_PATH)
    dist = build_distribution(records)
    for model, counts in sorted(dist.items()):
        plot_model(model, counts, OUT_DIR)


if __name__ == "__main__":
    main()
