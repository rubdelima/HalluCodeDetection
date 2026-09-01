from __future__ import annotations

from rich.table import Table

from src.constants.training import TrainingHyperparameters
from src.schemas.training import TrainingResult


def format_accuracy(value: float) -> str:
    return f"{value * 100:.1f}"


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{remainder:02d}s"


def nd() -> str:
    return "[dim]ND[/]"


def metric_cell(value: float | None, color: str) -> str:
    if value is None:
        return nd()
    return f"[{color}]{format_accuracy(value)}[/]"


def render_training_status_table(rows: list[dict[str, object]]) -> Table:
    table = Table(title="Training Hyperparameters", show_header=True, header_style="bold")
    table.add_column("Base model", style="magenta")
    table.add_column("params1", style="bright_cyan")
    table.add_column("params2", style="bright_magenta")
    table.add_column("ratio", justify="right", style="bright_cyan")
    table.add_column("time", justify="right", style="white")
    table.add_column("Acc", justify="right")
    table.add_column("MF1", justify="right")
    table.add_column("MR", justify="right")

    for row in rows:
        hyperparameters = row["hyperparameters"]
        if not isinstance(hyperparameters, TrainingHyperparameters):
            continue

        result = row.get("result")
        result = result if isinstance(result, TrainingResult) else None
        table.add_row(
            hyperparameters.model_name.id.removeprefix("google/"),
            f"r={hyperparameters.lora_r}; alpha={hyperparameters.lora_alpha}; dropout={hyperparameters.lora_dropout:g}",
            f"lr={hyperparameters.learning_rate:g}; epochs={hyperparameters.num_epochs}; bias={hyperparameters.bias}",
            ":".join(map(str, hyperparameters.class_balance_ratio)),
            format_duration(result.training_time) if result else nd(),
            metric_cell(result.val_acc if result else None, "yellow"),
            metric_cell(result.val_macro_f1 if result else None, "magenta"),
            metric_cell(result.val_macro_recall if result else None, "cyan"),
        )

    return table
