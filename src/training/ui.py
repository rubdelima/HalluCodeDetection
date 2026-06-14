from __future__ import annotations

from rich.table import Table

from src.constants.training import TrainingHyperparameters
from src.schemas.training import TrainingResult


def format_accuracy(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{remainder:02d}s"


def nd() -> str:
    return "[dim]ND[/]"


def qlora_cell(value: bool) -> str:
    return "[green]yes[/]" if value else "[yellow]no[/]"


def status_cell(status: str) -> str:
    if status == "pending":
        return "[yellow]pending[/]"
    if status == "done":
        return "[green]done[/]"
    if status == "json-only":
        return "[blue]json-only[/]"
    return status


def saved_cell(result: TrainingResult | None) -> str:
    if result is None:
        return nd()
    return "[green]yes[/]" if result.saved_model else "[dim]no[/]"


def metric_cell(value: float | None, color: str) -> str:
    if value is None:
        return nd()
    return f"[{color}]{format_accuracy(value)}[/]"


def render_training_status_table(rows: list[dict[str, object]]) -> Table:
    table = Table(title="Training Hyperparameters", show_header=True, header_style="bold")
    table.add_column("Status", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("QLoRA", justify="center")
    table.add_column("r", justify="right", style="bright_cyan")
    table.add_column("alpha", justify="right", style="blue")
    table.add_column("dropout", justify="right", style="yellow")
    table.add_column("lr", justify="right", style="bright_magenta")
    table.add_column("epochs", justify="right", style="bright_blue")
    table.add_column("bias", style="white")
    table.add_column("optimizer", style="cyan")
    table.add_column("time", justify="right", style="white")
    table.add_column("train", justify="right")
    table.add_column("val", justify="right")
    table.add_column("test", justify="right")
    table.add_column("saved", justify="center")
    table.add_column("path", style="blue")

    for row in rows:
        hyperparameters = row["hyperparameters"]
        if not isinstance(hyperparameters, TrainingHyperparameters):
            continue

        result = row.get("result")
        result = result if isinstance(result, TrainingResult) else None
        table.add_row(
            status_cell(str(row.get("status", "-"))),
            hyperparameters.model_name.id,
            qlora_cell(hyperparameters.use_qlora),
            str(hyperparameters.lora_r),
            str(hyperparameters.lora_alpha),
            f"{hyperparameters.lora_dropout:g}",
            f"{hyperparameters.learning_rate:g}",
            str(hyperparameters.num_epochs),
            hyperparameters.bias,
            hyperparameters.optimizer,
            format_duration(result.training_time) if result else nd(),
            metric_cell(result.train_acc if result else None, "green"),
            metric_cell(result.val_acc if result else None, "yellow"),
            metric_cell(result.test_acc if result else None, "bold green"),
            saved_cell(result),
            result.model_path if result and result.model_path else nd(),
        )

    return table
