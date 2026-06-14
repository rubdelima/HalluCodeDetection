from __future__ import annotations

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.core import ui
from src.schemas.evaluation import EvaluationResume


LEVEL_COLUMNS = (
    ("correct", "green"),
    ("functional_error", "yellow"),
    ("runtime_error", "orange3"),
    ("syntax_error", "red"),
)


def format_accuracy(value: float) -> str:
    return f"{value * 100:.2f}%"


def get_progress_evaluation() -> Progress:
    return Progress(
        SpinnerColumn(style="bold cyan"),
        TextColumn("[bold cyan]{task.fields[provider]}[/]"),
        TextColumn("[magenta]{task.fields[model]}[/]"),
        TextColumn("[white]{task.fields[sample]}[/]"),
        TextColumn("[bold]Parsed:[/] [cyan]{task.fields[parsed]}[/]"),
        TextColumn("[bold]Acc:[/] [green]{task.fields[accuracy]}[/]"),
        TextColumn("[green]C:{task.fields[correct]}[/]"),
        TextColumn("[yellow]F:{task.fields[functional_error]}[/]"),
        TextColumn("[orange3]R:{task.fields[runtime_error]}[/]"),
        TextColumn("[red]S:{task.fields[syntax_error]}[/]"),
        BarColumn(bar_width=None, complete_style="green", finished_style="bold green"),
        TextColumn("[bold]{task.completed}/{task.total}[/]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=ui.console,
    )


def add_evaluation_task(
    progress: Progress,
    total: int,
    provider: str,
    model: str,
    resume: EvaluationResume,
) -> TaskID:
    return progress.add_task(
        "evaluate",
        total=total,
        provider=provider,
        model=model,
        sample="-",
        parsed=resume.parsed_responses,
        accuracy=format_accuracy(resume.overall_accuracy),
        **resume.corrects_by_level,
    )


def update_evaluation_task(
    progress: Progress,
    task_id: TaskID,
    resume: EvaluationResume,
    *,
    sample: str,
    advance: int = 0,
) -> None:
    progress.update(
        task_id,
        advance=advance,
        sample=sample,
        parsed=resume.parsed_responses,
        accuracy=format_accuracy(resume.overall_accuracy),
        **resume.corrects_by_level,
    )


def render_summary_table(rows: list[dict[str, object]]) -> Table:
    table = Table(title="Evaluation Summary", show_header=True, header_style="bold")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Parsed", justify="right", style="cyan")
    table.add_column("Total", justify="right", style="blue")
    for level, style in LEVEL_COLUMNS:
        table.add_column(level, justify="right", style=style)
    table.add_column("Accuracy", justify="right", style="green")

    for row in rows:
        table.add_row(
            str(row.get("provider", "-")),
            str(row.get("model", "-")),
            str(row.get("parsed", 0)),
            str(row.get("total", 0)),
            *(str(row.get(level, 0)) for level, _ in LEVEL_COLUMNS),
            str(row.get("accuracy", "0.00%")),
        )

    return table
