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
from src.evaluations.metrics import recalls_by_level
from src.schemas.evaluation import EvaluationResume


LEVEL_COLUMNS = (
    ("correct", "CR", "green"),
    ("functional", "FR", "yellow"),
    ("runtime", "RR", "orange3"),
    ("syntax", "SR", "red"),
)


def format_metric(value: float) -> str:
    """Compact percentage display used by the live evaluation status."""
    return f"{value * 100:.1f}"


def get_progress_evaluation() -> Progress:
    return Progress(
        SpinnerColumn(style="bold cyan"),
        TextColumn("[bold cyan]{task.fields[provider]}[/]"),
        TextColumn("[magenta]{task.fields[model]}[/]"),
        TextColumn("[white]{task.fields[sample]}[/]"),
        TextColumn("[green]Acc:{task.fields[accuracy]}[/]"),
        TextColumn("[magenta]MF1:{task.fields[macro_f1]}[/]"),
        TextColumn("[green]CR:{task.fields[correct]}[/]"),
        TextColumn("[yellow]FR:{task.fields[functional]}[/]"),
        TextColumn("[orange3]RR:{task.fields[runtime]}[/]"),
        TextColumn("[red]SR:{task.fields[syntax]}[/]"),
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
        accuracy=format_metric(resume.overall_accuracy),
        macro_f1=format_metric(resume.macro_f1),
        **{level: format_metric(value) for level, value in recalls_by_level(resume).items()},
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
        accuracy=format_metric(resume.overall_accuracy),
        macro_f1=format_metric(resume.macro_f1),
        **{level: format_metric(value) for level, value in recalls_by_level(resume).items()},
    )


def render_summary_table(rows: list[dict[str, object]]) -> Table:
    table = Table(title="Evaluation Summary", show_header=True, header_style="bold")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Parsed", justify="right", style="cyan")
    table.add_column("Total", justify="right", style="blue")
    table.add_column("Skipped", justify="right", style="yellow")
    table.add_column("Acc", justify="right", style="green")
    table.add_column("MF1", justify="right", style="magenta")
    for _, short_name, style in LEVEL_COLUMNS:
        table.add_column(short_name, justify="right", style=style)

    for row in rows:
        table.add_row(
            str(row.get("provider", "-")),
            str(row.get("model", "-")),
            str(row.get("parsed", 0)),
            str(row.get("total", 0)),
            str(row.get("skipped", 0)),
            str(row.get("accuracy", "0.0")),
            str(row.get("macro_f1", "0.0")),
            *(str(row.get(level, "0.0")) for level, _, _ in LEVEL_COLUMNS),
        )

    return table
