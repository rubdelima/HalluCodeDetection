from __future__ import annotations

from rich.table import Table
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.core import ui


def render_grid_table(rows: list[dict[str, str]]) -> Table:
    table = Table(title="Training Grid", show_header=True, header_style="bold")
    table.add_column("Config")
    table.add_column("Status")
    table.add_column("Score")
    for row in rows:
        table.add_row(row.get("config", "-"), row.get("status", "-"), row.get("score", "-"))
    return table


def build_grid_progress(total: int) -> tuple[Progress, TaskID]:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.fields[config]}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=ui.console,
    )
    task_id = progress.add_task("grid", total=total, config="-")
    return progress, task_id
