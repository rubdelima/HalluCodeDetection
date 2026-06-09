from rich.table import Table
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

def render_summary_table(rows: list[dict[str, object]]) -> Table:
    table = Table(title="Evaluation Summary", show_header=True, header_style="bold")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Correct", style="green")
    table.add_column("Total", style="blue")
    table.add_column("Accuracy", style="blue")
    for row in rows:
        total = 0
        total_raw = row.get("total")
        if isinstance(total_raw, int):
            total = total_raw

        correct = 0
        correct_raw = row.get("correct")
        if isinstance(correct_raw, int):
            correct = correct_raw

        accuracy = (correct / total) if total else 0.0
        table.add_row(
            str(row.get("provider", "-")),
            str(row.get("model", "-")),
            str(correct),
            str(total),
            f"{accuracy:.4f}",
        )
    return table

def get_progress_evaluation():
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.fields[provider]}"),
        TextColumn("{task.fields[model]}"),
        TextColumn("{task.fields[sample]}"),
        TextColumn("C:{task.fields[c]}"),
        TextColumn("F:{task.fields[f]}"),
        TextColumn("R:{task.fields[r]}"),
        TextColumn("S:{task.fields[s]}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=ui.console,
    )