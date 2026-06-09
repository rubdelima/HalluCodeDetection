from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.console import Console

def get_augmentation_progress(console: Console)->Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.fields[model]}"),
        TextColumn("{task.fields[bench]}"),
        TextColumn("{task.fields[bench_id]}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    