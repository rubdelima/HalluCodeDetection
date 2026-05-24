from __future__ import annotations

from typing import Iterable, Mapping

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
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
from rich.text import Text


console = Console()


def format_pass_at_1(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    if total == 0:
        return "0.0%"
    return f"{(counts['correct'] / total) * 100:.1f}%"


def render_status_table(model_counts: dict[str, dict[str, int]]) -> Table:
    table = Table(title="Dataset Build", show_header=True, header_style="bold")
    table.add_column("Model", style="magenta")
    table.add_column("Correct", style="green")
    table.add_column("Functional", style="yellow")
    table.add_column("Runtime", style="orange3")
    table.add_column("Syntax", style="red")
    table.add_column("Pass@1", style="blue")
    for model in sorted(model_counts.keys()):
        counts = model_counts[model]
        table.add_row(
            model,
            str(counts["correct"]),
            str(counts["functional_error"]),
            str(counts["runtime_error"]),
            str(counts["syntax_error"]),
            format_pass_at_1(counts),
        )
    return table


def build_progress(total: int, counts: dict[str, int]) -> tuple[Progress, TaskID]:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.fields[model]}"),
        TextColumn("{task.fields[bench]}"),
        TextColumn("{task.fields[bench_id]}"),
        TextColumn("C:{task.fields[c]}"),
        TextColumn("F:{task.fields[f]}"),
        TextColumn("R:{task.fields[r]}"),
        TextColumn("S:{task.fields[s]}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task_id = progress.add_task(
        "build",
        total=total,
        model="-",
        bench="-",
        bench_id="-",
        c=counts["correct"],
        f=counts["functional_error"],
        r=counts["runtime_error"],
        s=counts["syntax_error"],
    )
    return progress, task_id


def stream_chat_chunks(
    chunk_iter: Iterable[Mapping[str, object]],
    spinner_length: int,
) -> str:
    content_buffer: list[str] = []
    thinking_buffer: list[str] = []
    content_preview = Text()
    thinking_preview = Text(style="yellow")

    def render_group(show_thinking: bool) -> Group:
        panels = []
        if show_thinking:
            panels.append(Panel(thinking_preview, title="Model Thinking", border_style="yellow"))
        panels.append(Panel(content_preview, title="Model Output", border_style="cyan"))
        return Group(*panels)

    show_thinking = False
    with Live(
        render_group(show_thinking),
        refresh_per_second=10,
        console=console,
        transient=True,
    ) as live:
        for chunk in chunk_iter:
            message_obj = chunk.get("message")
            message = message_obj if isinstance(message_obj, dict) else {}
            thinking_obj = message.get("thinking")
            thinking = thinking_obj if isinstance(thinking_obj, str) else None
            if thinking is None:
                fallback_thinking = chunk.get("thinking")
                thinking = fallback_thinking if isinstance(fallback_thinking, str) else None
            content_obj = message.get("content")
            content = content_obj if isinstance(content_obj, str) else ""

            if thinking:
                if not show_thinking:
                    show_thinking = True
                    live.update(render_group(show_thinking))
                thinking_buffer.append(thinking)
                thinking_preview.plain = "".join(thinking_buffer)[-spinner_length:]
                thinking_preview.no_wrap = False
            if content:
                content_buffer.append(content)
                content_preview.plain = "".join(content_buffer)[-spinner_length:]
                content_preview.no_wrap = False

    return "".join(content_buffer)
