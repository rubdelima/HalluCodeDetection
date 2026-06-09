from __future__ import annotations

from typing import Iterable, Mapping

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style
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


class MenuCompleter(Completer):
    def __init__(self, options: Mapping[str, str]) -> None:
        self.options = options

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor.strip().lower()
        for key, value in self.options.items():
            key_match = key.lower().startswith(text)
            value_match = value.lower().startswith(text) or text in value.lower()
            if text and not (key_match or value_match):
                continue
            yield Completion(
                value,
                start_position=-len(document.text_before_cursor),
                display=f"{key}  {value}",
                display_meta="click, Enter, or use arrows",
            )


def interactive_menu(options: Mapping[str, str]) -> str:
    option_lookup = {
        lookup_key.lower(): value
        for key, value in options.items()
        for lookup_key in (key, value)
    }
    completer = MenuCompleter(options)
    bindings = KeyBindings()

    @bindings.add("down")
    def _(event) -> None:
        buffer = event.current_buffer
        if buffer.complete_state:
            buffer.complete_next()
        else:
            buffer.start_completion(select_first=True)

    @bindings.add("up")
    def _(event) -> None:
        buffer = event.current_buffer
        if buffer.complete_state:
            buffer.complete_previous()
        else:
            buffer.start_completion(select_last=True)

    session = PromptSession(
        completer=completer,
        complete_while_typing=True,
        complete_style=CompleteStyle.MULTI_COLUMN,
        key_bindings=bindings,
        mouse_support=True,
        style=Style.from_dict(
            {
                "completion-menu.completion": "bg:#202020 #ffffff",
                "completion-menu.completion.current": "bg:#005f87 #ffffff",
                "completion-menu.meta.completion": "bg:#202020 #aaaaaa",
                "completion-menu.meta.completion.current": "bg:#005f87 #ffffff",
            }
        ),
        bottom_toolbar=HTML("Up/Down: navigate | Type: filter | Click: select | Enter: confirm"),
    )

    console.print("Select an option:")
    for key, value in options.items():
        console.print(f"  [bold cyan]{key}[/]) {value}")

    while True:
        try:
            choice = session.prompt("Choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "quit"

        if choice in option_lookup:
            return option_lookup[choice]

        matches = [
            value
            for key, value in options.items()
            if key.lower().startswith(choice) or value.lower().startswith(choice)
        ]
        unique_matches = set(matches)
        if choice and len(unique_matches) == 1:
            return next(iter(unique_matches))

        console.print("[red]Invalid option.[/] Type to filter, use arrows, or click an option.")


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

from dataclasses import dataclass

@dataclass
class OllamaResponse:
    content: str
    thoughts: str | None = None

def stream_chat_chunks(
    chunk_iter: Iterable[Mapping[str, object]],
    spinner_length: int,
) -> OllamaResponse:
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

    return OllamaResponse(content="".join(content_buffer), thoughts=None if not thinking_buffer else "".join(thinking_buffer))
