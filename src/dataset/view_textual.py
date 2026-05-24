from __future__ import annotations

import json
from pathlib import Path

from typing import cast

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.coordinate import Coordinate
from textual.widgets.data_table import RowKey
from textual.widgets import DataTable, Footer, Header, Input, Static


class DatasetViewer(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        layout: horizontal;
    }

    #filters {
        layout: horizontal;
        height: 3;
        padding: 0 1;
        border: solid $primary;
    }

    #filters Input {
        margin: 0 1 0 0;
        width: 1fr;
    }

    #table {
        width: 2fr;
        height: 1fr;
    }

    #details {
        width: 3fr;
        height: 1fr;
        border: solid $secondary;
        padding: 1;
    }

    #summary {
        height: 3;
        padding: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self, results: list[dict[str, object]]) -> None:
        super().__init__()
        self.results = results
        self.filtered_results: list[dict[str, object]] = list(results)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="summary")
        with Horizontal(id="filters"):
            yield Input(placeholder="Filter model", id="filter_model")
            yield Input(placeholder="Filter level", id="filter_level")
            yield Input(placeholder="Filter benchmark id", id="filter_id")
        with Horizontal(id="main"):
            yield DataTable(id="table")
            with Vertical(id="details"):
                yield Static("Select a row to view details.", id="detail_text")
        yield Footer()

    def on_mount(self) -> None:
        self._update_summary()
        self._populate_table()
        table = self.query_one("#table", DataTable)
        table.cursor_type = "row"
        table.show_cursor = True
        table.focus()
        if self.filtered_results:
            table.cursor_coordinate = Coordinate(0, 0)
            self._select_row_key(cast(RowKey, "0"))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._select_row_key(event.row_key)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._select_row_key(event.row_key)

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        table = self.query_one("#table", DataTable)
        row_key, _ = table.coordinate_to_cell_key(event.coordinate)
        self._select_row_key(row_key)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id not in {"filter_model", "filter_level", "filter_id"}:
            return
        self._apply_filters()

    def _update_summary(self) -> None:
        counts = {
            "correct": 0,
            "functional_error": 0,
            "runtime_error": 0,
            "syntax_error": 0,
        }
        for item in self.results:
            level = item.get("level")
            if level in counts:
                counts[level] += 1
        total = sum(counts.values())
        summary = (
            f"Total: {total} | "
            f"Correct: {counts['correct']} | "
            f"Functional: {counts['functional_error']} | "
            f"Runtime: {counts['runtime_error']} | "
            f"Syntax: {counts['syntax_error']}"
        )
        summary_widget = self.query_one("#summary", Static)
        summary_widget.update(summary)

    def _populate_table(self) -> None:
        table = self.query_one("#table", DataTable)
        table.clear(columns=True)
        table.add_columns("Model", "Benchmark", "ID", "Level")
        for idx, item in enumerate(self.filtered_results):
            table.add_row(
                str(item.get("model")),
                str(item.get("benchmark")),
                str(item.get("benchmark_id")),
                str(item.get("level")),
                key=str(idx),
            )

    def _apply_filters(self) -> None:
        filter_model = self.query_one("#filter_model", Input).value.strip().lower()
        filter_level = self.query_one("#filter_level", Input).value.strip().lower()
        filter_id = self.query_one("#filter_id", Input).value.strip().lower()

        def matches(item: dict[str, object]) -> bool:
            if filter_model and filter_model not in str(item.get("model", "")).lower():
                return False
            if filter_level and filter_level not in str(item.get("level", "")).lower():
                return False
            if filter_id and filter_id not in str(item.get("benchmark_id", "")).lower():
                return False
            return True

        self.filtered_results = [item for item in self.results if matches(item)]
        self._populate_table()
        detail_text = self.query_one("#detail_text", Static)
        detail_text.update("Select a row to view details.")

    def _select_row_key(self, row_key: RowKey | None) -> None:
        if row_key is None:
            return
        try:
            index = int(str(row_key))
        except ValueError:
            return
        if not 0 <= index < len(self.filtered_results):
            return
        self._render_details(self.filtered_results[index])

    def _render_details(self, item: dict[str, object]) -> None:
        detail_text = self.query_one("#detail_text", Static)
        error = str(item.get("error") or "")
        code = str(item.get("code") or "")
        detail = (
            f"Model: {item.get('model')}\n"
            f"Benchmark: {item.get('benchmark')}\n"
            f"Benchmark ID: {item.get('benchmark_id')}\n"
            f"Level: {item.get('level')}\n\n"
            f"Error:\n{error}\n\n"
            f"Code:\n{code}\n"
        )
        detail_text.update(detail)


def _load_results(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    results: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def view_dataset(config: dict[str, object]) -> None:
    build_cfg = cast(dict[str, object], config.get("dataset_build", {}))
    results_dir = Path(str(build_cfg.get("results_dir", "data/")))
    results_path = results_dir / "dataset_base.json"
    results = _load_results(results_path)
    if not results:
        from src.core import ui

        ui.console.print("No results found.")
        return
    DatasetViewer(results).run()
