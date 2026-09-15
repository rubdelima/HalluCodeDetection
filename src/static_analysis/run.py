from __future__ import annotations

import csv
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

from src.constants import HalluCodeDetectionConfig
from src.core import ui
from src.dataset.judge_dataset import load_hallucination_dataset
from src.dataset.utils import append_jsonl, load_jsonl
from src.schemas.dataset import BaseResultRow
from src.static_analysis.tools import (
    ToolFinding,
    ToolSpec,
    run_tool,
    tool_specs,
    tool_version,
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    task_id: str
    source: str
    levels: frozenset[str]
    is_test: bool


SUMMARY_FIELDS = ["Ferramenta", "RR (T)", "RS (T)", "RR (G)", "RS (G)", "Avg T"]


def _sample_id(row: BaseResultRow) -> str:
    return f"{row.benchmark}:{row.benchmark_id}:{row.model}"


def _load_samples(config: HalluCodeDetectionConfig, limit: int | None) -> list[Sample]:
    base_path = Path(config.dataset_building_config.results_dir) / "dataset_base.json"
    base_rows = load_jsonl(base_path, BaseResultRow)
    if not base_rows:
        raise FileNotFoundError(f"No Phase-1 results found at {base_path}")

    # Force the complete correct class here so the held-out task assignment is
    # the same one used by the final balanced Phase-4 experiment.
    split = load_hallucination_dataset(config, correct_size=1.0)
    test_task_ids = {str(row["task_id"]) for row in split["test"]}
    if not test_task_ids:
        raise RuntimeError("The test split is empty; verify the Phase-2 judge data and dataset configuration.")

    unique: dict[str, Sample] = {}
    for row in base_rows:
        sample_id = _sample_id(row)
        task_id = row.task_id or f"{row.benchmark}/{row.benchmark_id}"
        unique[sample_id] = Sample(
            sample_id=sample_id,
            task_id=task_id,
            source=row.code,
            levels=frozenset(level.level_name for level in row.levels),
            is_test=task_id in test_task_ids,
        )
    samples = list(unique.values())
    return samples[:limit] if limit is not None else samples


def _load_existing(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    rows = load_jsonl(path, dict, quiet=True)
    return {
        (str(row.get("tool")), str(row.get("sample_id"))): row
        for row in rows
        if row.get("tool") and row.get("sample_id")
    }


def _recall(rows: Iterable[dict[str, object]], label: str, test_only: bool) -> float | None:
    selected = [row for row in rows if not test_only or bool(row.get("is_test"))]
    positives = sum(bool(row.get(f"expected_{label}")) for row in selected)
    if positives == 0:
        return None
    true_positives = sum(
        bool(row.get(f"expected_{label}")) and bool(row.get(f"predicted_{label}"))
        for row in selected
    )
    return 100.0 * true_positives / positives


def _summary_rows(
    specs: list[ToolSpec], existing: dict[tuple[str, str], dict[str, object]]
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for spec in specs:
        rows = [row for (tool, _), row in existing.items() if tool == spec.key and row.get("status") == "ok"]
        elapsed = sum(float(row.get("elapsed_seconds") or 0.0) for row in rows)
        average = elapsed / len(rows) if rows else None
        summary.append({
            "Ferramenta": spec.label,
            "RR (T)": _recall(rows, "runtime", True),
            "RS (T)": _recall(rows, "syntax", True),
            "RR (G)": _recall(rows, "runtime", False),
            "RS (G)": _recall(rows, "syntax", False),
            "Avg T": average,
        })
    return summary


def _format_percent(value: object) -> str:
    return "—" if value is None else f"{float(value):.1f}%"


def _render_table(summary: list[dict[str, object]]) -> Table:
    table = Table(title="Phase 5 — Static analysis", header_style="bold cyan")
    for field in SUMMARY_FIELDS:
        table.add_column(field, justify="left" if field == "Ferramenta" else "center")
    for row in summary:
        average = row["Avg T"]
        table.add_row(
            str(row["Ferramenta"]),
            _format_percent(row["RR (T)"]),
            _format_percent(row["RS (T)"]),
            _format_percent(row["RR (G)"]),
            _format_percent(row["RS (G)"]),
            "—" if average is None else f"{float(average):.4f} s",
        )
    return table


def _write_summary(path: Path, summary: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary:
            writer.writerow({
                key: "" if row[key] is None else f"{float(row[key]):.6f}"
                if key != "Ferramenta" else row[key]
                for key in SUMMARY_FIELDS
            })
    temporary.replace(path)


def _batches(items: list[Sample], size: int) -> Iterable[list[Sample]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _write_sources(directory: Path, samples: list[Sample]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for index, sample in enumerate(samples):
        path = directory / f"sample_{index:06d}.py"
        path.write_text(sample.source, encoding="utf-8")
        paths[sample.sample_id] = path.resolve()
    return paths


def run_static_analysis(
    config: HalluCodeDetectionConfig,
    *,
    selected_tools: list[str] | None = None,
    retry: bool = False,
    batch_size: int = 250,
    limit: int | None = None,
) -> None:
    if batch_size < 1:
        raise ValueError("Phase-5 batch size must be at least 1.")
    samples = _load_samples(config, limit)
    available = tool_specs()
    unknown = set(selected_tools or []) - {spec.key for spec in available}
    if unknown:
        raise ValueError(f"Unknown Phase-5 tools: {', '.join(sorted(unknown))}")
    specs = [spec for spec in available if not selected_tools or spec.key in selected_tools]

    output_dir = Path(config.dataset_building_config.results_dir) / "phase5"
    details_path = output_dir / "static_analysis_results.jsonl"
    summary_path = output_dir / "static_analysis_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(details_path)

    pending_by_tool: dict[str, list[Sample]] = {}
    for spec in specs:
        pending_by_tool[spec.key] = [
            sample for sample in samples
            if (spec.key, sample.sample_id) not in existing
            or (retry and existing[(spec.key, sample.sample_id)].get("status") != "ok")
        ]
    total = sum(len(items) for items in pending_by_tool.values())
    completed = 0
    started = time.perf_counter()

    ui.console.print(
        f"[cyan]Phase 5:[/] {len(samples)} unique generations; "
        f"{sum(sample.is_test for sample in samples)} in the held-out test split."
    )
    versions = {spec.key: tool_version(spec) for spec in specs}
    for spec in specs:
        ui.console.print(f"  {spec.label}: {versions[spec.key]}")

    with tempfile.TemporaryDirectory(prefix="hallucode-phase5-") as temp_name:
        source_paths = _write_sources(Path(temp_name), samples)
        path_to_sample = {path: sample for sample in samples for path in [source_paths[sample.sample_id]]}
        progress_text = tqdm.format_meter(completed, total, 0.0, unit="check", ncols=88)
        live = Live(
            Group(Panel(progress_text, title="tqdm progress"), _render_table(_summary_rows(specs, existing))),
            console=ui.console,
            refresh_per_second=4,
        )
        with live:
            for spec in specs:
                # CrossHair starts one isolated process per target; commit it
                # more frequently so Ctrl+C loses at most ten checks.
                effective_batch_size = min(batch_size, 10) if spec.key == "crosshair" else batch_size
                for batch in _batches(pending_by_tool[spec.key], effective_batch_size):
                    paths = [source_paths[sample.sample_id] for sample in batch]
                    try:
                        findings, elapsed = run_tool(spec, paths)
                        per_sample_elapsed = elapsed / len(batch)
                        result_rows = []
                        for path in paths:
                            sample = path_to_sample[path]
                            finding = findings.get(path, ToolFinding())
                            result_rows.append({
                                "tool": spec.key,
                                "tool_label": spec.label,
                                "tool_version": versions[spec.key],
                                "sample_id": sample.sample_id,
                                "task_id": sample.task_id,
                                "is_test": sample.is_test,
                                "expected_runtime": "runtime" in sample.levels,
                                "expected_syntax": "syntax" in sample.levels,
                                "predicted_runtime": finding.runtime,
                                "predicted_syntax": finding.syntax,
                                "elapsed_seconds": per_sample_elapsed,
                                "diagnostics": finding.diagnostics
                                + ([finding.error] if finding.error else []),
                                "status": "error" if finding.error else "ok",
                            })
                    except Exception as exc:  # noqa: BLE001 - one failed batch must not abort Phase 5
                        result_rows = [{
                            "tool": spec.key,
                            "tool_label": spec.label,
                            "tool_version": versions[spec.key],
                            "sample_id": sample.sample_id,
                            "task_id": sample.task_id,
                            "is_test": sample.is_test,
                            "expected_runtime": "runtime" in sample.levels,
                            "expected_syntax": "syntax" in sample.levels,
                            "predicted_runtime": False,
                            "predicted_syntax": False,
                            "elapsed_seconds": 0.0,
                            "diagnostics": [f"{type(exc).__name__}: {exc}"],
                            "status": "error",
                        } for sample in batch]

                    append_jsonl(details_path, result_rows)
                    for row in result_rows:
                        existing[(str(row["tool"]), str(row["sample_id"]))] = row
                    completed += len(batch)
                    elapsed_total = time.perf_counter() - started
                    progress_text = tqdm.format_meter(
                        completed, total, elapsed_total, unit="check", ncols=88,
                        postfix=f"tool={spec.key}",
                    )
                    summary = _summary_rows(specs, existing)
                    _write_summary(summary_path, summary)
                    live.update(Group(Panel(progress_text, title="tqdm progress"), _render_table(summary)))

    summary = _summary_rows(specs, existing)
    _write_summary(summary_path, summary)
    ui.console.print(f"[green]Summary CSV:[/] {summary_path}")
    ui.console.print(f"[green]Per-sample diagnostics:[/] {details_path}")
