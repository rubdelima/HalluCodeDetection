from __future__ import annotations

import csv
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from rich.table import Table

from src.constants import HalluCodeDetectionConfig
from src.constants.models import ModelInfo
from src.core import ui
from src.dataset.build import count_levels, count_levels_by_model
from src.dataset.load import load_evalplus, sample_examples
from src.dataset.utils import append_jsonl, load_jsonl
from src.evaluations.classify import run_evalplus_tests
from src.models import get_model_handler
from src.models.base import BaseModelHandler
from src.models.opencode_go_handler import OpenCodeGoHandler
from src.schemas.dataset import BaseResultRow, Level
from src.schemas.evalplus import EvalPlusExample
from src.schemas.phase6 import Phase6ResultRow, Phase6Round
from src.static_analysis import analyze_with_compile_and_pyright


@dataclass(frozen=True)
class Phase6Task:
    example: EvalPlusExample
    model: ModelInfo

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.example.benchmark_name, self.example.benchmark_id, self.model.id)


def _render_summary_table(
    model_counts: dict[str, dict[str, int]],
    model_cycle_totals: dict[str, int],
) -> Table:
    table = Table(title="Phase 6 — Generation with static-tool feedback")
    table.add_column("Model", style="magenta")
    table.add_column("Avg Cycles", justify="center", style="cyan")
    table.add_column("Correct", justify="center", style="green")
    table.add_column("Functional", justify="center", style="yellow")
    table.add_column("Runtime", justify="center", style="orange3")
    table.add_column("Syntax", justify="center", style="red")
    for model in sorted(model_counts):
        counts = model_counts[model]
        completed = counts["total"]
        average = model_cycle_totals.get(model, 0) / completed if completed else 0.0
        table.add_row(
            model,
            f"{average:.2f}" if completed else "-",
            str(counts["correct"]),
            str(counts["functional"]),
            str(counts["runtime"]),
            str(counts["syntax"]),
        )
    return table


def _comparison_rows(
    phase6_rows: list[Phase6ResultRow],
    phase1_rows: list[BaseResultRow],
    model_ids: set[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    phase1_by_key = {
        (row.benchmark, row.benchmark_id, row.model): row for row in phase1_rows
    }
    rows: list[dict[str, str]] = []
    coverage: dict[str, int] = {}

    for model_id in sorted(model_ids):
        paired_phase6: list[BaseResultRow] = []
        paired_phase1: list[BaseResultRow] = []
        cycle_total = 0
        for row in phase6_rows:
            if row.model != model_id:
                continue
            current = _as_base_result(row)
            baseline = phase1_by_key.get(
                (row.benchmark, row.benchmark_id, row.model)
            )
            if current is None or baseline is None:
                continue
            paired_phase6.append(current)
            paired_phase1.append(baseline)
            cycle_total += len(row.rounds)

        total = len(paired_phase6)
        coverage[model_id] = total
        if not total:
            rows.append(
                {
                    "Model": model_id,
                    "Avg Cycles": "-",
                    "Correct": "-",
                    "Functional": "-",
                    "Runtime": "-",
                    "Syntax": "-",
                }
            )
            continue

        current_counts = count_levels(paired_phase6)
        baseline_counts = count_levels(paired_phase1)
        average_cycles = cycle_total / total
        result = {
            "Model": model_id,
            "Avg Cycles": f"{average_cycles:.2f} ({average_cycles - 1.0:+.2f})",
        }
        for label, heading in (
            ("correct", "Correct"),
            ("functional", "Functional"),
            ("runtime", "Runtime"),
            ("syntax", "Syntax"),
        ):
            current = current_counts[label] / total * 100
            baseline = baseline_counts[label] / total * 100
            result[heading] = f"{current:.1f}% ({current - baseline:+.1f})"
        rows.append(result)

    return rows, coverage


def _render_comparison_table(
    rows: list[dict[str, str]],
    coverage: dict[str, int],
) -> Table:
    table = Table(title="Phase 6 compared with Phase 1")
    table.add_column("Model", style="magenta")
    table.add_column("Avg Cycles", justify="center", style="cyan")
    table.add_column("Correct", justify="center", style="green")
    table.add_column("Functional", justify="center", style="yellow")
    table.add_column("Runtime", justify="center", style="orange3")
    table.add_column("Syntax", justify="center", style="red")
    for row in rows:
        table.add_row(*(row[field] for field in row))
    paired = ", ".join(f"{model}: n={coverage[model]}" for model in sorted(coverage))
    table.caption = (
        f"Phase-6 value (delta from Phase 1); class deltas are percentage points. {paired}."
    )
    return table


def _write_comparison_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["Model", "Avg Cycles", "Correct", "Functional", "Runtime", "Syntax"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(
    path: Path,
    model_counts: dict[str, dict[str, int]],
    model_cycle_totals: dict[str, int],
) -> None:
    fields = ["Model", "Avg Cycles", "Correct", "Functional", "Runtime", "Syntax"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in sorted(model_counts):
            counts = model_counts[model]
            completed = counts["total"]
            average = model_cycle_totals.get(model, 0) / completed if completed else 0.0
            writer.writerow(
                {
                    "Model": model,
                    "Avg Cycles": f"{average:.4f}" if completed else "",
                    "Correct": counts["correct"],
                    "Functional": counts["functional"],
                    "Runtime": counts["runtime"],
                    "Syntax": counts["syntax"],
                }
            )


def _as_base_result(row: Phase6ResultRow) -> BaseResultRow | None:
    if row.error or not row.levels:
        return None
    return BaseResultRow(
        benchmark=row.benchmark,
        benchmark_id=row.benchmark_id,
        task_id=row.task_id,
        model=row.model,
        levels=row.levels,
        tc_ok=row.tc_ok,
        tc_fail=row.tc_fail,
        code=row.final_code,
        base_passed=row.base_passed,
        plus_passed=row.plus_passed,
        exception_type=row.exception_type,
        failure_stage=row.failure_stage,
    )


def _evaluate_task(
    task: Phase6Task,
    handler: BaseModelHandler,
    *,
    max_rounds: int,
    generation_temperature: float,
    review_temperature: float,
    interaction_mode: str,
    tests_timeout: int,
    on_cycle: Callable[[Phase6Task, int], None] | None = None,
    initial_code: str | None = None,
) -> Phase6ResultRow:
    example = task.example
    rounds: list[Phase6Round] = []
    code = ""
    try:
        developer_history: list[dict[str, str]] | None = None
        analyzer_history: list[dict[str, str]] | None = None
        if initial_code is not None:
            code = initial_code.strip()
            if interaction_mode == "agentic":
                developer_history = handler.developer_agent_history(example, code)
                analyzer_history = handler.analyzer_agent_history()
        elif interaction_mode == "agentic":
            code, developer_history = handler.start_developer_agent(
                example, generation_temperature
            )
            analyzer_history = handler.analyzer_agent_history()
        else:
            code = handler.generate_code(example, generation_temperature).strip()
        for round_number in range(1, max_rounds + 1):
            feedback = analyze_with_compile_and_pyright(code, example)
            round_row = Phase6Round(
                round_number=round_number,
                code=code,
                tool_feedback=feedback,
            )
            rounds.append(round_row)
            if on_cycle is not None:
                on_cycle(task, round_number)
            if interaction_mode == "agentic":
                if developer_history is None:
                    developer_history = handler.developer_agent_history(example, code)
                if analyzer_history is None:
                    analyzer_history = handler.analyzer_agent_history()
                review = handler.analyze_with_analyzer_agent(
                    messages=analyzer_history,
                    example=example,
                    code=code,
                    tool_feedback=feedback.as_prompt(),
                    round_number=round_number,
                    max_rounds=max_rounds,
                    temperature=review_temperature,
                )
                round_row.decision = review.decision
                round_row.assessment = review.assessment
                round_row.review_raw_response = review.raw_response
                round_row.review_thoughts = review.thoughts
                round_row.critique = review.assessment
                round_row.critique_raw_response = review.raw_response
                if review.decision == "submit" or round_number == max_rounds:
                    break
                code = handler.revise_with_developer_feedback(
                    messages=developer_history,
                    assessment=review.assessment,
                    tool_feedback=feedback.as_prompt(),
                    round_number=round_number,
                    max_rounds=max_rounds,
                    temperature=generation_temperature,
                )
                continue

            if round_number == max_rounds:
                break

            review = handler.review_generated_code(
                example=example,
                code=code,
                tool_feedback=feedback.as_prompt(),
                round_number=round_number,
                max_rounds=max_rounds,
                temperature=review_temperature,
            )
            round_row.decision = review.decision
            round_row.assessment = review.assessment
            round_row.review_raw_response = review.raw_response
            round_row.review_thoughts = review.thoughts
            if review.decision == "submit":
                break
            code = review.code

        # EvalPlus is deliberately called only here. No test result is ever
        # included in a generation or self-review prompt.
        result = run_evalplus_tests(code, example, tests_timeout)
        return Phase6ResultRow(
            benchmark=example.benchmark_name,
            benchmark_id=example.benchmark_id,
            task_id=example.task_id,
            model=task.model.id,
            rounds=rounds,
            final_code=code,
            levels=[Level(level.level_name, list(level.evidences)) for level in result.levels],
            tc_ok=result.tc_ok,
            tc_fail=result.tc_fail,
            base_passed=result.base_passed,
            plus_passed=result.plus_passed,
            exception_type=result.exception_type,
            failure_stage=result.failure_stage,
        )
    except Exception as exc:  # noqa: BLE001 - persist failures for --retry
        return Phase6ResultRow(
            benchmark=example.benchmark_name,
            benchmark_id=example.benchmark_id,
            task_id=example.task_id,
            model=task.model.id,
            rounds=rounds,
            final_code=code,
            error=f"{type(exc).__name__}: {exc}"[:2_000],
        )


def _select_models(
    configured: list[ModelInfo], model_names: list[str] | None
) -> list[ModelInfo]:
    if model_names is None:
        return configured
    requested = set(model_names)
    selected = [model for model in configured if model.id in requested]
    missing = requested - {model.id for model in configured}
    if missing:
        available = ", ".join(model.id for model in configured)
        raise ValueError(
            f"Phase-6 model(s) not configured: {', '.join(sorted(missing))}. "
            f"Available: {available}"
        )
    return selected


def run_phase6(
    config: HalluCodeDetectionConfig,
    *,
    retry: bool = False,
    model_names: list[str] | None = None,
    limit: int | None = None,
    max_rounds: int | None = None,
) -> None:
    phase = config.phase6_config
    models = _select_models(phase.models, model_names)
    if phase.num_ctx is not None or phase.max_tokens is not None:
        models = [
            model.model_copy(update={
                **({"num_ctx": phase.num_ctx} if phase.num_ctx is not None else {}),
                **({"max_tokens": phase.max_tokens} if phase.max_tokens is not None else {}),
            })
            if model.type == "ollama" else model
            for model in models
        ]
    if not models:
        ui.console.print("[yellow]Skipping Phase 6: no models configured.[/]")
        return

    output_dir = Path(phase.results_dir or Path(config.dataset_building_config.results_dir) / "phase6")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = (
        "generation_with_tools_smoke.jsonl"
        if limit is not None or max_rounds is not None
        else "generation_with_tools.jsonl"
    )
    output_path = output_dir / output_name
    if phase.overwrite and output_path.exists():
        output_path.unlink()
    existing_rows = load_jsonl(output_path, Phase6ResultRow, quiet=True)
    existing = {
        (row.benchmark, row.benchmark_id, row.model): row for row in existing_rows
    }
    phase1_path = Path(phase.phase1_results_path)
    phase1_rows = load_jsonl(phase1_path, BaseResultRow, quiet=True)
    if not phase1_rows:
        raise FileNotFoundError(
            f"Phase-1 comparison results were not found at {phase1_path}"
        )
    # The models may be executed separately with --model_name, but their
    # aggregate table must remain shared and preserve previously run models.
    summary_model_ids = {
        model.id for model in phase.models
    } | {row.model for row in existing.values()}
    summary_existing_rows = list(existing.values())

    examples: list[EvalPlusExample] = []
    for benchmark in config.dataset_building_config.datasets_base:
        examples.extend(load_evalplus(benchmark))
    examples = sample_examples(examples, phase.dataset_load, phase.random_seed)
    if limit is not None:
        if limit < 1:
            raise ValueError("--phase6_limit must be at least 1.")
        examples = examples[:limit]

    tasks = [Phase6Task(example, model) for model in models for example in examples]
    tasks = [
        task for task in tasks
        if task.key not in existing or (retry and existing[task.key].error is not None)
    ]
    completed_base = [
        item for row in summary_existing_rows if (item := _as_base_result(row))
    ]
    model_counts = count_levels_by_model(completed_base)
    model_cycle_totals = {
        model_id: sum(
            len(row.rounds)
            for row in summary_existing_rows
            if row.model == model_id and _as_base_result(row) is not None
        )
        for model_id in summary_model_ids
    }
    for model_id in summary_model_ids:
        model_counts.setdefault(
            model_id,
            {"correct": 0, "functional": 0, "runtime": 0, "syntax": 0, "total": 0},
        )
    totals = {"correct": 0, "functional": 0, "runtime": 0, "syntax": 0, "total": 0}
    for counts in model_counts.values():
        for key in totals:
            totals[key] += counts[key]
    rounds_to_run = max_rounds if max_rounds is not None else phase.max_rounds
    if rounds_to_run < 1:
        raise ValueError("--phase6_rounds must be at least 1.")
    summary_path = output_dir / output_name.replace(".jsonl", "_summary.csv")
    comparison_path = output_dir / output_name.replace(".jsonl", "_comparison.csv")

    def refresh_comparison() -> tuple[list[dict[str, str]], dict[str, int]]:
        comparison, coverage = _comparison_rows(
            list(existing.values()), phase1_rows, summary_model_ids
        )
        _write_comparison_csv(comparison_path, comparison)
        return comparison, coverage

    if not tasks:
        _write_summary_csv(summary_path, model_counts, model_cycle_totals)
        comparison, coverage = refresh_comparison()
        ui.console.print(_render_comparison_table(comparison, coverage))
        ui.console.print("[green]All selected Phase-6 tasks are already complete.[/]")
        return

    ui.console.print(
        f"[cyan]Phase 6: {len(tasks)} pending task(s), up to "
        f"{rounds_to_run} cycle(s) per task. The task counter advances only after "
        "all of its cycles and EvalPlus finish.[/]"
    )
    progress, progress_id = ui.build_phase6_progress(
        len(tasks), totals, rounds_to_run
    )
    progress_lock = Lock()
    observed_cycles = 0

    def report_cycle(task: Phase6Task, round_number: int) -> None:
        nonlocal observed_cycles
        with progress_lock:
            observed_cycles += 1
            progress.update(
                progress_id,
                model=task.model.id,
                bench=task.example.benchmark_name,
                bench_id=str(task.example.benchmark_id),
                round=str(round_number),
                cycles=observed_cycles,
            )

    def record(task: Phase6Task, row: Phase6ResultRow) -> None:
        append_jsonl(output_path, [row.model_dump()])
        existing[task.key] = row
        if row.error:
            ui.console.print(
                f"[yellow]Phase 6 skipped {row.task_id} ({row.model}): {row.error}[/]"
            )
        else:
            model_cycle_totals[row.model] += len(row.rounds)
            names = {level.level_name for level in row.levels}
            model_counts[row.model]["total"] += 1
            totals["total"] += 1
            if names == {"correct"}:
                model_counts[row.model]["correct"] += 1
                totals["correct"] += 1
            else:
                for name in names:
                    if name in totals and name != "correct":
                        model_counts[row.model][name] += 1
                        totals[name] += 1
        # Keep the aggregate resumable too; a later Ctrl+C cannot lose the
        # summary for tasks that have already reached the JSONL checkpoint.
        _write_summary_csv(summary_path, model_counts, model_cycle_totals)
        refresh_comparison()
        progress.update(
            progress_id,
            advance=1,
            model=task.model.id,
            bench=task.example.benchmark_name,
            bench_id=str(task.example.benchmark_id),
            c=totals["correct"], f=totals["functional"],
            r=totals["runtime"], s=totals["syntax"],
        )

    started = time.monotonic()
    with progress:
        for model in models:
            model_tasks = [task for task in tasks if task.model.id == model.id]
            if not model_tasks:
                continue
            handler = get_model_handler(model)
            try:
                if model.type == "opencode_go" and model.threads > 1:
                    # Probe one real task before creating a burst of remote
                    # requests. Its generated code is reused, so the probe is
                    # part of the experiment rather than an extra API call.
                    first_task = model_tasks.pop(0)
                    if not isinstance(handler, OpenCodeGoHandler):
                        raise TypeError("OpenCode Go model has an incompatible handler.")
                    ui.console.print(
                        f"[cyan]Checking {model.id} availability (30s maximum)...[/]"
                    )
                    try:
                        first_code = handler.generate_code_with_request_policy(
                            first_task.example,
                            phase.generation_temperature,
                            request_timeout=min(model.request_timeout, 30),
                            max_retries=0,
                        )
                    except Exception as exc:  # noqa: BLE001
                        ui.console.print(
                            f"[red]Phase 6 did not start {model.id}: the provider "
                            f"failed its availability check ({type(exc).__name__}: {exc}). "
                            "No failed experiment row was saved; rerun later.[/]"
                        )
                        continue
                    first_row = _evaluate_task(
                        first_task,
                        handler,
                        max_rounds=rounds_to_run,
                        generation_temperature=phase.generation_temperature,
                        review_temperature=phase.review_temperature,
                        interaction_mode=phase.interaction_mode,
                        tests_timeout=config.dataset_building_config.tests_timeout,
                        on_cycle=report_cycle,
                        initial_code=first_code,
                    )
                    record(first_task, first_row)
                    if not model_tasks:
                        continue
                    executor = ThreadPoolExecutor(
                        max_workers=model.threads,
                        thread_name_prefix=f"phase6-{model.id}",
                    )
                    futures = {
                        executor.submit(
                            _evaluate_task,
                            task,
                            handler,
                            max_rounds=rounds_to_run,
                            generation_temperature=phase.generation_temperature,
                            review_temperature=phase.review_temperature,
                            interaction_mode=phase.interaction_mode,
                            tests_timeout=config.dataset_building_config.tests_timeout,
                            on_cycle=report_cycle,
                        ): task
                        for task in model_tasks
                    }
                    try:
                        for future in as_completed(futures):
                            task = futures[future]
                            record(task, future.result())
                    except BaseException:
                        for future in futures:
                            future.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    else:
                        executor.shutdown(wait=True)
                else:
                    for task in model_tasks:
                        row = _evaluate_task(
                            task,
                            handler,
                            max_rounds=rounds_to_run,
                            generation_temperature=phase.generation_temperature,
                            review_temperature=phase.review_temperature,
                            interaction_mode=phase.interaction_mode,
                            tests_timeout=config.dataset_building_config.tests_timeout,
                            on_cycle=report_cycle,
                        )
                        record(task, row)
            finally:
                handler.close()

    _write_summary_csv(summary_path, model_counts, model_cycle_totals)
    comparison, coverage = refresh_comparison()
    ui.console.print(_render_comparison_table(comparison, coverage))
    ui.console.print(
        f"[green]Phase 6 finished in {time.monotonic() - started:.1f}s:[/] "
        f"{output_path}, {summary_path}, and {comparison_path}"
    )
