"""Check static-tool feedback consistency for the tool-assisted experiments.

The static-tool survey (phase 5) analyzes the **raw** model output, whereas the
tool-assisted generation and verification experiments (phases 6 and 7) analyze
the **reconstructed candidate** produced by ``build_candidate_code``.  The
candidate is the exact source that the EvalPlus harness executes and is also the
source used to build the feedback that is shown to the model.

This script re-derives both analyses for every code that appears in phases 6
and 7 and flags every example where the two forms disagree.  Those are exactly
the examples affected by the raw-vs-candidate inconsistency.  No language model
is executed: only ``compile`` and ``pyright`` are run.

Outputs
-------
A JSON file (``--output``) with the lists of examples that can be kept and the
examples that should be re-evaluated, plus a short summary printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.load import load_all_examples
from src.evaluations.classify import build_candidate_code
from src.static_analysis.tools import run_tool, tool_specs


def analyze_source(source: str) -> dict[str, object]:
    """Run compile + pyright on an arbitrary source string."""
    specs = {spec.key: spec for spec in tool_specs()}
    with tempfile.TemporaryDirectory(prefix="feedback-check-") as temp_name:
        path = Path(temp_name) / "candidate.py"
        path.write_text(source, encoding="utf-8")

        compile_findings, _ = run_tool(specs["compile"], [path])
        compile_finding = compile_findings[path.resolve()]
        if compile_finding.syntax:
            return {
                "compile_ok": False,
                "pyright_ok": None,
                "diagnostics": list(compile_finding.diagnostics),
            }

        pyright_findings, _ = run_tool(specs["pyright"], [path])
        pyright_finding = pyright_findings[path.resolve()]
        return {
            "compile_ok": True,
            "pyright_ok": not pyright_finding.runtime,
            "diagnostics": list(pyright_finding.diagnostics),
        }


def same_analysis(left: dict[str, object], right: dict[str, object]) -> bool:
    return (
        left["compile_ok"] == right["compile_ok"]
        and left["pyright_ok"] == right["pyright_ok"]
        and left["diagnostics"] == right["diagnostics"]
    )


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_phase6(path: Path, examples: dict, limit: int | None) -> dict[str, list]:
    list_phase6_ok: list[dict[str, object]] = []
    list_phase6_error: list[dict[str, object]] = []
    rows = load_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    for row in rows:
        if row.get("error"):
            continue
        example = examples.get((row.get("benchmark"), row.get("benchmark_id")))
        if example is None:
            list_phase6_error.append({
                "file": str(path),
                "model": row.get("model"),
                "task_id": row.get("task_id"),
                "reason": "evalplus example not found",
            })
            continue
        divergent_rounds: list[dict[str, object]] = []
        for round_row in row.get("rounds", []):
            raw = round_row.get("code", "")
            candidate = build_candidate_code(example, raw)
            raw_analysis = analyze_source(raw)
            candidate_analysis = analyze_source(candidate)
            if not same_analysis(raw_analysis, candidate_analysis):
                divergent_rounds.append({
                    "round_number": round_row.get("round_number"),
                    "stored_feedback": round_row.get("tool_feedback"),
                    "raw_analysis": raw_analysis,
                    "candidate_analysis": candidate_analysis,
                })
        record = {
            "file": str(path),
            "model": row.get("model"),
            "task_id": row.get("task_id"),
            "benchmark": row.get("benchmark"),
            "benchmark_id": row.get("benchmark_id"),
        }
        if divergent_rounds:
            record["divergent_rounds"] = divergent_rounds
            list_phase6_error.append(record)
        else:
            list_phase6_ok.append(record)
    return {"ok": list_phase6_ok, "error": list_phase6_error}


def check_phase7(path: Path, examples: dict, base_by_sample: dict, limit: int | None) -> dict[str, list]:
    list_phase7_ok: list[dict[str, object]] = []
    list_phase7_error: list[dict[str, object]] = []
    rows = load_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    for row in rows:
        if row.get("error"):
            continue
        example = examples.get((row.get("benchmark"), row.get("benchmark_id")))
        base = base_by_sample.get(str(row.get("sample_id")))
        if example is None or base is None:
            list_phase7_error.append({
                "file": str(path),
                "model_id": row.get("model_id"),
                "sample_id": row.get("sample_id"),
                "reason": "evalplus example or base code not found",
            })
            continue
        raw = str(base.get("code", ""))
        candidate = build_candidate_code(example, raw)
        raw_analysis = analyze_source(raw)
        candidate_analysis = analyze_source(candidate)
        record = {
            "file": str(path),
            "model_id": row.get("model_id"),
            "sample_id": row.get("sample_id"),
            "response_model": row.get("response_model"),
            "task_id": row.get("task_id"),
            "stored_feedback": row.get("tool_feedback"),
        }
        if not same_analysis(raw_analysis, candidate_analysis):
            record["raw_analysis"] = raw_analysis
            record["candidate_analysis"] = candidate_analysis
            list_phase7_error.append(record)
        else:
            list_phase7_ok.append(record)
    return {"ok": list_phase7_ok, "error": list_phase7_error}


DEFAULT_PHASE6 = [
    "results/phase6_local_agentic/generation_with_tools.jsonl",
    "results/phase6_local_tools_legacy_direct/generation_with_tools.jsonl",
    "data/results/evalplus/phase6/generation_with_tools.jsonl",
]
DEFAULT_PHASE7 = [
    "results/phase7_local_tools/classification_with_tools.jsonl",
    "data/results/evalplus/phase7/classification_with_tools.jsonl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase6", action="append", default=None,
                        help="Phase-6 JSONL; repeatable. Defaults to the known result files.")
    parser.add_argument("--phase7", action="append", default=None,
                        help="Phase-7 JSONL; repeatable. Defaults to the known result files.")
    parser.add_argument("--base", type=str, default="data/results/evalplus/dataset_base.json",
                        help="Raw phase-1 generations used by phase 7.")
    parser.add_argument("--output", type=str, default="results/feedback_consistency.json",
                        help="Where to write the JSON report.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum rows per file (smoke test only).")
    args = parser.parse_args()

    examples = load_all_examples(["humaneval", "mbpp"])
    base_rows = load_jsonl(Path(args.base))
    base_by_sample = {
        f"{row['benchmark']}:{row['benchmark_id']}:{row['model']}": row
        for row in base_rows
    }

    phase6_paths = [Path(p) for p in (args.phase6 or DEFAULT_PHASE6)]
    phase7_paths = [Path(p) for p in (args.phase7 or DEFAULT_PHASE7)]

    report: dict[str, object] = {"phase6": {}, "phase7": {}}
    for path in phase6_paths:
        result = check_phase6(path, examples, args.limit)
        report["phase6"][str(path)] = result
        print(f"[phase6] {path}: ok={len(result['ok'])} error={len(result['error'])}")
    for path in phase7_paths:
        result = check_phase7(path, examples, base_by_sample, args.limit)
        report["phase7"][str(path)] = result
        print(f"[phase7] {path}: ok={len(result['ok'])} error={len(result['error'])}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
