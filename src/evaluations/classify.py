from __future__ import annotations

import base64
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from src.core.log import get_logger
from src.schemas.dataset import Level
from src.schemas.evalplus import EvalPlusExample

log = get_logger("classify")


@dataclass(frozen=True)
class ClassificationResult:
    levels: list[Level]
    tc_ok: int
    tc_fail: int
    base_passed: bool | None
    plus_passed: bool | None
    exception_type: str | None
    failure_stage: str | None

    @property
    def primary_level_name(self) -> str:
        return self.levels[0].level_name if self.levels else "correct"


def _remove_fences(code: str) -> str:
    lines = code.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _indent_body(body: str) -> str:
    return "\n".join(f"    {line}" if line.strip() else "" for line in body.splitlines())


def _normalize_completion_body(code: str) -> str:
    lines = _remove_fences(code).splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return "pass"
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if min(indents) == 0:
        return "\n".join(("    " + line if line.strip() else "") for line in lines)
    return "\n".join(lines)


def _has_entry_point_definition(code: str, entry_point: str) -> bool:
    return re.search(
        rf"^\s*def\s+{re.escape(entry_point)}\s*\(",
        code,
        re.MULTILINE,
    ) is not None


def _build_candidate_code(example: EvalPlusExample, generated: str) -> str:
    cleaned = _remove_fences(generated).strip()
    if _has_entry_point_definition(cleaned, example.entry_point):
        return cleaned
    if example.is_completion:
        body = _normalize_completion_body(generated)
        return example.prompt.rstrip() + "\n" + body
    if example.function_signature:
        return f"{example.function_signature}\n{_indent_body(cleaned)}"
    return cleaned


def build_candidate_code(example: EvalPlusExample, generated: str) -> str:
    """Build the complete Python source used by EvalPlus without executing it."""
    return _build_candidate_code(example, generated)


_HARNESS = r"""
import base64
import copy
import json
import math
import pickle
import resource
import sys

try:
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
except Exception:
    pass

CANDIDATE_CODE = {candidate_code!r}
CANONICAL_CODE = {canonical_code!r}
ENTRY_POINT = {entry_point!r}
DATASET = {dataset!r}
ATOL = {atol!r}
OUTPUT_NOT_NONE = {output_not_none!r}
PAYLOAD = {payload!r}

MAX_EVIDENCES = 20

_ctx = pickle.loads(base64.b64decode(PAYLOAD))
BASE_INPUTS = _ctx["base_input"]
PLUS_INPUTS = _ctx["plus_input"]

MBPP_OUTPUT_NOT_NONE_TASKS = {{"check_str", "text_match_three", "text_starta_endb"}}


def _is_floats(x):
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) for i in x)
    return False


def _poly(xs, x):
    return sum(coeff * math.pow(x, i) for i, coeff in enumerate(xs))


def _allclose(a, b, atol):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_allclose(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= atol
        except (TypeError, ValueError):
            return False
    return a == b


def _match(out, exp, inp, atol, entry_point, dataset):
    exact_match = out == exp
    if dataset == "mbpp":
        if entry_point == "are_equivalent":
            exact_match = True
        elif entry_point == "sum_div":
            exact_match = exact_match or out == 0
    if dataset == "humaneval" and entry_point == "find_zero":
        try:
            exact_match = _poly(*inp, out) <= atol
        except Exception:
            exact_match = False
    effective_atol = atol
    if atol == 0 and _is_floats(exp):
        effective_atol = 1e-6
    if not exact_match and effective_atol != 0:
        exact_match = _allclose(out, exp, effective_atol)
    return exact_match


candidate_ns = {{}}
try:
    exec(CANDIDATE_CODE, candidate_ns)
    candidate_fn = candidate_ns[ENTRY_POINT]
except BaseException as exc:
    print(json.dumps({{
        "exec_error": type(exc).__name__,
        "message": str(exc),
    }}))
    sys.exit(0)

canonical_ns = {{}}
try:
    exec(CANONICAL_CODE, canonical_ns)
    canonical_fn = canonical_ns[ENTRY_POINT]
except BaseException:
    canonical_fn = None


def run(inputs, stage):
    runtime_seen = None
    passed = 0
    total = 0
    runtime_evidences = []
    functional_evidences = []
    for inp in inputs:
        total += 1
        try:
            expected = canonical_fn(*copy.deepcopy(inp))
            if OUTPUT_NOT_NONE:
                expected = expected is not None
        except BaseException:
            expected = None
        try:
            out = candidate_fn(*copy.deepcopy(inp))
        except BaseException as exc:
            if runtime_seen is None:
                runtime_seen = type(exc).__name__
            if len(runtime_evidences) < MAX_EVIDENCES:
                runtime_evidences.append(
                    "[{{}}] {{}}: {{}} (input={{!r}})".format(stage, type(exc).__name__, exc, inp)
                )
            continue
        if canonical_fn is None:
            continue
        if OUTPUT_NOT_NONE:
            out = out is not None
        if _match(out, expected, inp, ATOL, ENTRY_POINT, DATASET):
            passed += 1
        elif len(functional_evidences) < MAX_EVIDENCES:
            functional_evidences.append(
                "[{{}}] input={{!r}}: expected {{!r}}, got {{!r}}".format(stage, inp, expected, out)
            )
    return passed, total, runtime_evidences, functional_evidences, runtime_seen


base_passed_count, base_total, base_runtime_ev, base_func_ev, base_runtime = run(BASE_INPUTS, "base")
plus_passed_count, plus_total, plus_runtime_ev, plus_func_ev, plus_runtime = run(PLUS_INPUTS, "plus")
print(json.dumps({{
    "base_passed_count": base_passed_count,
    "base_total": base_total,
    "plus_passed_count": plus_passed_count,
    "plus_total": plus_total,
    "runtime_evidences": base_runtime_ev + plus_runtime_ev,
    "functional_evidences": base_func_ev + plus_func_ev,
    "runtime_seen": base_runtime or plus_runtime,
}}))
"""


def _encode_payload(example: EvalPlusExample) -> str:
    payload = {
        "base_input": example.base_input,
        "plus_input": example.plus_input,
    }
    return base64.b64encode(pickle.dumps(payload)).decode("ascii")


def _run_harness(
    candidate_code: str,
    example: EvalPlusExample,
    timeout_seconds: int,
) -> dict[str, Any]:
    harness = _HARNESS.format(
        candidate_code=candidate_code,
        canonical_code=example.canonical_solution,
        entry_point=example.entry_point,
        dataset=example.benchmark_name,
        atol=example.atol,
        output_not_none=example.entry_point in _OUTPUT_NOT_NONE_TASKS,
        payload=_encode_payload(example),
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(harness)
        temp_path = handle.name

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        log.warning("Classification subprocess timed out after %ds (%s/%s).",
                    timeout_seconds, example.benchmark_name, example.benchmark_id)
        return {"timeout": True}
    finally:
        os.unlink(temp_path)
    log.debug("Classification subprocess for %s/%s finished in %.2fs.",
              example.benchmark_name, example.benchmark_id, time.monotonic() - t0)

    stdout = (result.stdout or "").strip()
    for line in stdout.splitlines():
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    stderr = (result.stderr or "").strip()
    log.warning("Classification subprocess produced no result for %s/%s (rc=%s, stderr=%r).",
                example.benchmark_name, example.benchmark_id, result.returncode, stderr[:500])
    return {"stderr": stderr}


_OUTPUT_NOT_NONE_TASKS = {"check_str", "text_match_three", "text_starta_endb"}


def _total_tests(example: EvalPlusExample) -> int:
    return len(example.base_input) + len(example.plus_input)


def run_evalplus_tests(
    code: str,
    example: EvalPlusExample,
    timeout_seconds: int = 30,
) -> ClassificationResult:
    candidate_code = _build_candidate_code(example, code)
    total = _total_tests(example)

    try:
        compile(candidate_code, "<generated>", "exec")
    except SyntaxError as exc:
        return ClassificationResult(
            levels=[Level("syntax", [f"SyntaxError: {exc.msg} (line {exc.lineno})"])],
            tc_ok=0,
            tc_fail=total,
            base_passed=False,
            plus_passed=False,
            exception_type=type(exc).__name__,
            failure_stage="syntax",
        )

    result = _run_harness(candidate_code, example, timeout_seconds)

    if result.get("timeout"):
        return ClassificationResult(
            levels=[Level("runtime", ["Timeout"])],
            tc_ok=0,
            tc_fail=total,
            base_passed=None,
            plus_passed=None,
            exception_type="Timeout",
            failure_stage="runtime",
        )

    if "exec_error" in result:
        evidence = f"{result['exec_error']}: {result.get('message', '')}".strip()
        return ClassificationResult(
            levels=[Level("runtime", [evidence])],
            tc_ok=0,
            tc_fail=total,
            base_passed=False,
            plus_passed=False,
            exception_type=result["exec_error"],
            failure_stage="runtime",
        )

    if "base_total" not in result:
        stderr = result.get("stderr", "")
        return ClassificationResult(
            levels=[Level("runtime", [stderr or "Execution failed"])],
            tc_ok=0,
            tc_fail=total,
            base_passed=None,
            plus_passed=None,
            exception_type=None,
            failure_stage="runtime",
        )

    base_passed_count = int(result["base_passed_count"])
    base_total = int(result["base_total"])
    plus_passed_count = int(result["plus_passed_count"])
    plus_total = int(result["plus_total"])
    runtime_evidences = [str(e) for e in result.get("runtime_evidences", [])]
    functional_evidences = [str(e) for e in result.get("functional_evidences", [])]
    runtime_seen = result.get("runtime_seen")

    tc_ok = base_passed_count + plus_passed_count
    tc_fail = (base_total + plus_total) - tc_ok
    base_passed = base_passed_count == base_total
    plus_passed = plus_passed_count == plus_total

    levels: list[Level] = []
    if runtime_evidences:
        levels.append(Level("runtime", runtime_evidences))
    if functional_evidences:
        levels.append(Level("functional", functional_evidences))
    if not levels:
        levels.append(Level("correct", []))

    if runtime_evidences:
        failure_stage = "runtime"
        exception_type = runtime_seen
    elif not base_passed:
        failure_stage = "base"
        exception_type = None
    elif not plus_passed:
        failure_stage = "plus"
        exception_type = None
    else:
        failure_stage = None
        exception_type = None

    return ClassificationResult(
        levels=levels,
        tc_ok=tc_ok,
        tc_fail=tc_fail,
        base_passed=base_passed,
        plus_passed=plus_passed,
        exception_type=exception_type,
        failure_stage=failure_stage,
    )
