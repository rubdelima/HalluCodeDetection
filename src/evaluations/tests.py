from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import Iterable


def _strip_code_fences(code: str) -> str:
    fenced = re.match(r"^```(?:python)?\s*(.*?)```\s*$", code, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return code.strip()


def _wrap_in_signature(body: str, signature: str) -> str:
    if not body.strip():
        return f"{signature}\n    pass"
    indented = "\n".join(f"    {line}" if line.strip() else "" for line in body.splitlines())
    return f"{signature}\n{indented}"


def _serialize_tests(tests: Iterable[str]) -> str:
    return "[" + ", ".join(repr(test) for test in tests) + "]"


def build_executable(code: str, tests: list[str], signature: str | None) -> str:
    cleaned = _strip_code_fences(code)
    lines = cleaned.splitlines()
    first_line = lines[0].strip() if lines else ""

    if signature and not first_line.startswith("def "):
        cleaned = _wrap_in_signature(cleaned, signature)

    tests_literal = _serialize_tests(tests)
    harness = """
import ast
import sys

__tests = {tests_literal}

def __assert_with_values(test_src: str) -> str:
    node = ast.parse(test_src).body[0]
    if not isinstance(node, ast.Assert):
        return f"FunctionalError in test case: {{test_src}}"
    expr = node.test
    if (
        isinstance(expr, ast.Compare)
        and len(expr.ops) == 1
        and isinstance(expr.ops[0], ast.Eq)
        and len(expr.comparators) == 1
    ):
        left_val = eval(compile(ast.Expression(expr.left), "<assert>", "eval"), globals(), globals())
        right_val = eval(
            compile(ast.Expression(expr.comparators[0]), "<assert>", "eval"),
            globals(),
            globals(),
        )
        return (
            "FunctionalError in test case: "
            f"{{test_src}}. Was expected {{right_val!r}} but was obtained {{left_val!r}}"
        )
    return f"FunctionalError in test case: {{test_src}}"

__failures = []

for __test in __tests:
    try:
        exec(__test, globals(), globals())
    except AssertionError:
        __failures.append(__assert_with_values(__test))
    except Exception:
        raise

if __failures:
    sys.stderr.write("FUNCTIONAL_ERROR\\n" + "\\n".join(__failures))
    sys.exit(1)
""".format(tests_literal=tests_literal).strip()

    return "\n\n".join([cleaned, "# Tests", harness])


def _format_syntax_error(exc: SyntaxError, source: str) -> str:
    line = ""
    if exc.lineno is not None:
        lines = source.splitlines()
        if 1 <= exc.lineno <= len(lines):
            line = lines[exc.lineno - 1]
    pointer = ""
    if exc.offset is not None and line:
        pointer = " " * (exc.offset - 1) + "^"
    location = f"line {exc.lineno}" if exc.lineno else "unknown line"
    details = "\n".join(part for part in [line, pointer] if part)
    if details:
        return f"SyntaxError: {exc.msg} ({location})\n{details}"
    return f"SyntaxError: {exc}"


def run_tests(
    code: str,
    tests: list[str],
    timeout_seconds: int,
    signature: str | None,
) -> tuple[str, str]:
    combined = build_executable(code, tests, signature)
    try:
        compile(combined, "<generated>", "exec")
    except SyntaxError as exc:
        return "syntax_error", _format_syntax_error(exc, combined)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(combined)
        temp_path = handle.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return "runtime_error", "Timeout"
    finally:
        os.unlink(temp_path)

    if result.returncode == 0:
        return "correct", ""

    stderr = (result.stderr or "").strip()
    if stderr.startswith("FUNCTIONAL_ERROR"):
        return "functional_error", stderr.split("\n", 1)[1].strip()
    if "AssertionError" in stderr:
        return "functional_error", stderr
    if "SyntaxError" in stderr:
        return "syntax_error", stderr
    return "runtime_error", stderr
