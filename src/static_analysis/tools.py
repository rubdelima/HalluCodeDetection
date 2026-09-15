from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


@dataclass
class ToolFinding:
    syntax: bool = False
    runtime: bool = False
    diagnostics: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    key: str
    label: str
    executable: str | None
    command: Callable[[list[Path]], list[str]] | None
    parser: Callable[[str, str, Iterable[Path]], dict[Path, ToolFinding]] | None
    accepted_exit_codes: frozenset[int] = frozenset({0, 1})
    timeout_per_sample: float = 2.0


PYLINT_RUNTIME_CODES = {
    "E0601", "E0602", "E0606", "E1101", "E1111", "E1120", "E1121",
    "E1123", "E1124", "E1125", "E1126", "E1127", "E1128", "E1129",
    "E1130", "E1131", "E1132", "E1133", "E1134", "E1135", "E1136",
    "E1137", "E1138", "E1139", "E1140", "E1200", "E1205", "E1206",
    "E1300", "E1301", "E1302", "E1303", "E1304", "E1305", "E1306",
    "E1307", "E1310", "E1507",
}

PYRIGHT_RUNTIME_RULES = {
    "reportArgumentType", "reportAssignmentType", "reportAttributeAccessIssue",
    "reportCallIssue", "reportGeneralTypeIssues", "reportIndexIssue",
    "reportInvalidTypeArguments", "reportOperatorIssue", "reportOptionalCall",
    "reportOptionalContextManager", "reportOptionalEnter", "reportOptionalIterable",
    "reportOptionalMemberAccess", "reportOptionalOperand", "reportOptionalSubscript",
    "reportPossiblyUnboundVariable", "reportReturnType", "reportUndefinedVariable",
    "reportUnboundVariable",
}

RUFF_RUNTIME_CODES = {"F405", "F406", "F821", "F822", "F823"}


def empty_findings(paths: Iterable[Path]) -> dict[Path, ToolFinding]:
    return {path.resolve(): ToolFinding() for path in paths}


def add_finding(
    findings: dict[Path, ToolFinding],
    raw_path: str | Path,
    *,
    syntax: bool = False,
    runtime: bool = False,
    diagnostic: str,
) -> None:
    path = Path(raw_path).resolve()
    finding = findings.get(path)
    if finding is None:
        return
    finding.syntax |= syntax
    finding.runtime |= runtime
    if diagnostic and len(finding.diagnostics) < 20:
        finding.diagnostics.append(diagnostic)


def parse_ruff(stdout: str, stderr: str, paths: Iterable[Path]) -> dict[Path, ToolFinding]:
    findings = empty_findings(paths)
    try:
        messages = json.loads(stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid Ruff JSON: {exc}; stderr={stderr[-500:]}") from exc
    for message in messages:
        code = str(message.get("code") or "")
        text = str(message.get("message") or code)
        add_finding(
            findings,
            message.get("filename", ""),
            syntax=code == "E999" or code == "invalid-syntax" or "syntax error" in text.lower(),
            runtime=code in RUFF_RUNTIME_CODES,
            diagnostic=f"{code}: {text}".strip(": "),
        )
    return findings


def parse_pyright(stdout: str, stderr: str, paths: Iterable[Path]) -> dict[Path, ToolFinding]:
    findings = empty_findings(paths)
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid Pyright JSON: {exc}; stderr={stderr[-500:]}") from exc
    for message in payload.get("generalDiagnostics", []):
        rule = str(message.get("rule") or "")
        text = str(message.get("message") or rule)
        lowered = text.lower()
        is_syntax = not rule and any(
            marker in lowered
            for marker in ("expected", "unexpected indentation", "invalid syntax", "unterminated")
        )
        is_runtime = rule in PYRIGHT_RUNTIME_RULES
        if not is_syntax and not is_runtime:
            continue
        add_finding(
            findings,
            message.get("file", ""),
            syntax=is_syntax,
            runtime=is_runtime,
            diagnostic=f"{rule}: {text}".strip(": "),
        )
    return findings


def parse_pylint(stdout: str, stderr: str, paths: Iterable[Path]) -> dict[Path, ToolFinding]:
    findings = empty_findings(paths)
    try:
        messages = json.loads(stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid Pylint JSON: {exc}; stderr={stderr[-500:]}") from exc
    for message in messages:
        code = str(message.get("message-id") or "")
        text = str(message.get("message") or code)
        add_finding(
            findings,
            message.get("path") or message.get("absolutePath") or "",
            syntax=code in {"E0001", "F0010"},
            runtime=code in PYLINT_RUNTIME_CODES,
            diagnostic=f"{code}: {text}".strip(": "),
        )
    return findings


def parse_semgrep(stdout: str, stderr: str, paths: Iterable[Path]) -> dict[Path, ToolFinding]:
    findings = empty_findings(paths)
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid Semgrep JSON: {exc}; stderr={stderr[-500:]}") from exc
    for result in payload.get("results", []):
        extra = result.get("extra") or {}
        add_finding(
            findings,
            result.get("path", ""),
            runtime=True,
            diagnostic=f"{result.get('check_id', '')}: {extra.get('message', '')}".strip(": "),
        )
    for error in payload.get("errors", []):
        error_type = str(error.get("type") or error.get("error_type") or "")
        text = str(error.get("message") or error)
        lowered = f"{error_type} {text}".lower()
        raw_path = error.get("path")
        if not raw_path and isinstance(error.get("location"), dict):
            raw_path = error["location"].get("path")
        if raw_path and any(marker in lowered for marker in ("syntax", "parse error", "parsing")):
            add_finding(
                findings,
                raw_path,
                syntax=True,
                diagnostic=f"{error_type}: {text}".strip(": "),
            )
    return findings


CROSSHAIR_LINE = re.compile(r"^(.*\.py):(\d+):\s+error:\s+(.*)$")


def parse_crosshair(stdout: str, stderr: str, paths: Iterable[Path]) -> dict[Path, ToolFinding]:
    findings = empty_findings(paths)
    for line in stdout.splitlines():
        match = CROSSHAIR_LINE.match(line.strip())
        if not match:
            continue
        raw_path, _, message = match.groups()
        lowered = message.lower()
        is_syntax = any(marker in lowered for marker in ("syntaxerror", "indentationerror", "parse"))
        add_finding(
            findings,
            raw_path,
            syntax=is_syntax,
            runtime=not is_syntax,
            diagnostic=message,
        )
    return findings


def tool_specs() -> list[ToolSpec]:
    semgrep_config = Path(__file__).with_name("semgrep.yml").resolve()
    pylint_codes = ",".join(sorted(PYLINT_RUNTIME_CODES | {"E0001", "F0010"}))
    return [
        ToolSpec("compile", "Python compile", None, None, None, timeout_per_sample=0.0),
        ToolSpec(
            "ruff", "Ruff", "ruff",
            lambda paths: [
                "ruff", "check", "--output-format=json", "--no-cache",
                "--select", ",".join(sorted(RUFF_RUNTIME_CODES)), *map(str, paths),
            ],
            parse_ruff,
        ),
        ToolSpec(
            "pyright", "Pyright", "pyright",
            lambda paths: [
                "pyright", "--outputjson", "--level", "error", "--pythonversion", "3.13",
                *map(str, paths),
            ],
            parse_pyright,
        ),
        ToolSpec(
            "pylint", "Pylint", "pylint",
            lambda paths: [
                "pylint", "--output-format=json", "--reports=no", "--score=no",
                "--persistent=no", "--jobs=1", "--disable=all", f"--enable={pylint_codes}",
                *map(str, paths),
            ],
            parse_pylint,
            accepted_exit_codes=frozenset(range(32)),
        ),
        ToolSpec(
            "semgrep", "Semgrep", "semgrep",
            lambda paths: [
                "semgrep", "scan", "--config", str(semgrep_config), "--json",
                "--metrics=off", "--timeout=2", "--timeout-threshold=1", "--quiet",
                *map(str, paths),
            ],
            parse_semgrep,
            accepted_exit_codes=frozenset({0}),
            timeout_per_sample=3.0,
        ),
        ToolSpec(
            "crosshair", "CrossHair (experimental)", "crosshair",
            lambda paths: [
                "crosshair", "check", "--analysis_kind=asserts,PEP316",
                "--max_uninteresting_iterations=2", "--per_path_timeout=0.25",
                "--per_condition_timeout=0.5", *map(str, paths),
            ],
            parse_crosshair,
            accepted_exit_codes=frozenset({0, 1}),
            timeout_per_sample=2.0,
        ),
    ]


def resolve_executable(name: str) -> str | None:
    executable = shutil.which(name)
    if executable:
        return executable
    sibling = Path(sys.executable).absolute().parent / name
    return str(sibling) if sibling.is_file() else None


def tool_version(spec: ToolSpec) -> str:
    if spec.key == "compile":
        return f"Python {sys.version.split()[0]}"
    if spec.key == "crosshair":
        try:
            return f"crosshair-tool {version('crosshair-tool')}"
        except PackageNotFoundError:
            return "missing"
    if spec.key == "semgrep":
        try:
            return f"semgrep {version('semgrep')}"
        except PackageNotFoundError:
            return "missing"
    executable = resolve_executable(spec.executable or "")
    if not executable:
        return "missing"
    flag = "version" if spec.key == "crosshair" else "--version"
    try:
        environment = os.environ.copy()
        environment["SEMGREP_SETTINGS_FILE"] = str(
            Path(tempfile.gettempdir()) / "hallucode-semgrep-settings.yml"
        )
        environment["SEMGREP_LOG_FILE"] = str(
            Path(tempfile.gettempdir()) / "hallucode-semgrep.log"
        )
        result = subprocess.run(
            [executable, flag], capture_output=True, text=True, timeout=10, check=False,
            env=environment,
        )
        lines = (result.stdout or result.stderr).strip().splitlines()
        return lines[-1] if lines else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def run_compile(paths: list[Path]) -> tuple[dict[Path, ToolFinding], float]:
    findings = empty_findings(paths)
    started = time.perf_counter()
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except (SyntaxError, UnicodeError, ValueError) as exc:
            add_finding(findings, path, syntax=True, diagnostic=f"{type(exc).__name__}: {exc}")
    return findings, time.perf_counter() - started


def _tool_environment(paths: list[Path]) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "SEMGREP_SEND_METRICS": "off",
        "SEMGREP_SETTINGS_FILE": str(paths[0].parent / ".semgrep-settings.yml"),
        "SEMGREP_LOG_FILE": str(paths[0].parent / ".semgrep.log"),
        "NO_COLOR": "1",
    })
    return environment


def run_crosshair(
    spec: ToolSpec, executable: str, paths: list[Path]
) -> tuple[dict[Path, ToolFinding], float]:
    """Run targets independently so one import/syntax failure cannot hide a batch."""
    assert spec.command is not None and spec.parser is not None
    findings = empty_findings(paths)
    environment = _tool_environment(paths)
    started = time.perf_counter()
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except (SyntaxError, UnicodeError, ValueError) as exc:
            add_finding(findings, path, syntax=True, diagnostic=f"{type(exc).__name__}: {exc}")
            continue

        command = spec.command([path])
        command[0] = executable
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=environment,
                timeout=max(5.0, spec.timeout_per_sample + 2.0),
                check=False,
            )
        except subprocess.TimeoutExpired:
            findings[path.resolve()].error = "CrossHair target timeout"
            continue
        parsed = spec.parser(result.stdout, result.stderr, [path])[path.resolve()]
        findings[path.resolve()] = parsed
        if result.returncode not in spec.accepted_exit_codes:
            details = (result.stderr or result.stdout)[-500:].replace("\n", " ")
            findings[path.resolve()].error = f"CrossHair exited with {result.returncode}: {details}"
    return findings, time.perf_counter() - started


def run_tool(spec: ToolSpec, paths: list[Path]) -> tuple[dict[Path, ToolFinding], float]:
    if spec.key == "compile":
        return run_compile(paths)
    executable = resolve_executable(spec.executable or "")
    if not executable:
        raise RuntimeError(f"Executable '{spec.executable}' is unavailable. Run uv sync.")
    if spec.key == "crosshair":
        return run_crosshair(spec, executable, paths)
    assert spec.command is not None and spec.parser is not None
    command = spec.command(paths)
    command[0] = executable
    environment = _tool_environment(paths)
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            timeout=max(45.0, spec.timeout_per_sample * len(paths) + 10.0),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{spec.label} exceeded the batch timeout") from exc
    elapsed = time.perf_counter() - started
    if result.returncode not in spec.accepted_exit_codes:
        details = (result.stderr or result.stdout)[-1000:]
        raise RuntimeError(f"{spec.label} exited with {result.returncode}: {details}")
    return spec.parser(result.stdout, result.stderr, paths), elapsed
