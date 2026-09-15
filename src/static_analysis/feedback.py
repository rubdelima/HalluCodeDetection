from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from src.evaluations.classify import build_candidate_code
from src.schemas.evalplus import EvalPlusExample
from src.static_analysis.tools import ToolFinding, run_tool, tool_specs


class StaticToolFeedback(BaseModel):
    compile_ok: bool
    pyright_ok: bool | None = None
    compile_diagnostics: list[str] = Field(default_factory=list)
    pyright_diagnostics: list[str] = Field(default_factory=list)
    compile_seconds: float = 0.0
    pyright_seconds: float = 0.0

    @property
    def clean(self) -> bool:
        return self.compile_ok and self.pyright_ok is True

    def as_prompt(self) -> str:
        compile_status = "PASS" if self.compile_ok else "FAIL"
        lines = [f"Python compile: {compile_status}"]
        lines.extend(f"- {message}" for message in self.compile_diagnostics)
        if self.pyright_ok is None:
            lines.append("Pyright: NOT RUN (the source did not compile)")
        else:
            lines.append(f"Pyright runtime checks: {'PASS' if self.pyright_ok else 'FAIL'}")
            lines.extend(f"- {message}" for message in self.pyright_diagnostics)
        if self.clean:
            lines.append(
                "No syntax or statically detectable runtime issue was reported. "
                "This does not prove functional correctness."
            )
        return "\n".join(lines)


def analyze_with_compile_and_pyright(
    generated_code: str,
    example: EvalPlusExample,
) -> StaticToolFeedback:
    """Analyze the exact complete source used later by the EvalPlus harness."""
    candidate = build_candidate_code(example, generated_code)
    specs = {spec.key: spec for spec in tool_specs()}
    with tempfile.TemporaryDirectory(prefix="hallucode-static-feedback-") as temp_name:
        path = Path(temp_name) / "candidate.py"
        path.write_text(candidate, encoding="utf-8")

        compile_findings, compile_seconds = run_tool(specs["compile"], [path])
        compile_finding = compile_findings[path.resolve()]
        if compile_finding.syntax:
            return StaticToolFeedback(
                compile_ok=False,
                compile_diagnostics=compile_finding.diagnostics,
                compile_seconds=compile_seconds,
            )

        pyright_findings, pyright_seconds = run_tool(specs["pyright"], [path])
        pyright_finding: ToolFinding = pyright_findings[path.resolve()]
        if pyright_finding.error:
            raise RuntimeError(pyright_finding.error)
        return StaticToolFeedback(
            compile_ok=True,
            pyright_ok=not pyright_finding.runtime,
            pyright_diagnostics=pyright_finding.diagnostics,
            compile_seconds=compile_seconds,
            pyright_seconds=pyright_seconds,
        )
