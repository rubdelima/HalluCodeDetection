from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from src.models.prompts import (
    analyse_hallucination_prompt,
    build_generation_prompt,
    build_judge_prompt,
    build_tool_assisted_judge_prompt,
    build_tool_review_prompt,
    build_analyzer_initial_prompt,
    build_analyzer_feedback_prompt,
    build_developer_feedback_prompt,
    judge_system_prompt,
    solve_problem_system,
    tool_review_system_prompt,
    developer_agent_system_prompt,
    analyzer_agent_system_prompt,
)
from src.schemas.dataset import BaseResultRow, JudgeExplanation, JudgeResultRow
from src.schemas.judge import JudgeAnalysis, JudgeResponse


class SolveExample(Protocol):
    prompt: str
    function_signature: str | None
    is_completion: bool

@dataclass
class GenerateResult:
    content: str
    thoughts: str | None = None


@dataclass
class CodeReviewResult:
    decision: str
    assessment: str
    code: str
    raw_response: str
    thoughts: str | None = None

class BaseModelHandler(ABC):
    @abstractmethod
    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def _generate(
        self,
        messages: list[dict[str, str]],
        temperature: float=0.0,
    ) -> GenerateResult:
        raise NotImplementedError

    @staticmethod
    def _generation_messages(example: SolveExample) -> list[dict[str, str]]:
        prompt = build_generation_prompt(
            example.prompt,
            example.function_signature,
            example.is_completion,
        )
        return [
            {"role": "system", "content": solve_problem_system},
            {"role": "user", "content": prompt},
        ]

    def generate_code(
        self,
        example: SolveExample,
        temperature: float,
    ) -> str:
        messages = self._generation_messages(example)
        return self._generate(messages, temperature).content

    def start_developer_agent(
        self,
        example: SolveExample,
        temperature: float,
    ) -> tuple[str, list[dict[str, str]]]:
        """Generate the first candidate and retain the developer-only history."""
        messages = [
            {"role": "system", "content": developer_agent_system_prompt},
            *self._generation_messages(example)[1:],
        ]
        code = self._generate(messages, temperature).content.strip()
        messages.append({"role": "assistant", "content": code})
        return code, messages

    def developer_agent_history(
        self,
        example: SolveExample,
        code: str,
    ) -> list[dict[str, str]]:
        """Seed developer history when a provider pre-generated its first code."""
        return [
            {"role": "system", "content": developer_agent_system_prompt},
            *self._generation_messages(example)[1:],
            {"role": "assistant", "content": code},
        ]

    def revise_with_developer_feedback(
        self,
        messages: list[dict[str, str]],
        assessment: str,
        tool_feedback: str,
        round_number: int,
        max_rounds: int,
        temperature: float,
    ) -> CodeReviewResult:
        messages.append(
            {
                "role": "user",
                "content": build_developer_feedback_prompt(
                    assessment, tool_feedback, round_number, max_rounds
                ),
            }
        )
        code = self._strip_markdown_fence(self._generate(messages, temperature).content)
        messages.append({"role": "assistant", "content": code})
        return code

    def analyze_with_analyzer_agent(
        self,
        messages: list[dict[str, str]],
        example: SolveExample,
        code: str,
        tool_feedback: str,
        round_number: int,
        max_rounds: int,
        temperature: float,
    ) -> str:
        prompt = (
            build_analyzer_initial_prompt(
                example.prompt, example.function_signature, example.is_completion,
                code, tool_feedback, round_number, max_rounds,
            )
            if len(messages) == 1
            else build_analyzer_feedback_prompt(
                code, tool_feedback, round_number, max_rounds
            )
        )
        messages.append({"role": "user", "content": prompt})
        response = self._generate(messages, temperature)
        messages.append({"role": "assistant", "content": response.content})
        payload = self._extract_json_payload(response.content)
        if not payload:
            return CodeReviewResult(
                decision="revise",
                assessment="The Analyzer did not return a valid decision JSON; revise using the static-tool feedback.",
                code=code,
                raw_response=response.content,
                thoughts=response.thoughts,
            )
        decision = str(payload.get("decision", "revise")).strip().lower()
        if decision not in {"submit", "revise"}:
            decision = "revise"
        return CodeReviewResult(
            decision=decision,
            assessment=str(payload.get("assessment", "")).strip(),
            code=code,
            raw_response=response.content,
            thoughts=response.thoughts,
        )

    @staticmethod
    def analyzer_agent_history() -> list[dict[str, str]]:
        return [{"role": "system", "content": analyzer_agent_system_prompt}]

    def review_generated_code(
        self,
        example: SolveExample,
        code: str,
        tool_feedback: str,
        round_number: int,
        max_rounds: int,
        temperature: float,
    ) -> CodeReviewResult:
        messages = [
            {"role": "system", "content": tool_review_system_prompt},
            {
                "role": "user",
                "content": build_tool_review_prompt(
                    example.prompt,
                    example.function_signature,
                    example.is_completion,
                    code,
                    tool_feedback,
                    round_number,
                    max_rounds,
                ),
            },
        ]
        response = self._generate(messages, temperature)
        payload = self._extract_json_payload(response.content)
        if payload:
            decision = str(payload.get("decision", "revise")).strip().lower()
            assessment = str(payload.get("assessment", "")).strip()
            replacement = payload.get("code")
            if decision not in {"submit", "revise"}:
                decision = "revise"
            if not isinstance(replacement, str) or not replacement.strip():
                replacement = code
            return CodeReviewResult(
                decision=decision,
                assessment=assessment,
                code=replacement.strip(),
                raw_response=response.content,
                thoughts=response.thoughts,
            )

        # A code-only answer is a useful fallback for models that ignore the
        # requested JSON during a repair turn.
        replacement = self._strip_markdown_fence(response.content)
        return CodeReviewResult(
            decision="revise" if replacement else "submit",
            assessment="Response did not contain the requested JSON.",
            code=replacement or code,
            raw_response=response.content,
            thoughts=response.thoughts,
        )
    
    def analyze_hallucination(
        self,
        example_prompt: str,
        base_result: BaseResultRow,
        temperature: float,
    ) -> JudgeResultRow:
        user_prompt = build_judge_prompt(
            example_prompt,
            base_result.code,
            base_result.model,
            base_result.levels,
        )
        
        messages = [
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        text =  self._generate(messages, temperature).content
        extracted_dict = self._extract_json_payload(text)
        explanations = self._parse_explanations(
            extracted_dict, text, base_result.primary_level_name
        )
        
        return JudgeResultRow(
            benchmark=base_result.benchmark,
            benchmark_id=base_result.benchmark_id,
            response_model=base_result.model,
            judge_model=self.model,
            explanations=explanations,
        )

    @staticmethod
    def _parse_explanations(
        extracted: dict[str, object] | None,
        raw_text: str,
        fallback_level: str,
    ) -> list[JudgeExplanation]:
        if extracted and isinstance(extracted.get("levels"), list):
            items: list[JudgeExplanation] = []
            for entry in extracted["levels"]:
                if not isinstance(entry, dict):
                    continue
                level = entry.get("level")
                explanation = entry.get("explanation")
                if isinstance(level, str) and isinstance(explanation, str):
                    items.append(JudgeExplanation(level=level.lower(), explanation=explanation))
            if items:
                return items
        explanation = ""
        if extracted:
            explanation = str(extracted.get("explanation", ""))
        if not explanation:
            explanation = raw_text.strip()
        return [JudgeExplanation(level=fallback_level, explanation=explanation)]
    
    def generate_judge(
        self,
        example_prompt: str,
        code: str,
        temperature: float,
    ) -> JudgeResponse:
        messages = [
            {
                "role": "system",
                "content": analyse_hallucination_prompt.format(
                    problem_description=example_prompt
                ),
            },
            {"role": "user", "content": code},
        ]
        response = self._generate(messages, temperature)
        text = response.content
        extracted_dict = self._extract_json_payload(text)

        levels: list[JudgeExplanation] = []
        if extracted_dict and isinstance(extracted_dict.get("levels"), list):
            for entry in extracted_dict["levels"]:
                if not isinstance(entry, dict):
                    continue
                level = entry.get("level")
                explanation = entry.get("explanation")
                # Classification is the evaluation target.  A missing explanation
                # must not erase an otherwise valid class prediction.
                if isinstance(level, str):
                    levels.append(JudgeExplanation(
                        level=level.lower(),
                        explanation=explanation if isinstance(explanation, str) else "",
                    ))

        analysis = JudgeAnalysis(levels=levels) if levels else None
        return JudgeResponse(
            analysis=analysis,
            raw_response=text,
            thoughts=response.thoughts,
        )

    def generate_judge_with_tools(
        self,
        example_prompt: str,
        code: str,
        tool_feedback: str,
        temperature: float,
    ) -> JudgeResponse:
        messages = [
            {
                "role": "system",
                "content": analyse_hallucination_prompt.format(
                    problem_description=example_prompt
                ),
            },
            {
                "role": "user",
                "content": build_tool_assisted_judge_prompt(code, tool_feedback),
            },
        ]
        response = self._generate(messages, temperature)
        text = response.content
        extracted_dict = self._extract_json_payload(text)
        levels: list[JudgeExplanation] = []
        if extracted_dict and isinstance(extracted_dict.get("levels"), list):
            for entry in extracted_dict["levels"]:
                if not isinstance(entry, dict):
                    continue
                level = entry.get("level")
                explanation = entry.get("explanation")
                if isinstance(level, str):
                    levels.append(JudgeExplanation(
                        level=level.lower(),
                        explanation=explanation if isinstance(explanation, str) else "",
                    ))
        return JudgeResponse(
            analysis=JudgeAnalysis(levels=levels) if levels else None,
            raw_response=text,
            thoughts=response.thoughts,
        )
    
    @staticmethod
    def _extract_json_payload(text: str) -> dict[str, object] | None:
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        cleaned = text.strip()
        fenced = re.search(r"```(?:python)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        return (fenced.group(1) if fenced else cleaned).strip()
