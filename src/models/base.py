from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol
import re
import json

from src.schemas.dataset import BaseResultRow, JudgeResultRow, JudgeExplanation
from src.schemas.judge import JudgeResponse, JudgeAnalysis

from dataclasses import dataclass
from src.models.prompts import *


class SolveExample(Protocol):
    prompt: str
    function_signature: str | None
    is_completion: bool

@dataclass
class GenerateResult:
    content: str
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

    def generate_code(
        self,
        example: SolveExample,
        temperature: float,
    ) -> str:
        prompt = build_generation_prompt(
            example.prompt,
            example.function_signature,
            example.is_completion,
        )
        messages = [
            {"role": "system", "content": solve_problem_system},
            {"role": "user", "content": prompt},
        ]
        return self._generate(messages, temperature).content
    
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
