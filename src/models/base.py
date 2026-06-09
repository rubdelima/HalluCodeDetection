from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol
import re
import json

from src.schemas.dataset import BaseResultRow, JudgeResultRow
from src.schemas.judge import JudgeResponse, JudgeAnalysis

from dataclasses import dataclass
from src.models.prompts import *


class SolveExample(Protocol):
    prompt: str
    function_signature: str | None

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
        prompt = build_solve_prompt(example.prompt, example.function_signature)
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
        user_prompt = build_judge_prompt(example_prompt, base_result.code, base_result.level, base_result.error)
        
        messages = [
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        text =  self._generate(messages, temperature).content
        extracted_dict = self._extract_json_payload(text)
        explanation = extracted_dict.get("explanation", "") if extracted_dict else ""
        
        return JudgeResultRow(
            benchmark=base_result.benchmark,
            benchmark_id=base_result.benchmark_id,
            response_model=base_result.code,
            judge_model=self.model,
            explanation=explanation, #type:ignore
        )
    
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
        try:
            extracted_dict = self._extract_json_payload(text)
            analysis = JudgeAnalysis(
                    level=extracted_dict["level"].lower(), #type:ignore
                    analysis=extracted_dict["explanation"] #type:ignore
            )
        except Exception:
            analysis = None
        
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
    