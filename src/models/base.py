from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class SolveExample(Protocol):
    prompt: str
    function_signature: str | None

class BaseModelHandler(ABC):
    @abstractmethod
    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate_code(
        self,
        example: SolveExample,
        options: dict[str, object] | None,
        spinner_length: int = 600,
    ) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def generate_judge(
        self,
        example_prompt: str,
        code: str,
        level: str,
        error: str,
        options: dict[str, object] | None,
        spinner_length: int = 600,
    ) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def analyze_hallucination(
        self,
        example_prompt: str,
        code: str,
        options: dict[str, object] | None,
        spinner_length: int = 600,
    ) -> str:
        raise NotImplementedError
    