from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


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
        example: Any,
        options: dict | None,
        spinner_length: int,
    ) -> str:
        raise NotImplementedError
    
    