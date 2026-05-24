from __future__ import annotations

import subprocess

import ollama

from src.core import ui
from src.models.base import BaseModelHandler
from src.models.prompts import build_solve_prompt, solve_problem_system


class OllamaHandler(BaseModelHandler):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": "oi"}],
            stream=False,
            keep_alive="-1m",
        )

    def close(self) -> None:
        subprocess.run(["ollama", "stop", self.model], check=False)

    def generate_code(
        self,
        example: object,
        options: dict | None,
        spinner_length: int,
    ) -> str:
        prompt = build_solve_prompt(example.prompt, example.function_signature)
        messages = [
            {"role": "system", "content": solve_problem_system},
            {"role": "user", "content": prompt},
        ]
        stream = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options=options or None,
            keep_alive="-1m",
        )
        return ui.stream_chat_chunks(stream, spinner_length)
