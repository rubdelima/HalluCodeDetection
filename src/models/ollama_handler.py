from __future__ import annotations

import subprocess
import ollama

from src.core import ui
from src.models.base import BaseModelHandler, SolveExample
from src.models.prompts import (
    analyse_hallucination_prompt,
    build_judge_prompt,
    build_solve_prompt,
    judge_system_prompt,
    solve_problem_system,
)


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
        example: SolveExample,
        options: dict[str, object] | None,
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
        chunk_iter = (chunk.model_dump() for chunk in stream)
        return ui.stream_chat_chunks(chunk_iter, spinner_length)

    def generate_judge(
        self,
        example_prompt: str,
        code: str,
        level: str,
        error: str,
        options: dict[str, object] | None,
        spinner_length: int,
    ) -> str:
        user_prompt = build_judge_prompt(example_prompt, code, level, error)
        messages = [
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        stream = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options=options or None,
            keep_alive="-1m",
        )
        chunk_iter = (chunk.model_dump() for chunk in stream)
        return ui.stream_chat_chunks(chunk_iter, spinner_length)

    def analyze_hallucination(
        self,
        example_prompt: str,
        code: str,
        options: dict[str, object] | None,
        spinner_length: int,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": analyse_hallucination_prompt.format(
                    problem_description=example_prompt
                ),
            },
            {"role": "user", "content": code},
        ]
        stream = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options=options or None,
            keep_alive="-1m",
        )
        chunk_iter = (chunk.model_dump() for chunk in stream)
        return ui.stream_chat_chunks(chunk_iter, spinner_length)
