from __future__ import annotations

import subprocess
import ollama

from src.core import ui
from src.models.base import BaseModelHandler, GenerateResult

class OllamaHandler(BaseModelHandler):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.spinner_length = 600
        ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": "oi"}],
            stream=False,
            keep_alive="-1m",
        )

    def close(self) -> None:
        subprocess.run(["ollama", "stop", self.model], check=False)
    
    def _generate(self, messages:list[dict[str, str]], temperature:float=0.0)-> GenerateResult:
        stream = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                "temperature": temperature,
            },
            keep_alive="-1m",
        )
        chunk_iter = (chunk.model_dump() for chunk in stream)
        result = ui.stream_chat_chunks(chunk_iter, self.spinner_length)
        return GenerateResult(content=result.content, thoughts=result.thoughts)
