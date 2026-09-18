from __future__ import annotations

import subprocess
import time

import ollama

from src.core import ui
from src.core.log import get_logger
from src.models.base import BaseModelHandler, GenerateResult

log = get_logger("ollama")


class OllamaHandler(BaseModelHandler):
    def __init__(
        self,
        model: str,
        num_ctx: int | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> None:
        super().__init__(model)
        self.spinner_length = 600
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.think = think if think is not None else False
        log.info("Loading model %s (num_ctx=%s, think=%s) ...", model, self.num_ctx, self.think)
        t0 = time.monotonic()
        options = {"num_ctx": self.num_ctx} if self.num_ctx is not None else None
        ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": "oi"}],
            stream=False,
            keep_alive="-1m",
            think=self.think,
            options=options,
        )
        log.info("Model %s loaded in %.2fs.", model, time.monotonic() - t0)

    def close(self) -> None:
        log.info("Stopping model %s ...", self.model)
        t0 = time.monotonic()
        subprocess.run(["ollama", "stop", self.model], check=False)
        log.info("Model %s stopped in %.2fs.", self.model, time.monotonic() - t0)

    def _generate(self, messages: list[dict[str, str]], temperature: float = 0.0) -> GenerateResult:
        log.info("Generating with %s (temperature=%s, think=%s) ...", self.model, temperature, self.think)
        t0 = time.monotonic()
        options: dict[str, object] = {"temperature": temperature}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        stream = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options=options,
            keep_alive="-1m",
            think=self.think,
        )
        chunk_iter = (chunk.model_dump() for chunk in stream)
        result = ui.stream_chat_chunks(chunk_iter, self.spinner_length)
        log.info("Generated with %s in %.2fs.", self.model, time.monotonic() - t0)
        return GenerateResult(content=result.content, thoughts=result.thoughts)
