from __future__ import annotations

import json
import os
import random
import time
import uuid
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from src.core.log import get_logger
from src.models.base import BaseModelHandler, GenerateResult, SolveExample

log = get_logger("opencode_go")

_CHAT_COMPLETIONS_URL = "https://opencode.ai/zen/go/v1/chat/completions"
_TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}


class OpenCodeGoHandler(BaseModelHandler):
    """OpenCode Go models exposed through its OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        max_tokens: int = 4096,
        request_timeout: int = 120,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        reasoning_effort: Literal["none", "low", "medium", "high", "max"] | None = None,
    ) -> None:
        super().__init__(model)
        load_dotenv()
        api_key = os.getenv("OPENCODE_GO_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENCODE_GO_API_KEY is not set. Add it to .env or export it before running."
            )

        self._api_key = api_key
        self._max_tokens = max_tokens
        self._request_timeout = request_timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._reasoning_effort = reasoning_effort

    def close(self) -> None:
        # Remote requests are stateless; there is no local model process to stop.
        return None

    def _generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> GenerateResult:
        return self._generate_with_request_policy(messages, temperature)

    def generate_code_with_request_policy(
        self,
        example: SolveExample,
        temperature: float,
        *,
        request_timeout: int,
        max_retries: int,
    ) -> str:
        """Generate code with a one-call policy, used for availability probes."""
        messages = self._generation_messages(example)
        return self._generate_with_request_policy(
            messages,
            temperature,
            request_timeout=request_timeout,
            max_retries=max_retries,
        ).content

    def _generate_with_request_policy(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        *,
        request_timeout: int | None = None,
        max_retries: int | None = None,
    ) -> GenerateResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if self._reasoning_effort is not None:
            # OpenCode Go currently ignores Xiaomi's native `thinking` switch
            # for MiMo, but honors the OpenAI-compatible effort override.
            payload["reasoning_effort"] = self._reasoning_effort
        else:
            payload["thinking"] = {"type": "disabled"}
        log.info("Generating with OpenCode Go model %s (temperature=%s) ...", self.model, temperature)
        started_at = time.monotonic()
        raw_response = self._request_with_retries(
            payload,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )

        try:
            body: dict[str, Any] = json.loads(raw_response)
            choices = body["choices"]
            message = choices[0]["message"]
            content = message.get("content") or ""
            thoughts = message.get("reasoning_content") or message.get("reasoning")
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unexpected OpenCode Go response for {self.model}: {raw_response[:500]}"
            ) from exc

        if not isinstance(content, str):
            raise TypeError(f"OpenCode Go returned non-text content for {self.model}.")
        if thoughts is not None and not isinstance(thoughts, str):
            thoughts = None

        log.info("Generated with OpenCode Go model %s in %.2fs.", self.model, time.monotonic() - started_at)
        return GenerateResult(content=content, thoughts=thoughts)

    def _request_with_retries(
        self,
        payload: dict[str, Any],
        *,
        request_timeout: int | None = None,
        max_retries: int | None = None,
    ) -> str:
        timeout = self._request_timeout if request_timeout is None else request_timeout
        retries = self._max_retries if max_retries is None else max_retries
        session_id = f"hallucodedetection-{uuid.uuid4()}"
        for attempt in range(retries + 1):
            request = Request(
                _CHAT_COMPLETIONS_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "HalluCodeDetection/0.1",
                    "x-opencode-session": session_id,
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=timeout) as response:
                    raw_response: str = response.read().decode("utf-8")
                    return raw_response
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code not in _TRANSIENT_HTTP_CODES or attempt >= retries:
                    raise RuntimeError(
                        f"OpenCode Go returned HTTP {exc.code} for {self.model}: {detail}"
                    ) from exc
                retry_after = exc.headers.get("Retry-After")
                delay = self._retry_delay(attempt, retry_after)
                log.warning(
                    "Transient OpenCode Go HTTP %d for %s; retrying in %.1fs (%d/%d).",
                    exc.code,
                    self.model,
                    delay,
                    attempt + 1,
                    retries,
                )
                time.sleep(delay)
            except (TimeoutError, URLError) as exc:
                if attempt >= retries:
                    if isinstance(exc, URLError):
                        message = str(exc.reason)
                    else:
                        message = str(exc)
                    raise RuntimeError(
                        f"OpenCode Go request failed for {self.model}: {message}"
                    ) from exc
                delay = self._retry_delay(attempt)
                log.warning(
                    "Transient OpenCode Go connection failure for %s; retrying in %.1fs (%d/%d).",
                    self.model,
                    delay,
                    attempt + 1,
                    retries,
                )
                time.sleep(delay)

        raise RuntimeError(f"OpenCode Go request failed unexpectedly for {self.model}.")

    def _retry_delay(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        exponential = self._retry_base_delay * (2**attempt)
        return float(
            min(exponential + random.uniform(0.0, self._retry_base_delay), 60.0)
        )
