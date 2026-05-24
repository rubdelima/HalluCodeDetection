from __future__ import annotations

import json
import re
from typing import Iterable

import torch

from src.models.prompts import analyse_hallucination_prompt


def _normalize_response(text: str) -> str:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    return "\n".join(line.strip() for line in cleaned.splitlines()).strip()


def _extract_json_payload(text: str) -> dict[str, object] | None:
    cleaned = _normalize_response(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    payload_text = fenced.group(1) if fenced else cleaned
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_level(value: object) -> str:
    return str(value).strip().lower()


def evaluate_accuracy(
    model,
    tokenizer,
    dataset: Iterable[dict[str, str]],
    max_new_tokens: int = 256,
) -> tuple[float, int, int]:
    total = 0
    correct = 0
    target_device = torch.device("cpu")
    if hasattr(model, "parameters"):
        try:
            first_parameter = next(model.parameters())
            if isinstance(first_parameter, torch.Tensor) and first_parameter.device.type != "meta":
                target_device = first_parameter.device
        except StopIteration:
            pass
    for sample in dataset:
        messages = [
            {
                "role": "system",
                "content": analyse_hallucination_prompt.format(
                    problem_description=sample["problem_description"]
                ),
            },
            {"role": "user", "content": sample["generated_code"]},
        ]
        expected_level = sample["level"]
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            # Normalize inputs: can be tensor or dict of tensors
            if isinstance(inputs, dict):
                seq_len = None
                for k, v in list(inputs.items()):
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(target_device)
                        if k == "input_ids":
                            seq_len = inputs[k].shape[1]
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                }
                with torch.no_grad():
                    output = model.generate(**inputs, **gen_kwargs)
                if seq_len is None:
                    # fallback: assume input length is first dimension
                    seq_len = inputs[next(iter(inputs))].shape[1]
                response_tokens = output[:, seq_len:]
                text = tokenizer.decode(response_tokens[0], skip_special_tokens=True)
            else:
                # single tensor
                input_ids = inputs.to(target_device)
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                response_tokens = output[:, input_ids.shape[1] :]
                text = tokenizer.decode(response_tokens[0], skip_special_tokens=True)
        except Exception:
            continue

        payload = _extract_json_payload(text)
        if payload is None:
            continue
        predicted_level = payload.get("level")
        total += 1
        if _normalize_level(predicted_level) == _normalize_level(expected_level):
            correct += 1

    accuracy = correct / total if total else 0.0
    return accuracy, correct, total
