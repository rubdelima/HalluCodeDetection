from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core import ui
from src.models.base import BaseModelHandler, SolveExample
from src.models.prompts import (
    analyse_hallucination_prompt,
    build_judge_prompt,
    build_solve_prompt,
    judge_system_prompt,
    solve_problem_system,
)


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> Any:
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return input_ids


def _resolve_generation_kwargs(options: dict[str, object] | None) -> dict[str, object]:
    if not options:
        return {}
    allowed = {
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "do_sample",
        "repetition_penalty",
    }
    return {key: options[key] for key in options if key in allowed}


class GemmaHandler(BaseModelHandler):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = model
        self.tokenizer: Any
        self.model_instance: Any
        model_path = Path(model)
        device_map = "auto" if self.device == "cuda" else None

        if model_path.is_dir() and (model_path / "adapter_config.json").exists():
            with (model_path / "adapter_config.json").open("r", encoding="utf-8") as handle:
                adapter_config = json.load(handle)
            base_model_id = str(adapter_config.get("base_model_name_or_path") or model)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            base_model = cast(
                Any,
                AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype=dtype,
                    device_map=device_map,
                ),
            )
            self.model_instance = cast(Any, PeftModel.from_pretrained(base_model, str(model_path)))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model_instance = cast(
                Any,
                AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=dtype,
                    device_map=device_map,
                ),
            )

    def close(self) -> None:
        del self.model_instance
        del self.tokenizer
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _generate(
        self,
        messages: list[dict[str, str]],
        options: dict[str, object] | None,
        spinner_length: int,
    ) -> str:
        inputs = _apply_chat_template(self.tokenizer, messages)
        target_device = self.device
        if hasattr(self.model_instance, "parameters"):
            try:
                first_parameter = next(self.model_instance.parameters())
                if isinstance(first_parameter, torch.Tensor) and first_parameter.device.type != "meta":
                    target_device = str(first_parameter.device)
            except StopIteration:
                pass

        gen_kwargs = _resolve_generation_kwargs(options)
        max_new_tokens = int(str(gen_kwargs.pop("max_new_tokens", 512)))

        if isinstance(inputs, dict):
            typed_inputs = cast(dict[str, Any], inputs)
            seq_len = None
            for key, value in list(typed_inputs.items()):
                if isinstance(value, torch.Tensor):
                    typed_inputs[key] = value.to(target_device)
                    if key == "input_ids":
                        seq_len = typed_inputs[key].shape[1]
            with torch.no_grad():
                output = self.model_instance.generate(
                    **typed_inputs,
                    max_new_tokens=max_new_tokens,
                    **cast(dict[str, Any], gen_kwargs),
                )
            if seq_len is None:
                first_tensor = next(
                    (value for value in typed_inputs.values() if isinstance(value, torch.Tensor)),
                    None,
                )
                if first_tensor is None:
                    raise ValueError("Tokenizer did not return input tensors.")
                seq_len = first_tensor.shape[1]
        else:
            input_ids = cast(torch.Tensor, inputs).to(target_device)
            with torch.no_grad():
                output = self.model_instance.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs,
                )
            seq_len = input_ids.shape[1]

        response_tokens = output[:, seq_len:]
        text = str(self.tokenizer.decode(response_tokens[0], skip_special_tokens=True))
        return ui.stream_chat_chunks(({"message": {"content": text}},), spinner_length)

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
        return self._generate(messages, options, spinner_length)

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
        return self._generate(messages, options, spinner_length)

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
        return self._generate(messages, options, spinner_length)
