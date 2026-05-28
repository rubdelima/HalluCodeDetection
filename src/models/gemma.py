from __future__ import annotations

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.core import ui
from src.models.base import BaseModelHandler, SolveExample
from src.models.prompts import (
    analyse_hallucination_prompt,
    build_judge_prompt,
    build_solve_prompt,
    judge_system_prompt,
    solve_problem_system,
)
class GemmaHandler(BaseModelHandler):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        
        self.model = model
        self.processor = AutoProcessor.from_pretrained(model)
        self.model_instance = AutoModelForImageTextToText.from_pretrained(
            model, device_map="auto"
        ).eval()
        

    def close(self) -> None:
        del self.model_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(
        self,
        messages: list[dict[str, str]],
        options: dict[str, object] | None,
        spinner_length: int = 600,
    ) -> str:
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model_instance.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = self.model_instance.generate(**inputs, max_new_tokens=4096, do_sample=False)
            generation = generation[0][input_len:]
        
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        
        return decoded

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