from __future__ import annotations

import torch

from src.models.base import BaseModelHandler, GenerateResult
from src.models.loading import load_model, load_text_tokenizer

class GemmaHandler(BaseModelHandler):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        
        self.model = model
        self.tokenizer = load_text_tokenizer(model)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model_instance = load_model(model, device_map="auto", dtype=dtype).eval()
        

    def close(self) -> None:
        del self.model_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(
        self,
        messages: list[dict[str, str]],
        temperature: float=0.0,
    ) -> GenerateResult:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model_instance.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = self.model_instance.generate(**inputs, max_new_tokens=4096, do_sample=False, temperature=temperature)
            generation = generation[0][input_len:]
        
        decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
        
        return GenerateResult(content=decoded) #type:ignore
