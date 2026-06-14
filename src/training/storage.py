from __future__ import annotations

import gc
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.schemas.training import TrainingResult


def remove_model_dir(path: str | None) -> None:
    if not path:
        return

    model_path = Path(path)
    if model_path.exists():
        shutil.rmtree(model_path)


def save_merged_model(
    *,
    base_model_id: str,
    adapter_path: Path,
    merged_model_path: Path,
    dtype: torch.dtype,
) -> None:
    base_model = None
    peft_model = None
    merged_model = None
    processor = None
    try:
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(
            merged_model_path,
            safe_serialization=True,
            max_shard_size="1GB",
        )

        processor = AutoProcessor.from_pretrained(str(adapter_path))
        processor.save_pretrained(merged_model_path)
    finally:
        del processor
        del merged_model
        del peft_model
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def apply_saved_model_policy(
    result: TrainingResult,
    results: list[TrainingResult],
    max_saved_models: int,
) -> TrainingResult:
    if max_saved_models <= 0:
        remove_model_dir(result.model_path)
        result.model_path = None
        result.saved_model = False
        return result

    saved_results = [
        item
        for item in results
        if item.saved_model and item.model_path and Path(item.model_path).exists()
    ]
    ranked_results = sorted(
        [*saved_results, result],
        key=lambda item: item.test_acc,
        reverse=True,
    )
    keep_ids = {id(item) for item in ranked_results[:max_saved_models]}

    for item in ranked_results:
        if id(item) in keep_ids:
            item.saved_model = True
            continue

        remove_model_dir(item.model_path)
        item.model_path = None
        item.saved_model = False

    return result
