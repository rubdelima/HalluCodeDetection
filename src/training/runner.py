from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase
from typing import Any, cast
from trl import SFTConfig, SFTTrainer

from src.core import ui
from src.training.config import TrainingSettings, TrainingPaths
from src.training.eval import evaluate_accuracy
from src.training.helpers import grid_config_label, safe_name
from src.training.registry import (
    RegistryEntry,
    RunRecord,
    append_history,
    compute_run_id,
    has_run,
    register_model,
)
from src.training.ui import build_grid_progress, render_grid_table


def _has_saved_adapter(output_dir: Path) -> bool:
    return any(
        (output_dir / filename).exists()
        for filename in (
            "adapter_config.json",
            "adapter_model.bin",
            "adapter_model.safetensors",
            "pytorch_model.bin",
            "config.json",
        )
    )


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[Path] = []
    if not output_dir.exists():
        return None
    for child in output_dir.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            candidates.append(child)
    if not candidates:
        return None

    def checkpoint_key(path: Path) -> int:
        match = re.search(r"checkpoint-(\d+)$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=checkpoint_key)


def _cleanup_checkpoints(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            shutil.rmtree(child, ignore_errors=True)


def _build_trainer(
    model_id: str,
    model_kwargs: dict[str, object],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
    settings: TrainingSettings,
    report_to: str,
    dtype: torch.dtype,
    train_dataset,
    output_dir: Path,
) -> tuple[SFTTrainer, Any]:
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
        ensure_weight_tying=True,
    )

    args = SFTConfig(
        output_dir=str(output_dir),
        max_length=settings.max_length,
        num_train_epochs=settings.num_train_epochs,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        optim="adamw_torch_fused",
        logging_steps=settings.logging_steps,
        save_strategy="epoch",
        save_total_limit=settings.max_saved_models,
        eval_strategy="no",
        learning_rate=learning_rate,
        fp16=True if dtype == torch.float16 else False,
        bf16=True if dtype == torch.bfloat16 else False,
        max_grad_norm=settings.max_grad_norm,
        lr_scheduler_type=settings.lr_scheduler_type,
        push_to_hub=settings.push_to_hub,
        report_to=report_to,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
        },
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    return trainer, tokenizer


def run_grid_search(
    settings: TrainingSettings,
    paths: TrainingPaths,
    train_dataset,
    validation_dataset,
    grid: list[tuple[int, int, float, float]],
) -> None:
    use_cuda = torch.cuda.is_available()
    dtype = torch.float32
    if use_cuda:
        major, _minor = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16

    report_to = settings.report_to
    if report_to == "tensorboard":
        if importlib.util.find_spec("tensorboard") is None and importlib.util.find_spec(
            "tensorboardX"
        ) is None:
            ui.console.print("TensorBoard not found; disabling report_to.")
            report_to = "none"

    total_runs = len(settings.use_qlora) * len(grid)
    grid_rows: list[dict[str, str]] = []
    progress, task_id = build_grid_progress(total_runs)
    task_id = cast(Any, task_id)

    with progress:
        for qlora in settings.use_qlora:
            if qlora and not use_cuda:
                raise ValueError("QLoRA requires CUDA.")

            for lora_r, lora_alpha, lora_dropout, learning_rate in grid:
                config_label = grid_config_label(
                    qlora, lora_r, lora_alpha, lora_dropout, learning_rate
                )
                progress.update(task_id, config=config_label)

                run_config = {
                    "model_id": settings.model_id,
                    "qlora": qlora,
                    "dataset_load": settings.dataset_load,
                    "correct_size": settings.correct_size,
                    "random_seed": settings.random_seed,
                    "validation_size": settings.validation_size,
                    "test_size": settings.test_size,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "learning_rate": learning_rate,
                    "max_length": settings.max_length,
                    "num_train_epochs": settings.num_train_epochs,
                    "per_device_train_batch_size": settings.per_device_train_batch_size,
                    "logging_steps": settings.logging_steps,
                    "max_grad_norm": settings.max_grad_norm,
                    "lr_scheduler_type": settings.lr_scheduler_type,
                    "push_to_hub": settings.push_to_hub,
                    "report_to": report_to,
                    "judge_model": paths.judge_model,
                    "train_size": len(train_dataset),
                    "validation_size_count": len(validation_dataset),
                }

                run_id = compute_run_id(run_config)
                if has_run(paths.history_path, run_id):
                    grid_rows.append(
                        {"config": config_label, "status": "skipped", "score": "-"}
                    )
                    progress.update(task_id, advance=1)
                    ui.console.print(render_grid_table(grid_rows))
                    continue

                suffix = "qlora" if qlora else "lora"
                output_dir = paths.output_root / (
                    f"{safe_name(settings.model_id)}_{suffix}"
                    f"_r{lora_r}_a{lora_alpha}_d{lora_dropout}_lr{learning_rate}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                completed_run = has_run(paths.history_path, run_id)
                saved_adapter = _has_saved_adapter(output_dir)
                latest_checkpoint = _latest_checkpoint(output_dir)

                if completed_run and saved_adapter:
                    ui.console.print(
                        f"Found completed run in {output_dir}, skipping training."
                    )
                    grid_rows.append({"config": config_label, "status": "skipped", "score": "-"})
                    progress.update(task_id, advance=1)
                    ui.console.print(render_grid_table(grid_rows))
                    continue
                model_kwargs: dict[str, object] = {
                    "torch_dtype": dtype,
                    "device_map": "auto" if use_cuda else None,
                }
                if qlora:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_quant_storage=dtype,
                    )

                ui.console.print(
                    "Training "
                    f"{settings.model_id} ({suffix}) "
                    f"r={lora_r} a={lora_alpha} d={lora_dropout} lr={learning_rate} "
                    f"on {len(train_dataset)} samples."
                )

                trainer, tokenizer = _build_trainer(
                    settings.model_id,
                    model_kwargs,
                    lora_r,
                    lora_alpha,
                    lora_dropout,
                    learning_rate,
                    settings,
                    report_to,
                    dtype,
                    train_dataset,
                    output_dir,
                )
                trained = False
                if saved_adapter:
                    ui.console.print(f"Loading existing adapter from {output_dir}.")
                    trainer.model = PeftModel.from_pretrained(trainer.model, str(output_dir))
                elif latest_checkpoint is not None:
                    ui.console.print(f"Resuming from checkpoint {latest_checkpoint}.")
                    trainer.train(resume_from_checkpoint=str(latest_checkpoint))
                    trained = True
                else:
                    trainer.train()
                    trained = True

                if trained:
                    try:
                        trainer.save_model()
                    except Exception:
                        ui.console.print("Warning: failed to save model after training.")

                # Run evaluation, but if it fails don't force re-training (model already saved)
                try:
                    accuracy, correct_count, total_count = evaluate_accuracy(
                        trainer.model,
                        tokenizer,
                        validation_dataset,
                    )
                    eval_score = accuracy
                except Exception as exc:  # evaluation errors shouldn't require re-training
                    ui.console.print(f"Evaluation failed: {exc}")
                    eval_score = 0.0
                    correct_count = 0
                    total_count = 0

                run_config["eval_score"] = eval_score
                run_config["validation_correct"] = correct_count
                run_config["validation_total"] = total_count

                entry = RegistryEntry(
                    path=str(output_dir),
                    eval_score=eval_score,
                    run_config=run_config,
                )

                saved = register_model(
                    paths.registry_path,
                    entry,
                    settings.max_saved_models,
                )
                if saved:
                    trainer.save_model()
                    with (output_dir / "run_config.json").open(
                        "w", encoding="utf-8"
                    ) as handle:
                        json.dump(run_config, handle, indent=2, ensure_ascii=True)
                    _cleanup_checkpoints(output_dir)
                else:
                    ui.console.print(
                        "Skipping save: model did not beat the current registry."
                    )
                    shutil.rmtree(output_dir, ignore_errors=True)

                status = "saved" if saved else "rejected"
                append_history(
                    paths.history_path,
                    RunRecord(
                        run_id=run_id,
                        eval_score=eval_score,
                        status=status,
                        run_config=run_config,
                    ),
                )

                grid_rows.append(
                    {
                        "config": config_label,
                        "status": status,
                        "score": f"{eval_score:.4f}",
                    }
                )
                progress.update(task_id, advance=1)
                ui.console.print(render_grid_table(grid_rows))
