from pathlib import Path

from src.schemas.dataset import BaseResultRow, JudgeResultRow
from src.dataset.judge_dataset import build_dataset, load_jsonl, select_records, stratified_split
from src.dateset.utils import to_conversation

from src.constants import HalluCodeDetectionConfig
from src.constants.training import TrainingHyperparameters
from src.constants.models import ModelInfo

import torch

from trl.trainer.sft_trainer import SFTTrainer
from uuid import uuid4
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from src.training.hyperparams import get_trainer
from src.evaluations import evaluate_model
from src.core import ui
import time
from src.schemas.training import TrainingResult
from typing import Optional

def train_model(
    model_name:str,
    hyperparameters: TrainingHyperparameters,
    dataset
)->Optional[TrainingResult]:
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    quantization_config = None
    
    if hyperparameters.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        hyperparameters.model_name.id,
        dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )
    
    processor = AutoProcessor.from_pretrained(hyperparameters.model_name.id)
    
    trainer = get_trainer(hyperparameters, model, model_name, processor, dataset)
    try:
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        model_info = ModelInfo(
            name=model_name,
            id=model_name,
            type=hyperparameters.model_name.type,
            size=hyperparameters.model_name.size,
            quantization="4-bit" if hyperparameters.use_qlora else None
        )
        
        train_result = evaluate_model(
            dataset_split=dataset['train'],
            model_info=model_info
        )
        validation_result = evaluate_model(
            dataset_split=dataset['validation'],
            model_info=model_info
        )
        
        train_result = evaluate_model(
            dataset_split=dataset['test'],
            model_info=model_info
        )
        
        return TrainingResult(
            **hyperparameters.model_dump(),
            training_time=training_time,
            train_acc=train_result.overall_accuracy,
            val_acc=validation_result.overall_accuracy,
            test_acc=train_result.overall_accuracy,
        )
        
    except Exception as e:
        ui.console.print(f"Error training model {hyperparameters.model_name.id}: {e}")
    
    finally:
        del model
        del processor
        torch.cuda.empty_cache()
    
    return None
        
def train_models(config: HalluCodeDetectionConfig):
    
    results_dir = Path(config.dataset_building_config.results_dir)
    base_path = results_dir / "dataset_base.json"
    judge_path = results_dir / "dataset_judge.jsonl"
    base_results = load_jsonl(base_path, BaseResultRow)
    
    judge_results = load_jsonl(judge_path, JudgeResultRow)

    records = select_records(base_results, judge_results)

    dataset = build_dataset(records)
    dataset = stratified_split(dataset, validation_size=0.1, test_size=0.2, seed=42)
    
    models_path = Path(config.training_config.models_path)
    models_path.mkdir(parents=True, exist_ok=True)
    
    for hyperparameters in config.training_config.hyperparameters:
        model_name = f"{models_path}/{uuid4().hex[:8]}"
        train_model(model_name, hyperparameters, dataset)
        
        