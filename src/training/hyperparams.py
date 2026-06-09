from trl.trainer.sft_trainer import SFTTrainer
from src.constants.training import TrainingHyperparameters
from peft import LoraConfig
from trl.trainer.sft_config import SFTConfig
from src.dataset.utils import to_conversation

def get_trainer(
        hyperparameters: TrainingHyperparameters, 
        model,
        model_name: str,
        processor,
        dataset
    ) -> SFTTrainer:
    
    conversation_dataset = {split: to_conversation(dataset[split]) for split in ["train", "validation", "test"]}

    peft_config = LoraConfig(
        lora_alpha=hyperparameters.lora_alpha,
        lora_dropout=hyperparameters.lora_dropout,
        r=hyperparameters.lora_r,
        bias=hyperparameters.bias,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
        ensure_weight_tying=True,
    )

    args = SFTConfig(
        output_dir=model_name,
        max_length=512,
        num_train_epochs=hyperparameters.num_epochs,
        per_device_train_batch_size=1,
        optim=hyperparameters.optimizer,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=hyperparameters.learning_rate,
        fp16=True if model.dtype == torch.float16 else False, #type:ignore
        bf16=True if model.dtype == torch.bfloat16 else False, #type:ignore
        max_grad_norm=0.3,
        lr_scheduler_type="constant",
        push_to_hub=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
        },
    )
    
    return SFTTrainer(
        model=model, #type:ignore
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=processor,#type:ignore
    )