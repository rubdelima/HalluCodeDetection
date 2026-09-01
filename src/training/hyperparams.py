from trl.trainer.sft_trainer import SFTTrainer
from src.constants.training import TrainingHyperparameters
from peft import LoraConfig
from trl.trainer.sft_config import SFTConfig
from src.dataset.utils import to_conversation
import torch
import torch.nn.functional as F


CLASS_LABELS = ("correct", "functional", "runtime", "syntax")


class ClassificationWeightedSFTTrainer(SFTTrainer):
    """SFT loss that gives extra weight to tokens used to emit class labels."""

    def __init__(self, *args, classification_token_ids: set[int], classification_loss_weight: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.classification_token_ids = tuple(classification_token_ids)
        self.classification_loss_weight = classification_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        model_inputs = dict(inputs)
        model_inputs["use_cache"] = False
        outputs = model(**model_inputs)
        logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid = shift_labels.ne(-100)

        token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        class_token_ids = torch.tensor(
            self.classification_token_ids,
            device=shift_labels.device,
            dtype=shift_labels.dtype,
        )
        is_class_token = torch.isin(shift_labels, class_token_ids) & valid
        weights = torch.ones_like(token_loss)
        weights = torch.where(
            is_class_token,
            torch.full_like(weights, self.classification_loss_weight),
            weights,
        )
        loss = (token_loss * weights * valid).sum() / (weights * valid).sum().clamp_min(1)
        return (loss, outputs) if return_outputs else loss


def _class_label_token_ids(tokenizer) -> set[int]:
    """Return token IDs that encode class names with and without leading whitespace."""
    token_ids: set[int] = set()
    for label in CLASS_LABELS:
        for text in (label, f" {label}"):
            token_ids.update(tokenizer.encode(text, add_special_tokens=False))
    return token_ids

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
        # Validation is measured after training by the project's evaluator.
        # Avoiding in-epoch evaluation/checkpointing prevents a second peak in
        # memory at every epoch boundary.
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        learning_rate=hyperparameters.learning_rate,
        fp16=True if model.dtype == torch.float16 else False, #type:ignore
        bf16=True if model.dtype == torch.bfloat16 else False, #type:ignore
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        lr_scheduler_type="constant",
        push_to_hub=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
        },
    )
    
    trainer_kwargs = dict(
        model=model, #type:ignore
        args=args,
        train_dataset=conversation_dataset["train"],
        eval_dataset=conversation_dataset["validation"],
        peft_config=peft_config,
        processing_class=processor,#type:ignore
    )
    if hyperparameters.classification_loss_strategy == "label_weighted":
        return ClassificationWeightedSFTTrainer(
            **trainer_kwargs,
            classification_token_ids=_class_label_token_ids(processor),
            classification_loss_weight=hyperparameters.classification_loss_weight,
        )
    return SFTTrainer(**trainer_kwargs)
