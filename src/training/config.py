from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast


def _to_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _to_int(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _to_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return default


def _to_bool_list(value: object, default: list[bool]) -> list[bool]:
    if isinstance(value, list) and all(isinstance(item, bool) for item in value):
        return value
    if isinstance(value, bool):
        return [value]
    return default


def _to_int_list(value: object, default: list[int]) -> list[int]:
    if isinstance(value, list):
        parsed: list[int] = []
        for item in value:
            if isinstance(item, int):
                parsed.append(item)
            elif isinstance(item, float):
                parsed.append(int(item))
            elif isinstance(item, str):
                try:
                    parsed.append(int(item))
                except ValueError:
                    continue
        return parsed or default
    if isinstance(value, (int, float)):
        return [int(value)]
    return default


def _to_float_list(value: object, default: list[float]) -> list[float]:
    if isinstance(value, list):
        parsed: list[float] = []
        for item in value:
            if isinstance(item, (int, float)):
                parsed.append(float(item))
            elif isinstance(item, str):
                try:
                    parsed.append(float(item))
                except ValueError:
                    continue
        return parsed or default
    if isinstance(value, (int, float)):
        return [float(value)]
    return default


@dataclass(frozen=True)
class TrainingSettings:
    dataset_load: float
    correct_size: float
    random_seed: int
    validation_size: float
    test_size: float
    model_id: str
    use_qlora: list[bool]
    lora_r_list: list[int]
    lora_alpha_list: list[int]
    lora_dropout_list: list[float]
    learning_rate_list: list[float]
    max_length: int
    num_train_epochs: int
    per_device_train_batch_size: int
    logging_steps: int
    max_grad_norm: float
    lr_scheduler_type: str
    push_to_hub: bool
    report_to: str
    output_root: Path
    max_saved_models: int


@dataclass(frozen=True)
class TrainingPaths:
    base_path: Path
    judge_path: Path
    output_root: Path
    registry_path: Path
    history_path: Path
    judge_model: str | None


def parse_training_settings(
    config: dict[str, object],
    model_name: str | None,
) -> tuple[TrainingSettings, TrainingPaths]:
    training_cfg = cast(dict[str, object], config.get("training", {}))
    build_cfg = cast(dict[str, object], config.get("dataset_build", {}))

    dataset_load = _to_float(training_cfg.get("dataset_load", 1.0), 1.0)
    correct_size = _to_float(training_cfg.get("correct_size", 1.0), 1.0)
    random_seed = _to_int(training_cfg.get("random_seed", 42), 42)
    validation_size = _to_float(training_cfg.get("validation_size", 0.1), 0.1)
    test_size = _to_float(training_cfg.get("test_size", 0.2), 0.2)

    model_id_value = model_name or training_cfg.get("model")
    if not isinstance(model_id_value, str) or not model_id_value:
        raise ValueError("No training model configured.")
    model_id = model_id_value

    use_qlora = _to_bool_list(training_cfg.get("use_qlora", [False]), [False])

    lora_r_list = _to_int_list(training_cfg.get("lora_r", [16]), [16])
    lora_alpha_list = _to_int_list(training_cfg.get("lora_alpha", [16]), [16])
    lora_dropout_list = _to_float_list(training_cfg.get("lora_dropout", [0.05]), [0.05])
    learning_rate_list = _to_float_list(training_cfg.get("learning_rate", [5e-5]), [5e-5])

    max_length = _to_int(training_cfg.get("max_length", 512), 512)
    num_train_epochs = _to_int(training_cfg.get("num_train_epochs", 3), 3)

    logging_steps = _to_int(training_cfg.get("logging_steps", 10), 10)
    max_grad_norm = _to_float(training_cfg.get("max_grad_norm", 0.3), 0.3)
    lr_scheduler_type = str(training_cfg.get("lr_scheduler_type", "constant"))
    push_to_hub = _to_bool(training_cfg.get("push_to_hub", False), False)
    report_to = str(training_cfg.get("report_to", "tensorboard"))

    output_root = Path(str(training_cfg.get("model_output_dir", "data/trained_models/")))
    output_root.mkdir(parents=True, exist_ok=True)

    max_saved_models = _to_int(training_cfg.get("max_salved_models", 5), 5)

    results_dir = Path(str(build_cfg.get("results_dir", "data/results/")))
    base_path = results_dir / "dataset_base.json"
    judge_path = results_dir / "dataset_judge.jsonl"

    judge_model_value = build_cfg.get("judge_model")
    judge_model = str(judge_model_value) if judge_model_value else None

    settings = TrainingSettings(
        dataset_load=dataset_load,
        correct_size=correct_size,
        random_seed=random_seed,
        validation_size=validation_size,
        test_size=test_size,
        model_id=model_id,
        use_qlora=use_qlora,
        lora_r_list=lora_r_list,
        lora_alpha_list=lora_alpha_list,
        lora_dropout_list=lora_dropout_list,
        learning_rate_list=learning_rate_list,
        max_length=max_length,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        logging_steps=logging_steps,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        push_to_hub=push_to_hub,
        report_to=report_to,
        output_root=output_root,
        max_saved_models=max_saved_models,
    )

    paths = TrainingPaths(
        base_path=base_path,
        judge_path=judge_path,
        output_root=output_root,
        registry_path=output_root / "registry.json",
        history_path=output_root / "run_history.jsonl",
        judge_model=judge_model,
    )

    return settings, paths
