from __future__ import annotations


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned.strip("_") or "model"


def grid_config_label(
    qlora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
) -> str:
    mode = "qlora" if qlora else "lora"
    return f"{mode} r={lora_r} a={lora_alpha} d={lora_dropout} lr={learning_rate}"
