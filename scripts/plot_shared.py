"""Constantes e utilitários compartilhados entre os scripts de visualização."""

import json
from pathlib import Path

# Pasta de saída para todas as figuras
OUT_DIR = Path("figures")

# Ordem canônica das classes
LEVEL_ORDER = ["correct", "functional_error", "runtime_error", "syntax_error"]

# Rótulos em português para legendas
LEVEL_LABELS: dict[str, str] = {
    "correct": "Correto",
    "functional_error": "Erro Funcional",
    "runtime_error": "Erro de Execução",
    "syntax_error": "Erro de Sintaxe",
}

# Rótulos em português com quebra de linha (para eixos de matriz de confusão)
LEVEL_LABELS_AXIS: list[str] = [
    "Correto",
    "Erro\nFuncional",
    "Erro de\nExecução",
    "Erro de\nSintaxe",
]

# Paleta de cores acessível para daltônicos (Okabe-Ito)
COLORS: dict[str, str] = {
    "correct": "#009E73",        # verde
    "functional_error": "#E69F00",  # laranja
    "runtime_error": "#56B4E9",  # azul celeste
    "syntax_error": "#CC79A7",   # rosa
}

# Nomes de exibição dos modelos
MODEL_DISPLAY: dict[str, str] = {
    "gemma3:1b": "Gemma 3 1B",
    "gemma3:4b": "Gemma 3 4B",
    "gemma4:e4b": "Gemma 4 E4B",
    "gpt-oss:20b": "GPT-OSS 20B",
    "qwen2.5-coder:7b": "Qwen 2.5-Coder 7B",
    "qwen3.5:9b": "Qwen 3.5 9B",
}

# Rótulos dos splits em português
SPLIT_LABELS: dict[str, str] = {
    "train": "Treino",
    "validation": "Validação",
    "test": "Teste",
}


def load_jsonl(path: Path) -> list[dict]:
    """Carrega um arquivo JSONL ignorando linhas malformadas."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records
