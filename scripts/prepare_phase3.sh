#!/usr/bin/env bash
# Download every configured training checkpoint before starting Phase 3.
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${workspace_root}/config.yaml"
ssd_root="/mnt/ssd/HalluCode"
optuna_db="${workspace_root}/data/results/evalplus/optuna_trials.db"
reset_search=false
start_training=false

usage() {
  cat <<'EOF'
Usage: ./scripts/prepare_phase3.sh [--reset-search] [--train]

Downloads the training models listed in config.yaml directly into their
local_path directories on the SSD. --reset-search removes the Optuna database
only when it has no completed trials. --train starts Phase 3 after preparation.
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --reset-search) reset_search=true ;;
    --train) start_training=true ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "${config_path}" ]]; then
  printf 'Configuration not found: %s\n' "${config_path}" >&2
  exit 1
fi

if [[ ! -d "${ssd_root}" || ! -w "${ssd_root}" ]]; then
  printf 'SSD path is unavailable or not writable: %s\n' "${ssd_root}" >&2
  exit 1
fi

mkdir -p "${ssd_root}/models" "${ssd_root}/checkpoints"
export HF_XET_CACHE="${ssd_root}/hf-xet-cache"
export TMPDIR="${ssd_root}/tmp"
# Reduces CUDA allocator fragmentation during the long QLoRA training run.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_XET_CACHE}" "${TMPDIR}"

if [[ "${reset_search}" == true ]]; then
  uv run python - "${optuna_db}" <<'PY'
from pathlib import Path
import sys

import optuna
from optuna.trial import TrialState

database = Path(sys.argv[1])
if not database.exists():
    print("No Optuna database to reset.")
    raise SystemExit(0)

study = optuna.load_study(
    study_name="hallucination_detection",
    storage=f"sqlite:///{database.resolve()}",
)
completed = [trial.number for trial in study.trials if trial.state == TrialState.COMPLETE]
if completed:
    raise SystemExit(
        f"Refusing to reset: completed trials exist ({completed}). Preserve the study or reset it manually."
    )

for suffix in ("", "-wal", "-shm"):
    database.with_name(database.name + suffix).unlink(missing_ok=True)
print("Removed the failed/interrupted Optuna study.")
PY
fi

uv run python - "${config_path}" <<'PY'
from pathlib import Path
import sys

import yaml
from huggingface_hub import snapshot_download

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
training_models = config.get("training", {}).get("models", [])
gemma_models = {item["id"]: item for item in config.get("models", {}).get("gemma", [])}
if not training_models:
    raise SystemExit("No training.models entries found in config.yaml.")

for model_id in training_models:
    model = gemma_models.get(model_id, {})
    local_path = model.get("local_path")
    if not local_path:
        raise SystemExit(f"No models.gemma.local_path configured for {model_id}.")
    print(f"Downloading/verifying {model_id} in {local_path}...")
    snapshot_download(repo_id=model_id, local_dir=local_path)

print("Models are ready in their SSD local_path directories.")
PY

if [[ "${start_training}" == true ]]; then
  exec uv run python "${workspace_root}/main.py" --config "${config_path}" --train_model
fi

printf '\nPreparation complete. Start Phase 3 with:\n'
printf '  ./scripts/prepare_phase3.sh --train\n'
