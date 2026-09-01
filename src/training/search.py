"""Persistent Optuna search over the project's finite training grid."""

from __future__ import annotations

import json
from pathlib import Path

import optuna
from optuna.trial import TrialState

from src.constants.training import TrainingConfig, TrainingHyperparameters
from src.schemas.training import TrainingResult
from src.training.state import metric_for_search


_TRIAL_FIELDS = (
    "model_name", "use_qlora", "lora_r", "lora_alpha", "lora_dropout",
    "learning_rate", "num_epochs", "bias", "optimizer", "classification_loss_strategy",
    "classification_loss_weight", "class_balance_ratio",
)


def create_study(config: TrainingConfig, storage_path: Path, previous_results: list[TrainingResult]) -> optuna.Study:
    """Open the resumable study and import pre-Optuna JSONL results once."""
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        seed=config.random_seed,
        n_startup_trials=config.optuna_startup_trials,
        multivariate=True,
        constant_liar=True,
    )
    study = optuna.create_study(
        study_name=config.optuna_study_name,
        storage=f"sqlite:///{storage_path.resolve()}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    _mark_interrupted_trials_failed(study)
    _import_legacy_results(study, config.hyperparameters, previous_results, config.optuna_target)
    return study


def completed_trial_count(study: optuna.Study) -> int:
    """Count evaluated configurations; pruned duplicate suggestions do not count."""
    return sum(
        trial.state in (TrialState.COMPLETE, TrialState.FAIL)
        for trial in study.get_trials(deepcopy=False)
    )


def ask_next_trial(study: optuna.Study, candidates: list[TrainingHyperparameters]):
    """Ask Optuna for one untried point from the configured finite grid."""
    candidates_by_params = {_params_key(_trial_params(item)): item for item in candidates}
    used = {
        _params_key(trial.params)
        for trial in study.get_trials(deepcopy=False)
        if trial.state in (TrialState.COMPLETE, TrialState.FAIL) and trial.params
    }
    if all(_params_key(_trial_params(item)) in used for item in candidates):
        return None

    # TPE samples a product space and can suggest a previously evaluated point.
    # Record those suggestions as PRUNED, then ask again without training twice.
    max_duplicate_attempts = max(100, len(candidates) * 2)
    for _ in range(max_duplicate_attempts):
        trial = study.ask()
        params = _suggest_parameters(trial, candidates)
        hyperparameters = candidates_by_params.get(_params_key(params))
        if hyperparameters is not None and _params_key(_trial_params(hyperparameters)) not in used:
            return trial, hyperparameters
        study.tell(trial, state=TrialState.PRUNED)

    raise RuntimeError("Optuna repeatedly suggested configurations already evaluated.")


def complete_trial(trial: optuna.trial.Trial, result: TrainingResult, target: str) -> None:
    """Persist the validation objective and lightweight run metadata."""
    trial.set_user_attr("run_path", result.run_path)
    trial.set_user_attr("model_path", result.model_path)
    trial.set_user_attr("train_acc", result.train_acc)
    trial.set_user_attr("test_acc", result.test_acc)
    trial.set_user_attr("training_time", result.training_time)
    trial.set_user_attr("val_acc", result.val_acc)
    trial.set_user_attr("val_macro_f1", result.val_macro_f1)
    trial.set_user_attr("val_macro_recall", result.val_macro_recall)
    value = metric_for_search(result, target)  # type: ignore[arg-type]
    if value is None:
        raise ValueError(f"Training result is missing the configured Optuna target: {target}")
    trial.study.tell(trial, value)


def fail_trial(trial: optuna.trial.Trial, error: Exception) -> None:
    """Persist failures so a resumed search does not retry the same point."""
    trial.set_user_attr("error", f"{type(error).__name__}: {error}"[:2000])
    trial.study.tell(trial, state=TrialState.FAIL)


def _mark_interrupted_trials_failed(study: optuna.Study) -> None:
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(trial.number, state=TrialState.FAIL)


def _import_legacy_results(
    study: optuna.Study,
    candidates: list[TrainingHyperparameters],
    previous_results: list[TrainingResult],
    target: str,
) -> None:
    known = {_params_key(trial.params) for trial in study.get_trials(deepcopy=False) if trial.params}
    candidate_keys = {_params_key(_trial_params(candidate)) for candidate in candidates}
    distributions = _distributions(candidates)
    for result in previous_results:
        params = _trial_params(result)
        params_key = _params_key(params)
        if params_key not in candidate_keys or params_key in known:
            continue
        value = metric_for_search(result, target)  # type: ignore[arg-type]
        # Results from earlier searches do not have Macro F1/Recall and must
        # not be mixed into a study using either as its target.
        if value is None:
            continue
        study.add_trial(optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            value=value,
            user_attrs={"imported_from_jsonl": True, "test_acc": result.test_acc},
        ))
        known.add(params_key)


def _suggest_parameters(trial: optuna.trial.Trial, candidates: list[TrainingHyperparameters]) -> dict[str, object]:
    return {
        name: trial.suggest_categorical(name, distribution.choices)
        for name, distribution in _distributions(candidates).items()
    }


def _distributions(candidates: list[TrainingHyperparameters]) -> dict[str, optuna.distributions.CategoricalDistribution]:
    if not candidates:
        raise ValueError("No training hyperparameters configured.")
    values = {name: [] for name in _TRIAL_FIELDS}
    for candidate in candidates:
        for name, value in _trial_params(candidate).items():
            if value not in values[name]:
                values[name].append(value)
    return {name: optuna.distributions.CategoricalDistribution(choices) for name, choices in values.items()}


def _trial_params(hyperparameters: TrainingHyperparameters) -> dict[str, object]:
    return {
        "model_name": hyperparameters.model_name.id,
        "use_qlora": hyperparameters.use_qlora,
        "lora_r": hyperparameters.lora_r,
        "lora_alpha": hyperparameters.lora_alpha,
        "lora_dropout": hyperparameters.lora_dropout,
        "learning_rate": hyperparameters.learning_rate,
        "num_epochs": hyperparameters.num_epochs,
        "bias": hyperparameters.bias,
        "optimizer": hyperparameters.optimizer,
        "classification_loss_strategy": hyperparameters.classification_loss_strategy,
        "classification_loss_weight": hyperparameters.classification_loss_weight,
        "class_balance_ratio": ":".join(map(str, hyperparameters.class_balance_ratio)),
    }


def _params_key(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))
