from __future__ import annotations

import math
import random
from collections import Counter

from src.constants.training import TrainingConfig, TrainingHyperparameters
from src.schemas.training import TrainingResult
from src.training.state import hyperparameters_key, metric_for_search


def select_pending_hyperparameters(
    all_hyperparameters: list[TrainingHyperparameters],
    previous_results: list[TrainingResult],
    config: TrainingConfig,
    limit: int | None = None,
) -> list[TrainingHyperparameters]:
    trained_keys = {hyperparameters_key(item) for item in previous_results}
    pending = [
        hyperparameters
        for hyperparameters in all_hyperparameters
        if hyperparameters_key(hyperparameters) not in trained_keys
    ]
    limit = min(len(pending) if limit is None else limit, len(pending))

    if config.search_strategy == "grid":
        return pending[:limit]
    if config.search_strategy == "random":
        return _select_random_hyperparameters(pending, limit, config.random_seed)
    if config.search_strategy == "bayesian":
        return _select_bayesian_hyperparameters(pending, previous_results, config, limit)

    return pending[:limit]


def _select_random_hyperparameters(
    pending: list[TrainingHyperparameters],
    limit: int,
    seed: int,
) -> list[TrainingHyperparameters]:
    rng = random.Random(seed)
    selected = pending.copy()
    rng.shuffle(selected)
    return selected[:limit]


def _select_bayesian_hyperparameters(
    pending: list[TrainingHyperparameters],
    previous_results: list[TrainingResult],
    config: TrainingConfig,
    limit: int,
) -> list[TrainingHyperparameters]:
    rng = random.Random(config.random_seed)
    selected: list[TrainingHyperparameters] = []
    available = pending.copy()

    observations = sorted(previous_results, key=metric_for_search, reverse=True)
    if len(observations) < config.bayesian_random_starts:
        return _select_random_hyperparameters(available, limit, config.random_seed)

    good_count = max(1, math.ceil(len(observations) * config.bayesian_good_quantile))
    good_results = observations[:good_count]
    bad_results = observations[good_count:] or observations[-good_count:]

    while available and len(selected) < limit:
        sample_size = min(config.bayesian_candidates, len(available))
        candidates = rng.sample(available, sample_size)
        best_candidate = max(
            candidates,
            key=lambda candidate: _bayesian_candidate_score(candidate, good_results, bad_results),
        )
        selected.append(best_candidate)
        available.remove(best_candidate)

    return selected


def _bayesian_candidate_score(
    candidate: TrainingHyperparameters,
    good_results: list[TrainingResult],
    bad_results: list[TrainingResult],
) -> float:
    score = 0.0
    candidate_values = _value_signature(candidate)
    good_counts = {field: Counter() for field in candidate_values}
    bad_counts = {field: Counter() for field in candidate_values}

    for result in good_results:
        for field, value in _value_signature(result).items():
            good_counts[field][value] += 1

    for result in bad_results:
        for field, value in _value_signature(result).items():
            bad_counts[field][value] += 1

    for field, value in candidate_values.items():
        good_total = sum(good_counts[field].values())
        bad_total = sum(bad_counts[field].values())
        cardinality = max(
            1,
            len(set(good_counts[field]) | set(bad_counts[field]) | {value}),
        )
        good_prob = (good_counts[field][value] + 1) / (good_total + cardinality)
        bad_prob = (bad_counts[field][value] + 1) / (bad_total + cardinality)
        score += math.log(good_prob / bad_prob)

    return score


def _value_signature(hyperparameters: TrainingHyperparameters) -> dict[str, object]:
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
    }
