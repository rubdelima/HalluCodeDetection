from __future__ import annotations

from src.schemas.evaluation import BASE_LEVEL_DICT, EvaluationResume


LEVELS = tuple(BASE_LEVEL_DICT)


def recalls_by_level(resume: EvaluationResume) -> dict[str, float]:
    """Return multilabel recall for each hallucination class."""
    return {
        level: (
            resume.true_positives[level] / (resume.true_positives[level] + resume.false_negatives[level])
            if resume.true_positives[level] + resume.false_negatives[level]
            else 0.0
        )
        for level in LEVELS
    }


def add_classification_metrics(
    resume: EvaluationResume,
    expected_levels: list[str],
    predicted_levels: list[str],
) -> None:
    """Accumulate multilabel classification metrics for one generated response."""
    expected = set(expected_levels)
    predicted = set(predicted_levels)

    for level in LEVELS:
        if level in expected and level in predicted:
            resume.true_positives[level] += 1
        elif level not in expected and level in predicted:
            resume.false_positives[level] += 1
        elif level in expected and level not in predicted:
            resume.false_negatives[level] += 1

    f1_scores: list[float] = []
    recall_scores = list(recalls_by_level(resume).values())
    for level in LEVELS:
        tp = resume.true_positives[level]
        fp = resume.false_positives[level]
        fn = resume.false_negatives[level]
        f1_scores.append((2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    resume.macro_f1 = sum(f1_scores) / len(f1_scores)
    resume.macro_recall = sum(recall_scores) / len(recall_scores)
