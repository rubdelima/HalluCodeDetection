from __future__ import annotations

from src.models.prompts import analyse_hallucination_prompt, get_target_response


def create_conversation(
    problem_description: str,
    generated_code: str,
    level: str,
    explanation: str,
) -> dict[str, list[dict[str, str]]]:
    return {
        "messages": [
            {
                "role": "system",
                "content": analyse_hallucination_prompt.format(
                    problem_description=problem_description
                ),
            },
            {"role": "user", "content": generated_code},
            {"role": "assistant", "content": get_target_response(level, explanation)},
        ]
    }
