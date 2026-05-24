import json

solve_problem_system = """
You are a helpful Python assistant that solves coding problems.
Return only a single function implementation and nothing else.
Do not include tests, asserts, or example usage.
If you need imports, place them inside the function body.
""".strip()


def build_solve_prompt(problem: str, signature: str | None) -> str:
    signature_block = f"Required signature: {signature}" if signature else ""
    return (
        "Solve the following problem. "
        "Return only the function implementation (no extra text).\n\n"
        f"Problem:\n{problem}\n\n"
        f"{signature_block}\n"
    ).strip()


judge_system_prompt = (
    "You are a strict code reviewer for a hallucination detection dataset. "
    "Return ONLY a valid JSON object with the key 'explanation'. "
    "Explain why the code is correct or why it is incorrect, using the provided error if any."
)


def build_judge_prompt(example_prompt: str, code: str, level: str, error: str) -> str:
    return (
        "Analyze the candidate implementation.\n\n"
        f"Problem:\n{example_prompt}\n\n"
        f"Candidate function:\n{code}\n\n"
        f"Level: {level}\n"
        f"Error: {error}\n\n"
        "Return a JSON object: {\"explanation\": \"...\"}."
    )

analyse_hallucination_prompt = """
# Task Description
You are an experienced code analyst, specializing in identifying and explaining possible code hallucinations.

Given the problem description and the user's code, you must analyze whether the code contains any hallucinations. The types of hallucinations are:

- syntax_error: When the provided code has a syntax error in the Python language, such as incorrect or incomplete code.

- runtime_error: When the code is "compilable" but has an error when executed, such as a call to an invalid function, an undefined variable, etc.

- functional_error: When the code is executable, but it has some deviation from the correct functioning of what was requested.

- correct: When the code is well-formed and has no functional errors.

Assume that the code has a sequential line, that is, if it is at a lower level (such as syntax), even if it has a high level (such as functional_error), it should only be classified as a syntax error.

# Problem Description
{problem_description}

# Output Instructions

You should return a JSON with the following fields:

- level : [syntax_error | runtime_error | functional_error | [Correct]

- Explanation: Explanation of the answer

Ex:

```json
{{
    "level": "syntax_error",
    "explanation": "The code has a missing parenthesis on line 3, which causes a syntax error."
}}
```
"""

def get_target_response(level, explanation) -> str:
    json_str = json.dumps({
        "level": level,
        "explanation": explanation
    }, ensure_ascii=False)
    target_reponse = f"```json\n{json_str}\n```"
    return target_reponse
