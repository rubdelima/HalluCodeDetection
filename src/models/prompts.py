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


def build_generation_prompt(
    problem: str,
    signature: str | None,
    is_completion: bool,
) -> str:
    if is_completion:
        return (
            "Complete the following Python function.\n"
            "Return only the function body (properly indented), no extra text, "
            "no Markdown code fences.\n\n"
            f"{problem}"
        ).strip()
    return build_solve_prompt(problem, signature)


tool_review_system_prompt = """
You are reviewing your own Python solution with feedback from static-analysis tools.
Decide whether to submit the current solution or revise it. The tools can miss logical
and input-dependent errors, so also inspect the implementation against the problem.
Return only valid JSON, without Markdown fences, using this schema:
{"decision":"submit|revise","assessment":"short explanation","code":"complete replacement code"}
When decision is submit, copy the current code unchanged into `code`. When revising,
return the complete replacement implementation. Do not include tests or example usage.
""".strip()


developer_agent_system_prompt = """
You are the Developer agent in a software-development workflow. Produce only a
complete Python implementation for the requested task. Do not provide analysis,
JSON, Markdown fences, tests, or example usage. When the user gives evaluator and
tool feedback, revise the previous implementation accordingly.
""".strip()


analyzer_agent_system_prompt = """
You are the Analyzer agent in a software-development workflow. You never write
or modify code. Examine the candidate code against the task and static-tool
feedback, then decide whether it should be submitted or revised. Return only
valid JSON without Markdown fences:
{"decision":"submit|revise","assessment":"concise justification"}
""".strip()


def build_tool_review_prompt(
    problem: str,
    signature: str | None,
    is_completion: bool,
    code: str,
    tool_feedback: str,
    round_number: int,
    max_rounds: int,
) -> str:
    expected_form = (
        "The code may be a function body or the complete function implementation."
        if is_completion
        else f"The required function signature is: {signature or 'specified in the problem'}"
    )
    return (
        f"Problem:\n{problem}\n\n"
        f"{expected_form}\n\n"
        f"Current candidate (round {round_number} of {max_rounds}):\n{code}\n\n"
        f"Static-analysis feedback:\n{tool_feedback}\n\n"
        "Perform a self-assessment and return the required JSON."
    )


def build_analyzer_initial_prompt(
    problem: str,
    signature: str | None,
    is_completion: bool,
    code: str,
    tool_feedback: str,
    round_number: int,
    max_rounds: int,
) -> str:
    expected_form = (
        "The candidate may be a function body or a complete function implementation."
        if is_completion
        else f"The required function signature is: {signature or 'specified in the task'}"
    )
    return (
        f"Task:\n{problem}\n\n{expected_form}\n\n"
        f"Candidate code (cycle {round_number} of {max_rounds}):\n{code}\n\n"
        f"Static-tool feedback:\n{tool_feedback}\n\n"
        "Analyze this candidate and return the required decision JSON."
    )


def build_analyzer_feedback_prompt(
    code: str,
    tool_feedback: str,
    round_number: int,
    max_rounds: int,
) -> str:
    return (
        f"New candidate code (cycle {round_number} of {max_rounds}):\n{code}\n\n"
        f"Static-tool feedback:\n{tool_feedback}\n\n"
        "Using the task and previous discussion, analyze this new candidate and return "
        "the required decision JSON."
    )


def build_developer_feedback_prompt(
    assessment: str,
    tool_feedback: str,
    round_number: int,
    max_rounds: int,
) -> str:
    return (
        f"The Analyzer requested revision after cycle {round_number} of {max_rounds}.\n"
        f"Analyzer feedback:\n{assessment}\n\n"
        f"Static-tool feedback:\n{tool_feedback}\n\n"
        "Revise the previous candidate. Return only the complete Python implementation."
    )


def build_tool_assisted_judge_prompt(code: str, tool_feedback: str) -> str:
    return (
        "Candidate code:\n"
        f"{code}\n\n"
        "Static-analysis feedback (use it as evidence, but remember that it can be incomplete):\n"
        f"{tool_feedback}\n\n"
        "Classify the candidate according to the requested JSON schema."
    )


judge_system_prompt = (
    "You are a strict code reviewer for a hallucination detection dataset. "
    "Return ONLY a valid JSON object with the key 'levels', where 'levels' is a list of objects, "
    "each with the keys 'level' and 'explanation'. "
    "For every error type present in the code, write a GENERAL explanation of why the code can fail that way "
    "(do NOT mention specific test cases or specific input values). "
    "If the code is correct, return a single entry with level 'correct'."
)


def build_judge_prompt(
    example_prompt: str,
    code: str,
    model: str,
    levels: list,
) -> str:
    parts = [
        "Analyze the candidate implementation.",
        "",
        f"Model that generated the code: {model}",
        "",
        f"Problem:\n{example_prompt}",
        "",
        f"Candidate function:\n{code}",
        "",
        "The code was classified with the following error types:",
        ", ".join(lv.level_name for lv in levels),
        "",
        (
            "For EACH error type above, write a GENERAL explanation of why the code can fail that way. "
            "Do not describe specific test cases or specific inputs. "
            "Generalize the failure condition (the kind of input or situation that triggers the error and why)."
        ),
        "",
        (
            "Return ONLY a JSON object with the following format: "
            '{"levels": [{"level": "<type>", "explanation": "<general reason>"}, ...]}.'
        ),
    ]
    return "\n".join(parts)

analyse_hallucination_prompt = """
# Task Description
You are an experienced code analyst, specializing in identifying and explaining possible code hallucinations.

Given the problem description and the user's code, you must analyze whether the code contains any hallucinations. The types of hallucinations are:

- syntax: When the provided code has a syntax error in the Python language, such as incorrect or incomplete code.

- runtime: When the code is "compilable" but has an error when executed, such as a call to an invalid function, an undefined variable, etc.

- functional: When the code is executable, but it has some deviation from the correct functioning of what was requested.

- correct: When the code is well-formed and has no functional errors.

Assume that the code has a sequential line, that is, if it is at a lower level (such as syntax), even if it has a high level (such as functional), it should only be classified as a syntax error.

# Problem Description
{problem_description}

# Output Instructions

You should return a JSON with the field 'levels', which is a list of objects, each with the keys 'level' and 'explanation'.
For every error type present in the code, write a GENERAL explanation of why it can fail that way (do NOT mention specific test cases or specific inputs). A code can have more than one error type.

Ex:

```json
{{
    "levels": [
        {{"level": "syntax", "explanation": "The code has a missing parenthesis, so it cannot be parsed."}}
    ]
}}
```
"""

def get_target_response(levels, explanations) -> str:
    items = []
    for entry in explanations:
        if isinstance(entry, dict):
            items.append({"level": entry.get("level"), "explanation": entry.get("explanation")})
        else:
            items.append({"level": entry.level, "explanation": entry.explanation})
    json_str = json.dumps({"levels": items}, ensure_ascii=False)
    target_reponse = f"```json\n{json_str}\n```"
    return target_reponse
