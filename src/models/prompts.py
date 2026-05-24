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