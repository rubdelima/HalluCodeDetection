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