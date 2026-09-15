#!/usr/bin/env python3
"""Statically validate citations in article/sections against article/refs.bib."""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECTIONS = PROJECT_ROOT / "article" / "sections"
DEFAULT_BIB = PROJECT_ROOT / "article" / "refs.bib"

CITE_COMMANDS = {
    "autocite",
    "cite",
    "citealp",
    "citealt",
    "citeauthor",
    "citep",
    "citet",
    "citeyear",
    "citeyearpar",
    "footcite",
    "nocite",
    "parencite",
    "smartcite",
    "supercite",
    "textcite",
}
COMMON_CITE_TYPOS = {
    "autocit",
    "ciet",
    "cit",
    "citee",
    "citeet",
    "citte",
    "ctie",
    "footcit",
    "parencit",
    "textcit",
}
KEY_PATTERN = re.compile(r"^[A-Za-z0-9_:.+/-]+$")
COMMAND_PATTERN = re.compile(r"\\([A-Za-z]+)(\*)?")
BIB_ENTRY_PATTERN = re.compile(r"(?m)^\s*@([A-Za-z]+)\s*([({])")
MANUAL_CITATION_PATTERNS = (
    re.compile(r"\b[A-Z][A-Za-z'’.-]+(?:\s+et\s+al\.)?\s*\((?:19|20)\d{2}[a-z]?\)"),
    re.compile(r"\([A-Z][A-Za-z'’.-]+(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2}[a-z]?\)"),
)
VERBATIM_ENV_PATTERN = re.compile(
    r"\\begin\{(?:verbatim\*?|lstlisting|minted)\}.*?"
    r"\\end\{(?:verbatim\*?|lstlisting|minted)\}",
    re.DOTALL,
)


@dataclass(frozen=True)
class Issue:
    severity: str
    path: Path
    line: int
    message: str


@dataclass(frozen=True)
class Citation:
    key: str
    path: Path
    line: int


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def blank_except_newlines(value: str) -> str:
    return "".join("\n" if char == "\n" else " " for char in value)


def strip_tex_comments(text: str) -> str:
    cleaned: list[str] = []
    for line in text.splitlines(keepends=True):
        comment_at: int | None = None
        for index, char in enumerate(line):
            if char != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                comment_at = index
                break
        if comment_at is None:
            cleaned.append(line)
        else:
            suffix = line[comment_at:]
            cleaned.append(line[:comment_at] + blank_except_newlines(suffix))
    return "".join(cleaned)


def strip_verbatim(text: str) -> str:
    text = VERBATIM_ENV_PATTERN.sub(lambda match: blank_except_newlines(match.group(0)), text)
    # LaTeX \verb uses the next character as its delimiter.
    return re.sub(
        r"\\verb\*?(?P<delimiter>[^A-Za-z\s]).*?(?P=delimiter)",
        lambda match: blank_except_newlines(match.group(0)),
        text,
    )


def read_balanced(text: str, start: int, opening: str, closing: str) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != opening:
        return None
    depth = 1
    cursor = start + 1
    while cursor < len(text):
        char = text[cursor]
        escaped = cursor > 0 and text[cursor - 1] == "\\"
        if not escaped and char == opening:
            depth += 1
        elif not escaped and char == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1:cursor], cursor + 1
        cursor += 1
    return None


def skip_whitespace(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def parse_citations(path: Path) -> tuple[list[Citation], list[Issue], str, list[tuple[int, int]]]:
    original = path.read_text(encoding="utf-8")
    text = strip_verbatim(strip_tex_comments(original))
    citations: list[Citation] = []
    issues: list[Issue] = []
    citation_spans: list[tuple[int, int]] = []

    for match in COMMAND_PATTERN.finditer(text):
        command = match.group(1)
        if command not in CITE_COMMANDS:
            if command.lower().startswith("cite") or command.lower() in COMMON_CITE_TYPOS:
                suggestion = difflib.get_close_matches(command, CITE_COMMANDS, n=1, cutoff=0.6)
                hint = f"; perhaps \\{suggestion[0]}" if suggestion else ""
                issues.append(Issue(
                    "error", path, line_number(text, match.start()),
                    f"unknown citation command \\{command}{hint}",
                ))
            continue

        cursor = skip_whitespace(text, match.end())
        for _ in range(2):
            if cursor >= len(text) or text[cursor] != "[":
                break
            optional = read_balanced(text, cursor, "[", "]")
            if optional is None:
                issues.append(Issue(
                    "error", path, line_number(text, match.start()),
                    f"unclosed optional argument in \\{command}",
                ))
                cursor = len(text)
                break
            _, cursor = optional
            cursor = skip_whitespace(text, cursor)

        if cursor >= len(text) or text[cursor] != "{":
            issues.append(Issue(
                "error", path, line_number(text, match.start()),
                f"\\{command} must be followed by a braced citation key",
            ))
            continue
        required = read_balanced(text, cursor, "{", "}")
        if required is None:
            issues.append(Issue(
                "error", path, line_number(text, match.start()),
                f"unclosed citation argument in \\{command}",
            ))
            continue

        raw_keys, end = required
        citation_spans.append((match.start(), end))
        for raw_key in raw_keys.split(","):
            key = raw_key.strip()
            key_line = line_number(text, cursor + 1 + raw_keys.find(raw_key))
            if not key:
                issues.append(Issue("error", path, key_line, f"empty key in \\{command}"))
                continue
            if command == "nocite" and key == "*":
                continue
            if raw_key != key:
                issues.append(Issue(
                    "warning", path, key_line,
                    f"citation key {raw_key!r} has surrounding whitespace",
                ))
            if not KEY_PATTERN.fullmatch(key):
                issues.append(Issue(
                    "error", path, key_line,
                    f"citation key {key!r} contains invalid characters or whitespace",
                ))
                continue
            citations.append(Citation(key, path, key_line))
    return citations, issues, text, citation_spans


def find_entry_end(text: str, start: int, opening: str) -> int | None:
    closing = "}" if opening == "{" else ")"
    depth = 1
    in_quote = False
    escaped = False
    cursor = start + 1
    while cursor < len(text):
        char = text[cursor]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            in_quote = not in_quote
        elif not in_quote:
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return cursor + 1
        cursor += 1
    return None


def parse_bibliography(path: Path) -> tuple[dict[str, int], list[Issue]]:
    text = path.read_text(encoding="utf-8")
    entries: dict[str, int] = {}
    casefolded: dict[str, str] = {}
    issues: list[Issue] = []

    for match in BIB_ENTRY_PATTERN.finditer(text):
        entry_type = match.group(1).lower()
        opening = match.group(2)
        entry_start = match.end() - 1
        end = find_entry_end(text, entry_start, opening)
        entry_line = line_number(text, match.start())
        if end is None:
            issues.append(Issue(
                "error", path, entry_line,
                f"unterminated @{entry_type} entry",
            ))
            continue
        if entry_type in {"comment", "preamble", "string"}:
            continue

        body = text[entry_start + 1:end - 1]
        comma = body.find(",")
        if comma < 0:
            issues.append(Issue(
                "error", path, entry_line,
                f"@{entry_type} entry is missing the comma after its key",
            ))
            continue
        key = body[:comma].strip()
        if not key or not KEY_PATTERN.fullmatch(key):
            issues.append(Issue(
                "error", path, entry_line,
                f"invalid or empty BibTeX key {key!r}",
            ))
            continue
        if key in entries:
            issues.append(Issue(
                "error", path, entry_line,
                f"duplicate BibTeX key {key!r}; first declared on line {entries[key]}",
            ))
            continue
        folded = key.casefold()
        if folded in casefolded:
            issues.append(Issue(
                "warning", path, entry_line,
                f"BibTeX key {key!r} differs only by case from {casefolded[folded]!r}",
            ))
        casefolded[folded] = key
        entries[key] = entry_line
    return entries, issues


def manual_citation_issues(
    path: Path,
    text: str,
    citation_spans: list[tuple[int, int]],
) -> list[Issue]:
    masked = list(text)
    for start, end in citation_spans:
        for index in range(start, end):
            if masked[index] != "\n":
                masked[index] = " "
    remaining = "".join(masked)
    issues: list[Issue] = []
    seen: set[tuple[int, str]] = set()
    for pattern in MANUAL_CITATION_PATTERNS:
        for match in pattern.finditer(remaining):
            found = match.group(0)
            line = line_number(remaining, match.start())
            if (line, found) in seen:
                continue
            seen.add((line, found))
            issues.append(Issue(
                "warning", path, line,
                f"possible manually written author-year citation {found!r}; use a citation command",
            ))
    return issues


def validate(
    sections_dir: Path,
    bib_path: Path,
    *,
    check_manual: bool = True,
) -> tuple[list[Issue], int, int, int]:
    issues: list[Issue] = []
    if not sections_dir.is_dir():
        return [Issue("error", sections_dir, 1, "sections directory does not exist")], 0, 0, 0
    if not bib_path.is_file():
        return [Issue("error", bib_path, 1, "bibliography file does not exist")], 0, 0, 0

    bib_entries, bib_issues = parse_bibliography(bib_path)
    issues.extend(bib_issues)
    citations: list[Citation] = []
    tex_files = sorted(sections_dir.rglob("*.tex"))
    for path in tex_files:
        found, tex_issues, text, spans = parse_citations(path)
        citations.extend(found)
        issues.extend(tex_issues)
        if check_manual:
            issues.extend(manual_citation_issues(path, text, spans))

    folded_bib = {key.casefold(): key for key in bib_entries}
    for citation in citations:
        if citation.key in bib_entries:
            continue
        if citation.key.casefold() in folded_bib:
            issues.append(Issue(
                "error", citation.path, citation.line,
                f"citation key {citation.key!r} has incorrect capitalization; "
                f"BibTeX contains {folded_bib[citation.key.casefold()]!r}",
            ))
            continue
        suggestions = difflib.get_close_matches(citation.key, bib_entries, n=3, cutoff=0.55)
        hint = f"; closest: {', '.join(suggestions)}" if suggestions else ""
        issues.append(Issue(
            "error", citation.path, citation.line,
            f"citation key {citation.key!r} is missing from {bib_path.name}{hint}",
        ))

    cited_keys = {citation.key for citation in citations}
    for key, line in bib_entries.items():
        if key not in cited_keys:
            issues.append(Issue(
                "warning", bib_path, line,
                f"BibTeX entry {key!r} is not cited in article/sections",
            ))

    issues.sort(key=lambda item: (str(item.path), item.line, item.severity, item.message))
    return issues, len(tex_files), len(citations), len(bib_entries)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check article/sections citation keys against article/refs.bib.",
    )
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--bib", type=Path, default=DEFAULT_BIB)
    parser.add_argument(
        "--no-manual-citation-check",
        action="store_true",
        help="Do not warn about probable author-year citations written without \\cite.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a failure status for warnings as well as errors.",
    )
    args = parser.parse_args()

    issues, tex_count, citation_count, entry_count = validate(
        args.sections,
        args.bib,
        check_manual=not args.no_manual_citation_check,
    )
    for issue in issues:
        print(
            f"{issue.severity.upper():7} {display_path(issue.path)}:{issue.line}: "
            f"{issue.message}"
        )

    errors = sum(issue.severity == "error" for issue in issues)
    warnings = sum(issue.severity == "warning" for issue in issues)
    print(
        f"Checked {tex_count} TeX files, {citation_count} citation uses, "
        f"and {entry_count} BibTeX entries: {errors} error(s), {warnings} warning(s)."
    )
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
