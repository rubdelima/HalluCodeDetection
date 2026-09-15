from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_article_references import validate


class ArticleReferenceCheckTests(unittest.TestCase):
    def validate_text(self, tex: str, bib: str):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            sections = root / "sections"
            sections.mkdir()
            (sections / "sample.tex").write_text(tex, encoding="utf-8")
            bib_path = root / "refs.bib"
            bib_path.write_text(bib, encoding="utf-8")
            return validate(sections, bib_path)

    def test_valid_citations_and_commented_citations(self) -> None:
        issues, tex_count, citation_count, entry_count = self.validate_text(
            "Text \\citep[see][p.~2]{known}. % \\cite{ignored}\n",
            "@article{known, title={Known}, year={2025}}\n",
        )
        self.assertEqual((tex_count, citation_count, entry_count), (1, 1, 1))
        self.assertFalse(issues)

    def test_missing_key_has_suggestion(self) -> None:
        issues, *_ = self.validate_text(
            "Text \\cite{knownn}.\n",
            "@article{known, title={Known}, year={2025}}\n",
        )
        errors = [issue.message for issue in issues if issue.severity == "error"]
        self.assertTrue(any("missing" in message and "known" in message for message in errors))

    def test_misspelled_command_is_an_error(self) -> None:
        issues, *_ = self.validate_text(
            "Text \\ctie{known}.\n",
            "@article{known, title={Known}, year={2025}}\n",
        )
        self.assertTrue(any("unknown citation command" in issue.message for issue in issues))

    def test_duplicate_bibtex_key_is_an_error(self) -> None:
        issues, *_ = self.validate_text(
            "Text \\cite{known}.\n",
            (
                "@article{known, title={First}, year={2024}}\n"
                "@article{known, title={Second}, year={2025}}\n"
            ),
        )
        self.assertTrue(any("duplicate BibTeX key" in issue.message for issue in issues))


if __name__ == "__main__":
    unittest.main()
