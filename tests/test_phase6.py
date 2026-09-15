from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.phase6.run import _comparison_rows, _write_summary_csv
from src.schemas.dataset import BaseResultRow, Level
from src.schemas.phase6 import Phase6ResultRow, Phase6Round
from src.static_analysis.feedback import StaticToolFeedback


class Phase6SummaryTests(unittest.TestCase):
    def test_summary_contains_average_cycles_and_level_counts(self) -> None:
        counts = {
            "deepseek-v4.1-flash": {
                "correct": 2,
                "functional": 1,
                "runtime": 0,
                "syntax": 0,
                "total": 3,
            }
        }
        cycle_totals = {"deepseek-v4.1-flash": 5}

        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "summary.csv"
            _write_summary_csv(path, counts, cycle_totals)
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Model"], "deepseek-v4.1-flash")
        self.assertEqual(rows[0]["Avg Cycles"], "1.6667")
        self.assertEqual(rows[0]["Correct"], "2")
        self.assertEqual(rows[0]["Functional"], "1")

    def test_comparison_uses_matched_tasks_and_percentage_point_deltas(self) -> None:
        model = "deepseek-v4.1-flash"
        baseline = [
            BaseResultRow(
                benchmark="humaneval",
                benchmark_id=index,
                task_id=f"HumanEval/{index}",
                model=model,
                levels=[Level(level)],
                tc_ok=0,
                tc_fail=0,
            )
            for index, level in ((1, "correct"), (2, "functional"))
        ]
        feedback = StaticToolFeedback(compile_ok=True, pyright_ok=True)
        phase6 = [
            Phase6ResultRow(
                benchmark="humaneval",
                benchmark_id=index,
                task_id=f"HumanEval/{index}",
                model=model,
                rounds=[
                    Phase6Round(round_number=round_number, code="pass", tool_feedback=feedback)
                    for round_number in range(1, cycles + 1)
                ],
                levels=[Level("correct")],
            )
            for index, cycles in ((1, 1), (2, 2))
        ]

        rows, coverage = _comparison_rows(phase6, baseline, {model})

        self.assertEqual(coverage[model], 2)
        self.assertEqual(rows[0]["Avg Cycles"], "1.50 (+0.50)")
        self.assertEqual(rows[0]["Correct"], "100.0% (+50.0)")
        self.assertEqual(rows[0]["Functional"], "0.0% (-50.0)")


if __name__ == "__main__":
    unittest.main()
