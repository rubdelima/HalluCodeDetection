from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import HalluCodeDetectionConfig
from src.dataset.judge_dataset import load_hallucination_dataset
from src.schemas.evaluation import EvaluationSummaryRow


def sample_key(sample: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(sample["problem_description"]),
        str(sample["generated_code"]),
        str(sample["level"]),
        str(sample["explanation"]),
    )


def load_rows(path: Path) -> list[EvaluationSummaryRow]:
    if not path.exists():
        return []

    rows: list[EvaluationSummaryRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(EvaluationSummaryRow(**json.loads(line)))
    return rows


def append_rows(path: Path, rows: list[EvaluationSummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(), ensure_ascii=True) + "\n")


def build_unique_index(dataset) -> dict[tuple[str, str, str, str], int]:
    indexes_by_key: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for idx, sample in enumerate(dataset):
        indexes_by_key[sample_key(sample)].append(idx)

    return {
        key: indexes[0]
        for key, indexes in indexes_by_key.items()
        if len(indexes) == 1
    }


def migrate(args: argparse.Namespace) -> None:
    config = HalluCodeDetectionConfig(args.config)
    results_dir = Path(config.dataset_building_config.results_dir)
    source_path = Path(args.source) if args.source else results_dir / "evaluation_results.jsonl"
    target_path = Path(args.target) if args.target else results_dir / "evaluation_results_full_correct.jsonl"

    old_test = load_hallucination_dataset(config)["test"]
    new_test = load_hallucination_dataset(config, correct_size=1.0)["test"]

    old_index_to_key = {
        idx: sample_key(sample)
        for idx, sample in enumerate(old_test)
    }
    new_key_to_index = build_unique_index(new_test)
    new_samples_by_index = {
        idx: sample
        for idx, sample in enumerate(new_test)
    }

    source_rows = load_rows(source_path)
    target_rows = load_rows(target_path)
    existing_target_keys = {
        (row.model_id, row.kind, row.sample_index)
        for row in target_rows
    }

    migrated_rows: list[EvaluationSummaryRow] = []
    skipped_no_sample = 0
    skipped_duplicate = 0

    for row in source_rows:
        key = old_index_to_key.get(row.sample_index)
        if key is None or key not in new_key_to_index:
            skipped_no_sample += 1
            continue

        new_index = new_key_to_index[key]
        target_key = (row.model_id, row.kind, new_index)
        if target_key in existing_target_keys:
            skipped_duplicate += 1
            continue

        expected_level = new_samples_by_index[new_index]["level"]
        migrated = row.model_copy(
            update={
                "sample_index": new_index,
                "expected_level": expected_level,
                "correct": row.predicted_level == expected_level
                if row.predicted_level is not None
                else False,
            }
        )
        migrated_rows.append(migrated)
        existing_target_keys.add(target_key)

    print(f"source: {source_path} ({len(source_rows)} rows)")
    print(f"target: {target_path} ({len(target_rows)} existing rows)")
    print(f"old test size: {len(old_test)}")
    print(f"new test size: {len(new_test)}")
    print(f"migratable rows: {len(migrated_rows)}")
    print(f"skipped duplicates: {skipped_duplicate}")
    print(f"skipped without intersection: {skipped_no_sample}")

    if args.dry_run:
        print("dry run: no rows written")
        return

    append_rows(target_path, migrated_rows)
    print(f"wrote {len(migrated_rows)} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate completed evaluation rows from the sampled test split to the full-correct test split."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    migrate(parse_args())
