#!/usr/bin/env python3
"""Compile Module B sources, migrate provenance, and emit a deterministic manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
DATA_ROOT = WORKSPACE / "data" / "module_b_hutao"
CATEGORY_DIR = DATA_ROOT / "categories"
ALL_SAMPLES = WORKSPACE / "all_samples.jsonl"
SCHEMA_PATH = DATA_ROOT / "schema.json"
SPLITS = ("train", "validation", "test")
EXPECTED_RECORDS = 430
OLD_EXTERNAL_SOURCE_PATTERN = re.compile(
    r"^data1/(train|valid|test)\.jsonl:L([1-9][0-9]*)$"
)
NEW_EXTERNAL_SOURCE_PATTERN = re.compile(r"^all_samples\.jsonl:L([1-9][0-9]*)$")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(record)
    return records


def load_jsonl_with_lines(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append((line_number, record))
    return records


def render_jsonl(records: list[dict[str, Any]]) -> str:
    return "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ) + ("\n" if records else "")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_write(path, render_jsonl(records))


def write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_external_line_map() -> tuple[dict[tuple[str, int], int], set[int]]:
    relative_counts: Counter[str] = Counter()
    mapping: dict[tuple[str, int], int] = {}
    physical_lines: set[int] = set()
    for physical_line, record in load_jsonl_with_lines(ALL_SAMPLES):
        split = record.get("split")
        if split not in {"train", "valid", "test"}:
            raise ValueError(
                f"{ALL_SAMPLES}:{physical_line}: unsupported split {split!r}"
            )
        relative_counts[split] += 1
        mapping[(split, relative_counts[split])] = physical_line
        physical_lines.add(physical_line)
    if sum(relative_counts.values()) != 270:
        raise ValueError(f"{ALL_SAMPLES}: expected 270 records")
    return mapping, physical_lines


def migrate_external_source(
    record: dict[str, Any],
    line_map: dict[tuple[str, int], int],
    physical_lines: set[int],
) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict) or "external_source" not in metadata:
        return False
    source_ref = metadata["external_source"]
    if not isinstance(source_ref, str):
        raise ValueError(f"{record.get('id')}: external_source must be a string")
    old_match = OLD_EXTERNAL_SOURCE_PATTERN.fullmatch(source_ref)
    if old_match:
        source_split, relative_line_text = old_match.groups()
        key = (source_split, int(relative_line_text))
        if key not in line_map:
            raise ValueError(f"{record.get('id')}: unresolved external_source {source_ref}")
        metadata["external_source"] = f"all_samples.jsonl:L{line_map[key]}"
        return True
    new_match = NEW_EXTERNAL_SOURCE_PATTERN.fullmatch(source_ref)
    if not new_match or int(new_match.group(1)) not in physical_lines:
        raise ValueError(f"{record.get('id')}: invalid external_source {source_ref!r}")
    return False


def source_kind(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    source = metadata.get("source") if isinstance(metadata, dict) else None
    if isinstance(source, dict) and source.get("dataset") == "all_samples.jsonl":
        return "imported_all_samples"
    return "curated_module_b"


def supervised_assistant_turns(record: dict[str, Any]) -> int:
    assistant_turns = sum(
        message.get("role") == "assistant" for message in record["messages"]
    )
    if record["metadata"].get("assistant_turn_policy") == "final_only":
        return 1
    return assistant_turns


def build_manifest(
    records: list[dict[str, Any]], category_files: list[Path]
) -> dict[str, Any]:
    split_counts = Counter(record["metadata"]["split"] for record in records)
    capability_counts = Counter(record["metadata"]["capability"] for record in records)
    capability_split_counts: dict[str, dict[str, int]] = {}
    for capability in sorted(capability_counts):
        capability_split_counts[capability] = {
            split: sum(
                record["metadata"]["capability"] == capability
                and record["metadata"]["split"] == split
                for record in records
            )
            for split in SPLITS
        }

    kind_counts = Counter(source_kind(record) for record in records)
    kind_split_counts = {
        kind: {
            split: sum(
                source_kind(record) == kind
                and record["metadata"]["split"] == split
                for record in records
            )
            for split in SPLITS
        }
        for kind in sorted(kind_counts)
    }
    raw_assistant_turns = [
        sum(message["role"] == "assistant" for message in record["messages"])
        for record in records
    ]
    raw_user_turns = [
        sum(message["role"] == "user" for message in record["messages"])
        for record in records
    ]
    supervised_turn_counts = {
        split: sum(
            supervised_assistant_turns(record)
            for record in records
            if record["metadata"]["split"] == split
        )
        for split in SPLITS
    }
    assistant_lengths = [
        len(message["content"])
        for record in records
        for message in record["messages"]
        if message["role"] == "assistant"
    ]
    external_refs = [
        record["metadata"]["external_source"]
        for record in records
        if "external_source" in record["metadata"]
    ]
    normalized_split = {"train": "train", "valid": "validation", "test": "test"}
    conflict_overrides: dict[str, dict[str, str]] = {}
    for record in records:
        if source_kind(record) != "imported_all_samples":
            continue
        source = record["metadata"]["source"]
        original_split = source["original_split"]
        assigned_split = record["metadata"]["split"]
        if assigned_split != normalized_split[original_split]:
            conflict_overrides[source["scene"]] = {
                "scene": source["scene"],
                "scenario_group": record["metadata"]["scenario_group"],
                "original_split": original_split,
                "assigned_split": assigned_split,
            }

    files: dict[str, Any] = {}
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "all.jsonl"):
        path = DATA_ROOT / name
        files[name] = {
            "records": len(load_jsonl(path)),
            "sha256": sha256_file(path),
        }
    files["schema.json"] = {"sha256": sha256_file(SCHEMA_PATH)}

    source_files = []
    for path in category_files:
        source_files.append(
            {
                "file": str(path.relative_to(WORKSPACE)),
                "records": len(load_jsonl(path)),
                "sha256": sha256_file(path),
            }
        )

    return {
        "schema_version": "2.0",
        "dataset_name": "hutao_persona_grounded_sft_zh",
        "version": "2.0",
        "profile": "modern_safety_adapted_character_assistant",
        "language": "zh-CN",
        "format": "messages_jsonl",
        "generated_by": "scripts/module_b/build_dataset.py",
        "counts": {
            "records": len(records),
            "scenario_groups": len(
                {record["metadata"]["scenario_group"] for record in records}
            ),
            "splits": {split: split_counts[split] for split in SPLITS},
            "source_records": dict(sorted(kind_counts.items())),
            "source_records_by_split": kind_split_counts,
            "single_turn_records": sum(turns == 1 for turns in raw_assistant_turns),
            "multi_turn_records": sum(turns > 1 for turns in raw_assistant_turns),
            "user_turns": sum(raw_user_turns),
            "assistant_turns_raw": sum(raw_assistant_turns),
            "assistant_turns_supervised": sum(supervised_turn_counts.values()),
            "assistant_turns_supervised_by_split": supervised_turn_counts,
            "external_source_records": len(external_refs),
            "external_source_refs": len(set(external_refs)),
        },
        "assistant_content_statistics": {
            "total_characters": sum(assistant_lengths),
            "minimum_per_turn": min(assistant_lengths),
            "median_per_turn": statistics.median(assistant_lengths),
            "mean_per_turn": round(statistics.mean(assistant_lengths), 2),
            "maximum_per_turn": max(assistant_lengths),
        },
        "capabilities": dict(sorted(capability_counts.items())),
        "capabilities_by_split": capability_split_counts,
        "assistant_turn_policy": {
            "curated_module_b": "all",
            "imported_all_samples": "final_only",
        },
        "split_policy": {
            "unit": "scenario_group",
            "imported_base_mapping": {
                "train": "train",
                "valid": "validation",
                "test": "test",
            },
            "conflict_aware_overrides": [
                conflict_overrides[scene] for scene in sorted(conflict_overrides)
            ],
            "override_count": len(conflict_overrides),
            "distribution_preserved": True,
            "reason": (
                "Swap high-confidence semantic conflicts with legacy held-out "
                "concepts while preserving imported split and category totals."
            ),
        },
        "external_source_policy": {
            "field": "metadata.external_source",
            "source_file": "all_samples.jsonl",
            "reference_format": "all_samples.jsonl:L<physical-line-number>",
            "records": len(external_refs),
            "unique_references": len(set(external_refs)),
        },
        "inputs": {
            "all_samples.jsonl": {
                "records": 270,
                "sha256": sha256_file(ALL_SAMPLES),
            },
            "category_sources": source_files,
        },
        "files": files,
    }


def main() -> None:
    category_files = sorted(CATEGORY_DIR.glob("*.jsonl"))
    if not category_files:
        raise SystemExit(f"No category JSONL files found in {CATEGORY_DIR}")
    imported_path = CATEGORY_DIR / "imported_all_samples.jsonl"
    if imported_path not in category_files:
        raise SystemExit(
            "Missing imported source. Run scripts/module_b/import_all_samples.py first."
        )

    line_map, physical_lines = build_external_line_map()
    records: list[dict[str, Any]] = []
    for path in category_files:
        source_records = load_jsonl(path)
        changed = False
        for record in source_records:
            changed = (
                migrate_external_source(record, line_map, physical_lines) or changed
            )
        if changed:
            # Persist the migration so canonical category sources no longer depend
            # on the removed data1/ directory. Subsequent builds are byte-stable.
            write_jsonl(path, source_records)
        records.extend(source_records)

    id_counts = Counter(record.get("id") for record in records)
    duplicate_ids = sorted(record_id for record_id, count in id_counts.items() if count > 1)
    if duplicate_ids:
        raise SystemExit(f"Duplicate ids: {duplicate_ids}")
    if len(records) != EXPECTED_RECORDS:
        raise SystemExit(
            f"Compiled {len(records)} records; expected {EXPECTED_RECORDS}. "
            "Re-run the importer and check category sources."
        )

    records.sort(key=lambda record: record["id"])
    write_jsonl(DATA_ROOT / "all.jsonl", records)

    summary: dict[str, int] = {"all": len(records)}
    for split in SPLITS:
        split_records = [
            record for record in records if record["metadata"]["split"] == split
        ]
        write_jsonl(DATA_ROOT / f"{split}.jsonl", split_records)
        summary[split] = len(split_records)

    manifest = build_manifest(records, category_files)
    write_json(DATA_ROOT / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
