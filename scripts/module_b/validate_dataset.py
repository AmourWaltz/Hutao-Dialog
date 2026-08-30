#!/usr/bin/env python3
"""Validate Module B structure, provenance, split isolation, and content gates."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.module_b.build_dataset import build_manifest, sha256_file
    from scripts.module_b.import_all_samples import (
        EXPECTED_SYSTEM,
        SOURCE_DATASET,
        load_source_records,
        normalize_record,
    )
except ModuleNotFoundError:  # Direct execution from scripts/module_b/.
    from build_dataset import build_manifest, sha256_file
    from import_all_samples import (  # type: ignore
        EXPECTED_SYSTEM,
        SOURCE_DATASET,
        load_source_records,
        normalize_record,
    )


WORKSPACE = Path(__file__).resolve().parents[2]
DATA_ROOT = WORKSPACE / "data" / "module_b_hutao"
CATEGORY_DIR = DATA_ROOT / "categories"
SCHEMA_PATH = DATA_ROOT / "schema.json"
MODULE_A_INDEX = WORKSPACE / "output" / "module_a_hutao" / "representative_corpus_index.jsonl"
SPLITS = ("train", "validation", "test")
CAPABILITIES = {
    "daily_chat",
    "wordplay_poetry",
    "business_humor",
    "relationship_sensitive",
    "professional_funeral",
    "worldview_life_death",
    "empathy_grief_support",
    "crisis_leadership",
    "knowledge_boundary",
}
CAPABILITY_BY_CODE = {
    "DLY": "daily_chat",
    "WDP": "wordplay_poetry",
    "BUS": "business_humor",
    "REL": "relationship_sensitive",
    "PRO": "professional_funeral",
    "WLD": "worldview_life_death",
    "EMP": "empathy_grief_support",
    "CRI": "crisis_leadership",
}
CAPABILITY_CODES = "DLY|WDP|BUS|REL|PRO|WLD|EMP|CRI"
CURATED_ID_PATTERN = re.compile(
    rf"^HT-({CAPABILITY_CODES})-G(0[1-9]|10)-V([12])$"
)
IMPORTED_ID_PATTERN = re.compile(
    r"^hutao-(business_professional|daily_playful|grief_support|"
    r"knowledge_boundary|life_death_values|liyue_relationships|"
    r"poetry_wordplay|safety_crisis|traveler_paimon)-[a-z0-9_]+-v([012])$"
)
EXTERNAL_SOURCE_PATTERN = re.compile(r"^all_samples\.jsonl:L([1-9][0-9]*)$")
RECORD_KEYS = {"id", "messages", "metadata"}
METADATA_REQUIRED_KEYS = {
    "split",
    "capability",
    "scenario_group",
    "register",
    "seriousness",
    "relationship",
    "humor_types",
    "death_topic_mode",
    "source_basis",
    "construction",
    "contains_verbatim_game_text",
    "risk_flags",
}
CURATED_OPTIONAL_KEYS = {"external_source"}
IMPORTED_REQUIRED_KEYS = {"assistant_turn_policy", "source"}
SERIOUS_SALES_TERMS = ("优惠", "套餐", "折扣", "买一送一", "预订", "客户", "哈哈", "嘿嘿")
PROFESSIONAL_JOKE_TERMS = ("优惠", "套餐", "折扣", "买一送一", "哈哈", "嘿嘿")
SERIOUS_JOKE_TERMS = ("哈哈", "嘿嘿", "买一送一")
STYLE_MARKERS = ("本堂主", "胡桃", "嘿嘿", "哎呀", "嘛", "啦", "喽")
EXPECTED_SPLIT_COUNTS = {"train": 344, "validation": 43, "test": 43}
EXPECTED_SOURCE_COUNTS = {"curated_module_b": 160, "imported_all_samples": 270}
EXPECTED_SOURCE_SPLIT_COUNTS = {
    "curated_module_b": {"train": 128, "validation": 16, "test": 16},
    "imported_all_samples": {"train": 216, "validation": 27, "test": 27},
}
EXPECTED_CAPABILITY_SPLIT_COUNTS = {
    "business_humor": {"train": 16, "validation": 2, "test": 2},
    "crisis_leadership": {"train": 34, "validation": 5, "test": 5},
    "daily_chat": {"train": 46, "validation": 5, "test": 5},
    "empathy_grief_support": {"train": 46, "validation": 5, "test": 5},
    "knowledge_boundary": {"train": 18, "validation": 3, "test": 3},
    "professional_funeral": {"train": 46, "validation": 5, "test": 5},
    "relationship_sensitive": {"train": 64, "validation": 8, "test": 8},
    "wordplay_poetry": {"train": 28, "validation": 5, "test": 5},
    "worldview_life_death": {"train": 46, "validation": 5, "test": 5},
}
EXPECTED_SUPERVISED_TURNS = {"train": 406, "validation": 50, "test": 50}


def load_jsonl(path: Path, annotate: bool = False) -> list[dict[str, Any]]:
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
        if annotate:
            record["_source_file"] = path.name
            record["_line_number"] = line_number
        records.append(record)
    return records


def clean(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def assistant_messages(record: dict[str, Any]) -> list[str]:
    return [
        message["content"]
        for message in record["messages"]
        if message["role"] == "assistant"
    ]


def user_text(record: dict[str, Any]) -> str:
    return "\n".join(
        message["content"]
        for message in record["messages"]
        if message["role"] == "user"
    )


def supervised_assistant_text(record: dict[str, Any]) -> str:
    values = assistant_messages(record)
    if record["metadata"].get("assistant_turn_policy") == "final_only":
        return values[-1]
    return "\n".join(values)


def source_kind(record: dict[str, Any]) -> str:
    source = record.get("metadata", {}).get("source")
    if isinstance(source, dict) and source.get("dataset") == SOURCE_DATASET:
        return "imported_all_samples"
    return "curated_module_b"


def protected_excerpts() -> list[str]:
    if not MODULE_A_INDEX.exists():
        return []
    excerpts: list[str] = []
    for line in MODULE_A_INDEX.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        excerpt = json.loads(line).get("text_excerpt", "").strip()
        if len(excerpt) >= 8:
            excerpts.append(excerpt)
    return excerpts


def validate_schema(
    records: list[dict[str, Any]], errors: list[str], warnings: list[str]
) -> dict[str, Any]:
    try:
        from jsonschema import Draft7Validator
    except ImportError:
        warnings.append("jsonschema is unavailable; schema validation was skipped")
        return {"status": "skipped", "records_checked": 0, "errors": 0}

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)
    schema_error_count = 0
    for record in records:
        source = f"{record.get('_source_file')}:{record.get('_line_number')}"
        for issue in validator.iter_errors(clean(record)):
            schema_error_count += 1
            path = ".".join(str(item) for item in issue.absolute_path) or "<record>"
            errors.append(
                f"{source} {record.get('id')}: schema {path}: {issue.message}"
            )
    return {
        "status": "pass" if schema_error_count == 0 else "fail",
        "records_checked": len(records),
        "errors": schema_error_count,
    }


def validate_messages(record: dict[str, Any], source: str, errors: list[str]) -> bool:
    record_id = record.get("id", "<missing-id>")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        errors.append(f"{source} {record_id}: messages must contain system, user, assistant")
        return False
    roles = [message.get("role") if isinstance(message, dict) else None for message in messages]
    expected_roles = ["system"] + [
        "user" if index % 2 else "assistant" for index in range(1, len(messages))
    ]
    if roles != expected_roles or roles[-1] != "assistant":
        errors.append(f"{source} {record_id}: invalid role order {roles}")
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            errors.append(f"{source} {record_id}: malformed or empty message")
    if isinstance(messages[0], dict) and messages[0].get("content") != EXPECTED_SYSTEM:
        errors.append(f"{source} {record_id}: non-standard system message")
    return True


def validate_common_metadata(
    record: dict[str, Any], source: str, imported: bool, errors: list[str]
) -> bool:
    record_id = record.get("id", "<missing-id>")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        errors.append(f"{source} {record_id}: metadata must be an object")
        return False
    expected_keys = set(METADATA_REQUIRED_KEYS)
    expected_keys |= IMPORTED_REQUIRED_KEYS if imported else set()
    allowed_keys = expected_keys | (set() if imported else CURATED_OPTIONAL_KEYS)
    if set(metadata) != expected_keys and not (
        not imported and set(metadata) == expected_keys | {"external_source"}
    ):
        errors.append(
            f"{source} {record_id}: metadata keys missing="
            f"{sorted(expected_keys - set(metadata))} unexpected="
            f"{sorted(set(metadata) - allowed_keys)}"
        )
        return False
    if metadata.get("split") not in SPLITS:
        errors.append(f"{source} {record_id}: invalid split {metadata.get('split')!r}")
    if metadata.get("capability") not in CAPABILITIES:
        errors.append(
            f"{source} {record_id}: unknown capability {metadata.get('capability')!r}"
        )
    seriousness = metadata.get("seriousness")
    if not isinstance(seriousness, int) or isinstance(seriousness, bool) or not 1 <= seriousness <= 5:
        errors.append(f"{source} {record_id}: seriousness must be int 1..5")
    for key in ("humor_types", "source_basis", "risk_flags"):
        values = metadata.get(key)
        if (
            not isinstance(values, list)
            or any(not isinstance(item, str) or not item for item in values)
            or len(values) != len(set(values))
        ):
            errors.append(f"{source} {record_id}: malformed or duplicate {key}")
    if metadata.get("construction") != "synthetic_persona_grounded":
        errors.append(f"{source} {record_id}: invalid construction")
    if metadata.get("contains_verbatim_game_text") is not False:
        errors.append(f"{source} {record_id}: verbatim flag must be false")
    for field in ("register", "relationship", "death_topic_mode", "scenario_group"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            errors.append(f"{source} {record_id}: metadata {field} must be non-empty")
    return True


def external_record(
    source_ref: str, source_by_line: dict[int, dict[str, Any]]
) -> tuple[dict[str, Any] | None, str | None]:
    match = EXTERNAL_SOURCE_PATTERN.fullmatch(source_ref)
    if not match:
        return None, f"invalid external_source {source_ref!r}"
    line_number = int(match.group(1))
    record = source_by_line.get(line_number)
    if record is None:
        return None, f"external_source physical line does not exist: {source_ref!r}"
    return record, None


def validate_curated_record(
    record: dict[str, Any],
    source: str,
    source_by_line: dict[int, dict[str, Any]],
    external_targets: set[str],
    errors: list[str],
) -> tuple[str | None, str | None]:
    record_id = record.get("id", "<missing-id>")
    metadata = record["metadata"]
    match = CURATED_ID_PATTERN.fullmatch(str(record_id))
    if not match:
        errors.append(f"{source} {record_id}: invalid curated id pattern")
        return None, None
    code, group_number_text, variant = match.groups()
    group_number = int(group_number_text)
    expected_group = record_id[3:-3]
    if metadata["scenario_group"] != expected_group:
        errors.append(
            f"{source} {record_id}: scenario_group {metadata['scenario_group']!r} "
            f"!= {expected_group!r}"
        )
    if metadata["capability"] != CAPABILITY_BY_CODE[code]:
        errors.append(f"{source} {record_id}: capability/id mismatch")
    expected_split = (
        "train"
        if group_number <= 6 or group_number in {9, 10}
        else "validation"
        if group_number == 7
        else "test"
    )
    if metadata["split"] != expected_split:
        errors.append(f"{source} {record_id}: group must be in {expected_split}")
    basis = metadata.get("source_basis", [])
    if not basis or any(not re.fullmatch(r"A(0[1-9]|1[0-8])", item) for item in basis):
        errors.append(f"{source} {record_id}: invalid source_basis {basis}")

    has_external = "external_source" in metadata
    if group_number in {9, 10} and not has_external:
        errors.append(f"{source} {record_id}: G09/G10 requires external_source")
    if group_number <= 8 and has_external:
        errors.append(f"{source} {record_id}: G01-G08 must not contain external_source")
    if has_external:
        source_ref = metadata["external_source"]
        if not isinstance(source_ref, str):
            errors.append(f"{source} {record_id}: external_source must be a string")
        else:
            candidate, external_error = external_record(source_ref, source_by_line)
            if external_error:
                errors.append(f"{source} {record_id}: {external_error}")
            elif candidate is not None:
                if candidate.get("split") != "train":
                    errors.append(
                        f"{source} {record_id}: external concept must come from source train"
                    )
                candidate_assistants = [
                    message["content"]
                    for message in candidate.get("messages", [])
                    if message.get("role") == "assistant"
                ]
                if not candidate_assistants:
                    errors.append(f"{source} {record_id}: external record has no assistant reply")
                else:
                    normalized_candidate = normalize(candidate_assistants[-1])
                    normalized_record_turns = {
                        normalize(value) for value in assistant_messages(record)
                    }
                    normalized_all = normalize(supervised_assistant_text(record))
                    if normalized_candidate in normalized_record_turns or normalized_candidate == normalized_all:
                        errors.append(
                            f"{source} {record_id}: assistant exactly copies {source_ref}"
                        )
                    elif len(normalized_candidate) >= 20 and normalized_candidate in normalized_all:
                        errors.append(
                            f"{source} {record_id}: assistant contains full reply from {source_ref}"
                        )
                    copied_turns = [
                        value
                        for value in assistant_messages(record)
                        if normalize(value) in external_targets
                    ]
                    if copied_turns:
                        errors.append(
                            f"{source} {record_id}: assistant copies another external candidate reply"
                        )
    return metadata["scenario_group"], variant


def validate_imported_record(
    record: dict[str, Any],
    source: str,
    expected_imports_by_line: dict[int, dict[str, Any]],
    errors: list[str],
) -> tuple[str | None, str | None]:
    record_id = record.get("id", "<missing-id>")
    metadata = record["metadata"]
    match = IMPORTED_ID_PATTERN.fullmatch(str(record_id))
    if not match:
        errors.append(f"{source} {record_id}: invalid imported id pattern")
        return None, None
    source_meta = metadata.get("source")
    if not isinstance(source_meta, dict):
        errors.append(f"{source} {record_id}: imported record has no source object")
        return None, None
    line_number = source_meta.get("line")
    if not isinstance(line_number, int) or isinstance(line_number, bool):
        errors.append(f"{source} {record_id}: source.line must be an integer")
        return metadata.get("scenario_group"), match.group(2)
    expected = expected_imports_by_line.get(line_number)
    if expected is None:
        errors.append(f"{source} {record_id}: source.line {line_number} does not resolve")
    elif clean(record) != expected:
        errors.append(
            f"{source} {record_id}: imported record differs from deterministic normalization "
            f"of all_samples.jsonl:L{line_number}"
        )
    if metadata.get("assistant_turn_policy") != "final_only":
        errors.append(f"{source} {record_id}: imported policy must be final_only")
    return metadata.get("scenario_group"), match.group(2)


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        path = DATA_ROOT / f"{split}.jsonl"
        if not path.exists():
            errors.append(f"missing compiled split: {path}")
            continue
        split_records = load_jsonl(path, annotate=True)
        for record in split_records:
            if record.get("metadata", {}).get("split") != split:
                errors.append(f"{record.get('id')}: metadata split does not match {path.name}")
        records.extend(split_records)

    source_rows = load_source_records()
    source_by_line = {line_number: record for line_number, record in source_rows}
    expected_imports = [
        normalize_record(record, line_number) for line_number, record in source_rows
    ]
    expected_imports_by_line = {
        record["metadata"]["source"]["line"]: record for record in expected_imports
    }
    imported_category_path = CATEGORY_DIR / "imported_all_samples.jsonl"
    if not imported_category_path.exists():
        errors.append(f"missing imported category source: {imported_category_path}")
    else:
        imported_category_records = load_jsonl(imported_category_path)
        if imported_category_records != expected_imports:
            errors.append(
                "imported_all_samples.jsonl is stale; re-run import_all_samples.py"
            )

    schema_validation = validate_schema(records, errors, warnings)
    seen_ids: set[str] = set()
    imported_source_lines: set[int] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    group_variants: dict[str, set[str]] = defaultdict(set)
    group_kinds: dict[str, set[str]] = defaultdict(set)
    normalized_users: dict[str, list[str]] = defaultdict(list)
    normalized_targets: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    excerpts = protected_excerpts()
    external_targets = {
        normalize(message["content"])
        for _, source_record in source_rows
        for message in source_record["messages"]
        if message["role"] == "assistant" and normalize(message["content"])
    }

    for record in records:
        record_id = record.get("id", "<missing-id>")
        source = f"{record.get('_source_file')}:{record.get('_line_number')}"
        if set(clean(record)) != RECORD_KEYS:
            errors.append(f"{source} {record_id}: record keys {sorted(clean(record))}")
            continue
        if record_id in seen_ids:
            errors.append(f"{source} {record_id}: duplicate id")
        seen_ids.add(record_id)
        if not validate_messages(record, source, errors):
            continue

        imported = source_kind(record) == "imported_all_samples"
        if not validate_common_metadata(record, source, imported, errors):
            continue
        metadata = record["metadata"]
        if imported:
            group, variant = validate_imported_record(
                record, source, expected_imports_by_line, errors
            )
            source_line = metadata.get("source", {}).get("line")
            if isinstance(source_line, int):
                if source_line in imported_source_lines:
                    errors.append(
                        f"{source} {record_id}: duplicate imported source line {source_line}"
                    )
                imported_source_lines.add(source_line)
        else:
            group, variant = validate_curated_record(
                record, source, source_by_line, external_targets, errors
            )

        if isinstance(group, str) and isinstance(variant, str):
            group_splits[group].add(metadata["split"])
            group_variants[group].add(variant)
            group_kinds[group].add("imported" if imported else "curated")

        target = supervised_assistant_text(record)
        if not 15 <= len(target) <= 420:
            errors.append(
                f"{source} {record_id}: supervised assistant chars {len(target)} outside 15..420"
            )
        if any(phrase in target for phrase in ("作为一个AI", "根据角色设定", "在《原神》中")):
            errors.append(f"{source} {record_id}: meta-assistant phrasing detected")
        capability = metadata["capability"]
        seriousness = metadata["seriousness"]
        if seriousness >= 4 and capability in {"empathy_grief_support", "professional_funeral"}:
            forbidden_terms = (
                SERIOUS_SALES_TERMS
                if capability == "empathy_grief_support"
                else PROFESSIONAL_JOKE_TERMS
            )
            found = [term for term in forbidden_terms if term in target]
            if found:
                errors.append(
                    f"{source} {record_id}: serious response contains sales/joke terms {found}"
                )
        if seriousness >= 4 and capability in {
            "business_humor",
            "crisis_leadership",
            "empathy_grief_support",
            "professional_funeral",
            "worldview_life_death",
        }:
            found = [term for term in SERIOUS_JOKE_TERMS if term in target]
            if found:
                errors.append(
                    f"{source} {record_id}: high-seriousness response contains joke markers {found}"
                )
        if "self_harm" in metadata["risk_flags"]:
            safety_terms = ("急救", "应急", "报警", "热线", "可信任", "陪着你", "药")
            if not any(term in target for term in safety_terms):
                errors.append(
                    f"{source} {record_id}: self-harm response lacks immediate safety action"
                )
        for excerpt in excerpts:
            if excerpt in target:
                errors.append(
                    f"{source} {record_id}: contains protected verbatim excerpt {excerpt!r}"
                )

        normalized_users[normalize(user_text(record))].append(record_id)
        normalized_targets[normalize(target)].append(
            (record_id, metadata["scenario_group"], "imported" if imported else "curated")
        )

    for group, splits in sorted(group_splits.items()):
        if len(splits) != 1:
            errors.append(f"scenario group {group} leaks across splits: {sorted(splits)}")
        if group_kinds[group] == {"imported"}:
            if group_variants[group] != {"0", "1", "2"}:
                errors.append(
                    f"imported group {group} variants {sorted(group_variants[group])} != ['0', '1', '2']"
                )
        elif group_kinds[group] == {"curated"}:
            if group_variants[group] != {"1", "2"}:
                errors.append(
                    f"curated group {group} variants {sorted(group_variants[group])} != ['1', '2']"
                )
        else:
            errors.append(f"scenario group {group} mixes source kinds")

    for normalized_value, ids in normalized_users.items():
        if normalized_value and len(ids) > 1:
            errors.append(f"duplicate normalized joined user text: {ids}")

    allowed_within_imported_group_duplicates = 0
    for normalized_value, entries in normalized_targets.items():
        if not normalized_value or len(entries) == 1:
            continue
        groups = {entry[1] for entry in entries}
        kinds = {entry[2] for entry in entries}
        if len(groups) == 1 and kinds == {"imported"}:
            allowed_within_imported_group_duplicates += 1
            continue
        errors.append(
            "duplicate normalized supervised assistant text across groups: "
            + repr([entry[0] for entry in entries])
        )

    split_counts = Counter(record["metadata"]["split"] for record in records)
    source_counts = Counter(source_kind(record) for record in records)
    source_split_counts = {
        kind: {
            split: sum(
                source_kind(record) == kind and record["metadata"]["split"] == split
                for record in records
            )
            for split in SPLITS
        }
        for kind in EXPECTED_SOURCE_COUNTS
    }
    capability_counts = Counter(record["metadata"]["capability"] for record in records)
    capability_split_counts = {
        capability: {
            split: sum(
                record["metadata"]["capability"] == capability
                and record["metadata"]["split"] == split
                for record in records
            )
            for split in SPLITS
        }
        for capability in sorted(CAPABILITIES)
    }
    supervised_turns = {
        split: sum(
            1
            if record["metadata"].get("assistant_turn_policy") == "final_only"
            else len(assistant_messages(record))
            for record in records
            if record["metadata"]["split"] == split
        )
        for split in SPLITS
    }
    if len(records) != 430:
        errors.append(f"record count {len(records)} != 430")
    if {split: split_counts[split] for split in SPLITS} != EXPECTED_SPLIT_COUNTS:
        errors.append(f"split counts {dict(split_counts)} != {EXPECTED_SPLIT_COUNTS}")
    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        errors.append(f"source counts {dict(source_counts)} != {EXPECTED_SOURCE_COUNTS}")
    if source_split_counts != EXPECTED_SOURCE_SPLIT_COUNTS:
        errors.append(
            f"source split counts {source_split_counts} != {EXPECTED_SOURCE_SPLIT_COUNTS}"
        )
    if capability_split_counts != EXPECTED_CAPABILITY_SPLIT_COUNTS:
        errors.append(
            "capability split counts differ from expected: "
            + repr(capability_split_counts)
        )
    if supervised_turns != EXPECTED_SUPERVISED_TURNS:
        errors.append(
            f"supervised assistant turns {supervised_turns} != {EXPECTED_SUPERVISED_TURNS}"
        )
    if len(imported_source_lines) != 270:
        errors.append(f"imported source line coverage {len(imported_source_lines)} != 270")

    all_path = DATA_ROOT / "all.jsonl"
    all_records: list[dict[str, Any]] = []
    if not all_path.exists():
        errors.append(f"missing compiled dataset: {all_path}")
    else:
        all_records = load_jsonl(all_path)
        clean_splits = [clean(record) for record in records]
        if all_records != sorted(clean_splits, key=lambda record: record["id"]):
            errors.append("all.jsonl does not exactly match sorted union of split files")

    category_files = sorted(CATEGORY_DIR.glob("*.jsonl"))
    category_union = [record for path in category_files for record in load_jsonl(path)]
    if all_records and all_records != sorted(category_union, key=lambda record: record["id"]):
        errors.append("all.jsonl does not exactly match sorted category source union")

    manifest_path = DATA_ROOT / "manifest.json"
    if not manifest_path.exists():
        errors.append(f"missing manifest: {manifest_path}")
    elif all_records:
        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest = build_manifest(all_records, category_files)
        if actual_manifest != expected_manifest:
            errors.append("manifest.json differs from deterministic dataset manifest")

    target_lengths = [len(supervised_assistant_text(record)) for record in records]
    raw_assistant_turns = sum(len(assistant_messages(record)) for record in records)
    multi_turn_count = sum(len(assistant_messages(record)) > 1 for record in records)
    marker_counts = {
        marker: sum(marker in supervised_assistant_text(record) for record in records)
        for marker in STYLE_MARKERS
    }
    if marker_counts["本堂主"] > len(records) * 0.55:
        warnings.append("'本堂主' appears in more than 55% of records")

    report = {
        "status": "pass" if not errors else "fail",
        "records": len(records),
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "source_counts": dict(sorted(source_counts.items())),
        "source_split_counts": source_split_counts,
        "capability_counts": dict(sorted(capability_counts.items())),
        "capability_split_counts": capability_split_counts,
        "message_turns": {
            "assistant_raw": raw_assistant_turns,
            "assistant_supervised": sum(supervised_turns.values()),
            "assistant_supervised_by_split": supervised_turns,
        },
        "multi_turn_records": multi_turn_count,
        "multi_turn_ratio": round(multi_turn_count / len(records), 4) if records else 0,
        "supervised_assistant_characters": {
            "minimum": min(target_lengths) if target_lengths else 0,
            "median": statistics.median(target_lengths) if target_lengths else 0,
            "mean": round(statistics.mean(target_lengths), 2) if target_lengths else 0,
            "maximum": max(target_lengths) if target_lengths else 0,
            "total": sum(target_lengths),
        },
        "duplicate_policy": {
            "allowed_within_imported_group_targets": allowed_within_imported_group_duplicates,
            "cross_group_target_duplicates": sum(
                "duplicate normalized supervised assistant text across groups" in error
                for error in errors
            ),
        },
        "provenance": {
            "imported_records_resolved": len(imported_source_lines),
            "all_samples_sha256": sha256_file(WORKSPACE / "all_samples.jsonl"),
        },
        "schema_validation": schema_validation,
        "style_marker_record_counts": marker_counts,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
