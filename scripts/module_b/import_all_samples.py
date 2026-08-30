#!/usr/bin/env python3
"""Normalize the root all_samples.jsonl corpus into the Module B schema.

The import is deterministic and idempotent: the same source bytes produce the
same normalized JSONL bytes.  Imported records keep a physical-line pointer,
the complete non-message source metadata, the original system prompt, and a
canonical SHA-256 of the complete source record.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE_PATH = WORKSPACE / "all_samples.jsonl"
OUTPUT_PATH = (
    WORKSPACE
    / "data"
    / "module_b_hutao"
    / "categories"
    / "imported_all_samples.jsonl"
)
EXPECTED_SYSTEM = "你是胡桃，以符合角色设定且适合当前情境的方式回答。"
SOURCE_DATASET = "all_samples.jsonl"

CATEGORY_TO_CAPABILITY = {
    "business_professional": "professional_funeral",
    "daily_playful": "daily_chat",
    "grief_support": "empathy_grief_support",
    "knowledge_boundary": "knowledge_boundary",
    "life_death_values": "worldview_life_death",
    "liyue_relationships": "relationship_sensitive",
    "poetry_wordplay": "wordplay_poetry",
    "safety_crisis": "crisis_leadership",
    "traveler_paimon": "relationship_sensitive",
}
SPLIT_MAP = {"train": "train", "valid": "validation", "test": "test"}
SCENE_SPLIT_OVERRIDES = {
    "public_promotion": "test",
    "family_ceremony_dispute": "train",
    "long_queue": "validation",
    "rainy_departure": "train",
    "immediate_self_harm": "validation",
    "revenge_weapon": "train",
    "hearing_ghosts": "test",
    "panic_attack": "train",
    "wishes_vs_family": "test",
    "wish_immortality": "train",
    "qiqi_burial_meme": "train",
    "zhongli_authority": "train",
    "funeral_unlucky": "validation",
    "laomeng_error": "test",
}
SOURCE_KEYS = {
    "id",
    "group_id",
    "source_group_id",
    "variant",
    "split",
    "category",
    "scene",
    "register",
    "relation",
    "capabilities",
    "risk_tags",
    "origin",
    "authenticity",
    "timeline",
    "source_ids",
    "style_anchor_ids",
    "verified_against_game",
    "review_status",
    "messages",
}
LIST_SOURCE_FIELDS = (
    "capabilities",
    "risk_tags",
    "source_ids",
    "style_anchor_ids",
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_source_records(path: Path = SOURCE_PATH) -> list[tuple[int, dict[str, Any]]]:
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


def _validate_string_list(record: dict[str, Any], field: str, line_number: int) -> None:
    value = record.get(field)
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(
            f"{SOURCE_PATH}:{line_number}: {field} must be a unique string list"
        )


def validate_source_record(record: dict[str, Any], line_number: int) -> None:
    if set(record) != SOURCE_KEYS:
        missing = sorted(SOURCE_KEYS - set(record))
        unexpected = sorted(set(record) - SOURCE_KEYS)
        raise ValueError(
            f"{SOURCE_PATH}:{line_number}: source keys "
            f"missing={missing} unexpected={unexpected}"
        )

    record_id = record["id"]
    group_id = record["group_id"]
    variant = record["variant"]
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"{SOURCE_PATH}:{line_number}: id must be non-empty")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError(f"{SOURCE_PATH}:{line_number}: group_id must be non-empty")
    if record["source_group_id"] != group_id:
        raise ValueError(f"{SOURCE_PATH}:{line_number}: source_group_id mismatch")
    if not isinstance(variant, int) or isinstance(variant, bool) or variant not in {0, 1, 2}:
        raise ValueError(f"{SOURCE_PATH}:{line_number}: variant must be 0, 1, or 2")
    if record_id != f"{group_id}-v{variant}":
        raise ValueError(f"{SOURCE_PATH}:{line_number}: id/group/variant mismatch")
    if record["split"] not in SPLIT_MAP:
        raise ValueError(f"{SOURCE_PATH}:{line_number}: unknown split {record['split']!r}")
    if record["category"] not in CATEGORY_TO_CAPABILITY:
        raise ValueError(
            f"{SOURCE_PATH}:{line_number}: unknown category {record['category']!r}"
        )
    if record["authenticity"] != "synthetic_grounded_not_verbatim":
        raise ValueError(f"{SOURCE_PATH}:{line_number}: unsupported authenticity")
    if record["verified_against_game"] != "not_applicable_synthetic":
        raise ValueError(f"{SOURCE_PATH}:{line_number}: unsupported verification status")

    for field in LIST_SOURCE_FIELDS:
        _validate_string_list(record, field, line_number)
    for field in (
        "scene",
        "register",
        "relation",
        "origin",
        "timeline",
        "review_status",
    ):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{SOURCE_PATH}:{line_number}: {field} must be non-empty")

    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError(f"{SOURCE_PATH}:{line_number}: messages are incomplete")
    roles = [message.get("role") if isinstance(message, dict) else None for message in messages]
    expected_roles = ["system"] + [
        "user" if index % 2 else "assistant" for index in range(1, len(messages))
    ]
    if roles != expected_roles or roles[-1] != "assistant":
        raise ValueError(
            f"{SOURCE_PATH}:{line_number}: invalid message roles {roles}"
        )
    for message in messages:
        if (
            set(message) != {"role", "content"}
            or not isinstance(message["content"], str)
            or not message["content"].strip()
        ):
            raise ValueError(f"{SOURCE_PATH}:{line_number}: malformed message")


def infer_seriousness(record: dict[str, Any]) -> int:
    risks = set(record["risk_tags"])
    if risks & {
        "crime",
        "harm_request",
        "medical",
        "mental_health",
        "panic",
        "safety_sensitive",
        "self_harm",
    }:
        return 5
    if risks & {
        "conflict",
        "death_topic",
        "grief",
        "legal",
        "professional_boundary",
        "supernatural_claim",
    }:
        return 4
    if record["register"] == "boundary":
        return 3
    return 2


def infer_humor_types(record: dict[str, Any]) -> list[str]:
    allowlisted = {"doggerel", "style_playful", "wordplay"}
    return [item for item in record["capabilities"] if item in allowlisted]


def infer_death_topic_mode(record: dict[str, Any]) -> str:
    risks = set(record["risk_tags"])
    if "self_harm" in risks:
        return "self_harm_crisis"
    if "grief" in risks:
        return "grief_context"
    if "death_topic" in risks:
        return "explicit_discussion"
    return "none"


def source_provenance(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    """Return a complete, non-lossy pointer and metadata description."""
    return {
        "dataset": SOURCE_DATASET,
        "line": line_number,
        "record_sha256": canonical_sha256(record),
        "record_id": record["id"],
        "group_id": record["group_id"],
        "source_group_id": record["source_group_id"],
        "variant": record["variant"],
        "original_split": record["split"],
        "category": record["category"],
        "scene": record["scene"],
        "register": record["register"],
        "relation": record["relation"],
        "capabilities": list(record["capabilities"]),
        "risk_tags": list(record["risk_tags"]),
        "origin": record["origin"],
        "authenticity": record["authenticity"],
        "timeline": record["timeline"],
        "source_ids": list(record["source_ids"]),
        "style_anchor_ids": list(record["style_anchor_ids"]),
        "verified_against_game": record["verified_against_game"],
        "review_status": record["review_status"],
        "original_system": record["messages"][0]["content"],
    }


def normalize_record(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    validate_source_record(record, line_number)
    messages = [dict(message) for message in record["messages"]]
    messages[0]["content"] = EXPECTED_SYSTEM
    return {
        "id": record["id"],
        "messages": messages,
        "metadata": {
            "split": SCENE_SPLIT_OVERRIDES.get(
                record["scene"], SPLIT_MAP[record["split"]]
            ),
            "capability": CATEGORY_TO_CAPABILITY[record["category"]],
            "scenario_group": f"EXT-{record['group_id']}",
            "register": record["register"],
            "seriousness": infer_seriousness(record),
            "relationship": record["relation"],
            "humor_types": infer_humor_types(record),
            "death_topic_mode": infer_death_topic_mode(record),
            "source_basis": [],
            "construction": "synthetic_persona_grounded",
            "contains_verbatim_game_text": False,
            "risk_flags": list(record["risk_tags"]),
            "assistant_turn_policy": "final_only",
            "source": source_provenance(record, line_number),
        },
    }


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


def main() -> None:
    source_records = load_source_records()
    ids = [record["id"] for _, record in source_records]
    duplicate_ids = sorted({record_id for record_id in ids if ids.count(record_id) > 1})
    if duplicate_ids:
        raise SystemExit(f"Duplicate source ids: {duplicate_ids}")
    normalized = [
        normalize_record(record, line_number)
        for line_number, record in source_records
    ]
    if len(normalized) != 270:
        raise SystemExit(f"Expected 270 source records, found {len(normalized)}")
    base_distribution = Counter(
        (record["category"], SPLIT_MAP[record["split"]])
        for _, record in source_records
    )
    imported_distribution = Counter(
        (record["metadata"]["source"]["category"], record["metadata"]["split"])
        for record in normalized
    )
    if imported_distribution != base_distribution:
        raise SystemExit("Conflict-aware split overrides changed category distributions")
    split_counts = Counter(record["metadata"]["split"] for record in normalized)
    if split_counts != Counter({"train": 216, "validation": 27, "test": 27}):
        raise SystemExit(f"Unexpected imported split counts: {dict(split_counts)}")
    overridden_scenes = {
        record["metadata"]["source"]["scene"]
        for record in normalized
        if record["metadata"]["split"]
        != SPLIT_MAP[record["metadata"]["source"]["original_split"]]
    }
    if overridden_scenes != set(SCENE_SPLIT_OVERRIDES):
        raise SystemExit(
            f"Split override coverage mismatch: {sorted(overridden_scenes)}"
        )
    atomic_write(OUTPUT_PATH, render_jsonl(normalized))
    print(
        json.dumps(
            {
                "source": str(SOURCE_PATH.relative_to(WORKSPACE)),
                "output": str(OUTPUT_PATH.relative_to(WORKSPACE)),
                "records": len(normalized),
                "split_counts": dict(sorted(split_counts.items())),
                "split_overrides": len(overridden_scenes),
                "sha256": hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
