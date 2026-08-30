#!/usr/bin/env python3
"""Shared, dependency-light helpers for the Module C/D experiment code."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


WORKSPACE = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")
CAPABILITIES = (
    "daily_chat",
    "wordplay_poetry",
    "business_humor",
    "relationship_sensitive",
    "professional_funeral",
    "worldview_life_death",
    "empathy_grief_support",
    "crisis_leadership",
    "knowledge_boundary",
)
EXPECTED_GROUPS = {
    "train": {"G01", "G02", "G03", "G04", "G05", "G06", "G09", "G10"},
    "validation": {"G07"},
    "test": {"G08"},
}


class ExperimentError(RuntimeError):
    """Raised when an experiment invariant is violated."""


TOKENIZER_IDENTITY_FIELDS = (
    "chat_template_sha256",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "padding_side",
    "chat_template_kwargs",
)


def canonical_tokenizer_identity(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a tokenizer identity while allowing a deliberately absent BOS.

    Qwen3 has no BOS token, so ``bos_token_id=None`` is an identity value rather
    than missing evidence. Every identity key must still be present, and EOS/PAD
    remain mandatory integer token IDs.
    """
    if not isinstance(snapshot, Mapping):
        raise ExperimentError("Tokenizer identity must be an object")
    missing = [field for field in TOKENIZER_IDENTITY_FIELDS if field not in snapshot]
    if missing:
        raise ExperimentError(
            "Tokenizer identity is missing fields: {}".format(missing)
        )
    identity = {field: snapshot[field] for field in TOKENIZER_IDENTITY_FIELDS}
    template_sha = identity["chat_template_sha256"]
    if not isinstance(template_sha, str) or not template_sha:
        raise ExperimentError("Tokenizer chat-template identity is empty")
    bos_token_id = identity["bos_token_id"]
    if bos_token_id is not None and (
        not isinstance(bos_token_id, int)
        or isinstance(bos_token_id, bool)
        or bos_token_id < 0
    ):
        raise ExperimentError(
            "Tokenizer bos_token_id must be a non-negative integer or null"
        )
    for field in ("eos_token_id", "pad_token_id"):
        value = identity[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ExperimentError(
                "Tokenizer {} must be a non-negative integer".format(field)
            )
    if identity["padding_side"] not in {"left", "right"}:
        raise ExperimentError("Tokenizer padding_side must be left or right")
    if not isinstance(identity["chat_template_kwargs"], Mapping):
        raise ExperimentError("Tokenizer chat_template_kwargs must be an object")
    identity["chat_template_kwargs"] = dict(identity["chat_template_kwargs"])
    return identity


def workspace_path(value: str) -> Path:
    """Resolve a workspace-relative path without requiring it to exist."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else WORKSPACE / path


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExperimentError("Missing JSON file: {}".format(path)) from exc
    except json.JSONDecodeError as exc:
        raise ExperimentError("Invalid JSON in {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise ExperimentError("Expected a JSON object in {}".format(path))
    return value


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ExperimentError("Missing JSONL file: {}".format(path)) from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentError(
                "Invalid JSON in {}:{}: {}".format(path, line_number, exc)
            ) from exc
        if not isinstance(record, dict):
            raise ExperimentError(
                "Expected an object in {}:{}, got {}".format(
                    path, line_number, type(record).__name__
                )
            )
        records.append(record)
    return records


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records
    ]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ExperimentError(
            "SHA-256 mismatch for {}: expected {}, got {}".format(
                path, expected, actual
            )
        )


def validate_source_record(record: Mapping[str, Any], expected_split: str) -> None:
    """Validate the invariants relied on by the turn-expansion code."""
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ExperimentError("Source record has no non-empty id")

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ExperimentError("{} has no metadata object".format(record_id))
    if metadata.get("split") != expected_split:
        raise ExperimentError(
            "{} declares split {!r}, expected {!r}".format(
                record_id, metadata.get("split"), expected_split
            )
        )
    if metadata.get("capability") not in CAPABILITIES:
        raise ExperimentError(
            "{} has unknown capability {!r}".format(
                record_id, metadata.get("capability")
            )
        )

    scenario_group = metadata.get("scenario_group")
    source = metadata.get("source")
    imported_from_all_samples = (
        isinstance(source, Mapping)
        and source.get("dataset") == "all_samples.jsonl"
    )
    if not isinstance(scenario_group, str) or not scenario_group.strip():
        raise ExperimentError("{} has invalid scenario_group".format(record_id))
    if not imported_from_all_samples:
        if "-G" not in scenario_group:
            raise ExperimentError("{} has invalid scenario_group".format(record_id))
        group_suffix = scenario_group.rsplit("-", 1)[-1]
        if group_suffix not in EXPECTED_GROUPS[expected_split]:
            raise ExperimentError(
                "{} group {} cannot appear in {}".format(
                    record_id, scenario_group, expected_split
                )
            )

    assistant_turn_policy = metadata.get("assistant_turn_policy", "all")
    if (
        not isinstance(assistant_turn_policy, str)
        or assistant_turn_policy not in {"all", "final_only"}
    ):
        raise ExperimentError(
            "{} has unsupported assistant_turn_policy {!r}".format(
                record_id, assistant_turn_policy
            )
        )
    if imported_from_all_samples and assistant_turn_policy != "final_only":
        raise ExperimentError(
            "{} imported records must use final_only supervision".format(record_id)
        )
    if not imported_from_all_samples and assistant_turn_policy != "all":
        raise ExperimentError(
            "{} curated records must supervise all assistant turns".format(record_id)
        )

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ExperimentError(
            "{} needs at least system/user/assistant".format(record_id)
        )
    roles: List[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ExperimentError(
                "{} message {} must contain only role/content".format(record_id, index)
            )
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ExperimentError(
                "{} message {} has invalid role {!r}".format(record_id, index, role)
            )
        if not isinstance(content, str) or not content.strip():
            raise ExperimentError(
                "{} message {} has empty content".format(record_id, index)
            )
        roles.append(role)

    expected_roles = ["system"] + [
        "user" if index % 2 else "assistant" for index in range(1, len(messages))
    ]
    if roles != expected_roles or roles[-1] != "assistant":
        raise ExperimentError(
            "{} has invalid role sequence {}".format(record_id, roles)
        )


def expand_assistant_turns(
    record: Mapping[str, Any], expected_split: str
) -> List[Dict[str, Any]]:
    """Create one contextual example per policy-selected assistant target."""
    validate_source_record(record, expected_split)
    messages = record["messages"]
    assistant_total = sum(message["role"] == "assistant" for message in messages)
    assistant_turn_policy = record["metadata"].get("assistant_turn_policy", "all")
    examples: List[Dict[str, Any]] = []
    assistant_index = 0
    for message_index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        assistant_index += 1
        if (
            assistant_turn_policy == "final_only"
            and assistant_index != assistant_total
        ):
            continue
        examples.append(
            {
                "id": "{}::A{}".format(record["id"], assistant_index),
                "source_record_id": record["id"],
                "assistant_turn_index": assistant_index,
                "assistant_turn_count": assistant_total,
                "prompt": deepcopy(messages[:message_index]),
                "completion": [deepcopy(message)],
                "metadata": deepcopy(record["metadata"]),
            }
        )
    return examples


def source_record_counts_by_capability(
    examples: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    """Count unique frozen source records per capability.

    A multi-turn record contributes multiple derived examples but exactly one
    source-record count.  Conflicting capability labels for the same source
    record indicate a corrupted derived snapshot and fail closed.
    """
    capability_by_record: Dict[str, str] = {}
    for example in examples:
        record_id = example.get("source_record_id")
        metadata = example.get("metadata")
        capability = (
            metadata.get("capability") if isinstance(metadata, Mapping) else None
        )
        if not isinstance(record_id, str) or not record_id:
            raise ExperimentError("Derived example has no source_record_id")
        if capability not in CAPABILITIES:
            raise ExperimentError(
                "{} has unknown capability {!r}".format(record_id, capability)
            )
        previous = capability_by_record.setdefault(record_id, capability)
        if previous != capability:
            raise ExperimentError(
                "{} has conflicting capabilities {!r} and {!r}".format(
                    record_id, previous, capability
                )
            )
    return dict(sorted(Counter(capability_by_record.values()).items()))


def summarize_derived(
    source_records: Sequence[Mapping[str, Any]], examples: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    source_capabilities = Counter(
        record["metadata"]["capability"] for record in source_records
    )
    example_capabilities = Counter(
        example["metadata"]["capability"] for example in examples
    )
    multi_turn_records = sum(
        sum(message["role"] == "assistant" for message in record["messages"]) > 1
        for record in source_records
    )
    return {
        "source_records": len(source_records),
        "derived_examples": len(examples),
        "multi_turn_source_records": multi_turn_records,
        "source_records_by_capability": dict(sorted(source_capabilities.items())),
        "derived_examples_by_capability": dict(sorted(example_capabilities.items())),
    }


def git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(WORKSPACE),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def package_versions(names: Sequence[str]) -> Dict[str, Optional[str]]:
    try:
        from importlib import metadata
    except ImportError:  # Python 3.7 compatibility for local lightweight checks.
        try:
            import importlib_metadata as metadata  # type: ignore
        except ImportError:
            return {name: None for name in names}

    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_snapshot() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": git_commit(),
        "packages": package_versions(
            (
                "torch",
                "transformers",
                "trl",
                "peft",
                "datasets",
                "accelerate",
                "bitsandbytes",
                "safetensors",
            )
        ),
    }
