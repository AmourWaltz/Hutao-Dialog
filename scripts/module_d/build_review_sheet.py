#!/usr/bin/env python3
"""Build a seeded, model-blind A/B review CSV and a separate answer key."""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts.module_c.common import (
    ExperimentError,
    canonical_tokenizer_identity,
    load_jsonl,
    validate_source_record,
    workspace_path,
)
from scripts.module_d.rubric import (
    ERROR_TAGS,
    GUARD_DIMENSIONS,
    PERSONA_LAYERS,
    PREFERENCE_DIMENSIONS,
    RUBRIC_SCHEMA_VERSION,
    SCORE_DIMENSIONS,
    public_rubric_payload,
    rubric_sha256,
)


COMPARISON_SCHEMA_VERSION = "module_d.comparison.v1"
KEY_SCHEMA_VERSION = "module_d.blind_key.v2"
GENERATION_MANIFEST_SCHEMA_VERSION = "module_d.generation_manifest.v1"
REGISTERED_SOURCE_SHA256 = {
    "validation": "42562316c1a2fa3f83313154c75b08ff53b6ab5fd19526e315ba3be08cd8af0d",
    "test": "d9e1f88a9ac180e2f08330f01b7093542ee6a8d59665745ad70280e1341ccf2c",
}
REGISTERED_SOURCE_ROOT = "data/module_b_hutao"
REVIEW_FIELDS = (
    "review_id",
    "split",
    "capability",
    "scenario_group",
    "mode",
    "assistant_turn_index",
    "latest_user_message",
    "context_a",
    "response_a",
    "context_b",
    "response_b",
    "surface_style_a_score",
    "surface_style_b_score",
    "knowledge_relationship_a_score",
    "knowledge_relationship_b_score",
    "value_worldview_a_score",
    "value_worldview_b_score",
    "task_completion_a_score",
    "task_completion_b_score",
    "safety_ethics_a_score",
    "safety_ethics_b_score",
    "critical_failure_a",
    "critical_failure_b",
    "error_tags_a",
    "error_tags_b",
    "surface_style_preference",
    "knowledge_relationship_preference",
    "value_worldview_preference",
    "preference",
    "reviewer_id",
    "notes",
)


class BlindReviewError(ValueError):
    """Raised when comparison records cannot form a valid blind review."""


def text_sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_sha256(messages):
    payload = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_comparisons(path):
    comparison_path = Path(path)
    if not comparison_path.is_file():
        raise BlindReviewError("missing comparison JSONL: %s" % comparison_path)
    records = []
    seen_ids = set()
    with comparison_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except ValueError as exc:
                raise BlindReviewError(
                    "%s:%d: invalid JSON: %s" % (comparison_path, line_number, exc)
                )
            _validate_comparison(record, "%s:%d" % (comparison_path, line_number))
            eval_id = record["eval_id"]
            if eval_id in seen_ids:
                raise BlindReviewError("duplicate eval_id %s" % eval_id)
            seen_ids.add(eval_id)
            records.append(record)
    if not records:
        raise BlindReviewError("comparison JSONL is empty")
    return records


def _validate_candidate(candidate, expected_variant, source):
    if not isinstance(candidate, dict):
        raise BlindReviewError("%s: missing %s candidate" % (source, expected_variant))
    if candidate.get("variant") != expected_variant:
        raise BlindReviewError(
            "%s: candidate variant must be %s" % (source, expected_variant)
        )
    if (
        not isinstance(candidate.get("model_label"), str)
        or not candidate["model_label"].strip()
    ):
        raise BlindReviewError("%s: invalid model_label" % source)
    if (
        not isinstance(candidate.get("response"), str)
        or not candidate["response"].strip()
    ):
        raise BlindReviewError("%s: empty candidate response" % source)
    messages = candidate.get("prompt_messages")
    if not isinstance(messages, list) or not messages:
        raise BlindReviewError("%s: missing prompt_messages" % source)
    for message in messages:
        if not isinstance(message, dict) or set(message.keys()) != set(
            ("role", "content")
        ):
            raise BlindReviewError("%s: malformed prompt message" % source)
    if candidate.get("prompt_sha256") != prompt_sha256(messages):
        raise BlindReviewError(
            "%s: %s prompt SHA-256 is invalid" % (source, expected_variant)
        )


def _validate_comparison(record, source="comparison"):
    if not isinstance(record, dict):
        raise BlindReviewError("%s: comparison must be an object" % source)
    if record.get("schema_version") != COMPARISON_SCHEMA_VERSION:
        raise BlindReviewError("%s: unsupported comparison schema" % source)
    for field in (
        "eval_id",
        "record_id",
        "split",
        "capability",
        "scenario_group",
        "mode",
        "assistant_turn_index",
        "latest_user_message",
        "seriousness",
        "risk_flags",
    ):
        if field not in record:
            raise BlindReviewError("%s: missing field %s" % (source, field))
    if not isinstance(record["eval_id"], str) or not record["eval_id"]:
        raise BlindReviewError("%s: invalid eval_id" % source)
    if not isinstance(record["assistant_turn_index"], int) or isinstance(
        record["assistant_turn_index"], bool
    ):
        raise BlindReviewError("%s: invalid assistant_turn_index" % source)
    _validate_candidate(record.get("base"), "base", source)
    _validate_candidate(record.get("lora"), "lora", source)
    if record["base"]["model_label"] == record["lora"]["model_label"]:
        raise BlindReviewError("%s: Base and LoRA model labels must differ" % source)
    prompts_equal = (
        record["base"]["prompt_messages"] == record["lora"]["prompt_messages"]
    )
    if record.get("prompt_equal") is not prompts_equal:
        raise BlindReviewError("%s: prompt_equal flag is inconsistent" % source)
    if record.get("mode") == "controlled_gold_history" and not prompts_equal:
        raise BlindReviewError("%s: controlled comparison is not same-prompt" % source)
    generation = record.get("generation")
    if not isinstance(generation, dict) or not isinstance(
        generation.get("config"), dict
    ):
        raise BlindReviewError("%s: missing generation configuration" % source)
    config = generation["config"]
    if config.get("do_sample") is not False or config.get("num_beams") != 1:
        raise BlindReviewError("%s: comparison is not deterministic greedy" % source)


def _stable_item_seed(root_seed, split, record_id, mode, assistant_turn_index):
    """Mirror the generator's stable, process-independent per-turn seed."""
    material = "%d\0%s\0%s\0%s\0%d" % (
        root_seed,
        split,
        record_id,
        mode,
        assistant_turn_index,
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _exact_json_value(actual, expected):
    """Compare JSON-shaped values without treating bool and int as equal."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_json_value(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_value(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _copy_messages(messages):
    return [
        {"role": message["role"], "content": message["content"]} for message in messages
    ]


def _assistant_turn_policy(record, source):
    """Return the registered assistant-turn evaluation policy for a record."""
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise BlindReviewError("%s: metadata must be an object" % source)
    policy = metadata.get("assistant_turn_policy", "all")
    if policy not in ("all", "final_only"):
        raise BlindReviewError(
            "%s: unsupported assistant_turn_policy %r" % (source, policy)
        )
    return policy


def _manifest_generation_contract(manifest):
    config = manifest.get("generation_config")
    expected_keys = set(("max_new_tokens", "do_sample", "num_beams", "seed"))
    if not isinstance(config, dict) or set(config) != expected_keys:
        raise BlindReviewError(
            "generation manifest has an incomplete generation configuration"
        )
    max_new_tokens = config["max_new_tokens"]
    root_seed = config["seed"]
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens < 1
    ):
        raise BlindReviewError("generation manifest has invalid max_new_tokens")
    if config["do_sample"] is not False:
        raise BlindReviewError("generation manifest is not deterministic greedy")
    if (
        not isinstance(config["num_beams"], int)
        or isinstance(config["num_beams"], bool)
        or config["num_beams"] != 1
    ):
        raise BlindReviewError("generation manifest is not deterministic greedy")
    if not isinstance(root_seed, int) or isinstance(root_seed, bool):
        raise BlindReviewError("generation manifest seed must be an integer")
    if manifest.get("python_hash_seed") != str(root_seed):
        raise BlindReviewError(
            "generation manifest PYTHONHASHSEED differs from its root seed"
        )
    return (
        root_seed,
        {"max_new_tokens": max_new_tokens, "do_sample": False, "num_beams": 1,},
    )


def _load_registered_source_records(splits, source_hashes):
    if set(source_hashes) != set(splits):
        raise BlindReviewError(
            "generation manifest source hashes do not exactly match its splits"
        )
    data_root = workspace_path(REGISTERED_SOURCE_ROOT)
    records = []
    seen_ids = set()
    for split in splits:
        registered_hash = REGISTERED_SOURCE_SHA256.get(split)
        if registered_hash is None:
            raise BlindReviewError(
                "only registered validation/test sources may be reviewed"
            )
        if source_hashes.get(split) != registered_hash:
            raise BlindReviewError(
                "generation manifest source hash is not registered for %s" % split
            )
        source_path = data_root / (split + ".jsonl")
        if not source_path.is_file():
            raise BlindReviewError("missing registered source: %s" % source_path)
        if file_sha256(source_path) != registered_hash:
            raise BlindReviewError(
                "registered source has changed on disk: %s" % source_path
            )
        try:
            split_records = load_jsonl(source_path)
            for record in split_records:
                validate_source_record(record, split)
        except ExperimentError as exc:
            raise BlindReviewError("registered source is invalid: %s" % exc)
        if not split_records:
            raise BlindReviewError("registered source is empty: %s" % source_path)
        for record in split_records:
            record_id = record["id"]
            if record_id in seen_ids:
                raise BlindReviewError(
                    "duplicate record id in registered sources: %s" % record_id
                )
            seen_ids.add(record_id)
            records.append((split, record))
    return records


def _validate_frozen_turn_binding(manifest, comparisons, splits, mode):
    """Bind every comparison row to one real turn in the frozen source."""
    if mode not in ("controlled_gold_history", "rollout"):
        raise BlindReviewError("unsupported generation history mode")
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise BlindReviewError("generation manifest has no source hashes")
    root_seed, row_generation_config = _manifest_generation_contract(manifest)
    source_records = _load_registered_source_records(splits, source_hashes)

    actual_by_id = {}
    for row_number, record in enumerate(comparisons, 1):
        source = "comparison row %d" % row_number
        _validate_comparison(record, source)
        eval_id = record["eval_id"]
        if eval_id in actual_by_id:
            raise BlindReviewError("duplicate eval_id %s" % eval_id)
        actual_by_id[eval_id] = record

    expected_by_id = {}
    for split, source_record in source_records:
        metadata = source_record["metadata"]
        assistant_total = sum(
            message["role"] == "assistant"
            for message in source_record["messages"]
        )
        assistant_turn_policy = _assistant_turn_policy(
            source_record, "%s:%s" % (split, source_record["id"])
        )
        assistant_turn_index = 0
        latest_user = None
        gold_history = []
        for message in source_record["messages"]:
            if message["role"] != "assistant":
                if message["role"] == "user":
                    latest_user = message["content"]
                gold_history.append(dict(message))
                continue
            assistant_turn_index += 1
            evaluate_turn = (
                assistant_turn_policy == "all"
                or assistant_turn_index == assistant_total
            )
            if not evaluate_turn:
                gold_history.append(dict(message))
                continue
            eval_id = "%s:%s:%s:T%02d" % (
                split,
                source_record["id"],
                mode,
                assistant_turn_index,
            )
            if eval_id in expected_by_id:
                raise BlindReviewError(
                    "registered source produced duplicate eval_id %s" % eval_id
                )
            expected_by_id[eval_id] = {
                "eval_id": eval_id,
                "record_id": source_record["id"],
                "split": split,
                "capability": metadata["capability"],
                "scenario_group": metadata["scenario_group"],
                "seriousness": metadata.get("seriousness"),
                "risk_flags": list(metadata.get("risk_flags", [])),
                "mode": mode,
                "assistant_turn_index": assistant_turn_index,
                "latest_user_message": latest_user,
                "gold_response": message["content"],
                "seed": _stable_item_seed(
                    root_seed, split, source_record["id"], mode, assistant_turn_index,
                ),
                "gold_prompt": _copy_messages(gold_history),
            }
            gold_history.append(dict(message))

    if not _exact_json_value(manifest.get("records"), len(source_records)):
        raise BlindReviewError(
            "generation manifest record count differs from frozen source"
        )
    expected_ids = set(expected_by_id)
    actual_ids = set(actual_by_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise BlindReviewError(
            "comparison turns differ from frozen source; missing=%r unexpected=%r"
            % (missing, unexpected)
        )

    frozen_fields = (
        "eval_id",
        "record_id",
        "split",
        "capability",
        "scenario_group",
        "seriousness",
        "risk_flags",
        "mode",
        "assistant_turn_index",
        "latest_user_message",
        "gold_response",
    )
    for eval_id in sorted(expected_ids):
        actual = actual_by_id[eval_id]
        expected = expected_by_id[eval_id]
        for field in frozen_fields:
            if not _exact_json_value(actual.get(field), expected[field]):
                raise BlindReviewError(
                    "%s: frozen field %s differs from registered source"
                    % (eval_id, field)
                )
        generation = actual.get("generation")
        if not isinstance(generation, dict):
            raise BlindReviewError("%s: missing generation evidence" % eval_id)
        if not _exact_json_value(generation.get("seed"), expected["seed"]):
            raise BlindReviewError("%s: per-item seed is inconsistent" % eval_id)
        if not _exact_json_value(generation.get("config"), row_generation_config):
            raise BlindReviewError(
                "%s: generation config differs from manifest" % eval_id
            )
        if mode == "controlled_gold_history":
            for variant in ("base", "lora"):
                if not _exact_json_value(
                    actual[variant]["prompt_messages"], expected["gold_prompt"]
                ):
                    raise BlindReviewError(
                        "%s: %s prompt is not the exact gold history"
                        % (eval_id, variant.capitalize())
                    )
        elif expected["assistant_turn_index"] == 1:
            for variant in ("base", "lora"):
                if not _exact_json_value(
                    actual[variant]["prompt_messages"], expected["gold_prompt"]
                ):
                    raise BlindReviewError(
                        "%s: first-turn %s rollout prompt is not gold history"
                        % (eval_id, variant.capitalize())
                    )

    if mode == "controlled_gold_history":
        return

    # Replay rollout records in source order.  Every frozen system/user turn
    # must be retained, and histories may differ only through the recorded
    # prior response of the same candidate.
    for split, source_record in source_records:
        base_history = []
        lora_history = []
        assistant_total = sum(
            message["role"] == "assistant"
            for message in source_record["messages"]
        )
        assistant_turn_policy = _assistant_turn_policy(
            source_record, "%s:%s" % (split, source_record["id"])
        )
        assistant_turn_index = 0
        for message in source_record["messages"]:
            if message["role"] != "assistant":
                base_history.append(dict(message))
                lora_history.append(dict(message))
                continue
            assistant_turn_index += 1
            evaluate_turn = (
                assistant_turn_policy == "all"
                or assistant_turn_index == assistant_total
            )
            if not evaluate_turn:
                # Skipped bridge turns were never generated, so both rollout
                # histories must retain the registered gold response.
                base_history.append(dict(message))
                lora_history.append(dict(message))
                continue
            eval_id = "%s:%s:%s:T%02d" % (
                split,
                source_record["id"],
                mode,
                assistant_turn_index,
            )
            actual = actual_by_id[eval_id]
            if not _exact_json_value(actual["base"]["prompt_messages"], base_history):
                raise BlindReviewError(
                    "%s: Base prompt is not the required history" % eval_id
                )
            if not _exact_json_value(actual["lora"]["prompt_messages"], lora_history):
                raise BlindReviewError(
                    "%s: LoRA prompt is not the required history" % eval_id
                )
            base_history.append(
                {"role": "assistant", "content": actual["base"]["response"]}
            )
            lora_history.append(
                {"role": "assistant", "content": actual["lora"]["response"]}
            )


def validate_generation_manifest(manifest_path, comparisons_path, comparisons):
    path = Path(manifest_path)
    if not path.is_file():
        raise BlindReviewError("missing generation manifest: %s" % path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except ValueError as exc:
        raise BlindReviewError("invalid generation manifest: %s" % exc)
    if manifest.get("schema_version") != GENERATION_MANIFEST_SCHEMA_VERSION:
        raise BlindReviewError("unsupported generation manifest schema")
    comparison_path = Path(comparisons_path)
    if Path(manifest.get("output", "")).resolve() != comparison_path.resolve():
        raise BlindReviewError(
            "generation manifest points to a different comparison file"
        )
    if manifest.get("output_sha256") != file_sha256(comparison_path):
        raise BlindReviewError("comparison JSONL differs from generation manifest")
    if manifest.get("comparisons") != len(comparisons):
        raise BlindReviewError("generation manifest comparison count is inconsistent")
    splits = sorted(set(record["split"] for record in comparisons))
    modes = sorted(set(record["mode"] for record in comparisons))
    if manifest.get("splits") != splits:
        raise BlindReviewError("generation manifest split list is inconsistent")
    if len(modes) != 1 or manifest.get("mode") != modes[0]:
        raise BlindReviewError("generation manifest mode is inconsistent")
    _validate_frozen_turn_binding(manifest, comparisons, splits, modes[0])
    base_runtime = manifest.get("base_runtime")
    lora_runtime = manifest.get("lora_runtime")
    if not isinstance(base_runtime, dict) or not isinstance(lora_runtime, dict):
        raise BlindReviewError("generation manifest lacks runtime model identities")
    try:
        base_tokenizer = canonical_tokenizer_identity(base_runtime)
        lora_tokenizer = canonical_tokenizer_identity(lora_runtime)
    except ExperimentError as exc:
        raise BlindReviewError(
            "generation manifest tokenizer identity is invalid: %s" % exc
        ) from exc
    if base_tokenizer != lora_tokenizer:
        raise BlindReviewError("Base and LoRA tokenizer identities differ")
    for field in (
        "revision",
        "resolved_commit",
        "dtype_requested",
        "dtype_actual_first_parameter",
        "first_parameter_device",
        "attention_implementation_requested",
        "attention_implementation_resolved",
        "chat_template_sha256",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "padding_side",
        "chat_template_kwargs",
        "cuda_device_count",
        "hf_device_map",
    ):
        if base_runtime.get(field) != lora_runtime.get(field):
            raise BlindReviewError("Base and LoRA runtime %s differs" % field)
    template_kwargs = manifest.get("chat_template_kwargs")
    if (
        not isinstance(template_kwargs, dict)
        or base_runtime.get("chat_template_kwargs") != template_kwargs
        or lora_runtime.get("chat_template_kwargs") != template_kwargs
    ):
        raise BlindReviewError(
            "generation manifest chat-template arguments are inconsistent"
        )
    attention_implementation = manifest.get("attention_implementation")
    if (
        not isinstance(attention_implementation, str)
        or not attention_implementation
        or base_runtime.get("attention_implementation_requested")
        != attention_implementation
        or base_runtime.get("attention_implementation_resolved")
        != attention_implementation
    ):
        raise BlindReviewError(
            "generation manifest attention implementation is inconsistent"
        )
    if (
        base_runtime.get("model_name_or_path") != manifest.get("base_model")
        or lora_runtime.get("model_name_or_path") != manifest.get("base_model")
        or base_runtime.get("revision") != manifest.get("base_revision")
        or lora_runtime.get("revision") != manifest.get("base_revision")
    ):
        raise BlindReviewError("generation manifest Base identity is inconsistent")
    manifest_adapter = manifest.get("lora_adapter")
    if manifest_adapter:
        if (
            Path(lora_runtime.get("adapter_path", "")).resolve()
            != Path(manifest_adapter).resolve()
        ):
            raise BlindReviewError("generation manifest adapter path is inconsistent")
    if splits == ["test"]:
        if not manifest_adapter:
            raise BlindReviewError("frozen test manifest has no LoRA adapter path")
        adapter_provenance = manifest.get("adapter_provenance")
        # Manifests created before coursework mode did not carry this field;
        # preserve their original strict checkpoint-selection interpretation.
        if adapter_provenance is None:
            adapter_provenance = "checkpoint_selection"
        if adapter_provenance == "training_final":
            for field in (
                "selected_adapter_sha256",
                "selected_adapter_config_sha256",
                "experiment_config_sha256",
                "run_manifest",
                "run_manifest_sha256",
            ):
                if not manifest.get(field):
                    raise BlindReviewError(
                        "direct-final test manifest is missing %s" % field
                    )
            if (
                manifest.get("selection_manifest") is not None
                or manifest.get("selection_manifest_sha256") is not None
            ):
                raise BlindReviewError(
                    "direct-final test manifest must not name a selection manifest"
                )
            run_manifest_path = Path(manifest["run_manifest"])
            if (
                not run_manifest_path.is_file()
                or file_sha256(run_manifest_path)
                != manifest.get("run_manifest_sha256")
            ):
                raise BlindReviewError(
                    "direct-final training manifest is missing or changed"
                )
            try:
                from scripts.module_d.generate_comparison import (
                    validate_test_final_adapter,
                    validate_test_generation_contract,
                    validate_test_runtime_identity,
                )

                binding = validate_test_final_adapter(
                    manifest_adapter,
                    manifest.get("base_model"),
                    manifest.get("base_revision"),
                )
                generation_config = manifest.get("generation_config", {})
                validate_test_generation_contract(
                    binding,
                    dtype=base_runtime.get("dtype_requested"),
                    attention_implementation=manifest.get(
                        "attention_implementation"
                    ),
                    seed=generation_config.get("seed"),
                    max_new_tokens=generation_config.get("max_new_tokens"),
                    chat_template_kwargs=template_kwargs,
                    python_hash_seed=manifest.get("python_hash_seed"),
                )
                validate_test_runtime_identity(binding, base_runtime, lora_runtime)
            except Exception as exc:
                raise BlindReviewError(
                    "direct-final adapter evidence could not be revalidated: %s"
                    % exc
                )
            selected = binding["selected"]
            if (
                Path(binding["run_manifest"]).resolve()
                != run_manifest_path.resolve()
                or binding.get("run_manifest_sha256")
                != manifest.get("run_manifest_sha256")
                or binding.get("experiment_config_sha256")
                != manifest.get("experiment_config_sha256")
                or selected.get("adapter_sha256")
                != manifest.get("selected_adapter_sha256")
                or selected.get("adapter_config_sha256")
                != manifest.get("selected_adapter_config_sha256")
                or lora_runtime.get("adapter_sha256")
                != manifest.get("selected_adapter_sha256")
                or lora_runtime.get("adapter_config_sha256")
                != manifest.get("selected_adapter_config_sha256")
                or binding.get("model", {}).get("chat_template_kwargs", {})
                != template_kwargs
            ):
                raise BlindReviewError(
                    "direct-final training or adapter identity is inconsistent"
                )
        elif adapter_provenance == "checkpoint_selection":
            for field in (
                "selection_manifest_sha256",
                "selected_adapter_sha256",
                "selected_adapter_config_sha256",
                "experiment_config_sha256",
            ):
                if not manifest.get(field):
                    raise BlindReviewError(
                        "frozen test generation manifest is missing %s" % field
                    )
            selection_path = Path(manifest.get("selection_manifest", ""))
            if not selection_path.is_file() or file_sha256(
                selection_path
            ) != manifest.get("selection_manifest_sha256"):
                raise BlindReviewError(
                    "test selection manifest is missing or changed"
                )
            try:
                with selection_path.open("r", encoding="utf-8") as handle:
                    selection = json.load(handle)
            except ValueError as exc:
                raise BlindReviewError("invalid test selection manifest: %s" % exc)
            selected = selection.get("selected")
            if (
                selection.get("schema_version")
                != "module_c.checkpoint_selection.v1"
                or selection.get("status") != "selected"
                or selection.get("test_access_authorised_after_this_manifest")
                is not True
                or not isinstance(selected, dict)
                or selected.get("adapter_sha256")
                != manifest.get("selected_adapter_sha256")
                or selected.get("adapter_config_sha256")
                != manifest.get("selected_adapter_config_sha256")
                or lora_runtime.get("adapter_sha256")
                != manifest.get("selected_adapter_sha256")
                or lora_runtime.get("adapter_config_sha256")
                != manifest.get("selected_adapter_config_sha256")
                or selection.get("experiment_config_sha256")
                != manifest.get("experiment_config_sha256")
                or selection.get("model", {}).get("chat_template_kwargs", {})
                != template_kwargs
            ):
                raise BlindReviewError(
                    "test selection or adapter identity is inconsistent"
                )
            try:
                # Re-run Module C's evidence validator here as well as at
                # generation time for users who opt into strict mode.
                from scripts.module_d.generate_comparison import (
                    validate_test_selection,
                )

                revalidated = validate_test_selection(
                    selection_path,
                    manifest_adapter,
                    manifest.get("base_model"),
                    manifest.get("base_revision"),
                )
            except Exception as exc:
                raise BlindReviewError(
                    "test selection evidence could not be revalidated: %s" % exc
                )
            if revalidated != selection:
                raise BlindReviewError(
                    "test selection changed during review validation"
                )
        else:
            raise BlindReviewError(
                "unsupported test adapter provenance: %s" % adapter_provenance
            )
    return manifest


def format_transcript(messages):
    """Render messages into a compact transcript suitable for a CSV cell."""
    role_labels = {
        "system": "[SYSTEM]",
        "user": "[USER]",
        "assistant": "[ASSISTANT]",
    }
    lines = []
    for message in messages:
        role = message.get("role")
        lines.append(
            "%s %s" % (role_labels.get(role, "[%s]" % role), message["content"])
        )
    return "\n".join(lines)


def _blank_scoring_fields(row):
    for dimension in SCORE_DIMENSIONS:
        row[dimension + "_a_score"] = ""
        row[dimension + "_b_score"] = ""
    row["critical_failure_a"] = ""
    row["critical_failure_b"] = ""
    row["error_tags_a"] = ""
    row["error_tags_b"] = ""
    for dimension in PREFERENCE_DIMENSIONS:
        row[dimension + "_preference"] = ""
    row["preference"] = ""
    row["reviewer_id"] = ""
    row["notes"] = ""


def build_blind_review(comparisons, seed=42):
    """Return deterministic CSV rows and a separate Base/LoRA answer key."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise BlindReviewError("seed must be an integer")
    ordered = sorted(comparisons, key=lambda item: item.get("eval_id", ""))
    seen_ids = set()
    for record in ordered:
        _validate_comparison(record)
        if record["eval_id"] in seen_ids:
            raise BlindReviewError("duplicate eval_id %s" % record["eval_id"])
        seen_ids.add(record["eval_id"])
    splits = set(record["split"] for record in ordered)
    modes = set(record["mode"] for record in ordered)
    if len(splits) != 1 or len(modes) != 1:
        raise BlindReviewError(
            "one blind sheet must contain exactly one split and mode"
        )

    rng = random.Random(seed)
    rng.shuffle(ordered)
    rows = []
    key_rows = {}
    for row_number, comparison in enumerate(ordered, 1):
        review_id = "R%04d" % row_number
        swap = bool(rng.getrandbits(1))
        if swap:
            candidate_a = comparison["lora"]
            candidate_b = comparison["base"]
        else:
            candidate_a = comparison["base"]
            candidate_b = comparison["lora"]

        row = {
            "review_id": review_id,
            "split": comparison["split"],
            "capability": comparison["capability"],
            "scenario_group": comparison["scenario_group"],
            "mode": comparison["mode"],
            "assistant_turn_index": str(comparison["assistant_turn_index"]),
            "latest_user_message": comparison["latest_user_message"],
            "context_a": format_transcript(candidate_a["prompt_messages"]),
            "response_a": candidate_a["response"],
            "context_b": format_transcript(candidate_b["prompt_messages"]),
            "response_b": candidate_b["response"],
        }
        _blank_scoring_fields(row)
        rows.append(row)
        context_a = row["context_a"]
        context_b = row["context_b"]
        key_rows[review_id] = {
            "eval_id": comparison["eval_id"],
            "record_id": comparison["record_id"],
            "split": comparison["split"],
            "capability": comparison["capability"],
            "scenario_group": comparison["scenario_group"],
            "mode": comparison["mode"],
            "assistant_turn_index": comparison["assistant_turn_index"],
            "seriousness": comparison["seriousness"],
            "risk_flags": list(comparison["risk_flags"]),
            "latest_user_message_sha256": text_sha256(
                comparison["latest_user_message"]
            ),
            "a": {
                "variant": candidate_a["variant"],
                "model_label": candidate_a["model_label"],
                "context_sha256": text_sha256(context_a),
                "response_sha256": text_sha256(candidate_a["response"]),
            },
            "b": {
                "variant": candidate_b["variant"],
                "model_label": candidate_b["model_label"],
                "context_sha256": text_sha256(context_b),
                "response_sha256": text_sha256(candidate_b["response"]),
            },
        }

    key = {
        "schema_version": KEY_SCHEMA_VERSION,
        "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_sha256": rubric_sha256(),
        "rubric": public_rubric_payload(),
        "seed": seed,
        "review_rows": len(rows),
        "persona_layers": list(PERSONA_LAYERS),
        "guard_dimensions": list(GUARD_DIMENSIONS),
        "score_dimensions": list(SCORE_DIMENSIONS),
        "preference_dimensions": list(PREFERENCE_DIMENSIONS),
        "allowed_error_tags": list(ERROR_TAGS),
        "rows": key_rows,
    }
    return rows, key


def write_review_csv(rows, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_blind_key(key, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(key, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def write_rubric_json(output_path):
    """Write reviewer-facing criteria without any blind-key/model identity."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            public_rubric_payload(),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", required=True, help="comparison JSONL")
    parser.add_argument(
        "--generation-manifest",
        required=True,
        help="manifest emitted beside the comparison JSONL",
    )
    parser.add_argument("--review-csv", required=True)
    parser.add_argument(
        "--key-json",
        required=True,
        help="secret A/B key; do not send this file to reviewers",
    )
    parser.add_argument(
        "--rubric-json",
        help="optional reviewer-facing rubric JSON without model identities",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    comparisons = load_comparisons(args.comparisons)
    validate_generation_manifest(
        args.generation_manifest, args.comparisons, comparisons
    )
    rows, key = build_blind_review(comparisons, seed=args.seed)
    key.update(
        {
            "comparison_file": str(Path(args.comparisons).resolve()),
            "comparison_file_sha256": file_sha256(args.comparisons),
            "generation_manifest": str(Path(args.generation_manifest).resolve()),
            "generation_manifest_sha256": file_sha256(args.generation_manifest),
        }
    )
    output_paths = [Path(args.review_csv).resolve(), Path(args.key_json).resolve()]
    if args.rubric_json:
        output_paths.append(Path(args.rubric_json).resolve())
    input_paths = [
        Path(args.comparisons).resolve(),
        Path(args.generation_manifest).resolve(),
    ]
    if len(set(output_paths)) != len(output_paths) or any(
        output_path in input_paths for output_path in output_paths
    ):
        raise BlindReviewError(
            "review/key/rubric outputs must be distinct from each other and inputs"
        )
    write_review_csv(rows, args.review_csv)
    write_blind_key(key, args.key_json)
    if args.rubric_json:
        write_rubric_json(args.rubric_json)
    print(
        json.dumps(
            {
                "status": "ok",
                "review_rows": len(rows),
                "review_csv": str(Path(args.review_csv)),
                "key_json": str(Path(args.key_json)),
                "rubric_json": (
                    str(Path(args.rubric_json)) if args.rubric_json else None
                ),
                "seed": args.seed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
