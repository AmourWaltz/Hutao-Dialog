#!/usr/bin/env python3
"""Generate deterministic Base/LoRA comparisons from Module B evaluation data.

The reusable functions in this module use only the Python standard library.
PyTorch, Transformers, and PEFT are imported lazily by
``TransformersTextGenerator`` when real model generation is requested.  Tests
can therefore inject a small callable without installing any ML dependencies.

Two history policies are supported:

``controlled_gold_history``
    Base and LoRA receive identical prompts.  Earlier assistant turns come from
    the gold Module B conversation, isolating the response at the current turn.

``rollout``
    Each model receives its own earlier generated assistant turns.  User and
    system messages remain identical, while later model contexts may diverge.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path

from scripts.module_c.common import ExperimentError, canonical_tokenizer_identity


SCHEMA_VERSION = "module_d.comparison.v1"
VALID_SPLITS = ("validation", "test")
VALID_MODES = ("controlled_gold_history", "rollout")
MODE_ALIASES = {
    "controlled-gold-history": "controlled_gold_history",
    "controlled_gold_history": "controlled_gold_history",
    "rollout": "rollout",
}
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "module_b_hutao"
WORKSPACE = Path(__file__).resolve().parents[2]
REGISTERED_SOURCE_SHA256 = {
    "validation": "42562316c1a2fa3f83313154c75b08ff53b6ab5fd19526e315ba3be08cd8af0d",
    "test": "d9e1f88a9ac180e2f08330f01b7093542ee6a8d59665745ad70280e1341ccf2c",
}
DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 192,
    "do_sample": False,
    "num_beams": 1,
}
SELECTION_SCHEMA_VERSION = "module_c.checkpoint_selection.v1"


class EvaluationDataError(ValueError):
    """Raised when evaluation JSONL does not satisfy the expected contract."""


def _copy_messages(messages):
    return [
        {"role": message["role"], "content": message["content"]} for message in messages
    ]


def _validate_messages(messages, source):
    if not isinstance(messages, list) or len(messages) < 3:
        raise EvaluationDataError(
            "%s: messages must contain at least system, user, assistant" % source
        )
    expected_roles = ["system"]
    for index in range(1, len(messages)):
        expected_roles.append("user" if index % 2 else "assistant")
    roles = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise EvaluationDataError(
                "%s: message %d is not an object" % (source, index)
            )
        if set(message.keys()) != set(("role", "content")):
            raise EvaluationDataError(
                "%s: message %d must contain only role/content" % (source, index)
            )
        if (
            not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise EvaluationDataError(
                "%s: message %d has empty content" % (source, index)
            )
        roles.append(message.get("role"))
    if roles != expected_roles or roles[-1] != "assistant":
        raise EvaluationDataError("%s: invalid role order %r" % (source, roles))


def load_evaluation_records(
    data_root=DEFAULT_DATA_ROOT,
    splits=("test",),
    expected_sha256=REGISTERED_SOURCE_SHA256,
):
    """Load raw records only from Module B ``validation`` and/or ``test``.

    Args:
        data_root: Directory containing ``validation.jsonl`` and ``test.jsonl``.
        splits: A split string or an iterable of split strings.

    Returns:
        A list of dictionaries.  The original ``messages`` content is retained,
        and ``_source_split``/``_source_line`` are added for traceability.
    """
    if isinstance(splits, str):
        splits = (splits,)
    splits = tuple(splits)
    if not splits:
        raise EvaluationDataError("at least one evaluation split is required")
    invalid = [split for split in splits if split not in VALID_SPLITS]
    if invalid:
        raise EvaluationDataError(
            "only validation/test may be evaluated; invalid splits: %r" % invalid
        )

    root = Path(data_root)
    records = []
    seen_ids = set()
    for split in splits:
        path = root / (split + ".jsonl")
        if not path.is_file():
            raise EvaluationDataError("missing evaluation split: %s" % path)
        raw_bytes = path.read_bytes()
        source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if expected_sha256 is not None:
            expected = expected_sha256.get(split)
            if expected is None:
                raise EvaluationDataError(
                    "no registered SHA-256 for evaluation split %s" % split
                )
            if source_sha256 != expected:
                raise EvaluationDataError(
                    "%s: SHA-256 is %s, expected %s" % (path, source_sha256, expected)
                )
        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvaluationDataError("%s: invalid UTF-8: %s" % (path, exc))
        for line_number, raw_line in enumerate(raw_text.splitlines(), 1):
            if not raw_line.strip():
                continue
            source = "%s:%d" % (path, line_number)
            try:
                record = json.loads(raw_line)
            except ValueError as exc:
                raise EvaluationDataError("%s: invalid JSON: %s" % (source, exc))
            if not isinstance(record, dict):
                raise EvaluationDataError("%s: record must be an object" % source)
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise EvaluationDataError("%s: missing id" % source)
            if record_id in seen_ids:
                raise EvaluationDataError("%s: duplicate id %s" % (source, record_id))
            seen_ids.add(record_id)
            metadata = record.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("split") != split:
                raise EvaluationDataError(
                    "%s: metadata.split must equal %s" % (source, split)
                )
            _validate_messages(record.get("messages"), source)
            copied = copy.deepcopy(record)
            copied["_source_split"] = split
            copied["_source_line"] = line_number
            copied["_source_sha256"] = source_sha256
            records.append(copied)
    return records


def normalize_mode(mode):
    try:
        return MODE_ALIASES[mode]
    except KeyError:
        raise ValueError("unsupported history mode: %r" % mode)


def normalized_generation_config(generation_config=None):
    """Return a validated greedy-generation configuration.

    Sampling is deliberately rejected so Base and LoRA use a deterministic
    decoding policy.  A stable per-item seed is still passed to generators and
    recorded to cover any backend operations that consult an RNG.
    """
    config = dict(DEFAULT_GENERATION_CONFIG)
    if generation_config:
        config.update(generation_config)
    if config.get("do_sample") is not False:
        raise ValueError("deterministic comparison requires do_sample=False")
    if config.get("num_beams", 1) != 1:
        raise ValueError("deterministic comparison requires num_beams=1")
    max_new_tokens = config.get("max_new_tokens")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise ValueError("max_new_tokens must be a positive integer")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    return config


def prompt_sha256(messages):
    payload = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_test_selection(
    selection_manifest, adapter_path, base_model, base_revision
):
    """Require a frozen validation decision before the CLI may open test."""
    path = Path(selection_manifest)
    if not path.is_file():
        raise EvaluationDataError("missing checkpoint selection manifest: %s" % path)
    with path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise EvaluationDataError("unsupported checkpoint selection schema")
    if selection.get("status") != "selected" or not isinstance(
        selection.get("selected"), dict
    ):
        raise EvaluationDataError("selection manifest has no selected checkpoint")
    if selection.get("test_access_authorised_after_this_manifest") is not True:
        raise EvaluationDataError("selection manifest does not authorise test access")
    expected_steps = selection.get("expected_checkpoint_steps")
    candidates = selection.get("candidates")
    if (
        not isinstance(expected_steps, list)
        or not expected_steps
        or any(
            isinstance(step, bool) or not isinstance(step, int) or step < 1
            for step in expected_steps
        )
        or len(set(expected_steps)) != len(expected_steps)
        or sorted(expected_steps) != expected_steps
        or not isinstance(candidates, list)
        or len(candidates) != len(expected_steps)
    ):
        raise EvaluationDataError("selection manifest has an incomplete candidate set")
    candidate_steps = []
    candidate_adapter_paths = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise EvaluationDataError("selection candidate is malformed")
        checkpoint_step = candidate.get("checkpoint_step")
        if (
            isinstance(checkpoint_step, bool)
            or not isinstance(checkpoint_step, int)
            or checkpoint_step < 1
        ):
            raise EvaluationDataError("selection candidate step is invalid")
        candidate_steps.append(checkpoint_step)
        if (
            candidate.get("integrity_pass") is not True
            or candidate.get("integrity_failures") != []
        ):
            raise EvaluationDataError("selection contains an unverified candidate")
        if candidate.get("model") != selection.get("model"):
            raise EvaluationDataError("selection candidate Base identity differs")
        candidate_adapter_path = Path(candidate.get("adapter_path", "")).resolve()
        expected_checkpoint_name = "checkpoint-{}".format(checkpoint_step)
        if candidate_adapter_path.name != expected_checkpoint_name:
            raise EvaluationDataError("selection candidate checkpoint path differs")
        candidate_adapter_paths.append(candidate_adapter_path)
        candidate_adapter_file = candidate_adapter_path / "adapter_model.safetensors"
        candidate_adapter_config = candidate_adapter_path / "adapter_config.json"
        if (
            not candidate_adapter_file.is_file()
            or file_sha256(candidate_adapter_file) != candidate.get("adapter_sha256")
            or not candidate_adapter_config.is_file()
            or file_sha256(candidate_adapter_config)
            != candidate.get("adapter_config_sha256")
        ):
            raise EvaluationDataError(
                "selection candidate adapter artifact is missing or changed"
            )
        for path_field, hash_field in (
            ("metric_file", "metric_file_sha256"),
            ("safety_review_file", "safety_review_file_sha256"),
        ):
            evidence_path = Path(candidate.get(path_field, ""))
            if not evidence_path.is_file() or file_sha256(
                evidence_path
            ) != candidate.get(hash_field):
                raise EvaluationDataError(
                    "selection candidate evidence is missing or changed"
                )
    if sorted(candidate_steps) != expected_steps:
        raise EvaluationDataError(
            "selection candidates differ from expected checkpoints"
        )
    if len(set(candidate_adapter_paths)) != len(candidate_adapter_paths):
        raise EvaluationDataError("selection candidate adapter paths are not unique")

    run_manifest_path = Path(selection.get("run_manifest", ""))
    if not run_manifest_path.is_file() or file_sha256(
        run_manifest_path
    ) != selection.get("run_manifest_sha256"):
        raise EvaluationDataError("canonical training manifest is missing or changed")
    with run_manifest_path.open("r", encoding="utf-8") as handle:
        run_manifest = json.load(handle)
    if (
        run_manifest.get("mode") != "main"
        or run_manifest.get("status") != "complete"
        or run_manifest.get("config_sha256")
        != selection.get("experiment_config_sha256")
        or run_manifest.get("config", {}).get("model") != selection.get("model")
    ):
        raise EvaluationDataError("canonical training manifest is inconsistent")

    raw_config_path = run_manifest.get("config_path")
    if not isinstance(raw_config_path, str) or not raw_config_path:
        raise EvaluationDataError("canonical training manifest has no config path")
    config_path = Path(raw_config_path)
    if not config_path.is_absolute():
        config_path = WORKSPACE / config_path
    if not config_path.is_file() or file_sha256(config_path) != selection.get(
        "experiment_config_sha256"
    ):
        raise EvaluationDataError("canonical experiment config is missing or changed")
    try:
        # Import lazily to avoid a module-import cycle. Recomputing from the
        # frozen metrics and human reviews prevents self-declared selection
        # booleans from opening the held-out test split.
        from scripts.module_c.select_checkpoint import select

        recomputed = select(
            config_path,
            [
                (Path(candidate["metric_file"]), Path(candidate["safety_review_file"]))
                for candidate in candidates
            ],
            output_path=None,
        )
    except Exception as exc:
        raise EvaluationDataError(
            "checkpoint selection evidence could not be revalidated: %s" % exc
        )
    if recomputed != selection:
        raise EvaluationDataError("selection manifest differs from recomputed result")

    selected = selection["selected"]
    if selected not in candidates:
        raise EvaluationDataError("selected checkpoint is absent from candidate set")
    if selected.get("safety_pass") is not True or selected.get("safety_failures") != []:
        raise EvaluationDataError("selected checkpoint did not pass the safety gate")
    selected_model = selected.get("model")
    if selected_model != selection.get("model"):
        raise EvaluationDataError(
            "selected checkpoint model identity differs from selection manifest"
        )
    if not isinstance(selected_model, dict):
        raise EvaluationDataError("selection manifest does not freeze the Base model")
    if selected_model.get("name") != base_model:
        raise EvaluationDataError("--base-model differs from selected training Base")
    if selected_model.get("revision") != base_revision:
        raise EvaluationDataError(
            "--base-revision differs from selected training revision"
        )
    expected_path = Path(selected.get("adapter_path", "")).expanduser().resolve()
    actual_path = Path(adapter_path).expanduser().resolve()
    if actual_path != expected_path:
        raise EvaluationDataError(
            "--lora-adapter differs from selected checkpoint: %s != %s"
            % (actual_path, expected_path)
        )
    adapter_file = actual_path / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise EvaluationDataError(
            "selected adapter weights are missing: %s" % adapter_file
        )
    actual_sha = file_sha256(adapter_file)
    if actual_sha != selected.get("adapter_sha256"):
        raise EvaluationDataError("selected adapter SHA-256 does not match weights")
    adapter_config_file = actual_path / "adapter_config.json"
    if not adapter_config_file.is_file():
        raise EvaluationDataError(
            "selected adapter configuration is missing: %s" % adapter_config_file
        )
    if file_sha256(adapter_config_file) != selected.get("adapter_config_sha256"):
        raise EvaluationDataError("selected adapter_config.json SHA-256 does not match")
    with adapter_config_file.open("r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    if (
        adapter_config.get("base_model_name_or_path") != base_model
        or adapter_config.get("revision") != base_revision
    ):
        raise EvaluationDataError("selected adapter names a different Base revision")
    return selection


def validate_test_final_adapter(adapter_path, base_model, base_revision):
    """Bind a coursework test run directly to training's ``adapter-final``.

    This deliberately skips checkpoint selection, validation metrics, and the
    Module C safety-review gate.  It still proves that the requested adapter is
    the final artifact written by a completed main training run and that it was
    trained against the requested immutable Base revision.
    """
    actual_path = Path(adapter_path).expanduser().resolve()
    if actual_path.name != "adapter-final":
        raise EvaluationDataError(
            "--use-final-adapter requires an adapter-final directory: %s"
            % actual_path
        )
    if not actual_path.is_dir():
        raise EvaluationDataError("final adapter directory is missing: %s" % actual_path)

    adapter_file = actual_path / "adapter_model.safetensors"
    adapter_config_file = actual_path / "adapter_config.json"
    if not adapter_file.is_file() or adapter_file.stat().st_size < 1:
        raise EvaluationDataError("final adapter weights are missing: %s" % adapter_file)
    if not adapter_config_file.is_file() or adapter_config_file.stat().st_size < 1:
        raise EvaluationDataError(
            "final adapter configuration is missing: %s" % adapter_config_file
        )
    try:
        with adapter_config_file.open("r", encoding="utf-8") as handle:
            adapter_config = json.load(handle)
    except (OSError, ValueError) as exc:
        raise EvaluationDataError(
            "invalid final adapter configuration: %s" % exc
        )
    if not isinstance(adapter_config, dict):
        raise EvaluationDataError("final adapter configuration must be an object")
    if (
        adapter_config.get("base_model_name_or_path") != base_model
        or adapter_config.get("revision") != base_revision
    ):
        raise EvaluationDataError(
            "final adapter names a different Base model or revision"
        )

    run_manifest_path = actual_path.parent / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise EvaluationDataError(
            "completed training manifest is missing: %s" % run_manifest_path
        )
    try:
        with run_manifest_path.open("r", encoding="utf-8") as handle:
            run_manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise EvaluationDataError("invalid training manifest: %s" % exc)
    if not isinstance(run_manifest, dict):
        raise EvaluationDataError("training manifest must be an object")
    if run_manifest.get("mode") != "main" or run_manifest.get("status") != "complete":
        raise EvaluationDataError(
            "--use-final-adapter requires a completed main training run"
        )

    registered_config = run_manifest.get("config")
    registered_model = (
        registered_config.get("model")
        if isinstance(registered_config, dict)
        else None
    )
    if not isinstance(registered_model, dict):
        raise EvaluationDataError("training manifest has no registered Base model")
    if (
        registered_model.get("name") != base_model
        or registered_model.get("revision") != base_revision
    ):
        raise EvaluationDataError(
            "training manifest names a different Base model or revision"
        )

    recorded_adapter_path = run_manifest.get("adapter_path")
    if not isinstance(recorded_adapter_path, str) or not recorded_adapter_path:
        raise EvaluationDataError("training manifest has no final adapter path")
    recorded_path = Path(recorded_adapter_path).expanduser()
    if not recorded_path.is_absolute():
        recorded_path = WORKSPACE / recorded_path
    if recorded_path.resolve() != actual_path:
        raise EvaluationDataError(
            "--lora-adapter differs from training's final adapter: %s != %s"
            % (actual_path, recorded_path.resolve())
        )

    adapter_sha256 = file_sha256(adapter_file)
    if run_manifest.get("adapter_model_sha256") != adapter_sha256:
        raise EvaluationDataError(
            "final adapter weights changed after training completed"
        )
    experiment_config_sha256 = run_manifest.get("config_sha256")
    if (
        not isinstance(experiment_config_sha256, str)
        or not experiment_config_sha256
    ):
        raise EvaluationDataError("training manifest has no experiment config SHA-256")
    try:
        canonical_tokenizer_identity(run_manifest.get("tokenizer"))
    except ExperimentError as exc:
        raise EvaluationDataError(
            "training manifest tokenizer identity is invalid: %s" % exc
        )

    return {
        "adapter_provenance": "training_final",
        "model": registered_model,
        "run_manifest": str(run_manifest_path.resolve()),
        "run_manifest_sha256": file_sha256(run_manifest_path),
        "experiment_config_sha256": experiment_config_sha256,
        "selected": {
            "adapter_path": str(actual_path),
            "adapter_sha256": adapter_sha256,
            "adapter_config_sha256": file_sha256(adapter_config_file),
        },
    }


def validate_test_runtime_identity(selection, base_audit, lora_audit):
    """Bind test inference to the model/tokenizer identity recorded at training."""
    run_manifest_path = Path(selection["run_manifest"])
    with run_manifest_path.open("r", encoding="utf-8") as handle:
        run_manifest = json.load(handle)
    tokenizer = run_manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise EvaluationDataError(
            "canonical training manifest lacks tokenizer identity"
        )
    try:
        tokenizer_identity = canonical_tokenizer_identity(tokenizer)
    except ExperimentError as exc:
        raise EvaluationDataError(
            "canonical training tokenizer identity is invalid: %s" % exc
        ) from exc
    model = selection["model"]
    attention_implementation = model.get("attention_implementation")
    expected_runtime = {
        "model_name_or_path": model["name"],
        "revision": model["revision"],
        "resolved_commit": model["revision"],
        "dtype_requested": model["dtype"],
        "dtype_actual_first_parameter": "torch.%s" % model["dtype"],
        "first_parameter_device": "cuda:0",
        "attention_implementation_requested": attention_implementation,
        "attention_implementation_resolved": attention_implementation,
        "cuda_device_count": run_manifest.get("config", {})
        .get("runtime", {})
        .get("visible_cuda_devices"),
        **tokenizer_identity,
    }
    if any(
        value is None
        for field, value in expected_runtime.items()
        if field != "bos_token_id"
    ):
        raise EvaluationDataError("canonical runtime identity is incomplete")
    for name, audit in (("Base", base_audit), ("LoRA", lora_audit)):
        try:
            audit_tokenizer = canonical_tokenizer_identity(audit)
        except ExperimentError as exc:
            raise EvaluationDataError(
                "%s runtime tokenizer identity is invalid: %s" % (name, exc)
            ) from exc
        if audit_tokenizer != tokenizer_identity or any(
            field not in audit or audit[field] != value
            for field, value in expected_runtime.items()
        ):
            raise EvaluationDataError(
                "%s test runtime differs from canonical training identity" % name
            )
    selected = selection["selected"]
    if lora_audit.get("adapter_sha256") != selected.get(
        "adapter_sha256"
    ) or lora_audit.get("adapter_config_sha256") != selected.get(
        "adapter_config_sha256"
    ):
        raise EvaluationDataError("LoRA runtime used different adapter artifacts")


def validate_test_generation_contract(
    selection,
    dtype,
    attention_implementation,
    seed,
    max_new_tokens,
    chat_template_kwargs,
    python_hash_seed,
):
    """Bind every reportable test-generation option to the training config."""
    run_manifest_path = Path(selection["run_manifest"])
    with run_manifest_path.open("r", encoding="utf-8") as handle:
        run_manifest = json.load(handle)
    registered_config = run_manifest.get("config")
    if not isinstance(registered_config, dict):
        raise EvaluationDataError("canonical run has no registered config")
    if registered_config.get("model") != selection.get("model"):
        raise EvaluationDataError("selection and training model configs differ")
    registered_generation = registered_config.get("generation")
    if not isinstance(registered_generation, dict) or set(registered_generation) != {
        "do_sample",
        "num_beams",
        "max_new_tokens",
        "seed",
    }:
        raise EvaluationDataError("canonical generation config is incomplete")
    if (
        registered_generation.get("do_sample") is not False
        or registered_generation.get("num_beams") != 1
    ):
        raise EvaluationDataError("canonical test generation must use greedy decoding")
    if dtype != selection["model"].get("dtype"):
        raise EvaluationDataError("test dtype differs from selected training dtype")
    if attention_implementation != selection["model"].get("attention_implementation"):
        raise EvaluationDataError("test attention implementation differs from training")
    if seed != registered_generation.get("seed"):
        raise EvaluationDataError("test seed differs from the registered seed")
    if max_new_tokens != registered_generation.get("max_new_tokens"):
        raise EvaluationDataError(
            "test max_new_tokens differs from the registered limit"
        )
    registered_template_kwargs = selection["model"].get(
        "chat_template_kwargs", {}
    )
    if chat_template_kwargs != registered_template_kwargs:
        raise EvaluationDataError(
            "test chat-template arguments differ from canonical training"
        )
    if python_hash_seed != str(seed):
        raise EvaluationDataError(
            "test generation requires PYTHONHASHSEED=%d before Python starts" % seed
        )
    return {
        "dtype": dtype,
        "attention_implementation": attention_implementation,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "python_hash_seed": python_hash_seed,
        "chat_template_kwargs": dict(chat_template_kwargs),
    }


def stable_item_seed(root_seed, split, record_id, mode, assistant_turn_index):
    material = "%d\0%s\0%s\0%s\0%d" % (
        root_seed,
        split,
        record_id,
        mode,
        assistant_turn_index,
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _generator_label(generator, fallback):
    label = getattr(generator, "model_label", fallback)
    if not isinstance(label, str) or not label.strip():
        raise ValueError("generator model_label must be a non-empty string")
    return label


def _invoke_generator(generator, messages, generation_config, seed):
    prompt = _copy_messages(messages)
    if hasattr(generator, "generate"):
        response = generator.generate(prompt, dict(generation_config), seed)
    elif callable(generator):
        response = generator(prompt, dict(generation_config), seed)
    else:
        raise TypeError("generator must be callable or expose generate()")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("generator returned an empty or non-string response")
    return response.strip()


def _assistant_turn_policy(record, source):
    """Return the registered assistant-turn evaluation policy for a record."""
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise EvaluationDataError("%s: metadata must be an object" % source)
    policy = metadata.get("assistant_turn_policy", "all")
    if policy not in ("all", "final_only"):
        raise EvaluationDataError(
            "%s: unsupported assistant_turn_policy %r" % (source, policy)
        )
    return policy


def _make_comparison_item(
    record,
    mode,
    assistant_turn_index,
    latest_user,
    gold_response,
    base_prompt,
    lora_prompt,
    base_response,
    lora_response,
    base_label,
    lora_label,
    generation_config,
    item_seed,
):
    split = record["_source_split"]
    record_id = record["id"]
    metadata = record["metadata"]
    base_digest = prompt_sha256(base_prompt)
    lora_digest = prompt_sha256(lora_prompt)
    eval_id = "%s:%s:%s:T%02d" % (split, record_id, mode, assistant_turn_index,)
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_id": eval_id,
        "record_id": record_id,
        "split": split,
        "capability": metadata.get("capability", "unknown"),
        "scenario_group": metadata.get("scenario_group", "unknown"),
        "seriousness": metadata.get("seriousness"),
        "risk_flags": list(metadata.get("risk_flags", [])),
        "mode": mode,
        "assistant_turn_index": assistant_turn_index,
        "latest_user_message": latest_user,
        "gold_response": gold_response,
        "prompt_equal": base_digest == lora_digest,
        "generation": {"seed": item_seed, "config": dict(generation_config),},
        "base": {
            "variant": "base",
            "model_label": base_label,
            "prompt_messages": _copy_messages(base_prompt),
            "prompt_sha256": base_digest,
            "response": base_response,
        },
        "lora": {
            "variant": "lora",
            "model_label": lora_label,
            "prompt_messages": _copy_messages(lora_prompt),
            "prompt_sha256": lora_digest,
            "response": lora_response,
        },
    }


def generate_comparisons(
    records,
    base_generator,
    lora_generator,
    mode="controlled_gold_history",
    generation_config=None,
    seed=42,
):
    """Generate Base/LoRA comparisons for each record's registered target turns.

    Records default to evaluating every assistant turn.  Imported records may
    set ``metadata.assistant_turn_policy`` to ``final_only``; their earlier
    assistant messages are then retained as frozen gold history, but are not
    generated or scored as standalone turns.
    """
    mode = normalize_mode(mode)
    config = normalized_generation_config(generation_config)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    base_label = _generator_label(base_generator, "base")
    lora_label = _generator_label(lora_generator, "lora")
    if base_label == lora_label:
        raise ValueError("Base and LoRA generators must have distinct model labels")

    comparisons = []
    seen_eval_ids = set()
    for record in records:
        source = "%s:%s" % (record.get("_source_split"), record.get("id"))
        _validate_messages(record.get("messages"), source)
        if record.get("_source_split") not in VALID_SPLITS:
            raise EvaluationDataError("%s: invalid evaluation split" % source)

        gold_messages = _copy_messages(record["messages"])
        assistant_total = sum(
            message["role"] == "assistant" for message in gold_messages
        )
        assistant_turn_policy = _assistant_turn_policy(record, source)
        base_history = []
        lora_history = []
        assistant_turn_index = 0
        latest_user = None
        for message in gold_messages:
            role = message["role"]
            if role != "assistant":
                if role == "user":
                    latest_user = message["content"]
                base_history.append(dict(message))
                lora_history.append(dict(message))
                continue

            assistant_turn_index += 1
            evaluate_turn = (
                assistant_turn_policy == "all"
                or assistant_turn_index == assistant_total
            )
            if not evaluate_turn:
                # A skipped bridge response is still part of the frozen source
                # conversation.  Even rollout mode must use that gold bridge,
                # because no candidate response was generated for this turn.
                base_history.append(dict(message))
                lora_history.append(dict(message))
                continue
            item_seed = stable_item_seed(
                seed, record["_source_split"], record["id"], mode, assistant_turn_index,
            )
            base_prompt = _copy_messages(base_history)
            lora_prompt = _copy_messages(lora_history)
            if mode == "controlled_gold_history" and base_prompt != lora_prompt:
                raise AssertionError("controlled prompts unexpectedly diverged")

            base_response = _invoke_generator(
                base_generator, base_prompt, config, item_seed
            )
            lora_response = _invoke_generator(
                lora_generator, lora_prompt, config, item_seed
            )
            item = _make_comparison_item(
                record=record,
                mode=mode,
                assistant_turn_index=assistant_turn_index,
                latest_user=latest_user,
                gold_response=message["content"],
                base_prompt=base_prompt,
                lora_prompt=lora_prompt,
                base_response=base_response,
                lora_response=lora_response,
                base_label=base_label,
                lora_label=lora_label,
                generation_config=config,
                item_seed=item_seed,
            )
            if item["eval_id"] in seen_eval_ids:
                raise EvaluationDataError("duplicate eval_id %s" % item["eval_id"])
            seen_eval_ids.add(item["eval_id"])
            comparisons.append(item)

            if mode == "controlled_gold_history":
                base_history.append(dict(message))
                lora_history.append(dict(message))
            else:
                base_history.append({"role": "assistant", "content": base_response})
                lora_history.append({"role": "assistant", "content": lora_response})
    return comparisons


def write_jsonl(records, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


class TransformersTextGenerator(object):
    """Lazy Hugging Face generator for a base model or PEFT adapter."""

    def __init__(
        self,
        model_name_or_path,
        model_label,
        adapter_path=None,
        revision=None,
        device_map="auto",
        torch_dtype="auto",
        attention_implementation="eager",
        chat_template_kwargs=None,
        trust_remote_code=False,
        deterministic_algorithms=True,
    ):
        self.model_name_or_path = model_name_or_path
        self.model_label = model_label
        self.adapter_path = adapter_path
        self.revision = revision
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.attention_implementation = attention_implementation
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.trust_remote_code = trust_remote_code
        self.deterministic_algorithms = deterministic_algorithms
        self._torch = None
        self._tokenizer = None
        self._model = None

    def _resolve_dtype(self, torch_module):
        if self.torch_dtype in (None, "auto"):
            return "auto"
        if not isinstance(self.torch_dtype, str):
            return self.torch_dtype
        aliases = {
            "float16": "float16",
            "fp16": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "float32": "float32",
            "fp32": "float32",
        }
        attribute = aliases.get(self.torch_dtype.lower())
        if attribute is None:
            raise ValueError("unsupported torch dtype %r" % self.torch_dtype)
        return getattr(torch_module, attribute)

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "real generation requires torch and transformers; install them "
                "in the Module D experiment environment"
            ) from exc

        tokenizer_kwargs = {"trust_remote_code": self.trust_remote_code}
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "dtype": self._resolve_dtype(torch),
            "attn_implementation": self.attention_implementation,
        }
        if self.revision:
            tokenizer_kwargs["revision"] = self.revision
            model_kwargs["revision"] = self.revision
        if self.device_map:
            model_kwargs["device_map"] = self.device_map

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, **tokenizer_kwargs
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, **model_kwargs
        )
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "loading a LoRA adapter requires the peft package"
                ) from exc
            model = PeftModel.from_pretrained(
                model,
                self.adapter_path,
                is_trainable=False,
                autocast_adapter_dtype=True,
            )
            adapter_dtypes = {
                str(parameter.dtype)
                for name, parameter in model.named_parameters()
                if "lora_" in name
            }
            if adapter_dtypes != {"torch.float32"}:
                raise RuntimeError(
                    "selected adapter is not FP32 as registered: %r" % adapter_dtypes
                )
        resolved_attention = getattr(model.config, "_attn_implementation", None)
        if resolved_attention != self.attention_implementation:
            raise RuntimeError(
                "resolved attention implementation differs from requested: %s != %s"
                % (resolved_attention, self.attention_implementation)
            )
        model.eval()

        if self.deterministic_algorithms:
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def audit_metadata(self):
        self._load()
        template = getattr(self._tokenizer, "chat_template", None)
        if not isinstance(template, str) or not template:
            raise RuntimeError("tokenizer has no chat template")
        adapter_sha = None
        adapter_config_sha = None
        if self.adapter_path:
            adapter_file = Path(self.adapter_path) / "adapter_model.safetensors"
            if not adapter_file.is_file():
                raise RuntimeError("adapter weights are missing: %s" % adapter_file)
            adapter_sha = file_sha256(adapter_file)
            adapter_config_file = Path(self.adapter_path) / "adapter_config.json"
            if not adapter_config_file.is_file():
                raise RuntimeError(
                    "adapter configuration is missing: %s" % adapter_config_file
                )
            adapter_config_sha = file_sha256(adapter_config_file)
        device_map = getattr(self._model, "hf_device_map", None)
        if isinstance(device_map, dict):
            device_map = dict(
                (str(key), str(value)) for key, value in device_map.items()
            )
        resolved_commit = getattr(self._model.config, "_commit_hash", None)
        if (
            self.revision
            and re.fullmatch(r"[0-9a-fA-F]{40}", self.revision)
            and resolved_commit != self.revision
        ):
            raise RuntimeError(
                "resolved model commit differs from requested revision: %s != %s"
                % (resolved_commit, self.revision)
            )
        try:
            first_parameter = next(self._model.parameters())
            actual_dtype = str(first_parameter.dtype)
            actual_device = str(first_parameter.device)
        except StopIteration:
            actual_dtype = None
            actual_device = None
        return {
            "model_name_or_path": self.model_name_or_path,
            "model_label": self.model_label,
            "revision": self.revision,
            "resolved_commit": resolved_commit,
            "adapter_path": self.adapter_path,
            "adapter_sha256": adapter_sha,
            "adapter_config_sha256": adapter_config_sha,
            "dtype_requested": str(self.torch_dtype),
            "dtype_actual_first_parameter": actual_dtype,
            "first_parameter_device": actual_device,
            "attention_implementation_requested": self.attention_implementation,
            "attention_implementation_resolved": getattr(
                self._model.config, "_attn_implementation", None
            ),
            "chat_template_sha256": hashlib.sha256(
                template.encode("utf-8")
            ).hexdigest(),
            "bos_token_id": self._tokenizer.bos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.pad_token_id,
            "padding_side": self._tokenizer.padding_side,
            "chat_template_kwargs": dict(self.chat_template_kwargs),
            "cuda_device_count": int(self._torch.cuda.device_count()),
            "hf_device_map": device_map,
        }

    def generate(self, messages, generation_config, seed):
        self._load()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs
        )
        # The rendered chat template already owns all special-token placement.
        # Adding them again here can duplicate BOS/EOS for some tokenizers.
        model_inputs = tokenizer(
            [prompt_text], return_tensors="pt", add_special_tokens=False
        )
        device = getattr(model, "device", None)
        if device is not None and str(device) != "meta":
            model_inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in model_inputs.items()
            }
        prompt_length = model_inputs["input_ids"].shape[-1]
        generation_config = dict(generation_config)
        generation_config.setdefault("eos_token_id", tokenizer.eos_token_id)
        generation_config.setdefault("pad_token_id", tokenizer.pad_token_id)
        with torch.inference_mode():
            generated = model.generate(**model_inputs, **generation_config)
        output_ids = generated[0][prompt_length:]
        return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def _parse_json_object(value, argument_name):
    try:
        parsed = json.loads(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "%s must be valid JSON: %s" % (argument_name, exc)
        )
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("%s must be a JSON object" % argument_name)
    return parsed


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--split",
        action="append",
        choices=VALID_SPLITS,
        help="repeat to evaluate both splits; defaults to test",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(sorted(MODE_ALIASES.keys())),
        default="controlled_gold_history",
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-model", help="defaults to --base-model")
    parser.add_argument("--lora-adapter")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--lora-revision")
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--lora-label", default="lora")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--chat-template-kwargs",
        default={},
        type=lambda value: _parse_json_object(value, "--chat-template-kwargs"),
        help='JSON object, e.g. {"enable_thinking": false} for Qwen3',
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int)
    adapter_source = parser.add_mutually_exclusive_group()
    adapter_source.add_argument(
        "--selection-manifest",
        help="strict mode: proves validation selected this exact checkpoint",
    )
    adapter_source.add_argument(
        "--use-final-adapter",
        action="store_true",
        help=(
            "coursework mode: test the completed training run's adapter-final "
            "without checkpoint selection or the Module C safety gate"
        ),
    )
    parser.add_argument(
        "--manifest-output", help="generation manifest; defaults beside --output",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if (
        not args.lora_adapter
        and (args.lora_model or args.base_model) == args.base_model
    ):
        parser.error(
            "provide --lora-adapter or a distinct --lora-model for the LoRA candidate"
        )

    splits = tuple(args.split or ("test",))
    if len(splits) != 1:
        parser.error("generate exactly one split per run")
    if "test" in splits and args.limit is not None:
        parser.error("--limit is forbidden for the reportable frozen test run")
    if not Path(args.base_model).is_dir() and not re.fullmatch(
        r"[0-9a-fA-F]{40}", args.base_revision
    ):
        parser.error("Hub --base-revision must be an immutable 40-character commit")
    test_binding = None
    adapter_provenance = None
    if "test" in splits:
        if not args.lora_adapter:
            parser.error("test generation requires --lora-adapter")
        try:
            if args.selection_manifest:
                test_binding = validate_test_selection(
                    args.selection_manifest,
                    args.lora_adapter,
                    args.base_model,
                    args.base_revision,
                )
                adapter_provenance = "checkpoint_selection"
            elif args.use_final_adapter:
                test_binding = validate_test_final_adapter(
                    args.lora_adapter,
                    args.base_model,
                    args.base_revision,
                )
                adapter_provenance = "training_final"
            else:
                parser.error(
                    "test generation requires --selection-manifest (strict mode) "
                    "or --use-final-adapter (coursework mode)"
                )
            validate_test_generation_contract(
                test_binding,
                dtype=args.dtype,
                attention_implementation=args.attention_implementation,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                chat_template_kwargs=args.chat_template_kwargs,
                python_hash_seed=os.environ.get("PYTHONHASHSEED"),
            )
        except EvaluationDataError as exc:
            parser.error(str(exc))
    elif args.use_final_adapter:
        parser.error("--use-final-adapter is only valid for test generation")
    if args.lora_adapter:
        lora_model_candidate = args.lora_model or args.base_model
        lora_revision_candidate = args.lora_revision or args.base_revision
        if lora_model_candidate != args.base_model:
            parser.error("adapter comparison requires the same Base model path")
        if lora_revision_candidate != args.base_revision:
            parser.error("adapter comparison requires the same Base model revision")

    records = load_evaluation_records(
        data_root=args.data_root,
        splits=splits,
        expected_sha256=REGISTERED_SOURCE_SHA256,
    )
    if args.limit is not None:
        records = records[: args.limit]
    lora_model = args.lora_model or args.base_model
    base_generator = TransformersTextGenerator(
        model_name_or_path=args.base_model,
        model_label=args.base_label,
        revision=args.base_revision,
        device_map=args.device_map,
        torch_dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        chat_template_kwargs=args.chat_template_kwargs,
        trust_remote_code=args.trust_remote_code,
    )
    lora_generator = TransformersTextGenerator(
        model_name_or_path=lora_model,
        model_label=args.lora_label,
        adapter_path=args.lora_adapter,
        revision=args.lora_revision or args.base_revision,
        device_map=args.device_map,
        torch_dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        chat_template_kwargs=args.chat_template_kwargs,
        trust_remote_code=args.trust_remote_code,
    )
    output_path = Path(args.output)
    manifest_path = (
        Path(args.manifest_output)
        if args.manifest_output
        else output_path.with_suffix(output_path.suffix + ".manifest.json")
    )
    if output_path.resolve() == manifest_path.resolve():
        parser.error("--output and --manifest-output must be different files")
    protected_inputs = [
        (Path(args.data_root) / (split + ".jsonl")).resolve() for split in splits
    ]
    if args.selection_manifest:
        protected_inputs.append(Path(args.selection_manifest).resolve())
    if test_binding:
        protected_inputs.append(Path(test_binding["run_manifest"]).resolve())
    if args.lora_adapter:
        protected_inputs.extend(
            [
                (Path(args.lora_adapter) / "adapter_model.safetensors").resolve(),
                (Path(args.lora_adapter) / "adapter_config.json").resolve(),
            ]
        )
    if (
        output_path.resolve() in protected_inputs
        or manifest_path.resolve() in protected_inputs
    ):
        parser.error(
            "generation outputs must not overwrite data, selection, or adapter inputs"
        )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    base_audit = base_generator.audit_metadata()
    lora_audit = lora_generator.audit_metadata()
    try:
        base_tokenizer_identity = canonical_tokenizer_identity(base_audit)
        lora_tokenizer_identity = canonical_tokenizer_identity(lora_audit)
    except ExperimentError as exc:
        raise RuntimeError("Runtime tokenizer identity is invalid: %s" % exc) from exc
    if base_tokenizer_identity != lora_tokenizer_identity:
        raise RuntimeError("Base and LoRA runtime tokenizer identities differ")
    comparable_runtime_fields = (
        "model_name_or_path",
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
    )
    for field in comparable_runtime_fields:
        if base_audit.get(field) != lora_audit.get(field):
            raise RuntimeError("Base and LoRA runtime %s differs" % field)
    if test_binding:
        validate_test_runtime_identity(test_binding, base_audit, lora_audit)
    comparisons = generate_comparisons(
        records=records,
        base_generator=base_generator,
        lora_generator=lora_generator,
        mode=args.mode,
        generation_config={
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        },
        seed=args.seed,
    )
    write_jsonl(comparisons, args.output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    generation_manifest = {
        "schema_version": "module_d.generation_manifest.v1",
        "splits": list(splits),
        "records": len(records),
        "comparisons": len(comparisons),
        "mode": normalize_mode(args.mode),
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "lora_adapter": args.lora_adapter,
        "base_runtime": base_audit,
        "lora_runtime": lora_audit,
        "source_sha256": dict(
            (
                split,
                next(
                    record["_source_sha256"]
                    for record in records
                    if record["_source_split"] == split
                ),
            )
            for split in splits
        ),
        "adapter_provenance": adapter_provenance,
        "selected_adapter_sha256": test_binding["selected"]["adapter_sha256"]
        if test_binding
        else None,
        "selected_adapter_config_sha256": test_binding["selected"]["adapter_config_sha256"]
        if test_binding
        else None,
        "experiment_config_sha256": test_binding.get("experiment_config_sha256")
        if test_binding
        else None,
        "run_manifest": test_binding.get("run_manifest") if test_binding else None,
        "run_manifest_sha256": test_binding.get("run_manifest_sha256")
        if test_binding
        else None,
        "selection_manifest": args.selection_manifest,
        "selection_manifest_sha256": file_sha256(args.selection_manifest)
        if args.selection_manifest
        else None,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "seed": args.seed,
        },
        "attention_implementation": args.attention_implementation,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "chat_template_kwargs": args.chat_template_kwargs,
        "output": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            generation_manifest, handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "ok",
                "records": len(records),
                "comparisons": len(comparisons),
                "mode": normalize_mode(args.mode),
                "output": str(Path(args.output)),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
