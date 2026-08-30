#!/usr/bin/env python3
"""Select a validation checkpoint using NLL plus the frozen safety gate."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scripts.module_c.common import (
    ExperimentError,
    load_json,
    load_jsonl,
    sha256_file,
    source_record_counts_by_capability,
    workspace_path,
    write_json,
)
from scripts.module_c.make_safety_review import (
    COMMON_CHECKS,
    EXPECTED_ASSISTANT_TURNS,
    V2_EXTRA_CHECKS,
    validate_registered_validation_comparisons,
)
from scripts.module_d.build_review_sheet import (
    load_comparisons,
    validate_generation_manifest,
)
from scripts.module_c.train_lora import (
    checkpoint_artifact_snapshot,
    validate_config,
    verify_training_data,
)


DEFAULT_CONFIG = "configs/module_c/hutao_qwen3_1p7b_lora_bf16.json"


def _close(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    )


def _recompute_validation_metrics(
    metrics: Mapping[str, Any], expected_examples: Sequence[Mapping[str, Any]]
) -> Tuple[Optional[float], List[str]]:
    """Recompute every reported aggregate from the frozen per-example rows."""
    failures: List[str] = []
    expected = {example["id"]: example for example in expected_examples}
    expected_records_per_capability = source_record_counts_by_capability(
        expected_examples
    )
    if not expected_records_per_capability:
        return None, ["frozen validation contains no capabilities"]
    if len(expected) != len(expected_examples):
        return None, ["frozen validation contains duplicate example IDs"]
    rows = metrics.get("per_example")
    if not isinstance(rows, list) or len(rows) != len(expected):
        return None, ["metric per_example coverage is incomplete"]
    actual_ids = [row.get("id") for row in rows if isinstance(row, dict)]
    if (
        len(actual_ids) != len(rows)
        or set(actual_ids) != set(expected)
        or len(set(actual_ids)) != len(actual_ids)
    ):
        return None, ["metric per_example IDs differ from frozen validation"]

    record_totals: Dict[str, Dict[str, Any]] = {}
    total_nll = 0.0
    total_tokens = 0
    for row in rows:
        example = expected[row["id"]]
        source_record_id = example["source_record_id"]
        capability = example["metadata"]["capability"]
        if (
            row.get("source_record_id") != source_record_id
            or row.get("capability") != capability
        ):
            failures.append("{} trace metadata differs".format(row["id"]))
            continue
        tokens = row.get("supervised_tokens")
        nll_sum = row.get("nll_sum")
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens < 1
            or isinstance(nll_sum, bool)
            or not isinstance(nll_sum, (int, float))
            or not math.isfinite(float(nll_sum))
            or float(nll_sum) < 0.0
        ):
            failures.append("{} has invalid token count or NLL".format(row["id"]))
            continue
        mean_nll = float(nll_sum) / tokens
        if not _close(row.get("mean_nll"), mean_nll):
            failures.append("{} mean_nll is inconsistent".format(row["id"]))
        aggregate = record_totals.setdefault(
            source_record_id, {"capability": capability, "nll_sum": 0.0, "tokens": 0},
        )
        aggregate["nll_sum"] += float(nll_sum)
        aggregate["tokens"] += tokens
        total_nll += float(nll_sum)
        total_tokens += tokens

    if failures:
        return None, failures
    declared_records = metrics.get("per_record")
    if not isinstance(declared_records, dict) or set(declared_records) != set(
        record_totals
    ):
        return None, ["metric per_record coverage is inconsistent"]
    capability_values: Dict[str, List[float]] = defaultdict(list)
    for record_id, aggregate in record_totals.items():
        mean_nll = aggregate["nll_sum"] / aggregate["tokens"]
        declared = declared_records[record_id]
        if (
            declared.get("capability") != aggregate["capability"]
            or declared.get("supervised_tokens") != aggregate["tokens"]
            or not _close(declared.get("mean_nll"), mean_nll)
        ):
            failures.append("{} per_record aggregate is inconsistent".format(record_id))
        capability_values[aggregate["capability"]].append(mean_nll)

    declared_capabilities = metrics.get("per_capability")
    if not isinstance(declared_capabilities, dict) or set(declared_capabilities) != set(
        expected_records_per_capability
    ):
        failures.append("metric per_capability coverage is inconsistent")
    capability_means: List[float] = []
    for capability, expected_record_count in expected_records_per_capability.items():
        values = capability_values.get(capability, [])
        if len(values) != expected_record_count:
            failures.append(
                "{} contains {} validation records; expected {}".format(
                    capability, len(values), expected_record_count
                )
            )
            continue
        mean_nll = sum(values) / len(values)
        capability_means.append(mean_nll)
        declared = (
            declared_capabilities.get(capability, {})
            if isinstance(declared_capabilities, dict)
            else {}
        )
        if declared.get("records") != expected_record_count or not _close(
            declared.get("mean_record_nll"), mean_nll
        ):
            failures.append("{} aggregate is inconsistent".format(capability))

    if failures or len(capability_means) != len(expected_records_per_capability):
        return None, failures
    macro_nll = sum(capability_means) / len(capability_means)
    token_weighted_nll = total_nll / total_tokens
    if not _close(metrics.get("capability_macro_nll"), macro_nll):
        failures.append("capability_macro_nll differs from per-example values")
    if not _close(metrics.get("token_weighted_nll"), token_weighted_nll):
        failures.append("token_weighted_nll differs from per-example values")
    return (None if failures else macro_nll), failures


def _validate_safety_review(
    metric: Mapping[str, Any],
    review: Mapping[str, Any],
    required_records: Sequence[str],
    config_sha256: str,
    config: Mapping[str, Any],
) -> Tuple[bool, List[str], List[str]]:
    """Separate provenance/integrity failures from genuine human safety failures."""
    integrity_failures: List[str] = []
    safety_failures: List[str] = []
    if review.get("schema_version") != "module_c.safety_gate_review.v1":
        integrity_failures.append("unsupported safety review schema")
    if review.get("experiment_config_sha256") != config_sha256:
        integrity_failures.append("safety review config_sha256 mismatch")
    if review.get("adapter_sha256") != metric.get("adapter_sha256"):
        integrity_failures.append("adapter_sha256 mismatch")
    if review.get("adapter_config_sha256") != metric.get("adapter_config_sha256"):
        integrity_failures.append("adapter_config_sha256 mismatch")
    try:
        if (
            Path(review.get("adapter_path", "")).resolve()
            != Path(metric.get("adapter_path", "")).resolve()
        ):
            integrity_failures.append("adapter_path mismatch")
        comparison_path = Path(review.get("comparison_file", ""))
        generation_manifest_path = Path(review.get("generation_manifest", ""))
        if not comparison_path.is_file():
            integrity_failures.append("safety comparison file is missing")
            comparison_rows: List[Dict[str, Any]] = []
        else:
            if sha256_file(comparison_path) != review.get("comparison_file_sha256"):
                integrity_failures.append("safety comparison file hash mismatch")
            comparison_rows = load_comparisons(comparison_path)
        if not generation_manifest_path.is_file():
            integrity_failures.append("safety generation manifest is missing")
            generation_manifest: Dict[str, Any] = {}
        else:
            if sha256_file(generation_manifest_path) != review.get(
                "generation_manifest_sha256"
            ):
                integrity_failures.append("safety generation manifest hash mismatch")
            generation_manifest = validate_generation_manifest(
                generation_manifest_path, comparison_path, comparison_rows
            )
            validate_registered_validation_comparisons(
                dict(config), comparison_rows, generation_manifest
            )
        lora_runtime = generation_manifest.get("lora_runtime", {})
        if lora_runtime.get("adapter_sha256") != metric.get("adapter_sha256"):
            integrity_failures.append(
                "generated outputs used different adapter weights"
            )
        if lora_runtime.get("adapter_config_sha256") != metric.get(
            "adapter_config_sha256"
        ):
            integrity_failures.append(
                "generated outputs used different adapter configuration"
            )
    except (ExperimentError, OSError, ValueError, KeyError, TypeError) as exc:
        integrity_failures.append("safety provenance validation failed: {}".format(exc))
        comparison_rows = []
    records = review.get("records")
    if not isinstance(records, dict):
        integrity_failures.append("safety review has no records object")
        return False, integrity_failures, safety_failures
    if set(records) != set(required_records):
        integrity_failures.append(
            "safety review record set differs from the frozen gate"
        )
    for record_id in required_records:
        value = records.get(record_id)
        if not isinstance(value, dict):
            integrity_failures.append("{} review entry is missing".format(record_id))
            continue
        if value.get("pass") is not True:
            safety_failures.append("{} did not pass".format(record_id))
        checks = value.get("checks")
        expected_checks = set(COMMON_CHECKS)
        if record_id.endswith("V2"):
            expected_checks.update(V2_EXTRA_CHECKS)
        if not isinstance(checks, dict) or set(checks) != expected_checks:
            integrity_failures.append("{} checklist schema differs".format(record_id))
        elif any(check_value is not True for check_value in checks.values()):
            safety_failures.append(
                "{} checklist is incomplete or failed".format(record_id)
            )
        if (
            not isinstance(value.get("reviewer_id"), str)
            or not value["reviewer_id"].strip()
        ):
            integrity_failures.append("{} reviewer_id is empty".format(record_id))
        expected_outputs = []
        for row in comparison_rows:
            if row.get("record_id") != record_id:
                continue
            expected_outputs.append(
                {
                    "eval_id": row["eval_id"],
                    "assistant_turn_index": row["assistant_turn_index"],
                    "latest_user_message": row["latest_user_message"],
                    "prompt_messages": row["lora"]["prompt_messages"],
                    "lora_response": row["lora"]["response"],
                }
            )
        expected_outputs.sort(key=lambda item: item["assistant_turn_index"])
        if value.get("outputs") != expected_outputs:
            integrity_failures.append(
                "{} reviewed outputs differ from generation".format(record_id)
            )
        actual_turns = tuple(item["assistant_turn_index"] for item in expected_outputs)
        if actual_turns != EXPECTED_ASSISTANT_TURNS[record_id]:
            integrity_failures.append(
                "{} assistant turns are incomplete".format(record_id)
            )
    passed = not integrity_failures and not safety_failures
    return passed, integrity_failures, safety_failures


def select(
    config_path: Path,
    candidates: Sequence[Tuple[Path, Path]],
    output_path: Optional[Path],
) -> Dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    config_sha256 = sha256_file(config_path)
    expected_data_snapshot = verify_training_data(config)
    canonical_run = workspace_path(config["training"]["output_dir"]).resolve()
    run_manifest_path = canonical_run / "run_manifest.json"
    run_manifest = load_json(run_manifest_path)
    if (
        run_manifest.get("mode") != "main"
        or run_manifest.get("status") != "complete"
        or run_manifest.get("config_sha256") != config_sha256
        or run_manifest.get("config") != config
        or run_manifest.get("data") != expected_data_snapshot
        or run_manifest.get("runtime_check", {}).get("mismatches")
        or run_manifest.get("runtime_check", {}).get("mismatch_override") is True
        or run_manifest.get("hardware", {}).get("cuda_available") is not True
    ):
        raise ExperimentError(
            "Canonical run_manifest.json is incomplete or inconsistent"
        )
    run_manifest_sha256 = sha256_file(run_manifest_path)
    validation_source = load_jsonl(
        workspace_path(config["data"]["source_dir"]) / "validation.jsonl"
    )
    expected_validation_ids = {record["id"] for record in validation_source}
    validation_examples = load_jsonl(
        workspace_path(config["data"]["derived_dir"]) / "validation.jsonl"
    )
    selection = config["checkpoint_selection"]
    required_records = selection["required_safety_records"]
    inspected: List[Dict[str, Any]] = []

    for metric_path, review_path in candidates:
        metric = load_json(metric_path)
        review = load_json(review_path)
        adapter_path = workspace_path(metric["adapter_path"])
        adapter_model_path = adapter_path / "adapter_model.safetensors"
        adapter_config_path = adapter_path / "adapter_config.json"
        integrity_failures: List[str] = []
        checkpoint_match = re.fullmatch(r"checkpoint-([0-9]+)", adapter_path.name)
        if adapter_path.resolve().parent != canonical_run or checkpoint_match is None:
            integrity_failures.append(
                "adapter is not a checkpoint-N inside the canonical output_dir"
            )
            checkpoint_step = 10 ** 18
        else:
            checkpoint_step = int(checkpoint_match.group(1))
        if metric.get("status") != "scored_unreviewed_for_safety":
            integrity_failures.append("metric status is invalid")
        metric_value, aggregation_failures = _recompute_validation_metrics(
            metric.get("metrics", {}), validation_examples
        )
        integrity_failures.extend(aggregation_failures)
        if metric.get("config_sha256") != config_sha256:
            integrity_failures.append("metric config_sha256 mismatch")
        if metric.get("model") != config.get("model"):
            integrity_failures.append("metric Base model identity mismatch")
        if metric.get("resolved_model_revision") != config["model"]["revision"]:
            integrity_failures.append("metric resolved Base revision mismatch")
        if metric.get("tokenizer") != run_manifest.get("tokenizer"):
            integrity_failures.append("metric tokenizer differs from training")
        if metric.get("evaluation_base_precision") != "{}_unquantized".format(
            config["model"]["dtype"]
        ):
            integrity_failures.append("metric evaluation precision is inconsistent")
        if metric.get("data") != expected_data_snapshot:
            integrity_failures.append("metric validation data snapshot mismatch")
        runtime_check = metric.get("runtime_check")
        if (
            not isinstance(runtime_check, dict)
            or runtime_check.get("mismatches")
            or runtime_check.get("mismatch_override") is True
        ):
            integrity_failures.append(
                "metric was not produced in the canonical runtime"
            )
        hardware = metric.get("hardware")
        if config["runtime"].get("require_cuda", True) and (
            not isinstance(hardware, dict) or hardware.get("cuda_available") is not True
        ):
            integrity_failures.append("metric was not produced on CUDA")
        determinism = metric.get("determinism")
        expected_seed = int(config["training"]["seed"])
        if (
            not isinstance(determinism, dict)
            or determinism.get("seed") != expected_seed
            or determinism.get("python_hash_seed") != str(expected_seed)
            or determinism.get("cublas_workspace_config") != ":4096:8"
            or determinism.get("deterministic_algorithms") is not True
            or determinism.get("tf32_matmul") is not False
            or determinism.get("tf32_cudnn") is not False
        ):
            integrity_failures.append(
                "metric was not produced with the registered deterministic settings"
            )
        per_record = metric.get("metrics", {}).get("per_record")
        if (
            not isinstance(per_record, dict)
            or set(per_record) != expected_validation_ids
        ):
            integrity_failures.append("metric validation record coverage is incomplete")
        if not adapter_model_path.is_file():
            integrity_failures.append("adapter weights are missing")
        elif sha256_file(adapter_model_path) != metric.get("adapter_sha256"):
            integrity_failures.append("adapter weights changed after validation")
        if not adapter_config_path.is_file():
            integrity_failures.append("adapter_config.json is missing")
        elif sha256_file(adapter_config_path) != metric.get("adapter_config_sha256"):
            integrity_failures.append("adapter_config.json changed after validation")
        else:
            adapter_config = load_json(adapter_config_path)
            if (
                adapter_config.get("base_model_name_or_path") != config["model"]["name"]
                or adapter_config.get("revision") != config["model"]["revision"]
            ):
                integrity_failures.append("adapter Base identity is inconsistent")
        if (
            Path(metric.get("run_manifest", "")).resolve()
            != run_manifest_path.resolve()
        ):
            integrity_failures.append("metric points to a different run manifest")
        if metric.get("run_manifest_sha256") != run_manifest_sha256:
            integrity_failures.append("run manifest changed after validation")
        try:
            checkpoint_artifacts = checkpoint_artifact_snapshot(
                adapter_path, require_grad_scaler=config["model"]["dtype"] == "float16",
            )
        except ExperimentError as exc:
            checkpoint_artifacts = None
            integrity_failures.append(str(exc))
        if checkpoint_artifacts != metric.get("checkpoint_artifacts"):
            integrity_failures.append("checkpoint state changed after validation")
        if metric.get("checkpoint_step") != checkpoint_step:
            integrity_failures.append("metric checkpoint step is inconsistent")
        passed, review_integrity_failures, safety_failures = _validate_safety_review(
            metric, review, required_records, config_sha256, config
        )
        integrity_failures.extend(review_integrity_failures)
        passed = passed and not integrity_failures
        inspected.append(
            {
                "metric_file": str(metric_path),
                "metric_file_sha256": sha256_file(metric_path),
                "safety_review_file": str(review_path),
                "safety_review_file_sha256": sha256_file(review_path),
                "adapter_path": metric["adapter_path"],
                "adapter_sha256": metric["adapter_sha256"],
                "adapter_config_sha256": metric.get("adapter_config_sha256"),
                "model": metric.get("model"),
                "capability_macro_nll": metric_value,
                "checkpoint_step": checkpoint_step,
                "integrity_pass": not integrity_failures,
                "integrity_failures": integrity_failures,
                "safety_pass": passed,
                "safety_failures": safety_failures,
            }
        )

    train_examples = int(config["data"]["expected_derived_examples"]["train"])
    micro_batch = int(config["training"]["per_device_train_batch_size"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    steps_per_epoch = int(
        math.ceil(math.ceil(train_examples / float(micro_batch)) / float(accumulation))
    )
    epochs = float(config["training"]["num_train_epochs"])
    if not epochs.is_integer():
        raise ExperimentError("Checkpoint selection requires an integer epoch count")
    expected_steps = {steps_per_epoch * epoch for epoch in range(1, int(epochs) + 1)}
    actual_steps = [candidate["checkpoint_step"] for candidate in inspected]
    if set(actual_steps) != expected_steps or len(actual_steps) != len(expected_steps):
        result = {
            "schema_version": "module_c.checkpoint_selection.v1",
            "status": "failed_incomplete_candidate_set",
            "selected": None,
            "candidates": inspected,
            "expected_checkpoint_steps": sorted(expected_steps),
            "selection_rule": selection,
            "experiment_config_sha256": config_sha256,
            "model": config["model"],
            "run_manifest": str(run_manifest_path),
            "run_manifest_sha256": run_manifest_sha256,
        }
        if output_path is not None:
            write_json(output_path, result)
        raise ExperimentError(
            "Checkpoint candidates must cover every saved epoch: {}".format(
                sorted(expected_steps)
            )
        )

    candidates_with_integrity_failures = [
        candidate for candidate in inspected if not candidate["integrity_pass"]
    ]
    if candidates_with_integrity_failures:
        result = {
            "schema_version": "module_c.checkpoint_selection.v1",
            "status": "failed_candidate_integrity",
            "selected": None,
            "candidates": inspected,
            "expected_checkpoint_steps": sorted(expected_steps),
            "selection_rule": selection,
            "experiment_config_sha256": config_sha256,
            "model": config["model"],
            "run_manifest": str(run_manifest_path),
            "run_manifest_sha256": run_manifest_sha256,
        }
        if output_path is not None:
            write_json(output_path, result)
        raise ExperimentError(
            "At least one checkpoint candidate failed integrity validation"
        )

    eligible = [candidate for candidate in inspected if candidate["safety_pass"]]
    if not eligible:
        result = {
            "schema_version": "module_c.checkpoint_selection.v1",
            "status": "failed_no_safe_checkpoint",
            "selected": None,
            "candidates": inspected,
            "selection_rule": selection,
            "experiment_config_sha256": config_sha256,
            "model": config["model"],
            "run_manifest": str(run_manifest_path),
            "run_manifest_sha256": run_manifest_sha256,
            "expected_checkpoint_steps": sorted(expected_steps),
        }
        if output_path is not None:
            write_json(output_path, result)
        raise ExperimentError("No checkpoint passed the frozen validation safety gate")

    eligible.sort(
        key=lambda item: (item["capability_macro_nll"], item["checkpoint_step"])
    )
    best = eligible[0]
    tolerance = float(selection["relative_tie_tolerance"])
    tied = [
        item
        for item in eligible
        if abs(item["capability_macro_nll"] - best["capability_macro_nll"])
        <= abs(best["capability_macro_nll"]) * tolerance
    ]
    if selection.get("prefer_earlier_on_tie", True):
        best = min(tied, key=lambda item: item["checkpoint_step"])

    result = {
        "schema_version": "module_c.checkpoint_selection.v1",
        "status": "selected",
        "selected": best,
        "candidates": inspected,
        "selection_rule": selection,
        "experiment_config_sha256": config_sha256,
        "model": config["model"],
        "run_manifest": str(run_manifest_path),
        "run_manifest_sha256": run_manifest_sha256,
        "expected_checkpoint_steps": sorted(expected_steps),
        "test_access_authorised_after_this_manifest": True,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--candidate",
        nargs=2,
        action="append",
        metavar=("METRICS_JSON", "SAFETY_REVIEW_JSON"),
        required=True,
        help="Repeat once per checkpoint.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_paths = [
        (workspace_path(pair[0]), workspace_path(pair[1])) for pair in args.candidate
    ]
    result = select(
        workspace_path(args.config), candidate_paths, workspace_path(args.output)
    )
    print("Selected {}".format(result["selected"]["adapter_path"]))


if __name__ == "__main__":
    main()
