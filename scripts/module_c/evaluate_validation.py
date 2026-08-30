#!/usr/bin/env python3
"""Compute completion-only validation NLL for one saved adapter checkpoint."""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from scripts.module_c.common import (
    ExperimentError,
    environment_snapshot,
    load_json,
    load_jsonl,
    sha256_file,
    source_record_counts_by_capability,
    workspace_path,
    write_json,
)
from scripts.module_c.tokenization import tokenize_completion_example
from scripts.module_c.train_lora import (
    _assert_lora_adapter_dtype,
    _assert_model_revision,
    _hardware_snapshot,
    _model_load_kwargs,
    _tokenizer_snapshot,
    checkpoint_artifact_snapshot,
    validate_config,
    verify_runtime,
    verify_training_data,
)


DEFAULT_CONFIG = "configs/module_c/hutao_qwen3_1p7b_lora_bf16.json"


def evaluate(
    config_path: Path,
    adapter_path: Path,
    output_path: Path,
    allow_version_mismatch: bool,
    allow_non_cuda: bool,
) -> Dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = load_json(config_path)
    validate_config(config)
    seed = int(config["training"]["seed"])
    if bool(config["training"]["full_determinism"]) and os.environ.get(
        "PYTHONHASHSEED"
    ) != str(seed):
        raise ExperimentError(
            "Launch canonical validation with PYTHONHASHSEED={} set before Python "
            "starts".format(seed)
        )
    runtime = verify_runtime(config, allow_version_mismatch)
    data_snapshot = verify_training_data(config)
    canonical_run = workspace_path(config["training"]["output_dir"]).resolve()
    resolved_adapter = adapter_path.resolve()
    checkpoint_match = re.fullmatch(r"checkpoint-([0-9]+)", resolved_adapter.name)
    if resolved_adapter.parent != canonical_run or checkpoint_match is None:
        raise ExperimentError(
            "Validation adapter must be a checkpoint-N inside {}".format(canonical_run)
        )
    checkpoint_step = int(checkpoint_match.group(1))
    run_manifest_path = canonical_run / "run_manifest.json"
    run_manifest = load_json(run_manifest_path)
    if run_manifest.get("mode") != "main" or run_manifest.get("status") != "complete":
        raise ExperimentError("Validation requires a completed canonical main run")
    if run_manifest.get("config_sha256") != sha256_file(config_path):
        raise ExperimentError("Run manifest config hash differs from validation config")
    if run_manifest.get("data") != data_snapshot:
        raise ExperimentError("Run manifest data snapshot differs from validation data")
    if run_manifest.get("runtime_check", {}).get("mismatches"):
        raise ExperimentError("Run manifest contains a runtime mismatch")
    if run_manifest.get("hardware", {}).get("cuda_available") is not True:
        raise ExperimentError("Run manifest is not a CUDA training run")
    checkpoint_artifacts = checkpoint_artifact_snapshot(
        resolved_adapter, require_grad_scaler=config["model"]["dtype"] == "float16",
    )
    trainer_state = load_json(resolved_adapter / "trainer_state.json")
    if trainer_state.get("global_step") != checkpoint_step:
        raise ExperimentError("Checkpoint directory step differs from trainer_state")
    try:
        import torch
        import torch.nn.functional as functional
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise ExperimentError(
            "Validation scoring requires the training environment"
        ) from exc

    if config["runtime"].get("require_cuda", True) and not torch.cuda.is_available():
        if not allow_non_cuda:
            raise ExperimentError("Canonical validation scoring requires CUDA")
    try:
        actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ExperimentError("WORLD_SIZE must be an integer") from exc
    if actual_world_size != int(config["runtime"]["world_size"]):
        raise ExperimentError("Canonical validation scoring requires WORLD_SIZE=1")
    if torch.cuda.is_available() and torch.cuda.device_count() != int(
        config["runtime"]["visible_cuda_devices"]
    ):
        raise ExperimentError(
            "Canonical validation scoring requires one visible CUDA device"
        )
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    set_seed(seed, deterministic=bool(config["training"]["full_determinism"]))
    if not (adapter_path / "adapter_config.json").exists():
        raise ExperimentError("Not a PEFT adapter directory: {}".format(adapter_path))
    adapter_config_path = adapter_path / "adapter_config.json"
    adapter_model_path = adapter_path / "adapter_model.safetensors"
    if not adapter_model_path.is_file():
        raise ExperimentError("Missing adapter weights: {}".format(adapter_model_path))
    adapter_config = load_json(adapter_config_path)
    if adapter_config.get("base_model_name_or_path") != config["model"]["name"]:
        raise ExperimentError("Adapter configuration names a different Base model")
    if adapter_config.get("revision") != config["model"]["revision"]:
        raise ExperimentError("Adapter configuration does not freeze the Base revision")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], revision=config["model"]["revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    chat_template_kwargs = dict(config["model"]["chat_template_kwargs"])
    tokenizer_snapshot = _tokenizer_snapshot(tokenizer, chat_template_kwargs)
    if tokenizer_snapshot != run_manifest.get("tokenizer"):
        raise ExperimentError("Validation tokenizer differs from the training run")

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        # Evaluate both LoRA and QLoRA adapters in the same unquantized BF16
        # deployment form used by Module D generation.
        **_model_load_kwargs(config, torch, quantized=False)
    )
    resolved_model_revision = _assert_model_revision(
        base_model, config["model"]["revision"]
    )
    model = PeftModel.from_pretrained(
        base_model, str(adapter_path), is_trainable=False, autocast_adapter_dtype=True,
    )
    model.config.use_cache = False
    _assert_lora_adapter_dtype(model, torch.float32)
    model.eval()
    if not hasattr(model, "hf_device_map"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = model.device

    examples = load_jsonl(
        workspace_path(config["data"]["derived_dir"]) / "validation.jsonl"
    )
    source_records = {example["source_record_id"] for example in examples}
    expected_records_per_capability = source_record_counts_by_capability(examples)
    if len(examples) != config["data"]["expected_derived_examples"]["validation"]:
        raise ExperimentError("Validation turn-view count differs from config")
    if len(source_records) != config["data"]["expected_source_records"]["validation"]:
        raise ExperimentError("Validation source-record count differs from config")
    if not expected_records_per_capability:
        raise ExperimentError("Validation contains no capabilities")
    per_example: List[Dict[str, Any]] = []
    record_totals: Dict[str, Dict[str, Any]] = {}

    with torch.inference_mode():
        for example in examples:
            tokenized = tokenize_completion_example(
                example,
                tokenizer,
                int(config["data"]["max_length"]),
                chat_template_kwargs=chat_template_kwargs,
            )
            input_ids = torch.tensor(
                [tokenized["input_ids"]], dtype=torch.long, device=device
            )
            attention_mask = torch.tensor(
                [tokenized["attention_mask"]], dtype=torch.long, device=device
            )
            labels = torch.tensor(
                [tokenized["labels"]], dtype=torch.long, device=device
            )
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Match Transformers' causal-LM loss precision: cross entropy is
            # accumulated from float32 logits even when inference uses BF16.
            shift_logits = outputs.logits[:, :-1, :].float().contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_loss = functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            supervised_tokens = int((shift_labels != -100).sum().item())
            if supervised_tokens < 1:
                raise ExperimentError(
                    "{} has zero shifted labels".format(example["id"])
                )
            nll_sum = float(flat_loss.item())
            mean_nll = nll_sum / supervised_tokens
            if not math.isfinite(nll_sum) or not math.isfinite(mean_nll):
                raise ExperimentError(
                    "{} produced non-finite validation NLL".format(example["id"])
                )
            item = {
                "id": example["id"],
                "source_record_id": example["source_record_id"],
                "capability": example["metadata"]["capability"],
                "supervised_tokens": supervised_tokens,
                "nll_sum": nll_sum,
                "mean_nll": mean_nll,
            }
            per_example.append(item)
            aggregate = record_totals.setdefault(
                example["source_record_id"],
                {
                    "capability": example["metadata"]["capability"],
                    "nll_sum": 0.0,
                    "supervised_tokens": 0,
                },
            )
            aggregate["nll_sum"] += nll_sum
            aggregate["supervised_tokens"] += supervised_tokens

    per_record: Dict[str, Dict[str, Any]] = {}
    capability_values: Dict[str, List[float]] = defaultdict(list)
    for record_id, aggregate in sorted(record_totals.items()):
        mean_nll = aggregate["nll_sum"] / aggregate["supervised_tokens"]
        per_record[record_id] = {
            "capability": aggregate["capability"],
            "supervised_tokens": aggregate["supervised_tokens"],
            "mean_nll": mean_nll,
        }
        capability_values[aggregate["capability"]].append(mean_nll)

    per_capability = {
        capability: {
            "records": len(values),
            "mean_record_nll": sum(values) / len(values),
        }
        for capability, values in sorted(capability_values.items())
    }
    actual_records_per_capability = {
        capability: value["records"]
        for capability, value in per_capability.items()
    }
    if actual_records_per_capability != expected_records_per_capability:
        raise ExperimentError(
            "Validation capability source-record counts differ from frozen examples"
        )
    capability_macro_nll = sum(
        item["mean_record_nll"] for item in per_capability.values()
    ) / len(per_capability)
    token_weighted_nll = sum(item["nll_sum"] for item in per_example) / sum(
        item["supervised_tokens"] for item in per_example
    )

    result: Dict[str, Any] = {
        "status": "scored_unreviewed_for_safety",
        "config_sha256": sha256_file(config_path),
        "model": config["model"],
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_file(adapter_model_path),
        "adapter_config_sha256": sha256_file(adapter_config_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_artifacts": checkpoint_artifacts,
        "evaluation_base_precision": "{}_unquantized".format(config["model"]["dtype"]),
        "resolved_model_revision": resolved_model_revision,
        "tokenizer": tokenizer_snapshot,
        "run_manifest": str(run_manifest_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "metrics": {
            "token_weighted_nll": token_weighted_nll,
            "capability_macro_nll": capability_macro_nll,
            "per_capability": per_capability,
            "per_record": per_record,
            "per_example": per_example,
        },
        "safety_gate": {
            "status": "unreviewed",
            "required_records": config["checkpoint_selection"][
                "required_safety_records"
            ],
        },
        "runtime_check": runtime,
        "data": data_snapshot,
        "environment": environment_snapshot(),
        "hardware": _hardware_snapshot(torch),
        "determinism": {
            "seed": seed,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32)
            if torch.cuda.is_available()
            else None,
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)
            if torch.cuda.is_available()
            else None,
        },
    }
    write_json(output_path, result)
    del model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument("--allow-non-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        workspace_path(args.config),
        workspace_path(args.adapter),
        workspace_path(args.output),
        allow_version_mismatch=args.allow_version_mismatch,
        allow_non_cuda=args.allow_non_cuda,
    )
    print(
        "Validation capability_macro_nll={:.6f}".format(
            result["metrics"]["capability_macro_nll"]
        )
    )


if __name__ == "__main__":
    main()
