#!/usr/bin/env python3
"""Train the registered Hu Tao LoRA/QLoRA experiment on one CUDA GPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scripts.module_c.common import (
    CAPABILITIES,
    ExperimentError,
    canonical_tokenizer_identity,
    environment_snapshot,
    load_json,
    load_jsonl,
    package_versions,
    sha256_file,
    verify_sha256,
    workspace_path,
    write_json,
)
from scripts.module_c.tokenization import (
    CompletionOnlyDataCollator,
    tokenization_summary,
    tokenize_completion_example,
)


DEFAULT_CONFIG = "configs/module_c/hutao_qwen3_1p7b_lora_bf16.json"
REGISTERED_MODEL_NAME = "Qwen/Qwen3-1.7B"
REGISTERED_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
REGISTERED_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
REGISTERED_TRAINABLE_PARAMETERS = 3211264
REGISTERED_ADAPTER_TENSORS = 112
REGISTERED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
REGISTERED_CUBLAS_WORKSPACE_BYTES = 4096 * 1024 * 8
COMMON_CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state.pth",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_exact_requirements(path: Path) -> Dict[str, str]:
    requirements: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            nested = path.parent / line[3:].strip()
            requirements.update(_read_exact_requirements(nested))
            continue
        if "==" not in line:
            raise ExperimentError(
                "Every runtime requirement must be exact (name==version): {}".format(
                    line
                )
            )
        name, version = line.split("==", 1)
        requirements[name.strip()] = version.strip()
    return requirements


def _runtime_version_matches(name: str, expected: str, actual: Optional[str]) -> bool:
    """Accept a CUDA/ROCm local build suffix for an otherwise exact Torch pin."""
    if actual is None:
        return False
    if actual == expected:
        return True
    return name == "torch" and actual.split("+", 1)[0] == expected


def verify_runtime(config: Mapping[str, Any], allow_mismatch: bool) -> Dict[str, Any]:
    runtime = config["runtime"]
    expected_python = runtime["python_major_minor"]
    actual_python = "{}.{}".format(sys.version_info.major, sys.version_info.minor)
    requirements_path = workspace_path(runtime["requirements_file"])
    lock_path = workspace_path(runtime["lock_file"])
    if not lock_path.is_file():
        raise ExperimentError(
            "Registered transitive lock file is missing: {}".format(lock_path)
        )
    verify_sha256(requirements_path, runtime["requirements_sha256"])
    verify_sha256(lock_path, runtime["lock_sha256"])
    expected_packages = _read_exact_requirements(requirements_path)
    expected_locked_packages = _read_exact_requirements(lock_path)
    actual_packages = package_versions(tuple(expected_packages))
    actual_locked_packages = package_versions(tuple(expected_locked_packages))
    imported_modules: Dict[str, Dict[str, Optional[str]]] = {}
    mismatches = []
    if actual_python != expected_python:
        mismatches.append(
            "python: expected {}, got {}".format(expected_python, actual_python)
        )
    for name, expected in sorted(expected_locked_packages.items()):
        actual = actual_locked_packages.get(name)
        if actual != expected:
            mismatches.append(
                "locked {}: expected {}, got {}".format(
                    name, expected, actual or "missing"
                )
            )
    for name, expected in sorted(expected_packages.items()):
        locked = expected_locked_packages.get(name)
        if not _runtime_version_matches(name, expected, locked):
            mismatches.append(
                "direct requirement {}=={} is absent from or differs in the "
                "transitive lock ({})".format(name, expected, locked or "missing")
            )
    for name, expected in sorted(expected_packages.items()):
        actual = actual_packages.get(name)
        if not _runtime_version_matches(name, expected, actual):
            mismatches.append(
                "{}: expected {}, got {}".format(name, expected, actual or "missing")
            )
        try:
            module = importlib.import_module(name.replace("-", "_"))
            module_version = getattr(module, "__version__", None)
            module_file = getattr(module, "__file__", None)
            imported_modules[name] = {
                "version": str(module_version) if module_version is not None else None,
                "file": str(module_file) if module_file is not None else None,
                "error": None,
            }
            if not _runtime_version_matches(
                name,
                expected,
                str(module_version) if module_version is not None else None,
            ):
                mismatches.append(
                    "{} import: expected {}, got {}".format(
                        name, expected, module_version or "missing __version__"
                    )
                )
        except Exception as exc:  # Import failures are evidence, not a hard crash.
            imported_modules[name] = {
                "version": None,
                "file": None,
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
            mismatches.append("{} import failed: {}".format(name, exc))

    try:
        pip_check_process = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            universal_newlines=True,
        )
        pip_check = {
            "returncode": int(pip_check_process.returncode),
            "output": pip_check_process.stdout.strip(),
        }
    except OSError as exc:
        pip_check = {"returncode": None, "output": str(exc)}
    if pip_check["returncode"] != 0:
        mismatches.append("pip check failed: {}".format(pip_check["output"]))
    if mismatches and not allow_mismatch:
        raise ExperimentError(
            "Runtime does not match the registered lock:\n- " + "\n- ".join(mismatches)
        )
    return {
        "requirements_file": str(requirements_path.relative_to(workspace_path("."))),
        "requirements_sha256": sha256_file(requirements_path),
        "lock_file": str(lock_path.relative_to(workspace_path("."))),
        "lock_file_sha256": sha256_file(lock_path),
        "expected_python": expected_python,
        "actual_python": actual_python,
        "expected_packages": expected_packages,
        "actual_packages": actual_packages,
        "expected_locked_packages": expected_locked_packages,
        "actual_locked_packages": actual_locked_packages,
        "locked_package_count": len(expected_locked_packages),
        "imported_modules": imported_modules,
        "pip_check": pip_check,
        "mismatches": mismatches,
        "mismatch_override": bool(mismatches and allow_mismatch),
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "1.0":
        raise ExperimentError("Unsupported config schema_version")
    method = config.get("method", {}).get("name")
    if method not in {"lora", "qlora"}:
        raise ExperimentError("method.name must be lora or qlora")
    if config["data"].get("packing") is not False:
        raise ExperimentError("This registered experiment requires packing=false")
    if int(config["data"].get("max_length", 0)) < 1:
        raise ExperimentError("data.max_length must be positive")
    if config.get("model", {}).get("dtype") not in {"float16", "bfloat16"}:
        raise ExperimentError("model.dtype must be float16 or bfloat16")
    model_config = config.get("model", {})
    if (
        model_config.get("name") != REGISTERED_MODEL_NAME
        or model_config.get("revision") != REGISTERED_MODEL_REVISION
    ):
        raise ExperimentError(
            "Active experiment must use {} at revision {}; got {!r} at {!r}".format(
                REGISTERED_MODEL_NAME,
                REGISTERED_MODEL_REVISION,
                model_config.get("name"),
                model_config.get("revision"),
            )
        )
    if model_config.get("chat_template_kwargs") != REGISTERED_CHAT_TEMPLATE_KWARGS:
        raise ExperimentError(
            "Qwen3 training requires chat_template_kwargs={!r}".format(
                REGISTERED_CHAT_TEMPLATE_KWARGS
            )
        )
    targets = config["method"]["lora"].get("target_modules")
    if not isinstance(targets, list) or not targets:
        raise ExperimentError("LoRA target_modules must be a non-empty list")
    if config["method"]["lora"].get("adapter_dtype") != "float32":
        raise ExperimentError("This reproducible workflow requires FP32 LoRA adapters")
    if (
        int(config["method"]["lora"].get("expected_trainable_parameters", -1))
        != REGISTERED_TRAINABLE_PARAMETERS
    ):
        raise ExperimentError(
            "Qwen3-1.7B q_proj/v_proj LoRA requires expected_trainable_parameters={}"
            .format(REGISTERED_TRAINABLE_PARAMETERS)
        )
    if method == "qlora" and not isinstance(config["method"].get("quantization"), dict):
        raise ExperimentError("QLoRA config needs method.quantization")
    if int(config["runtime"].get("world_size", 0)) != 1:
        raise ExperimentError(
            "This registered experiment requires runtime.world_size=1"
        )
    if int(config["runtime"].get("visible_cuda_devices", 0)) != 1:
        raise ExperimentError(
            "This registered experiment requires exactly one visible CUDA device"
        )


def verify_training_data(config: Mapping[str, Any]) -> Dict[str, Any]:
    data_config = config["data"]
    source_dir = workspace_path(data_config["source_dir"])
    derived_dir = workspace_path(data_config["derived_dir"])
    derived_manifest_path = derived_dir / "manifest.json"
    verify_sha256(derived_manifest_path, data_config["expected_manifest_sha256"])
    derived_manifest = load_json(derived_manifest_path)
    result: Dict[str, Any] = {
        "source": {},
        "derived": {},
        "manifest": {
            "path": str(derived_manifest_path.relative_to(workspace_path("."))),
            "sha256": sha256_file(derived_manifest_path),
        },
    }

    # Deliberately do not parse the held-out test split in the training process.
    for split in ("train", "validation"):
        source_path = source_dir / "{}.jsonl".format(split)
        verify_sha256(source_path, data_config["expected_sha256"][split])
        result["source"][split] = {
            "path": str(source_path.relative_to(workspace_path("."))),
            "sha256": sha256_file(source_path),
        }

        derived_path = derived_dir / "{}.jsonl".format(split)
        expected_derived_hash = data_config["expected_derived_sha256"][split]
        verify_sha256(derived_path, expected_derived_hash)
        manifest_derived_hash = derived_manifest["splits"][split]["derived_sha256"]
        if manifest_derived_hash != expected_derived_hash:
            raise ExperimentError(
                "Derived manifest hash for {} differs from the registered config".format(
                    split
                )
            )
        records = load_jsonl(derived_path)
        expected_count = int(data_config["expected_derived_examples"][split])
        if len(records) != expected_count:
            raise ExperimentError(
                "{} contains {} examples; expected {}".format(
                    derived_path, len(records), expected_count
                )
            )
        result["derived"][split] = {
            "path": str(derived_path.relative_to(workspace_path("."))),
            "sha256": sha256_file(derived_path),
            "examples": len(records),
        }
    return result


def _balanced_smoke_subset(
    examples: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    for example in examples:
        capability = example["metadata"]["capability"]
        if capability not in selected:
            selected[capability] = dict(example)
    missing = sorted(set(CAPABILITIES) - set(selected))
    if missing:
        raise ExperimentError(
            "Smoke subset is missing capabilities: {}".format(missing)
        )
    return [selected[capability] for capability in CAPABILITIES]


def _tokenize_rows(
    examples: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
    chat_template_kwargs: Mapping[str, Any],
) -> Tuple[List[Dict[str, List[int]]], Dict[str, Any]]:
    training_rows: List[Dict[str, List[int]]] = []
    audit_rows: List[Dict[str, Any]] = []
    for example in examples:
        row = tokenize_completion_example(
            example,
            tokenizer,
            max_length,
            chat_template_kwargs=chat_template_kwargs,
        )
        training_rows.append(
            {
                "input_ids": row["input_ids"],
                "attention_mask": row["attention_mask"],
                "labels": row["labels"],
            }
        )
        audit_rows.append(row)
    return training_rows, tokenization_summary(audit_rows)


def _torch_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ExperimentError("Unsupported dtype: {}".format(name))
    return mapping[name]


def _hardware_snapshot(torch: Any) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        snapshot.update(
            {
                "device_index": index,
                "device_name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return snapshot


def _model_load_kwargs(
    config: Mapping[str, Any], torch: Any, quantized: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build immutable Base-model load options.

    QLoRA training/reload defaults to the registered 4-bit form. Callers may
    request ``quantized=False`` for the common BF16 deployment/evaluation form.
    """
    model_config = config["model"]
    kwargs: Dict[str, Any] = {
        "revision": model_config["revision"],
        "dtype": _torch_dtype(torch, model_config["dtype"]),
        "low_cpu_mem_usage": True,
        "attn_implementation": model_config.get("attention_implementation", "eager"),
    }
    use_quantization = (
        config["method"]["name"] == "qlora" if quantized is None else quantized
    )
    if use_quantization:
        if config["method"]["name"] != "qlora":
            raise ExperimentError("4-bit loading requires a QLoRA configuration")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ExperimentError(
                "QLoRA requires Transformers bitsandbytes support"
            ) from exc
        quant = config["method"]["quantization"]
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=bool(quant["load_in_4bit"]),
            bnb_4bit_quant_type=quant["quant_type"],
            bnb_4bit_use_double_quant=bool(quant["use_double_quant"]),
            bnb_4bit_compute_dtype=_torch_dtype(torch, quant["compute_dtype"]),
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    return kwargs


def _count_trainable_parameters(model: Any) -> Dict[str, Any]:
    trainable = 0
    total = 0
    unexpected_trainable: List[str] = []
    trainable_names: List[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
            if "lora_" not in name:
                unexpected_trainable.append(name)
    if unexpected_trainable:
        raise ExperimentError(
            "Non-LoRA parameters are trainable: {}".format(unexpected_trainable[:20])
        )
    return {
        "trainable": trainable,
        "total": total,
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
        "trainable_tensor_names": trainable_names,
    }


def _normalize_trainable_adapter_dtype(
    model: Any, torch: Any, dtype_name: str
) -> Dict[str, Any]:
    """Freeze a stable adapter dtype after Trainer has applied its QLoRA policy.

    TRL converts trainable QLoRA parameters to bfloat16 during construction,
    while PEFT checkpoint loading defaults to float32.  Normalizing the small
    adapter to float32 here makes fresh and resumed runs use the same dtype.
    """
    target_dtype = _torch_dtype(torch, dtype_name)
    names: List[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" not in name:
            raise ExperimentError(
                "Unexpected non-LoRA trainable parameter: {}".format(name)
            )
        parameter.data = parameter.data.to(target_dtype)
        names.append(name)
    if not names:
        raise ExperimentError("No trainable adapter parameters were found")
    dtype_counts: Dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            dtype_key = str(parameter.dtype)
            dtype_counts[dtype_key] = dtype_counts.get(dtype_key, 0) + 1
    expected_dtype_key = str(target_dtype)
    if set(dtype_counts) != {expected_dtype_key}:
        raise ExperimentError(
            "Trainable adapter dtype normalization failed: {}".format(dtype_counts)
        )
    return {
        "registered_dtype": dtype_name,
        "resolved_dtype": expected_dtype_key,
        "trainable_tensors": len(names),
        "dtype_tensor_counts": dtype_counts,
    }


def _assert_lora_adapter_dtype(model: Any, expected_dtype: Any) -> Dict[str, Any]:
    dtype_counts: Dict[str, int] = {}
    adapter_tensors = 0
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        adapter_tensors += 1
        dtype_key = str(parameter.dtype)
        dtype_counts[dtype_key] = dtype_counts.get(dtype_key, 0) + 1
    if adapter_tensors < 1 or set(dtype_counts) != {str(expected_dtype)}:
        raise ExperimentError(
            "LoRA adapter dtype differs from {}: {}".format(
                expected_dtype, dtype_counts
            )
        )
    return {
        "adapter_tensors": adapter_tensors,
        "dtype_tensor_counts": dtype_counts,
    }


def _snapshot_peft_adapter_state(
    model: Any,
    get_peft_model_state_dict: Any,
    torch: Any,
    expected_values: int,
    expected_tensors: int,
) -> Dict[str, Any]:
    """Clone the portable FP32 adapter state for a lossless reload audit."""
    state = get_peft_model_state_dict(model)
    if not isinstance(state, Mapping) or not state:
        raise ExperimentError("PEFT returned an empty or invalid adapter state dict")
    snapshot: Dict[str, Any] = {}
    value_count = 0
    for name in sorted(state):
        tensor = state[name]
        if "lora_" not in name or not hasattr(tensor, "detach"):
            raise ExperimentError(
                "Unexpected value in the portable adapter state: {}".format(name)
            )
        detached = tensor.detach().cpu().clone()
        if detached.dtype != torch.float32:
            raise ExperimentError(
                "Portable adapter tensor {} has dtype {}, expected torch.float32"
                .format(name, detached.dtype)
            )
        value_count += int(detached.numel())
        snapshot[name] = detached
    if len(snapshot) != int(expected_tensors):
        raise ExperimentError(
            "Portable adapter state has {} tensors, expected {}".format(
                len(snapshot), expected_tensors
            )
        )
    if value_count != int(expected_values):
        raise ExperimentError(
            "Portable adapter state has {} values, expected {}".format(
                value_count, expected_values
            )
        )
    return snapshot


def _assert_exact_adapter_state_roundtrip(
    reference: Mapping[str, Any],
    reloaded: Mapping[str, Any],
    torch: Any,
    expected_values: int,
    expected_tensors: int,
) -> Dict[str, Any]:
    """Require safetensors save/load to preserve every FP32 LoRA value exactly."""
    reference_keys = set(reference)
    reloaded_keys = set(reloaded)
    if reference_keys != reloaded_keys:
        raise ExperimentError(
            "Reloaded adapter state keys differ; missing={}, extra={}".format(
                sorted(reference_keys - reloaded_keys)[:20],
                sorted(reloaded_keys - reference_keys)[:20],
            )
        )
    if len(reference_keys) != int(expected_tensors):
        raise ExperimentError(
            "Reloaded adapter state has {} tensors, expected {}".format(
                len(reference_keys), expected_tensors
            )
        )
    value_count = 0
    for name in sorted(reference_keys):
        expected = reference[name]
        actual = reloaded[name]
        if expected.shape != actual.shape:
            raise ExperimentError(
                "Reloaded adapter tensor {} shape is {}, expected {}".format(
                    name, tuple(actual.shape), tuple(expected.shape)
                )
            )
        if expected.dtype != actual.dtype:
            raise ExperimentError(
                "Reloaded adapter tensor {} dtype is {}, expected {}".format(
                    name, actual.dtype, expected.dtype
                )
            )
        if not bool(torch.equal(actual, expected)):
            maximum = float(
                (actual.float() - expected.float()).abs().max().item()
            )
            raise ExperimentError(
                "Reloaded adapter tensor {} differs from the saved FP32 state; "
                "maximum absolute difference={}".format(name, maximum)
            )
        value_count += int(actual.numel())
    if value_count != int(expected_values):
        raise ExperimentError(
            "Reloaded adapter state has {} values, expected {}".format(
                value_count, expected_values
            )
        )
    return {
        "status": "pass",
        "tensor_count": len(reference_keys),
        "value_count": value_count,
        "dtype": "torch.float32",
        "comparison_rtol": 0.0,
        "comparison_atol": 0.0,
    }


def _assert_saved_peft_config(
    saved: Any, config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind the serialized adapter scaling and Base identity to the run config."""
    lora = config["method"]["lora"]
    task_type = getattr(getattr(saved, "task_type", None), "value", None)
    if task_type is None:
        task_type = str(getattr(saved, "task_type", ""))
        if task_type.startswith("TaskType."):
            task_type = task_type.split(".", 1)[1]
    actual = {
        "base_model_name_or_path": getattr(saved, "base_model_name_or_path", None),
        "revision": getattr(saved, "revision", None),
        "rank": int(getattr(saved, "r", -1)),
        "alpha": int(getattr(saved, "lora_alpha", -1)),
        "dropout": float(getattr(saved, "lora_dropout", -1.0)),
        "target_modules": sorted(getattr(saved, "target_modules", None) or []),
        "bias": getattr(saved, "bias", None),
        "task_type": task_type,
        "inference_mode": bool(getattr(saved, "inference_mode", False)),
    }
    expected = {
        "base_model_name_or_path": config["model"]["name"],
        "revision": config["model"]["revision"],
        "rank": int(lora["rank"]),
        "alpha": int(lora["alpha"]),
        "dropout": float(lora["dropout"]),
        "target_modules": sorted(lora["target_modules"]),
        "bias": lora["bias"],
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
    }
    if actual != expected:
        raise ExperimentError(
            "Saved PEFT config differs from the registered adapter: actual={!r}, "
            "expected={!r}".format(actual, expected)
        )
    return {"status": "pass", **actual}


def _unwrap_trained_model_for_inference(trainer: Any) -> Any:
    """Remove Accelerate's AMP forward wrapper before clean-model comparison."""
    accelerator = getattr(trainer, "accelerator", None)
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    if not callable(unwrap_model):
        raise ExperimentError("Trainer has no Accelerator unwrap_model API")
    model = unwrap_model(
        trainer.model,
        keep_fp32_wrapper=False,
        keep_torch_compile=False,
    )
    if hasattr(model, "_original_forward"):
        raise ExperimentError(
            "Accelerate's mixed-precision forward wrapper was not removed"
        )
    return model


def _logit_reload_diagnostics(
    reference: Any, reloaded: Any, torch: Any
) -> Dict[str, Any]:
    """Describe BF16 forward drift; adapter state and token IDs are hard gates."""
    if reference.shape != reloaded.shape:
        raise ExperimentError(
            "Reloaded logits shape is {}, expected {}".format(
                tuple(reloaded.shape), tuple(reference.shape)
            )
        )
    if not bool(torch.isfinite(reference).all().item()) or not bool(
        torch.isfinite(reloaded).all().item()
    ):
        raise ExperimentError("Reference or reloaded smoke logits are non-finite")
    difference = (reloaded - reference).abs()
    close = torch.isclose(reference, reloaded, rtol=1.6e-2, atol=1e-5)
    reference_top1 = int(reference.argmax().item())
    reloaded_top1 = int(reloaded.argmax().item())
    return {
        "values": int(reference.numel()),
        "maximum_absolute_difference": float(difference.max().item()),
        "mean_absolute_difference": float(difference.mean().item()),
        "bfloat16_default_rtol": 1.6e-2,
        "bfloat16_default_atol": 1e-5,
        "bfloat16_close_fraction": float(close.float().mean().item()),
        "reference_top1_token_id": reference_top1,
        "reloaded_top1_token_id": reloaded_top1,
        "top1_match": reference_top1 == reloaded_top1,
    }


def _assert_trainable_adapter_finite(model: Any, torch: Any) -> Dict[str, Any]:
    """Reject an adapter as soon as any trainable LoRA value is NaN or Inf."""
    checked = 0
    bad_names: List[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" not in name:
            raise ExperimentError(
                "Unexpected non-LoRA trainable parameter: {}".format(name)
            )
        checked += int(parameter.numel())
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            bad_names.append(name)
    if checked < 1:
        raise ExperimentError("No trainable adapter values were found")
    if bad_names:
        raise ExperimentError(
            "Training produced non-finite LoRA weights: {}".format(bad_names[:20])
        )
    return {"checked_values": checked, "nonfinite_tensors": 0}


def _assert_trainable_adapter_gradients_finite(
    model: Any, torch: Any
) -> Dict[str, Any]:
    """Reject non-finite LoRA gradients before optimizer.step mutates weights."""
    checked = 0
    bad_names: List[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if "lora_" not in name:
            raise ExperimentError(
                "Unexpected non-LoRA trainable gradient: {}".format(name)
            )
        checked += int(parameter.grad.numel())
        if not bool(torch.isfinite(parameter.grad.detach()).all().item()):
            bad_names.append(name)
    if checked < 1:
        raise ExperimentError("No trainable adapter gradients were found before step")
    if bad_names:
        raise ExperimentError(
            "Backward produced non-finite LoRA gradients: {}".format(bad_names[:20])
        )
    return {"checked_gradient_values": checked, "nonfinite_gradient_tensors": 0}


def _assert_logged_metrics_finite(
    values: Mapping[str, Any], label: str = "training metrics",
) -> None:
    """Fail on non-finite optimization metrics, including v5 string values."""
    metric_fragments = (
        "loss",
        "grad_norm",
        "entropy",
        "accuracy",
        "learning_rate",
    )
    bad: Dict[str, str] = {}
    for name, value in values.items():
        if not any(fragment in name for fragment in metric_fragments):
            continue
        if isinstance(value, bool) or value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            bad[name] = str(value)
    if bad:
        raise ExperimentError("{} contain non-finite values: {}".format(label, bad))


def _set_registered_cublas_workspace() -> str:
    """Restore the deterministic workspace after Trainer changes it to :16:8."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = REGISTERED_CUBLAS_WORKSPACE_CONFIG
    actual = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual != REGISTERED_CUBLAS_WORKSPACE_CONFIG:
        raise ExperimentError(
            "Could not register CUBLAS_WORKSPACE_CONFIG={}".format(
                REGISTERED_CUBLAS_WORKSPACE_CONFIG
            )
        )
    return actual


def _configure_registered_cublas_workspace(torch: Any) -> Dict[str, Any]:
    """Set both the deterministic environment and PyTorch's live override."""
    environment_value = _set_registered_cublas_workspace()
    result: Dict[str, Any] = {
        "environment_value": environment_value,
        "requested_size_bytes": REGISTERED_CUBLAS_WORKSPACE_BYTES,
        "api_override_available": False,
        "actual_size_bytes": None,
    }
    if not torch.cuda.is_available():
        return result
    workspace_api = getattr(torch.backends.cuda, "cublas_workspace_size", None)
    if not callable(workspace_api):
        # Keep diagnostic runs on older, explicitly overridden runtimes usable;
        # the exact PyTorch 2.13 lock provides this API.
        return result
    actual_size = int(workspace_api(REGISTERED_CUBLAS_WORKSPACE_BYTES))
    if actual_size != REGISTERED_CUBLAS_WORKSPACE_BYTES:
        raise ExperimentError(
            "PyTorch registered a cuBLAS workspace of {} bytes; expected {}".format(
                actual_size, REGISTERED_CUBLAS_WORKSPACE_BYTES
            )
        )
    result.update(
        {"api_override_available": True, "actual_size_bytes": actual_size,}
    )
    return result


def _pretrain_numerical_audit(
    trainer: Any,
    training_rows: Sequence[Mapping[str, Sequence[int]]],
    collator: Any,
    torch: Any,
) -> Dict[str, Any]:
    """Run real completion-only batches before allowing optimizer updates.

    This catches low-precision forward failures before a checkpoint containing
    NaN adapter values can be written. At most one example per registered
    capability is checked so the gate remains cheap for both run modes.
    """
    if not training_rows:
        raise ExperimentError("Numerical audit received an empty training split")
    batch_size = max(1, int(trainer.args.per_device_train_batch_size))
    audit_count = min(len(CAPABILITIES), len(training_rows))
    model = trainer.model
    device = next(model.parameters()).device
    was_training = bool(model.training)
    batches: List[Dict[str, Any]] = []
    model.eval()
    try:
        for start in range(0, audit_count, batch_size):
            stop = min(start + batch_size, audit_count)
            cpu_batch = collator(list(training_rows[start:stop]))
            labels = cpu_batch["labels"]
            attention_mask = cpu_batch["attention_mask"]
            input_ids = cpu_batch["input_ids"]
            shifted_supervised = (labels[:, 1:] != -100).sum(dim=1)
            if not bool((shifted_supervised > 0).all().item()):
                raise ExperimentError(
                    "Numerical audit found an example with no shifted supervision"
                )
            if not bool((labels[attention_mask == 0] == -100).all().item()):
                raise ExperimentError(
                    "Numerical audit found a padding token included in the loss"
                )
            supervised_mask = labels != -100
            if not bool(
                (labels[supervised_mask] == input_ids[supervised_mask]).all().item()
            ):
                raise ExperimentError(
                    "Numerical audit found labels that differ from supervised input ids"
                )

            device_batch = {
                name: tensor.to(device) for name, tensor in cpu_batch.items()
            }
            # In Transformers v5, CUDA mixed precision is owned by Accelerate;
            # Trainer.compute_loss_context_manager() is a null context on CUDA.
            # Use the same autocast policy that wraps the real training forward.
            with torch.no_grad(), trainer.accelerator.autocast():
                outputs = model(**device_batch, use_cache=False)
            loss = getattr(outputs, "loss", None)
            logits = getattr(outputs, "logits", None)
            if loss is None or logits is None:
                raise ExperimentError(
                    "Numerical audit requires model outputs containing loss and logits"
                )
            loss_finite = bool(torch.isfinite(loss).all().item())
            # TRL computes entropy for every shifted position before applying
            # the -100 label mask. Check that same full shifted-logit domain so
            # masked prompt/padding positions cannot leak NaN through NaN * 0.
            shifted_logits = logits[:, :-1, :]
            finite_mask = torch.isfinite(shifted_logits)
            finite_logits = int(finite_mask.sum().item())
            shifted_logit_values = int(shifted_logits.numel())
            nonfinite_logits = shifted_logit_values - finite_logits
            logits_finite = nonfinite_logits == 0
            batch_result = {
                "row_start": start,
                "row_stop": stop,
                "shifted_supervised_tokens": [
                    int(value) for value in shifted_supervised.tolist()
                ],
                "loss": float(loss.detach().float().cpu().item()),
                "loss_finite": loss_finite,
                "logits_dtype": str(logits.dtype),
                "shifted_logit_values": shifted_logit_values,
                "nonfinite_shifted_logits": nonfinite_logits,
            }
            batches.append(batch_result)
            del (
                outputs,
                loss,
                logits,
                shifted_logits,
                finite_mask,
                device_batch,
            )
            if not loss_finite or not logits_finite:
                model_dtype = "bfloat16" if trainer.args.bf16 else "float16"
                raise ExperimentError(
                    "Pre-training numerical audit failed for model dtype {}: {}. "
                    "Do not use the resulting adapter; on Ampere GPUs use the "
                    "registered BF16 experiment.".format(model_dtype, batch_result)
                )
    finally:
        model.train(was_training)
    return {
        "status": "pass",
        "examples": audit_count,
        "batches": batches,
    }


def _validate_lora_coverage(
    model: Any, target_modules: Sequence[str]
) -> Dict[str, Any]:
    expected_layers = int(model.config.num_hidden_layers)
    coverage: Dict[str, set] = {target: set() for target in target_modules}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or ".lora_A." not in name:
            continue
        for target in target_modules:
            marker = ".{}.lora_A.".format(target)
            if marker in name:
                match = name.split(".layers.", 1)
                if len(match) != 2:
                    raise ExperimentError("Cannot identify layer in {}".format(name))
                layer_text = match[1].split(".", 1)[0]
                coverage[target].add(int(layer_text))
    result = {target: sorted(layers) for target, layers in coverage.items()}
    for target, layers in result.items():
        if len(layers) != expected_layers:
            raise ExperimentError(
                "LoRA target {} covers {} layers; expected {}".format(
                    target, len(layers), expected_layers
                )
            )
    return {"expected_layers": expected_layers, "layers_by_target": result}


def _tokenizer_snapshot(
    tokenizer: Any, chat_template_kwargs: Mapping[str, Any] = None
) -> Dict[str, Any]:
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template and hasattr(tokenizer, "get_chat_template"):
        chat_template = tokenizer.get_chat_template()
    if not isinstance(chat_template, str) or not chat_template:
        raise ExperimentError("Tokenizer has no chat template")
    return canonical_tokenizer_identity({
        "chat_template_sha256": _text_sha256(chat_template),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "padding_side": tokenizer.padding_side,
        "chat_template_kwargs": dict(chat_template_kwargs or {}),
    })


def _assert_model_revision(model: Any, expected_revision: str) -> str:
    """Require the Hub loader to resolve the immutable registered commit."""
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", expected_revision):
        raise ExperimentError("Registered Hub revision must be a 40-character commit")
    if resolved_revision != expected_revision:
        raise ExperimentError(
            "Resolved Base commit is {}, expected {}".format(
                resolved_revision, expected_revision
            )
        )
    return resolved_revision


def _ensure_output_and_resume_safety(
    output_dir: Path,
    resume_from_checkpoint: Optional[str],
    config_sha256: str,
    data_snapshot: Mapping[str, Any],
    require_grad_scaler: bool,
    expected_mode: str,
) -> Optional[str]:
    if resume_from_checkpoint is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ExperimentError(
                "Output directory is not empty; refuse to overwrite a prior run: {}".format(
                    output_dir
                )
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        return None

    checkpoint = workspace_path(resume_from_checkpoint).resolve()
    if not checkpoint.is_dir():
        raise ExperimentError("Resume checkpoint does not exist: {}".format(checkpoint))
    checkpoint_match = re.fullmatch(r"checkpoint-([0-9]+)", checkpoint.name)
    if checkpoint.parent != output_dir.resolve() or checkpoint_match is None:
        raise ExperimentError(
            "Resume path must be a direct checkpoint-N child of the output_dir"
        )
    previous_manifest = load_json(output_dir / "run_manifest.json")
    if previous_manifest.get("mode") != expected_mode:
        raise ExperimentError("Resume mode differs from the original run")
    if previous_manifest.get("status") not in {
        "initialising",
        "training",
        "failed",
    }:
        raise ExperimentError(
            "Resume requires an incomplete or failed run, got status {!r}".format(
                previous_manifest.get("status")
            )
        )
    if previous_manifest.get("config_sha256") != config_sha256:
        raise ExperimentError("Resume config hash differs from the original run")
    if previous_manifest.get("data") != data_snapshot:
        raise ExperimentError("Resume data snapshot differs from the original run")
    checkpoint_artifact_snapshot(checkpoint, require_grad_scaler)
    trainer_state = load_json(checkpoint / "trainer_state.json")
    expected_step = int(checkpoint_match.group(1))
    if trainer_state.get("global_step") != expected_step:
        raise ExperimentError(
            "Resume checkpoint name differs from trainer_state.global_step"
        )
    later_steps = []
    for candidate in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-([0-9]+)", candidate.name)
        if match and candidate.is_dir() and int(match.group(1)) > expected_step:
            later_steps.append(int(match.group(1)))
    if later_steps:
        raise ExperimentError(
            "Refuse to resume from checkpoint-{} while later checkpoints exist: {}".format(
                expected_step, sorted(later_steps)
            )
        )
    return str(checkpoint)


def checkpoint_artifact_snapshot(
    checkpoint: Path, require_grad_scaler: bool
) -> Dict[str, str]:
    """Hash all state needed for an exact single-GPU Trainer resume."""
    required = list(COMMON_CHECKPOINT_FILES)
    if require_grad_scaler:
        required.append("scaler.pt")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise ExperimentError(
            "Checkpoint is incomplete; missing {} from {}".format(missing, checkpoint)
        )
    return {name: sha256_file(checkpoint / name) for name in sorted(required)}


def _build_sft_args(config: Mapping[str, Any], output_dir: Path, smoke: bool) -> Any:
    from trl import SFTConfig

    data = config["data"]
    training = config["training"]
    dtype = config["model"]["dtype"]
    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training["num_train_epochs"]),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "lr_scheduler_type": training["lr_scheduler_type"],
        # Transformers v5 folds warmup_ratio into warmup_steps: a float below
        # one is still interpreted as a ratio of total optimizer steps.
        "warmup_steps": float(training["warmup_ratio"]),
        "max_grad_norm": float(training["max_grad_norm"]),
        "logging_strategy": "steps",
        "logging_steps": int(training["logging_steps"]),
        # Do not turn NaN/Inf loss into a misleading historical average.
        "logging_nan_inf_filter": False,
        "eval_strategy": training["eval_strategy"],
        "save_strategy": training["save_strategy"],
        "save_total_limit": int(training["save_total_limit"]),
        "seed": int(training["seed"]),
        "data_seed": int(training["data_seed"]),
        "full_determinism": bool(training["full_determinism"]),
        "dataloader_num_workers": int(training["dataloader_num_workers"]),
        "dataloader_drop_last": bool(training["dataloader_drop_last"]),
        "auto_find_batch_size": bool(training["auto_find_batch_size"]),
        "tf32": bool(training["tf32"]),
        "gradient_checkpointing": bool(training["gradient_checkpointing"]),
        "report_to": list(training["report_to"]),
        "fp16": dtype == "float16",
        "bf16": dtype == "bfloat16",
        "packing": False,
        "max_length": int(data["max_length"]),
        "pad_to_multiple_of": int(data["pad_to_multiple_of"]),
        "loss_type": "nll",
        "assistant_only_loss": False,
        "completion_only_loss": False,
        "padding_free": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "save_only_model": False,
        "ignore_data_skip": False,
        "restore_callback_states_from_checkpoint": True,
        "optim": training.get("optimizer", "adamw_torch"),
    }
    if smoke:
        kwargs.update(
            {
                "output_dir": str(output_dir),
                "num_train_epochs": 1.0,
                "max_steps": 2,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "eval_strategy": "steps",
                "eval_steps": 2,
                "save_strategy": "steps",
                "save_steps": 2,
                "save_total_limit": 1,
            }
        )
    # Transformers v5 removed TrainingArguments.logging_dir. Its TensorBoard
    # callback reads this environment variable when the trainer is created.
    if "tensorboard" in kwargs["report_to"]:
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(output_dir / "tensorboard")
    return SFTConfig(**kwargs)


def _generate_smoke_sample(
    model: Any,
    tokenizer: Any,
    prompt_messages: Sequence[Mapping[str, str]],
    max_new_tokens: int,
    chat_template_kwargs: Mapping[str, Any],
) -> Tuple[str, List[int]]:
    import torch

    model.eval()
    encoded = tokenizer.apply_chat_template(
        list(prompt_messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **dict(chat_template_kwargs)
    )
    encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    token_ids = [int(value) for value in new_tokens.detach().cpu().tolist()]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, token_ids


def _last_token_logits(
    model: Any,
    tokenizer: Any,
    prompt_messages: Sequence[Mapping[str, str]],
    chat_template_kwargs: Mapping[str, Any],
) -> Any:
    import torch

    encoded = tokenizer.apply_chat_template(
        list(prompt_messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **dict(chat_template_kwargs)
    )
    encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
    model.eval()
    with torch.inference_mode():
        logits = model(**encoded).logits[0, -1].float().cpu()
    return logits


def train(
    config_path: Path,
    smoke: bool,
    resume_from_checkpoint: Optional[str],
    allow_version_mismatch: bool,
    allow_non_cuda: bool,
) -> Dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _set_registered_cublas_workspace()

    config = load_json(config_path)
    if config.get("experiment_status") not in {"canonical", "fallback"}:
        raise ExperimentError(
            "This configuration is not active. Use "
            "configs/module_c/hutao_qwen3_1p7b_lora_bf16.json instead."
        )
    validate_config(config)
    seed = int(config["training"]["seed"])
    if bool(config["training"]["full_determinism"]) and os.environ.get(
        "PYTHONHASHSEED"
    ) != str(seed):
        raise ExperimentError(
            "Launch the canonical run with PYTHONHASHSEED={} set before Python starts".format(
                seed
            )
        )
    runtime_check = verify_runtime(config, allow_version_mismatch)
    data_snapshot = verify_training_data(config)

    try:
        import torch
        from datasets import Dataset
        from peft import (
            LoraConfig,
            PeftConfig,
            PeftModel,
            TaskType,
            get_peft_model_state_dict,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainerCallback,
            set_seed,
        )
        from trl import SFTTrainer
    except ImportError as exc:
        raise ExperimentError(
            "Training dependencies are missing; install the registered requirements file"
        ) from exc

    if config["runtime"].get("require_cuda", True) and not torch.cuda.is_available():
        if not allow_non_cuda:
            raise ExperimentError(
                "The registered main experiment requires CUDA; use --allow-non-cuda only "
                "for an explicitly non-canonical diagnostic run"
            )
    if config["method"]["name"] == "qlora" and not torch.cuda.is_available():
        raise ExperimentError(
            "The registered bitsandbytes QLoRA fallback requires CUDA"
        )
    if (
        config["model"]["dtype"] == "bfloat16"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        raise ExperimentError(
            "The registered BF16 experiment requires a CUDA GPU with BF16 support"
        )
    _configure_registered_cublas_workspace(torch)

    configured_world_size = int(config["runtime"]["world_size"])
    try:
        actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ExperimentError("WORLD_SIZE must be an integer") from exc
    if actual_world_size != configured_world_size:
        raise ExperimentError(
            "WORLD_SIZE is {}, expected {}".format(
                actual_world_size, configured_world_size
            )
        )
    if torch.cuda.is_available() and torch.cuda.device_count() != int(
        config["runtime"]["visible_cuda_devices"]
    ):
        raise ExperimentError(
            "Canonical training requires exactly one visible CUDA device; "
            "set CUDA_VISIBLE_DEVICES to one GPU"
        )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    set_seed(seed, deterministic=bool(config["training"]["full_determinism"]))

    canonical_output = workspace_path(config["training"]["output_dir"])
    output_dir = (
        canonical_output.with_name(canonical_output.name + "-smoke")
        if smoke
        else canonical_output
    )
    config_sha = sha256_file(config_path)
    resume_from_checkpoint = _ensure_output_and_resume_safety(
        output_dir,
        resume_from_checkpoint,
        config_sha,
        data_snapshot,
        require_grad_scaler=config["model"]["dtype"] == "float16",
        expected_mode="smoke" if smoke else "main",
    )
    resume_evidence = None
    if resume_from_checkpoint is not None:
        resume_path = Path(resume_from_checkpoint)
        resume_evidence = {
            "checkpoint_artifacts": checkpoint_artifact_snapshot(
                resume_path, require_grad_scaler=config["model"]["dtype"] == "float16",
            ),
            "previous_run_manifest_sha256": sha256_file(
                output_dir / "run_manifest.json"
            ),
        }

    manifest: Dict[str, Any] = {
        "status": "initialising",
        "started_at_utc": utc_now(),
        "mode": "smoke" if smoke else "main",
        "config_path": str(config_path.relative_to(workspace_path("."))),
        "config_sha256": config_sha,
        "config": config,
        "runtime_check": runtime_check,
        "environment": environment_snapshot(),
        "hardware": _hardware_snapshot(torch),
        "data": data_snapshot,
        "resume_from_checkpoint": resume_from_checkpoint,
        "resume_evidence": resume_evidence,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], revision=config["model"]["revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    chat_template_kwargs = dict(config["model"]["chat_template_kwargs"])
    tokenizer_info = _tokenizer_snapshot(tokenizer, chat_template_kwargs)
    manifest["tokenizer"] = tokenizer_info
    write_json(output_dir / "run_manifest.json", manifest)

    derived_dir = workspace_path(config["data"]["derived_dir"])
    train_examples = load_jsonl(derived_dir / "train.jsonl")
    validation_examples = load_jsonl(derived_dir / "validation.jsonl")
    if smoke:
        train_examples = _balanced_smoke_subset(train_examples)
        validation_examples = _balanced_smoke_subset(validation_examples)

    train_rows, train_token_summary = _tokenize_rows(
        train_examples,
        tokenizer,
        int(config["data"]["max_length"]),
        chat_template_kwargs,
    )
    validation_rows, validation_token_summary = _tokenize_rows(
        validation_examples,
        tokenizer,
        int(config["data"]["max_length"]),
        chat_template_kwargs,
    )
    token_audit = {
        "train": train_token_summary,
        "validation": validation_token_summary,
        "zero_truncation_required": True,
    }
    write_json(output_dir / "tokenization_audit.json", token_audit)

    train_dataset = Dataset.from_list(train_rows)
    validation_dataset = Dataset.from_list(validation_rows)
    collator = CompletionOnlyDataCollator(
        tokenizer.pad_token_id, int(config["data"]["pad_to_multiple_of"])
    )

    load_kwargs = _model_load_kwargs(config, torch)
    model = AutoModelForCausalLM.from_pretrained(config["model"]["name"], **load_kwargs)
    resolved_model_revision = _assert_model_revision(model, config["model"]["revision"])
    model.config.use_cache = False
    if config["method"]["name"] == "qlora":
        if getattr(model, "is_loaded_in_4bit", False) is not True:
            raise ExperimentError("QLoRA model did not load in 4-bit")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(
                config["training"]["gradient_checkpointing"]
            ),
        )

    lora = config["method"]["lora"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias=lora["bias"],
    )
    model = get_peft_model(
        model,
        lora_config,
        revision=config["model"]["revision"],
        autocast_adapter_dtype=False,
    )
    parameter_counts = _count_trainable_parameters(model)
    expected_trainable = int(lora["expected_trainable_parameters"])
    if parameter_counts["trainable"] != expected_trainable:
        raise ExperimentError(
            "Trainable parameter count is {}, expected {}".format(
                parameter_counts["trainable"], expected_trainable
            )
        )
    lora_coverage = _validate_lora_coverage(model, lora["target_modules"])
    sft_args = _build_sft_args(config, output_dir, smoke)
    registered_adapter_dtype = lora["adapter_dtype"]

    class RegisteredSFTTrainer(SFTTrainer):
        """Restore the registered adapter dtype after PEFT checkpoint loading."""

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(
                model, inputs, num_items_in_batch=num_items_in_batch
            )
            if not bool(torch.isfinite(loss).all().item()):
                raise ExperimentError(
                    "Training produced a non-finite loss before optimizer.step"
                )
            return loss

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            active_model = model if model is not None else self.model
            _normalize_trainable_adapter_dtype(
                active_model, torch, registered_adapter_dtype
            )
            self.resume_adapter_finite_audit = _assert_trainable_adapter_finite(
                active_model, torch
            )
            return result

    class FiniteMetricCallback(TrainerCallback):
        """Stop immediately instead of checkpointing NaN/Inf training state."""

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            active_model = kwargs.get("model", model)
            _assert_trainable_adapter_gradients_finite(active_model, torch)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            _assert_logged_metrics_finite(
                logs or {}, "training log at step {}".format(state.global_step)
            )
            return control

    trainer = RegisteredSFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[FiniteMetricCallback()],
    )
    cublas_workspace = None
    adapter_dtype = None
    try:
        # full_determinism rewrites the environment to :16:8; restore the
        # registered live 32 MiB workspace before the first model forward.
        cublas_workspace = _configure_registered_cublas_workspace(torch)
        adapter_dtype = _normalize_trainable_adapter_dtype(
            trainer.model, torch, registered_adapter_dtype
        )
        numerical_audit = _pretrain_numerical_audit(
            trainer, train_rows, collator, torch
        )
        numerical_audit["scope"] = (
            "fresh_initialization_before_checkpoint_load"
            if resume_from_checkpoint is not None
            else "fresh_training_model"
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "failure_stage": "pretrain_setup_or_numerical_audit",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "tokenization": token_audit,
                "parameters": parameter_counts,
                "adapter_dtype": adapter_dtype,
                "lora_coverage": lora_coverage,
                "determinism": {
                    "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                    "cublas_workspace": cublas_workspace,
                },
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    micro_batches = int(
        math.ceil(len(train_rows) / float(sft_args.per_device_train_batch_size))
    )
    expected_steps_per_epoch = int(
        math.ceil(micro_batches / float(sft_args.gradient_accumulation_steps))
    )
    manifest.update(
        {
            "status": "training",
            "tokenization": token_audit,
            "parameters": parameter_counts,
            "adapter_dtype": adapter_dtype,
            "lora_coverage": lora_coverage,
            "numerical_audit": numerical_audit,
            "determinism": {
                "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                "cublas_workspace": cublas_workspace,
                "deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "tf32_matmul": (
                    bool(torch.backends.cuda.matmul.allow_tf32)
                    if torch.cuda.is_available()
                    else None
                ),
                "tf32_cudnn": (
                    bool(torch.backends.cudnn.allow_tf32)
                    if torch.cuda.is_available()
                    else None
                ),
            },
            "resolved_model_revision": resolved_model_revision,
            "expected_optimizer_steps_per_epoch": expected_steps_per_epoch,
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)

    started = time.monotonic()
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        wall_time = time.monotonic() - started
        _assert_logged_metrics_finite(train_result.metrics, "final training metrics")
        trainer.save_state()
        # Accelerator wraps the training model's forward in a BF16 autocast
        # context. A freshly loaded PEFT model has no such wrapper, so comparing
        # them directly exercises different LoRA numerical paths. Remove the
        # wrapper before saving and before every clean-reload inference check.
        trained_model = _unwrap_trained_model_for_inference(trainer)
        adapter_finite = _assert_trainable_adapter_finite(trained_model, torch)
        final_adapter = output_dir / "adapter-final"
        _assert_lora_adapter_dtype(trained_model, torch.float32)
        reference_adapter_state = None
        if smoke:
            reference_adapter_state = _snapshot_peft_adapter_state(
                trained_model,
                get_peft_model_state_dict,
                torch,
                expected_trainable,
                REGISTERED_ADAPTER_TENSORS,
            )
        trained_model.save_pretrained(str(final_adapter), safe_serialization=True)
        tokenizer.save_pretrained(str(final_adapter))
        adapter_model = final_adapter / "adapter_model.safetensors"
        if not adapter_model.is_file():
            raise ExperimentError("Final adapter weights were not written")
        saved_peft_config = PeftConfig.from_pretrained(str(final_adapter))
        adapter_config_audit = _assert_saved_peft_config(saved_peft_config, config)
        adapter_sha = sha256_file(adapter_model)
        metrics = dict(train_result.metrics)
        metrics["adapter_finite_audit"] = adapter_finite
        metrics["saved_adapter_config_audit"] = adapter_config_audit
        metrics["inference_model_unwrap"] = {
            "accelerator_unwrapped": True,
            "keep_fp32_wrapper": False,
            "keep_torch_compile": False,
        }
        if resume_from_checkpoint is not None:
            metrics["resume_adapter_finite_audit"] = getattr(
                trainer, "resume_adapter_finite_audit", None
            )
        metrics["wall_time_seconds_measured"] = wall_time
        if torch.cuda.is_available():
            metrics["max_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated()
            )
            metrics["max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        write_json(output_dir / "train_metrics.json", metrics)
        write_json(
            output_dir / "log_history.json", {"log_history": trainer.state.log_history}
        )

        if smoke:
            smoke_prompt = validation_examples[0]["prompt"]
            smoke_device = next(trained_model.parameters()).device
            reference_logits = _last_token_logits(
                trained_model, tokenizer, smoke_prompt, chat_template_kwargs
            )
            generated, generated_token_ids = _generate_smoke_sample(
                trained_model,
                tokenizer,
                smoke_prompt,
                min(64, int(config["generation"]["max_new_tokens"])),
                chat_template_kwargs,
            )
            if not generated or not generated_token_ids:
                raise ExperimentError("Smoke generation produced an empty response")

            # Release the trained instance, then prove that the saved adapter can
            # be loaded into a clean copy of the exact base revision.
            del trainer
            del model
            del trained_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            reloaded_base = AutoModelForCausalLM.from_pretrained(
                config["model"]["name"], **_model_load_kwargs(config, torch)
            )
            _assert_model_revision(reloaded_base, config["model"]["revision"])
            reloaded_model = PeftModel.from_pretrained(
                reloaded_base,
                str(final_adapter),
                is_trainable=False,
                # PEFT creates destination LoRA tensors from the low-precision Base.
                # Enable its documented upcast so FP32 checkpoint values are not
                # first retained in a lower-precision inference adapter.
                autocast_adapter_dtype=True,
            )
            reloaded_model.config.use_cache = False
            if not isinstance(getattr(reloaded_model, "hf_device_map", None), dict):
                reloaded_model.to(smoke_device)
            if reloaded_model.device != smoke_device:
                raise ExperimentError(
                    "Reloaded smoke model is on {}, expected {}".format(
                        reloaded_model.device, smoke_device
                    )
                )
            _assert_lora_adapter_dtype(reloaded_model, torch.float32)
            if reference_adapter_state is None:
                raise ExperimentError("Smoke adapter reference state is missing")
            reloaded_adapter_state = _snapshot_peft_adapter_state(
                reloaded_model,
                get_peft_model_state_dict,
                torch,
                expected_trainable,
                REGISTERED_ADAPTER_TENSORS,
            )
            adapter_roundtrip = _assert_exact_adapter_state_roundtrip(
                reference_adapter_state,
                reloaded_adapter_state,
                torch,
                expected_trainable,
                REGISTERED_ADAPTER_TENSORS,
            )
            del reference_adapter_state, reloaded_adapter_state
            reloaded_logits = _last_token_logits(
                reloaded_model, tokenizer, smoke_prompt, chat_template_kwargs
            )
            logit_diagnostics = _logit_reload_diagnostics(
                reference_logits, reloaded_logits, torch
            )
            reloaded_response, reloaded_token_ids = _generate_smoke_sample(
                reloaded_model,
                tokenizer,
                smoke_prompt,
                min(64, int(config["generation"]["max_new_tokens"])),
                chat_template_kwargs,
            )
            smoke_audit = {
                "status": "adapter_roundtrip_pass_generation_check_pending",
                "example_id": validation_examples[0]["id"],
                "response": generated,
                "reloaded_response": reloaded_response,
                "adapter_config_reload": "pass",
                "adapter_weight_reload": "pass",
                "adapter_state_roundtrip": adapter_roundtrip,
                "reference_inference_context": (
                    "accelerator_unwrapped_without_fp32_wrapper"
                ),
                "reloaded_inference_context": "clean_peft_model",
                "generated_token_ids": generated_token_ids,
                "reloaded_token_ids": reloaded_token_ids,
                "logit_reload_diagnostics": logit_diagnostics,
            }
            write_json(output_dir / "smoke_generation.json", smoke_audit)
            if reloaded_token_ids != generated_token_ids:
                raise ExperimentError(
                    "Adapter reload changed the deterministic smoke token IDs"
                )
            if reloaded_response != generated:
                raise ExperimentError(
                    "Adapter reload changed the deterministic smoke response"
                )
            smoke_audit.update(
                {
                    "status": "pass",
                    "deterministic_response_match": True,
                    "deterministic_token_ids_match": True,
                }
            )
            write_json(output_dir / "smoke_generation.json", smoke_audit)
            del reloaded_model, reloaded_base

        manifest.update(
            {
                "status": "complete",
                "finished_at_utc": utc_now(),
                "metrics": metrics,
                "adapter_path": str(final_adapter.relative_to(workspace_path("."))),
                "adapter_model_sha256": adapter_sha,
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the balanced, 2-step integration smoke test.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Diagnostic only: record and allow a runtime lock mismatch.",
    )
    parser.add_argument(
        "--allow-non-cuda",
        action="store_true",
        help="Diagnostic only: allow a non-canonical CPU/MPS run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(
        workspace_path(args.config),
        smoke=args.smoke,
        resume_from_checkpoint=args.resume_from_checkpoint,
        allow_version_mismatch=args.allow_version_mismatch,
        allow_non_cuda=args.allow_non_cuda,
    )
    print(
        "Training {}: {}".format(
            result["status"], result.get("adapter_path", "no adapter")
        )
    )


if __name__ == "__main__":
    main()
