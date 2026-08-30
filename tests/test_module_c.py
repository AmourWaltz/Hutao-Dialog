from __future__ import annotations

import json
import math
import os
import copy
import sys
import tempfile
import types
import unittest
from collections import UserDict
from pathlib import Path
from unittest.mock import patch

from scripts.module_c.common import (
    CAPABILITIES,
    ExperimentError,
    canonical_tokenizer_identity,
    expand_assistant_turns,
    load_json,
    load_jsonl,
    sha256_file,
    workspace_path,
)
from scripts.module_c.export_logs import export
from scripts.module_c.make_safety_review import make_template
from scripts.module_c.select_checkpoint import select
from scripts.module_c.train_lora import (
    REGISTERED_CHAT_TEMPLATE_KWARGS,
    REGISTERED_ADAPTER_TENSORS,
    REGISTERED_CUBLAS_WORKSPACE_CONFIG,
    REGISTERED_CUBLAS_WORKSPACE_BYTES,
    REGISTERED_MODEL_NAME,
    REGISTERED_MODEL_REVISION,
    REGISTERED_TRAINABLE_PARAMETERS,
    _assert_exact_adapter_state_roundtrip,
    _assert_saved_peft_config,
    _assert_trainable_adapter_gradients_finite,
    _assert_logged_metrics_finite,
    _build_sft_args,
    _configure_registered_cublas_workspace,
    _logit_reload_diagnostics,
    _read_exact_requirements,
    _runtime_version_matches,
    _set_registered_cublas_workspace,
    _snapshot_peft_adapter_state,
    _unwrap_trained_model_for_inference,
    checkpoint_artifact_snapshot,
    validate_config,
    verify_training_data,
)
from scripts.module_c.tokenization import (
    CompletionOnlyDataCollator,
    tokenize_completion_example,
)
from scripts.module_d.generate_comparison import (
    EvaluationDataError,
    prompt_sha256,
    stable_item_seed,
    validate_test_generation_contract,
    validate_test_runtime_identity,
    validate_test_selection,
)


class FakeTokenizer:
    eos_token_id = 9001
    pad_token_id = 0

    @staticmethod
    def _render(messages, add_generation_prompt):
        text = "".join(
            "<|im_start|>{}\n{}<|im_end|>\n".format(message["role"], message["content"])
            for message in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    @classmethod
    def _encode(cls, text):
        ids = []
        cursor = 0
        marker = "<|im_end|>"
        while cursor < len(text):
            if text.startswith(marker, cursor):
                ids.append(cls.eos_token_id)
                cursor += len(marker)
            else:
                ids.append(ord(text[cursor]) + 10)
                cursor += 1
        return ids

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, return_tensors=None, **kwargs
    ):
        text = self._render(messages, add_generation_prompt)
        return self._encode(text) if tokenize else text


class BrokenPrefixTokenizer(FakeTokenizer):
    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, return_tensors=None, **kwargs
    ):
        value = super().apply_chat_template(
            messages, tokenize, add_generation_prompt, return_tensors, **kwargs
        )
        if tokenize and not add_generation_prompt:
            return [123456] + value
        return value


class V5BatchEncodingTokenizer(FakeTokenizer):
    """Mimic Transformers v5's Mapping-based BatchEncoding return value."""

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, return_tensors=None, **kwargs
    ):
        value = super().apply_chat_template(
            messages, tokenize, add_generation_prompt, return_tensors, **kwargs
        )
        if tokenize:
            return UserDict({"input_ids": value, "attention_mask": [1] * len(value)})
        return value


class KwargRecordingTokenizer(FakeTokenizer):
    def __init__(self):
        self.template_kwargs = []

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, return_tensors=None, **kwargs
    ):
        self.template_kwargs.append(dict(kwargs))
        return super().apply_chat_template(
            messages,
            tokenize,
            add_generation_prompt,
            return_tensors,
            **kwargs
        )


class ModuleCDataTests(unittest.TestCase):
    def test_full_cuda_lock_contains_all_exact_direct_requirements(self):
        direct = _read_exact_requirements(workspace_path("requirements-module-c.txt"))
        locked = _read_exact_requirements(
            workspace_path("requirements-module-c-lock-cu126.txt")
        )
        self.assertEqual(len(locked), 84)
        self.assertEqual(locked["torch"], "2.13.0+cu126")
        for name, version in direct.items():
            if name == "torch":
                self.assertTrue(_runtime_version_matches(name, version, locked[name]))
            else:
                self.assertEqual(locked[name], version)

    def test_runtime_version_allows_only_torch_local_build_suffix(self):
        self.assertTrue(_runtime_version_matches("torch", "2.13.0", "2.13.0+cu126"))
        self.assertTrue(_runtime_version_matches("torch", "2.13.0", "2.13.0"))
        self.assertFalse(_runtime_version_matches("torch", "2.13.0", "2.12.0+cu126"))
        self.assertFalse(
            _runtime_version_matches("transformers", "5.15.0", "5.15.0+local")
        )

    def test_real_data_expands_to_registered_counts(self):
        config = load_json(
            workspace_path("configs/module_c/hutao_qwen3_1p7b_lora_bf16.json")
        )
        expected = config["data"]["expected_derived_examples"]
        for split, expected_count in expected.items():
            records = load_jsonl(
                workspace_path("data/module_b_hutao/{}.jsonl".format(split))
            )
            examples = []
            for record in records:
                examples.extend(expand_assistant_turns(record, split))
            self.assertEqual(len(examples), expected_count)
            self.assertEqual(
                len({example["id"] for example in examples}), expected_count
            )

    def test_second_assistant_target_keeps_gold_history(self):
        records = load_jsonl(workspace_path("data/module_b_hutao/train.jsonl"))
        multi = next(
            record
            for record in records
            if sum(message["role"] == "assistant" for message in record["messages"])
            == 2
        )
        examples = expand_assistant_turns(multi, "train")
        self.assertEqual(len(examples), 2)
        self.assertEqual(
            [message["role"] for message in examples[1]["prompt"]],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(examples[1]["completion"][0]["role"], "assistant")

    def test_imported_multiturn_record_supervises_only_final_turn(self):
        records = load_jsonl(workspace_path("data/module_b_hutao/train.jsonl"))
        imported = next(
            record
            for record in records
            if record["metadata"].get("assistant_turn_policy") == "final_only"
            and sum(
                message["role"] == "assistant" for message in record["messages"]
            )
            == 2
        )

        examples = expand_assistant_turns(imported, "train")

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["id"], imported["id"] + "::A2")
        self.assertEqual(examples[0]["assistant_turn_index"], 2)
        self.assertEqual(examples[0]["assistant_turn_count"], 2)
        self.assertEqual(
            examples[0]["prompt"],
            imported["messages"][:-1],
        )
        self.assertEqual(
            [message["role"] for message in examples[0]["prompt"]],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(examples[0]["completion"], [imported["messages"][-1]])

    def test_derived_capability_counts_match_registered_manifest(self):
        rows = load_jsonl(workspace_path("data/module_c_hutao/train.jsonl"))
        counts = {capability: 0 for capability in CAPABILITIES}
        for row in rows:
            counts[row["metadata"]["capability"]] += 1
        manifest = load_json(workspace_path("data/module_c_hutao/manifest.json"))
        self.assertEqual(
            counts,
            manifest["splits"]["train"]["derived_examples_by_capability"],
        )

    def test_source_hashes_match_registered_config(self):
        config = load_json(
            workspace_path("configs/module_c/hutao_qwen3_1p7b_lora_bf16.json")
        )
        for split, expected in config["data"]["expected_sha256"].items():
            actual = sha256_file(
                workspace_path("data/module_b_hutao/{}.jsonl".format(split))
            )
            self.assertEqual(actual, expected)

    def test_derived_hashes_match_registered_config(self):
        config = load_json(
            workspace_path("configs/module_c/hutao_qwen3_1p7b_lora_bf16.json")
        )
        for split, expected in config["data"]["expected_derived_sha256"].items():
            actual = sha256_file(
                workspace_path("data/module_c_hutao/{}.jsonl".format(split))
            )
            self.assertEqual(actual, expected)
        self.assertEqual(
            sha256_file(workspace_path("data/module_c_hutao/manifest.json")),
            config["data"]["expected_manifest_sha256"],
        )

    def test_qwen3_preflight_binds_non_thinking_template(self):
        preflight = load_json(
            workspace_path("output/module_c_hutao/preflight.json")
        )
        self.assertEqual(preflight["status"], "pass")
        self.assertEqual(preflight["model"]["name"], REGISTERED_MODEL_NAME)
        self.assertEqual(
            preflight["model"]["revision"], REGISTERED_MODEL_REVISION
        )
        self.assertEqual(
            preflight["tokenizer_artifact"]["chat_template_kwargs"],
            REGISTERED_CHAT_TEMPLATE_KWARGS,
        )
        self.assertEqual(preflight["splits"]["train"]["over_max_length"], 0)
        self.assertEqual(
            preflight["splits"]["validation"]["over_max_length"], 0
        )
        supervised_samples = [
            sample["supervised_text"]
            for split in ("train", "validation")
            for sample in preflight["splits"][split][
                "decoded_supervision_samples"
            ]
        ]
        self.assertTrue(all("<think>" not in text for text in supervised_samples))


class ModuleCTokenizationTests(unittest.TestCase):
    def setUp(self):
        self.example = {
            "id": "example::A1",
            "prompt": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
            "completion": [{"role": "assistant", "content": "answer"}],
        }

    def test_completion_only_mask_is_exact(self):
        row = tokenize_completion_example(self.example, FakeTokenizer(), max_length=512)
        self.assertEqual(len(row["input_ids"]), len(row["labels"]))
        self.assertTrue(
            all(label == -100 for label in row["labels"][: row["prompt_tokens"]])
        )
        self.assertEqual(
            row["labels"][row["prompt_tokens"] :],
            row["input_ids"][row["prompt_tokens"] :],
        )
        self.assertIn(FakeTokenizer.eos_token_id, row["labels"])

    def test_transformers_v5_batch_encoding_is_supported(self):
        row = tokenize_completion_example(
            self.example, V5BatchEncodingTokenizer(), max_length=512
        )
        self.assertGreater(row["prompt_tokens"], 0)
        self.assertGreater(row["supervised_tokens"], 0)
        self.assertEqual(len(row["input_ids"]), len(row["labels"]))

    def test_qwen3_non_thinking_kwargs_reach_all_template_calls(self):
        tokenizer = KwargRecordingTokenizer()
        row = tokenize_completion_example(
            self.example,
            tokenizer,
            max_length=512,
            chat_template_kwargs={"enable_thinking": False},
        )
        self.assertGreater(row["supervised_tokens"], 0)
        self.assertEqual(
            tokenizer.template_kwargs,
            [{"enable_thinking": False}] * 4,
        )

    def test_prefix_mismatch_fails_closed(self):
        with self.assertRaises(ExperimentError):
            tokenize_completion_example(
                self.example, BrokenPrefixTokenizer(), max_length=512
            )

    def test_overlength_fails_instead_of_truncating(self):
        with self.assertRaises(ExperimentError):
            tokenize_completion_example(self.example, FakeTokenizer(), max_length=8)

    def test_prompt_must_end_in_user(self):
        invalid = dict(self.example)
        invalid["prompt"] = invalid["prompt"] + [
            {"role": "assistant", "content": "history"}
        ]
        with self.assertRaises(ExperimentError):
            tokenize_completion_example(invalid, FakeTokenizer(), max_length=512)

    def test_collator_right_pads_inputs_attention_and_labels(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        collator = CompletionOnlyDataCollator(pad_token_id=0, pad_to_multiple_of=4)
        batch = collator(
            [
                {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
                {
                    "input_ids": [3, 4, 5],
                    "attention_mask": [1, 1, 1],
                    "labels": [-100, 4, 5],
                },
            ]
        )
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        self.assertEqual(batch["input_ids"][0].tolist(), [1, 2, 0, 0])
        self.assertEqual(batch["attention_mask"][0].tolist(), [1, 1, 0, 0])
        self.assertEqual(batch["labels"][0].tolist(), [-100, 2, -100, -100])
        self.assertEqual(batch["input_ids"].dtype, torch.long)


class ModuleCSFTConfigTests(unittest.TestCase):
    def test_transformers_v5_training_arguments_are_used(self):
        class StrictSFTConfig:
            def __init__(self, **kwargs):
                if "warmup_ratio" in kwargs or "logging_dir" in kwargs:
                    raise TypeError("removed Transformers v5 argument")
                self.kwargs = kwargs

        fake_trl = types.ModuleType("trl")
        fake_trl.SFTConfig = StrictSFTConfig
        config = load_json(
            workspace_path("configs/module_c/hutao_qwen3_1p7b_lora_bf16.json")
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "smoke"
            with patch.dict(sys.modules, {"trl": fake_trl}), patch.dict(
                os.environ, {"TENSORBOARD_LOGGING_DIR": "external-value"}
            ):
                args = _build_sft_args(config, output_dir, smoke=True)
                self.assertEqual(args.kwargs["warmup_steps"], 0.1)
                self.assertEqual(args.kwargs["max_steps"], 2)
                self.assertEqual(args.kwargs["per_device_train_batch_size"], 4)
                self.assertEqual(args.kwargs["gradient_accumulation_steps"], 1)
                self.assertEqual(
                    math.ceil(args.kwargs["max_steps"] * args.kwargs["warmup_steps"]),
                    1,
                )
                self.assertNotIn("warmup_ratio", args.kwargs)
                self.assertNotIn("logging_dir", args.kwargs)
                self.assertEqual(
                    os.environ["TENSORBOARD_LOGGING_DIR"],
                    str(output_dir / "tensorboard"),
                )
                self.assertFalse(args.kwargs["logging_nan_inf_filter"])

    def test_qwen3_bf16_config_has_a_distinct_experiment_identity(self):
        fp16 = load_json(workspace_path("configs/module_c/hutao_qwen25_1p5b_lora.json"))
        bf16 = load_json(
            workspace_path("configs/module_c/hutao_qwen3_1p7b_lora_bf16.json")
        )
        self.assertEqual(fp16["model"]["dtype"], "float16")
        self.assertEqual(bf16["model"]["dtype"], "bfloat16")
        self.assertEqual(fp16["experiment_status"], "failed_diagnostic_do_not_train")
        self.assertEqual(bf16["experiment_status"], "canonical")
        self.assertEqual(bf16["model"]["name"], REGISTERED_MODEL_NAME)
        self.assertEqual(bf16["model"]["revision"], REGISTERED_MODEL_REVISION)
        self.assertEqual(
            bf16["model"]["chat_template_kwargs"],
            REGISTERED_CHAT_TEMPLATE_KWARGS,
        )
        self.assertEqual(
            bf16["method"]["lora"]["expected_trainable_parameters"],
            REGISTERED_TRAINABLE_PARAMETERS,
        )
        self.assertEqual(REGISTERED_ADAPTER_TENSORS, 28 * 2 * 2)
        self.assertNotEqual(fp16["experiment_name"], bf16["experiment_name"])
        self.assertNotEqual(
            fp16["training"]["output_dir"], bf16["training"]["output_dir"]
        )
        self.assertEqual(fp16["data"], bf16["data"])
        validate_config(bf16)

        hybrid = copy.deepcopy(bf16)
        hybrid["model"]["revision"] = fp16["model"]["revision"]
        with self.assertRaises(ExperimentError):
            validate_config(hybrid)

    def test_bf16_sft_arguments_disable_fp16(self):
        class CaptureSFTConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_trl = types.ModuleType("trl")
        fake_trl.SFTConfig = CaptureSFTConfig
        config = load_json(
            workspace_path("configs/module_c/hutao_qwen3_1p7b_lora_bf16.json")
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"trl": fake_trl}):
                args = _build_sft_args(config, Path(directory), smoke=True)
        self.assertTrue(args.kwargs["bf16"])
        self.assertFalse(args.kwargs["fp16"])
        self.assertFalse(args.kwargs["logging_nan_inf_filter"])

    def test_qwen3_tokenizer_identity_allows_explicit_null_bos_only(self):
        identity = {
            "chat_template_sha256": "template",
            "bos_token_id": None,
            "eos_token_id": 151645,
            "pad_token_id": 151643,
            "padding_side": "right",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        self.assertEqual(
            canonical_tokenizer_identity(identity)["bos_token_id"], None
        )

        missing_bos = dict(identity)
        del missing_bos["bos_token_id"]
        with self.assertRaises(ExperimentError):
            canonical_tokenizer_identity(missing_bos)

        missing_eos = dict(identity)
        missing_eos["eos_token_id"] = None
        with self.assertRaises(ExperimentError):
            canonical_tokenizer_identity(missing_eos)

    def test_smoke_unwrap_removes_accelerate_precision_wrappers(self):
        class FakeModel:
            pass

        class FakeAccelerator:
            def __init__(self):
                self.call = None

            def unwrap_model(self, model, **kwargs):
                self.call = (model, kwargs)
                return model

        class FakeTrainer:
            def __init__(self):
                self.model = FakeModel()
                self.accelerator = FakeAccelerator()

        trainer = FakeTrainer()
        actual = _unwrap_trained_model_for_inference(trainer)
        self.assertIs(actual, trainer.model)
        self.assertIs(trainer.accelerator.call[0], trainer.model)
        self.assertEqual(
            trainer.accelerator.call[1],
            {"keep_fp32_wrapper": False, "keep_torch_compile": False},
        )

    def test_exact_fp32_adapter_state_roundtrip_is_a_hard_gate(self):
        import torch

        source = {
            "base.layers.0.q_proj.lora_A.weight": torch.tensor(
                [[1.0, 2.0]], dtype=torch.float32
            ),
            "base.layers.0.q_proj.lora_B.weight": torch.tensor(
                [[3.0], [4.0]], dtype=torch.float32
            ),
        }

        class FakeModel:
            pass

        snapshot = _snapshot_peft_adapter_state(
            FakeModel(),
            lambda model: source,
            torch,
            expected_values=4,
            expected_tensors=2,
        )
        reloaded = {name: value.clone() for name, value in snapshot.items()}
        audit = _assert_exact_adapter_state_roundtrip(
            snapshot, reloaded, torch, expected_values=4, expected_tensors=2
        )
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["value_count"], 4)
        self.assertEqual(audit["comparison_rtol"], 0.0)

        changed = {name: value.clone() for name, value in reloaded.items()}
        changed["base.layers.0.q_proj.lora_A.weight"][0, 0] += 1.0
        with self.assertRaises(ExperimentError):
            _assert_exact_adapter_state_roundtrip(
                snapshot, changed, torch, expected_values=4, expected_tensors=2
            )

        wrong_dtype = {name: value.clone() for name, value in reloaded.items()}
        wrong_dtype["base.layers.0.q_proj.lora_A.weight"] = wrong_dtype[
            "base.layers.0.q_proj.lora_A.weight"
        ].to(torch.bfloat16)
        with self.assertRaises(ExperimentError):
            _assert_exact_adapter_state_roundtrip(
                snapshot, wrong_dtype, torch, expected_values=4, expected_tensors=2
            )

    def test_saved_peft_config_is_bound_to_registered_lora(self):
        config = load_json(
            workspace_path("configs/module_c/hutao_qwen3_1p7b_lora_bf16.json")
        )
        lora = config["method"]["lora"]
        saved = types.SimpleNamespace(
            base_model_name_or_path=config["model"]["name"],
            revision=config["model"]["revision"],
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=set(lora["target_modules"]),
            bias=lora["bias"],
            task_type="CAUSAL_LM",
            inference_mode=True,
        )
        audit = _assert_saved_peft_config(saved, config)
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["target_modules"], ["q_proj", "v_proj"])

        saved.revision = "wrong-revision"
        with self.assertRaises(ExperimentError):
            _assert_saved_peft_config(saved, config)

    def test_bf16_logit_difference_is_diagnostic_after_exact_state_reload(self):
        import torch

        reference = torch.tensor([10.0, 8.0, -0.01], dtype=torch.float32)
        reloaded = torch.tensor([10.46875, 8.25, 0.0], dtype=torch.float32)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(
                reloaded, reference, rtol=1e-4, atol=1e-4
            )
        audit = _logit_reload_diagnostics(reference, reloaded, torch)
        self.assertAlmostEqual(audit["maximum_absolute_difference"], 0.46875)
        self.assertTrue(audit["top1_match"])
        self.assertEqual(audit["values"], 3)

    def test_nonfinite_metric_values_are_rejected_even_when_strings(self):
        _assert_logged_metrics_finite({"loss": "1.25", "grad_norm": 0.5, "epoch": 1.0})
        with self.assertRaises(ExperimentError):
            _assert_logged_metrics_finite(
                {"loss": "nan", "grad_norm": "inf", "epoch": 1.0}
            )

    def test_registered_cublas_workspace_replaces_trainer_value(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            actual = _set_registered_cublas_workspace()
            self.assertEqual(actual, REGISTERED_CUBLAS_WORKSPACE_CONFIG)
            self.assertEqual(
                os.environ["CUBLAS_WORKSPACE_CONFIG"],
                REGISTERED_CUBLAS_WORKSPACE_CONFIG,
            )

    def test_registered_cublas_workspace_uses_live_torch_override(self):
        class FakeWorkspaceAPI:
            def __init__(self):
                self.value = None

            def __call__(self, value):
                self.value = value
                return value

        workspace_api = FakeWorkspaceAPI()

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            cublas_workspace_size = workspace_api

        class FakeBackends:
            cuda = FakeCuda()

        class FakeTorch:
            cuda = FakeCuda()
            backends = FakeBackends()

        result = _configure_registered_cublas_workspace(FakeTorch())
        self.assertTrue(result["api_override_available"])
        self.assertEqual(result["actual_size_bytes"], REGISTERED_CUBLAS_WORKSPACE_BYTES)
        self.assertEqual(workspace_api.value, REGISTERED_CUBLAS_WORKSPACE_BYTES)

    def test_nonfinite_adapter_gradient_is_rejected_before_step(self):
        class FakeGradient:
            def __init__(self, finite, values=4):
                self.finite = finite
                self.values = values

            def detach(self):
                return self

            def all(self):
                return self

            def item(self):
                return self.finite

            def numel(self):
                return self.values

        class FakeParameter:
            requires_grad = True

            def __init__(self, finite):
                self.grad = FakeGradient(finite)

        class FakeModel:
            def __init__(self, finite):
                self.finite = finite

            def named_parameters(self):
                return [("layer.lora_A.weight", FakeParameter(self.finite))]

        class FakeTorch:
            @staticmethod
            def isfinite(value):
                return value

        result = _assert_trainable_adapter_gradients_finite(
            FakeModel(True), FakeTorch()
        )
        self.assertEqual(result["checked_gradient_values"], 4)
        with self.assertRaises(ExperimentError):
            _assert_trainable_adapter_gradients_finite(FakeModel(False), FakeTorch())


class ModuleCSelectionAndLogTests(unittest.TestCase):
    def _write_json(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def _make_checkpoint(self, adapter_path, config, step):
        adapter_path.mkdir(parents=True)
        (adapter_path / "adapter_model.safetensors").write_bytes(
            "weights-{}".format(step).encode("utf-8")
        )
        self._write_json(
            adapter_path / "adapter_config.json",
            {
                "r": 16,
                "base_model_name_or_path": config["model"]["name"],
                "revision": config["model"]["revision"],
            },
        )
        self._write_json(adapter_path / "trainer_state.json", {"global_step": step})
        state_files = [
            "optimizer.pt",
            "scheduler.pt",
            "training_args.bin",
            "rng_state.pth",
        ]
        require_grad_scaler = config["model"]["dtype"] == "float16"
        if require_grad_scaler:
            state_files.append("scaler.pt")
        for name in state_files:
            (adapter_path / name).write_bytes(name.encode("utf-8"))
        return checkpoint_artifact_snapshot(
            adapter_path, require_grad_scaler=require_grad_scaler
        )

    def _write_run_manifest(self, config_path, config, data_snapshot):
        run_dir = Path(config["training"]["output_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "run_manifest.json"
        self._write_json(
            manifest_path,
            {
                "mode": "main",
                "status": "complete",
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
                "config": config,
                "data": data_snapshot,
                "runtime_check": {"mismatches": [], "mismatch_override": False},
                "hardware": {"cuda_available": True},
                "tokenizer": {
                    "chat_template_sha256": "template",
                    "bos_token_id": None,
                    "eos_token_id": 151645,
                    "pad_token_id": 151643,
                    "padding_side": "right",
                    "chat_template_kwargs": config["model"].get(
                        "chat_template_kwargs", {}
                    ),
                },
            },
        )
        return manifest_path

    def _fake_validation_metrics(self, mean_nll):
        examples = load_jsonl(workspace_path("data/module_c_hutao/validation.jsonl"))
        per_example = [
            {
                "id": example["id"],
                "source_record_id": example["source_record_id"],
                "capability": example["metadata"]["capability"],
                "supervised_tokens": 1,
                "nll_sum": mean_nll,
                "mean_nll": mean_nll,
            }
            for example in examples
        ]
        per_record = {}
        for example in examples:
            record_id = example["source_record_id"]
            value = per_record.setdefault(
                record_id,
                {
                    "capability": example["metadata"]["capability"],
                    "supervised_tokens": 0,
                    "mean_nll": mean_nll,
                },
            )
            value["supervised_tokens"] += 1
        records_by_capability = {
            capability: {
                example["source_record_id"]
                for example in examples
                if example["metadata"]["capability"] == capability
            }
            for capability in CAPABILITIES
        }
        return {
            "token_weighted_nll": mean_nll,
            "capability_macro_nll": mean_nll,
            "per_capability": {
                capability: {
                    "records": len(records_by_capability[capability]),
                    "mean_record_nll": mean_nll,
                }
                for capability in CAPABILITIES
            },
            "per_record": per_record,
            "per_example": per_example,
        }

    def _make_completed_safety_review(self, root, config_path, adapter_path, suffix):
        adapter_sha = sha256_file(adapter_path / "adapter_model.safetensors")
        adapter_config_sha = sha256_file(adapter_path / "adapter_config.json")
        comparison_path = root / ("comparison-{}.jsonl".format(suffix))
        rows = []
        validation_examples = load_jsonl(
            workspace_path("data/module_c_hutao/validation.jsonl")
        )
        for example in validation_examples:
            record_id = example["source_record_id"]
            turn = example["assistant_turn_index"]
            prompt = example["prompt"]
            digest = prompt_sha256(prompt)
            rows.append(
                {
                    "schema_version": "module_d.comparison.v1",
                    "eval_id": "validation:{}:controlled_gold_history:T{:02d}".format(
                        record_id, turn
                    ),
                    "record_id": record_id,
                    "split": "validation",
                    "capability": example["metadata"]["capability"],
                    "scenario_group": example["metadata"]["scenario_group"],
                    "seriousness": example["metadata"]["seriousness"],
                    "risk_flags": example["metadata"]["risk_flags"],
                    "mode": "controlled_gold_history",
                    "assistant_turn_index": turn,
                    "latest_user_message": next(
                        message["content"]
                        for message in reversed(prompt)
                        if message["role"] == "user"
                    ),
                    "gold_response": example["completion"][0]["content"],
                    "prompt_equal": True,
                    "generation": {
                        "seed": stable_item_seed(
                            42,
                            "validation",
                            record_id,
                            "controlled_gold_history",
                            turn,
                        ),
                        "config": {
                            "max_new_tokens": 192,
                            "do_sample": False,
                            "num_beams": 1,
                        },
                    },
                    "base": {
                        "variant": "base",
                        "model_label": "base",
                        "prompt_messages": prompt,
                        "prompt_sha256": digest,
                        "response": "base response",
                    },
                    "lora": {
                        "variant": "lora",
                        "model_label": "lora-{}".format(suffix),
                        "prompt_messages": prompt,
                        "prompt_sha256": digest,
                        "response": "safe lora response",
                    },
                }
            )
        comparison_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        manifest_path = root / ("generation-{}.json".format(suffix))
        config = load_json(config_path)
        self._write_json(
            manifest_path,
            {
                "schema_version": "module_d.generation_manifest.v1",
                "splits": ["validation"],
                "comparisons": len(rows),
                "records": config["data"]["expected_source_records"]["validation"],
                "mode": "controlled_gold_history",
                "base_model": config["model"]["name"],
                "base_revision": config["model"]["revision"],
                "generation_config": {
                    "max_new_tokens": 192,
                    "do_sample": False,
                    "num_beams": 1,
                    "seed": 42,
                },
                "chat_template_kwargs": config["model"].get(
                    "chat_template_kwargs", {}
                ),
                "source_sha256": {
                    "validation": config["data"]["expected_sha256"]["validation"]
                },
                "base_runtime": {
                    "model_name_or_path": config["model"]["name"],
                    "revision": config["model"]["revision"],
                    "resolved_commit": config["model"]["revision"],
                    "chat_template_sha256": "template",
                    "dtype_requested": config["model"]["dtype"],
                    "dtype_actual_first_parameter": "torch.{}".format(
                        config["model"]["dtype"]
                    ),
                    "first_parameter_device": "cuda:0",
                    "attention_implementation_requested": "eager",
                    "attention_implementation_resolved": "eager",
                    "cuda_device_count": 1,
                    "bos_token_id": None,
                    "eos_token_id": 151645,
                    "pad_token_id": 151643,
                    "padding_side": "right",
                    "chat_template_kwargs": config["model"].get(
                        "chat_template_kwargs", {}
                    ),
                },
                "lora_runtime": {
                    "model_name_or_path": config["model"]["name"],
                    "revision": config["model"]["revision"],
                    "resolved_commit": config["model"]["revision"],
                    "chat_template_sha256": "template",
                    "dtype_requested": config["model"]["dtype"],
                    "dtype_actual_first_parameter": "torch.{}".format(
                        config["model"]["dtype"]
                    ),
                    "first_parameter_device": "cuda:0",
                    "attention_implementation_requested": "eager",
                    "attention_implementation_resolved": "eager",
                    "cuda_device_count": 1,
                    "bos_token_id": None,
                    "eos_token_id": 151645,
                    "pad_token_id": 151643,
                    "padding_side": "right",
                    "chat_template_kwargs": config["model"].get(
                        "chat_template_kwargs", {}
                    ),
                    "adapter_path": str(adapter_path),
                    "adapter_sha256": adapter_sha,
                    "adapter_config_sha256": adapter_config_sha,
                },
                "output": str(comparison_path.resolve()),
                "output_sha256": sha256_file(comparison_path),
                "attention_implementation": "eager",
                "python_hash_seed": "42",
            },
        )
        review_path = root / ("review-{}.json".format(suffix))
        review = make_template(
            config_path, comparison_path, manifest_path, adapter_path, review_path,
        )
        for value in review["records"].values():
            value["pass"] = True
            value["checks"] = dict((name, True) for name in value["checks"])
            value["reviewer_id"] = "reviewer-01"
        self._write_json(review_path, review)
        return review_path

    def test_checkpoint_selection_requires_safety_and_prefers_earlier_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_json(
                workspace_path("configs/module_c/hutao_qwen3_1p7b_lora_bf16.json")
            )
            config["training"]["output_dir"] = str(root / "run")
            config_path = root / "config.json"
            self._write_json(config_path, config)
            data_snapshot = verify_training_data(config)
            run_manifest_path = self._write_run_manifest(
                config_path, config, data_snapshot
            )
            candidates = []
            for step, nll in (
                (26, 1.0),
                (52, 0.997),
                (78, 1.05),
                (104, 1.08),
                (130, 1.10),
            ):
                adapter_path = root / "run" / "checkpoint-{}".format(step)
                checkpoint_artifacts = self._make_checkpoint(adapter_path, config, step)
                adapter_sha = sha256_file(adapter_path / "adapter_model.safetensors")
                adapter_config_sha = sha256_file(adapter_path / "adapter_config.json")
                metric_path = root / "metric-{}.json".format(step)
                review_path = root / "review-{}.json".format(step)
                self._write_json(
                    metric_path,
                    {
                        "config_sha256": sha256_file(config_path),
                        "model": config["model"],
                        "adapter_path": str(adapter_path),
                        "adapter_sha256": adapter_sha,
                        "adapter_config_sha256": adapter_config_sha,
                        "status": "scored_unreviewed_for_safety",
                        "evaluation_base_precision": "{}_unquantized".format(
                            config["model"]["dtype"]
                        ),
                        "resolved_model_revision": config["model"]["revision"],
                        "tokenizer": load_json(run_manifest_path)["tokenizer"],
                        "metrics": self._fake_validation_metrics(nll),
                        "data": data_snapshot,
                        "runtime_check": {"mismatches": []},
                        "hardware": {"cuda_available": True},
                        "determinism": {
                            "seed": 42,
                            "python_hash_seed": "42",
                            "cublas_workspace_config": ":4096:8",
                            "deterministic_algorithms": True,
                            "tf32_matmul": False,
                            "tf32_cudnn": False,
                        },
                        "checkpoint_step": step,
                        "checkpoint_artifacts": checkpoint_artifacts,
                        "run_manifest": str(run_manifest_path),
                        "run_manifest_sha256": sha256_file(run_manifest_path),
                    },
                )
                review_path = self._make_completed_safety_review(
                    root, config_path, adapter_path, str(step),
                )
                candidates.append((metric_path, review_path))
            selection_path = root / "selected.json"
            result = select(config_path, candidates, selection_path)
            self.assertEqual(result["selected"]["checkpoint_step"], 26)
            selected_adapter = root / "run" / "checkpoint-26"
            unlocked = validate_test_selection(
                selection_path,
                selected_adapter,
                config["model"]["name"],
                config["model"]["revision"],
            )
            self.assertEqual(unlocked, result)
            runtime = {
                "model_name_or_path": config["model"]["name"],
                "revision": config["model"]["revision"],
                "resolved_commit": config["model"]["revision"],
                "dtype_requested": config["model"]["dtype"],
                "dtype_actual_first_parameter": "torch.{}".format(
                    config["model"]["dtype"]
                ),
                "first_parameter_device": "cuda:0",
                "attention_implementation_requested": "eager",
                "attention_implementation_resolved": "eager",
                "chat_template_sha256": "template",
                "bos_token_id": None,
                "eos_token_id": 151645,
                "pad_token_id": 151643,
                "padding_side": "right",
                "chat_template_kwargs": config["model"].get(
                    "chat_template_kwargs", {}
                ),
                "cuda_device_count": 1,
            }
            lora_runtime = dict(runtime)
            lora_runtime["adapter_sha256"] = result["selected"]["adapter_sha256"]
            lora_runtime["adapter_config_sha256"] = result["selected"][
                "adapter_config_sha256"
            ]
            validate_test_runtime_identity(result, runtime, lora_runtime)
            missing_bos_runtime = dict(runtime)
            del missing_bos_runtime["bos_token_id"]
            with self.assertRaises(EvaluationDataError):
                validate_test_runtime_identity(
                    result, missing_bos_runtime, lora_runtime
                )
            mismatched_bos_runtime = dict(runtime)
            mismatched_bos_runtime["bos_token_id"] = 151643
            with self.assertRaises(EvaluationDataError):
                validate_test_runtime_identity(
                    result, mismatched_bos_runtime, lora_runtime
                )
            contract = validate_test_generation_contract(
                result,
                dtype=config["model"]["dtype"],
                attention_implementation="eager",
                seed=42,
                max_new_tokens=192,
                chat_template_kwargs=config["model"].get(
                    "chat_template_kwargs", {}
                ),
                python_hash_seed="42",
            )
            self.assertEqual(contract["max_new_tokens"], 192)
            with self.assertRaises(EvaluationDataError):
                validate_test_generation_contract(
                    result,
                    dtype=config["model"]["dtype"],
                    attention_implementation="eager",
                    seed=42,
                    max_new_tokens=192,
                    chat_template_kwargs={"enable_thinking": True},
                    python_hash_seed="42",
                )
            with self.assertRaises(EvaluationDataError):
                validate_test_generation_contract(
                    result,
                    dtype=config["model"]["dtype"],
                    attention_implementation="eager",
                    seed=42,
                    max_new_tokens=1,
                    chat_template_kwargs=config["model"].get(
                        "chat_template_kwargs", {}
                    ),
                    python_hash_seed="42",
                )

            tampered_metric_path = candidates[-1][0]
            tampered_metric = load_json(tampered_metric_path)
            tampered_metric["metrics"]["capability_macro_nll"] += 0.25
            self._write_json(tampered_metric_path, tampered_metric)
            with self.assertRaises(ExperimentError):
                select(config_path, candidates, root / "integrity-failed.json")
            integrity_result = load_json(root / "integrity-failed.json")
            self.assertEqual(integrity_result["status"], "failed_candidate_integrity")

    def test_checkpoint_selection_rejects_all_unsafe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_json(
                workspace_path("configs/module_c/hutao_qwen3_1p7b_lora_bf16.json")
            )
            config["training"]["output_dir"] = str(root / "run")
            config_path = root / "config.json"
            self._write_json(config_path, config)
            data_snapshot = verify_training_data(config)
            run_manifest_path = self._write_run_manifest(
                config_path, config, data_snapshot
            )
            candidates = []
            for step in (26, 52, 78, 104, 130):
                adapter_path = root / "run" / "checkpoint-{}".format(step)
                checkpoint_artifacts = self._make_checkpoint(adapter_path, config, step)
                adapter_sha = sha256_file(adapter_path / "adapter_model.safetensors")
                adapter_config_sha = sha256_file(adapter_path / "adapter_config.json")
                metric_path = root / "metric-{}.json".format(step)
                self._write_json(
                    metric_path,
                    {
                        "config_sha256": sha256_file(config_path),
                        "model": config["model"],
                        "adapter_path": str(adapter_path),
                        "adapter_sha256": adapter_sha,
                        "adapter_config_sha256": adapter_config_sha,
                        "status": "scored_unreviewed_for_safety",
                        "evaluation_base_precision": "{}_unquantized".format(
                            config["model"]["dtype"]
                        ),
                        "resolved_model_revision": config["model"]["revision"],
                        "tokenizer": load_json(run_manifest_path)["tokenizer"],
                        "metrics": self._fake_validation_metrics(1.0),
                        "data": data_snapshot,
                        "runtime_check": {"mismatches": []},
                        "hardware": {"cuda_available": True},
                        "determinism": {
                            "seed": 42,
                            "python_hash_seed": "42",
                            "cublas_workspace_config": ":4096:8",
                            "deterministic_algorithms": True,
                            "tf32_matmul": False,
                            "tf32_cudnn": False,
                        },
                        "checkpoint_step": step,
                        "checkpoint_artifacts": checkpoint_artifacts,
                        "run_manifest": str(run_manifest_path),
                        "run_manifest_sha256": sha256_file(run_manifest_path),
                    },
                )
                review_path = self._make_completed_safety_review(
                    root, config_path, adapter_path, "unsafe-{}".format(step)
                )
                review = load_json(review_path)
                review["records"]["HT-WLD-G07-V2"]["pass"] = False
                first_check = next(iter(review["records"]["HT-WLD-G07-V2"]["checks"]))
                review["records"]["HT-WLD-G07-V2"]["checks"][first_check] = False
                self._write_json(review_path, review)
                candidates.append((metric_path, review_path))
            with self.assertRaises(ExperimentError):
                select(
                    config_path, candidates, root / "selected.json",
                )
            failed = json.loads((root / "selected.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed_no_safe_checkpoint")

    def test_log_export_produces_csv_and_svg(self):
        history = {
            "log_history": [
                {"step": 1, "epoch": 0.1, "loss": 2.0, "learning_rate": 1e-4},
                {"step": 2, "epoch": 1.0, "eval_loss": 1.8},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "history.json"
            source.write_text(json.dumps(history), encoding="utf-8")
            points = export(source, root / "curve.csv", root / "curve.svg", "test")
            self.assertEqual(len(points), 2)
            self.assertIn(
                "train_loss", (root / "curve.csv").read_text(encoding="utf-8")
            )
            self.assertIn("<svg", (root / "curve.svg").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
