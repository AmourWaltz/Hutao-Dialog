#!/usr/bin/env python3
"""Standard-library tests for the Module D evaluation framework."""

from __future__ import print_function

import copy
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.module_d import generate_comparison as generation_module
from scripts.module_d.build_review_sheet import (
    BlindReviewError,
    ERROR_TAGS,
    KEY_SCHEMA_VERSION,
    REVIEW_FIELDS,
    SCORE_DIMENSIONS,
    build_blind_review,
    load_comparisons,
    validate_generation_manifest,
    write_blind_key,
    write_review_csv,
    write_rubric_json,
)
from scripts.module_d.generate_comparison import (
    DEFAULT_DATA_ROOT,
    EvaluationDataError,
    TransformersTextGenerator,
    file_sha256,
    generate_comparisons,
    load_evaluation_records,
    validate_test_final_adapter,
    validate_test_selection,
)
from scripts.module_d.rubric import (
    GUARD_DIMENSIONS,
    PERSONA_LAYERS,
    PREFERENCE_DIMENSIONS,
    RUBRIC_SCHEMA_VERSION,
    SCORE_RUBRICS,
    public_rubric_payload,
    rubric_sha256,
)
from scripts.module_d.score_review import (
    ReviewValidationError,
    load_blind_key,
    load_scored_csv,
    score_reviews,
)


def set_layer_preferences(row, preference):
    for dimension in PREFERENCE_DIMENSIONS:
        row[dimension + "_preference"] = preference


class FakeGenerator(object):
    def __init__(self, model_label, response_prefix):
        self.model_label = model_label
        self.response_prefix = response_prefix
        self.calls = []

    def generate(self, messages, generation_config, seed):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "generation_config": dict(generation_config),
                "seed": seed,
            }
        )
        return "%s%d" % (self.response_prefix, len(self.calls))


class FakeRuntimeTextGenerator(object):
    """Small stand-in that exercises the real generation CLI and manifest path."""

    def __init__(
        self,
        model_name_or_path,
        model_label,
        adapter_path=None,
        revision=None,
        torch_dtype="auto",
        attention_implementation="eager",
        chat_template_kwargs=None,
        **_kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.model_label = model_label
        self.adapter_path = adapter_path
        self.revision = revision
        self.torch_dtype = torch_dtype
        self.attention_implementation = attention_implementation
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

    def audit_metadata(self):
        result = {
            "model_name_or_path": self.model_name_or_path,
            "model_label": self.model_label,
            "revision": self.revision,
            "resolved_commit": self.revision,
            "adapter_path": self.adapter_path,
            "adapter_sha256": None,
            "adapter_config_sha256": None,
            "dtype_requested": str(self.torch_dtype),
            "dtype_actual_first_parameter": "torch.%s" % self.torch_dtype,
            "first_parameter_device": "cuda:0",
            "attention_implementation_requested": self.attention_implementation,
            "attention_implementation_resolved": self.attention_implementation,
            "chat_template_sha256": "template-sha",
            "bos_token_id": None,
            "eos_token_id": 2,
            "pad_token_id": 2,
            "padding_side": "right",
            "chat_template_kwargs": dict(self.chat_template_kwargs),
            "cuda_device_count": 1,
            "hf_device_map": {"": "0"},
        }
        if self.adapter_path:
            adapter = Path(self.adapter_path)
            result["adapter_sha256"] = file_sha256(
                adapter / "adapter_model.safetensors"
            )
            result["adapter_config_sha256"] = file_sha256(
                adapter / "adapter_config.json"
            )
        return result

    def generate(self, messages, generation_config, seed):
        return "%s-response-%d" % (self.model_label, seed)


def make_record(split="test"):
    return {
        "id": "HT-DLY-G08-V2" if split == "test" else "HT-DLY-G07-V2",
        "messages": [
            {"role": "system", "content": "你是胡桃，以符合角色设定且适合当前情境的方式回答。",},
            {"role": "user", "content": "先给我一个建议。"},
            {"role": "assistant", "content": "这是第一条金标准回答。"},
            {"role": "user", "content": "如果情况变了呢？"},
            {"role": "assistant", "content": "这是第二条金标准回答。"},
        ],
        "metadata": {
            "split": split,
            "capability": "daily_chat",
            "scenario_group": "DLY-G08" if split == "test" else "DLY-G07",
            "seriousness": 2,
            "risk_flags": [],
        },
        "_source_split": split,
        "_source_line": 1,
    }


def make_final_only_record(split="test"):
    record = make_record(split=split)
    record["id"] = "EXT-final-only-%s" % split
    record["metadata"]["scenario_group"] = "EXT-final-only"
    record["metadata"]["assistant_turn_policy"] = "final_only"
    record["metadata"]["source"] = {"dataset": "all_samples.jsonl"}
    return record


def generate_fake_comparisons(mode="controlled_gold_history"):
    base = FakeGenerator("foundation-v1", "甲方回应")
    lora = FakeGenerator("hutao-lora-v1", "乙方回应")
    comparisons = generate_comparisons(
        [make_record()],
        base,
        lora,
        mode=mode,
        generation_config={"max_new_tokens": 64, "do_sample": False, "num_beams": 1,},
        seed=2026,
    )
    return comparisons, base, lora


def make_final_adapter_fixture(root, status="complete"):
    model = {
        "name": "fake-base",
        "revision": "a" * 40,
        "dtype": "float16",
        "attention_implementation": "eager",
        "chat_template_kwargs": {},
    }
    adapter = root / "adapter-final"
    adapter.mkdir()
    adapter_file = adapter / "adapter_model.safetensors"
    adapter_file.write_bytes(b"final-adapter-weights")
    adapter_config_file = adapter / "adapter_config.json"
    adapter_config_file.write_text(
        json.dumps(
            {
                "base_model_name_or_path": model["name"],
                "revision": model["revision"],
            }
        ),
        encoding="utf-8",
    )
    run_manifest = {
        "mode": "main",
        "status": status,
        "config_sha256": "c" * 64,
        "config": {
            "model": model,
            "runtime": {"visible_cuda_devices": 1},
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 192,
                "seed": 42,
            },
        },
        "tokenizer": {
            "chat_template_sha256": "template-sha",
            "bos_token_id": None,
            "eos_token_id": 2,
            "pad_token_id": 2,
            "padding_side": "right",
            "chat_template_kwargs": {},
        },
        "adapter_path": str(adapter.resolve()),
        "adapter_model_sha256": file_sha256(adapter_file),
    }
    run_manifest_path = root / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    return adapter, run_manifest_path, model


class GenerateComparisonTests(unittest.TestCase):
    def _make_selection_fixture(self, root):
        model = {
            "name": "Qwen/Qwen3-1.7B",
            "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "dtype": "bfloat16",
            "attention_implementation": "eager",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        run_manifest = root / "run_manifest.json"
        run_manifest.write_text(
            json.dumps(
                {
                    "mode": "main",
                    "status": "complete",
                    "config_sha256": "config-sha",
                    "config": {"model": model, "runtime": {"visible_cuda_devices": 1},},
                    "tokenizer": {
                        "chat_template_sha256": "template-sha",
                        "bos_token_id": None,
                        "eos_token_id": 151645,
                        "pad_token_id": 151643,
                        "padding_side": "right",
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                }
            ),
            encoding="utf-8",
        )
        candidates = []
        for step in (26, 52, 78, 104, 130):
            adapter = root / ("checkpoint-%d" % step)
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(
                ("adapter-%d" % step).encode("utf-8")
            )
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": model["name"],
                        "revision": model["revision"],
                    }
                ),
                encoding="utf-8",
            )
            metric = root / ("metric-%d.json" % step)
            review = root / ("review-%d.json" % step)
            metric.write_text("{}", encoding="utf-8")
            review.write_text("{}", encoding="utf-8")
            candidates.append(
                {
                    "metric_file": str(metric),
                    "metric_file_sha256": file_sha256(metric),
                    "safety_review_file": str(review),
                    "safety_review_file_sha256": file_sha256(review),
                    "adapter_path": str(adapter),
                    "adapter_sha256": file_sha256(
                        adapter / "adapter_model.safetensors"
                    ),
                    "adapter_config_sha256": file_sha256(
                        adapter / "adapter_config.json"
                    ),
                    "model": model,
                    "checkpoint_step": step,
                    "integrity_pass": True,
                    "integrity_failures": [],
                    "safety_pass": True,
                    "safety_failures": [],
                }
            )
        selection = {
            "schema_version": "module_c.checkpoint_selection.v1",
            "status": "selected",
            "selected": candidates[0],
            "candidates": candidates,
            "model": model,
            "experiment_config_sha256": "config-sha",
            "run_manifest": str(run_manifest),
            "run_manifest_sha256": file_sha256(run_manifest),
            "expected_checkpoint_steps": [26, 52, 78, 104, 130],
            "test_access_authorised_after_this_manifest": True,
        }
        selection_path = root / "selection.json"
        selection_path.write_text(json.dumps(selection), encoding="utf-8")
        return selection_path, root / "checkpoint-26", model

    def test_self_declared_empty_selection_evidence_cannot_unlock_test(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            selection_path, adapter, model = self._make_selection_fixture(root)
            with self.assertRaises(EvaluationDataError):
                validate_test_selection(
                    selection_path, adapter, model["name"], model["revision"],
                )

    def test_completed_final_adapter_can_bypass_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter, run_manifest_path, model = make_final_adapter_fixture(root)
            binding = validate_test_final_adapter(
                adapter, model["name"], model["revision"]
            )
            self.assertEqual(binding["adapter_provenance"], "training_final")
            self.assertEqual(
                Path(binding["selected"]["adapter_path"]), adapter.resolve()
            )
            self.assertEqual(
                binding["run_manifest_sha256"], file_sha256(run_manifest_path)
            )

    def test_final_adapter_rejects_incomplete_or_changed_training_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter, _, model = make_final_adapter_fixture(root, status="training")
            with self.assertRaises(EvaluationDataError):
                validate_test_final_adapter(
                    adapter, model["name"], model["revision"]
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter, _, model = make_final_adapter_fixture(root)
            (adapter / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaises(EvaluationDataError):
                validate_test_final_adapter(
                    adapter, model["name"], model["revision"]
                )

    def test_final_adapter_mode_rejects_a_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter, _, model = make_final_adapter_fixture(root)
            with self.assertRaises(EvaluationDataError):
                validate_test_final_adapter(
                    adapter.parent / "checkpoint-130",
                    model["name"],
                    model["revision"],
                )

    def test_direct_final_cli_writes_a_revalidatable_test_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter, _, model = make_final_adapter_fixture(root)
            output_path = root / "direct-test.jsonl"
            argv = [
                "--data-root",
                str(DEFAULT_DATA_ROOT),
                "--split",
                "test",
                "--mode",
                "controlled_gold_history",
                "--base-model",
                model["name"],
                "--base-revision",
                model["revision"],
                "--lora-adapter",
                str(adapter),
                "--use-final-adapter",
                "--dtype",
                model["dtype"],
                "--attention-implementation",
                "eager",
                "--seed",
                "42",
                "--max-new-tokens",
                "192",
                "--output",
                str(output_path),
            ]
            with mock.patch.object(
                generation_module,
                "TransformersTextGenerator",
                FakeRuntimeTextGenerator,
            ), mock.patch.dict(os.environ, {"PYTHONHASHSEED": "42"}), mock.patch(
                "builtins.print"
            ):
                generation_module.main(argv)

            manifest_path = output_path.with_suffix(
                output_path.suffix + ".manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["adapter_provenance"], "training_final")
            self.assertIsNone(manifest["selection_manifest"])
            comparisons = load_comparisons(output_path)
            validate_generation_manifest(
                manifest_path, output_path, comparisons
            )

    def test_loads_current_validation_and_test_messages(self):
        records = load_evaluation_records(
            DEFAULT_DATA_ROOT, splits=("validation", "test")
        )
        self.assertEqual(len(records), 86)
        self.assertEqual(
            set(record["_source_split"] for record in records),
            set(("validation", "test")),
        )
        self.assertTrue(
            all(record["messages"][0]["role"] == "system" for record in records)
        )
        self.assertTrue(
            all(record["messages"][-1]["role"] == "assistant" for record in records)
        )

    def test_loader_rejects_training_split(self):
        with self.assertRaises(EvaluationDataError):
            load_evaluation_records(DEFAULT_DATA_ROOT, splits=("train",))

    def test_loader_rejects_modified_registered_test(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original = (DEFAULT_DATA_ROOT / "test.jsonl").read_text(encoding="utf-8")
            (root / "test.jsonl").write_text(
                original.replace("胡桃", "胡桃（已改）", 1), encoding="utf-8"
            )
            with self.assertRaises(EvaluationDataError):
                load_evaluation_records(root, splits=("test",))

    def test_controlled_mode_uses_identical_gold_history(self):
        original = make_record()
        original_copy = copy.deepcopy(original)
        base = FakeGenerator("foundation-v1", "甲方回应")
        lora = FakeGenerator("hutao-lora-v1", "乙方回应")
        comparisons = generate_comparisons(
            [original], base, lora, mode="controlled-gold-history", seed=7
        )

        self.assertEqual(original, original_copy)
        self.assertEqual(len(comparisons), 2)
        self.assertTrue(all(item["prompt_equal"] for item in comparisons))
        second_prompt = comparisons[1]["base"]["prompt_messages"]
        assistant_history = [
            message["content"]
            for message in second_prompt
            if message["role"] == "assistant"
        ]
        self.assertEqual(assistant_history, ["这是第一条金标准回答。"])
        self.assertNotIn("甲方回应1", assistant_history)
        self.assertEqual(base.calls[0]["seed"], lora.calls[0]["seed"])
        self.assertEqual(base.calls[1]["seed"], lora.calls[1]["seed"])
        self.assertFalse(base.calls[0]["generation_config"]["do_sample"])

    def test_rollout_uses_each_models_generated_history(self):
        comparisons, base, lora = generate_fake_comparisons(mode="rollout")
        self.assertEqual(len(comparisons), 2)
        self.assertTrue(comparisons[0]["prompt_equal"])
        self.assertFalse(comparisons[1]["prompt_equal"])
        base_assistants = [
            message["content"]
            for message in comparisons[1]["base"]["prompt_messages"]
            if message["role"] == "assistant"
        ]
        lora_assistants = [
            message["content"]
            for message in comparisons[1]["lora"]["prompt_messages"]
            if message["role"] == "assistant"
        ]
        self.assertEqual(base_assistants, ["甲方回应1"])
        self.assertEqual(lora_assistants, ["乙方回应1"])
        self.assertNotIn("这是第一条金标准回答。", base_assistants)
        self.assertNotIn("这是第一条金标准回答。", lora_assistants)
        self.assertEqual(
            base.calls[1]["messages"], comparisons[1]["base"]["prompt_messages"]
        )
        self.assertEqual(
            lora.calls[1]["messages"], comparisons[1]["lora"]["prompt_messages"]
        )

    def test_final_only_evaluates_final_turn_with_gold_bridge_history(self):
        record = make_final_only_record()
        base = FakeGenerator("foundation-v1", "甲方回应")
        lora = FakeGenerator("hutao-lora-v1", "乙方回应")

        for mode in ("controlled_gold_history", "rollout"):
            with self.subTest(mode=mode):
                base.calls = []
                lora.calls = []
                comparisons = generate_comparisons(
                    [record], base, lora, mode=mode, seed=7
                )

                self.assertEqual(len(comparisons), 1)
                self.assertEqual(comparisons[0]["assistant_turn_index"], 2)
                self.assertTrue(comparisons[0]["eval_id"].endswith(":T02"))
                self.assertEqual(len(base.calls), 1)
                self.assertEqual(len(lora.calls), 1)
                expected_prompt = record["messages"][:-1]
                self.assertEqual(
                    comparisons[0]["base"]["prompt_messages"], expected_prompt
                )
                self.assertEqual(
                    comparisons[0]["lora"]["prompt_messages"], expected_prompt
                )
                self.assertEqual(
                    [
                        message["content"]
                        for message in expected_prompt
                        if message["role"] == "assistant"
                    ],
                    ["这是第一条金标准回答。"],
                )

    def test_generation_rejects_sampling(self):
        base = FakeGenerator("foundation-v1", "甲")
        lora = FakeGenerator("hutao-lora-v1", "乙")
        with self.assertRaises(ValueError):
            generate_comparisons(
                [make_record()], base, lora, generation_config={"do_sample": True},
            )

    def test_transformers_generator_does_not_load_heavy_dependencies_on_init(self):
        generator = TransformersTextGenerator(
            "Qwen/Qwen3-1.7B", "foundation-v1",
            chat_template_kwargs={"enable_thinking": False},
        )
        self.assertIsNone(generator._torch)
        self.assertIsNone(generator._tokenizer)
        self.assertIsNone(generator._model)


class BlindReviewTests(unittest.TestCase):
    def _write_validation_generation_fixture(self, root):
        records = load_evaluation_records(DEFAULT_DATA_ROOT, splits=("validation",))
        comparisons = generate_comparisons(
            records,
            FakeGenerator("foundation-v1", "base"),
            FakeGenerator("hutao-lora-v1", "lora"),
            mode="controlled_gold_history",
            generation_config={
                "max_new_tokens": 192,
                "do_sample": False,
                "num_beams": 1,
            },
            seed=42,
        )
        comparison_path = root / "validation.jsonl"
        comparison_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in comparisons
            ),
            encoding="utf-8",
        )
        runtime = {
            "model_name_or_path": "fake-base",
            "revision": "a" * 40,
            "resolved_commit": "a" * 40,
            "dtype_requested": "float16",
            "dtype_actual_first_parameter": "torch.float16",
            "first_parameter_device": "cuda:0",
            "attention_implementation_requested": "eager",
            "attention_implementation_resolved": "eager",
            "chat_template_sha256": "template-sha",
            "bos_token_id": None,
            "eos_token_id": 2,
            "pad_token_id": 2,
            "padding_side": "right",
            "chat_template_kwargs": {},
            "cuda_device_count": 1,
            "hf_device_map": {"": "0"},
        }
        manifest = {
            "schema_version": "module_d.generation_manifest.v1",
            "splits": ["validation"],
            "records": len(records),
            "comparisons": len(comparisons),
            "mode": "controlled_gold_history",
            "base_model": "fake-base",
            "base_revision": "a" * 40,
            "lora_adapter": None,
            "base_runtime": dict(runtime),
            "lora_runtime": dict(runtime),
            "source_sha256": {"validation": records[0]["_source_sha256"]},
            "generation_config": {
                "max_new_tokens": 192,
                "do_sample": False,
                "num_beams": 1,
                "seed": 42,
            },
            "attention_implementation": "eager",
            "python_hash_seed": "42",
            "chat_template_kwargs": {},
            "output": str(comparison_path.resolve()),
            "output_sha256": file_sha256(comparison_path),
        }
        manifest_path = root / "validation.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return comparisons, comparison_path, manifest, manifest_path

    def _write_direct_final_test_fixture(self, root):
        records = load_evaluation_records(DEFAULT_DATA_ROOT, splits=("test",))
        comparisons = generate_comparisons(
            records,
            FakeGenerator("foundation-v1", "base"),
            FakeGenerator("hutao-lora-v1", "lora"),
            mode="controlled_gold_history",
            generation_config={
                "max_new_tokens": 192,
                "do_sample": False,
                "num_beams": 1,
            },
            seed=42,
        )
        comparison_path = root / "test.jsonl"
        comparison_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in comparisons
            ),
            encoding="utf-8",
        )
        adapter, run_manifest_path, model = make_final_adapter_fixture(root)
        runtime = {
            "model_name_or_path": model["name"],
            "revision": model["revision"],
            "resolved_commit": model["revision"],
            "dtype_requested": model["dtype"],
            "dtype_actual_first_parameter": "torch.%s" % model["dtype"],
            "first_parameter_device": "cuda:0",
            "attention_implementation_requested": "eager",
            "attention_implementation_resolved": "eager",
            "chat_template_sha256": "template-sha",
            "bos_token_id": None,
            "eos_token_id": 2,
            "pad_token_id": 2,
            "padding_side": "right",
            "chat_template_kwargs": {},
            "cuda_device_count": 1,
            "hf_device_map": {"": "0"},
        }
        lora_runtime = dict(runtime)
        lora_runtime.update(
            {
                "adapter_path": str(adapter),
                "adapter_sha256": file_sha256(
                    adapter / "adapter_model.safetensors"
                ),
                "adapter_config_sha256": file_sha256(
                    adapter / "adapter_config.json"
                ),
            }
        )
        manifest = {
            "schema_version": "module_d.generation_manifest.v1",
            "splits": ["test"],
            "records": len(records),
            "comparisons": len(comparisons),
            "mode": "controlled_gold_history",
            "base_model": model["name"],
            "base_revision": model["revision"],
            "lora_adapter": str(adapter),
            "base_runtime": dict(runtime),
            "lora_runtime": lora_runtime,
            "source_sha256": {"test": records[0]["_source_sha256"]},
            "adapter_provenance": "training_final",
            "selected_adapter_sha256": lora_runtime["adapter_sha256"],
            "selected_adapter_config_sha256": lora_runtime[
                "adapter_config_sha256"
            ],
            "experiment_config_sha256": "c" * 64,
            "run_manifest": str(run_manifest_path),
            "run_manifest_sha256": file_sha256(run_manifest_path),
            "selection_manifest": None,
            "selection_manifest_sha256": None,
            "generation_config": {
                "max_new_tokens": 192,
                "do_sample": False,
                "num_beams": 1,
                "seed": 42,
            },
            "attention_implementation": "eager",
            "python_hash_seed": "42",
            "chat_template_kwargs": {},
            "output": str(comparison_path.resolve()),
            "output_sha256": file_sha256(comparison_path),
        }
        manifest_path = root / "test.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return comparisons, comparison_path, manifest, manifest_path, adapter

    def test_generation_manifest_binds_every_frozen_turn(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (
                comparisons,
                comparison_path,
                manifest,
                manifest_path,
            ) = self._write_validation_generation_fixture(root)
            validated = validate_generation_manifest(
                manifest_path, comparison_path, comparisons
            )
            self.assertEqual(validated["comparisons"], 50)

            missing_bos = copy.deepcopy(manifest)
            del missing_bos["base_runtime"]["bos_token_id"]
            manifest_path.write_text(json.dumps(missing_bos), encoding="utf-8")
            with self.assertRaises(BlindReviewError):
                validate_generation_manifest(
                    manifest_path, comparison_path, comparisons
                )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            mismatched_template = copy.deepcopy(manifest)
            mismatched_template["chat_template_kwargs"] = {
                "enable_thinking": False
            }
            manifest_path.write_text(
                json.dumps(mismatched_template), encoding="utf-8"
            )
            with self.assertRaises(BlindReviewError):
                validate_generation_manifest(
                    manifest_path, comparison_path, comparisons
                )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            comparisons[0]["latest_user_message"] += "（篡改）"
            comparison_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in comparisons
                ),
                encoding="utf-8",
            )
            manifest["output_sha256"] = file_sha256(comparison_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            tampered_rows = load_comparisons(comparison_path)
            with self.assertRaises(BlindReviewError):
                validate_generation_manifest(
                    manifest_path, comparison_path, tampered_rows
                )

    def test_direct_final_test_manifest_supports_downstream_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (
                comparisons,
                comparison_path,
                manifest,
                manifest_path,
                adapter,
            ) = self._write_direct_final_test_fixture(root)
            validated = validate_generation_manifest(
                manifest_path, comparison_path, comparisons
            )
            self.assertEqual(validated["adapter_provenance"], "training_final")

            changed_hash = copy.deepcopy(manifest)
            changed_hash["selected_adapter_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(changed_hash), encoding="utf-8")
            with self.assertRaises(BlindReviewError):
                validate_generation_manifest(
                    manifest_path, comparison_path, comparisons
                )

            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaises(BlindReviewError):
                validate_generation_manifest(
                    manifest_path, comparison_path, comparisons
                )

    def test_shared_v2_rubric_has_three_persona_layers_and_complete_anchors(self):
        self.assertEqual(
            PERSONA_LAYERS,
            ("surface_style", "knowledge_relationship", "value_worldview"),
        )
        self.assertEqual(GUARD_DIMENSIONS, ("task_completion", "safety_ethics"))
        self.assertEqual(SCORE_DIMENSIONS, PERSONA_LAYERS + GUARD_DIMENSIONS)
        self.assertEqual(RUBRIC_SCHEMA_VERSION, "module_d.rubric.v2")
        for dimension in SCORE_DIMENSIONS:
            dimension_rubric = SCORE_RUBRICS[dimension]
            self.assertGreaterEqual(len(dimension_rubric["indicators"]), 2)
            self.assertLessEqual(len(dimension_rubric["indicators"]), 3)
            self.assertEqual(set(dimension_rubric["score_anchors"]), set(range(1, 6)))
            self.assertTrue(
                all(
                    isinstance(anchor, str) and len(anchor) >= 20
                    for anchor in dimension_rubric["score_anchors"].values()
                )
            )
        self.assertIn("fabricated_lore", ERROR_TAGS)
        self.assertIn("value_conflict", ERROR_TAGS)
        payload = public_rubric_payload()
        self.assertEqual(payload["schema_version"], RUBRIC_SCHEMA_VERSION)
        self.assertEqual(len(rubric_sha256()), 64)
        payload["score_dimensions"].append("tampered")
        self.assertNotIn("tampered", public_rubric_payload()["score_dimensions"])

    def test_blinding_is_seeded_and_model_labels_are_not_in_csv_rows(self):
        comparisons, _, _ = generate_fake_comparisons()
        rows_one, key_one = build_blind_review(comparisons, seed=31415)
        rows_two, key_two = build_blind_review(comparisons, seed=31415)
        self.assertEqual(rows_one, rows_two)
        self.assertEqual(key_one, key_two)
        self.assertEqual(tuple(rows_one[0].keys()), REVIEW_FIELDS)
        serialized_rows = json.dumps(rows_one, ensure_ascii=False)
        self.assertNotIn("foundation-v1", serialized_rows)
        self.assertNotIn("hutao-lora-v1", serialized_rows)
        serialized_key = json.dumps(key_one, ensure_ascii=False)
        self.assertIn("foundation-v1", serialized_key)
        self.assertIn("hutao-lora-v1", serialized_key)
        self.assertEqual(key_one["schema_version"], KEY_SCHEMA_VERSION)
        self.assertEqual(key_one["rubric_sha256"], rubric_sha256())
        self.assertEqual(
            key_one["preference_dimensions"], list(PREFERENCE_DIMENSIONS)
        )

    def test_blinding_rejects_tampered_prompt_or_stale_prompt_hash(self):
        comparisons, _, _ = generate_fake_comparisons()
        comparisons[0]["lora"]["prompt_messages"][-1]["content"] = "tampered"
        with self.assertRaises(BlindReviewError):
            build_blind_review(comparisons, seed=42)

    def test_scoring_unblinds_and_summarizes_by_model_and_capability(self):
        comparisons, _, _ = generate_fake_comparisons()
        rows, key = build_blind_review(comparisons, seed=9)
        for index, row in enumerate(rows):
            key_row = key["rows"][row["review_id"]]
            for side_name in ("a", "b"):
                is_lora = key_row[side_name]["variant"] == "lora"
                for dimension in SCORE_DIMENSIONS:
                    row[dimension + "_" + side_name + "_score"] = (
                        "5" if is_lora else "3"
                    )
                row["critical_failure_" + side_name] = (
                    "yes" if (not is_lora and index == 0) else "no"
                )
            row["preference"] = "A" if key_row["a"]["variant"] == "lora" else "B"
            set_layer_preferences(row, row["preference"])
            row["reviewer_id"] = "reviewer-01"
            if "yes" in (row["critical_failure_a"], row["critical_failure_b"],):
                row["notes"] = "Base 在该题出现预注册的严重失败。"

        summary = score_reviews(rows, key)
        base_stats = summary["by_model"]["foundation-v1"]
        lora_stats = summary["by_model"]["hutao-lora-v1"]
        self.assertEqual(base_stats["mean_score"], 3.0)
        self.assertEqual(lora_stats["mean_score"], 5.0)
        self.assertEqual(base_stats["persona_mean_score"], 3.0)
        self.assertEqual(lora_stats["persona_mean_score"], 5.0)
        self.assertEqual(base_stats["critical_failures"], 1)
        self.assertEqual(lora_stats["critical_failures"], 0)
        self.assertEqual(lora_stats["preference"]["wins"], 2)
        self.assertEqual(base_stats["preference"]["losses"], 2)
        self.assertEqual(lora_stats["persona_preference"]["wins"], 6)
        self.assertEqual(
            lora_stats["preference_by_layer"]["surface_style"]["wins"], 2
        )
        self.assertEqual(lora_stats["safety_gate_pass_rate"], 1.0)
        self.assertEqual(summary["lora_minus_base"]["mean_score"], 2.0)
        self.assertEqual(summary["lora_minus_base"]["persona_mean_score"], 2.0)
        self.assertEqual(
            summary["persona_mean_score"], {"base": 3.0, "lora": 5.0, "delta": 2.0}
        )
        self.assertEqual(
            summary["before_after"]["knowledge_relationship"],
            {"base": 3.0, "lora": 5.0, "delta": 2.0},
        )
        self.assertEqual(
            summary["lora_minus_base_by_capability"]["daily_chat"]["mean_score"], 2.0,
        )
        self.assertEqual(len(summary["per_review"]), 2)
        self.assertEqual(summary["per_review"][0]["preference_variant"], "lora")
        self.assertEqual(
            summary["per_review"][0]["preference_variant_by_layer"][
                "value_worldview"
            ],
            "lora",
        )
        self.assertIn("daily_chat", summary["by_capability"])
        self.assertEqual(summary["reviewers"], ["reviewer-01"])
        self.assertEqual(summary["schema_version"], "module_d.review_summary.v2")
        self.assertTrue(summary["decision"]["persona_or_preference_gain_pass"])
        self.assertIn("human_quantitative_pass", summary["decision"])
        self.assertTrue(
            summary["decision"]["automatic_quantitative_pass_deprecated"]
        )

    def test_persona_mean_excludes_guards_and_core_regressions_do_not_compensate(self):
        comparisons, _, _ = generate_fake_comparisons()
        rows, key = build_blind_review(comparisons, seed=10)
        for row in rows:
            key_row = key["rows"][row["review_id"]]
            for side_name in ("a", "b"):
                is_lora = key_row[side_name]["variant"] == "lora"
                row["surface_style_" + side_name + "_score"] = (
                    "5" if is_lora else "4"
                )
                row["knowledge_relationship_" + side_name + "_score"] = (
                    "3" if is_lora else "4"
                )
                row["value_worldview_" + side_name + "_score"] = (
                    "5" if is_lora else "4"
                )
                row["task_completion_" + side_name + "_score"] = (
                    "3" if is_lora else "4"
                )
                row["safety_ethics_" + side_name + "_score"] = "5"
                row["critical_failure_" + side_name] = "no"
            lora_preference = "A" if key_row["a"]["variant"] == "lora" else "B"
            set_layer_preferences(row, lora_preference)
            row["preference"] = lora_preference
            row["reviewer_id"] = "reviewer-01"

        summary = score_reviews(rows, key)
        self.assertEqual(summary["persona_mean_score"]["base"], 4.0)
        self.assertEqual(summary["persona_mean_score"]["lora"], 4.3333)
        self.assertEqual(summary["persona_mean_score"]["delta"], 0.3333)
        self.assertTrue(summary["decision"]["persona_or_preference_gain_pass"])
        self.assertEqual(
            summary["decision"]["persona_preference_decisive_win_rate"], 1.0
        )
        self.assertFalse(summary["decision"]["core_utility_non_regression_pass"])
        self.assertEqual(summary["decision"]["knowledge_relationship_delta"], -1.0)
        self.assertEqual(summary["decision"]["task_completion_delta"], -1.0)

    def test_score_validation_rejects_bad_score_flag_and_preference(self):
        comparisons, _, _ = generate_fake_comparisons()
        clean_rows, key = build_blind_review(comparisons, seed=11)
        for row in clean_rows:
            for dimension in SCORE_DIMENSIONS:
                row[dimension + "_a_score"] = "4"
                row[dimension + "_b_score"] = "4"
            row["critical_failure_a"] = "no"
            row["critical_failure_b"] = "no"
            row["preference"] = "Tie"
            set_layer_preferences(row, "Tie")
            row["reviewer_id"] = "reviewer-01"

        mutations = (
            ("score", "surface_style_a_score", "6"),
            ("critical failure", "critical_failure_b", "maybe"),
            ("layer preference", "knowledge_relationship_preference", "LoRA"),
            ("preference", "preference", "LoRA"),
        )
        for label, field, value in mutations:
            with self.subTest(label=label):
                bad_rows = copy.deepcopy(clean_rows)
                bad_rows[0][field] = value
                with self.assertRaises(ReviewValidationError):
                    score_reviews(bad_rows, key)

    def test_critical_failure_requires_written_notes(self):
        comparisons, _, _ = generate_fake_comparisons()
        rows, key = build_blind_review(comparisons, seed=12)
        for row in rows:
            for dimension in SCORE_DIMENSIONS:
                row[dimension + "_a_score"] = "4"
                row[dimension + "_b_score"] = "4"
            row["critical_failure_a"] = "yes"
            row["critical_failure_b"] = "no"
            row["preference"] = "Tie"
            set_layer_preferences(row, "Tie")
            row["reviewer_id"] = "reviewer-01"
        with self.assertRaises(ReviewValidationError):
            score_reviews(rows, key)

    def test_scoring_rejects_multiple_labels_for_one_variant(self):
        comparisons, _, _ = generate_fake_comparisons()
        rows, key = build_blind_review(comparisons, seed=13)
        for row in rows:
            for dimension in SCORE_DIMENSIONS:
                row[dimension + "_a_score"] = "4"
                row[dimension + "_b_score"] = "4"
            row["critical_failure_a"] = "no"
            row["critical_failure_b"] = "no"
            row["preference"] = "Tie"
            set_layer_preferences(row, "Tie")
            row["reviewer_id"] = "reviewer-01"
        second_key_row = key["rows"][rows[1]["review_id"]]
        base_side = "a" if second_key_row["a"]["variant"] == "base" else "b"
        second_key_row[base_side]["model_label"] = "foundation-alias"
        with self.assertRaises(ReviewValidationError):
            score_reviews(rows, key)

    def test_scoring_rejects_a_tampered_embedded_rubric(self):
        comparisons, _, _ = generate_fake_comparisons()
        rows, key = build_blind_review(comparisons, seed=15)
        for row in rows:
            for dimension in SCORE_DIMENSIONS:
                row[dimension + "_a_score"] = "4"
                row[dimension + "_b_score"] = "4"
            row["critical_failure_a"] = "no"
            row["critical_failure_b"] = "no"
            row["preference"] = "Tie"
            set_layer_preferences(row, "Tie")
            row["reviewer_id"] = "reviewer-01"
        key["rubric"]["scoring_rules"][0] += "（篡改）"
        with self.assertRaises(ReviewValidationError):
            score_reviews(rows, key)

    def test_protocol_rejects_missing_turn_replaced_by_t99(self):
        records = load_evaluation_records(DEFAULT_DATA_ROOT, splits=("test",))
        comparisons = generate_comparisons(
            records,
            FakeGenerator("foundation-v1", "base"),
            FakeGenerator("hutao-lora-v1", "lora"),
            mode="controlled_gold_history",
            seed=42,
        )
        rows, key = build_blind_review(comparisons, seed=14)
        for row in rows:
            for dimension in SCORE_DIMENSIONS:
                row[dimension + "_a_score"] = "4"
                row[dimension + "_b_score"] = "4"
            row["critical_failure_a"] = "no"
            row["critical_failure_b"] = "no"
            row["preference"] = "Tie"
            set_layer_preferences(row, "Tie")
            row["reviewer_id"] = "reviewer-01"
        key["rows"][rows[0]["review_id"]][
            "eval_id"
        ] = "test:HT-BUS-G08-V1:controlled_gold_history:T99"
        summary = score_reviews(rows, key)
        self.assertFalse(summary["decision"]["protocol_complete"])
        self.assertTrue(summary["evaluation_scope"]["missing_eval_ids"])
        self.assertTrue(summary["evaluation_scope"]["extra_eval_ids"])

    def test_scoring_rejects_modified_prompt_or_response_and_missing_reviewer(self):
        comparisons, _, _ = generate_fake_comparisons()
        clean_rows, key = build_blind_review(comparisons, seed=17)
        for row in clean_rows:
            for dimension in SCORE_DIMENSIONS:
                row[dimension + "_a_score"] = "4"
                row[dimension + "_b_score"] = "4"
            row["critical_failure_a"] = "no"
            row["critical_failure_b"] = "no"
            row["preference"] = "Tie"
            set_layer_preferences(row, "Tie")
            row["reviewer_id"] = "reviewer-01"

        mutations = (
            ("latest_user_message", "edited"),
            ("context_a", "edited"),
            ("response_b", "edited"),
            ("reviewer_id", ""),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                bad_rows = copy.deepcopy(clean_rows)
                bad_rows[0][field] = value
                with self.assertRaises(ReviewValidationError):
                    score_reviews(bad_rows, key)

    def test_review_csv_and_key_round_trip_with_standard_library(self):
        comparisons, _, _ = generate_fake_comparisons()
        rows, key = build_blind_review(comparisons, seed=99)
        for row in rows:
            for dimension in SCORE_DIMENSIONS:
                row[dimension + "_a_score"] = "4"
                row[dimension + "_b_score"] = "3"
            row["critical_failure_a"] = "no"
            row["critical_failure_b"] = "no"
            row["preference"] = "A"
            set_layer_preferences(row, "A")
            row["reviewer_id"] = "reviewer-01"
        with tempfile.TemporaryDirectory() as temporary_directory:
            review_path = Path(temporary_directory) / "review.csv"
            key_path = Path(temporary_directory) / "key.json"
            rubric_path = Path(temporary_directory) / "rubric.json"
            write_review_csv(rows, review_path)
            write_blind_key(key, key_path)
            write_rubric_json(rubric_path)
            loaded_rows = load_scored_csv(review_path)
            loaded_key = load_blind_key(key_path)
            loaded_rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded_rows, rows)
            self.assertEqual(loaded_key, key)
            self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)
            self.assertEqual(loaded_rubric, public_rubric_payload())
            self.assertNotIn("foundation-v1", rubric_path.read_text(encoding="utf-8"))
            summary = score_reviews(loaded_rows, loaded_key)
            self.assertEqual(summary["review_rows"], 2)

    def test_critical_safety_gate_is_non_compensatory(self):
        records = load_evaluation_records(DEFAULT_DATA_ROOT, splits=("test",))
        comparisons = generate_comparisons(
            records,
            FakeGenerator("foundation-v1", "base"),
            FakeGenerator("hutao-lora-v1", "lora"),
            mode="controlled_gold_history",
            seed=42,
        )
        rows, key = build_blind_review(comparisons, seed=42)
        for row in rows:
            key_row = key["rows"][row["review_id"]]
            for side_name in ("a", "b"):
                for dimension in SCORE_DIMENSIONS:
                    row[dimension + "_" + side_name + "_score"] = "5"
                row["critical_failure_" + side_name] = "no"
            row["preference"] = "Tie"
            set_layer_preferences(row, "Tie")
            row["reviewer_id"] = "reviewer-01"
            if key_row["eval_id"] == ("test:HT-CRI-G08-V1:controlled_gold_history:T01"):
                lora_side = "a" if key_row["a"]["variant"] == "lora" else "b"
                row["critical_failure_" + lora_side] = "yes"
                row["notes"] = "LoRA 对关键危机场景响应不足。"

        summary = score_reviews(rows, key)
        self.assertTrue(summary["critical_safety_gate"]["base"]["pass"])
        self.assertFalse(summary["critical_safety_gate"]["lora"]["pass"])
        self.assertFalse(summary["decision"]["no_new_lora_critical_failures"])
        self.assertFalse(summary["decision"]["automatic_quantitative_pass"])


if __name__ == "__main__":
    unittest.main()
