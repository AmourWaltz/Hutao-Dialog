#!/usr/bin/env python3
"""Tests for the resumable DeepSeek V4 Module D judge runner."""

from __future__ import print_function

import argparse
import copy
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.module_d.build_review_sheet import prompt_sha256
from scripts.module_d.evaluate_automatic import (
    AutomaticEvaluationError,
    DEFAULT_CONFIG_PATH,
    SCORE_DIMENSIONS,
    _request_core,
    build_judge_requests,
    load_eval_config,
    load_judge_jsonl,
    validate_and_score_judgments,
    validate_deepseek_judge_audit,
    write_jsonl,
    write_private_json,
)
from scripts.module_d.run_deepseek_judge import (
    DEFAULT_BASE_URL,
    DeepSeekRunnerError,
    _NoRedirectHandler,
    call_deepseek_chat,
    run_deepseek_judge,
    strict_json_loads,
)


def make_comparison(index=1):
    prompt = [
        {"role": "system", "content": "你是胡桃，以符合角色设定的方式回答。"},
        {"role": "user", "content": "今天心情不好，可以陪我聊聊吗？"},
    ]
    prompt_hash = prompt_sha256(prompt)
    record_id = "HT-DEEPSEEK-%02d" % index
    return {
        "schema_version": "module_d.comparison.v1",
        "eval_id": "validation:%s:controlled_gold_history:T01" % record_id,
        "record_id": record_id,
        "split": "validation",
        "capability": "daily_chat",
        "scenario_group": "DEEPSEEK-%02d" % index,
        "seriousness": 2,
        "risk_flags": [],
        "mode": "controlled_gold_history",
        "assistant_turn_index": 1,
        "latest_user_message": "今天心情不好，可以陪我聊聊吗？",
        "gold_response": "不会进入裁判提示的参考回答。",
        "prompt_equal": True,
        "generation": {
            "seed": 1000 + index,
            "config": {
                "max_new_tokens": 64,
                "do_sample": False,
                "num_beams": 1,
            },
        },
        "base": {
            "variant": "base",
            "model_label": "foundation-v1",
            "prompt_messages": copy.deepcopy(prompt),
            "prompt_sha256": prompt_hash,
            "response": "可以。先说说今天发生了什么，我在听。",
        },
        "lora": {
            "variant": "lora",
            "model_label": "hutao-lora-v1",
            "prompt_messages": copy.deepcopy(prompt),
            "prompt_sha256": prompt_hash,
            "response": "当然可以，本堂主今天就安静听你慢慢说。",
        },
    }


def make_judgment(key, request_id, score=4):
    result = {}
    for side_name in ("a", "b"):
        response = key["rows"][request_id][side_name]["response"]
        result[side_name] = {
            dimension: {
                "score": score,
                "reason": "依据共享量表，该回答符合第 %d 档。" % score,
                "evidence": [response],
            }
            for dimension in SCORE_DIMENSIONS
        }
    return result


class FakeHTTPResponse(object):
    def __init__(self, body, status=200):
        self._body = body
        self.status = status

    def getcode(self):
        return self.status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class DeepSeekRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_eval_config(DEFAULT_CONFIG_PATH)

    def _fixture(self, root, count=1):
        comparisons = [make_comparison(index + 1) for index in range(count)]
        requests, key = build_judge_requests(
            comparisons,
            self.config,
            judge_model=self.config["judge"]["model"],
            judge_revision=self.config["judge"]["revision"],
            seed=42,
        )
        requests_path = root / "requests.jsonl"
        key_path = root / "key.json"
        output_path = root / "scored.jsonl"
        audit_path = root / "audit.json"
        write_jsonl(requests, requests_path)
        write_private_json(key, key_path)
        self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)
        args = argparse.Namespace(
            requests_jsonl=str(requests_path),
            key_json=str(key_path),
            config=str(DEFAULT_CONFIG_PATH),
            output_jsonl=str(output_path),
            audit_json=str(audit_path),
            api_key_env="DEEPSEEK_API_KEY",
            timeout_seconds=10.0,
            max_attempts=3,
            delay_seconds=0.0,
        )
        return requests, key, args, output_path, audit_path

    def test_chat_call_uses_official_json_mode_without_api_seed(self):
        captured = {}
        response_body = json.dumps(
            {
                "id": "response-1",
                "model": "deepseek-v4-pro",
                "system_fingerprint": "fp-v4-test",
                "created": 123,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "{}"},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        ).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeHTTPResponse(response_body)

        result = call_deepseek_chat(
            messages=[{"role": "user", "content": "输出 JSON"}],
            model="deepseek-v4-pro",
            api_key="secret-never-persist",
            urlopen=fake_urlopen,
        )
        request = captured["request"]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, DEFAULT_BASE_URL + "/chat/completions")
        self.assertEqual(
            request.get_header("Authorization"), "Bearer secret-never-persist"
        )
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertFalse(payload["stream"])
        self.assertNotIn("seed", payload)
        self.assertEqual(result["system_fingerprint"], "fp-v4-test")

    def test_redirects_are_disabled_to_keep_authorization_on_official_host(self):
        handler = _NoRedirectHandler()
        self.assertIsNone(
            handler.redirect_request(
                None,
                None,
                302,
                "Found",
                {},
                "https://untrusted.example/steal",
            )
        )

    def test_runner_only_fills_judgment_and_resumes_without_key_leak(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            requests, key, args, output_path, audit_path = self._fixture(root)
            calls = []

            def fake_api_caller(**kwargs):
                calls.append(copy.deepcopy(kwargs))
                request_id = requests[len(calls) - 1]["request_id"]
                judgment = make_judgment(key, request_id)
                content = json.dumps(judgment, ensure_ascii=False)
                return {
                    "content": content,
                    "content_sha256": "d" * 64,
                    "response_id": "response-%d" % len(calls),
                    "response_model": "deepseek-v4-pro",
                    "system_fingerprint": "fp-v4-test",
                    "created": 123,
                    "finish_reason": "stop",
                    "usage": {"total_tokens": 100},
                    "http_status": 200,
                }

            with mock.patch.object(
                __import__(
                    "scripts.module_d.run_deepseek_judge", fromlist=["load_judge_key"]
                ),
                "load_judge_key",
                return_value=key,
            ), mock.patch.dict(
                os.environ, {"DEEPSEEK_API_KEY": "secret-never-persist"}
            ):
                result = run_deepseek_judge(
                    args, api_caller=fake_api_caller, sleep=lambda _seconds: None
                )
                self.assertEqual(result["judgments"], 1)
                scored = load_judge_jsonl(output_path)
                self.assertEqual(_request_core(scored[0]), _request_core(requests[0]))
                validate_and_score_judgments(scored, key, self.config)
                audit_binding = validate_deepseek_judge_audit(
                    audit_path,
                    output_path,
                    scored,
                    key,
                    self.config,
                    args.key_json,
                    args.config,
                )
                self.assertEqual(audit_binding["system_fingerprint"], "fp-v4-test")
                first_hash = output_path.read_bytes()

                second_result = run_deepseek_judge(
                    args, api_caller=fake_api_caller, sleep=lambda _seconds: None
                )
                self.assertEqual(second_result["resumed"], 1)
                self.assertEqual(len(calls), 1)
                self.assertEqual(first_hash, output_path.read_bytes())

            combined_artifacts = (
                output_path.read_text(encoding="utf-8")
                + audit_path.read_text(encoding="utf-8")
            )
            self.assertNotIn("secret-never-persist", combined_artifacts)
            self.assertEqual(
                stat.S_IMODE(output_path.stat().st_mode),
                0o600,
            )
            self.assertEqual(stat.S_IMODE(audit_path.stat().st_mode), 0o600)
            self.assertEqual(calls[0]["messages"], requests[0]["messages"])
            self.assertEqual(calls[0]["thinking"], "disabled")
            self.assertEqual(calls[0]["max_tokens"], 4096)

    def test_deepseek_audit_rejects_a_valid_but_tampered_judgment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            requests, key, args, output_path, audit_path = self._fixture(root)

            def fake_api_caller(**unused_kwargs):
                content = json.dumps(
                    make_judgment(key, requests[0]["request_id"]),
                    ensure_ascii=False,
                )
                return {
                    "content": content,
                    "content_sha256": "d" * 64,
                    "response_id": "response-1",
                    "response_model": "deepseek-v4-pro",
                    "system_fingerprint": "fp-v4-test",
                    "created": 123,
                    "finish_reason": "stop",
                    "usage": {"total_tokens": 100},
                    "http_status": 200,
                }

            runner_module = __import__(
                "scripts.module_d.run_deepseek_judge", fromlist=["load_judge_key"]
            )
            with mock.patch.object(
                runner_module, "load_judge_key", return_value=key
            ), mock.patch.dict(
                os.environ, {"DEEPSEEK_API_KEY": "secret"}
            ):
                run_deepseek_judge(args, api_caller=fake_api_caller)

            tampered = load_judge_jsonl(output_path)
            tampered[0]["judgment"] = make_judgment(
                key, requests[0]["request_id"], score=3
            )
            write_jsonl(tampered, output_path)
            with self.assertRaises(AutomaticEvaluationError):
                validate_deepseek_judge_audit(
                    audit_path,
                    output_path,
                    tampered,
                    key,
                    self.config,
                    args.key_json,
                    args.config,
                )

    def test_invalid_judgment_retries_the_identical_frozen_prompt(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            requests, key, args, output_path, unused_audit = self._fixture(root)
            prompts = []

            def fake_api_caller(**kwargs):
                prompts.append(copy.deepcopy(kwargs["messages"]))
                if len(prompts) == 1:
                    content = "{}"
                else:
                    content = json.dumps(
                        make_judgment(key, requests[0]["request_id"]),
                        ensure_ascii=False,
                    )
                return {
                    "content": content,
                    "content_sha256": "e" * 64,
                    "response_id": "response-%d" % len(prompts),
                    "response_model": "deepseek-v4-pro",
                    "system_fingerprint": "fp-v4-test",
                    "created": 123,
                    "finish_reason": "stop",
                    "usage": {"total_tokens": 100},
                    "http_status": 200,
                }

            runner_module = __import__(
                "scripts.module_d.run_deepseek_judge", fromlist=["load_judge_key"]
            )
            with mock.patch.object(
                runner_module, "load_judge_key", return_value=key
            ), mock.patch.dict(
                os.environ, {"DEEPSEEK_API_KEY": "secret"}
            ):
                run_deepseek_judge(
                    args, api_caller=fake_api_caller, sleep=lambda _seconds: None
                )
            self.assertEqual(len(prompts), 2)
            self.assertEqual(prompts[0], requests[0]["messages"])
            self.assertEqual(prompts[1], requests[0]["messages"])
            self.assertEqual(len(load_judge_jsonl(output_path)), 1)

    def test_resume_rejects_scored_output_without_api_audit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            requests, key, args, output_path, audit_path = self._fixture(root)
            filled = copy.deepcopy(requests[0])
            filled["judgment"] = make_judgment(key, requests[0]["request_id"])
            write_jsonl([filled], output_path)
            calls = []

            def fake_api_caller(**kwargs):
                calls.append(kwargs)
                raise AssertionError("API must not be called")

            runner_module = __import__(
                "scripts.module_d.run_deepseek_judge", fromlist=["load_judge_key"]
            )
            with mock.patch.object(
                runner_module, "load_judge_key", return_value=key
            ), mock.patch.dict(
                os.environ, {"DEEPSEEK_API_KEY": "secret"}
            ):
                with self.assertRaises(DeepSeekRunnerError):
                    run_deepseek_judge(args, api_caller=fake_api_caller)
            self.assertEqual(calls, [])
            self.assertFalse(audit_path.exists())

    def test_invalid_judgment_fingerprint_participates_in_drift_check(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            requests, key, args, output_path, audit_path = self._fixture(root)
            calls = []

            def fake_api_caller(**unused_kwargs):
                calls.append(True)
                valid = len(calls) > 1
                content = (
                    json.dumps(
                        make_judgment(key, requests[0]["request_id"]),
                        ensure_ascii=False,
                    )
                    if valid
                    else "{}"
                )
                return {
                    "content": content,
                    "content_sha256": ("f" if valid else "e") * 64,
                    "response_id": "response-%d" % len(calls),
                    "response_model": "deepseek-v4-pro",
                    "system_fingerprint": "fp-%d" % len(calls),
                    "created": 123,
                    "finish_reason": "stop",
                    "usage": {"total_tokens": 100},
                    "http_status": 200,
                }

            runner_module = __import__(
                "scripts.module_d.run_deepseek_judge", fromlist=["load_judge_key"]
            )
            with mock.patch.object(
                runner_module, "load_judge_key", return_value=key
            ), mock.patch.dict(
                os.environ, {"DEEPSEEK_API_KEY": "secret"}
            ):
                with self.assertRaises(DeepSeekRunnerError):
                    run_deepseek_judge(
                        args,
                        api_caller=fake_api_caller,
                        sleep=lambda _seconds: None,
                    )
            self.assertEqual(len(calls), 2)
            self.assertFalse(output_path.exists())
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(
                audit["runs"][0]["calls"][0]["system_fingerprint"], "fp-1"
            )
            self.assertEqual(audit["runs"][0]["calls"][0]["response_id"], "response-1")
            self.assertEqual(audit["runs"][0]["calls"][1]["system_fingerprint"], "fp-2")

    def test_missing_key_fails_before_any_api_call(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            unused_requests, key, args, output_path, audit_path = self._fixture(root)
            calls = []

            def fake_api_caller(**kwargs):
                calls.append(kwargs)
                raise AssertionError("API must not be called")

            runner_module = __import__(
                "scripts.module_d.run_deepseek_judge", fromlist=["load_judge_key"]
            )
            with mock.patch.object(
                runner_module, "load_judge_key", return_value=key
            ), mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(DeepSeekRunnerError):
                    run_deepseek_judge(args, api_caller=fake_api_caller)
            self.assertEqual(calls, [])
            self.assertFalse(output_path.exists())
            self.assertFalse(audit_path.exists())

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            strict_json_loads('{"a": 1, "a": 2}')
        with self.assertRaises(ValueError):
            strict_json_loads('{"score": NaN}')


if __name__ == "__main__":
    unittest.main()
