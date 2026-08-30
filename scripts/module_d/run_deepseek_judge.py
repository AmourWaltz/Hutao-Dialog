#!/usr/bin/env python3
"""Fill Module D blind-judge requests with the DeepSeek V4 Chat API.

The API key is read only from an environment variable (``DEEPSEEK_API_KEY``
by default).  It is never accepted as a command-line value and is never
written to the scored JSONL or audit JSON.  The runner is resumable: every
validated judgment is atomically persisted, and a later invocation skips
already completed request IDs.
"""

from __future__ import print_function

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request


WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts.module_d.evaluate_automatic import (  # noqa: E402
    AutomaticEvaluationError,
    DEEPSEEK_RUNNER_SCHEMA_VERSION,
    JUDGE_REQUEST_SCHEMA_VERSION,
    _ensure_distinct_paths,
    _request_core,
    _validate_judgment,
    canonical_json_sha256,
    file_sha256,
    load_eval_config,
    load_judge_jsonl,
    load_judge_key,
    text_sha256,
)


RUNNER_SCHEMA_VERSION = DEEPSEEK_RUNNER_SCHEMA_VERSION
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
SUPPORTED_MODELS = ("deepseek-v4-pro", "deepseek-v4-flash")
REQUEST_FIELDS = {
    "schema_version",
    "request_id",
    "judge",
    "rubric_schema_version",
    "rubric_sha256",
    "config_schema_version",
    "config_sha256",
    "prompt_sha256",
    "messages",
    "required_output",
    "judgment",
}


class DeepSeekRunnerError(RuntimeError):
    """Raised when DeepSeek execution or local result validation fails."""


class DeepSeekAPIError(DeepSeekRunnerError):
    """An API error carrying whether retrying the same request is reasonable."""

    def __init__(self, message, retryable=False, status=None):
        super(DeepSeekAPIError, self).__init__(message)
        self.retryable = bool(retryable)
        self.status = status


class _NoRedirectHandler(urllib_request.HTTPRedirectHandler):
    """Keep the Bearer credential pinned to the configured official host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib_request.build_opener(_NoRedirectHandler())


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(value, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_jsonl(records, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_api_error_message(raw_body, fallback):
    """Extract a short provider message without ever including request headers."""
    try:
        body = json.loads(raw_body.decode("utf-8", errors="replace"))
    except (TypeError, ValueError):
        body = None
    message = None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
        elif isinstance(error, str):
            message = error
    if not isinstance(message, str) or not message.strip():
        message = fallback
    return " ".join(message.strip().split())[:300]


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _reject_nonfinite_constant(value):
    raise ValueError("non-finite JSON number: %s" % value)


def strict_json_loads(value):
    """Parse standards-compliant JSON while rejecting duplicate object keys."""
    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_nonfinite_constant,
    )


def call_deepseek_chat(
    messages,
    model,
    api_key,
    base_url=DEFAULT_BASE_URL,
    timeout_seconds=120,
    max_tokens=4096,
    thinking="disabled",
    reasoning_effort="high",
    urlopen=None,
):
    """Call DeepSeek's OpenAI-compatible Chat Completions endpoint once."""
    if model not in SUPPORTED_MODELS:
        raise DeepSeekRunnerError(
            "DeepSeek V4 model must be one of %r" % (SUPPORTED_MODELS,)
        )
    if base_url != DEFAULT_BASE_URL:
        raise DeepSeekRunnerError(
            "DeepSeek API endpoint must be %s" % DEFAULT_BASE_URL
        )
    if not isinstance(api_key, str) or not api_key.strip():
        raise DeepSeekRunnerError("DeepSeek API key is empty")
    if thinking not in ("enabled", "disabled"):
        raise DeepSeekRunnerError("thinking must be enabled or disabled")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise DeepSeekRunnerError("max_tokens must be a positive integer")
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
    ):
        raise DeepSeekRunnerError("timeout_seconds must be positive")
    if not isinstance(messages, list) or not messages:
        raise DeepSeekRunnerError("judge messages must be a non-empty list")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "stream": False,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "thinking": {"type": thinking},
    }
    if thinking == "enabled":
        payload["reasoning_effort"] = reasoning_effort
    endpoint = base_url.rstrip("/") + "/chat/completions"
    request = urllib_request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer %s" % api_key.strip(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    # urllib's default redirect handler can forward request headers to a new
    # location.  The official endpoint should answer directly, so redirects
    # are rejected rather than risking forwarding the Bearer credential.
    opener = urlopen or _NO_REDIRECT_OPENER.open
    try:
        with opener(request, timeout=float(timeout_seconds)) as response:
            status = int(getattr(response, "status", response.getcode()))
            raw_body = response.read()
    except urllib_error.HTTPError as exc:
        raw_body = exc.read()
        detail = _safe_api_error_message(raw_body, str(exc.reason))
        retryable = exc.code == 429 or exc.code in (500, 502, 503, 504)
        raise DeepSeekAPIError(
            "DeepSeek API HTTP %d: %s" % (exc.code, detail),
            retryable=retryable,
            status=exc.code,
        )
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise DeepSeekAPIError(
            "DeepSeek API network error: %s" % exc,
            retryable=True,
            status=None,
        )
    if status < 200 or status >= 300:
        detail = _safe_api_error_message(raw_body, "unexpected HTTP response")
        raise DeepSeekAPIError(
            "DeepSeek API HTTP %d: %s" % (status, detail),
            retryable=status == 429 or status in (500, 502, 503, 504),
            status=status,
        )
    try:
        body = strict_json_loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise DeepSeekAPIError(
            "DeepSeek API returned invalid JSON: %s" % exc,
            retryable=True,
            status=status,
        )
    if not isinstance(body, dict):
        raise DeepSeekAPIError(
            "DeepSeek API response must be an object",
            retryable=True,
            status=status,
        )
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise DeepSeekAPIError(
            "DeepSeek API response must contain exactly one choice",
            retryable=True,
            status=status,
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise DeepSeekAPIError("DeepSeek API choice is malformed", retryable=True)
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise DeepSeekAPIError(
            "DeepSeek API finish_reason is %r" % finish_reason,
            retryable=finish_reason in ("length", "insufficient_system_resource"),
            status=status,
        )
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekAPIError(
            "DeepSeek API returned empty content", retryable=True, status=status
        )
    response_model = body.get("model")
    if response_model != model:
        raise DeepSeekAPIError(
            "DeepSeek API returned model %r, expected %r"
            % (response_model, model),
            retryable=False,
            status=status,
        )
    system_fingerprint = body.get("system_fingerprint")
    if not isinstance(system_fingerprint, str) or not system_fingerprint.strip():
        raise DeepSeekAPIError(
            "DeepSeek API response has no system_fingerprint",
            retryable=True,
            status=status,
        )
    return {
        "content": content.strip(),
        "content_sha256": text_sha256(content.strip()),
        "response_id": body.get("id"),
        "response_model": response_model,
        "system_fingerprint": system_fingerprint,
        "created": body.get("created"),
        "finish_reason": finish_reason,
        "usage": body.get("usage"),
        "http_status": status,
    }


def _validate_prepared_requests(records, key):
    if not isinstance(records, list) or not records:
        raise DeepSeekRunnerError("prepared judge requests are empty")
    expected_ids = set(key["rows"])
    actual_ids = []
    validated = []
    for record in records:
        request_id = record.get("request_id") if isinstance(record, dict) else None
        if not isinstance(record, dict) or set(record) != REQUEST_FIELDS:
            raise DeepSeekRunnerError(
                "%s: prepared request fields differ from the frozen schema"
                % (request_id or "unknown request")
            )
        if record.get("schema_version") != JUDGE_REQUEST_SCHEMA_VERSION:
            raise DeepSeekRunnerError(
                "%s: unsupported judge request schema" % request_id
            )
        if not isinstance(request_id, str) or not request_id:
            raise DeepSeekRunnerError("prepared request has no request_id")
        if request_id in actual_ids:
            raise DeepSeekRunnerError("duplicate request_id %s" % request_id)
        actual_ids.append(request_id)
        if request_id not in key["rows"]:
            raise DeepSeekRunnerError("unknown request_id %s" % request_id)
        if record.get("judgment") is not None:
            raise DeepSeekRunnerError(
                "%s: input requests must have judgment=null" % request_id
            )
        if record.get("judge") != key.get("judge"):
            raise DeepSeekRunnerError("%s: judge identity differs from key" % request_id)
        core = _request_core(record)
        if canonical_json_sha256(core) != key["rows"][request_id].get(
            "request_sha256"
        ):
            raise DeepSeekRunnerError("%s: request hash differs from key" % request_id)
        if canonical_json_sha256(record.get("messages")) != record.get(
            "prompt_sha256"
        ):
            raise DeepSeekRunnerError("%s: prompt hash is invalid" % request_id)
        validated.append(copy.deepcopy(record))
    if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
        raise DeepSeekRunnerError("prepared request coverage differs from judge key")
    return validated


def _load_resumable_results(output_path, inputs, key, config):
    path = Path(output_path)
    if not path.exists():
        return {}
    existing = load_judge_jsonl(path)
    inputs_by_id = dict((item["request_id"], item) for item in inputs)
    completed = {}
    for record in existing:
        request_id = record.get("request_id") if isinstance(record, dict) else None
        if request_id not in inputs_by_id or request_id in completed:
            raise DeepSeekRunnerError(
                "existing output has an unknown or duplicate request_id: %r"
                % request_id
            )
        if _request_core(record) != _request_core(inputs_by_id[request_id]):
            raise DeepSeekRunnerError(
                "%s: existing output request core differs from input" % request_id
            )
        try:
            normalized = _validate_judgment(
                record.get("judgment"), request_id, key["rows"][request_id], config
            )
        except AutomaticEvaluationError as exc:
            raise DeepSeekRunnerError(
                "%s: existing judgment is invalid: %s" % (request_id, exc)
            )
        copied = copy.deepcopy(record)
        copied["judgment"] = normalized
        completed[request_id] = copied
    return completed


def _parse_judgment(content, request_id, key_row, config):
    try:
        parsed = strict_json_loads(content)
    except ValueError as exc:
        raise AutomaticEvaluationError("model output is not JSON: %s" % exc)
    return _validate_judgment(parsed, request_id, key_row, config)


def _persist_results(inputs, completed, output_path):
    ordered = [
        completed[item["request_id"]]
        for item in inputs
        if item["request_id"] in completed
    ]
    _atomic_write_jsonl(ordered, output_path)


def _initial_audit(args, key, requests_path, key_path, config_path):
    audit_path = Path(args.audit_json)
    identity = {
        "requests_jsonl": str(requests_path.resolve()),
        "requests_jsonl_sha256": file_sha256(requests_path),
        "key_json": str(key_path.resolve()),
        "key_json_sha256": file_sha256(key_path),
        "config_file": str(config_path.resolve()),
        "config_file_sha256": file_sha256(config_path),
        "output_jsonl": str(Path(args.output_jsonl).resolve()),
        "audit_json": str(Path(args.audit_json).resolve()),
        "judge": key["judge"],
        "provider": "deepseek",
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "credential_present": True,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort
        if args.thinking == "enabled"
        else None,
        "max_tokens": args.max_tokens,
        "api_seed_supported": False,
        "seed_scope": "blind_order_only_api_has_no_seed_parameter",
    }
    if audit_path.exists():
        try:
            with audit_path.open("r", encoding="utf-8") as handle:
                audit = json.load(handle)
        except (OSError, ValueError) as exc:
            raise DeepSeekRunnerError("invalid existing audit JSON: %s" % exc)
        if (
            not isinstance(audit, dict)
            or audit.get("schema_version") != RUNNER_SCHEMA_VERSION
            or audit.get("identity") != identity
            or not isinstance(audit.get("runs"), list)
        ):
            raise DeepSeekRunnerError(
                "existing audit JSON belongs to a different judge run"
            )
        return audit
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "identity": identity,
        "runs": [],
    }


def _provider_response_audit(response):
    """Return non-content API metadata that is safe to persist."""
    return {
        "http_status": response.get("http_status"),
        "response_id": response.get("response_id"),
        "response_model": response.get("response_model"),
        "system_fingerprint": response.get("system_fingerprint"),
        "created": response.get("created"),
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage"),
        "content_sha256": response.get("content_sha256"),
    }


def _known_audit_fingerprints(audit):
    fingerprints = set(
        call.get("system_fingerprint")
        for prior_run in audit.get("runs", [])
        for call in prior_run.get("calls", [])
        if isinstance(call.get("system_fingerprint"), str)
        and call.get("system_fingerprint")
    )
    if len(fingerprints) > 1:
        raise DeepSeekRunnerError(
            "existing audit contains multiple DeepSeek system_fingerprints"
        )
    return fingerprints


def _validate_completed_against_audit(completed, audit, key):
    """Require every resumed result to have a matching successful API call."""
    successful_calls = {}
    for prior_run in audit.get("runs", []):
        for call in prior_run.get("calls", []):
            if call.get("status") != "ok":
                continue
            request_id = call.get("request_id")
            successful_calls.setdefault(request_id, []).append(call)
    for request_id, record in completed.items():
        expected_request_sha256 = key["rows"][request_id]["request_sha256"]
        expected_judgment_sha256 = canonical_json_sha256(record["judgment"])
        matches = [
            call
            for call in successful_calls.get(request_id, [])
            if call.get("request_sha256") == expected_request_sha256
            and call.get("prompt_sha256") == record["prompt_sha256"]
            and call.get("judgment_sha256") == expected_judgment_sha256
            and call.get("response_model") == key["judge"]["model"]
            and isinstance(call.get("system_fingerprint"), str)
            and call.get("system_fingerprint")
        ]
        if not matches:
            raise DeepSeekRunnerError(
                "%s: existing judgment has no matching successful API audit call"
                % request_id
            )


def run_deepseek_judge(args, api_caller=call_deepseek_chat, sleep=time.sleep):
    config_path = Path(args.config)
    requests_path = Path(args.requests_jsonl)
    key_path = Path(args.key_json)
    output_path = Path(args.output_jsonl)
    audit_path = Path(args.audit_json)
    _ensure_distinct_paths(
        (output_path, audit_path),
        (requests_path, key_path, config_path),
    )

    config = load_eval_config(config_path)
    # Scoring-affecting provider parameters come only from the hashed config;
    # the runner CLI deliberately offers no override for them.
    args.base_url = config["judge"]["base_url"]
    args.thinking = config["judge"]["thinking"]
    args.max_tokens = int(config["judge"]["max_tokens"])
    args.reasoning_effort = None
    key = load_judge_key(key_path, require_provenance=True, config=config)
    model = key.get("judge", {}).get("model")
    if (
        model not in SUPPORTED_MODELS
        or model != config["judge"]["model"]
        or key.get("judge", {}).get("revision")
        != config["judge"]["revision"]
    ):
        raise DeepSeekRunnerError(
            "prepared DeepSeek model/revision differs from the evaluation config"
        )
    api_key = os.environ.get(args.api_key_env)
    if not isinstance(api_key, str) or not api_key.strip():
        raise DeepSeekRunnerError(
            "missing API key: export %s before running" % args.api_key_env
        )

    if output_path.exists() and not audit_path.exists():
        raise DeepSeekRunnerError(
            "existing scored output has no matching audit JSON; refuse unaudited resume"
        )
    inputs = _validate_prepared_requests(load_judge_jsonl(requests_path), key)
    audit = _initial_audit(args, key, requests_path, key_path, config_path)
    completed = _load_resumable_results(
        output_path, inputs, key, config
    )
    _validate_completed_against_audit(completed, audit, key)
    known_fingerprints = _known_audit_fingerprints(audit)
    run_audit = {
        "started_at_utc": utc_now(),
        "completed_before_resume": len(completed),
        "calls": [],
        "status": "running",
    }
    audit.pop("summary", None)
    audit["runs"].append(run_audit)
    _atomic_write_json(audit, audit_path)

    total = len(inputs)
    try:
        for index, record in enumerate(inputs, 1):
            request_id = record["request_id"]
            if request_id in completed:
                continue
            messages = copy.deepcopy(record["messages"])
            last_error = None
            attempts_used = 0
            for attempt in range(1, args.max_attempts + 1):
                attempts_used = attempt
                call_audit = {
                    "request_id": request_id,
                    "request_sha256": key["rows"][request_id]["request_sha256"],
                    "prompt_sha256": record["prompt_sha256"],
                    "attempt": attempt,
                    "started_at_utc": utc_now(),
                }
                started = time.monotonic()
                try:
                    response = api_caller(
                        messages=messages,
                        model=model,
                        api_key=api_key,
                        base_url=args.base_url,
                        timeout_seconds=args.timeout_seconds,
                        max_tokens=args.max_tokens,
                        thinking=args.thinking,
                        reasoning_effort="high",
                    )
                    fingerprint = response.get("system_fingerprint")
                    if known_fingerprints and fingerprint not in known_fingerprints:
                        call_audit.update(_provider_response_audit(response))
                        raise DeepSeekAPIError(
                            "DeepSeek system_fingerprint drifted within one judge run",
                            retryable=False,
                            status=response.get("http_status"),
                        )
                    known_fingerprints.add(fingerprint)
                    normalized = _parse_judgment(
                        response["content"],
                        request_id,
                        key["rows"][request_id],
                        config,
                    )
                    call_audit.update(_provider_response_audit(response))
                    call_audit.update(
                        {
                            "status": "ok",
                            "finished_at_utc": utc_now(),
                            "elapsed_seconds": round(
                                time.monotonic() - started, 6
                            ),
                            "judgment_sha256": canonical_json_sha256(normalized),
                        }
                    )
                    run_audit["calls"].append(call_audit)
                    # Commit API metadata before the scored row.  If the process
                    # dies between the two atomic writes, a rerun may call the API
                    # again, but it cannot silently lose the original call audit.
                    _atomic_write_json(audit, audit_path)
                    filled = copy.deepcopy(record)
                    filled["judgment"] = normalized
                    completed[request_id] = filled
                    _persist_results(inputs, completed, output_path)
                    print(
                        "[%d/%d] %s judged" % (index, total, request_id),
                        file=sys.stderr,
                    )
                    if args.delay_seconds:
                        sleep(args.delay_seconds)
                    last_error = None
                    break
                except AutomaticEvaluationError as exc:
                    last_error = "judgment validation failed: %s" % exc
                    call_audit.update(_provider_response_audit(response))
                    call_audit.update(
                        {
                            "status": "invalid_judgment",
                            "finished_at_utc": utc_now(),
                            "elapsed_seconds": round(
                                time.monotonic() - started, 6
                            ),
                            "error": last_error,
                        }
                    )
                    run_audit["calls"].append(call_audit)
                    _atomic_write_json(audit, audit_path)
                except DeepSeekAPIError as exc:
                    last_error = str(exc)
                    call_audit.update(
                        {
                            "status": "api_error",
                            "finished_at_utc": utc_now(),
                            "elapsed_seconds": round(
                                time.monotonic() - started, 6
                            ),
                            "http_status": exc.status,
                            "retryable": exc.retryable,
                            "error": last_error,
                        }
                    )
                    run_audit["calls"].append(call_audit)
                    _atomic_write_json(audit, audit_path)
                    if not exc.retryable:
                        break
                    if attempt < args.max_attempts:
                        sleep(min(2 ** (attempt - 1), 8))
            if last_error is not None:
                raise DeepSeekRunnerError(
                    "%s failed after %d attempt(s): %s"
                    % (request_id, attempts_used, last_error)
                )
        run_audit.update(
            {
                "status": "complete",
                "finished_at_utc": utc_now(),
                "completed_after_run": len(completed),
            }
        )
        audit["summary"] = {
            "status": "complete",
            "requests": total,
            "judgments": len(completed),
            "system_fingerprints": sorted(known_fingerprints),
            "output_jsonl": str(output_path.resolve()),
            "output_jsonl_sha256": file_sha256(output_path),
        }
        _atomic_write_json(audit, audit_path)
    except Exception:
        run_audit.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "completed_after_run": len(completed),
            }
        )
        _atomic_write_json(audit, audit_path)
        raise

    return {
        "status": "ok",
        "requests": total,
        "judgments": len(completed),
        "resumed": run_audit["completed_before_resume"],
        "output_jsonl": str(output_path),
        "audit_json": str(audit_path),
        "judge_model": model,
        "judge_revision": key["judge"]["revision"],
    }


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-jsonl", required=True)
    parser.add_argument("--key-json", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--audit-json",
        help="defaults to OUTPUT_JSONL.audit.json",
    )
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not isinstance(args.api_key_env, str) or not args.api_key_env.strip():
        parser.error("--api-key-env must be non-empty")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be non-negative")
    if args.audit_json is None:
        args.audit_json = args.output_jsonl + ".audit.json"
    try:
        result = run_deepseek_judge(args)
    except (AutomaticEvaluationError, DeepSeekRunnerError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
