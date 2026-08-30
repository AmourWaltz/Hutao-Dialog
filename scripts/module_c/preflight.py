#!/usr/bin/env python3
"""Audit chat-template boundaries, sequence lengths, and supervised spans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from scripts.module_c.common import (
    ExperimentError,
    environment_snapshot,
    load_json,
    load_jsonl,
    sha256_file,
    verify_sha256,
    workspace_path,
    write_json,
)
from scripts.module_c.tokenization import (
    tokenization_summary,
    tokenize_completion_example,
)


DEFAULT_CONFIG = "configs/module_c/hutao_qwen3_1p7b_lora_bf16.json"
DEFAULT_OUTPUT = "output/module_c_hutao/preflight.json"
PREFLIGHT_SPLITS = ("train", "validation")


class OfflineChatTemplateTokenizer:
    """Exact no-tools renderer backed by an offline tokenizer snapshot."""

    def __init__(self, tokenizer_json: Path) -> None:
        try:
            from tokenizers import Tokenizer
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError as exc:
            raise ExperimentError(
                "The optional offline mode requires tokenizers and Jinja2"
            ) from exc
        # Keep the snapshot path instead of resolving the Hub symlink to its
        # blob; tokenizer_config.json is adjacent inside the snapshot.
        self.tokenizer_json = tokenizer_json.absolute()
        self._tokenizer = Tokenizer.from_file(str(self.tokenizer_json))
        self.eos_token_id = self._tokenizer.token_to_id("<|im_end|>")
        if self.eos_token_id is None:
            raise ExperimentError("Offline tokenizer has no <|im_end|> token")
        tokenizer_config_path = self.tokenizer_json.with_name("tokenizer_config.json")
        if not tokenizer_config_path.is_file():
            raise ExperimentError(
                "Offline audit also requires adjacent tokenizer_config.json"
            )
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        chat_template = tokenizer_config.get("chat_template")
        if not isinstance(chat_template, str) or not chat_template:
            raise ExperimentError("Offline tokenizer config has no chat_template")
        self.tokenizer_config_path = tokenizer_config_path.absolute()
        self.chat_template = chat_template
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True,
        )
        self._chat_template = environment.from_string(chat_template)

    def _render(
        self,
        messages: Sequence[Mapping[str, str]],
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> str:
        return self._chat_template.render(
            messages=list(messages),
            add_generation_prompt=add_generation_prompt,
            tools=None,
            **dict(template_kwargs)
        )

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: Any = None,
        **template_kwargs: Any
    ) -> Any:
        rendered = self._render(messages, add_generation_prompt, template_kwargs)
        if not tokenize:
            return rendered
        if return_tensors is not None:
            raise ExperimentError("Offline tokenizer only supports plain token ids")
        return self._tokenizer.encode(rendered, add_special_tokens=False).ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def audit_metadata(self) -> Dict[str, Any]:
        return {
            "tokenizer_json": str(self.tokenizer_json),
            "tokenizer_json_sha256": sha256_file(self.tokenizer_json),
            "tokenizer_config_json": str(self.tokenizer_config_path),
            "tokenizer_config_json_sha256": sha256_file(self.tokenizer_config_path),
            "chat_template_sha256": hashlib.sha256(
                self.chat_template.encode("utf-8")
            ).hexdigest(),
            "eos_token_id": self.eos_token_id,
            "renderer": "registered_jinja_chat_template_without_tools",
        }


def load_tokenizer(config: Mapping[str, Any], tokenizer_json: str = None) -> Any:
    if tokenizer_json:
        return OfflineChatTemplateTokenizer(workspace_path(tokenizer_json))

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ExperimentError(
            "Transformers is unavailable. Install requirements-module-c.txt or "
            "pass --tokenizer-json for an offline structural audit."
        ) from exc
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"], revision=model_config["revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def run_preflight(
    config_path: Path, output_path: Path, tokenizer_json: str = None
) -> Dict[str, Any]:
    config = load_json(config_path)
    tokenizer = load_tokenizer(config, tokenizer_json)
    data_config = config["data"]
    derived_dir = workspace_path(data_config["derived_dir"])
    max_length = int(data_config["max_length"])
    chat_template_kwargs = dict(config["model"].get("chat_template_kwargs", {}))
    split_results: Dict[str, Any] = {}

    # Test is deliberately excluded: Module D opens it only after validation
    # has frozen a checkpoint and written a selection manifest.
    for split in PREFLIGHT_SPLITS:
        path = derived_dir / "{}.jsonl".format(split)
        verify_sha256(path, data_config["expected_derived_sha256"][split])
        examples = load_jsonl(path)
        rows: List[Dict[str, Any]] = []
        decoded_samples: List[Dict[str, Any]] = []
        for index, example in enumerate(examples):
            tokenized = tokenize_completion_example(
                example,
                tokenizer,
                max_length,
                chat_template_kwargs=chat_template_kwargs,
            )
            rows.append(tokenized)
            if index < 2 and hasattr(tokenizer, "decode"):
                supervised = [
                    token_id
                    for token_id, label in zip(
                        tokenized["input_ids"], tokenized["labels"]
                    )
                    if label != -100
                ]
                decoded_samples.append(
                    {
                        "id": example["id"],
                        "supervised_text": tokenizer.decode(supervised),
                    }
                )
        summary = tokenization_summary(rows)
        summary.update(
            {
                "file": str(path.relative_to(workspace_path("."))),
                "sha256": sha256_file(path),
                "over_max_length": sum(
                    int(row["sequence_length"]) > max_length for row in rows
                ),
                "decoded_supervision_samples": decoded_samples,
            }
        )
        split_results[split] = summary

    tokenizer_artifact = (
        tokenizer.audit_metadata()
        if hasattr(tokenizer, "audit_metadata")
        else {
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "chat_template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode("utf-8")
            ).hexdigest(),
        }
    )
    tokenizer_artifact["chat_template_kwargs"] = chat_template_kwargs
    result: Dict[str, Any] = {
        "status": "pass",
        "scope": "tokenizer_and_loss_mask_preflight_only",
        "model": config["model"],
        "max_length": max_length,
        "tokenizer_backend": (
            "offline_tokenizer_json"
            if tokenizer_json
            else "transformers_auto_tokenizer"
        ),
        "tokenizer_artifact": tokenizer_artifact,
        "splits": split_results,
        "environment": environment_snapshot(),
        "limitations": [
            "This preflight does not load model weights or execute a backward pass.",
            "GPU, loss, adapter, wall-time and memory results remain unmeasured.",
            "The held-out test split is not read by this preflight.",
        ],
    }
    write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tokenizer-json",
        default=None,
        help="Optional local tokenizer.json for an offline structural audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preflight(
        workspace_path(args.config),
        workspace_path(args.output),
        tokenizer_json=args.tokenizer_json,
    )
    maxima = {
        split: result["splits"][split]["sequence_length"]["maximum"]
        for split in PREFLIGHT_SPLITS
    }
    print("Preflight passed; maximum sequence lengths: {}".format(maxima))


if __name__ == "__main__":
    main()
