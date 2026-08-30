#!/usr/bin/env python3
"""Explicit completion-only tokenization and dynamic padding utilities."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from scripts.module_c.common import ExperimentError


IGNORE_INDEX = -100


def _normalise_token_ids(value: Any, label: str) -> List[int]:
    # Transformers v5 returns BatchEncoding by default. BatchEncoding follows
    # the Mapping protocol (via UserDict) but is not a built-in dict.
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise ExperimentError("Tokenizer returned invalid {} token ids".format(label))
    return value


def _normalise_chat_template_kwargs(
    value: Mapping[str, Any] = None,
) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not key for key in value
    ):
        raise ExperimentError("chat_template_kwargs must be an object with string keys")
    return dict(value)


def apply_chat_template_ids(
    tokenizer: Any, messages: Sequence[Mapping[str, str]], add_generation_prompt: bool,
    chat_template_kwargs: Mapping[str, Any] = None,
) -> List[int]:
    template_kwargs = _normalise_chat_template_kwargs(chat_template_kwargs)
    try:
        value = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors=None,
            **template_kwargs
        )
    except TypeError:
        # Small fake tokenizers used by unit tests need not expose return_tensors.
        value = tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=add_generation_prompt,
            **template_kwargs
        )
    return _normalise_token_ids(value, "chat-template")


def apply_chat_template_text(
    tokenizer: Any, messages: Sequence[Mapping[str, str]], add_generation_prompt: bool,
    chat_template_kwargs: Mapping[str, Any] = None,
) -> str:
    template_kwargs = _normalise_chat_template_kwargs(chat_template_kwargs)
    value = tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=add_generation_prompt,
        **template_kwargs
    )
    if not isinstance(value, str):
        raise ExperimentError("Tokenizer returned a non-text chat template")
    return value


def tokenize_completion_example(
    example: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
    chat_template_kwargs: Mapping[str, Any] = None,
) -> Dict[str, Any]:
    """Tokenize one derived row and mask every token before the completion."""
    example_id = example.get("id", "<missing-id>")
    prompt = example.get("prompt")
    completion = example.get("completion")
    if not isinstance(prompt, list) or not prompt:
        raise ExperimentError("{} has no prompt messages".format(example_id))
    if not isinstance(prompt[-1], dict) or prompt[-1].get("role") != "user":
        raise ExperimentError(
            "{} prompt must end with a user message".format(example_id)
        )
    if (
        not isinstance(completion, list)
        or len(completion) != 1
        or completion[0].get("role") != "assistant"
    ):
        raise ExperimentError(
            "{} must contain exactly one assistant completion".format(example_id)
        )

    prefix_text = apply_chat_template_text(
        tokenizer,
        prompt,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    full_text = apply_chat_template_text(
        tokenizer,
        prompt + completion,
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    if not full_text.startswith(prefix_text):
        raise ExperimentError(
            "{}: rendered generation prompt is not an exact text prefix".format(
                example_id
            )
        )

    prefix_ids = apply_chat_template_ids(
        tokenizer,
        prompt,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    full_ids = apply_chat_template_ids(
        tokenizer,
        prompt + completion,
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ExperimentError(
            "{}: generation prompt is not an exact prefix of the full template; "
            "refuse to guess the loss boundary".format(example_id)
        )
    if len(full_ids) > max_length:
        raise ExperimentError(
            "{} has {} tokens, above max_length={}; assistant targets must not "
            "be silently truncated".format(example_id, len(full_ids), max_length)
        )
    supervised_ids = full_ids[len(prefix_ids) :]
    if not supervised_ids:
        raise ExperimentError("{} has zero supervised tokens".format(example_id))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and int(eos_token_id) not in supervised_ids:
        raise ExperimentError(
            "{} supervised span does not contain tokenizer.eos_token_id".format(
                example_id
            )
        )

    labels = [IGNORE_INDEX] * len(prefix_ids) + list(supervised_ids)
    if labels[: len(prefix_ids)] != [IGNORE_INDEX] * len(prefix_ids):
        raise ExperimentError(
            "{} prompt labels are not fully masked".format(example_id)
        )
    if labels[len(prefix_ids) :] != full_ids[len(prefix_ids) :]:
        raise ExperimentError(
            "{} completion labels differ from input ids".format(example_id)
        )
    return {
        "input_ids": list(full_ids),
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "sequence_length": len(full_ids),
        "prompt_tokens": len(prefix_ids),
        "supervised_tokens": len(supervised_ids),
    }


def percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        raise ExperimentError("Cannot compute a percentile of an empty sequence")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def tokenization_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    lengths = [int(row["sequence_length"]) for row in rows]
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    supervised_tokens = sum(int(row["supervised_tokens"]) for row in rows)
    return {
        "examples": len(rows),
        "sequence_length": {
            "minimum": min(lengths),
            "median": round(percentile(lengths, 0.5), 2),
            "p95": round(percentile(lengths, 0.95), 2),
            "maximum": max(lengths),
        },
        "prompt_tokens": prompt_tokens,
        "supervised_tokens": supervised_tokens,
        "zero_supervision_examples": sum(
            int(row["supervised_tokens"]) == 0 for row in rows
        ),
    }


class CompletionOnlyDataCollator:
    """Right-pad pre-tokenized examples while preserving explicit labels."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 1) -> None:
        if pad_token_id is None:
            raise ExperimentError("Tokenizer must define pad_token_id")
        if pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be >= 1")
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ExperimentError("Cannot collate an empty batch")
        try:
            import torch
        except ImportError as exc:
            raise ExperimentError(
                "PyTorch is required to collate training batches"
            ) from exc

        maximum = max(len(feature["input_ids"]) for feature in features)
        remainder = maximum % self.pad_to_multiple_of
        if remainder:
            maximum += self.pad_to_multiple_of - remainder

        batch_input_ids: List[List[int]] = []
        batch_attention: List[List[int]] = []
        batch_labels: List[List[int]] = []
        for feature in features:
            input_ids = list(feature["input_ids"])
            attention = list(feature.get("attention_mask", [1] * len(input_ids)))
            labels = list(feature["labels"])
            if not (len(input_ids) == len(attention) == len(labels)):
                raise ExperimentError("input_ids/attention_mask/labels length mismatch")
            padding = maximum - len(input_ids)
            batch_input_ids.append(input_ids + [self.pad_token_id] * padding)
            batch_attention.append(attention + [0] * padding)
            batch_labels.append(labels + [IGNORE_INDEX] * padding)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def capability_counts(examples: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(example["metadata"]["capability"] for example in examples)
    return dict(sorted(counts.items()))
